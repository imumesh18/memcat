"""Redis and Valkey store (``pip install memcat[redis]``, redis-py 5.0.1 or newer).

Key names, where ``{ns}`` is the namespace and ``{hex}`` the 32-character digest::

    memcat:{ns}:e:{hex}     string        entry blob, TTL refreshed on hit
    memcat:{ns}:t:{hex}     string        unix time the entry was last written or served
    memcat:{ns}:a:{hex}     HyperLogLog   distinct calls that missed it; deleted once admitted
    memcat:{ns}:lock:{hex}  string        render lock holder's random token, 60 s
    memcat:{ns}:new         pub/sub       digest of each new entry

Every command and script touches one key, so nothing depends on keys sharing a slot.

Each operation has its own time budget (the hot-path ``MGET`` uses ``op_timeout_ms``;
writes 2 s, bulk pages and ``get_bulk`` 5 s, ``idle_seconds`` 10 s). Commands wait for a free pooled
connection within that budget rather than failing at once, and subscriptions use
their own connections so they never take one from the pool.

redis-py clients belong to the event loop that created them, so the store keeps one
client per running loop: a script that calls ``asyncio.run()`` twice still works.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import weakref
from collections.abc import AsyncIterator, Hashable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from loguru import logger

from memcat.core.store.base import DEFAULT_TTL_S, RESYNC, StoreError

__all__ = ["RedisStore"]

_ADMIT_LUA = """
redis.call('PFADD', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[3])
local n = redis.call('PFCOUNT', KEYS[1])
if n >= tonumber(ARGV[2]) then
  redis.call('DEL', KEYS[1])
  return 1
end
return 0
"""
# Delete the lock only while it still holds our token: a render that outlived its
# lock must not release the lock another process has taken since.
_UNLOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
_ADMISSION_TTL_S = DEFAULT_TTL_S  # a clause not heard again for this long starts over
_WRITE_TIMEOUT_S = 2.0  # put, touch, publish, admission, locks
_BULK_TIMEOUT_S = 5.0  # one SCAN page, UNLINK or PUBLISH batch
_IDLE_TIMEOUT_S = 10.0  # one idle_seconds batch
# Socket reads only guard against a connection that hangs outside any budget; each
# operation's own budget (asyncio.wait_for) is what callers see.
_SOCKET_TIMEOUT_S = max(_WRITE_TIMEOUT_S, _BULK_TIMEOUT_S, _IDLE_TIMEOUT_S)
_UNLINK_BATCH = 500
_SCHEMES = ("redis://", "rediss://", "unix://")


class _PrefixSet:
    """Matches hex digests against a set of byte prefixes, grouped by length."""

    def __init__(self, prefixes: Sequence[bytes]) -> None:
        by_len: dict[int, set[str]] = {}
        for p in prefixes:
            by_len.setdefault(len(p) * 2, set()).add(p.hex())
        self._by_len = sorted(by_len.items())
        hexes = {h for _, hs in self._by_len for h in hs}
        self.only = next(iter(hexes)) if len(hexes) == 1 else None

    def __bool__(self) -> bool:
        return bool(self._by_len)

    def __call__(self, h: str) -> bool:
        return any(h[:n] in hs for n, hs in self._by_len)


@dataclass
class _Conn:
    """The clients for one event loop."""

    client: Any  # commands, through a blocking pool of pool_size connections
    sub: Any  # pub/sub only, one connection per subscription, never from the pool
    admit: Any
    unlock: Any


class RedisStore:
    """Shared store on Redis 6+ or Valkey 7+.

    Args:
        url: ``redis://`` or ``rediss://`` URL.
        pool_size: Maximum command connections per event loop. A command waits for a
            free one within its time budget. Subscriptions have their own connections.
        op_timeout_ms: Reads slower than this count as a miss and as a store error.
        client: An existing ``redis.asyncio.Redis`` to use instead of ``url``. It is
            used as given: on its own event loop only, and for subscriptions too.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        pool_size: int = 10,
        op_timeout_ms: int = 50,
        client: Any | None = None,
    ):
        if client is None:
            if find_spec("redis") is None:
                raise ImportError("RedisStore needs the extra: pip install 'memcat[redis]'")
            if not url.startswith(_SCHEMES):
                raise ValueError("RedisStore url must start with redis://, rediss:// or unix://")
        self._url = url
        self._pool_size = pool_size
        self._timeout = op_timeout_ms / 1000.0
        self._given = client
        self._owns_client = client is None
        self._conns: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Conn] = (
            weakref.WeakKeyDictionary()
        )

    def identity(self) -> Hashable:
        if not self._owns_client:
            return ("redis-client", id(self._given))
        return ("redis", self._url, self._pool_size, self._timeout)

    def _conn(self) -> _Conn:
        loop = asyncio.get_running_loop()
        conn = self._conns.get(loop)
        if conn is None:
            # The connections hold their loop, so the weak keys alone never let a
            # finished loop go (one per asyncio.run()): drop those explicitly.
            for old in [lp for lp in self._conns if lp.is_closed()]:
                del self._conns[old]
            conn = self._connect()
            self._conns[loop] = conn
        return conn

    def _connect(self) -> _Conn:
        if self._given is not None:
            c = self._given
            return _Conn(c, c, c.register_script(_ADMIT_LUA), c.register_script(_UNLOCK_LUA))
        import redis.asyncio as aioredis

        common: dict[str, Any] = {
            "socket_timeout": max(_SOCKET_TIMEOUT_S, self._timeout),
            "socket_connect_timeout": 1.0,
            "health_check_interval": 30,
        }
        pool = aioredis.BlockingConnectionPool.from_url(
            self._url,
            max_connections=self._pool_size,
            # Backstop only: every command's own budget cancels the wait sooner.
            timeout=_SOCKET_TIMEOUT_S,
            **common,
        )
        client = aioredis.Redis.from_pool(pool)
        sub = aioredis.Redis.from_url(self._url, **common)
        return _Conn(
            client, sub, client.register_script(_ADMIT_LUA), client.register_script(_UNLOCK_LUA)
        )

    def bind(self, namespace: str) -> _BoundRedisStore:
        return _BoundRedisStore(self, namespace)

    async def aclose(self) -> None:
        """Close this event loop's connections (clients of other loops are dropped)."""
        loop = asyncio.get_running_loop()
        conns = dict(self._conns)
        self._conns.clear()
        if not self._owns_client:
            return
        for owner, conn in conns.items():
            if owner is not loop:
                continue  # its loop is closed or busy elsewhere: nothing safe to await
            for c in (conn.client, conn.sub):
                with contextlib.suppress(Exception):
                    await c.aclose()


class _BoundRedisStore:
    def __init__(self, parent: RedisStore, namespace: str):
        self._p = parent
        self.namespace = namespace
        self._prefix = f"memcat:{namespace}:"
        # Lock tokens this process holds, by digest and the task that took the lock
        # (worker.py takes and releases a lock in one task).
        self._tokens: dict[tuple[bytes, asyncio.Task[Any] | None], bytes] = {}

    @property
    def read_timeout_s(self) -> float:
        """Budget of ``get_many`` on the call path (``op_timeout_ms``)."""
        return self._p._timeout

    @property
    def _r(self) -> Any:
        return self._p._conn().client

    def _e(self, d: bytes) -> str:
        return f"{self._prefix}e:{d.hex()}"

    def _t(self, d: bytes) -> str:
        return f"{self._prefix}t:{d.hex()}"

    def _a(self, d: bytes) -> str:
        return f"{self._prefix}a:{d.hex()}"

    def _lock(self, d: bytes) -> str:
        return f"{self._prefix}lock:{d.hex()}"

    async def _timed(self, coro: Any, timeout: float | None = None) -> Any:
        try:
            return await asyncio.wait_for(coro, timeout or self._p._timeout)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise StoreError(f"{type(e).__name__}: {e}") from None

    async def _scan_keys(self, match: str) -> AsyncIterator[str]:
        """``SCAN`` with each page timed, so a stalled server fails as a StoreError."""
        it: Any = self._r.scan_iter(match=match, count=1000)
        try:
            while True:
                try:
                    key = await asyncio.wait_for(it.__anext__(), _BULK_TIMEOUT_S)
                except StopAsyncIteration:
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    raise StoreError(f"{type(e).__name__}: {e}") from None
                yield key.decode() if isinstance(key, bytes) else key
        finally:
            with contextlib.suppress(Exception):
                await it.aclose()

    async def _unlink(self, keys: Sequence[str]) -> int:
        return int(await self._timed(self._r.unlink(*keys), _BULK_TIMEOUT_S))

    async def _mget(self, digests: Sequence[bytes], timeout: float) -> list[bytes | None]:
        if not digests:
            return []
        res = await self._timed(self._r.mget([self._e(d) for d in digests]), timeout)
        return [bytes(x) if x is not None else None for x in res]

    async def get_many(self, digests: Sequence[bytes]) -> list[bytes | None]:
        return await self._mget(digests, self._p._timeout)

    async def get_bulk(self, digests: Sequence[bytes]) -> list[bytes | None]:
        """``get_many`` with the bulk budget instead of the hot-path one, for commands
        that read every entry (``memcat export`` and ``stats``)."""
        return await self._mget(digests, _BULK_TIMEOUT_S)

    async def put(self, digest: bytes, blob: bytes, ttl_s: int) -> None:
        pipe = self._r.pipeline(transaction=False)
        pipe.set(self._e(digest), blob, ex=ttl_s)
        pipe.set(self._t(digest), int(time.time()), ex=ttl_s)
        await self._timed(pipe.execute(), _WRITE_TIMEOUT_S)

    async def touch(self, digests: Sequence[bytes], ttl_s: int) -> None:
        if not digests:
            return
        now = int(time.time())
        pipe = self._r.pipeline(transaction=False)
        for d in digests:
            pipe.expire(self._e(d), ttl_s)
            # XX: never create a timestamp for an entry that is gone (purged meanwhile).
            pipe.set(self._t(d), now, ex=ttl_s, xx=True)
        await self._timed(pipe.execute(), _WRITE_TIMEOUT_S)

    async def delete(self, digests: Sequence[bytes] | None = None) -> int:
        if digests is None:
            n = 0
            batch: list[str] = []
            async for k in self._scan_keys(f"{self._prefix}*"):
                if k.startswith(f"{self._prefix}e:"):
                    n += 1
                batch.append(k)
                if len(batch) >= _UNLINK_BATCH:
                    await self._unlink(batch)
                    batch = []
            if batch:
                await self._unlink(batch)
            return n
        if not digests:
            return 0
        n = 0
        for i in range(0, len(digests), _UNLINK_BATCH):
            chunk = digests[i : i + _UNLINK_BATCH]
            n += await self._unlink([self._e(d) for d in chunk])
            await self._unlink([self._t(d) for d in chunk])
        await self.forget_admission(digests)
        return n

    async def scan(self) -> AsyncIterator[bytes]:
        start = len(self._prefix) + 2
        async for k in self._scan_keys(f"{self._prefix}e:*"):
            try:
                d = bytes.fromhex(k[start:])
            except ValueError:
                continue
            if len(d) == 16:
                yield d

    async def find_prefixes(self, prefixes: Sequence[bytes]) -> list[bytes]:
        """Digests of entries and admission counts starting with any of ``prefixes``:
        one ``SCAN`` of the entry keys and one of the admission keys, however many."""
        match = _PrefixSet(prefixes)
        if not match:
            return []
        out: set[bytes] = set()
        for kind in ("e", "a"):
            start = len(self._prefix) + len(kind) + 1
            # With a single prefix, let the server filter; otherwise filter here.
            pattern = f"{self._prefix}{kind}:{match.only or ''}*"
            async for k in self._scan_keys(pattern):
                h = k[start:]
                if not match(h):
                    continue
                try:
                    d = bytes.fromhex(h)
                except ValueError:
                    continue
                if len(d) == 16:
                    out.add(d)
        return sorted(out)

    async def publish_new(self, digest: bytes) -> None:
        await self._timed(self._r.publish(f"{self._prefix}new", digest), _WRITE_TIMEOUT_S)

    async def publish_many(self, messages: Sequence[bytes]) -> None:
        """Publish several notifications in pipelined round trips (bulk purges)."""
        channel = f"{self._prefix}new"
        for i in range(0, len(messages), _UNLINK_BATCH):
            pipe = self._r.pipeline(transaction=False)
            for m in messages[i : i + _UNLINK_BATCH]:
                pipe.publish(channel, m)
            await self._timed(pipe.execute(), _BULK_TIMEOUT_S)

    async def subscribe_new(self) -> AsyncIterator[bytes]:
        backoff = 0.5
        lost = False
        while True:
            pubsub: Any = None
            try:
                pubsub = self._p._conn().sub.pubsub(ignore_subscribe_messages=True)
                await pubsub.subscribe(f"{self._prefix}new")
                backoff = 0.5
                if lost:
                    lost = False
                    yield RESYNC
                while True:
                    msg = await pubsub.get_message(timeout=30.0)
                    if msg is None:
                        continue
                    data = msg.get("data")
                    if isinstance(data, (bytes, bytearray)) and len(data) in (1, 16, 17):
                        yield bytes(data)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                lost = True
                logger.debug(f"memcat: redis subscription lost ({type(e).__name__}); retrying")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                if pubsub is not None:
                    with contextlib.suppress(Exception):
                        await pubsub.aclose()

    async def observe(self, digest: bytes, call_id: str, k: int) -> bool:
        admit = self._p._conn().admit
        res = await self._timed(
            admit(keys=[self._a(digest)], args=[call_id, k, _ADMISSION_TTL_S]), _WRITE_TIMEOUT_S
        )
        return int(res) == 1

    async def forget_admission(self, digests: Sequence[bytes] | None = None) -> None:
        if digests is None:
            batch: list[str] = []
            async for k in self._scan_keys(f"{self._prefix}a:*"):
                batch.append(k)
                if len(batch) >= _UNLINK_BATCH:
                    await self._unlink(batch)
                    batch = []
            if batch:
                await self._unlink(batch)
            return
        for i in range(0, len(digests), _UNLINK_BATCH):
            await self._unlink([self._a(d) for d in digests[i : i + _UNLINK_BATCH]])

    async def try_lock(self, digest: bytes, ttl_s: int) -> bool:
        token = os.urandom(16).hex().encode()
        ok = bool(
            await self._timed(
                self._r.set(self._lock(digest), token, nx=True, ex=ttl_s), _WRITE_TIMEOUT_S
            )
        )
        if ok:
            if len(self._tokens) >= 1024:  # forget locks whose task ended without unlock
                for key in [k for k in self._tokens if k[1] is not None and k[1].done()]:
                    del self._tokens[key]
            self._tokens[(digest, asyncio.current_task())] = token
        return ok

    def _take_token(self, digest: bytes) -> bytes | None:
        token = self._tokens.pop((digest, asyncio.current_task()), None)
        if token is None:  # released from another task: only if exactly one holder
            held = [k for k in self._tokens if k[0] == digest]
            if len(held) == 1:
                token = self._tokens.pop(held[0])
        return token

    async def unlock(self, digest: bytes) -> None:
        token = self._take_token(digest)
        if token is None:
            return  # not ours (never taken, or already released)
        unlock = self._p._conn().unlock
        await self._timed(unlock(keys=[self._lock(digest)], args=[token]), _WRITE_TIMEOUT_S)

    async def idle_seconds(self, digests: Sequence[bytes]) -> list[float | None]:
        """Seconds since each entry was last written or served; None when it is gone.

        Read from the timestamp ``put`` and ``touch`` keep beside each entry.
        """
        if not digests:
            return []
        pipe = self._r.pipeline(transaction=False)
        for d in digests:
            pipe.get(self._t(d))
            pipe.ttl(self._e(d))
        res = await self._timed(pipe.execute(), _IDLE_TIMEOUT_S)
        now = time.time()
        out: list[float | None] = []
        for i in range(len(digests)):
            stamp, ttl = res[2 * i], int(res[2 * i + 1])
            if ttl == -2 or stamp is None:  # the entry is gone
                out.append(None)
            else:
                try:
                    out.append(max(0.0, now - float(stamp)))
                except ValueError:
                    out.append(None)
        return out

    async def close(self) -> None:
        return None
