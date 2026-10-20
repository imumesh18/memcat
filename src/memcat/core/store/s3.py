"""S3 / R2 / MinIO store (``pip install memcat[s3]``).

For backups and cold storage: the target of ``memcat warm``, ``export`` and
``import``. Each operation may take up to 10 s before it fails as a
:class:`StoreError`, so it is not meant as the store a live call reads from.
Objects live at ``{prefix}/{namespace}/{hex}`` with no TTL (bucket lifecycle
rules decide their lifetime). Admission counts and render locks are process-local.

Purges reach other processes through small event objects at
``{prefix}/{namespace}/_events/{unix_ns:020d}-{writer}-{id}``, one per purge (any
number of digests). Each process polls that prefix every ``events_poll_s`` seconds
and drops what was purged from its memory and its admission counts. New entries
are not announced (that would cost a PUT per entry). Event objects older than a day
are deleted by later purges, a bounded amount each time; a lifecycle rule expiring
``{prefix}/*/_events/`` after one day keeps that prefix small regardless. Hosts'
clocks may disagree by up to ``_EVENT_SKEW_S`` (60 s) without losing events.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import datetime as dt
import functools
import hashlib
import os
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from collections.abc import AsyncIterator, Hashable, Sequence
from importlib.util import find_spec
from typing import Any

from loguru import logger

from memcat.core.store.base import RESYNC, StoreError
from memcat.core.store.memory import MemoryStore

_S3_TIMEOUT_S = 10.0  # per request
_DELETE_BATCH = 1000  # the S3 DeleteObjects limit
_CONCURRENCY = 10  # requests in flight per store: boto3's default connection pool size
_MTIME_CACHE = 4096  # LastModified remembered from listings, for idle_seconds
# Keys the credential fingerprint, so identity() never holds anything derived from a
# secret that outlives this process.
_FINGERPRINT_KEY = os.urandom(16)
# Same values as memcat.core.index.PURGE_ONE / PURGE_ALL (not imported: no cycle).
_PURGE_ONE = b"\x00"  # + 16-byte digest
_PURGE_ALL = b"\x01"
_EVENTS = "_events/"  # never valid hex, so never parsed as an entry
_EVENT_MAGIC = b"MCE1"
_EVENT_BATCH = 100_000  # purge markers per event object
_EVENT_RETENTION_S = 86_400.0  # event objects older than this are cleaned up
_EVENT_CLEANUP_KEYS = 1000  # at most one listing page of old events per publish
_EVENT_SKEW_S = 60.0  # tolerated clock difference between hosts
_EVENT_SEEN_MAX = 10_000


def _encode_events(messages: Sequence[bytes]) -> bytes:
    return _EVENT_MAGIC + b"".join(messages)


def _decode_events(body: bytes) -> list[bytes]:
    if not body.startswith(_EVENT_MAGIC):
        return []
    out: list[bytes] = []
    i = len(_EVENT_MAGIC)
    while i < len(body):
        if body[i : i + 1] == _PURGE_ALL:
            out.append(_PURGE_ALL)
            i += 1
        elif body[i : i + 1] == _PURGE_ONE and i + 17 <= len(body):
            out.append(body[i : i + 17])
            i += 17
        else:
            break  # unknown or truncated: ignore the rest
    return out


def _is_purge(m: bytes) -> bool:
    return m == _PURGE_ALL or (len(m) == 17 and m[:1] == _PURGE_ONE)


def _event_ns(name: str) -> int | None:
    try:
        return int(name.split("-", 1)[0])
    except ValueError:
        return None


async def _guard(coro: Any) -> Any:
    try:
        return await asyncio.wait_for(coro, _S3_TIMEOUT_S)
    except asyncio.CancelledError:
        raise
    except StoreError:
        raise
    except Exception as e:
        raise StoreError(f"{type(e).__name__}: {e}") from None


__all__ = ["S3Store"]


class S3Store:
    """Object-storage store, for durable copies of a namespace (see the module docstring).

    Args:
        bucket: Bucket name.
        prefix: Key prefix inside the bucket.
        endpoint_url: Custom endpoint, e.g. Cloudflare R2 or MinIO.
        client: An existing boto3 S3 client.
        events_poll_s: How often each process lists the purge events of a namespace
            it uses (one LIST request per poll, plus one GET per new event).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "memcat",
        endpoint_url: str | None = None,
        *,
        client: Any | None = None,
        events_poll_s: float = 10.0,
        **client_kwargs: Any,
    ):
        if client is None and find_spec("boto3") is None:
            raise ImportError("S3Store needs the extra: pip install 'memcat[s3]'")
        self._events_poll_s = float(events_poll_s)
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._endpoint = endpoint_url
        self._client = client
        self._given = client is not None
        self._client_kwargs = client_kwargs
        self._client_lock = threading.Lock()
        # S3's own threads: a request stuck in boto3 past its budget keeps a thread
        # until boto3 gives up, and must not hold up other work on the default
        # executor (Opus decoding, Pipecat's own threads).
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_CONCURRENCY * 2, thread_name_prefix="memcat-s3"
        )
        self._local = MemoryStore(max_bytes=1)  # admission and locks only
        self._sem: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def identity(self) -> Hashable:
        """Bucket, prefix, endpoint, region and a keyed hash of the other client options
        (credentials included), so stores that may reach different data never share."""
        if self._given:
            return ("s3-client", id(self._client), self._bucket, self._prefix)
        kw = dict(self._client_kwargs)
        region = kw.pop("region_name", None)
        h = hashlib.blake2b(key=_FINGERPRINT_KEY, digest_size=16)
        for k in sorted(kw):
            h.update(f"{k}={kw[k]!r}\0".encode())
        return ("s3", self._bucket, self._prefix, self._endpoint, region, h.hexdigest())

    def _s3(self) -> Any:
        client = self._client
        if client is not None:
            return client
        with self._client_lock:  # called from worker threads: create exactly one
            if self._client is None:
                import boto3

                kw = dict(self._client_kwargs)
                if "config" not in kw:  # give up on a stalled socket near the budget
                    from botocore.config import Config

                    kw["config"] = Config(
                        connect_timeout=5,
                        read_timeout=_S3_TIMEOUT_S,
                        retries={"max_attempts": 2, "mode": "standard"},
                    )
                self._client = boto3.client("s3", endpoint_url=self._endpoint, **kw)
            return self._client

    def _limit(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._sem.get(loop)
        if sem is None:
            sem = self._sem[loop] = asyncio.Semaphore(_CONCURRENCY)
        return sem

    async def _call(self, fn: Any, *args: Any) -> Any:
        """Run one blocking S3 request in a thread, within the concurrency bound and
        the per-request budget (time spent waiting for a slot does not count)."""
        async with self._limit():
            loop = asyncio.get_running_loop()
            return await _guard(loop.run_in_executor(self._executor, functools.partial(fn, *args)))

    def bind(self, namespace: str) -> _BoundS3Store:
        return _BoundS3Store(self, namespace)


class _BoundS3Store:
    def __init__(self, parent: S3Store, namespace: str):
        self._p = parent
        self.namespace = namespace
        self._dir = f"{parent._prefix}/{namespace}/"
        self._local = parent._local.bind(namespace)
        self._mtimes: OrderedDict[bytes, dt.datetime] = OrderedDict()
        self._events = self._dir + _EVENTS
        self._writer = uuid.uuid4().hex[:12]  # this process's events are not its news

    def _key(self, d: bytes) -> str:
        return self._dir + d.hex()

    async def _get(self, d: bytes) -> bytes | None:
        def run() -> bytes | None:
            s3 = self._p._s3()
            try:
                obj = s3.get_object(Bucket=self._p._bucket, Key=self._key(d))
            except Exception as e:
                code = getattr(e, "response", {}).get("Error", {}).get("Code")
                if code in ("NoSuchKey", "404", "NotFound"):
                    return None
                raise
            return bytes(obj["Body"].read())

        return await self._p._call(run)

    async def get_many(self, digests: Sequence[bytes]) -> list[bytes | None]:
        # Each GET has its own budget; at most _CONCURRENCY run at once.
        return list(await asyncio.gather(*(self._get(d) for d in digests)))

    async def put(self, digest: bytes, blob: bytes, ttl_s: int) -> None:
        def run() -> None:
            self._p._s3().put_object(
                Bucket=self._p._bucket,
                Key=self._key(digest),
                Body=blob,
                ContentType="application/x-memcat-entry",
            )

        await self._p._call(run)
        self._mtimes.pop(digest, None)

    async def touch(self, digests: Sequence[bytes], ttl_s: int) -> None:
        return None  # lifetime is managed by bucket lifecycle rules

    async def delete(self, digests: Sequence[bytes] | None = None) -> int:
        extra: list[str] = []
        if digests is None:
            digests = [d async for d in self.scan()]
            # The whole namespace goes, old purge events included.
            extra = [o["Key"] async for o in self._list_raw(self._events)]
        keys = [self._key(d) for d in digests]

        def run(chunk: list[str]) -> list[dict[str, Any]]:
            res = self._p._s3().delete_objects(
                Bucket=self._p._bucket,
                Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
            )
            return list(res.get("Errors") or [])

        n = 0
        failed: list[dict[str, Any]] = []
        for i in range(0, len(keys), _DELETE_BATCH):
            chunk = keys[i : i + _DELETE_BATCH]
            errors = await self._p._call(run, chunk)
            failed += errors
            n += len(chunk) - len(errors)
        for i in range(0, len(extra), _DELETE_BATCH):
            with contextlib.suppress(StoreError):  # events also age out on their own
                await self._p._call(run, extra[i : i + _DELETE_BATCH])
        await self._local.forget_admission(digests)
        for d in digests:
            self._mtimes.pop(d, None)
        if failed:
            first = failed[0]
            raise StoreError(
                f"{len(failed)} of {len(keys)} objects not deleted "
                f"({first.get('Code')}: {first.get('Message')}; key {first.get('Key')})"
            )
        return n

    async def find_prefixes(self, prefixes: Sequence[bytes]) -> list[bytes]:
        """Digests of entries and local admission counts starting with any of
        ``prefixes``: one prefix listing each, several at a time."""
        hexes = sorted({p.hex() for p in prefixes})
        # A listing per prefix pages only through matches; drop prefixes another covers.
        hexes = [h for h in hexes if not any(h != o and h.startswith(o) for o in hexes)]

        async def one(h: str) -> list[bytes]:
            return [d async for d in self._list(h)]

        out: set[bytes] = set()
        for found in await asyncio.gather(*(one(h) for h in hexes)):
            out.update(found)
        out |= set(await self._local.find_prefixes(prefixes))
        return sorted(out)

    async def scan(self) -> AsyncIterator[bytes]:
        async for d in self._list(""):
            yield d

    async def _list_raw(
        self, prefix: str, start_after: str | None = None, max_pages: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        def page(token: str | None) -> tuple[list[dict[str, Any]], str | None]:
            kw: dict[str, Any] = {"Bucket": self._p._bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            elif start_after:
                kw["StartAfter"] = start_after
            res = self._p._s3().list_objects_v2(**kw)
            objs = list(res.get("Contents", []))
            return objs, res.get("NextContinuationToken") if res.get("IsTruncated") else None

        token: str | None = None
        pages = 0
        while True:
            objs, token = await self._p._call(page, token)
            pages += 1
            for o in objs:
                yield o
            if token is None or (max_pages is not None and pages >= max_pages):
                return

    async def _list(self, hex_prefix: str) -> AsyncIterator[bytes]:
        async for o in self._list_raw(self._dir + hex_prefix):
            name = o["Key"][len(self._dir) :]
            if len(name) != 32:
                continue  # purge events and anything else that is not an entry
            try:
                d = bytes.fromhex(name)
            except ValueError:
                continue
            mtime = o.get("LastModified")
            if isinstance(mtime, dt.datetime):
                self._remember(d, mtime)
            yield d

    def _remember(self, d: bytes, mtime: dt.datetime) -> None:
        self._mtimes[d] = mtime
        self._mtimes.move_to_end(d)
        while len(self._mtimes) > _MTIME_CACHE:
            self._mtimes.popitem(last=False)

    async def idle_seconds(self, digests: Sequence[bytes]) -> list[float | None]:
        """Seconds since each object was written (S3 keeps no access time).

        Uses LastModified from the listing ``scan`` just read when it has it, and
        otherwise one HEAD per object, several at a time.
        """

        def run(d: bytes) -> dt.datetime | None:
            s3 = self._p._s3()
            try:
                head = s3.head_object(Bucket=self._p._bucket, Key=self._key(d))
            except Exception:
                return None
            return head["LastModified"]

        async def one(d: bytes) -> float | None:
            mtime = self._mtimes.get(d)
            if mtime is None:
                mtime = await self._p._call(run, d)
            if mtime is None:
                return None
            return (dt.datetime.now(dt.UTC) - mtime).total_seconds()

        return list(await asyncio.gather(*(one(d) for d in digests)))

    async def publish_new(self, digest: bytes) -> None:
        await self.publish_many([digest])

    async def publish_many(self, messages: Sequence[bytes]) -> None:
        """Write purge markers as one event object per batch; new-entry
        notifications are not sent (S3 has no cheap channel for them)."""
        purges = [m for m in messages if _is_purge(m)]
        if not purges:
            return
        for i in range(0, len(purges), _EVENT_BATCH):
            body = _encode_events(purges[i : i + _EVENT_BATCH])

            def run(body: bytes = body) -> None:
                # Named when the PUT starts, not before waiting for a slot, so the
                # time in the name stays close to when other processes can list it.
                key = f"{self._events}{time.time_ns():020d}-{self._writer}-{uuid.uuid4().hex[:8]}"
                self._p._s3().put_object(
                    Bucket=self._p._bucket,
                    Key=key,
                    Body=body,
                    ContentType="application/x-memcat-events",
                )

            await self._p._call(run)
        try:
            await self._clean_events()
        except StoreError as e:
            logger.debug(f"memcat: s3 event cleanup failed ({e})")

    async def _clean_events(self) -> None:
        """Delete event objects past retention: the oldest page at most (bounded)."""
        cutoff = time.time_ns() - int(_EVENT_RETENTION_S * 1e9)
        old: list[str] = []
        async for o in self._list_raw(self._events, max_pages=1):
            ns = _event_ns(o["Key"][len(self._events) :])
            if ns is not None and ns < cutoff:
                old.append(o["Key"])
            if len(old) >= _EVENT_CLEANUP_KEYS:
                break
        if old:

            def run() -> None:
                self._p._s3().delete_objects(
                    Bucket=self._p._bucket,
                    Delete={"Objects": [{"Key": k} for k in old], "Quiet": True},
                )

            await self._p._call(run)

    async def _read_event(self, key: str) -> list[bytes]:
        def run() -> bytes | None:
            try:
                obj = self._p._s3().get_object(Bucket=self._p._bucket, Key=key)
            except Exception as e:
                code = getattr(e, "response", {}).get("Error", {}).get("Code")
                if code in ("NoSuchKey", "404", "NotFound"):
                    return None  # cleaned up already
                raise
            return bytes(obj["Body"].read())

        body = await self._p._call(run)
        return _decode_events(body) if body is not None else []

    async def _poll_events(
        self, after_ns: int, seen: OrderedDict[str, int]
    ) -> list[tuple[str, int]]:
        """New event keys (and their times) written after ``after_ns``, oldest first."""
        start = f"{self._events}{max(after_ns, 0):020d}"
        new: list[tuple[str, int]] = []
        async for o in self._list_raw(self._events, start_after=start):
            name = o["Key"][len(self._events) :]
            ns = _event_ns(name)
            if ns is None or o["Key"] in seen:
                continue
            new.append((o["Key"], ns))
        return new

    async def subscribe_new(self) -> AsyncIterator[bytes]:
        """Purge markers other processes wrote, polled every ``events_poll_s``.

        Starts from now: events written before the subscription are not replayed.
        Yields RESYNC when polling failed for longer than event retention, since
        events from that gap may already be cleaned up.
        """
        poll = self._p._events_poll_s
        skew_ns = int(_EVENT_SKEW_S * 1e9)
        seen: OrderedDict[str, int] = OrderedDict()
        cursor: int | None = None  # our clock at the last successful poll
        failing_since: float | None = None
        backoff = poll
        while True:
            began = time.time_ns()
            try:
                if cursor is None:
                    # Baseline: events from before the subscription are history. One
                    # written while this listing runs is news (a repeated purge is
                    # harmless; a missed one is not).
                    news: list[tuple[str, int]] = []
                    for key, ns in await self._poll_events(began - skew_ns, seen):
                        if ns < began:
                            seen[key] = ns
                        else:
                            news.append((key, ns))
                else:
                    news = await self._poll_events(cursor - skew_ns, seen)
            except StoreError as e:
                if failing_since is None:
                    failing_since = time.monotonic()
                logger.debug(f"memcat: s3 event poll failed ({e}); retrying")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max(poll, 30.0))
                continue
            gap = failing_since is not None and (
                time.monotonic() - failing_since >= _EVENT_RETENTION_S
            )
            failing_since = None
            backoff = poll
            cursor = began
            if gap:
                yield RESYNC
            for key, ns in news:
                seen[key] = ns
                if key[len(self._events) :].split("-")[1:2] == [self._writer]:
                    continue  # our own purge: already applied here
                try:
                    msgs = await self._read_event(key)
                except StoreError:
                    del seen[key]  # read it on the next poll
                    cursor = min(cursor, ns + skew_ns)
                    break
                for m in msgs:
                    if m == _PURGE_ALL:
                        await self._local.forget_admission(None)
                    else:
                        await self._local.forget_admission([m[1:]])
                    yield m
            # Forget keys that can no longer be listed again, and bound the rest.
            floor = cursor - skew_ns
            for key in [k for k, ns in seen.items() if ns < floor]:
                del seen[key]
            while len(seen) > _EVENT_SEEN_MAX:
                seen.popitem(last=False)
            await asyncio.sleep(poll)

    async def observe(self, digest: bytes, call_id: str, k: int) -> bool:
        return await self._local.observe(digest, call_id, k)

    async def forget_admission(self, digests: Sequence[bytes] | None = None) -> None:
        await self._local.forget_admission(digests)

    async def try_lock(self, digest: bytes, ttl_s: int) -> bool:
        return await self._local.try_lock(digest, ttl_s)

    async def unlock(self, digest: bytes) -> None:
        await self._local.unlock(digest)

    async def close(self) -> None:
        return None
