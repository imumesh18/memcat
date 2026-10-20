"""In-process store: a byte-bounded LRU with TTLs on a monotonic clock."""

from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

from memcat.core.store.base import RESYNC

__all__ = ["MemoryStore"]

# A subscriber this far behind loses its backlog and gets RESYNC instead (it re-reads
# the store), the same contract as a Redis subscription that reconnected.
_MAX_QUEUED = 10_000
_LOCK_SWEEP = 1024  # sweep expired locks whenever the table grows past this


@dataclass
class _Shared:
    max_bytes: int
    entries: OrderedDict[tuple[str, bytes], tuple[bytes, float]] = field(
        default_factory=OrderedDict
    )
    size: int = 0
    admission: OrderedDict[tuple[str, bytes], set[str]] = field(default_factory=OrderedDict)
    # key -> (expiry, holder token)
    locks: dict[tuple[str, bytes], tuple[float, bytes]] = field(default_factory=dict)
    lock_sweep_at: int = _LOCK_SWEEP
    # Each listener's queue with the loop it waits on: a publish from another
    # loop (another thread) must hand the message over through that loop.
    subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[bytes]]]] = field(
        default_factory=dict
    )


class MemoryStore:
    """Process-local store. Good for one process, tests and development.

    Args:
        max_bytes: Bound on stored blob bytes; least recently used entries go first.
        max_admission_keys: Bound on clauses being counted for admission.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024, max_admission_keys: int = 100_000):
        self._shared = _Shared(max_bytes=max_bytes)
        self._max_admission = max_admission_keys

    def identity(self) -> Hashable:
        return ("memory", id(self._shared))

    def bind(self, namespace: str) -> _BoundMemoryStore:
        return _BoundMemoryStore(self._shared, namespace, self._max_admission)


class _BoundMemoryStore:
    def __init__(self, shared: _Shared, namespace: str, max_admission: int):
        self._s = shared
        self.namespace = namespace
        self._max_admission = max_admission
        # Tokens of the locks this view took, by digest and the taking task.
        self._tokens: dict[tuple[bytes, asyncio.Task[Any] | None], bytes] = {}

    def _k(self, digest: bytes) -> tuple[str, bytes]:
        return (self.namespace, digest)

    def _live(self, key: tuple[str, bytes], now: float) -> bytes | None:
        item = self._s.entries.get(key)
        if item is None:
            return None
        blob, expires = item
        if expires <= now:
            self._drop(key)
            return None
        return blob

    def _drop(self, key: tuple[str, bytes]) -> bool:
        item = self._s.entries.pop(key, None)
        if item is None:
            return False
        self._s.size -= len(item[0])
        return True

    async def get_many(self, digests: Sequence[bytes]) -> list[bytes | None]:
        now = time.monotonic()
        out: list[bytes | None] = []
        for d in digests:
            key = self._k(d)
            blob = self._live(key, now)
            if blob is not None:
                self._s.entries.move_to_end(key)
            out.append(blob)
        return out

    async def put(self, digest: bytes, blob: bytes, ttl_s: int) -> None:
        key = self._k(digest)
        self._drop(key)
        if len(blob) > self._s.max_bytes:
            return
        self._s.entries[key] = (blob, time.monotonic() + ttl_s)
        self._s.size += len(blob)
        while self._s.size > self._s.max_bytes and self._s.entries:
            _, (b, _) = self._s.entries.popitem(last=False)
            self._s.size -= len(b)

    async def touch(self, digests: Sequence[bytes], ttl_s: int) -> None:
        now = time.monotonic()
        for d in digests:
            key = self._k(d)
            blob = self._live(key, now)
            if blob is not None:
                self._s.entries[key] = (blob, now + ttl_s)
                self._s.entries.move_to_end(key)

    async def delete(self, digests: Sequence[bytes] | None = None) -> int:
        if digests is None:
            keys = [k for k in self._s.entries if k[0] == self.namespace]
        else:
            keys = [self._k(d) for d in digests]
        n = sum(1 for k in keys if self._drop(k))
        await self.forget_admission(digests)
        return n

    async def scan(self) -> AsyncIterator[bytes]:
        now = time.monotonic()
        for (ns, d), (_, exp) in list(self._s.entries.items()):
            if ns == self.namespace and exp > now:
                yield d

    async def find_prefixes(self, prefixes: Sequence[bytes]) -> list[bytes]:
        """Digests of entries and admission counts starting with any of ``prefixes``."""
        ps = tuple(prefixes)
        if not ps:
            return []
        ns = self.namespace
        out = {d for (n, d) in self._s.entries if n == ns and d.startswith(ps)}
        out |= {d for (n, d) in self._s.admission if n == ns and d.startswith(ps)}
        return sorted(out)

    async def publish_new(self, digest: bytes) -> None:
        here = asyncio.get_running_loop()
        for loop, q in list(self._s.subscribers.get(self.namespace, [])):
            if loop is here:
                _deliver(q, digest)
            elif not loop.is_closed():
                loop.call_soon_threadsafe(_deliver, q, digest)

    async def publish_many(self, messages: Sequence[bytes]) -> None:
        for m in messages:
            await self.publish_new(m)

    async def subscribe_new(self) -> AsyncIterator[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue()
        entry = (asyncio.get_running_loop(), q)
        subs = self._s.subscribers.setdefault(self.namespace, [])
        subs.append(entry)
        try:
            while True:
                yield await q.get()
        finally:
            subs.remove(entry)

    async def observe(self, digest: bytes, call_id: str, k: int) -> bool:
        key = self._k(digest)
        seen = self._s.admission.get(key)
        if seen is None:
            seen = set()
            self._s.admission[key] = seen
            while len(self._s.admission) > self._max_admission:
                self._s.admission.popitem(last=False)
        else:
            self._s.admission.move_to_end(key)
        seen.add(call_id)
        if len(seen) >= k:
            del self._s.admission[key]
            return True
        return False

    async def forget_admission(self, digests: Sequence[bytes] | None = None) -> None:
        if digests is None:
            for key in [k for k in self._s.admission if k[0] == self.namespace]:
                del self._s.admission[key]
        else:
            for d in digests:
                self._s.admission.pop(self._k(d), None)

    async def try_lock(self, digest: bytes, ttl_s: int) -> bool:
        key = self._k(digest)
        now = time.monotonic()
        held = self._s.locks.get(key)
        if held is not None and held[0] > now:
            return False
        locks = self._s.locks
        if len(locks) >= self._s.lock_sweep_at:
            for k in [k for k, (exp, _) in locks.items() if exp <= now]:
                del locks[k]
            self._s.lock_sweep_at = max(_LOCK_SWEEP, 2 * len(locks))
        if len(self._tokens) >= _LOCK_SWEEP:
            for k in [k for k in self._tokens if k[1] is not None and k[1].done()]:
                del self._tokens[k]
        token = os.urandom(16)
        locks[key] = (now + ttl_s, token)
        self._tokens[(digest, asyncio.current_task())] = token
        return True

    async def unlock(self, digest: bytes) -> None:
        token = self._tokens.pop((digest, asyncio.current_task()), None)
        if token is None:  # released from another task: only if exactly one holder
            held = [k for k in self._tokens if k[0] == digest]
            if len(held) != 1:
                return
            token = self._tokens.pop(held[0])
        key = self._k(digest)
        cur = self._s.locks.get(key)
        if cur is not None and cur[1] == token:
            del self._s.locks[key]

    async def close(self) -> None:
        return None


def _deliver(q: asyncio.Queue[bytes], message: bytes) -> None:
    """Queue a notification on the listener's own loop. Never drops silently: a full
    queue is emptied and told to resync instead."""
    if q.qsize() >= _MAX_QUEUED:
        while not q.empty():
            q.get_nowait()
        q.put_nowait(RESYNC)
    q.put_nowait(message)
