"""In-process set of every stored digest, so a miss never costs a network call."""

from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Collection

import numpy as np
from loguru import logger

from memcat.core.store.base import RESYNC, BoundStore

__all__ = ["PURGE_ALL", "PURGE_ONE", "BloomFilter", "KeyIndex"]

# Notification payloads beside a plain 16-byte "new entry" digest.
PURGE_ONE = b"\x00"  # + 16-byte digest: that entry was deleted
PURGE_ALL = b"\x01"  # the whole namespace was deleted

_BLOOM_MIN_CAPACITY = 10_000_000  # about 12 MB at 1% false positives
_BLOOM_HEADROOM = 10  # a new filter has room for 10x the keys it starts with
_MAX_ABSENT = 100_000  # digests removed while in Bloom mode, remembered most recent first
_MASK64 = (1 << 64) - 1
_FOLLOW_HEALTHY_S = 10.0  # a subscription that lasted this long resets the retry backoff


class BloomFilter:
    """A Bloom filter over 16-byte digests (already uniform, so they are the hashes).

    ``count`` is the number of distinct digests added, as far as the filter can tell:
    adding a digest it already reports present is not counted again.
    """

    def __init__(self, capacity: int, fp_rate: float = 0.01):
        n = max(1024, capacity)
        self._m = max(8, int(-n * math.log(fp_rate) / (math.log(2) ** 2)))
        self._k = max(1, round(self._m / n * math.log(2)))
        self._bits = bytearray((self._m + 7) // 8)
        self.count = 0

    @classmethod
    def from_digests(cls, digests: Collection[bytes], capacity: int) -> BloomFilter:
        """Build a filter holding ``digests`` (distinct), vectorized: about 0.2 s per
        million digests. Run it off the event loop."""
        bf = cls(capacity)
        if digests:
            a = np.frombuffer(b"".join(digests), dtype="<u8").reshape(-1, 2)
            h2 = a[:, 1] | np.uint64(1)
            pos = a[:, 0].copy()
            m = np.uint64(bf._m)
            bits = np.frombuffer(bf._bits, dtype=np.uint8)
            with np.errstate(over="ignore"):
                for _ in range(bf._k):
                    p = pos % m
                    np.bitwise_or.at(
                        bits,
                        (p >> np.uint64(3)).astype(np.intp),
                        np.left_shift(np.uint8(1), (p & np.uint64(7)).astype(np.uint8)),
                    )
                    pos += h2  # wraps at 2**64, as _positions does
            bf.count = len(digests)
        return bf

    def _positions(self, d: bytes) -> list[int]:
        h1 = int.from_bytes(d[:8], "little")
        h2 = int.from_bytes(d[8:16], "little") | 1
        m = self._m
        return [((h1 + i * h2) & _MASK64) % m for i in range(self._k)]

    def add(self, d: bytes) -> None:
        bits = self._bits
        new = False
        for p in self._positions(d):
            byte, bit = p >> 3, 1 << (p & 7)
            if not bits[byte] & bit:
                new = True
                bits[byte] |= bit
        if new:
            self.count += 1

    def __contains__(self, d: bytes) -> bool:
        bits = self._bits
        return all(bits[p >> 3] & (1 << (p & 7)) for p in self._positions(d))

    @property
    def nbytes(self) -> int:
        return len(self._bits)


class KeyIndex:
    """Mirror of the store's digests: loaded by ``scan()``, kept fresh by pub/sub.

    Until the first scan finishes, :meth:`may_contain` answers True for unknown
    digests so lookups still reach the store. When the subscription reconnects
    (the store yields :data:`RESYNC`), the store is scanned again and the index
    replaced by what it holds, so entries announced while it was down are learned
    and entries purged meanwhile are forgotten (``on_resync`` then lets the owner
    drop them from its L1).

    A set costs about 91 bytes per key in CPython. Past ``bloom_after`` keys the
    index becomes a fixed-size Bloom filter with room for 10x the keys it holds
    then, and at least 10 million: about 12 MB with the default ``bloom_after``,
    i.e. 1.2 bytes per key of capacity at 1% false positives. A false positive
    costs one store read; past the filter's capacity the false-positive rate rises.
    Filters are built in a thread, never on the event loop.
    """

    def __init__(
        self,
        store: BoundStore,
        on_purge: Callable[[list[bytes] | None], None] | None = None,
        bloom_after: int = 1_000_000,
        min_resync_interval_s: float = 10.0,
        on_resync: Callable[[], None] | None = None,
    ):
        self._store = store
        self._on_purge = on_purge
        self._on_resync = on_resync
        self._keys: set[bytes] = set()  # set mode; while converting, keys added since
        self._frozen: set[bytes] | None = None  # the set a Bloom filter is being built from
        self._bloom: BloomFilter | None = None
        self._absent: OrderedDict[bytes, None] = OrderedDict()  # removed from frozen/Bloom
        self._bloom_after = bloom_after
        self._gen = 0  # bumped whenever the contents are replaced; stale builds are dropped
        self._convert_task: asyncio.Task[None] | None = None
        self._loaded = False
        self._follow_task: asyncio.Task[None] | None = None
        self._load_task: asyncio.Task[None] | None = None
        self._rescan = False
        # A connection that keeps dropping would otherwise rescan back to back.
        self._min_resync_s = min_resync_interval_s
        self._last_resync = -math.inf
        # Adds (True) and removals (False) seen while a scan runs, applied to its result.
        self._scan_ops: dict[bytes, bool] | None = None
        self._scan_cleared = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def is_bloom(self) -> bool:
        return self._bloom is not None or self._frozen is not None

    def __len__(self) -> int:
        n = len(self._keys)
        if self._frozen is not None:
            n += len(self._frozen)
        elif self._bloom is not None:
            n += self._bloom.count
        return n

    def may_contain(self, digest: bytes) -> bool:
        return digest in self or not self._loaded

    def __contains__(self, digest: bytes) -> bool:
        if digest in self._keys:
            return True
        if self._frozen is not None:
            return digest not in self._absent and digest in self._frozen
        if self._bloom is not None:
            return digest not in self._absent and digest in self._bloom
        return False

    def add(self, digest: bytes) -> None:
        if self._scan_ops is not None:
            self._scan_ops[digest] = True
        if self._bloom is not None or self._frozen is not None:
            self._absent.pop(digest, None)
            if self._frozen is None:
                assert self._bloom is not None
                self._bloom.add(digest)
                return
        self._keys.add(digest)
        if self._frozen is None and len(self._keys) > self._bloom_after:
            self._start_conversion()

    def discard(self, digest: bytes) -> None:
        if self._scan_ops is not None:
            self._scan_ops[digest] = False
        self._keys.discard(digest)
        if self._bloom is not None or self._frozen is not None:
            self._absent[digest] = None
            self._absent.move_to_end(digest)
            while len(self._absent) > _MAX_ABSENT:
                self._absent.popitem(last=False)

    def clear(self) -> None:
        if self._scan_ops is not None:
            self._scan_cleared = True
        self._replace(set(), None)

    def _replace(self, keys: set[bytes], bloom: BloomFilter | None) -> None:
        self._gen += 1
        self._keys = keys
        self._frozen = None
        self._bloom = bloom
        self._absent.clear()

    # --- Bloom conversion -----------------------------------------------------

    def _capacity(self, n: int) -> int:
        return max(n * _BLOOM_HEADROOM, _BLOOM_MIN_CAPACITY)

    def _start_conversion(self) -> None:
        frozen, gen = self._keys, self._gen
        self._frozen, self._keys = frozen, set()
        try:
            asyncio.get_running_loop()
        except RuntimeError:  # no event loop (scripts, tests): build in place
            self._finish_conversion(
                BloomFilter.from_digests(frozen, self._capacity(len(frozen))), gen
            )
            return
        self._convert_task = asyncio.ensure_future(self._convert(frozen, gen))

    async def _convert(self, frozen: set[bytes], gen: int) -> None:
        # ``frozen`` is never modified again (adds go to _keys, removals to _absent),
        # so the thread can read it while the loop keeps answering lookups.
        bloom = await asyncio.to_thread(
            BloomFilter.from_digests, frozen, self._capacity(len(frozen))
        )
        self._finish_conversion(bloom, gen)

    def _finish_conversion(self, bloom: BloomFilter, gen: int) -> None:
        self._convert_task = None
        if gen != self._gen:
            return  # cleared or replaced by a resync while the filter was built
        for d in self._keys:
            bloom.add(d)
        self._bloom, self._frozen, self._keys = bloom, None, set()

    # --- loading ------------------------------------------------------------

    def start(self) -> None:
        if self._follow_task is not None:
            return
        self._follow_task = asyncio.ensure_future(self._follow())
        self._load_task = asyncio.ensure_future(self._load())

    def resync(self) -> None:
        """Scan the store again (now, or as soon as a scan in progress ends; at most once
        per ``min_resync_interval_s``) and replace the index with what it holds."""
        self._rescan = True
        if self._load_task is None or self._load_task.done():
            self._load_task = asyncio.ensure_future(self._load())

    async def _load(self) -> None:
        backoff = 0.5
        initial = not self._loaded and not self._rescan
        self._rescan = False
        while True:
            if not initial:
                wait = self._last_resync + self._min_resync_s - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_resync = time.monotonic()
            try:
                self._scan_ops, self._scan_cleared = {}, False
                fresh: set[bytes] = set()
                async for d in self._store.scan():
                    fresh.add(d)
                    if len(fresh) % 10_000 == 0:
                        await asyncio.sleep(0)  # a scan that never awaits must not stall calls
                bloom = None
                if len(fresh) > self._bloom_after:
                    bloom = await asyncio.to_thread(
                        BloomFilter.from_digests, fresh, self._capacity(len(fresh))
                    )
                ops, cleared = self._scan_ops, self._scan_cleared
                if cleared:
                    self._rescan = True  # the namespace was wiped mid-scan: results are stale
                else:
                    self._install(fresh, bloom, ops)
                    self._loaded = True
                    logger.debug(f"memcat: key index loaded {len(self)} keys")
                    if self._on_resync is not None:
                        self._on_resync()
                if not self._rescan:
                    return
                self._rescan = False  # a reconnect or wipe during the scan: scan once more
                initial = False
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"memcat: key index scan failed ({type(e).__name__}); retrying")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                self._scan_ops = None

    def _install(
        self, fresh: set[bytes], bloom: BloomFilter | None, ops: dict[bytes, bool]
    ) -> None:
        """Replace the contents with a scan's result plus what changed while it ran."""
        if bloom is None:
            for d, present in ops.items():
                if present:
                    fresh.add(d)
                else:
                    fresh.discard(d)
            self._replace(fresh, None)
            return
        self._replace(set(), bloom)
        for d, present in ops.items():
            if present:
                bloom.add(d)
            elif d in bloom:
                self._absent[d] = None
        while len(self._absent) > _MAX_ABSENT:
            self._absent.popitem(last=False)

    async def _follow(self) -> None:
        backoff = 0.5
        first = True
        while True:
            if not first:
                self.resync()  # notifications sent since the last subscription are lost
            first = False
            began = time.monotonic()
            try:
                async for d in self._store.subscribe_new():
                    if len(d) == 16:
                        self.add(d)
                    elif d == RESYNC:
                        self.resync()
                    elif d == PURGE_ALL:
                        self.clear()
                        if self._on_purge is not None:
                            self._on_purge(None)
                    elif len(d) == 17 and d[:1] == PURGE_ONE:
                        self.discard(d[1:])
                        if self._on_purge is not None:
                            self._on_purge([d[1:]])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"memcat: key index subscription failed ({type(e).__name__})")
            if time.monotonic() - began >= _FOLLOW_HEALTHY_S:
                backoff = 0.5  # it was up for a while: a fresh outage starts the backoff over
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def stop(self) -> None:
        tasks = [
            t for t in (self._follow_task, self._load_task, self._convert_task) if t is not None
        ]
        self._follow_task = self._load_task = None
        if self._convert_task is not None:
            self._convert_task = None
            self._gen += 1  # the cancelled build never lands; the next add converts again
            if self._frozen is not None:
                self._keys |= self._frozen.difference(self._absent)
                self._frozen = None
                self._absent.clear()
        for t in tasks:
            t.cancel()
        # Collects the tasks' own cancellation; a cancellation of stop() itself still raises.
        await asyncio.gather(*tasks, return_exceptions=True)
