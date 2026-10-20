"""Hot-path lookup: L1 (decoded clips in process), then the key index, then one MGET."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from memcat.core import codec
from memcat.core.breaker import Breaker
from memcat.core.codec import CorruptEntryError, Entry
from memcat.core.index import KeyIndex
from memcat.core.metrics import Metrics
from memcat.core.store.base import BoundStore, StoreError

__all__ = ["Clip", "L1Cache", "Lookup"]

# The store read on the call path; a slower read is a miss and a store error. A store
# with its own budget (``read_timeout_s``: Redis ``op_timeout_ms``) gets that budget.
_READ_BUDGET_S = 0.1
_READ_MARGIN_S = 0.05


@dataclass(slots=True)
class Clip:
    """A clip ready to replay.

    ``pcm`` is set for clips decoded up front. Opus clips fetched from the store
    keep ``entry`` and are decoded whole before they play.
    """

    sample_rate: int
    duration_ms: int
    words: tuple[tuple[str, int], ...]
    lufs: float | None
    pcm: bytes | None = None
    entry: Entry | None = None
    flags: int = 0
    nbytes: int = field(default=0)

    def frames(self, frame_ms: int = 40) -> Iterator[bytes]:
        if self.pcm is not None:
            step = max(2, self.sample_rate * frame_ms // 1000 * 2)
            for i in range(0, len(self.pcm), step):
                yield self.pcm[i : i + step]

    @classmethod
    def from_entry(cls, e: Entry) -> Clip:
        if e.codec == codec.CODEC_OPUS:
            return cls(
                e.sample_rate,
                e.duration_ms,
                e.words,
                e.lufs,
                entry=e,
                flags=e.flags,
                nbytes=len(e.audio),
            )
        pcm = codec.to_pcm16(e)
        return cls(
            e.sample_rate, e.duration_ms, e.words, e.lufs, pcm=pcm, flags=e.flags, nbytes=len(pcm)
        )


# Approximate bytes a cached clip costs beyond its PCM (Clip object, dict slot, key).
_ENTRY_OVERHEAD = 128


class L1Cache:
    """Byte-bounded LRU of decoded clips, shared by every call in the process."""

    def __init__(self, max_bytes: int):
        self._max = max_bytes
        self._items: OrderedDict[bytes, Clip] = OrderedDict()
        self._size = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def size_bytes(self) -> int:
        return self._size

    def get(self, digest: bytes) -> Clip | None:
        clip = self._items.get(digest)
        if clip is not None:
            self._items.move_to_end(digest)
        return clip

    def put(self, digest: bytes, clip: Clip) -> None:
        old = self._items.pop(digest, None)
        if old is not None:
            self._size -= old.nbytes + _ENTRY_OVERHEAD
        if clip.nbytes + _ENTRY_OVERHEAD > self._max:
            return
        self._items[digest] = clip
        self._size += clip.nbytes + _ENTRY_OVERHEAD
        while self._size > self._max and self._items:
            _, c = self._items.popitem(last=False)
            self._size -= c.nbytes + _ENTRY_OVERHEAD

    def discard(self, digest: bytes) -> None:
        old = self._items.pop(digest, None)
        if old is not None:
            self._size -= old.nbytes + _ENTRY_OVERHEAD

    def clear(self) -> None:
        self._items.clear()
        self._size = 0

    def digests(self) -> list[bytes]:
        """A snapshot of the digests held."""
        return list(self._items)


def _never(_d: bytes, _t: float) -> bool:
    return False


class Lookup:
    """Resolve the digests of one sentence with at most one store round trip."""

    def __init__(
        self,
        store: BoundStore,
        l1: L1Cache,
        index: KeyIndex,
        breaker: Breaker,
        metrics: Metrics,
        on_hits: Callable[[list[bytes], int | None], None],
        on_corrupt: Callable[[bytes], None],
        on_decoded: Callable[[bytes, Entry], None],
        purged: Callable[[bytes, float], bool] | None = None,
    ):
        self._store = store
        self._l1 = l1
        self._index = index
        self._breaker = breaker
        self._metrics = metrics
        self._on_hits = on_hits
        self._on_corrupt = on_corrupt
        self._on_decoded = on_decoded
        self._purged: Callable[[bytes, float], bool] = purged or _never

    async def get_many(
        self, digests: Sequence[bytes | None], ttl_s: int | None = None
    ) -> list[Clip | None]:
        started = time.monotonic()
        out: list[Clip | None] = [None] * len(digests)
        want: list[int] = []
        for i, d in enumerate(digests):
            if d is None:
                continue
            clip = self._l1.get(d)
            if clip is not None:
                out[i] = clip
            elif self._index.may_contain(d):
                want.append(i)
        touched: list[bytes] = [d for i, d in enumerate(digests) if d is not None and out[i]]
        if want and self._breaker.allow():
            uniq: list[bytes] = []
            for i in want:
                d = digests[i]
                assert d is not None
                if d not in uniq:
                    uniq.append(d)
            try:
                own = getattr(self._store, "read_timeout_s", None)
                budget = _READ_BUDGET_S if own is None else own + _READ_MARGIN_S
                async with asyncio.timeout(budget):
                    blobs = await self._store.get_many(uniq)
            except asyncio.CancelledError:
                self._breaker.abandon()  # a barge-in must not leave a probe outstanding
                raise
            except (StoreError, TimeoutError):
                self._breaker.failure()
                self._metrics.inc("store_errors")
                blobs = None
            else:
                self._breaker.success()
            if blobs is not None:
                found: dict[bytes, Clip] = {}
                for d, blob in zip(uniq, blobs, strict=True):
                    if blob is None:
                        if self._index.loaded:
                            self._index.discard(d)  # expired since the index learned of it
                        continue
                    try:
                        e = codec.decode(blob)
                    except CorruptEntryError:
                        self._metrics.inc("corrupt_entries")
                        self._on_corrupt(d)
                        continue
                    if e.codec == codec.CODEC_OPUS and not codec.opus_available():
                        continue  # written by a process with PyAV; a miss here, not corrupt
                    if self._purged(d, started):
                        continue  # deleted while this read was in flight
                    clip = Clip.from_entry(e)
                    found[d] = clip
                    self._index.add(d)
                    if clip.pcm is not None:
                        self._l1.put(d, clip)
                    else:
                        self._on_decoded(d, e)  # worker decodes Opus fully into L1
                for i in want:
                    d = digests[i]
                    assert d is not None
                    if d in found:
                        out[i] = found[d]
                        touched.append(d)
        if touched:
            self._on_hits(touched, ttl_s)
        return out
