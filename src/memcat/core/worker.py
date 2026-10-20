"""Background work: admission, captures, renders, writes. Never blocks a call."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from loguru import logger

from memcat.core import audio, codec
from memcat.core.audio import LoudnessTracker
from memcat.core.breaker import Breaker
from memcat.core.codec import Entry
from memcat.core.index import KeyIndex
from memcat.core.lookup import Clip, L1Cache
from memcat.core.metrics import Metrics
from memcat.core.store.base import DEFAULT_TTL_S, BoundStore, StoreError

__all__ = ["Audio", "ObserveItem", "Policy", "RenderFn", "Tombstones", "Worker", "WorkerConfig"]

# Bounds on per-digest bookkeeping, so memory stays flat however many clauses pass.
_MAX_TOMBSTONES = 100_000  # beyond this, purges older than the window are forgotten
_MAX_FAILED = 50_000  # renders that recently failed (not retried for render_retry_s)
_MAX_TOUCHED = 200_000  # digests whose TTL was recently refreshed
_MAX_RECENT_PUTS = 100_000
# Longer than a purge notification takes to arrive: Redis pub/sub in milliseconds, S3
# events within events_poll_s (10 s by default) plus 60 s of tolerated clock skew.
_RECENT_PUT_S = 600.0
# A provider word clock this far from the clip's speech onset counts from elsewhere.
_WORD_DRIFT_S = 0.15
# What _store_audio did with a clip.
_STORED, _QUIET, _SKIPPED = "stored", "quiet", "skipped"


@dataclass(frozen=True, slots=True)
class Audio:
    """Synthesized speech for one clause: PCM16 mono plus word start times in seconds."""

    pcm: bytes
    sample_rate: int
    words: tuple[tuple[str, float], ...] = ()


RenderFn = Callable[[], Awaitable[Audio | None]]


class Tombstones:
    """Recent purges. Work that started before a purge of its digest (a queued
    capture, a running render, a lookup already in flight) must not bring the
    entry back."""

    WINDOW_S = 600.0

    def __init__(self) -> None:
        # Oldest mark first, so expiry pops from the front: O(1) amortized per mark.
        self._at: OrderedDict[bytes, float] = OrderedDict()
        self._epoch = float("-inf")  # last purge of the whole namespace

    def mark(self, digests: Sequence[bytes] | None) -> None:
        now = time.monotonic()
        if digests is None:
            self._epoch = now
            self._at.clear()
            return
        at = self._at
        for d in digests:
            at[d] = now
            at.move_to_end(d)
        if len(at) > _MAX_TOMBSTONES:
            cutoff = now - self.WINDOW_S
            while at:
                d, t = next(iter(at.items()))
                if t > cutoff:
                    break
                del at[d]

    def __len__(self) -> int:
        return len(self._at)

    def blocks(self, digest: bytes, since: float) -> bool:
        """True when ``digest`` was purged at or after ``since``."""
        if since <= self._epoch:
            return True
        t = self._at.get(digest)
        return t is not None and since <= t


@dataclass(frozen=True, slots=True)
class Policy:
    """Per-service rules that travel with each job: services sharing a runtime may
    differ (one provider forbids lossy codecs, another admits after 5 calls)."""

    admit_after_calls: int = 3
    codec: str = "auto"
    lossless_only: bool = False
    ttl_s: int = DEFAULT_TTL_S
    render: bool = True


@dataclass(slots=True)
class ObserveItem:
    digest: bytes
    chars: int
    captured: Audio | None = None
    render: RenderFn | None = None
    policy: Policy | None = None
    created: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class WorkerConfig:
    """Process-wide worker settings. ``policy`` applies to jobs that carry none."""

    policy: Policy = field(default_factory=Policy)
    queue_size: int = 1_000
    # Audio bytes queued jobs may hold; observes and loudness samples past it are dropped.
    queue_max_bytes: int = 64 * 1024 * 1024
    max_renders: int = 2
    render_queue_size: int = 256
    render_retry_s: float = 3600.0
    touch_every_s: float = 3600.0
    lock_ttl_s: int = 60
    drain_s: float = 2.0


@dataclass(slots=True)
class _Job:
    kind: str
    args: tuple[object, ...] = field(default_factory=tuple)
    nbytes: int = 0  # audio held by the job, counted against ``queue_max_bytes``


class Worker:
    """One per process, store and namespace, shared by every call."""

    def __init__(
        self,
        store: BoundStore,
        index: KeyIndex,
        l1: L1Cache,
        breaker: Breaker,
        metrics: Metrics,
        loudness: LoudnessTracker,
        config: WorkerConfig,
    ):
        self._store = store
        self._index = index
        self._l1 = l1
        self._breaker = breaker
        self._metrics = metrics
        self._loudness = loudness
        self.cfg = config
        self._q: asyncio.Queue[_Job] = asyncio.Queue(config.queue_size)
        self._rq: asyncio.Queue[tuple[bytes, RenderFn, int, Policy, float]] = asyncio.Queue(
            config.render_queue_size
        )
        self._tasks: list[asyncio.Task[None]] = []
        self._pending_renders: set[bytes] = set()
        self._failed_at: OrderedDict[bytes, float] = OrderedDict()
        self._touched_at: OrderedDict[bytes, float] = OrderedDict()
        self._put_at: OrderedDict[bytes, float] = OrderedDict()  # oldest first
        self._touch_pending: set[bytes] = set()  # queued or running touches
        self._promote_pending: set[bytes] = set()  # one decode per clip, however many hits
        self._queued_bytes = 0
        self._live_loudness_at: dict[str, float] = {}
        self.tomb = Tombstones()
        self._idle = asyncio.Event()
        self._idle.set()
        self._busy = 0

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._tasks:
            return
        self._tasks.append(asyncio.ensure_future(self._run()))
        for _ in range(self.cfg.max_renders):
            self._tasks.append(asyncio.ensure_future(self._run_renders()))

    async def drain(self, timeout: float | None = None) -> None:
        """Wait until queued jobs (not renders) are done, at most ``timeout`` seconds."""
        deadline = time.monotonic() + (self.cfg.drain_s if timeout is None else timeout)
        while (not self._q.empty() or self._busy) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

    async def drain_all(self, timeout: float = 30.0) -> None:
        """Wait for jobs and renders; for tests, ``warm`` and the CLI."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (
                self._q.empty()
                and self._rq.empty()
                and not self._busy
                and not self._pending_renders
            ):
                return
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        await self.drain()
        tasks, self._tasks = self._tasks, []
        for t in tasks:
            t.cancel()
        # Collects the tasks' own cancellation; a cancellation of stop() itself still raises.
        await asyncio.gather(*tasks, return_exceptions=True)

    # --- submission (hot path: never awaits) --------------------------------

    def _submit(self, kind: str, *args: object, nbytes: int = 0) -> bool:
        if nbytes and self._queued_bytes + nbytes > self.cfg.queue_max_bytes:
            self._metrics.inc("dropped_jobs")
            self._metrics.inc("dropped_bytes", nbytes)
            return False
        try:
            self._q.put_nowait(_Job(kind, args, nbytes))
        except asyncio.QueueFull:
            self._metrics.inc("dropped_jobs")
            return False
        self._queued_bytes += nbytes
        return True

    def observe(self, call_id: str, items: Sequence[ObserveItem]) -> None:
        if items:
            n = sum(len(it.captured.pcm) for it in items if it.captured is not None)
            self._submit("observe", call_id, list(items), nbytes=n)

    def warm(self, digest: bytes, render: RenderFn, policy: Policy | None = None) -> None:
        if digest in self._index or self._l1.get(digest) is not None:
            return
        self._queue_render(
            digest,
            render,
            codec.FLAG_RENDERED | codec.FLAG_WARMED,
            policy or self.cfg.policy,
            time.monotonic(),
        )

    def touch(self, digests: Sequence[bytes], ttl_s: int | None = None) -> None:
        """Refresh TTLs at most once per ``touch_every_s`` per digest. The refresh is
        recorded only once the store has applied it, so a dropped or failed touch is
        retried on the next hit."""
        now = time.monotonic()
        due: list[bytes] = []
        for d in digests:
            if d in self._touch_pending:
                continue
            last = self._touched_at.get(d)
            if last is None or now - last >= self.cfg.touch_every_s:
                due.append(d)
        if due and self._submit("touch", due, ttl_s or self.cfg.policy.ttl_s):
            self._touch_pending.update(due)

    def _touched(self, digests: list[bytes]) -> None:
        now = time.monotonic()
        for d in digests:
            self._touched_at[d] = now
            self._touched_at.move_to_end(d)
        while len(self._touched_at) > _MAX_TOUCHED:
            self._touched_at.popitem(last=False)

    def purged_elsewhere(self, digests: Sequence[bytes] | None) -> None:
        """Another process purged these (None: everything). Writes this process made
        shortly before may have landed after that purge's delete: delete them again."""
        at = self._put_at
        cutoff = time.monotonic() - _RECENT_PUT_S
        while at and next(iter(at.values())) < cutoff:
            at.popitem(last=False)
        again = list(at) if digests is None else [d for d in digests if d in at]
        for d in again:
            del at[d]
        if again:
            self._submit("delete", again)

    def delete_corrupt(self, digest: bytes) -> None:
        self._index.discard(digest)
        self._l1.discard(digest)
        self._submit("delete", [digest])

    def promote(self, digest: bytes, entry: Entry) -> None:
        if digest in self._promote_pending:
            return
        if self._submit("promote", digest, entry, time.monotonic()):
            self._promote_pending.add(digest)

    def live_loudness(self, voice: str, pcm: bytes, sample_rate: int) -> None:
        now = time.monotonic()
        if now - self._live_loudness_at.get(voice, 0.0) < 5.0 or len(pcm) < sample_rate * 2:
            return
        self._live_loudness_at[voice] = now
        self._submit("loudness", voice, pcm, sample_rate, nbytes=len(pcm))

    # --- processing ----------------------------------------------------------

    async def _run(self) -> None:
        while True:
            job = await self._q.get()
            self._busy += 1
            try:
                await self._handle(job)
            except asyncio.CancelledError:
                self._breaker.abandon()  # the job may have held the breaker's probe
                raise
            except Exception as e:
                # Some backends echo the request (clause text) in their errors: details
                # go to DEBUG only.
                logger.warning(f"memcat: worker job {job.kind} failed: {type(e).__name__}")
                logger.debug(f"memcat: worker job {job.kind} failed: {e}")
            finally:
                self._busy -= 1
                self._queued_bytes -= job.nbytes
                self._q.task_done()

    async def _handle(self, job: _Job) -> None:
        if job.kind == "observe":
            call_id, items = job.args
            assert isinstance(call_id, str) and isinstance(items, list)
            await self._observe(call_id, items)
        elif job.kind == "touch":
            ds, ttl = job.args
            assert isinstance(ds, list) and isinstance(ttl, int)
            try:
                if self._breaker.allow():
                    try:
                        await self._store.touch(ds, ttl)
                    except StoreError:
                        self._breaker.failure()
                        self._metrics.inc("store_errors")
                    else:
                        self._breaker.success()
                        self._touched(ds)
            finally:
                self._touch_pending.difference_update(ds)
        elif job.kind == "delete":
            (ds,) = job.args
            assert isinstance(ds, list)
            with contextlib.suppress(StoreError):
                await self._store.delete(ds)
        elif job.kind == "promote":
            d, e, since = job.args
            assert isinstance(d, bytes) and isinstance(e, Entry) and isinstance(since, float)
            try:
                pcm = await asyncio.to_thread(codec.to_pcm16, e)
            finally:
                self._promote_pending.discard(d)
            if self.tomb.blocks(d, since):
                return
            self._l1.put(
                d,
                Clip(
                    e.sample_rate,
                    e.duration_ms,
                    e.words,
                    e.lufs,
                    pcm=pcm,
                    flags=e.flags,
                    nbytes=len(pcm),
                ),
            )
        elif job.kind == "loudness":
            voice, pcm, sr = job.args
            assert isinstance(voice, str) and isinstance(pcm, bytes) and isinstance(sr, int)
            v = await asyncio.to_thread(audio.loudness, pcm, sr)
            if v is not None:
                self._loudness.add(voice, v)

    async def _observe(self, call_id: str, items: list[ObserveItem]) -> None:
        for it in items:
            try:
                if not await self._observe_one(call_id, it):
                    return  # the store is unavailable: the rest would fail the same way
            except asyncio.CancelledError:
                raise
            except Exception as e:  # one bad item never costs the rest of the batch
                logger.warning(f"memcat: observe failed: {type(e).__name__}")
                logger.debug(f"memcat: observe {it.digest.hex()} failed: {e}")

    async def _observe_one(self, call_id: str, it: ObserveItem) -> bool:
        """Count one sighting and store or render the clause once admitted. False when
        the breaker is open."""
        d = it.digest
        pol = it.policy or self.cfg.policy
        if d in self._index or d in self._pending_renders or self._l1.get(d) is not None:
            return True
        k = pol.admit_after_calls
        if k <= 1:
            admitted = True
        else:
            if not self._breaker.allow():
                return False
            try:
                admitted = await self._store.observe(d, call_id, k)
                self._breaker.success()
            except StoreError:
                self._breaker.failure()
                self._metrics.inc("store_errors")
                return True
        if not admitted or self.tomb.blocks(d, it.created):
            return True
        if it.captured is not None:
            await self._store_audio(d, it.captured, codec.FLAG_CAPTURED, pol, it.created)
        elif it.render is not None and pol.render:
            self._queue_render(d, it.render, codec.FLAG_RENDERED, pol, it.created)
        return True

    def _queue_render(
        self, digest: bytes, render: RenderFn, flags: int, policy: Policy, since: float
    ) -> None:
        if digest in self._pending_renders:
            return
        failed = self._failed_at.get(digest)
        if failed is not None and time.monotonic() - failed < self.cfg.render_retry_s:
            return
        try:
            self._rq.put_nowait((digest, render, flags, policy, since))
            self._pending_renders.add(digest)
        except asyncio.QueueFull:
            self._metrics.inc("dropped_jobs")

    async def _run_renders(self) -> None:
        while True:
            d, render, flags, pol, since = await self._rq.get()
            try:
                await self._render_one(d, render, flags, pol, since)
            except asyncio.CancelledError:
                self._breaker.abandon()
                raise
            except Exception as e:
                logger.warning(f"memcat: render failed: {type(e).__name__}")
                logger.debug(f"memcat: render {d.hex()} failed: {e}")
                self._mark_failed(d)
            finally:
                self._pending_renders.discard(d)
                self._rq.task_done()

    def _mark_failed(self, d: bytes) -> None:
        self._metrics.inc("render_failures")
        self._failed_at[d] = time.monotonic()
        while len(self._failed_at) > _MAX_FAILED:
            self._failed_at.popitem(last=False)

    async def _render_one(
        self, d: bytes, render: RenderFn, flags: int, pol: Policy, since: float
    ) -> None:
        if d in self._index:
            return
        if not self._breaker.allow():
            return  # the store is down: the render could not be stored
        try:
            if not self._index.loaded:
                # The index may not know this entry yet; never pay to render it twice.
                # Off the call path, so the bulk budget where the store has one.
                read = getattr(self._store, "get_bulk", self._store.get_many)
                (existing,) = await read([d])
                if existing is not None:
                    self._breaker.success()
                    self._index.add(d)
                    return
            locked = await self._store.try_lock(d, self.cfg.lock_ttl_s)
        except StoreError:
            self._breaker.failure()
            self._metrics.inc("store_errors")
            return
        self._breaker.success()
        if not locked:
            return
        try:
            result = await render()
            if result is None or not result.pcm:
                self._mark_failed(d)
                return
            self._metrics.inc("renders")
            if await self._store_audio(d, result, flags, pol, since) == _QUIET:
                # Nothing but silence: storing it is pointless, and rendering it again on
                # every admission would pay for it again. Back off like any failure.
                self._mark_failed(d)
        finally:
            with contextlib.suppress(StoreError):
                await self._store.unlock(d)

    def choose_codec(self, sample_rate: int, policy: Policy | None = None) -> int:
        pol = policy or self.cfg.policy
        lossless = pol.lossless_only
        mode = pol.codec
        if lossless or mode == "pcm16":
            return codec.CODEC_PCM16  # lossless_only overrides any lossy choice
        if mode == "mulaw":
            return codec.CODEC_MULAW
        opus_ok = sample_rate in codec.OPUS_RATES and codec.opus_available()
        if mode == "opus":
            return codec.CODEC_OPUS if opus_ok else codec.CODEC_PCM16
        if sample_rate <= 8000:  # auto
            return codec.CODEC_MULAW
        return codec.CODEC_OPUS if opus_ok else codec.CODEC_PCM16

    async def _store_audio(self, d: bytes, a: Audio, flags: int, pol: Policy, since: float) -> str:
        """Store one clip. Returns ``"stored"``, ``"quiet"`` (no speech in it) or
        ``"skipped"`` (purged meanwhile, or the store failed)."""
        if self.tomb.blocks(d, since):
            return _SKIPPED
        cod = self.choose_codec(a.sample_rate, pol)
        blob, clip = await asyncio.to_thread(_prepare, a, cod, flags)
        if blob is None or clip is None:
            return _QUIET
        if self.tomb.blocks(d, since):
            return _SKIPPED
        try:
            await self._store.put(d, blob, pol.ttl_s)
        except StoreError:
            self._metrics.inc("store_errors")
            self._breaker.failure()
            return _SKIPPED
        self._breaker.success()
        self._put_at[d] = time.monotonic()
        self._put_at.move_to_end(d)
        while len(self._put_at) > _MAX_RECENT_PUTS:
            self._put_at.popitem(last=False)
        if self.tomb.blocks(d, since):  # purged while the write was in flight
            with contextlib.suppress(StoreError):
                await self._store.delete([d])
            return _SKIPPED
        self._index.add(d)
        self._l1.put(d, clip)
        self._metrics.inc("bytes_stored", len(blob))
        self._metrics.inc("entries_stored")
        if flags & codec.FLAG_CAPTURED:
            self._metrics.inc("captures")
        with contextlib.suppress(StoreError):
            await self._store.publish_new(d)
        logger.debug(f"memcat: stored {d.hex()} ({len(blob)} bytes, codec {cod})")
        return _STORED


def _prepare(a: Audio, cod: int, flags: int) -> tuple[bytes | None, Clip | None]:
    """Trim, fade, measure and encode one clip (runs in a thread)."""
    sr = a.sample_rate
    if len(a.pcm) % 2:  # a stray half sample at the end
        a = Audio(a.pcm[:-1], sr, a.words)
    pcm, cut = audio.trim_silence(a.pcm, sr)
    if not pcm:
        return None, None
    pcm = audio.fade_edges(pcm, sr)
    lufs = audio.loudness(pcm, sr)
    shift_ms = cut * 1000 / sr
    raw_words = a.words
    if raw_words:
        # A provider's word clock can run ahead of this clip (it counts audio from an
        # earlier request). Re-anchor the first word where the speech starts, but only
        # when that start is known (leading silence was found) or the times cannot
        # belong to this clip; with a noise floor and no silence, trust the provider.
        onset_s = (cut + (sr * audio.EDGE_MS // 1000 if cut else 0)) / sr
        first = raw_words[0][1]
        drift = first - onset_s
        impossible = first < 0 or first >= len(a.pcm) / (2 * sr)
        if impossible or (cut > 0 and abs(drift) > _WORD_DRIFT_S):
            raw_words = tuple((w, t - drift) for w, t in raw_words)
    words = tuple((w, max(0, round(t * 1000 - shift_ms))) for w, t in raw_words if w and w.strip())
    duration_ms = len(pcm) * 1000 // (2 * sr)
    payload = codec.encode_audio(pcm, sr, cod)
    entry = Entry(
        codec=cod,
        sample_rate=sr,
        duration_ms=duration_ms,
        lufs=lufs,
        words=words,
        audio=payload,
        flags=flags,
    )
    blob = codec.encode(entry)
    # L1 holds exactly what any other process would decode from the store.
    decoded = codec.decode(blob)
    out = codec.to_pcm16(decoded)
    clip = Clip(sr, duration_ms, words, lufs, pcm=out, flags=flags, nbytes=len(out))
    return blob, clip
