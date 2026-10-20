"""MemcatTTSMixin: the cache, placed first in the MRO of a Pipecat TTS service."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from memcat.config import MemcatConfig
from memcat.core import audio as audio_ops
from memcat.core import codec as audio_codec
from memcat.core.chunker import Clause, split
from memcat.core.keys import KeyContext, Keyer, SettingsNotSerializableError
from memcat.core.lookup import Clip
from memcat.core.metrics import Metrics, price_for
from memcat.core.pcm import bytes_to_seconds
from memcat.core.runtime import Runtime, get_runtime
from memcat.core.worker import Audio, ObserveItem
from memcat.pipecat import compat
from memcat.pipecat.adapters.base import FlushBudget, Mode, ProviderAdapter
from memcat.pipecat.adapters.providers import select_adapter
from memcat.pipecat.contexts import CtxState, HttpCtx, LeadCtx, Segment, Stream
from memcat.pipecat.render import RenderPool

__all__ = ["MemcatTTSMixin"]

FRAME_MS = 40
_MAX_CAPTURE_S = 30.0
_LOUDNESS_SAMPLE_S = 4.0
_LOUDNESS_EVERY_S = 5.0  # a service offers one live loudness sample per voice at most this often
_STREAM_GIVEUP_S = 30.0  # longest a stream asked to end may wait to be sent
_KEEPALIVE_TICK_S = 0.25
_PROVIDER_LEAD_MS = 50  # typical lead silence of a clause synthesized alone
_MAX_HTTP_CTXS = 64  # HTTP contexts end without telling the service; keep the newest
_MAX_SPAN_TOTALS = 256  # per-context tracing totals kept for open spans
_RELEASE_TIMEOUT_S = 2.5  # wait for the runtime to flush on stop, then give up
_HOLD_ALL_MS = 60_000  # tail_keep_ms that holds every trailing silence until the stream ends
_MAX_CLOSED = 256  # stream ids Memcat closed itself; a late provider reply to one is expected
_CLEAR_WAIT_S = 2.0  # longest audio is dropped waiting for a "Cleared" (cursor routing)
_DEAD_CTX = "memcat-cleared"  # where audio sent before a "Cleared" is routed (and dropped)
_WORD_SLACK_S = 0.05  # word times may start this far before their audio

# Release flushes a cancelled service starts but does not wait for.
_DETACHED: set[asyncio.Future[Any]] = set()
# Opus clips being decoded for replay, so concurrent hits on one clip decode it once.
_DECODING: dict[tuple[int, bytes], asyncio.Future[bytes]] = {}


class _ProviderError(Exception):
    """An exception raised by the wrapped provider, carried past Memcat's fallback."""

    def __init__(self, original: BaseException):
        super().__init__(str(original))
        self.original = original


async def _provider_frames(
    gen: AsyncGenerator[Frame | None, None],
) -> AsyncGenerator[Frame | None, None]:
    """Iterate a provider's ``run_tts``, carrying what it raises as :class:`_ProviderError`."""
    async with contextlib.aclosing(gen):
        while True:
            try:
                f = await gen.__anext__()
            except StopAsyncIteration:
                return
            except Exception as e:
                raise _ProviderError(e) from e
            yield f


# Generic HTTP-mode classes whose run_tts was seen to yield audio inline. A service
# that yields None and delivers audio from a background task must not be replayed
# around (its audio would land after the cached clips), so it is probed first.
_INLINE_AUDIO: dict[type, bool] = {}


@dataclass(slots=True)
class _Run:
    hit: bool
    idx: list[int]
    text: str


@dataclass(slots=True)
class _Plan:
    clauses: list[Clause]
    digests: list[bytes | None]
    clips: list[Clip | None]
    key: KeyContext
    runs: list[_Run]
    settings: Any = None  # settings snapshot taken with ``key``, for renders of these clauses


async def _decode_shared(clip: Clip, digest: bytes | None) -> bytes:
    """Decode an Opus clip off the loop; concurrent replays of one digest share it."""
    if clip.entry is None:
        return b""
    entry = clip.entry
    if digest is None:
        return await asyncio.to_thread(audio_codec.to_pcm16, entry)
    loop = asyncio.get_running_loop()
    k = (id(loop), digest)
    fut = _DECODING.get(k)
    if fut is None:
        fut = asyncio.ensure_future(asyncio.to_thread(audio_codec.to_pcm16, entry))
        _DECODING[k] = fut
        fut.add_done_callback(lambda _f: _DECODING.pop(k, None))
    # One caller's barge-in must not cancel the decode the others wait on.
    return await asyncio.shield(fut)


def _rebase_words(
    raw: Sequence[tuple[str, float]], base: float, dur_s: float, *, exact: bool
) -> list[tuple[str, float]] | None:
    """Word times of one live run, made relative to the run's own audio.

    ``base`` is the provider's clock when the run began (``exact``) or Memcat's
    count of the context's live audio. A provider that restarts its clock per
    request reports offsets already relative; the first word tells which. Every
    word is then read the same way, and a set that does not fit the audio (out of
    range or out of order) means the clock was not what was assumed: ``None``,
    and the clip is not kept.
    """
    if not raw:
        return []
    off = base
    if not exact and raw[0][1] - base < -_WORD_SLACK_S:
        off = 0.0
    out: list[tuple[str, float]] = []
    prev = float("-inf")
    for w, ts in raw:
        rel = ts - off
        if rel < -_WORD_SLACK_S or rel > dur_s + _WORD_SLACK_S or rel < prev - 1e-6:
            return None
        prev = rel
        out.append((w, max(0.0, rel)))
    return out


def _forget_detached(fut: asyncio.Future[Any]) -> None:
    _DETACHED.discard(fut)
    if not fut.cancelled() and fut.exception() is not None:
        logger.debug(f"memcat: background release failed: {type(fut.exception()).__name__}")


class MemcatTTSMixin:
    """Caches clause audio for any Pipecat ``TTSService``. Use :func:`memcat.cached`."""

    _memcat_wrapped: type[TTSService]

    # ------------------------------------------------------------------ setup

    def __init__(self, *args: Any, memcat: MemcatConfig | None = None, **kwargs: Any):
        self._memcat_args = args
        self._memcat_kwargs = dict(kwargs)
        self._memcat_cfg = memcat or MemcatConfig()
        self._memcat_ctx: dict[str, CtxState] = {}
        self._memcat_streams: dict[str, Stream] = {}
        self._memcat_inflight: list[Stream] = []
        self._memcat_waiting: deque[Stream] = deque()
        self._memcat_http: dict[str, HttpCtx] = {}
        self._memcat_lead: dict[str, LeadCtx] = {}
        self._memcat_rt: Runtime | None = None
        self._memcat_pool: RenderPool | None = None
        self._memcat_call_id = uuid.uuid4().hex
        self._memcat_metrics = Metrics()
        self._memcat_warned: set[str] = set()
        self._memcat_snap: tuple[KeyContext | None, Any] = (None, None)
        self._memcat_ifs = False
        self._memcat_span_totals: dict[str, list[int]] = {}
        self._memcat_seq = 0
        self._memcat_tasks: set[asyncio.Task[None]] = set()  # stream drains and watchdogs
        self._memcat_closed: dict[str, None] = {}  # stream ids closed by Memcat, oldest first
        self._memcat_clears = 0  # "Clear"s sent whose "Cleared" has not come back
        self._memcat_clear_deadline = 0.0
        self._memcat_clear_ws: Any = None
        self._memcat_watched_ws: Any = None  # the socket whose "Cleared" replies are seen
        self._memcat_interrupting = False
        self._memcat_loudness_at: dict[str, float] = {}
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]
        svc: Any = self
        base = type(self)._memcat_wrapped
        self._memcat_adapter: ProviderAdapter = select_adapter(base)
        self._memcat_mode: Mode = self._memcat_adapter.mode_for(svc)
        self._memcat_keyer = Keyer(
            f"{base.__module__}.{base.__qualname__}",
            self._memcat_adapter.init_identity(self._memcat_kwargs),
        )
        self._memcat_budget: FlushBudget | None = self._memcat_adapter.flush_budget(
            self._memcat_cfg.max_stream_flushes_per_min
        )
        if self._memcat_adapter.watermarks_audio and not self._memcat_cfg.lossless_only:
            self._memcat_cfg = dataclasses.replace(self._memcat_cfg, lossless_only=True)
        self._memcat_policy = self._memcat_cfg.policy()
        self._memcat_disabled = not self._memcat_cfg.enabled or self._memcat_mode == "passthrough"
        if self._memcat_mode == "passthrough" and self._memcat_cfg.enabled:
            self._memcat_warn(
                "passthrough",
                f"memcat: {base.__name__} is a WebSocket service Memcat cannot replay into "
                "safely; caching is off for it",
            )
        if compat.is_token_mode(svc):
            self._memcat_disabled = True
            self._memcat_warn(
                "token", "memcat: TOKEN aggregation mode is not supported; caching is off"
            )

    def _memcat_warn(self, key: str, msg: str) -> None:
        if key not in self._memcat_warned:
            self._memcat_warned.add(key)
            logger.warning(msg)

    def _memcat_disable(self, why: str, exc: BaseException | None = None) -> None:
        """Turn Memcat off for this service. ``why`` must never contain clause text: it
        is logged at WARNING with only the exception type; the message goes to DEBUG."""
        self._memcat_disabled = True
        self._memcat_metrics.inc("fallbacks")
        detail = f" ({type(exc).__name__})" if exc is not None else ""
        self._memcat_warn("disabled", f"memcat: disabled for this service: {why}{detail}")
        if exc is not None:
            logger.debug(f"memcat: {why}: {type(exc).__name__}: {exc}")

    def _memcat_spawn(self, coro: Any, name: str) -> None:
        """Start a background task that stop and cancel will cancel."""
        task = self.create_task(coro, name)  # type: ignore[attr-defined]
        self._memcat_tasks.add(task)
        task.add_done_callback(self._memcat_tasks.discard)

    # -------------------------------------------------------------- lifecycle

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)  # type: ignore[misc]
        self._memcat_call_id = uuid.uuid4().hex
        if self._memcat_disabled:
            return
        try:
            rt = get_runtime(self._memcat_cfg)
            rt.acquire()
            self._memcat_rt = rt
            if self._memcat_cfg.render or self._memcat_cfg.warm:
                self._memcat_pool = RenderPool.shared(
                    type(self)._memcat_wrapped, self._memcat_args, self._memcat_kwargs
                )
            if self._memcat_cfg.warm:
                self._memcat_warm(self._memcat_cfg.warm)
        except Exception as e:
            self._memcat_disable("could not start", e)

    async def _memcat_release(self, wait: bool = True) -> None:
        tasks, self._memcat_tasks = list(self._memcat_tasks), set()
        for t in tasks:
            with contextlib.suppress(Exception):
                await self.cancel_task(t)  # type: ignore[attr-defined]
        rt, self._memcat_rt = self._memcat_rt, None
        if rt is None:
            return
        if wait:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(rt.release(), timeout=_RELEASE_TIMEOUT_S)
            return
        # A cancelled pipeline must stop now: the runtime flushes in the background.
        fut = asyncio.ensure_future(rt.release())
        _DETACHED.add(fut)
        fut.add_done_callback(_forget_detached)

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)  # type: ignore[misc]
        await self._memcat_release()

    async def cancel(self, frame: Any) -> None:
        await super().cancel(frame)  # type: ignore[misc]
        await self._memcat_release(wait=False)

    async def cleanup(self) -> None:
        await super().cleanup()  # type: ignore[misc]
        await self._memcat_release()

    # ------------------------------------------------------------ public API

    async def memcat_stats(self) -> dict[str, Any]:
        """Counters for this service, plus process-wide cache state."""
        out = self._memcat_metrics.snapshot()
        rt = self._memcat_rt
        if rt is not None:
            out["process"] = rt.metrics.snapshot()
            out["process"]["index_keys"] = len(rt.index)
            out["process"]["index_loaded"] = rt.index.loaded
            out["process"]["l1_entries"] = len(rt.l1)
            out["process"]["l1_bytes"] = rt.l1.size_bytes
            out["process"]["breaker_open"] = rt.breaker.is_open
        out["mode"] = self._memcat_mode
        out["adapter"] = self._memcat_adapter.name
        out["enabled"] = not self._memcat_disabled
        return out

    async def memcat_purge(self, text: str | None = None) -> int:
        """Delete cached audio and admission counts for ``text`` and each of its
        clauses, in every voice, setting and sample rate; with no text, everything."""
        rt = self._memcat_rt or get_runtime(self._memcat_cfg)
        if text is None:
            return await rt.purge(None)
        return await rt.purge_text(text, self._memcat_cfg)

    # ---------------------------------------------------------------- planning

    def _memcat_split(self, text: str) -> list[Clause]:
        return split(
            text,
            min_words=self._memcat_cfg.min_clause_words,
            extra_blocks=self._memcat_cfg.compiled_blocks,
        )

    def _memcat_key(self) -> KeyContext:
        svc: Any = self
        return self._memcat_keyer.context(compat.given_settings(svc), svc.sample_rate)

    def _memcat_settings_snapshot(self, key: KeyContext) -> Any:
        k, snap = self._memcat_snap
        if k is not key:
            snap = copy.deepcopy(compat.settings_object(self))  # type: ignore[arg-type]
            self._memcat_snap = (key, snap)
        return snap

    async def _memcat_plan(self, text: str) -> _Plan:
        rt = self._memcat_rt
        assert rt is not None
        clauses = self._memcat_split(text)
        key = self._memcat_key()
        digests: list[bytes | None] = [
            key.digest(c.text) if c.cacheable and c.text.strip() else None for c in clauses
        ]
        clips = await rt.lookup.get_many(digests, self._memcat_cfg.ttl_s)
        sr: int = self.sample_rate  # type: ignore[attr-defined]
        runs: list[_Run] = []
        for i, c in enumerate(clauses):
            clip = clips[i]
            hit = clip is not None and clip.sample_rate == sr and clip.duration_ms > 0
            if not hit:
                clips[i] = None
            if runs and runs[-1].hit == hit:
                runs[-1].idx.append(i)
                runs[-1].text += c.text
            else:
                runs.append(_Run(hit, [i], c.text))
        # Taken now, not when the audio is observed: the voice may change before then,
        # and a render must use the settings these digests were keyed under.
        settings = None
        if self._memcat_pool is not None and self._memcat_cfg.render:
            settings = self._memcat_settings_snapshot(key)
        return _Plan(clauses, digests, clips, key, runs, settings)

    def _memcat_account(
        self,
        plan: _Plan,
        live_idx: Sequence[int],
        cached_idx: Sequence[int],
        cid: str | None = None,
    ) -> None:
        m = self._memcat_metrics
        prm = self._memcat_rt.metrics if self._memcat_rt else None
        cached_chars = sum(len(plan.clauses[i].text) for i in cached_idx)
        live_chars = sum(len(plan.clauses[i].text) for i in live_idx)
        for mm in (m, prm):
            if mm is None:
                continue
            mm.inc("hits", len(cached_idx))
            mm.inc("misses", len(live_idx))
            mm.inc("chars_cached", cached_chars)
            mm.inc("chars_live", live_chars)
            mm.usd_saved += cached_chars * self._memcat_price_now() / 1e6
        if cid is not None:
            self._memcat_trace(cid, cached_chars, live_chars)

    def _memcat_trace(self, cid: str, cached_chars: int, live_chars: int) -> None:
        span = compat.tts_span(self, cid)
        if span is None:
            return
        try:
            tot = self._memcat_span_totals.setdefault(cid, [0, 0])
            tot[0] += cached_chars
            tot[1] += live_chars
            while len(self._memcat_span_totals) > _MAX_SPAN_TOTALS:
                self._memcat_span_totals.pop(next(iter(self._memcat_span_totals)))
            span.set_attribute("memcat.hit", tot[0] > 0)
            span.set_attribute("memcat.chars_cached", tot[0])
            span.set_attribute("memcat.chars_live", tot[1])
            span.set_attribute("memcat.adapter", self._memcat_adapter.name)
            span.set_attribute("memcat.mode", self._memcat_mode)
        except Exception:  # tracing must never break a call
            return

    def _memcat_price_now(self) -> float:
        try:
            model = compat.given_settings(self).get("model")  # type: ignore[arg-type]
        except Exception:
            model = None
        return price_for(
            self._memcat_adapter.name,
            model if isinstance(model, str) else None,
            self._memcat_cfg.prices,
        )

    def _memcat_observe_item(
        self, plan: _Plan, i: int, captured: Audio | None = None
    ) -> ObserveItem | None:
        d = plan.digests[i]
        if d is None:
            return None
        render = None
        pool = self._memcat_pool
        if pool is not None and captured is None and plan.settings is not None:
            render = pool.render_fn(
                plan.key.identity,
                plan.clauses[i].text,
                plan.settings,
                self.sample_rate,  # type: ignore[attr-defined]
                self._memcat_count_render,
            )
        return ObserveItem(d, len(plan.clauses[i].text), captured, render, self._memcat_policy)

    def _memcat_count_render(self, chars: int) -> None:
        self._memcat_metrics.inc("chars_rendered", chars)
        if self._memcat_rt is not None:
            self._memcat_rt.metrics.inc("chars_rendered", chars)

    def _memcat_observe(self, items: Sequence[ObserveItem | None]) -> None:
        rt = self._memcat_rt
        real = [it for it in items if it is not None]
        if rt is not None and real:
            rt.worker.observe(self._memcat_call_id, real)

    def _memcat_warm(self, phrases: Sequence[str]) -> None:
        rt, pool = self._memcat_rt, self._memcat_pool
        if rt is None or pool is None:
            return
        key = self._memcat_key()
        snap = self._memcat_settings_snapshot(key)
        for phrase in phrases:
            for c in self._memcat_split(phrase):
                if not c.cacheable or not c.text.strip():
                    continue
                d = key.digest(c.text)
                fn = pool.render_fn(
                    key.identity,
                    c.text,
                    snap,
                    self.sample_rate,  # type: ignore[attr-defined]
                    self._memcat_count_render,
                )
                rt.worker.warm(d, fn, self._memcat_policy)

    def _memcat_gap_ms(self, plan: _Plan, i: int) -> float:
        """Silence to add after clause ``i`` (both sides of the join are tight)."""
        if not self._memcat_cfg.trim_joins:
            return 0.0
        text = plan.clauses[i].text
        if i == len(plan.clauses) - 1:
            return max(0, audio_ops.SENTENCE_GAP_MS - audio_ops.EDGE_MS)
        pause = audio_ops.join_pause_ms(text, self._memcat_cfg.clause_pause_ms)
        return max(0, pause - 2 * audio_ops.EDGE_MS)

    def _memcat_gain(self, key: KeyContext, clip: Clip) -> float:
        if not self._memcat_cfg.gain_match or self._memcat_rt is None:
            return 0.0
        return self._memcat_rt.loudness.gain_for(key.voice_id, clip.lufs)

    async def _memcat_clip_chunks(
        self, key: KeyContext, clip: Clip, digest: bytes | None = None
    ) -> list[bytes]:
        """A clip's audio as gain-matched frames, decoded whole before any is sent,
        so a bad entry fails before the listener hears part of its clause.

        Decoded PCM is only sliced here. An Opus clip (fetched from the store, not yet
        promoted to L1) is decoded in a thread: ~5 ms per second of audio would
        otherwise stall every call on the loop. The listener waits for the decode
        either way; the thread hop adds well under a millisecond.
        """
        gain = self._memcat_gain(key, clip)
        if clip.pcm is None:
            pcm = await _decode_shared(clip, digest)
            step = max(2, clip.sample_rate * FRAME_MS // 1000 * 2)
            chunks = [pcm[i : i + step] for i in range(0, len(pcm), step)]
        else:
            chunks = list(clip.frames(FRAME_MS))
        return [audio_ops.apply_gain(c, gain) for c in chunks] if gain else chunks

    def _memcat_capture(self, pcm: bytearray, words: Sequence[tuple[str, float]]) -> Audio:
        return Audio(bytes(pcm), self.sample_rate, tuple(words))  # type: ignore[attr-defined]

    def _memcat_live_sample(self, key: KeyContext | None, pcm: bytearray) -> None:
        """Hand the worker the first seconds of a live stream for loudness matching.
        Checked before copying anything: most streams are not sampled."""
        rt = self._memcat_rt
        sr: int = self.sample_rate  # type: ignore[attr-defined]
        if rt is None or key is None or not self._memcat_cfg.gain_match:
            return
        if bytes_to_seconds(len(pcm), sr) < 1.0:
            return
        now = time.monotonic()
        if now - self._memcat_loudness_at.get(key.voice_id, -1e9) < _LOUDNESS_EVERY_S:
            return
        self._memcat_loudness_at[key.voice_id] = now
        rt.worker.live_loudness(key.voice_id, bytes(pcm[: int(_LOUDNESS_SAMPLE_S * 2 * sr)]), sr)

    def _memcat_ttfa(self, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1000.0
        self._memcat_metrics.observe_ttfa(ms)
        if self._memcat_rt is not None:
            self._memcat_rt.metrics.observe_ttfa(ms)

    # ----------------------------------------------------------------- run_tts

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:  # type: ignore[override]
        t0 = time.perf_counter()
        mode = self._memcat_mode
        managed = context_id in self._memcat_ctx
        gen: AsyncGenerator[Frame | None, None]
        if (self._memcat_disabled or self._memcat_rt is None) and not managed:
            gen = super().run_tts(text, context_id)  # type: ignore[misc]
        else:
            plan: _Plan | None = None
            if not self._memcat_disabled and self._memcat_rt is not None:
                try:
                    plan = await self._memcat_plan(text)
                except SettingsNotSerializableError as e:
                    self._memcat_disable(f"settings cannot be keyed ({e})")
                except Exception as e:
                    self._memcat_disable("planning failed", e)
            if mode == "segmented":
                gen = self._memcat_run_segmented(text, context_id, plan, t0)
            elif plan is None:
                gen = super().run_tts(text, context_id)  # type: ignore[misc]
            elif mode == "http":
                base = type(self)._memcat_wrapped
                verified = self._memcat_adapter.certified or _INLINE_AUDIO.get(base)
                if verified is None:
                    # First sentence of an unknown HTTP service: nothing is replayed, and
                    # whether its audio comes back inline is checked as it plays.
                    plan = _Plan(
                        plan.clauses,
                        plan.digests,
                        [None] * len(plan.clauses),
                        plan.key,
                        [_Run(False, list(range(len(plan.clauses))), text)],
                        plan.settings,
                    )
                if verified is False:
                    self._memcat_disable("audio arrives outside run_tts; caching is off")
                    gen = super().run_tts(text, context_id)  # type: ignore[misc]
                else:
                    gen = self._memcat_run_http(text, context_id, plan, t0)
            else:
                gen = self._memcat_run_leading(text, context_id, plan, t0)
        async with contextlib.aclosing(gen) as frames:
            async for f in frames:
                yield f

    # --------------------------------------------------------------- HTTP mode

    async def _memcat_run_http(
        self, text: str, cid: str, plan: _Plan, t0: float
    ) -> AsyncGenerator[Frame | None, None]:
        h = self._memcat_http.get(cid)
        if h is None:
            h = HttpCtx()
            self._memcat_http[cid] = h
            if len(self._memcat_http) > _MAX_HTTP_CTXS:
                self._memcat_http.pop(next(iter(self._memcat_http)))
        sr: int = self.sample_rate  # type: ignore[attr-defined]
        words_on = compat.uses_word_timestamps(self)  # type: ignore[arg-type]
        observe: list[ObserveItem | None] = []
        live_idx: list[int] = []
        cached_idx: list[int] = []
        first_audio = True
        pending_stop: TTSStoppedFrame | None = None
        done_chars = 0  # characters heard or handed to the provider: never sent again
        try:
            for r_i, run in enumerate(plan.runs):
                last = r_i == len(plan.runs) - 1
                if run.hit:
                    if not compat.push_start_frame(self) and not self.audio_context_available(cid):  # type: ignore[arg-type,attr-defined]
                        # The service opens its own context in run_tts, which has not
                        # run yet: open it the way the service would.
                        await self.create_audio_context(cid)  # type: ignore[attr-defined]
                        await self.start_ttfb_metrics()  # type: ignore[attr-defined]
                        yield TTSStartedFrame(context_id=cid)
                    for i in run.idx:
                        clip = plan.clips[i]
                        assert clip is not None
                        chunks = await self._memcat_clip_chunks(plan.key, clip, plan.digests[i])
                        if words_on and clip.words:
                            await super().add_word_timestamps(  # type: ignore[misc]
                                [(w, h.emitted_s + off / 1000.0) for w, off in clip.words],
                                cid,
                                includes_inter_frame_spaces=self._memcat_ifs or None,
                            )
                        done_chars += len(plan.clauses[i].text)
                        for chunk in chunks:
                            dur = bytes_to_seconds(len(chunk), sr)
                            h.emitted_s += dur
                            h.cached_s += dur
                            if first_audio:
                                first_audio = False
                                self._memcat_ttfa(t0)
                            yield TTSAudioRawFrame(chunk, sr, 1, context_id=cid)
                        cached_idx.append(i)
                        gap = self._memcat_gap_ms(plan, i)
                        if gap:
                            sil = self._memcat_silence(cid, gap)
                            dur = bytes_to_seconds(len(sil.audio), sr)
                            h.emitted_s += dur
                            h.cached_s += dur
                            yield sil
                        # Keep provider continuity (ElevenLabs previous_text) in order.
                        self._memcat_adapter.on_cached_text(self, plan.clauses[i].text)  # type: ignore[arg-type]
                    continue
                # live run
                single = len(run.idx) == 1 and plan.digests[run.idx[0]] is not None
                h.live_active = True
                h.capture = single
                h.word_clock = self._memcat_adapter.word_clock(self)  # type: ignore[arg-type]
                h.capture_base_s = h.live_s
                h.pcm = bytearray()
                h.words = []
                h.held = []
                h.trimmer = None
                after_hit = r_i > 0 and plan.runs[r_i - 1].hit
                before_hit = r_i + 1 < len(plan.runs) and plan.runs[r_i + 1].hit
                if self._memcat_cfg.trim_joins and (after_hit or before_hit):
                    # A clause synthesized alone carries edge silence it would not
                    # have mid-sentence; at a join Memcat sets the pause itself.
                    h.trimmer = audio_ops.EdgeTrimmer(
                        sr,
                        trim_lead=after_hit,
                        lead_keep_ms=audio_ops.EDGE_MS,
                        tail_keep_ms=audio_ops.EDGE_MS if before_hit else None,
                    )
                errored = False
                saw_none = inline_audio = False
                broken: Exception | None = None  # Memcat failed mid-run; the rest plays as is
                try:
                    async with contextlib.aclosing(
                        _provider_frames(super().run_tts(run.text, cid))  # type: ignore[misc]
                    ) as gen:
                        # Once asked, the provider speaks this run: it is never sent again.
                        done_chars += len(run.text)
                        async for f in gen:
                            if f is None:
                                saw_none = True
                            elif isinstance(f, ErrorFrame):
                                errored = True
                            elif isinstance(f, TTSStoppedFrame) and not last:
                                pending_stop = f  # keep one stop frame, at the end of the sentence
                                continue
                            elif isinstance(f, TTSStartedFrame) and r_i > 0:
                                continue  # one started frame per sentence
                            elif isinstance(f, TTSAudioRawFrame):
                                if not inline_audio:
                                    _INLINE_AUDIO.setdefault(type(self)._memcat_wrapped, True)
                                inline_audio = True
                                if broken is None:
                                    try:
                                        out = await self._memcat_http_audio(h, cid, f, sr)
                                    except Exception as e:
                                        broken = e
                                        out = await self._memcat_http_unhook(h, cid, sr)
                                        out.append(f)
                                    for o in out:
                                        h.emitted_s += bytes_to_seconds(len(o.audio), o.sample_rate)
                                        yield o
                                    continue
                                h.live_s += bytes_to_seconds(len(f.audio), f.sample_rate)
                                h.emitted_s += bytes_to_seconds(len(f.audio), f.sample_rate)
                            yield f
                    if broken is not None:
                        raise broken
                    tr = h.trimmer
                    if tr is not None:
                        rest = tr.finish()
                        h.trimmer = None
                        await self._memcat_release_http_words(h, cid, tr.lead_trimmed_s)
                        h.trimmed_s += tr.lead_trimmed_s + tr.tail_trimmed_s
                        if rest:
                            h.emitted_s += bytes_to_seconds(len(rest), sr)
                            yield TTSAudioRawFrame(rest, sr, 1, context_id=cid)
                        if before_hit:
                            gap = self._memcat_gap_ms(plan, run.idx[-1])
                            if gap:
                                sil = self._memcat_silence(cid, gap)
                                dur = bytes_to_seconds(len(sil.audio), sr)
                                h.emitted_s += dur
                                h.cached_s += dur
                                yield sil
                finally:
                    h.live_active = False
                live_idx.extend(run.idx)
                if saw_none and not inline_audio and not errored:
                    # This service delivers audio from a background task: anything
                    # replayed now would play before it. Say the rest live, stop.
                    _INLINE_AUDIO[type(self)._memcat_wrapped] = False
                    self._memcat_disable("audio arrives outside run_tts; caching is off")
                    rest = text[done_chars:]
                    done_chars = len(text)
                    live_idx.extend(i for r in plan.runs[r_i + 1 :] for i in r.idx)
                    if rest.strip():
                        async with contextlib.aclosing(
                            _provider_frames(super().run_tts(rest, cid))  # type: ignore[misc]
                        ) as gen2:
                            async for f2 in gen2:
                                yield f2
                    pending_stop = None
                    break
                cap = None
                if single and not errored and h.capture and h.pcm:
                    exact = h.word_clock is not None  # h.words are already offsets
                    words = _rebase_words(
                        h.words,
                        0.0 if exact else h.capture_base_s,
                        bytes_to_seconds(len(h.pcm), sr),
                        exact=exact,
                    )
                    if words is not None:
                        cap = self._memcat_capture(h.pcm, words)
                observe.extend(self._memcat_observe_item(plan, i, cap) for i in run.idx)
                self._memcat_live_sample(plan.key, h.pcm)
            if pending_stop is not None and plan.runs and plan.runs[-1].hit:
                yield pending_stop
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except _ProviderError as pe:
            # The provider failed, not Memcat: behave exactly as plain Pipecat would.
            self._memcat_account(plan, live_idx, cached_idx, cid)
            raise pe.original from None
        except Exception as e:
            self._memcat_disable("replay failed; the rest is sent live", e)
            rest = text[done_chars:]
            if rest.strip():
                async with contextlib.aclosing(super().run_tts(rest, cid)) as gen3:  # type: ignore[misc]
                    async for f in gen3:
                        yield f
        self._memcat_account(plan, live_idx, cached_idx, cid)
        self._memcat_observe(observe)

    async def _memcat_http_audio(
        self, h: HttpCtx, cid: str, f: TTSAudioRawFrame, sr: int
    ) -> list[TTSAudioRawFrame]:
        """Capture and join-trim one live audio frame; what to play in its place."""
        h.live_s += bytes_to_seconds(len(f.audio), f.sample_rate)
        native = f.sample_rate == sr and f.num_channels == 1
        if h.capture:
            if not native:
                h.capture = False  # a frame left out would leave a gap in the clip
            else:
                h.pcm.extend(f.audio)
                if bytes_to_seconds(len(h.pcm), sr) > _MAX_CAPTURE_S:
                    h.capture = False
        tr = h.trimmer
        if tr is None or not native:
            return [f]
        had = tr.seen_speech
        out = tr.feed(f.audio)
        if tr.seen_speech and not had:
            await self._memcat_release_http_words(h, cid)
        return [TTSAudioRawFrame(out, sr, 1, context_id=cid)] if out else []

    async def _memcat_http_unhook(self, h: HttpCtx, cid: str, sr: int) -> list[TTSAudioRawFrame]:
        """Memcat failed mid-run: stop touching this run, keep what was held back."""
        h.capture = False
        tr, h.trimmer = h.trimmer, None
        out: list[TTSAudioRawFrame] = []
        with contextlib.suppress(Exception):
            if tr is not None:
                rest = tr.finish()
                if rest:
                    out.append(TTSAudioRawFrame(rest, sr, 1, context_id=cid))
        held, h.held = h.held, []
        with contextlib.suppress(Exception):
            for w, ts, c, ifs in held:
                await super().append_to_audio_context(  # type: ignore[misc]
                    cid, compat.make_word_entry(w, ts + h.cached_s - h.trimmed_s, c, ifs)
                )
        return out

    def _memcat_http_word_shift(self, h: HttpCtx, lead: float | None = None) -> float:
        tr = h.trimmer
        lead_now = lead if lead is not None else (tr.lead_trimmed_s if tr is not None else 0.0)
        return h.cached_s - h.trimmed_s - lead_now

    async def _memcat_release_http_words(
        self, h: HttpCtx, cid: str, lead: float | None = None
    ) -> None:
        held, h.held = h.held, []
        shift = self._memcat_http_word_shift(h, lead)
        for w, ts, c, ifs in held:
            await super().append_to_audio_context(  # type: ignore[misc]
                cid, compat.make_word_entry(w, ts + shift, c, ifs)
            )

    # ------------------------------------------------------------ leading mode

    async def _memcat_run_leading(
        self, text: str, cid: str, plan: _Plan, t0: float
    ) -> AsyncGenerator[Frame | None, None]:
        lc = self._memcat_lead.get(cid)
        if lc is None:
            lc = LeadCtx()
            self._memcat_lead[cid] = lc
        cached_idx: list[int] = []
        live_idx: list[int] = []
        observe: list[ObserveItem | None] = []
        consumed = 0  # characters of clauses already replayed
        for run in plan.runs:
            if (
                run.hit
                and not lc.has_live
                and not self._memcat_disabled
                and compat.context_queue_exists(self, cid)  # type: ignore[arg-type]
            ):
                replayed: list[int] = []
                try:
                    await self._memcat_lead_replay(plan, run, cid, lc, t0, not cached_idx, replayed)
                except (GeneratorExit, asyncio.CancelledError):
                    raise
                except Exception as e:
                    self._memcat_disable("replay failed; the rest is sent live", e)
                cached_idx.extend(replayed)
                consumed += sum(len(plan.clauses[i].text) for i in replayed)
                if len(replayed) == len(run.idx):
                    continue
            # Live from the first clause not replayed (a failed run resumes mid-run).
            rest_idx = list(range(cached_idx[-1] + 1 if cached_idx else 0, len(plan.clauses)))
            if not text[consumed:].strip():
                break
            lc.has_live = True
            async with contextlib.aclosing(super().run_tts(text[consumed:], cid)) as _gen:  # type: ignore[misc]
                async for f in _gen:
                    yield f
            live_idx.extend(rest_idx)
            observe.extend(
                self._memcat_observe_item(plan, i) for i in rest_idx if plan.clips[i] is None
            )
            break
        self._memcat_account(plan, live_idx, cached_idx, cid)
        self._memcat_observe(observe)
        if not lc.has_live and cid != compat.turn_context_id(self):  # type: ignore[arg-type]
            await self.flush_audio(cid)  # a whole cached sentence context: close it now

    async def _memcat_lead_replay(
        self,
        plan: _Plan,
        run: _Run,
        cid: str,
        lc: LeadCtx,
        t0: float,
        first: bool,
        replayed: list[int],
    ) -> None:
        """Queue a run's clips; a clause counts as replayed from its first chunk."""
        sr: int = self.sample_rate  # type: ignore[attr-defined]
        words_on = compat.uses_word_timestamps(self)  # type: ignore[arg-type]
        for i in run.idx:
            clip = plan.clips[i]
            assert clip is not None
            # Everything is built before the first item is queued, and the clause
            # counts as replayed from its first audio on: a failure part-way through
            # cuts the clause short rather than sending it to the provider again.
            chunks = await self._memcat_clip_chunks(plan.key, clip, plan.digests[i])
            words = (
                [
                    compat.make_word_entry(w, lc.cached_s + off / 1000.0, cid, self._memcat_ifs)
                    for w, off in clip.words
                ]
                if words_on
                else []
            )
            for entry in words:
                await super().append_to_audio_context(cid, entry)  # type: ignore[misc]
            for n, chunk in enumerate(chunks):
                lc.cached_s += bytes_to_seconds(len(chunk), sr)
                if first:
                    first = False
                    self._memcat_ttfa(t0)
                if n == 0:
                    replayed.append(i)
                await super().append_to_audio_context(  # type: ignore[misc]
                    cid, TTSAudioRawFrame(chunk, sr, 1, context_id=cid)
                )
            if not chunks:
                replayed.append(i)
            gap = self._memcat_gap_ms(plan, i)
            if gap and i == run.idx[-1] and i < len(plan.clauses) - 1:
                # Live audio follows natively here, carrying the provider's own lead
                # silence (it cannot be trimmed in leading mode).
                gap = max(0.0, gap - _PROVIDER_LEAD_MS)
            if gap:
                sil = self._memcat_silence(cid, gap)
                lc.cached_s += bytes_to_seconds(len(sil.audio), sr)
                await super().append_to_audio_context(cid, sil)  # type: ignore[misc]

    # ---------------------------------------------------------- segmented mode

    async def _memcat_run_segmented(
        self, text: str, cid: str, plan: _Plan | None, t0: float
    ) -> AsyncGenerator[Frame | None, None]:
        st = self._memcat_ctx.get(cid)
        if st is None:
            st = CtxState(cid)
            self._memcat_ctx[cid] = st
        st.input_done = False
        if not compat.context_queue_exists(self, cid):  # type: ignore[arg-type]
            # The provider would open the context in its own run_tts (ElevenLabs).
            # Through self, so Pipecat's tracing (which wraps the instance's method)
            # opens the turn's span for it.
            await self.create_audio_context(cid)  # type: ignore[misc]
            await self.start_ttfb_metrics()  # type: ignore[attr-defined]
            await super().append_to_audio_context(cid, TTSStartedFrame(context_id=cid))  # type: ignore[misc]
        if plan is None:
            await self._memcat_send(st, text, [])
        else:
            done = [0]  # characters fully handled, so a failure resumes after them
            try:
                await self._memcat_segment_plan(st, plan, t0, done)
            except (GeneratorExit, asyncio.CancelledError):
                raise
            except Exception as e:
                self._memcat_disable("replay failed; the rest is sent live", e)
                rest = text[done[0] :]
                if rest.strip():
                    await self._memcat_send(st, rest, [])
        if cid != compat.turn_context_id(self):  # type: ignore[arg-type]
            # One context per sentence (reuse_context_id_within_turn=False): Pipecat
            # never flushes these, so end them here instead of stalling 3 s.
            await self._memcat_input_done(st)
        yield None

    async def _memcat_segment_plan(
        self, st: CtxState, plan: _Plan, t0: float, done: list[int]
    ) -> None:
        """Queue a sentence's runs in order. ``done[0]`` counts the characters already
        handed to the context or the provider, so a failure resumes after them."""
        sr: int = self.sample_rate  # type: ignore[attr-defined]
        words_on = compat.uses_word_timestamps(self)  # type: ignore[arg-type]
        cached_idx: list[int] = []
        live_idx: list[int] = []
        first = True
        prev_cached = False  # the previous run of this sentence was replayed from cache
        for r_i, run in enumerate(plan.runs):
            lead_gap = 0.0
            if run.hit and st.open_stream is not None:
                if self._memcat_budget is None or self._memcat_budget.can_spend_extra():
                    s = st.open_stream
                    st.open_stream = None
                    if r_i > 0:
                        # A live clause of this sentence meets this cached one.
                        s.tail_trim = True
                        lead_gap = self._memcat_gap_ms(plan, plan.runs[r_i - 1].idx[-1])
                    await self._memcat_end(s)
                else:
                    # No flush left in the provider's budget: say it live instead.
                    await self._memcat_send(
                        st, run.text, [(None, plan.clauses[i].text, False) for i in run.idx]
                    )
                    live_idx.extend(run.idx)
                    prev_cached = False
                    done[0] += len(run.text)
                    continue
            if run.hit:
                # Built whole before it is queued: a failure here leaves nothing heard.
                seg = Segment(done=True)
                offset = 0.0
                if lead_gap:
                    seg.items.append(self._memcat_silence(st.cid, lead_gap))
                    offset += lead_gap / 1000.0
                for i in run.idx:
                    clip = plan.clips[i]
                    assert clip is not None
                    chunks = await self._memcat_clip_chunks(plan.key, clip, plan.digests[i])
                    if words_on:
                        for w, off in clip.words:
                            seg.items.append(
                                compat.make_word_entry(
                                    w, offset + off / 1000.0, st.cid, self._memcat_ifs
                                )
                            )
                    for chunk in chunks:
                        offset += bytes_to_seconds(len(chunk), sr)
                        seg.items.append(TTSAudioRawFrame(chunk, sr, 1, context_id=st.cid))
                    gap = self._memcat_gap_ms(plan, i)
                    if gap:
                        seg.items.append(self._memcat_silence(st.cid, gap))
                        offset += gap / 1000.0
                    cached_idx.append(i)
                is_head = not st.segments
                st.segments.append(seg)
                done[0] += len(run.text)
                await self._memcat_release_ctx(st)
                if first and is_head:
                    first = False
                    self._memcat_ttfa(t0)
                prev_cached = True
            else:
                meta = [(plan.digests[i], plan.clauses[i].text, True) for i in run.idx]
                done[0] += len(run.text)  # once sent, it is the provider's to speak
                s = await self._memcat_send(st, run.text, meta, trim_lead=prev_cached)
                s.plans.append((plan, list(run.idx)))
                live_idx.extend(run.idx)
                prev_cached = False
        self._memcat_account(plan, live_idx, cached_idx, st.cid)

    # stream plumbing ------------------------------------------------------

    def _memcat_silence(self, cid: str, ms: float) -> TTSAudioRawFrame:
        sr: int = self.sample_rate  # type: ignore[attr-defined]
        return TTSAudioRawFrame(audio_ops.silence(sr, ms), sr, 1, context_id=cid)

    def _memcat_new_stream(self, st: CtxState, trim_lead: bool = False) -> Stream:
        self._memcat_seq += 1
        sid = f"{st.cid}-s{self._memcat_seq}"  # unique per service, even if the context id returns
        seg = Segment()
        s = Stream(
            sid=sid, cid=st.cid, seg=seg, registered=not self._memcat_adapter.self_creates_context
        )
        if self._memcat_cfg.trim_joins:
            # Trailing silence is held so it can be cut if a cached clip follows.
            s.trimmer = audio_ops.EdgeTrimmer(
                self.sample_rate,  # type: ignore[attr-defined]
                trim_lead=trim_lead,
                lead_keep_ms=audio_ops.EDGE_MS,
                tail_keep_ms=_HOLD_ALL_MS,
            )
        st.segments.append(seg)
        st.streams.append(s)
        st.open_stream = s
        self._memcat_streams[sid] = s
        self._memcat_waiting.append(s)
        return s

    async def _memcat_send(
        self,
        st: CtxState,
        text: str,
        meta: list[tuple[bytes | None, str, bool]],
        trim_lead: bool = False,
    ) -> Stream:
        s = st.open_stream or self._memcat_new_stream(st, trim_lead)
        s.clauses.extend(meta)
        if len(s.clauses) != 1:
            s.pcm = bytearray()
            s.words.clear()
        if s.state == "sent":
            await self._memcat_send_text(s, text)
            return s
        s.backlog.append(text)
        if (
            s.state == "waiting"
            and not self._memcat_interrupting
            and self._memcat_waiting
            and self._memcat_waiting[0] is s
            and len(self._memcat_inflight) < self._memcat_adapter.max_inflight
        ):
            self._memcat_waiting.popleft()
            s.state = "sending"
            s.started_at = time.monotonic()
            self._memcat_inflight.append(s)
            await self._memcat_drain(s)
        return s

    async def _memcat_drain(self, s: Stream) -> None:
        if s.plans and self._memcat_settings_changed(s):
            s.capture = False  # queued across a settings update: said in the new voice
        try:
            while s.backlog and s.state != "done":
                await self._memcat_send_text(s, s.backlog.popleft())
            if s.state == "sending":
                s.state = "sent"
            if s.end_requested and not s.ended and s.state != "done":
                await self._memcat_end(s)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # The provider failed: report it the way Pipecat would without Memcat.
            s.error = True
            with contextlib.suppress(Exception):
                await self.push_error(  # type: ignore[attr-defined]
                    error_msg=f"TTS provider error ({type(e).__name__})", exception=e
                )
            await self._memcat_stream_done(s)

    def _memcat_settings_changed(self, s: Stream) -> bool:
        try:
            key = self._memcat_key()
        except Exception:
            return True
        return any(plan.key != key for plan, _ in s.plans)

    async def _memcat_send_text(self, s: Stream, text: str) -> None:
        async with contextlib.aclosing(super().run_tts(text, s.sid)) as _gen:  # type: ignore[misc]
            async for f in _gen:
                if f is not None:
                    await self._memcat_route(s, f)

    async def _memcat_end(self, s: Stream) -> None:
        if s.state != "done" and not s.watched:
            # From here the context must outlive Pipecat's silence timeout until the
            # stream finishes, or cached clauses queued behind it would be lost.
            s.watched = True
            self._memcat_spawn(self._memcat_watch(s), f"memcat-watch-{s.sid}")
        if s.state in ("waiting", "sending"):
            s.end_requested = True
            return
        if s.ended or s.state == "done":
            return
        s.ended = True
        if self._memcat_budget is not None:
            st = self._memcat_ctx.get(s.cid)
            first = st is not None and not st.base_flush_done
            if st is not None:
                st.base_flush_done = True
            self._memcat_budget.spend(extra=not first)
        s.ended_at = time.monotonic()
        await self._memcat_adapter.end_stream(self, s.sid)

    async def _memcat_watch(self, s: Stream) -> None:
        """Keep an ended stream's context alive until the stream finishes, and close
        a stream whose provider goes quiet without confirming it (a dropped ``done``).

        Pipecat ends a context after ``stop_frame_timeout_s`` without an item, which
        would drop the cached clauses queued behind a slow stream; Memcat holds the
        context open instead and applies the same timeout to the stream itself.
        """
        grace = compat.stop_frame_timeout(self)  # type: ignore[arg-type]
        began = time.monotonic()
        while True:
            compat.keep_context_alive(self, s.cid)  # type: ignore[arg-type]
            await asyncio.sleep(_KEEPALIVE_TICK_S)
            if s.state == "done" or s.cid not in self._memcat_ctx:
                return
            now = time.monotonic()
            if not s.ended:
                # Queued behind another stream, or its text is still going out.
                if now - max(began, s.last_item_at) > _STREAM_GIVEUP_S:
                    await self._memcat_give_up(s, "never started")
                    return
                continue
            quiet = now - max(s.started_at, s.ended_at, s.last_item_at)
            if quiet > grace:
                await self._memcat_give_up(
                    s, "went quiet without finishing" if s.got_audio else "sent no audio"
                )
                return

    async def _memcat_give_up(self, s: Stream, why: str) -> None:
        """Close a stream the provider never finished: free its slot so later streams
        run, release what is queued behind it, and stop the provider generating it."""
        self._memcat_warn("stream-lost", f"memcat: a provider stream {why}; closing it")
        logger.debug(f"memcat: stream {s.sid} {why}; closing it")
        s.error = True  # a clip cut short must never be cached
        sent = s.state in ("sending", "sent")
        if sent:
            # Before the next stream can start: on a cursor-routed socket, its audio
            # must not be mixed with this one's.
            await self._memcat_abandon([s])
        await self._memcat_stream_done(s)

    async def _memcat_input_done(self, st: CtxState) -> None:
        st.input_done = True
        s = st.open_stream
        if s is not None:
            st.open_stream = None
            await self._memcat_end(s)
        await self._memcat_maybe_finalize(st)

    def _memcat_pump(self) -> None:
        if self._memcat_interrupting:
            return  # every context is being dropped: nothing queued may start now
        while (
            self._memcat_waiting and len(self._memcat_inflight) < self._memcat_adapter.max_inflight
        ):
            s = self._memcat_waiting.popleft()
            if s.state != "waiting":
                continue
            s.state = "sending"
            s.started_at = time.monotonic()
            self._memcat_inflight.append(s)
            self._memcat_spawn(self._memcat_drain(s), f"memcat-stream-{s.sid}")

    async def _memcat_stream_done(self, s: Stream) -> None:
        if s.state == "done":
            return
        try:
            for w, ts in self._memcat_adapter.pending_words(self):  # type: ignore[arg-type]
                await self._memcat_route(s, compat.make_word_entry(w, ts, s.sid, self._memcat_ifs))
        except Exception:
            logger.debug("memcat: could not flush the provider's last word")
        try:
            await self._memcat_finish_trim(s)
        except Exception:
            logger.debug("memcat: could not finish join trimming")
        s.state = "done"
        s.seg.done = True
        if s in self._memcat_inflight:
            self._memcat_inflight.remove(s)
        with contextlib.suppress(ValueError):
            self._memcat_waiting.remove(s)
        self._memcat_streams.pop(s.sid, None)
        self._memcat_forget_stream(s.sid)
        st = self._memcat_ctx.get(s.cid)
        if st is not None and st.open_stream is s:
            st.open_stream = None
        with contextlib.suppress(Exception):
            await self._memcat_adapter.on_stream_done(self, s.sid)
        self._memcat_observe_stream(s)
        if st is not None:
            await self._memcat_release_ctx(st)
        self._memcat_pump()

    def _memcat_observe_stream(self, s: Stream) -> None:
        plans: list[tuple[_Plan, list[int]]] = s.plans
        if not plans:
            return
        single = len(s.clauses) == 1 and s.clauses[0][0] is not None and not s.error and s.capture
        items: list[ObserveItem | None] = []
        for plan, idx in plans:
            for i in idx:
                cap = self._memcat_capture(s.pcm, s.words) if single and s.pcm else None
                items.append(self._memcat_observe_item(plan, i, cap))
        self._memcat_observe(items)
        self._memcat_live_sample(plans[-1][0].key, s.pcm)

    async def _memcat_route(self, s: Stream, item: Any) -> None:
        st = self._memcat_ctx.get(s.cid)
        if st is None or s.state == "done":
            return
        s.last_item_at = time.monotonic()
        if isinstance(item, (TTSStartedFrame, TTSStoppedFrame)):
            return  # the context has exactly one of each, owned by Memcat
        if isinstance(item, TTSAudioRawFrame):
            if s.error:
                return
            s.got_audio = True
            sr = item.sample_rate
            native = sr == self.sample_rate and item.num_channels == 1  # type: ignore[attr-defined]
            if s.capture and len(s.clauses) == 1:
                if not native:
                    s.capture = False
                else:
                    s.pcm.extend(item.audio)
                    if bytes_to_seconds(len(s.pcm), sr) > _MAX_CAPTURE_S:
                        s.capture = False  # too long to cache; keep what is here for loudness
            tr = s.trimmer
            if tr is not None and native:
                had = tr.seen_speech
                out = tr.feed(item.audio)
                if tr.seen_speech and not had:
                    await self._memcat_release_stream_words(st, s)
                if not out:
                    return
                item = TTSAudioRawFrame(out, sr, 1, context_id=s.sid)
        elif compat.is_word_entry(item):
            w, ts, c, ifs = compat.word_entry_fields(item)
            self._memcat_ifs = bool(ifs)
            if len(s.clauses) == 1:
                s.words.append((w, float(ts)))
            tr = s.trimmer
            if tr is not None:
                if not tr.seen_speech:
                    s.held.append(item)  # its shift is known once speech starts
                    return
                item = compat.make_word_entry(w, ts - tr.lead_trimmed_s, c, ifs)
        elif isinstance(item, ErrorFrame):
            s.error = True
        await self._memcat_push(st, s.seg, item)

    async def _memcat_release_stream_words(self, st: CtxState, s: Stream) -> None:
        held, s.held = s.held, []
        lead = s.trimmer.lead_trimmed_s if s.trimmer is not None else 0.0
        for it in held:
            w, ts, c, ifs = compat.word_entry_fields(it)
            await self._memcat_push(st, s.seg, compat.make_word_entry(w, ts - lead, c, ifs))

    async def _memcat_finish_trim(self, s: Stream) -> None:
        tr, s.trimmer = s.trimmer, None
        st = self._memcat_ctx.get(s.cid)
        if tr is None or st is None:
            return
        rest = tr.finish(audio_ops.EDGE_MS if s.tail_trim else None)
        s.trimmer = tr  # so held words get the lead shift
        await self._memcat_release_stream_words(st, s)
        s.trimmer = None
        if rest and not s.error:
            await self._memcat_push(
                st,
                s.seg,
                TTSAudioRawFrame(rest, self.sample_rate, 1, context_id=s.cid),  # type: ignore[attr-defined]
            )

    async def _memcat_push(self, st: CtxState, seg: Segment | None, item: Any) -> None:
        if seg is None:
            await self._memcat_forward(st, None, item)
            return
        if not st.releasing and st.segments and st.segments[0] is seg and not seg.items:
            if seg.base_s is None:
                seg.base_s = st.emitted_s
            await self._memcat_forward(st, seg, item)
        else:
            seg.items.append(item)

    async def _memcat_forward(self, st: CtxState, seg: Segment | None, item: Any) -> None:
        if isinstance(item, TTSAudioRawFrame):
            item.context_id = st.cid
            st.emitted_s += bytes_to_seconds(len(item.audio), item.sample_rate)
        elif compat.is_word_entry(item):
            w, ts, _, ifs = compat.word_entry_fields(item)
            base = seg.base_s if seg is not None and seg.base_s is not None else 0.0
            item = compat.make_word_entry(w, ts + base, st.cid, ifs)
        await super().append_to_audio_context(st.cid, item)  # type: ignore[misc]

    async def _memcat_release_ctx(self, st: CtxState) -> None:
        if st.releasing:
            return
        st.releasing = True
        try:
            while st.segments:
                seg = st.segments[0]
                if seg.base_s is None:
                    seg.base_s = st.emitted_s
                while seg.items:
                    batch, seg.items = seg.items, []
                    for it in batch:
                        await self._memcat_forward(st, seg, it)
                if seg.done:
                    st.segments.popleft()
                    continue
                break
        finally:
            st.releasing = False
        await self._memcat_maybe_finalize(st)

    async def _memcat_maybe_finalize(self, st: CtxState) -> None:
        if st.finalized or not st.input_done or st.segments or st.open_stream is not None:
            return
        if any(s.state != "done" for s in st.streams):
            return
        st.finalized = True
        await super().append_to_audio_context(st.cid, TTSStoppedFrame(context_id=st.cid))  # type: ignore[misc]
        await super().append_to_audio_context(st.cid, None)  # type: ignore[misc]

    async def _memcat_drop_ctx(self, st: CtxState, pump: bool = True) -> None:
        # All bookkeeping first, with no await: a cancellation in the provider calls
        # below must not leave a stream holding an in-flight slot for the rest of
        # the call.
        sent: list[Stream] = []
        for s in st.streams:
            if s.state != "done":
                if s.state in ("sending", "sent"):
                    sent.append(s)
                s.state = "done"
                s.seg.done = True
                if s in self._memcat_inflight:
                    self._memcat_inflight.remove(s)
                with contextlib.suppress(ValueError):
                    self._memcat_waiting.remove(s)
            self._memcat_streams.pop(s.sid, None)
            self._memcat_forget_stream(s.sid)
        st.streams.clear()
        st.segments.clear()
        st.open_stream = None
        try:
            if sent and self._memcat_adapter.routing == "context":
                await asyncio.shield(self._memcat_abandon(sent))
        finally:
            if pump:
                self._memcat_pump()

    async def _memcat_abandon(self, streams: Sequence[Stream]) -> None:
        """Stop the provider generating streams Memcat no longer wants."""
        if self._memcat_adapter.routing == "context":
            for s in streams:
                with contextlib.suppress(Exception):
                    await self._memcat_adapter.cancel_stream(self, s.sid)
        elif streams:
            await self._memcat_clear_socket(streams[-1].sid)

    async def _memcat_clear_socket(self, context_id: str) -> None:
        """Cursor routing: drop everything the socket still has queued ("Clear")."""
        self._memcat_note_clear()
        with contextlib.suppress(Exception):
            await self._memcat_super_interrupted(context_id)

    def _memcat_forget_stream(self, sid: str) -> None:
        self._memcat_closed[sid] = None
        while len(self._memcat_closed) > _MAX_CLOSED:
            self._memcat_closed.pop(next(iter(self._memcat_closed)))
        compat.drop_tts_span(self, sid)

    # cursor routing: audio sent before a "Cleared" belongs to no stream -------

    def _memcat_note_clear(self) -> None:
        if self._memcat_adapter.routing != "cursor":
            return
        ws = compat.current_websocket(self)  # type: ignore[arg-type]
        if ws is None or ws is not self._memcat_watched_ws:
            return  # its "Cleared" would never be seen: nothing is held back
        if ws is not self._memcat_clear_ws:
            self._memcat_clear_ws, self._memcat_clears = ws, 0
        self._memcat_clears += 1
        self._memcat_clear_deadline = time.monotonic() + _CLEAR_WAIT_S

    def _memcat_watching(self, ws: Any) -> None:
        self._memcat_watched_ws = ws

    def _memcat_on_cleared(self) -> None:
        self._memcat_clears = max(0, self._memcat_clears - 1)

    def _memcat_clearing(self) -> bool:
        if not self._memcat_clears:
            return False
        if (
            compat.current_websocket(self) is not self._memcat_clear_ws  # type: ignore[arg-type]
            or time.monotonic() > self._memcat_clear_deadline
        ):
            self._memcat_clears = 0  # reconnected, or the "Cleared" never came
            return False
        return True

    # ------------------------------------------------- Pipecat hook overrides

    async def append_to_audio_context(self, context_id: Any, frame: Any) -> None:  # type: ignore[override]
        if context_id:
            s = self._memcat_streams.get(context_id)
            if s is not None:
                if frame is not None:
                    await self._memcat_route(s, frame)
                return
            if context_id in self._memcat_closed or context_id == _DEAD_CTX:
                return  # a late reply for a stream Memcat already closed
            st = self._memcat_ctx.get(context_id)
            if st is not None and frame is not None and not st.finalized:
                tail = st.segments[-1] if st.segments else None
                await self._memcat_push(st, tail, frame)
                return
            if frame is not None and compat.is_word_entry(frame):
                h = self._memcat_http.get(context_id)
                if h is not None and h.live_active:
                    w, ts, c, ifs = compat.word_entry_fields(frame)
                    self._memcat_ifs = bool(ifs)
                    if h.capture:
                        # A provider that exposes its clock (ElevenLabs) added it to this
                        # word just now; it can be reset between requests, so it is
                        # read per word. Otherwise the raw time is rebased at capture.
                        clock = (
                            self._memcat_adapter.word_clock(self)  # type: ignore[arg-type]
                            if h.word_clock is not None
                            else None
                        )
                        h.words.append((w, float(ts) - clock if clock is not None else float(ts)))
                    if h.trimmer is not None and not h.trimmer.seen_speech:
                        h.held.append((w, ts, c, ifs))  # shift unknown until speech starts
                        return
                    frame = compat.make_word_entry(w, ts + self._memcat_http_word_shift(h), c, ifs)
                else:
                    lc = self._memcat_lead.get(context_id)
                    if lc is not None and lc.has_live and lc.cached_s:
                        w, ts, c, ifs = compat.word_entry_fields(frame)
                        frame = compat.make_word_entry(w, ts + lc.cached_s, c, ifs)
        await super().append_to_audio_context(context_id, frame)  # type: ignore[misc]

    async def create_audio_context(self, context_id: str) -> None:
        s = self._memcat_streams.get(context_id)
        if s is not None:
            s.registered = True
            return
        await super().create_audio_context(context_id)  # type: ignore[misc]

    def audio_context_available(self, context_id: str) -> bool:
        s = self._memcat_streams.get(context_id)
        if s is not None:
            return s.registered and s.state != "done"
        return super().audio_context_available(context_id)  # type: ignore[misc]

    async def remove_audio_context(self, context_id: str | None) -> None:
        if context_id:
            s = self._memcat_streams.get(context_id)
            if s is not None:
                await self._memcat_stream_done(s)
                return
            if context_id in self._memcat_closed or context_id == _DEAD_CTX:
                logger.debug(f"memcat: late end of stream {context_id}, already closed")
                return
            st = self._memcat_ctx.get(context_id)
            if st is not None:
                # Someone other than Memcat closed the context (e.g. a disconnect).
                if not st.finalized:
                    await self._memcat_drop_ctx(st)
                    st.finalized = True
                    await super().append_to_audio_context(context_id, None)  # type: ignore[misc]
                return
        await super().remove_audio_context(context_id)  # type: ignore[misc]

    def get_active_audio_context_id(self) -> str | None:
        if self._memcat_adapter.routing == "cursor":
            if self._memcat_clearing():
                # Audio (and a "Flushed") sent before the socket confirmed a "Clear"
                # belongs to the interrupted stream, never to the next one.
                return _DEAD_CTX
            if self._memcat_inflight:
                return self._memcat_inflight[0].sid
        return super().get_active_audio_context_id()  # type: ignore[misc]

    def _get_websocket(self) -> Any:
        # WebSocket services read their socket through this; a cursor-routed adapter
        # watches the receive loop for the end of a "Clear".
        ws = super()._get_websocket()  # type: ignore[misc]
        if self._memcat_disabled:
            return ws
        return self._memcat_adapter.wrap_socket(self, ws)

    async def _handle_interruption(self, frame: Any, direction: Any) -> None:
        # Pipecat interrupts every open context in turn. Dropping the first must not
        # start a queued stream of the next: its text would be sent, and billed, only
        # to be cancelled a moment later.
        self._memcat_interrupting = True
        try:
            await super()._handle_interruption(frame, direction)  # type: ignore[misc]
        finally:
            self._memcat_interrupting = False
            self._memcat_pump()

    async def flush_audio(self, context_id: str | None = None) -> None:
        if context_id:
            st = self._memcat_ctx.get(context_id)
            if st is not None:
                await self._memcat_input_done(st)
                return
            lc = self._memcat_lead.get(context_id)
            if lc is not None and not lc.has_live:
                if not lc.finalized and compat.context_queue_exists(self, context_id):  # type: ignore[arg-type]
                    lc.finalized = True
                    await super().append_to_audio_context(  # type: ignore[misc]
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await super().append_to_audio_context(context_id, None)  # type: ignore[misc]
                return
        await super().flush_audio(context_id)  # type: ignore[misc]

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        st = self._memcat_ctx.pop(context_id, None)
        self._memcat_http.pop(context_id, None)
        self._memcat_lead.pop(context_id, None)
        if st is not None:
            await self._memcat_drop_ctx(st)
            with contextlib.suppress(Exception):
                await self.stop_all_metrics()  # type: ignore[attr-defined]
            if self._memcat_adapter.routing == "cursor":
                await self._memcat_clear_socket(context_id)
            return
        self._memcat_note_clear()
        await super().on_audio_context_interrupted(context_id)  # type: ignore[misc]

    async def on_audio_context_completed(self, context_id: str) -> None:
        st = self._memcat_ctx.pop(context_id, None)
        self._memcat_http.pop(context_id, None)
        self._memcat_lead.pop(context_id, None)
        if st is not None:
            unfinished = any(x.state in ("sending", "sent") for x in st.streams)
            await self._memcat_drop_ctx(st, pump=False)
            if unfinished and self._memcat_adapter.routing == "cursor":
                # The context timed out with a stream still generating: clear the socket
                # before the next stream starts, so its audio cannot spill over.
                await self._memcat_clear_socket(context_id)
            self._memcat_pump()
        await super().on_audio_context_completed(context_id)  # type: ignore[misc]

    async def _close_context(self, context_id: str) -> None:
        # ElevenLabs closes the turn's context at turn end; Memcat already ended the
        # streams it opened for that context, and the provider never saw its id.
        if context_id not in self._memcat_ctx:
            await self._memcat_super_close_context(context_id)

    # helpers for adapters (call the provider, skipping Memcat's overrides) ----

    async def _memcat_super_flush(self, sid: str) -> None:
        await super().flush_audio(sid)  # type: ignore[misc]

    async def _memcat_super_close_context(self, sid: str) -> None:
        fn = getattr(super(), "_close_context", None)
        if fn is not None:
            await fn(sid)

    async def _memcat_super_interrupted(self, sid: str) -> None:
        await super().on_audio_context_interrupted(sid)  # type: ignore[misc]

    async def _memcat_super_completed(self, sid: str) -> None:
        await super().on_audio_context_completed(sid)  # type: ignore[misc]
