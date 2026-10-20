"""Render twin: synthesize one clause alone, outside any call.

A twin is a second instance of the wrapped provider class (without Memcat), built
from the constructor arguments ``cached()`` recorded, running in its own
one-processor pipeline. It hooks the audio-context path every Pipecat TTS
service goes through, so it sees raw audio and raw word timestamps (relative to
the clause start) for HTTP and WebSocket services alike.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import time
import weakref
from collections.abc import Callable
from typing import Any

import aiohttp
from loguru import logger
from pipecat.frames.frames import EndFrame, ErrorFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.services.tts_service import TTSService
from pipecat.workers.runner import WorkerRunner

from memcat.core.worker import Audio, RenderFn
from memcat.pipecat import compat

__all__ = ["RenderPool", "aclose_render_pools", "make_twin_class"]

RENDER_TIMEOUT_S = 15.0
IDLE_CLOSE_S = 60.0

_twin_classes: dict[type, type] = {}


def make_twin_class(base: type[TTSService]) -> type[TTSService]:
    """A subclass of ``base`` that reports the audio and words it produces."""
    cached_cls = _twin_classes.get(base)
    if cached_cls is not None:
        return cached_cls

    class _Twin(base):  # type: ignore[valid-type,misc]
        _memcat_sink: Callable[[str, Any], None] | None = None
        _memcat_done: Callable[[str], None] | None = None
        _memcat_error: Callable[[], None] | None = None

        async def append_to_audio_context(self, context_id: Any, frame: Any) -> None:  # type: ignore[override]
            sink = self._memcat_sink
            if sink is not None and context_id:
                sink(context_id, frame)  # None: the provider ended the context
            await super().append_to_audio_context(context_id, frame)

        async def push_frame(self, frame: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
            # Errors the service reports itself (push_error) travel upstream, never
            # through the audio context.
            on_error = self._memcat_error
            if on_error is not None and isinstance(frame, ErrorFrame):
                on_error()
            await super().push_frame(frame, *args, **kwargs)

        async def on_audio_context_completed(self, context_id: str) -> None:  # type: ignore[override]
            await super().on_audio_context_completed(context_id)
            done = self._memcat_done
            if done is not None:
                done(context_id)

    _Twin.__name__ = f"MemcatTwin{base.__name__}"
    _Twin.__qualname__ = _Twin.__name__
    _twin_classes[base] = _Twin
    return _Twin


class _TwinRunner:
    def __init__(
        self,
        base: type[TTSService],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        settings: Any,
        sample_rate: int,
    ):
        self._base = base
        self._args = args
        self._kwargs = kwargs
        self._settings = settings
        self._sample_rate = sample_rate
        self._lock = asyncio.Lock()
        self._service: Any = None
        self._worker: Any = None
        self._run_task: asyncio.Task[None] | None = None
        self.last_used = time.monotonic()
        self._pcm = bytearray()
        self._words: list[tuple[str, float]] = []
        self._ctx: str | None = None
        self._done = asyncio.Event()
        self._ended = False  # the provider closed the context (not Pipecat's timeout)
        self._failed = False
        self._owned: list[Any] = []

    def _own_sessions(self, kwargs: dict[str, Any]) -> None:
        # A twin can outlive the call that admitted the clause, so it never borrows
        # that call's HTTP session: each twin opens and closes its own.
        for k, v in list(kwargs.items()):
            if isinstance(v, aiohttp.ClientSession):
                own = aiohttp.ClientSession()
                self._owned.append(own)
                kwargs[k] = own

    async def _ensure(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            return
        cls = make_twin_class(self._base)
        kwargs = dict(self._kwargs)
        self._own_sessions(kwargs)
        svc: Any = cls(*self._args, **kwargs)
        if self._settings is not None:
            compat.replace_settings(svc, copy.deepcopy(self._settings))
        svc._memcat_sink = self._sink
        svc._memcat_done = self._on_done
        svc._memcat_error = self._on_error
        # No transport plays a twin's audio, so nothing would ever lift a pause
        # taken until "bot stopped speaking" (Deepgram, Sarvam, ElevenLabs).
        compat.disable_playback_pause(svc)
        self._service = svc
        worker: Any = PipelineWorker(
            Pipeline([svc]),
            params=PipelineParams(audio_out_sample_rate=self._sample_rate, enable_metrics=False),
            cancel_on_idle_timeout=False,
            idle_timeout_secs=None,
            enable_rtvi=False,
            enable_turn_tracking=False,
            check_dangling_tasks=False,
        )
        started = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def _started(_w: Any, _f: Any) -> None:
            started.set()

        self._worker = worker
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        self._run_task = asyncio.ensure_future(runner.run())
        await asyncio.wait_for(started.wait(), timeout=10.0)

    def _sink(self, context_id: str, item: Any) -> None:
        if self._ctx is None:
            self._ctx = context_id
        if context_id != self._ctx:
            return
        if item is None:
            self._ended = True
        elif isinstance(item, ErrorFrame):
            self._failed = True
        elif isinstance(item, TTSAudioRawFrame):
            if item.sample_rate == self._sample_rate and item.num_channels == 1:
                self._pcm.extend(item.audio)
        elif compat.is_word_entry(item):
            word, ts, _, _ = compat.word_entry_fields(item)
            self._words.append((word, float(ts)))

    def _on_error(self) -> None:
        self._failed = True

    def _on_done(self, context_id: str) -> None:
        if self._ctx is None or context_id == self._ctx:
            self._done.set()

    async def render(self, text: str) -> Audio | None:
        async with self._lock:
            self.last_used = time.monotonic()
            await self._ensure()
            # A stored clip is reused after any sentence, so it is rendered with no
            # context from whatever this twin rendered before.
            compat.clear_request_context(self._service)
            self._pcm = bytearray()
            self._words = []
            self._ctx = None
            self._ended = False
            self._failed = False
            self._done = asyncio.Event()
            await self._worker.queue_frame(TTSSpeakFrame(text, append_to_context=False))
            try:
                await asyncio.wait_for(self._done.wait(), timeout=RENDER_TIMEOUT_S)
            except TimeoutError:
                logger.debug("memcat: render timed out")
                await self.close()
                return None
            finally:
                self.last_used = time.monotonic()
            if self._failed or not self._ended or not self._pcm:
                # An error, or a context that ran out of time instead of being closed
                # by the provider: whatever audio arrived may be a clause cut short.
                logger.debug("memcat: render failed or incomplete; nothing kept")
                return None
            return Audio(bytes(self._pcm), self._sample_rate, tuple(self._words))

    async def close(self) -> None:
        task, self._run_task = self._run_task, None
        if task is not None:
            with contextlib.suppress(Exception):
                await self._worker.queue_frame(EndFrame())
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    task.cancel()  # close() itself was cancelled: stop the twin, then propagate
                    raise
                # Otherwise the twin's own task was cancelled: it is already done.
            except Exception:
                task.cancel()
                # Collects the task's own cancellation; a cancellation of close() still raises.
                await asyncio.gather(asyncio.wait_for(task, 2.0), return_exceptions=True)
        owned, self._owned = self._owned, []
        for sess in owned:
            with contextlib.suppress(Exception):
                await sess.close()


_pools: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, weakref.WeakSet[RenderPool]] = (
    weakref.WeakKeyDictionary()
)

# A clause reaches a twin already transformed and filtered by the live service; the
# twin must not apply them again (or share a stateful filter with that service).
_NOT_FOR_TWINS = ("memcat", "text_transforms", "text_filters")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, aiohttp.ClientSession) and isinstance(b, aiohttp.ClientSession):
        return True  # a twin opens its own session, whichever the service was given
    try:
        return type(a) is type(b) and bool(a == b)
    except Exception:
        return False


class RenderPool:
    """Twins for one provider configuration, shared by every call in the process
    built with the same constructor arguments.

    One twin per voice (settings + sample rate); each closes after 60 s idle. A pool
    lives while a service uses it or its twins run.
    """

    @classmethod
    def shared(
        cls, base: type[TTSService], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> RenderPool:
        pools = _pools.setdefault(asyncio.get_running_loop(), weakref.WeakSet())
        want = cls(base, args, kwargs)
        for pool in pools:
            if pool._base is base and pool._same_args(want):
                return pool
        pools.add(want)
        return want

    def __init__(self, base: type[TTSService], args: tuple[Any, ...], kwargs: dict[str, Any]):
        self._base = base
        self._args = tuple(args)
        self._kwargs = {k: v for k, v in kwargs.items() if k not in _NOT_FOR_TWINS}
        self._twins: dict[str, _TwinRunner] = {}
        self._reaper: asyncio.Task[None] | None = None

    def _same_args(self, other: RenderPool) -> bool:
        a, b = self._kwargs, other._kwargs
        return (
            len(self._args) == len(other._args)
            and all(_same(x, y) for x, y in zip(self._args, other._args, strict=True))
            and a.keys() == b.keys()
            and all(_same(a[k], b[k]) for k in a)
        )

    def render_fn(
        self,
        voice_key: str,
        text: str,
        settings: Any,
        sample_rate: int,
        on_chars: Callable[[int], None],
    ) -> RenderFn:
        """A render of ``text`` in this voice. ``settings`` must be a snapshot the
        caller will not mutate."""

        async def run() -> Audio | None:
            key = f"{voice_key}|{sample_rate}"
            twin = self._twins.get(key)
            if twin is None:
                twin = _TwinRunner(self._base, self._args, self._kwargs, settings, sample_rate)
                self._twins[key] = twin
            self._ensure_reaper()
            on_chars(len(text))
            return await twin.render(text.strip())

        return run

    def _ensure_reaper(self) -> None:
        # Registered again in case aclose_render_pools() ran since: it must reach
        # every running twin.
        _pools.setdefault(asyncio.get_running_loop(), weakref.WeakSet()).add(self)
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.ensure_future(self._reap())

    async def _reap(self) -> None:
        while self._twins:
            await asyncio.sleep(10.0)
            now = time.monotonic()
            for key, twin in list(self._twins.items()):
                if now - twin.last_used > IDLE_CLOSE_S and not twin._lock.locked():
                    self._twins.pop(key, None)
                    await twin.close()

    async def close(self) -> None:
        reaper, self._reaper = self._reaper, None
        if reaper is not None:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
        twins, self._twins = list(self._twins.values()), {}
        for t in twins:
            await t.close()


async def aclose_render_pools() -> None:
    """Close every twin on the running loop."""
    loop = asyncio.get_running_loop()
    for pool in list(_pools.pop(loop, ())):
        await pool.close()
