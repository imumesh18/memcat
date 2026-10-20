"""Run a TTS service in a real Pipecat pipeline and record what a listener hears."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test

from tests.fakes import SR, decode


@dataclass
class Heard:
    frames: list[Frame]
    pcm: bytes = b""
    words: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    started: int = 0
    stopped: int = 0
    word_pts: list[tuple[str, int]] = field(default_factory=list)
    upstream: list[Frame] = field(default_factory=list)  # errors travel upstream


class Playout(FrameProcessor):
    """Stands in for the output transport: reports the bot speaking upstream.

    Services built with ``pause_frame_processing=True`` (ElevenLabs, Deepgram) hold
    incoming frames after each turn until a ``BotStoppedSpeakingFrame`` comes back,
    which an output transport sends once audio stops arriving for ``idle_s`` (or at
    once on an interruption). :func:`hear` has no transport, so those services would
    wait forever there.
    """

    def __init__(self, idle_s: float = 0.12):
        super().__init__()
        self._idle_s = idle_s
        self._speaking = False
        self._timer: asyncio.Task[None] | None = None

    async def _stop_after_idle(self) -> None:
        await asyncio.sleep(self._idle_s)
        self._timer = None
        await self._stopped()

    async def _stopped(self) -> None:
        if self._speaking:
            self._speaking = False
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSAudioRawFrame):
                if not self._speaking:
                    self._speaking = True
                    await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
                self._cancel_timer()
                self._timer = asyncio.ensure_future(self._stop_after_idle())
            elif isinstance(frame, InterruptionFrame):
                self._cancel_timer()
                await self._stopped()
            elif isinstance(frame, (EndFrame, CancelFrame)):
                self._cancel_timer()
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        self._cancel_timer()
        await super().cleanup()


def turn(*chunks: str) -> list[Frame]:
    return [LLMFullResponseStartFrame(), *[TextFrame(c) for c in chunks], LLMFullResponseEndFrame()]


async def hear(tts: Any, frames: Sequence[Frame], *, metrics: bool = False) -> Heard:
    down, up = await run_test(
        tts,
        frames_to_send=list(frames),
        pipeline_params=PipelineParams(
            audio_out_sample_rate=SR,
            audio_in_sample_rate=SR,
            enable_metrics=metrics,
            enable_usage_metrics=metrics,
        ),
        start_timeout=5.0,
    )
    pcm = b"".join(f.audio for f in down if isinstance(f, TTSAudioRawFrame))
    h = Heard(list(down), pcm, decode(pcm), upstream=list(up))
    for f in down:
        if isinstance(f, TTSTextFrame):
            h.text.append(f.text)
            if f.pts is not None:
                h.word_pts.append((f.text, f.pts))
        elif isinstance(f, TTSStartedFrame):
            h.started += 1
        elif isinstance(f, TTSStoppedFrame):
            h.stopped += 1
    return h


class Driver:
    """A long-lived pipeline, one utterance at a time (used where run_test is too rigid)."""

    def __init__(self, tts: Any, *, tracing: bool = False):
        import asyncio

        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.processors.frame_processor import FrameProcessor

        outer = self

        class _Sink(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)
                outer.frames.append(frame)
                if isinstance(frame, TTSStoppedFrame):
                    outer._done.set()
                await self.push_frame(frame, direction)

        self.frames: list[Frame] = []
        self._done = asyncio.Event()
        self.worker = PipelineWorker(
            Pipeline([tts, _Sink()]),
            params=PipelineParams(audio_out_sample_rate=SR, audio_in_sample_rate=SR),
            cancel_on_idle_timeout=False,
            idle_timeout_secs=None,
            enable_rtvi=False,
            enable_tracing=tracing,
            check_dangling_tasks=False,
        )
        self._task: Any = None

    async def __aenter__(self) -> Driver:
        import asyncio

        from pipecat.workers.runner import WorkerRunner

        started = asyncio.Event()

        @self.worker.event_handler("on_pipeline_started")
        async def _s(_w: Any, _f: Any) -> None:
            started.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self._task = asyncio.ensure_future(runner.run())
        await asyncio.wait_for(started.wait(), 10)
        return self

    async def say(self, text: str) -> None:
        import asyncio

        from pipecat.frames.frames import TTSSpeakFrame

        self._done = asyncio.Event()
        await self.worker.queue_frame(TTSSpeakFrame(text))
        await asyncio.wait_for(self._done.wait(), 10)

    async def __aexit__(self, *exc: Any) -> None:
        import asyncio

        from pipecat.frames.frames import EndFrame

        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self._task, 10)
