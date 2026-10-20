"""Drive one TTS service in a long-lived Pipecat pipeline, one utterance at a time."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSSpeakFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner


class _Sink(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.t_first: float | None = None
        self.pcm = bytearray()
        self.done = asyncio.Event()

    def reset(self) -> None:
        self.t_first = None
        self.pcm = bytearray()
        self.done = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            if self.t_first is None:
                self.t_first = time.perf_counter()
            self.pcm.extend(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            self.done.set()
        await self.push_frame(frame, direction)


class Driver:
    def __init__(self, tts: Any, sample_rate: int):
        self.tts = tts
        self.sink = _Sink()
        self.worker = PipelineWorker(
            Pipeline([tts, self.sink]),
            params=PipelineParams(audio_out_sample_rate=sample_rate, enable_metrics=False),
            cancel_on_idle_timeout=False,
            idle_timeout_secs=None,
            enable_rtvi=False,
            check_dangling_tasks=False,
        )
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        started = asyncio.Event()

        @self.worker.event_handler("on_pipeline_started")
        async def _s(_w: Any, _f: Any) -> None:
            started.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self._task = asyncio.ensure_future(runner.run())
        await asyncio.wait_for(started.wait(), 30)

    async def speak(self, text: str, timeout: float = 60.0) -> tuple[float, bytes]:
        self.sink.reset()
        t0 = time.perf_counter()
        await self.worker.queue_frame(TTSSpeakFrame(text, append_to_context=False))
        await asyncio.wait_for(self.sink.done.wait(), timeout)
        first = self.sink.t_first or time.perf_counter()
        return (first - t0) * 1000.0, bytes(self.sink.pcm)

    async def stop(self) -> None:
        from pipecat.frames.frames import EndFrame

        await self.worker.queue_frame(EndFrame())
        if self._task:
            await asyncio.wait_for(self._task, 30)
