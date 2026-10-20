"""Hear Memcat work with no API keys: Kokoro (a local voice) speaks the same lines
twice; the second time they come from the cache in about a millisecond.

    uv add "pipecat-ai[kokoro]" memcat
    uv run examples/kokoro_local.py
"""

from __future__ import annotations

import asyncio
import time
import wave

from pipecat.frames.frames import EndFrame, TTSAudioRawFrame, TTSSpeakFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.workers.runner import WorkerRunner

import memcat
from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.runtime import get_runtime

SR = 24000
LINES = [
    "Thank you for calling, how can I help you today?",
    "Your order 48213 has shipped, and it should arrive soon.",
    "Is there anything else I can help with?",
]


class Recorder(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.pcm = bytearray()
        self.first: float | None = None
        self.done = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.first = self.first or time.perf_counter()
            self.pcm.extend(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            self.done.set()
        await self.push_frame(frame, direction)


async def speak_all(store: MemoryStore, label: str) -> bytes:
    tts = cached(KokoroTTSService)(
        settings=KokoroTTSService.Settings(voice="af_heart"),
        memcat=MemcatConfig(store=store, admit_after_calls=1),  # cache after one call (demo)
    )
    rec = Recorder()
    worker = PipelineWorker(Pipeline([tts, rec]), params=PipelineParams(audio_out_sample_rate=SR))
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.ensure_future(runner.run())
    await asyncio.sleep(0.5)
    out = bytearray()
    for line in LINES:
        rec.pcm, rec.first, rec.done = bytearray(), None, asyncio.Event()
        t0 = time.perf_counter()
        await worker.queue_frame(TTSSpeakFrame(line))
        await rec.done.wait()
        print(f"{label}: {(rec.first or t0) - t0:7.3f}s to first audio  {line}")
        out += rec.pcm
    print(
        f"{label}:",
        {
            k: v
            for k, v in (await tts.memcat_stats()).items()
            if k in ("hits", "misses", "hit_rate_chars")
        },
    )
    await worker.queue_frame(EndFrame())
    await task
    return bytes(out)


async def main() -> None:
    store = MemoryStore()
    await speak_all(store, "call 1")
    await get_runtime(MemcatConfig(store=store)).worker.drain_all(60)  # renders finish
    audio = await speak_all(store, "call 2")
    with wave.open("memcat_demo.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio)
    print("wrote memcat_demo.wav (call 2: cached and live clauses joined)")
    await memcat.aclose()


if __name__ == "__main__":
    asyncio.run(main())
