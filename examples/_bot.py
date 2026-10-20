"""The parts both example bots share: an LLM support agent with a cached voice."""

from __future__ import annotations

import os

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.workers.runner import WorkerRunner

from memcat import MemcatConfig, MemoryStore, RedisStore, cached

SYSTEM = (
    "You are Ava, the phone agent for ABC Shop. Keep answers short and friendly. "
    "Greet callers with 'Thank you for calling ABC Shop, how can I help you today?' "
    "and end with 'Is there anything else I can help with?'"
)

# Wrap the provider once at import time; every call builds its own instance.
CachedCartesia = cached(CartesiaTTSService)


def memcat_config() -> MemcatConfig:
    url = os.getenv("REDIS_URL")
    return MemcatConfig(
        # Redis lets every bot process share one cache; MemoryStore is per process.
        store=RedisStore(url) if url else MemoryStore(),
        namespace=os.getenv("MEMCAT_NAMESPACE", "abc-shop"),
        warm=[
            "Thank you for calling ABC Shop, how can I help you today?",
            "Is there anything else I can help with?",
            "One moment while I look that up.",
        ],
    )


async def run_bot(transport: BaseTransport, sample_rate: int, handle_sigint: bool) -> None:
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    tts = CachedCartesia(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
        memcat=memcat_config(),
    )
    context = LLMContext([{"role": "system", "content": SYSTEM}])
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=sample_rate,
            audio_out_sample_rate=sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def _connected(_transport, _client) -> None:
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _disconnected(_transport, _client) -> None:
        print("memcat:", await tts.memcat_stats())
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)
    await runner.run()
