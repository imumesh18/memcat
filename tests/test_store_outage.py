"""Redis dies in the middle of a call: every turn is still spoken, and no error
reaches the pipeline."""

from __future__ import annotations

import asyncio
import shutil

import pytest
from pipecat.frames.frames import ErrorFrame, TTSSpeakFrame
from pipecat.tests.utils import SleepFrame

import memcat
from memcat import MemcatConfig, RedisStore, cached
from memcat.core.runtime import get_runtime
from tests.fakes import FakeHTTPTTS, FakeWSTTS, words_of
from tests.harness import hear
from tests.test_chaos import _port, _redis

pytestmark = pytest.mark.skipif(shutil.which("redis-server") is None, reason="needs redis-server")

LINES = [
    "Thanks for calling, how can I help you today?",
    "Is there anything else I can help with?",
    "Thanks for waiting, I appreciate your patience.",
]


@pytest.mark.parametrize("cls", [FakeHTTPTTS, FakeWSTTS])
async def test_redis_dies_mid_call(cls) -> None:
    port = _port()
    proc = await _redis(port)
    url = f"redis://127.0.0.1:{port}/0"
    try:
        # Fill the store, then start a fresh process: its hits must come from Redis.
        cfg = MemcatConfig(store=RedisStore(url, op_timeout_ms=50), admit_after_calls=1)
        await hear(cached(cls)(memcat=cfg), [TTSSpeakFrame(x) for x in LINES])
        await get_runtime(cfg).worker.drain_all(10)
        await memcat.aclose()

        cfg2 = MemcatConfig(store=RedisStore(url, op_timeout_ms=50))
        tts = cached(cls)(memcat=cfg2)
        frames = []
        for line in LINES * 3:
            frames += [TTSSpeakFrame(line), SleepFrame(0.15)]

        async def kill_after_first_turns() -> None:
            await asyncio.sleep(0.5)
            proc.kill()

        killer = asyncio.ensure_future(kill_after_first_turns())
        h = await hear(tts, frames)
        await killer
        assert proc.poll() is not None  # Redis really died during the call
        assert h.words == words_of(" ".join(LINES * 3))
        assert not [f for f in h.frames + h.upstream if isinstance(f, ErrorFrame)]
        st = await tts.memcat_stats()
        assert st["hits"] >= 1  # turns before the outage came from Redis
        assert st["enabled"] is True
    finally:
        proc.kill()
        proc.wait()
