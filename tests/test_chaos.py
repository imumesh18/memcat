"""Kill the store mid-session: the bot keeps talking, then recovers."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess

import pytest
from pipecat.frames.frames import TTSSpeakFrame

import memcat
from memcat import MemcatConfig, RedisStore, cached
from memcat.core.runtime import get_runtime
from tests.fakes import FakeHTTPTTS, FakeWSTTS, words_of
from tests.harness import hear

pytestmark = pytest.mark.skipif(shutil.which("redis-server") is None, reason="needs redis-server")


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _redis(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return proc
        except OSError:
            await asyncio.sleep(0.02)
    raise RuntimeError("redis did not start")


@pytest.mark.parametrize("cls", [FakeHTTPTTS, FakeWSTTS])
async def test_store_dies_and_comes_back(cls) -> None:
    port = _port()
    proc = await _redis(port)
    url = f"redis://127.0.0.1:{port}/0"
    s = "Thanks for calling, how can I help you today?"
    try:
        cfg = MemcatConfig(store=RedisStore(url, op_timeout_ms=50), admit_after_calls=1)
        tts = cached(cls)(memcat=cfg)
        await hear(tts, [TTSSpeakFrame(s)])
        await get_runtime(cfg).worker.drain_all(10)
        assert len(get_runtime(cfg).index) == 2
        await memcat.aclose()  # a new process: L1 is empty, the store holds the audio

        proc.kill()
        proc.wait()
        cfg2 = MemcatConfig(store=RedisStore(url, op_timeout_ms=50))
        tts2 = cached(cls)(memcat=cfg2)
        rt = get_runtime(cfg2)
        rt.breaker._cooldown = 0.3
        h = await hear(tts2, [TTSSpeakFrame(s)] * 7)
        assert h.words == words_of(s) * 7  # every turn spoken, live
        assert rt.breaker.is_open or rt.metrics.counters["store_errors"] >= 1

        proc = await _redis(port)  # empty store after the restart
        await asyncio.sleep(0.4)
        tts3 = cached(cls)(
            memcat=MemcatConfig(store=RedisStore(url, op_timeout_ms=50), admit_after_calls=1)
        )
        h3 = await hear(tts3, [TTSSpeakFrame(s)] * 3)
        assert h3.words == words_of(s) * 3
        await get_runtime(tts3._memcat_cfg).worker.drain_all(10)
        assert not get_runtime(tts3._memcat_cfg).breaker.is_open
    finally:
        proc.kill()
        proc.wait()
