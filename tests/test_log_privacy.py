"""Clause text never reaches logs at INFO or above, even when things fail."""

from __future__ import annotations

import pytest
from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame

from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.lookup import Clip
from memcat.core.runtime import get_runtime
from tests.fakes import FakeHTTPTTS, FakeWSTTS
from tests.harness import hear

SECRET_CLAUSE = "Thanks for waiting today Margaret Hamilton"
SENTENCE = f"{SECRET_CLAUSE}, is there anything else I can help with?"


@pytest.fixture
def records():
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level="INFO")
    yield seen
    logger.remove(sink)


@pytest.mark.parametrize("cls", [FakeHTTPTTS, FakeWSTTS])
async def test_normal_calls_log_no_text(cls, records) -> None:
    store = MemoryStore()
    for _ in range(2):
        cfg = MemcatConfig(store=store, admit_after_calls=1)
        await hear(cached(cls)(memcat=cfg), [TTSSpeakFrame(SENTENCE)])
        await get_runtime(cfg).worker.drain_all(10)
    assert not [r for r in records if "Margaret" in r or "anything else" in r]


@pytest.mark.parametrize("cls", [FakeHTTPTTS, FakeWSTTS])
async def test_errors_log_no_text(cls, records, monkeypatch) -> None:
    store = MemoryStore()
    cfg = MemcatConfig(store=store, admit_after_calls=1)
    await hear(cached(cls)(memcat=cfg), [TTSSpeakFrame(SENTENCE)])
    await get_runtime(cfg).worker.drain_all(10)

    def boom(self, frame_ms=40):
        raise RuntimeError(f"cannot replay {SENTENCE}")
        yield b""

    monkeypatch.setattr(Clip, "frames", boom)
    tts = cached(cls)(memcat=MemcatConfig(store=store))
    await hear(tts, [TTSSpeakFrame(SENTENCE)])
    assert (await tts.memcat_stats())["enabled"] is False  # the error really happened
    assert any("disabled" in r for r in records)  # and its warning was captured
    assert not [r for r in records if "Margaret" in r or "anything else" in r], records
