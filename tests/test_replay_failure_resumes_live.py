"""A Memcat failure mid-sentence: the listener hears every clause exactly once, and the
failure is logged once without clause text. Also: what a live stream may capture,
and that background stream tasks end with the service."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from loguru import logger
from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame

from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.lookup import Clip
from memcat.pipecat import mixin
from memcat.pipecat.contexts import CtxState
from tests.fakes import FakeWSTTS, synth, words_of
from tests.harness import Driver, hear, turn
from tests.util import drain, make

P1 = "Thanks for waiting today, "
P2 = "we really appreciate your patience, "
P3 = "your order 48213 has shipped."
TEXT = P1 + P2 + P3  # hit, hit, live


class LeadingWS(FakeWSTTS):
    __memcat_adapter__ = None  # no certified adapter: leading mode


def build(variant: str, cfg: MemcatConfig) -> Any:
    if variant == "leading":
        tts = cached(LeadingWS)(memcat=cfg)
        assert tts._memcat_mode == "leading"
        return tts
    return make(variant, cfg)


def live_texts(tts: Any) -> list[str]:
    return [r if isinstance(r, str) else r[1] for r in tts.requests]


async def primed(variant: str) -> MemcatConfig:
    store = MemoryStore()
    cfg = MemcatConfig(store=store, admit_after_calls=1)
    warm = build(variant, cfg)
    await hear(warm, [TTSSpeakFrame(P1.strip()), TTSSpeakFrame(P2.strip())])
    await drain(cfg)
    return MemcatConfig(store=store)


def fail_after(monkeypatch: pytest.MonkeyPatch, n_frames: int) -> None:
    """Clips decode ``n_frames`` frames in total, then every decode raises."""
    real = Clip.frames
    left = [n_frames]

    def flaky(self: Clip, frame_ms: int = 40):
        for f in real(self, frame_ms):
            if left[0] <= 0:
                raise RuntimeError("bad clip")
            left[0] -= 1
            yield f

    monkeypatch.setattr(Clip, "frames", flaky)


# 0: fails on the first frame; 3: part way into P1; 14: part way into P2 (P1 is ~10 frames)
@pytest.mark.parametrize("n_frames", [0, 3, 14])
@pytest.mark.parametrize("variant", ["http", "ws-ctx", "ws-el", "ws-cursor", "leading"])
async def test_replay_failure_never_repeats_or_drops_a_clause(
    variant: str, n_frames: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = await primed(variant)
    fail_after(monkeypatch, n_frames)
    tts = build(variant, cfg)
    h = await hear(tts, turn(TEXT))
    assert h.words == words_of(TEXT)
    assert (await tts.memcat_stats())["enabled"] is False


@pytest.mark.parametrize("variant", ["http", "leading"])
async def test_clauses_replayed_before_a_failure_are_not_sent_live(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = await primed(variant)
    fail_after(monkeypatch, 14)  # P1 replays whole, P2 fails
    tts = build(variant, cfg)
    h = await hear(tts, turn(TEXT))
    assert h.words == words_of(TEXT)
    sent = " ".join(live_texts(tts))
    assert P1.strip() not in sent
    assert P2.strip() in sent and P3.strip() in sent


# --------------------------------------------------------------------- logging


@pytest.fixture
def records():
    out: list[Any] = []
    sink = logger.add(lambda m: out.append(m.record), level="DEBUG")
    yield out
    logger.remove(sink)


@pytest.mark.parametrize("where", ["plan", "replay"])
async def test_a_disabling_error_logs_one_warning_without_its_message(
    where: str, records: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "the caller's card ends 4242"
    cfg = await primed("http")
    if where == "plan":

        async def broken_plan(self: Any, text: str) -> Any:
            raise RuntimeError(secret)

        monkeypatch.setattr(mixin.MemcatTTSMixin, "_memcat_plan", broken_plan)
    else:

        def boom(self: Clip, frame_ms: int = 40):
            raise RuntimeError(secret)
            yield b""

        monkeypatch.setattr(Clip, "frames", boom)
    records.clear()
    tts = build("http", cfg)
    h = await hear(tts, turn(TEXT))
    assert h.words == words_of(TEXT)
    memcat_warnings = [
        r for r in records if r["level"].no >= 30 and r["message"].startswith("memcat:")
    ]
    assert len(memcat_warnings) == 1, [r["message"] for r in memcat_warnings]
    assert "RuntimeError" in memcat_warnings[0]["message"]
    assert all(secret not in r["message"] for r in records if r["level"].no >= 20)
    assert any(secret in r["message"] for r in records if r["level"].name == "DEBUG")


# --------------------------------------------------------------------- capture


async def _cached_after_one_hearing(cls: type, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Speak a clause alone, then again: was the second time served from cache?"""
    phrase = "Thanks for waiting today,"
    cfg = MemcatConfig(store=MemoryStore(), admit_after_calls=1, render=False)
    tts = cached(cls)(memcat=cfg)
    await hear(tts, [TTSSpeakFrame(phrase)])
    await drain(cfg)
    again = cached(cls)(memcat=cfg)
    await hear(again, [TTSSpeakFrame(phrase)])
    return not again.requests


async def test_a_live_stream_is_cached_when_short_and_native() -> None:
    assert await _cached_after_one_hearing(FakeWSTTS, pytest.MonkeyPatch())


async def test_a_live_stream_past_the_capture_cap_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mixin, "_MAX_CAPTURE_S", 0.2)  # the clause runs ~0.4 s
    assert not await _cached_after_one_hearing(FakeWSTTS, monkeypatch)


async def test_a_live_stream_in_another_format_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stereo(FakeWSTTS):
        async def _emit(self, ctx: str, text: str) -> None:
            pcm, _ = synth(text, self.sample_rate)
            step = self.sample_rate // 10 * 4
            for i in range(0, len(pcm), step):
                await self.append_to_audio_context(
                    ctx, TTSAudioRawFrame(pcm[i : i + step], self.sample_rate, 2, context_id=ctx)
                )

    assert not await _cached_after_one_hearing(Stereo, monkeypatch)


# ----------------------------------------------------------------------- tasks


async def test_stream_tasks_end_with_the_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued stream drained in the background is cancelled when the service stops."""
    started = asyncio.Event()

    async def never_answers(self: Any, s: Any) -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(mixin.MemcatTTSMixin, "_memcat_drain", never_answers)
    tts = make("ws-cursor", MemcatConfig(store=MemoryStore()))
    async with Driver(tts):
        st = CtxState("ctx")
        tts._memcat_new_stream(st)
        tts._memcat_pump()
        await asyncio.wait_for(started.wait(), 5)
        drains = [t for t in asyncio.all_tasks() if "memcat-stream" in t.get_name()]
        assert drains
    await asyncio.sleep(0)
    assert all(t.done() for t in drains)
