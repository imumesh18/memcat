"""Memcat against the real Pipecat Deepgram services, talking to a local Deepgram emulator.

``DeepgramTTSService`` is built with ``pause_frame_processing=True`` (hard-coded),
so after each turn it holds incoming frames until a ``BotStoppedSpeakingFrame``
comes back upstream. ``tests.harness.hear`` has no output transport, so the plain
service deadlocks there. :class:`tests.harness.Playout` stands in for ``BaseOutputTransport``:
it pushes ``BotStartedSpeakingFrame`` upstream on the first audio frame and
``BotStoppedSpeakingFrame`` after ``BOT_VAD_STOP_SECS`` (0.35 s) without audio,
or at once on an interruption, as the real transport does.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import aiohttp
import pytest
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.deepgram.tts import DeepgramHttpTTSService, DeepgramTTSService
from pipecat.tests.utils import SleepFrame
from pipecat.transports.base_output import BOT_VAD_STOP_SECS

from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.runtime import get_runtime
from memcat.pipecat import render as render_mod
from tests.emulators.deepgram import DeepgramEmulator
from tests.fakes import SR, words_of
from tests.harness import Heard, Playout, hear, turn
from tests.test_pipeline import MIXED, A, B, C, D, E
from tests.util import drain

# --------------------------------------------------------------------------- harness


async def listen(tts: Any, frames: Sequence[Frame]) -> Heard:
    return await asyncio.wait_for(hear(Pipeline([tts, Playout(BOT_VAD_STOP_SECS)]), frames), 60)


def ws(
    emu: DeepgramEmulator, cfg: MemcatConfig | None = None, *, seasoned: bool = True, **kw: Any
) -> Any:
    """The real service. ``seasoned`` simulates a call already two quiet minutes in, so
    the flush budget has seen the turn rate and may spend extra flushes; a fresh call
    spends none for its first 30 s."""
    args = {"api_key": "test", "base_url": emu.ws_url, "sample_rate": SR, **kw}
    if cfg is None:
        return DeepgramTTSService(**args)
    svc = cached(DeepgramTTSService)(memcat=cfg, **args)
    if seasoned and svc._memcat_budget is not None:
        svc._memcat_budget._t0 = time.monotonic() - 120
    return svc


def assert_closed(tts: Any) -> None:
    assert not tts._memcat_ctx and not tts._memcat_streams and not tts._memcat_inflight
    assert not tts._memcat_waiting
    assert not tts.get_audio_contexts()


async def prime(emu: DeepgramEmulator, cfg: MemcatConfig, phrases: list[str]) -> None:
    tts = ws(emu, cfg)
    await listen(tts, [TTSSpeakFrame(p) for p in phrases])
    await drain(cfg)


def split_words(texts: list[str]) -> list[str]:
    return " ".join(texts).split()


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(params=["on_speak", "on_flush"])
async def emu(request):
    e = await DeepgramEmulator(generate_on_flush=request.param == "on_flush").start()
    yield e
    await e.stop()
    assert not e.errors, e.errors


@pytest.fixture
def cfg_factory():
    store = MemoryStore()
    # render=False: clauses are cached only from captured live audio. The render twin
    # (a second DeepgramTTSService) deadlocks; see test_h_ws_render_fills_cache.
    return lambda **kw: MemcatConfig(store=store, **{"admit_after_calls": 1, "render": False, **kw})


def no_warnings(e: DeepgramEmulator) -> None:
    assert e.sent("Warning") == [] and e.rejected_flushes == []


# ------------------------------------------------------------------------------ tests


async def test_plain_service_speaks_through_emulator(emu) -> None:
    tts = ws(emu)
    h = await listen(
        tts, turn("Hello there, how are you? ", "I am fine.") + [TTSSpeakFrame("Bye now.")]
    )
    assert h.words == words_of("Hello there, how are you? I am fine. Bye now.")
    assert emu.conns[0].params == {
        "model": "aura-2-helena-en",
        "encoding": "linear16",
        "sample_rate": "16000",
    }
    assert emu.conns[0].headers["Authorization"] == "Token test"
    assert len(emu.messages("Flush")) == 2
    assert emu.messages()[-1] == {"type": "Close"}


async def test_a_empty_cache_matches_plain(emu, cfg_factory) -> None:
    frames = (
        turn(MIXED)
        + turn("Your order 1234 shipped, ", "it arrives Friday.")
        + [TTSSpeakFrame("Anything else?")]
    )
    h0 = await listen(ws(emu), frames)
    plain_speak = emu.speak_texts()
    n0 = len(emu.messages())
    tts = ws(emu, cfg_factory())
    h1 = await listen(tts, frames)
    assert h1.words == h0.words
    assert h1.text == h0.text
    assert (h1.started, h1.stopped) == (h0.started, h0.stopped) == (3, 3)
    assert split_words(emu.speak_texts()[len(plain_speak) :]) == split_words(plain_speak)
    assert [m["type"] for m in emu.messages()[n0:]] == [m["type"] for m in emu.messages()[:n0]]
    assert_closed(tts)
    no_warnings(emu)


async def test_b_mixed_sentence_order(emu, cfg_factory) -> None:
    """Both emulator modes (generate on Speak / on Flush) via the ``emu`` params (test g)."""
    await prime(emu, cfg_factory(), [A.strip(), C.strip(), E])
    n_speak = len(emu.speak_texts())
    tts = ws(emu, cfg_factory())
    h = await listen(tts, turn(MIXED))
    assert h.words == words_of(MIXED)
    assert words_of(" ".join(h.text)) == words_of(MIXED)
    assert (h.started, h.stopped) == (1, 1)
    sent = emu.speak_texts()[n_speak:]
    assert split_words(sent) == (B + D).split()
    # one Flush per live stream, each answered by its own Flushed
    conn = emu.conns[-1]
    assert [m["type"] for _, m in conn.received] == ["Speak", "Flush", "Speak", "Flush", "Close"]
    assert len([1 for _, k, _ in conn.sent if k == "Flushed"]) == 2
    st = await tts.memcat_stats()
    assert st["hits"] == 3 and st["misses"] == 2
    assert_closed(tts)
    no_warnings(emu)


async def test_c_multi_sentence_turn_hit_after_live(emu, cfg_factory) -> None:
    s1 = "Your order 4471 is confirmed. "
    s2 = "Thanks for waiting today, your parcel 8812 ships soon. "
    s3 = "Is there anything else I can help with?"
    await prime(emu, cfg_factory(), [A.strip(), s3])
    n_speak = len(emu.speak_texts())
    tts = ws(emu, cfg_factory())
    h = await listen(tts, turn(s1, s2, s3))
    assert h.words == words_of(s1 + s2 + s3)
    assert words_of(" ".join(h.text)) == words_of(s1 + s2 + s3)
    assert split_words(emu.speak_texts()[n_speak:]) == (s1 + "your parcel 8812 ships soon.").split()
    assert (h.started, h.stopped) == (1, 1)
    assert_closed(tts)
    no_warnings(emu)


async def test_d_flush_budget_holds(emu, cfg_factory) -> None:
    await prime(emu, cfg_factory(), [A.strip(), C.strip(), E])
    tts = ws(emu, cfg_factory(), seasoned=False)
    n = 12
    frames: list[Frame] = []
    for _ in range(n):
        frames += turn(MIXED)
    h = await listen(tts, frames)
    conn = emu.conns[-1]
    flushes = [t for t, m in conn.received if m["type"] == "Flush"]
    assert len(flushes) <= 20
    no_warnings(emu)
    assert h.words == words_of(MIXED) * n
    assert (h.started, h.stopped) == (n, n)
    sent = [m["text"] for _, m in conn.received if m["type"] == "Speak"]
    assert any(C.strip() in t for t in sent)  # budget ran low: hits went live
    assert not any(A.strip() in t for t in sent)  # a leading hit never needs a flush
    assert_closed(tts)


async def test_d2_flush_budget_never_worse_than_plain(emu, cfg_factory) -> None:
    await prime(emu, cfg_factory(), [A.strip(), C.strip(), E])
    n = 20  # plain Pipecat: exactly 20 Flush messages, within the limit
    frames: list[Frame] = []
    for _ in range(n):
        frames += turn(MIXED)
    h0 = await listen(ws(emu), frames)  # baseline: plain Pipecat on its own socket
    assert h0.words == words_of(MIXED) * n
    no_warnings(emu)
    tts = ws(emu, cfg_factory(), seasoned=False)  # a fresh call at 20 turns a minute
    h = await listen(tts, frames)
    no_warnings(emu)
    assert h.words == words_of(MIXED) * n


async def test_e_interruption_mid_turn(emu, cfg_factory) -> None:
    long = "Thanks for your patience today, " * 3
    await prime(emu, cfg_factory(), ["Thanks for your patience today,"])
    tts = ws(emu, cfg_factory())
    after = "Sure, the code is 7721, anything else I can help with?"
    frames = (
        turn(long + "and order 5522 is late, " + long)
        + [SleepFrame(0.01), InterruptionFrame(), SleepFrame(0.05)]
        + turn(after)
    )
    h = await listen(tts, frames)
    want = words_of(after)
    assert h.words[-len(want) :] == want
    assert "clear" in [m["type"].lower() for m in emu.messages()]
    assert_closed(tts)
    no_warnings(emu)


async def test_e2_interruption_while_live_stream_in_flight(emu, cfg_factory) -> None:
    """Interrupt while the live clause is being synthesized, then speak a mixed turn."""
    await prime(emu, cfg_factory(), [A.strip(), C.strip(), E])
    tts = ws(emu, cfg_factory())
    frames = turn(MIXED) + [SleepFrame(0.03), InterruptionFrame(), SleepFrame(0.1)] + turn(MIXED)
    h = await listen(tts, frames)
    want = words_of(MIXED)
    assert h.words[-len(want) :] == want
    assert_closed(tts)


async def test_f_http_mixed_order(cfg_factory) -> None:
    e = await DeepgramEmulator().start()
    try:
        async with aiohttp.ClientSession() as session:

            def make(cfg: MemcatConfig | None) -> Any:
                args = {
                    "api_key": "test",
                    "aiohttp_session": session,
                    "base_url": e.http_url,
                    "sample_rate": SR,
                }
                if cfg is None:
                    return DeepgramHttpTTSService(**args)
                return cached(DeepgramHttpTTSService)(memcat=cfg, **args)

            h0 = await listen(make(None), turn(MIXED))
            cfg = cfg_factory()
            await listen(make(cfg), [TTSSpeakFrame(p) for p in (A.strip(), C.strip(), E)])
            await drain(cfg)
            n = len(e.http_requests)
            tts = make(cfg_factory())
            h = await listen(tts, turn(MIXED))
            assert h.words == words_of(MIXED) == h0.words
            assert words_of(" ".join(h.text)) == words_of(MIXED)
            assert (h.started, h.stopped) == (1, 1) == (h0.started, h0.stopped)
            reqs = e.http_requests[n:]
            assert [r["body"]["text"].strip() for r in reqs] == [B.strip(), D.strip()]
            assert reqs[0]["params"] == {
                "model": "aura-2-helena-en",
                "encoding": "linear16",
                "sample_rate": "16000",
                "container": "none",
            }
            st = await tts.memcat_stats()
            assert st["hits"] == 3 and st["misses"] == 2
    finally:
        await e.stop()
    assert not e.errors


async def _unstick_twins() -> None:
    """Test cleanup only: lift the pause in any render twin so it can shut down."""
    loop = asyncio.get_running_loop()
    for pool in list(render_mod._pools.get(loop, ())):
        for twin in pool._twins.values():
            if twin._service is not None:
                twin._service._pause_frame_processing = False
                await twin._service.resume_processing_frames()
    await asyncio.sleep(0.5)  # let the stuck close() finish
    await asyncio.wait_for(render_mod.aclose_render_pools(), 10)


async def test_h_ws_render_fills_cache(emu, cfg_factory, monkeypatch) -> None:
    """Multi-clause live speech is cached by the render twin (a second DeepgramTTSService)."""
    monkeypatch.setattr(render_mod, "RENDER_TIMEOUT_S", 3.0)
    s = "Thanks for waiting today, your order 3321 shipped, and it should arrive soon."
    cfg = cfg_factory(render=True)
    tts = ws(emu, cfg)
    try:
        await listen(tts, turn(s))
        await get_runtime(cfg).worker.drain_all(8)
        m = get_runtime(cfg).metrics.snapshot()
        assert m["entries_stored"] == 2, m
        n_speak = len(emu.speak_texts())
        tts2 = ws(emu, cfg_factory())
        h = await listen(tts2, turn(s))
        assert h.words == words_of(s)
        assert split_words(emu.speak_texts()[n_speak:]) == ["your", "order", "3321", "shipped,"]
    finally:
        await _unstick_twins()
