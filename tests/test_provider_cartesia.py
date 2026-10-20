"""Memcat against the real Pipecat Cartesia services, talking to a local Cartesia emulator."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

import aiohttp
import pytest
from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame
from pipecat.services.cartesia.tts import CartesiaHttpTTSService, CartesiaTTSService
from pipecat.tests.utils import SleepFrame

import memcat
from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.runtime import get_runtime
from tests.emulators.cartesia import CartesiaEmulator
from tests.fakes import SR, decode_timed, words_of
from tests.harness import hear, turn

A = "Thanks for waiting today, "
B = "your order 48213 has shipped, "
C = "and it should arrive soon — "
D = "the tracking code is ZX81QP, "
E = "is there anything else I can help with?"
MIXED = A + B + C + D + E  # hit, live, hit, live, hit

STREAM = re.compile(r"-s\d+$")


@pytest.fixture
async def emu():
    async with CartesiaEmulator() as e:
        yield e
        await memcat.aclose()  # stop render twins before the server goes away


@pytest.fixture
def cfg_factory() -> Callable[..., MemcatConfig]:
    store = MemoryStore()
    return lambda **kw: MemcatConfig(store=store, **{"admit_after_calls": 1, **kw})


def ws_kwargs(emu: CartesiaEmulator) -> dict[str, Any]:
    return {
        "api_key": "test",
        "settings": CartesiaTTSService.Settings(voice="v"),
        "url": emu.ws_url,
        "sample_rate": SR,
    }


def plain_ws(emu: CartesiaEmulator) -> CartesiaTTSService:
    return CartesiaTTSService(**ws_kwargs(emu))


def cached_ws(emu: CartesiaEmulator, cfg: MemcatConfig) -> Any:
    return cached(CartesiaTTSService)(**ws_kwargs(emu), memcat=cfg)


async def prime(tts: Any, phrases: list[str]) -> None:
    await hear(tts, [TTSSpeakFrame(p) for p in phrases])
    await get_runtime(tts._memcat_cfg).worker.drain_all(20)


def sent_words(emu: CartesiaEmulator) -> list[str]:
    """Words the call itself sent live (render twins use their own contexts)."""
    return " ".join(t for c, t in emu.transcripts() if STREAM.search(c)).split()


def rendered(emu: CartesiaEmulator) -> list[str]:
    return [t for c, t in emu.transcripts() if not STREAM.search(c)]


def assert_streams_closed(emu: CartesiaEmulator) -> list[str]:
    """Every provider context on the wire was ended with continue=false and got done."""
    ctxs = emu.contexts()
    closed, done = emu.closed_contexts(), emu.done_contexts()
    for c in ctxs:
        assert c in closed, f"{c} never got continue=false"
        assert c in done, f"{c} never got done"
    assert not emu.violations, emu.violations
    return ctxs


# --------------------------------------------------------------------------- WS


async def test_ws_empty_cache_matches_plain(emu, cfg_factory) -> None:
    h0 = await hear(plain_ws(emu), turn(MIXED))
    assert h0.words == words_of(MIXED)
    plain_ctxs = assert_streams_closed(emu)
    emu.reset()

    tts = cached_ws(emu, cfg_factory())
    h1 = await hear(tts, turn(MIXED))
    assert h1.words == h0.words
    assert h1.text == h0.text
    assert words_of(" ".join(h1.text)) == words_of(MIXED)
    assert (h1.started, h1.stopped) == (h0.started, h0.stopped) == (1, 1)
    assert sent_words(emu) == MIXED.split()
    await get_runtime(tts._memcat_cfg).worker.drain_all(20)  # let render twins finish
    ctxs = assert_streams_closed(emu)
    assert len(plain_ctxs) == 1
    streams = [c for c in ctxs if STREAM.search(c)]
    assert len(streams) == 1  # one Memcat stream; the turn context never hits the wire
    # Everything else on the wire is a render twin saying a cacheable clause alone.
    assert sorted(t.strip() for t in rendered(emu)) == sorted([A.strip(), C.strip(), E])


async def test_ws_mixed_runs_keep_order(emu, cfg_factory) -> None:
    await prime(cached_ws(emu, cfg_factory()), [A.strip(), C.strip(), E])
    emu.reset()
    tts = cached_ws(emu, cfg_factory())
    t0 = time.perf_counter()
    h = await hear(tts, turn(MIXED))
    elapsed = time.perf_counter() - t0
    assert h.words == words_of(MIXED)
    assert words_of(" ".join(h.text)) == words_of(MIXED)
    assert (h.started, h.stopped) == (1, 1)
    assert sent_words(emu) == (B + D).split()
    ctxs = assert_streams_closed(emu)
    assert len(ctxs) == 2 and all(STREAM.search(c) for c in ctxs)
    assert not emu.cancelled_contexts()
    assert elapsed < 2.0, elapsed  # an open context would stall 3 s
    st = await tts.memcat_stats()
    assert st["hits"] == 3 and st["misses"] == 2


async def test_ws_multi_sentence_hit_after_live(emu, cfg_factory) -> None:
    s1 = "Your balance is 340 points today. "
    s2 = A + "your order 48213 has shipped. "
    s3 = E
    await prime(cached_ws(emu, cfg_factory()), [A.strip(), E])
    emu.reset()
    tts = cached_ws(emu, cfg_factory())
    t0 = time.perf_counter()
    h = await hear(tts, turn(s1, s2, s3))
    elapsed = time.perf_counter() - t0
    assert h.words == words_of(s1 + s2 + s3)
    assert words_of(" ".join(h.text)) == words_of(s1 + s2 + s3)
    assert (h.started, h.stopped) == (1, 1)
    assert sent_words(emu) == (s1 + "your order 48213 has shipped.").split()
    assert_streams_closed(emu)
    assert elapsed < 2.0, elapsed


async def test_ws_word_timestamps_rebased(emu, cfg_factory) -> None:
    await prime(cached_ws(emu, cfg_factory()), [A.strip(), C.strip(), E])
    tts = cached_ws(emu, cfg_factory())
    h = await hear(tts, turn(MIXED))
    onsets = decode_timed(h.pcm, SR)
    assert [w for w, _ in onsets] == words_of(MIXED)
    pts = [p for _, p in h.word_pts]
    assert len(pts) == len(onsets)
    assert pts == sorted(pts)
    for (w, onset), p in zip(onsets, pts, strict=True):
        delta = (p - pts[0]) / 1e9 - (onset - onsets[0][1])
        assert abs(delta) < 0.035, (w, delta)


async def test_ws_interruption_mid_turn(emu, cfg_factory) -> None:
    long = "Thanks for your patience today, " * 3
    await prime(cached_ws(emu, cfg_factory()), ["Thanks for your patience today,"])
    emu.reset()
    tts = cached_ws(emu, cfg_factory())
    after = "Sure, the code is 7721, anything else I can help with?"
    frames = (
        turn(long + "and order 5522 is late, " + long)
        + [SleepFrame(0.01), InterruptionFrame(), SleepFrame(0.05)]
        + turn(after)
    )
    h = await hear(tts, frames)
    want = words_of(after)
    assert h.words[-len(want) :] == want
    assert not tts._memcat_ctx and not tts._memcat_streams and not tts._memcat_inflight
    assert not emu.violations, emu.violations
    # The first turn's live stream was in flight: it was cancelled, not left open.
    first = [c for c, t in emu.transcripts() if "5522" in t]
    assert first, emu.transcripts()
    cancelled, done = emu.cancelled_contexts(), emu.done_contexts()
    for c in first:
        assert c in cancelled or c in done, c
    assert set(first) & cancelled, "the in-flight stream was never cancelled"
    # The next turn's stream ran to completion.
    last = [c for c, t in emu.transcripts() if "7721" in t]
    assert last and all(c in done and c not in cancelled for c in last)


async def test_ws_fully_cached_turn_is_fast(emu, cfg_factory) -> None:
    s = "Thanks for calling, how can I help you today?"
    await prime(cached_ws(emu, cfg_factory()), [s])
    emu.reset()
    tts = cached_ws(emu, cfg_factory())
    t0 = time.perf_counter()
    h = await hear(tts, turn(s) + [TTSSpeakFrame(s)])
    elapsed = time.perf_counter() - t0
    assert h.words == words_of(s) * 2
    assert emu.transcripts() == []
    assert elapsed < 1.5, elapsed
    assert (h.started, h.stopped) == (2, 2)


# ------------------------------------------------------------------------- HTTP


def http_tts(emu: CartesiaEmulator, session: aiohttp.ClientSession, cfg: MemcatConfig | None):
    kw: dict[str, Any] = {
        "api_key": "test",
        "settings": CartesiaTTSService.Settings(voice="v"),
        "base_url": emu.http_url,
        "aiohttp_session": session,
        "sample_rate": SR,
    }
    if cfg is None:
        return CartesiaHttpTTSService(**kw)
    return cached(CartesiaHttpTTSService)(**kw, memcat=cfg)


async def test_http_mixed_runs_keep_order(emu, cfg_factory) -> None:
    async with aiohttp.ClientSession() as session:
        h0 = await hear(http_tts(emu, session, None), turn(MIXED))
        assert h0.words == words_of(MIXED)
        await prime(http_tts(emu, session, cfg_factory()), [A.strip(), C.strip(), E])
        emu.reset()
        tts = http_tts(emu, session, cfg_factory())
        h = await hear(tts, turn(MIXED))
    assert h.words == words_of(MIXED)
    assert words_of(" ".join(h.text)) == words_of(MIXED)
    assert (h.started, h.stopped) == (h0.started, h0.stopped) == (1, 1)
    assert " ".join(r["transcript"] for r in emu.http_requests).split() == (B + D).split()
    st = await tts.memcat_stats()
    assert st["hits"] == 3 and st["misses"] == 2
