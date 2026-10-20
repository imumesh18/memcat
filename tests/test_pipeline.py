"""End-to-end behaviour in real Pipecat pipelines, for every kind of provider."""

from __future__ import annotations

import asyncio
import time

import pytest
from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame, TTSUpdateSettingsFrame
from pipecat.services.settings import TTSSettings
from pipecat.tests.utils import SleepFrame

from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.runtime import get_runtime
from tests.fakes import SR, FakeHTTPTTS, FakeWSTTS, decode_timed, words_of
from tests.harness import hear, turn
from tests.util import TIMESTAMPED, VARIANTS, WS, drain, make, prime

A = "Thanks for waiting today, "
B = "your order 48213 has shipped, "
C = "and it should arrive soon — "
D = "the tracking code is ZX81QP, "
E = "is there anything else I can help with?"
MIXED = A + B + C + D + E  # hit, live, hit, live, hit


@pytest.fixture
def cfg_factory():
    store = MemoryStore()
    return lambda **kw: MemcatConfig(store=store, **{"admit_after_calls": 1, **kw})


def live_texts(tts) -> list[str]:
    return [r if isinstance(r, str) else r[1] for r in tts.requests]


@pytest.mark.parametrize("variant", list(VARIANTS))
async def test_mixed_runs_keep_order(variant, cfg_factory) -> None:
    await prime(variant, cfg_factory, [A.strip(), C.strip(), E])
    tts = make(variant, cfg_factory())
    h = await hear(tts, turn(MIXED))
    assert h.words == words_of(MIXED)
    assert "".join(live_texts(tts)).split() == (B + D).split()
    assert words_of(" ".join(h.text)) == words_of(MIXED)
    assert (h.started, h.stopped) == (1, 1)
    st = await tts.memcat_stats()
    assert st["hits"] == 3 and st["misses"] == 2


@pytest.mark.parametrize("variant", list(VARIANTS))
async def test_multi_sentence_turn(variant, cfg_factory) -> None:
    s1 = "Hello and welcome to Acme support. "
    s2 = "Your balance is 340 points, and you can redeem them any time. "
    s3 = "Is there anything else I can help with?"
    await prime(variant, cfg_factory, [s1.strip(), "and you can redeem them any time.", s3])
    tts = make(variant, cfg_factory())
    h = await hear(tts, turn(s1, s2, s3))
    assert h.words == words_of(s1 + s2 + s3)
    assert words_of(" ".join(h.text)) == words_of(s1 + s2 + s3)
    assert live_texts(tts) == ["Your balance is 340 points, "]


@pytest.mark.parametrize("variant", WS)
async def test_ws_fully_cached_context_closes(variant, cfg_factory) -> None:
    s = "Thanks for calling, how can I help you today?"
    await prime(variant, cfg_factory, [s])
    tts = make(variant, cfg_factory())
    t0 = time.perf_counter()
    h = await hear(tts, turn(s) + [TTSSpeakFrame(s)])
    elapsed = time.perf_counter() - t0
    assert h.words == words_of(s) * 2
    assert live_texts(tts) == []
    assert elapsed < 1.5, elapsed  # a context left open would stall 3 s each
    assert (h.started, h.stopped) == (2, 2)


@pytest.mark.parametrize("variant", TIMESTAMPED)
async def test_word_timestamps_rebased(variant, cfg_factory) -> None:
    await prime(variant, cfg_factory, [A.strip(), C.strip(), E])
    tts = make(variant, cfg_factory())
    h = await hear(tts, turn(MIXED))
    onsets = decode_timed(h.pcm, SR)
    assert [w for w, _ in onsets] == words_of(MIXED)
    pts = [p for _, p in h.word_pts]
    assert len(pts) == len(onsets)
    assert pts == sorted(pts)
    for (w, onset), p in zip(onsets, pts, strict=True):
        delta = (p - pts[0]) / 1e9 - (onset - onsets[0][1])
        assert abs(delta) < 0.035, (w, delta)


@pytest.mark.parametrize("variant", list(VARIANTS))
async def test_interruption_mid_replay(variant, cfg_factory) -> None:
    long = "Thanks for your patience today, " * 3
    await prime(variant, cfg_factory, ["Thanks for your patience today,"])
    tts = make(variant, cfg_factory())
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


@pytest.mark.parametrize("variant", list(VARIANTS))
async def test_speak_frames_and_settings_change(variant, cfg_factory) -> None:
    s = "Please hold while I check that for you."
    await prime(variant, cfg_factory, [s])
    tts = make(variant, cfg_factory())
    h = await hear(
        tts,
        [
            TTSSpeakFrame(s),
            TTSUpdateSettingsFrame(delta=TTSSettings(voice="bob")),
            SleepFrame(0.05),
            TTSSpeakFrame(s),
        ],
    )
    assert h.words == words_of(s) * 2
    assert len(live_texts(tts)) == 1  # new voice: new key, spoken live


@pytest.mark.parametrize("variant", WS)
async def test_reuse_false_contexts_end_promptly(variant, cfg_factory) -> None:
    await prime(variant, cfg_factory, ["Let me check that for you."])
    tts = make(variant, cfg_factory(), reuse_context_id_within_turn=False)
    t0 = time.perf_counter()
    h = await hear(
        tts,
        turn(
            "Let me check that for you. ",
            "Order 2231 is on its way. ",
            "Let me check that for you.",
        ),
    )
    assert h.words == words_of(
        "Let me check that for you. Order 2231 is on its way. Let me check that for you."
    )
    assert time.perf_counter() - t0 < 2.0


async def test_el_every_stream_sends_init_first(cfg_factory) -> None:
    await prime("ws-el", cfg_factory, [A.strip(), C.strip(), E])
    tts = make("ws-el", cfg_factory())
    await hear(tts, turn(MIXED, MIXED))
    streams = {c for c, _ in tts.requests}
    assert streams and streams <= set(tts.inits)


async def test_cursor_flush_budget_falls_back_to_live(cfg_factory) -> None:
    await prime("ws-cursor", cfg_factory, [A.strip(), C.strip(), E])
    tts = make("ws-cursor", cfg_factory(max_stream_flushes_per_min=3))
    h = await hear(tts, turn(MIXED) + turn(MIXED) + turn(MIXED))
    assert h.words == words_of(MIXED) * 3
    assert tts.flushes <= 3 + 3  # extra flushes capped; turn-end flushes always sent


async def test_admission_needs_three_calls() -> None:
    store = MemoryStore()
    s = "Thanks for calling, how can I help you today?"
    for n in range(3):
        cfg = MemcatConfig(store=store, admit_after_calls=3)
        tts = cached(FakeHTTPTTS)(memcat=cfg)
        await hear(tts, [TTSSpeakFrame(s)])
        await drain(cfg)
        assert len(get_runtime(cfg).index) == (0 if n < 2 else 2)  # two clauses
    tts = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=store))
    await hear(tts, [TTSSpeakFrame(s)])
    assert tts.requests == []


async def test_http_capture_needs_no_render(cfg_factory) -> None:
    s = "How can I help you today?"  # one clause: its live audio is stored as is
    tts = make("http", cfg_factory())
    await hear(tts, [TTSSpeakFrame(s)])
    await drain(tts._memcat_cfg)
    m = get_runtime(tts._memcat_cfg).metrics.snapshot()
    assert m["captures"] == 1 and m["renders"] == 0 and m["entries_stored"] == 1


@pytest.mark.parametrize("variant", WS)
async def test_ws_render_fills_cache(variant, cfg_factory) -> None:
    s = "Thanks for waiting today, your order 3321 shipped, and it should arrive soon."
    tts = make(variant, cfg_factory())
    await hear(tts, turn(s))
    await drain(tts._memcat_cfg)
    m = get_runtime(tts._memcat_cfg).metrics.snapshot()
    assert m["entries_stored"] == 2
    tts2 = make(variant, cfg_factory())
    h = await hear(tts2, turn(s))
    assert h.words == words_of(s)
    assert live_texts(tts2) == ["your order 3321 shipped, "]


async def test_warm_skips_admission() -> None:
    store = MemoryStore()
    s = "Thank you for calling ABC Shop."
    cfg = MemcatConfig(store=store, warm=[s])
    tts = cached(FakeWSTTS)(memcat=cfg)
    await hear(tts, [SleepFrame(0.01)])
    await drain(cfg)
    tts2 = cached(FakeWSTTS)(memcat=MemcatConfig(store=store))
    h = await hear(tts2, [TTSSpeakFrame(s)])
    assert h.words == words_of(s) and tts2.requests == []


async def test_purge_removes_entries(cfg_factory) -> None:
    s = "Thanks for calling, how can I help you today?"
    await prime("http", cfg_factory, [s])
    tts = make("http", cfg_factory())
    await hear(tts, [TTSSpeakFrame(s)])
    assert tts.requests == []
    assert await tts.memcat_purge(s) >= 1
    tts2 = make("http", cfg_factory(admit_after_calls=3))
    await hear(tts2, [TTSSpeakFrame(s)])
    assert tts2.requests == [s]


@pytest.mark.parametrize("variant", list(VARIANTS))
async def test_kill_switch_is_pure_passthrough(variant) -> None:
    cls, kw = VARIANTS[variant]
    s = "Thanks for waiting, your order 1234 has shipped."
    plain = cls(**kw)
    h0 = await hear(plain, turn(s))
    tts = cached(cls)(memcat=MemcatConfig(enabled=False), **kw)
    h1 = await hear(tts, turn(s))
    assert h1.words == h0.words and h1.text == h0.text
    assert live_texts(tts) == live_texts(plain) or [r[1] for r in tts.requests] == [
        r[1] for r in plain.requests
    ]
    assert [type(f).__name__ for f in h1.frames] == [type(f).__name__ for f in h0.frames]


async def test_http_and_ws_share_nothing_across_namespaces() -> None:
    store = MemoryStore()
    s = "Thanks for calling, how can I help you today?"
    a = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=store, namespace="a", admit_after_calls=1))
    await hear(a, [TTSSpeakFrame(s)])
    await drain(a._memcat_cfg)
    b = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=store, namespace="b", admit_after_calls=1))
    await hear(b, [TTSSpeakFrame(s)])
    assert b.requests == [s]


async def test_concurrent_calls_share_one_runtime(cfg_factory) -> None:
    s = "Thanks for calling, how can I help you today?"
    await prime("ws-ctx", cfg_factory, [s])
    bots = [make("ws-ctx", cfg_factory()) for _ in range(8)]
    results = await asyncio.gather(*(hear(b, turn(s + " ", "Order 9981 is ready.")) for b in bots))
    for b, h in zip(bots, results, strict=True):
        assert h.words == words_of(s + " Order 9981 is ready.")
        assert live_texts(b) == ["Order 9981 is ready."]
    assert len({id(get_runtime(b._memcat_cfg)) for b in bots}) == 1


async def test_ws_leading_mode_hit_after_live_is_sent_live(cfg_factory) -> None:
    """A WebSocket service with no certified adapter only replays leading clauses."""

    class PlainWS(FakeWSTTS):
        __memcat_adapter__ = None

    warm = cached(PlainWS)(memcat=cfg_factory())
    await hear(warm, [TTSSpeakFrame(p) for p in (A.strip(), C.strip(), E)])
    await drain(warm._memcat_cfg)
    tts = cached(PlainWS)(memcat=cfg_factory())
    assert tts._memcat_mode == "leading"
    h = await hear(tts, turn(MIXED))
    assert h.words == words_of(MIXED)
    sent = " ".join(live_texts(tts))
    assert A.strip() not in sent
    assert C.strip() in sent and E in sent


@pytest.mark.parametrize("codec_name", ["pcm16", "mulaw", "opus"])
@pytest.mark.parametrize("variant", ["http", "ws-ctx"])
async def test_replay_from_store_after_restart(variant, codec_name) -> None:
    import memcat

    store = MemoryStore()
    await prime(
        variant,
        lambda: MemcatConfig(store=store, admit_after_calls=1, codec=codec_name),
        [A.strip(), C.strip(), E],
    )
    await memcat.aclose()  # new process: L1 empty, entries come from the store
    tts = make(variant, MemcatConfig(store=store, codec=codec_name))
    h = await hear(tts, turn(MIXED))
    assert h.words == words_of(MIXED)
    assert "".join(live_texts(tts)).split() == (B + D).split()
