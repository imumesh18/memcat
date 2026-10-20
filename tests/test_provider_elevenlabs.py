"""Memcat with Pipecat's real ElevenLabs services, against a local ElevenLabs.

The services are the shipped ``ElevenLabsTTSService`` (multi-context WebSocket) and
``ElevenLabsHttpTTSService`` (HTTP stream-with-timestamps); only the server on the
other end is local (:mod:`tests.emulators.elevenlabs`). Audio is tone-per-word, so
the listener's word order is decoded from what actually played.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import aiohttp
import pytest
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService, ElevenLabsTTSService
from pipecat.tests.utils import SleepFrame, run_test

import memcat
from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.runtime import get_runtime
from tests.emulators.elevenlabs import ElevenLabsEmulator
from tests.fakes import SR, _norm, decode, decode_timed, words_of
from tests.harness import Heard, Playout, turn
from tests.util import drain

A = "Thanks for waiting today, "
B = "your order 48213 has shipped, "
C = "and it should arrive soon — "
D = "the tracking code is ZX81QP, "
E = "is there anything else I can help with?"
MIXED = A + B + C + D + E  # hit, live, hit, live, hit

MODEL = "eleven_flash_v2_5"
VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75}


# ------------------------------------------------------------------ harness


async def hear(tts: Any, frames: Sequence[Frame]) -> Heard:
    """``tests.harness.hear`` with a playout stage after the service."""
    down, _ = await run_test(
        Pipeline([tts, Playout()]),
        frames_to_send=list(frames),
        pipeline_params=PipelineParams(audio_out_sample_rate=SR, audio_in_sample_rate=SR),
        start_timeout=5.0,
    )
    pcm = b"".join(f.audio for f in down if isinstance(f, TTSAudioRawFrame))
    h = Heard(list(down), pcm, decode(pcm))
    for f in down:
        if isinstance(f, TTSTextFrame):
            h.text.append(f.text)
            if f.pts is not None:
                h.word_pts.append((f.text, f.pts))
        elif isinstance(f, TTSStartedFrame):
            h.started += 1
        elif isinstance(f, TTSStoppedFrame):
            h.stopped += 1
    return h


# ----------------------------------------------------------------- fixtures


@pytest.fixture
async def emu():
    async with ElevenLabsEmulator() as e:
        yield e
        # Render twins outlive the call and keep their sockets open: close them
        # before the server goes away.
        await memcat.aclose()


@pytest.fixture
def cfg_factory():
    # render=False: these tests are about replay; clause audio is captured from
    # single-clause live streams. Rendering is covered by its own test.
    store = MemoryStore()
    return lambda **kw: MemcatConfig(store=store, **{"admit_after_calls": 1, "render": False, **kw})


@pytest.fixture
async def session():
    async with aiohttp.ClientSession() as s:
        yield s


def ws_kwargs(emu: ElevenLabsEmulator) -> dict[str, Any]:
    return {
        "api_key": "test",
        "url": emu.ws_url,
        "sample_rate": SR,
        "settings": ElevenLabsTTSService.Settings(voice="v", model=MODEL, **VOICE_SETTINGS),
    }


def http_kwargs(emu: ElevenLabsEmulator, session: aiohttp.ClientSession) -> dict[str, Any]:
    return {
        "api_key": "test",
        "aiohttp_session": session,
        "base_url": emu.http_url,
        "sample_rate": SR,
        "settings": ElevenLabsHttpTTSService.Settings(voice="v", model=MODEL, **VOICE_SETTINGS),
    }


def ws(emu: ElevenLabsEmulator, cfg: MemcatConfig | None = None, **extra: Any) -> Any:
    kw = {**ws_kwargs(emu), **extra}
    if cfg is None:
        return ElevenLabsTTSService(**kw)
    return cached(ElevenLabsTTSService)(memcat=cfg, **kw)


def http(emu: ElevenLabsEmulator, session: aiohttp.ClientSession, cfg: MemcatConfig) -> Any:
    return cached(ElevenLabsHttpTTSService)(memcat=cfg, **http_kwargs(emu, session))


async def prime_ws(emu: ElevenLabsEmulator, cfg_factory: Any, phrases: list[str]) -> None:
    """Speak each phrase alone so its clause audio is captured live."""
    cfg = cfg_factory()
    await hear(ws(emu, cfg), [TTSSpeakFrame(p) for p in phrases])
    await drain(cfg)


class Wire:
    """What the emulator saw from one point on."""

    def __init__(self, emu: ElevenLabsEmulator):
        self.emu = emu
        self.mark = len(emu.received)
        self.http_mark = len(emu.http_requests)
        self.known = set(emu.contexts)

    def contexts(self) -> list[Any]:
        cs = [c for cid, c in self.emu.contexts.items() if cid not in self.known]
        return sorted(cs, key=lambda c: c.opened_at)

    def texts(self) -> list[str]:
        new = {c.cid for c in self.contexts()}
        return [t for cid, t in self.emu.ws_texts() if cid in new]

    def http_texts(self) -> list[str]:
        return [r["json"]["text"] for r in self.emu.http_requests[self.http_mark :]]

    def assert_streams_well_formed(self, *, one_at_a_time: bool = True) -> None:
        cs = self.contexts()
        assert self.emu.errors == []
        for c in cs:
            msgs = self.emu.messages_for(c.cid)
            assert c.anomalies == [], (c.cid, c.anomalies)
            assert c.init is not None and msgs[0] is c.init, (c.cid, msgs[:2])
            assert c.init["voice_settings"] == VOICE_SETTINGS
            kinds = [
                "close" if m.get("close_context") else "flush" if m.get("flush") else "text"
                for m in msgs[1:]
                if m.get("text") != ""
            ]
            assert kinds and kinds[0] == "text", (c.cid, kinds)
            assert "flush" in kinds and kinds[-1] == "close", (c.cid, kinds)
            assert kinds.index("flush") > max(i for i, k in enumerate(kinds) if k == "text")
            assert c.final_at is not None, c.cid  # isFinal went back: nothing left open
        if one_at_a_time:
            for prev, nxt in zip(cs, cs[1:], strict=False):
                assert prev.final_at is not None and nxt.opened_at >= prev.final_at, (
                    prev.cid,
                    nxt.cid,
                )


def norm_join(texts: Sequence[str]) -> list[str]:
    return " ".join(texts).split()


# -------------------------------------------------------------------- tests


async def test_ws_plain_and_wrapped_match_with_empty_cache(emu, cfg_factory) -> None:
    s = "Hello and welcome to Acme support. " + MIXED
    h0 = await hear(ws(emu), turn(s))
    tts = ws(emu, cfg_factory())
    h1 = await hear(tts, turn(s))
    assert h0.words == words_of(s)
    assert h1.words == h0.words
    assert h1.text == h0.text
    assert (h1.started, h1.stopped) == (h0.started, h0.stopped) == (1, 1)
    assert not tts._memcat_ctx and not tts._memcat_streams and not tts._memcat_inflight


async def test_ws_mixed_runs_keep_order(emu, cfg_factory) -> None:
    await prime_ws(emu, cfg_factory, [A.strip(), C.strip(), E])
    tts = ws(emu, cfg_factory())
    wire = Wire(emu)
    h = await hear(tts, turn(MIXED))
    assert h.words == words_of(MIXED)
    assert words_of(" ".join(h.text)) == words_of(MIXED)
    assert (h.started, h.stopped) == (1, 1)
    # Only the live clauses went over the wire, each on its own provider stream.
    assert norm_join(wire.texts()) == (B + D).split()
    assert len(wire.contexts()) == 2
    assert all(c.cid.endswith(("-s1", "-s2")) for c in wire.contexts())
    wire.assert_streams_well_formed()
    st = await tts.memcat_stats()
    assert st["hits"] == 3 and st["misses"] == 2
    assert not tts._memcat_ctx and not tts._memcat_streams and not tts._memcat_inflight


async def test_ws_multi_sentence_turn_hit_after_live(emu, cfg_factory) -> None:
    s1 = "Your balance is 340 points today. "
    s2 = A + B.rstrip(", ") + ". "
    s3 = E
    await prime_ws(emu, cfg_factory, [A.strip(), E])
    tts = ws(emu, cfg_factory())
    wire = Wire(emu)
    h = await hear(tts, turn(s1, s2, s3))
    want = words_of(s1 + s2 + s3)
    assert h.words == want
    assert words_of(" ".join(h.text)) == want
    assert norm_join(wire.texts()) == (s1 + B.rstrip(", ") + ".").split()
    assert (h.started, h.stopped) == (1, 1)
    wire.assert_streams_well_formed()


def assert_pts_match_audio(h: Heard) -> None:
    """Each word's TTSTextFrame pts is where its tone starts (relative, +/-35 ms)."""
    onsets = decode_timed(h.pcm, SR)
    pts = [p for w, p in h.word_pts if _norm(w)]  # "—" has a timestamp but no audio
    assert len(pts) == len(onsets), (h.word_pts, onsets)
    assert pts == sorted(pts)
    for (w, onset), p in zip(onsets, pts, strict=True):
        delta = (p - pts[0]) / 1e9 - (onset - onsets[0][1])
        assert abs(delta) < 0.035, (w, delta)


async def test_ws_word_timestamps_rebased(emu, cfg_factory) -> None:
    # Primed with their trailing spaces: Pipecat's ElevenLabs WS service only emits
    # a word's timestamp once the alignment shows the space after it.
    await prime_ws(emu, cfg_factory, [A, C, E + " "])
    tts = ws(emu, cfg_factory())
    h = await hear(tts, turn(MIXED))
    assert [w for w, _ in decode_timed(h.pcm, SR)] == words_of(MIXED)
    assert_pts_match_audio(h)


async def test_ws_clip_from_sentence_final_clause_keeps_last_word(emu, cfg_factory) -> None:
    h0 = await hear(ws(emu), turn(MIXED))
    await prime_ws(emu, cfg_factory, [A.strip(), C.strip(), E])  # each said as a sentence
    h = await hear(ws(emu, cfg_factory()), turn(MIXED))
    assert h.words == words_of(MIXED)
    assert h.text == h0.text  # plain: one TTSTextFrame per word ("today,", "your", ...)


async def test_ws_interruption_mid_turn(emu, cfg_factory) -> None:
    long = "Thanks for your patience today, " * 3
    await prime_ws(emu, cfg_factory, ["Thanks for your patience today,"])
    tts = ws(emu, cfg_factory())
    wire = Wire(emu)
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
    assert norm_join(wire.texts()[-1:]) == after.split()
    assert len(h.words) < len(words_of(long + "and order 5522 is late, " + long + after))
    assert emu.errors == []
    live = [c for c in wire.contexts() if any("5522" in t for t in c.texts)]
    assert len(live) == 1 and live[0].closed  # the interrupted turn's stream was ended
    for c in wire.contexts():
        assert c.anomalies == [], (c.cid, c.anomalies)
        assert c.init is not None and c.init["voice_settings"] == VOICE_SETTINGS
        # Every stream was closed: by Memcat when it ended, or on interruption.
        assert c.closed, c.cid


async def test_ws_fully_cached_turn_is_fast(emu, cfg_factory) -> None:
    s = "Thanks for calling, how can I help you today?"
    await prime_ws(emu, cfg_factory, ["Thanks for calling, ", "how can I help you today?"])
    tts = ws(emu, cfg_factory())
    wire = Wire(emu)
    t0 = time.perf_counter()
    h = await hear(tts, turn(s) + [TTSSpeakFrame(s)])
    elapsed = time.perf_counter() - t0
    assert h.words == words_of(s) * 2
    assert wire.texts() == [] and wire.contexts() == []
    assert elapsed < 1.5, elapsed
    assert (h.started, h.stopped) == (2, 2)


async def test_ws_render_fills_cache(emu, cfg_factory) -> None:
    """Clauses spoken together in one stream are rendered alone by a twin service,
    one after another on the same twin."""
    s = "Thanks for waiting today, your order 3321 shipped, and it should arrive soon."
    cfg = cfg_factory(render=True)
    await hear(ws(emu, cfg), turn(s))
    # Two renders take ~0.1 s here. A twin of ElevenLabsTTSService that keeps
    # Pipecat's playback pause (no transport ever lifts it) would latch after one.
    await get_runtime(cfg).worker.drain_all(2.0)
    m = get_runtime(cfg).metrics.snapshot()
    assert m["entries_stored"] == 2 and m["renders"] == 2, m
    tts = ws(emu, cfg_factory())
    wire = Wire(emu)
    h = await hear(tts, turn(s))
    assert h.words == words_of(s)
    assert wire.texts() == ["your order 3321 shipped, "]


# --------------------------------------------------------------------- HTTP


async def prime_http(emu, session, cfg_factory, phrases: list[str]) -> None:
    cfg = cfg_factory()
    await hear(http(emu, session, cfg), [TTSSpeakFrame(p) for p in phrases])
    await drain(cfg)


async def test_http_mixed_runs_keep_order(emu, session, cfg_factory) -> None:
    await prime_http(emu, session, cfg_factory, [A.strip(), C.strip(), E])
    tts = http(emu, session, cfg_factory())
    wire = Wire(emu)
    h = await hear(tts, turn(MIXED))
    assert h.words == words_of(MIXED)
    assert words_of(" ".join(h.text)) == words_of(MIXED)
    assert (h.started, h.stopped) == (1, 1)
    assert norm_join(wire.http_texts()) == (B + D).split()
    for r in emu.http_requests[wire.http_mark :]:
        assert r["query"]["output_format"] == "pcm_16000"
        assert r["json"]["model_id"] == MODEL
        assert r["json"]["voice_settings"] == VOICE_SETTINGS
    assert_pts_match_audio(h)


async def test_http_previous_text_across_sentences(emu, session, cfg_factory) -> None:
    """A cached sentence still reaches the next live request as previous_text."""
    s1 = "Thanks for waiting today. "
    s2 = "Your order 48213 has shipped."
    await prime_http(emu, session, cfg_factory, [s1.strip()])
    tts = http(emu, session, cfg_factory())
    wire = Wire(emu)
    h = await hear(tts, turn(s1, s2))
    assert h.words == words_of(s1 + s2)
    reqs = emu.http_requests[wire.http_mark :]
    assert [r["json"]["text"].strip() for r in reqs] == [s2]
    assert s1.strip() in reqs[0]["json"].get("previous_text", "")


async def test_http_previous_text_within_sentence(emu, session, cfg_factory) -> None:
    """Each live request's previous_text holds everything said before it in the turn,
    cached clauses included, in spoken order."""
    nxt = "Your parcel 7781 is late today."
    await prime_http(emu, session, cfg_factory, [A.strip(), C.strip(), E])
    tts = http(emu, session, cfg_factory())
    wire = Wire(emu)
    h = await hear(tts, turn(MIXED + " ", nxt))
    assert h.words == words_of(MIXED + " " + nxt)
    reqs = emu.http_requests[wire.http_mark :]
    assert [r["json"]["text"].strip() for r in reqs] == [B.strip(), D.strip(), nxt]
    prev = [r["json"].get("previous_text", "").split() for r in reqs]
    assert prev[0] == A.split()
    assert prev[1] == (A + B + C).split()
    assert prev[2] == MIXED.split()
