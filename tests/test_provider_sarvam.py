"""Memcat with Pipecat's real Sarvam (Bulbul) services, against a local Sarvam emulator.

``SarvamTTSService`` passes ``pause_frame_processing=True`` (and cannot be told
otherwise: the kwarg is hard-coded in its ``super().__init__`` call), so after
each turn it waits for the ``BotStoppedSpeakingFrame`` an output transport sends.
These tests put a minimal output transport stand-in after the service
(:class:`FakeOutput`) that sends Bot*SpeakingFrame upstream on the same triggers
as ``BaseOutputTransport`` (first TTS audio; ``TTSStoppedFrame`` after audio;
3 s of idle as a fallback), just without real-time playback.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.sarvam.tts import SarvamHttpTTSService, SarvamTTSService
from pipecat.tests.utils import SleepFrame

import memcat
from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.runtime import get_runtime
from tests.emulators.sarvam import SarvamEmulator
from tests.fakes import SR, words_of
from tests.harness import Heard, hear, turn
from tests.util import drain

A = "Thanks for waiting today, "
B = "your order 48213 has shipped, "
C = "and it should arrive soon — "
D = "the tracking code is ZX81QP, "
E = "is there anything else I can help with?"
MIXED = A + B + C + D + E  # hit, live, hit, live, hit

VAD_STOP_FALLBACK_S = 3.0


class FakeOutput(FrameProcessor):
    """Sends BotStarted/BotStoppedSpeakingFrame upstream like BaseOutputTransport."""

    def __init__(self) -> None:
        super().__init__()
        self._speaking = False
        self._audio_received = False
        self._idle: asyncio.TimerHandle | None = None
        self._idle_task: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSAudioRawFrame):
                self._audio_received = True
                await self._started()
                self._arm_fallback()
            elif isinstance(frame, TTSStoppedFrame) and self._audio_received:
                await self._stopped()
            elif isinstance(frame, (EndFrame,)) and self._idle_task:
                self._idle_task.cancel()
        await self.push_frame(frame, direction)

    def _arm_fallback(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()

        async def fallback() -> None:
            await asyncio.sleep(VAD_STOP_FALLBACK_S)
            await self._stopped()

        self._idle_task = asyncio.ensure_future(fallback())

    async def _started(self) -> None:
        if self._speaking:
            return
        self._speaking = True
        await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _stopped(self) -> None:
        if not self._speaking:
            return
        self._speaking = False
        self._audio_received = False
        await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def cleanup(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
        await super().cleanup()


async def hear_ws(tts: Any, frames: list[Frame]) -> Heard:
    return await hear(Pipeline([tts, FakeOutput()]), frames)


@pytest.fixture
async def emu():
    e = await SarvamEmulator().start()
    yield e
    await memcat.aclose()  # close render twins while their server is still up
    await e.stop()


@pytest.fixture
def cfg_factory():
    store = MemoryStore()
    return lambda **kw: MemcatConfig(store=store, **{"admit_after_calls": 1, **kw})


def ws_service(emu: SarvamEmulator, cfg: MemcatConfig | None = None, **kw: Any) -> Any:
    """The service under test. ``svc.ports`` collects its sockets' client ports, so its
    traffic can be told apart from that of Memcat's render twins (same URL)."""
    args = {"api_key": "test", "url": emu.ws_url, "sample_rate": SR, **kw}
    cls = SarvamTTSService if cfg is None else cached(SarvamTTSService)
    svc = cls(**args) if cfg is None else cls(memcat=cfg, **args)
    svc.ports = set()

    @svc.event_handler("on_connected")
    async def _connected(s: Any) -> None:
        s.ports.add(s._websocket.local_address[1])

    return svc


async def prime_ws(emu: SarvamEmulator, cfg: MemcatConfig, phrases: list[str]) -> None:
    tts = ws_service(emu, cfg)
    await hear_ws(tts, [TTSSpeakFrame(p) for p in phrases])
    await drain(cfg)
    m = get_runtime(cfg).metrics.snapshot()
    # Leading mode never captures live audio; every clause is rendered by a twin.
    assert m["captures"] == 0 and m["entries_stored"] == m["renders"] >= len(phrases), m
    emu.conns.clear()


def assert_wire_ok(emu: SarvamEmulator, ports: set[int] | None = None) -> None:
    assert emu.of(ports)
    for c in emu.of(ports):
        assert c.path == "/text-to-speech/ws"
        assert c.query["model"] == ["bulbul:v3"]
        assert c.query["send_completion_event"] == ["true"]
        assert c.headers.get("api-subscription-key") == "test"
        if c.received:
            assert c.received[0]["type"] == "config"
            assert c.received[0]["data"]["speech_sample_rate"] == str(SR)


# --------------------------------------------------------------------- WebSocket


async def test_ws_plain_and_wrapped_match_on_empty_cache(emu, cfg_factory) -> None:
    s1 = "Hello and welcome to Acme support. "
    s2 = "Your balance is 340 points, and you can redeem them any time."
    frames = lambda: [*turn(s1, s2), TTSSpeakFrame("Please hold while I check that for you.")]  # noqa: E731
    plain = ws_service(emu)
    h0 = await hear_ws(plain, frames())
    sent0 = emu.texts(plain.ports)
    tts = ws_service(emu, cfg_factory())
    assert tts._memcat_mode == "leading"
    h1 = await hear_ws(tts, frames())
    sent1 = emu.texts(tts.ports)
    want = words_of(s1 + s2 + " Please hold while I check that for you.")
    assert h0.words == want
    assert h1.words == h0.words
    assert h1.text == h0.text
    assert (h1.started, h1.stopped) == (h0.started, h0.stopped) == (2, 2)
    assert sent1 == sent0
    assert [c.types() for c in emu.of(tts.ports)] == [c.types() for c in emu.of(plain.ports)]
    assert_wire_ok(emu)


async def test_ws_leading_mode_replays_only_leading_clauses(emu, cfg_factory) -> None:
    await prime_ws(emu, cfg_factory(), [A.strip(), C.strip(), E])
    tts = ws_service(emu, cfg_factory())
    h = await hear_ws(tts, turn(MIXED))
    assert h.words == words_of(MIXED)
    sent = " ".join(emu.texts(tts.ports))
    assert "Thanks" not in sent  # the leading cached clause was replayed, not sent
    assert words_of(sent) == words_of(B + C + D + E)  # the rest went live, once
    assert words_of(" ".join(h.text)) == words_of(MIXED)
    assert (h.started, h.stopped) == (1, 1)
    assert not tts._memcat_lead  # no context left open
    st = await tts.memcat_stats()
    assert st["hits"] == 1 and st["misses"] == 4
    assert [m["type"] for m in emu.of(tts.ports)[0].received] == ["config", "text", "flush"]
    assert_wire_ok(emu, tts.ports)


async def test_ws_fully_cached_contexts_close_promptly(emu, cfg_factory) -> None:
    s = "Thanks for calling, how can I help you today?"
    await prime_ws(emu, cfg_factory(), [s])
    tts = ws_service(emu, cfg_factory())
    t0 = time.perf_counter()
    h = await hear_ws(tts, [TTSSpeakFrame(s), *turn(s)])
    elapsed = time.perf_counter() - t0
    assert h.words == words_of(s) * 2
    assert emu.texts(tts.ports) == []
    assert [c.types() for c in emu.of(tts.ports)] == [["config"]]  # no text, no flush
    assert elapsed < 1.5, elapsed  # a context left open would stall 3 s
    assert (h.started, h.stopped) == (2, 2)
    assert words_of(" ".join(h.text)) == words_of(s) * 2


async def test_ws_interruption_mid_turn(emu, cfg_factory) -> None:
    long = "Thanks for your patience today, " * 3
    await prime_ws(emu, cfg_factory(), ["Thanks for your patience today,"])
    tts = ws_service(emu, cfg_factory())
    after = "Sure, the code is 7721, anything else I can help with?"
    frames = (
        turn(long + "and order 5522 is late, " + long)
        + [SleepFrame(0.01), InterruptionFrame(), SleepFrame(0.05)]
        + turn(after)
    )
    h = await hear_ws(tts, frames)
    want = words_of(after)
    assert h.words[-len(want) :] == want
    assert not tts._memcat_lead and not tts._memcat_ctx and not tts._memcat_streams
    conns = emu.of(tts.ports)
    assert len(conns) == 2  # InterruptibleTTSService reconnected on the interruption
    assert conns[0].closed
    assert words_of(" ".join(conns[-1].texts())) == want


async def test_ws_render_twin_renders_clause_after_clause(emu, cfg_factory) -> None:
    """Leading mode fills the cache only by rendering clauses in a twin. The twin
    has no output transport, so a twin that kept SarvamTTSService's pause until
    "bot stopped speaking" would render one clause and time out on the next."""
    cfg = cfg_factory()
    tts = ws_service(emu, cfg)
    phrases = [A.strip(), C.strip(), E]
    t0 = time.perf_counter()
    await hear_ws(tts, [TTSSpeakFrame(p) for p in phrases])
    await drain(cfg)
    m = get_runtime(cfg).metrics.snapshot()
    assert m["renders"] == 3 and m["entries_stored"] == 3 and m["render_failures"] == 0, m
    assert time.perf_counter() - t0 < 5.0  # a stuck twin times out after 15 s
    twin_texts = [t for c in emu.conns if c.peer_port not in tts.ports for t in c.texts()]
    assert sorted(twin_texts) == sorted(phrases)


# -------------------------------------------------------------------------- HTTP


P = "we really appreciate your patience, "  # cacheable, unlike B and D (digits, codes)


def http_session() -> aiohttp.ClientSession:
    # Render twins open their own sessions; this header tells our traffic apart.
    return aiohttp.ClientSession(headers={"x-client": "main"})


async def test_http_mixed_runs_keep_order_and_capture(emu, cfg_factory) -> None:
    async with http_session() as session:

        def make(cfg: MemcatConfig) -> Any:
            return cached(SarvamHttpTTSService)(
                memcat=cfg,
                api_key="test",
                aiohttp_session=session,
                base_url=emu.http_url,
                sample_rate=SR,
            )

        warm_cfg = cfg_factory()
        await hear(make(warm_cfg), [TTSSpeakFrame(p) for p in (A.strip(), C.strip(), E)])
        await drain(warm_cfg)
        m = get_runtime(warm_cfg).metrics.snapshot()
        assert m["captures"] == 3 and m["renders"] == 0 and m["entries_stored"] == 3, m
        emu.clear_http()

        cfg = cfg_factory()
        tts = make(cfg)
        h = await hear(tts, turn(MIXED))
        assert h.words == words_of(MIXED)
        assert emu.http_texts() == [B, D]  # only the live clauses were requested, in order
        assert words_of(" ".join(h.text)) == words_of(MIXED)
        assert (h.started, h.stopped) == (1, 1)
        st = await tts.memcat_stats()
        assert st["hits"] == 3 and st["misses"] == 2
        body = emu.http_requests[0]
        assert body["speech_sample_rate"] == SR and body["model"] == "bulbul:v3"
        assert all(hd.get("api-subscription-key") == "test" for hd in emu.http_headers)
        await drain(cfg)
        m = get_runtime(cfg).metrics.snapshot()
        assert m["captures"] == 3 and m["entries_stored"] == 3, m  # B, D are not cacheable

        # A cacheable clause spoken live on its own, between cached ones, is captured.
        emu.clear_http()
        mixed2 = A + P + C + E
        tts2 = make(cfg_factory())
        h2 = await hear(tts2, turn(mixed2))
        assert h2.words == words_of(mixed2)
        assert words_of(" ".join(h2.text)) == words_of(mixed2)
        assert emu.http_texts() == [P]
        await drain(cfg)
        m = get_runtime(cfg).metrics.snapshot()
        assert m["captures"] == 4 and m["renders"] == 0 and m["entries_stored"] == 4, m

        # Every cacheable clause is cached now: only B and D still go to the provider.
        emu.clear_http()
        h3 = await hear(make(cfg_factory()), turn(mixed2 + " ", MIXED))
        assert h3.words == words_of(mixed2 + " " + MIXED)
        assert emu.http_texts() == [B, D]


async def test_http_plain_and_wrapped_match_on_empty_cache(emu, cfg_factory) -> None:
    async with http_session() as session:
        kw = {"api_key": "test", "aiohttp_session": session, "base_url": emu.http_url}
        frames = lambda: turn(MIXED)  # noqa: E731
        h0 = await hear(SarvamHttpTTSService(sample_rate=SR, **kw), frames())
        sent0 = emu.http_texts("main")
        emu.clear_http()
        wrapped = cached(SarvamHttpTTSService)(memcat=cfg_factory(), sample_rate=SR, **kw)
        h1 = await hear(wrapped, frames())
        assert h0.words == words_of(MIXED)
        assert h1.words == h0.words and h1.text == h0.text
        assert (h1.started, h1.stopped) == (h0.started, h0.stopped) == (1, 1)
        assert emu.http_texts("main") == sent0 == [MIXED]
