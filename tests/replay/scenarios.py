"""ElevenLabs scenarios, run against whatever ``wire`` points at.

The same scenario runs three ways: against the local emulator to prove the
recorder and player (``tests/test_replay_machinery.py``), against the real API to
record (``MEMCAT_RECORD=1``), and against a recording (``tests/test_replay_elevenlabs.py``).
Each returns a ``Run``: what the listener received and when.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import aiohttp
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService, ElevenLabsTTSService
from pipecat.tests.utils import SleepFrame, run_test

import memcat
from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.runtime import get_runtime
from tests.fakes import _norm
from tests.harness import Playout, turn

A = "Thanks for waiting today, "
B = "your order 48213 has shipped, "
C = "and it should arrive soon — "
D = "the tracking code is ZX81QP, "
E = "is there anything else I can help with?"
MIXED = A + B + C + D + E  # hit, live, hit, live, hit
PRIMES = [A.strip(), C.strip(), E]
AFTER = "Sure, is there anything else I can help with?"

MODEL = "eleven_flash_v2_5"
VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75}


class Wire(Protocol):
    @property
    def ws_url(self) -> str: ...
    @property
    def http_url(self) -> str: ...


@dataclass
class Target:
    wire: Wire
    voice: str
    sample_rate: int
    session: aiohttp.ClientSession
    model: str = MODEL


@dataclass
class Run:
    pcm: bytes = b""
    words: list[str] = field(default_factory=list)
    stopped: int = 0
    ttfa: list[float] = field(default_factory=list)  # per turn: text in -> first audio out
    hits: int = 0
    enabled: bool = True
    chars_saved: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "words": self.words,
            "ttfa_ms": [round(x * 1000, 1) for x in self.ttfa],
            "hits": self.hits,
        }


class _In(FrameProcessor):
    """Stamps when each turn's text reaches the TTS."""

    def __init__(self, clock: _Clock):
        super().__init__()
        self._c = clock
        self._open = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._open = True
            elif isinstance(frame, TTSSpeakFrame) or (isinstance(frame, TextFrame) and self._open):
                self._open = False
                self._c.text_at.append(time.monotonic())
        await self.push_frame(frame, direction)


class _Out(FrameProcessor):
    """Stamps the first audio of each Pipecat context (one per turn)."""

    def __init__(self, clock: _Clock):
        super().__init__()
        self._c = clock

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, TTSAudioRawFrame)
            and frame.context_id not in self._c.audio_at
        ):
            self._c.audio_at[frame.context_id] = time.monotonic()
        await self.push_frame(frame, direction)


@dataclass
class _Clock:
    text_at: list[float] = field(default_factory=list)
    audio_at: dict[Any, float] = field(default_factory=dict)  # in order of first audio

    def ttfa(self) -> list[float]:
        return [a - t for t, a in zip(self.text_at, self.audio_at.values(), strict=False)]


async def hear(tts: Any, frames: Sequence[Frame], sr: int) -> Run:
    clock = _Clock()
    down, _ = await run_test(
        Pipeline([_In(clock), tts, _Out(clock), Playout()]),
        frames_to_send=list(frames),
        pipeline_params=PipelineParams(audio_out_sample_rate=sr, audio_in_sample_rate=sr),
        start_timeout=10.0,
    )
    r = Run(ttfa=clock.ttfa())
    r.pcm = b"".join(f.audio for f in down if isinstance(f, TTSAudioRawFrame))
    for f in down:
        if isinstance(f, TTSTextFrame):
            r.words += [w for w in (_norm(x) for x in f.text.split()) if w]
        elif isinstance(f, TTSStoppedFrame):
            r.stopped += 1
    if hasattr(tts, "memcat_stats"):
        st = await tts.memcat_stats()
        r.hits = int(st.get("hits", 0))
        r.enabled = bool(st.get("enabled", True))
        r.chars_saved = int(st.get("chars_cached", 0))
    return r


def _ws(t: Target, cfg: MemcatConfig | None, **extra: Any) -> Any:
    kw: dict[str, Any] = {
        "api_key": "replay-dummy-key",
        "url": t.wire.ws_url,
        "sample_rate": t.sample_rate,
        "settings": ElevenLabsTTSService.Settings(voice=t.voice, model=t.model, **VOICE_SETTINGS),
        **extra,
    }
    return (
        ElevenLabsTTSService(**kw)
        if cfg is None
        else cached(ElevenLabsTTSService)(memcat=cfg, **kw)
    )


def _http(t: Target, cfg: MemcatConfig | None) -> Any:
    kw: dict[str, Any] = {
        "api_key": "replay-dummy-key",
        "aiohttp_session": t.session,
        "base_url": t.wire.http_url,
        "sample_rate": t.sample_rate,
        "settings": ElevenLabsHttpTTSService.Settings(
            voice=t.voice, model=t.model, **VOICE_SETTINGS
        ),
    }
    if cfg is None:
        return ElevenLabsHttpTTSService(**kw)
    return cached(ElevenLabsHttpTTSService)(memcat=cfg, **kw)


def _cfg(**kw: Any) -> MemcatConfig:
    return MemcatConfig(store=MemoryStore(), **{"admit_after_calls": 1, "render": False, **kw})


async def _drain(cfg: MemcatConfig) -> None:
    await get_runtime(cfg).worker.drain_all(30)


def _speak_each(lines: Sequence[str]) -> list[Frame]:
    return [TTSSpeakFrame(x) for x in lines]


# ---------------------------------------------------------------- scenarios


async def plain_ws(t: Target) -> list[Run]:
    """No Memcat: the baseline every other scenario is compared with."""
    return [await hear(_ws(t, None), turn(MIXED), t.sample_rate)]


async def plain_http(t: Target) -> list[Run]:
    return [await hear(_http(t, None), [TTSSpeakFrame(MIXED)], t.sample_rate)]


async def memcat_ws_mixed(t: Target) -> list[Run]:
    """Three clauses captured live, then a sentence of hit, live, hit, live, hit."""
    cfg = _cfg()
    prime = await hear(_ws(t, cfg), _speak_each(PRIMES), t.sample_rate)
    await _drain(cfg)
    mixed = await hear(
        _ws(t, MemcatConfig(store=cfg.store, render=False)), turn(MIXED), t.sample_rate
    )
    return [prime, mixed]


async def memcat_ws_render(t: Target) -> list[Run]:
    """A cold sentence, clauses rendered by background twins, then the same sentence."""
    cfg = _cfg(render=True)
    cold = await hear(_ws(t, cfg), turn(MIXED), t.sample_rate)
    await _drain(cfg)
    warm = await hear(
        _ws(t, MemcatConfig(store=cfg.store, render=False)), turn(MIXED), t.sample_rate
    )
    return [cold, warm]


async def memcat_ws_interrupt(t: Target) -> list[Run]:
    """Interrupted after the cached opener, while the first live clause is pending."""
    cfg = _cfg()
    await hear(_ws(t, cfg), _speak_each(PRIMES), t.sample_rate)
    await _drain(cfg)
    frames = turn(MIXED) + [SleepFrame(0.15), InterruptionFrame(), SleepFrame(0.1)] + turn(AFTER)
    return [await hear(_ws(t, MemcatConfig(store=cfg.store, render=False)), frames, t.sample_rate)]


async def plain_ws_interrupt(t: Target) -> list[Run]:
    """No Memcat: the same barge-in, for comparison with ``memcat_ws_interrupt``."""
    frames = turn(MIXED) + [SleepFrame(0.15), InterruptionFrame(), SleepFrame(0.1)] + turn(AFTER)
    return [await hear(_ws(t, None), frames, t.sample_rate)]


async def memcat_ws_two_turns(t: Target) -> list[Run]:
    cfg = _cfg()
    await hear(_ws(t, cfg), _speak_each(PRIMES), t.sample_rate)
    await _drain(cfg)
    frames = turn(MIXED) + [SleepFrame(0.5)] + turn(AFTER)
    return [await hear(_ws(t, MemcatConfig(store=cfg.store, render=False)), frames, t.sample_rate)]


async def memcat_http_mixed(t: Target) -> list[Run]:
    """HTTP: clauses captured live, then the mixed sentence (previous_text kept).

    Each clause is primed in a call of its own, as admission would see it in
    production. Spoken back to back in one call, the ``previous_text`` each request
    carries would depend on when Pipecat's stop frame resets it, which is timing.
    """
    cfg = _cfg()
    primes = [await hear(_http(t, cfg), [TTSSpeakFrame(p)], t.sample_rate) for p in PRIMES]
    await _drain(cfg)
    mixed = await hear(
        _http(t, MemcatConfig(store=cfg.store, render=False)), [TTSSpeakFrame(MIXED)], t.sample_rate
    )
    return [*primes, mixed]


async def error_bad_voice(t: Target) -> list[Run]:
    """A voice id that does not exist: whatever the provider does, Memcat must not hang."""
    bad = Target(t.wire, "memcat-no-such-voice", t.sample_rate, t.session, t.model)
    return [await hear(_ws(bad, _cfg()), turn("Hello there, how are you today?"), t.sample_rate)]


Scenario = Callable[[Target], Awaitable[list[Run]]]

SCENARIOS: dict[str, Scenario] = {
    "plain_ws": plain_ws,
    "plain_http": plain_http,
    "memcat_ws_mixed": memcat_ws_mixed,
    "memcat_ws_render": memcat_ws_render,
    "memcat_ws_interrupt": memcat_ws_interrupt,
    "plain_ws_interrupt": plain_ws_interrupt,
    "memcat_ws_two_turns": memcat_ws_two_turns,
    "memcat_http_mixed": memcat_http_mixed,
    "error_bad_voice": error_bad_voice,
}


async def run(name: str, t: Target) -> list[Run]:
    try:
        return await SCENARIOS[name](t)
    finally:
        await memcat.aclose()  # close render twins before the wire goes away
