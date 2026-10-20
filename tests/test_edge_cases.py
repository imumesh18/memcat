"""Edge cases where Memcat must still sound exactly like plain Pipecat."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame

import memcat
from memcat import MemcatConfig, MemoryStore, cached
from memcat.core import codec
from memcat.core.runtime import get_runtime
from memcat.core.worker import Audio, ObserveItem, Policy
from tests.fakes import SR, FakeELWSTTS, FakeHTTPTTS, FakeWSTTS, _settings, synth, words_of
from tests.harness import hear, turn
from tests.util import make, prime

A = "Thanks for waiting today, "
B = "your order 48213 has shipped, "
C = "and it should arrive soon."


@pytest.mark.parametrize("variant", ["ws-ctx", "ws-el", "ws-cursor"])
async def test_slow_provider_first_byte_is_not_dropped(variant) -> None:
    """The stream watchdog waits for a slow provider's first audio."""
    store = MemoryStore()
    cf = lambda: MemcatConfig(store=store, admit_after_calls=1)  # noqa: E731
    await prime(variant, cf, [A.strip(), C])
    tts = make(variant, cf(), delay=(1.8, 1.8))
    h = await hear(tts, turn(A + B + C))
    assert h.words == words_of(A + B + C)


@pytest.mark.parametrize("cls", [FakeWSTTS, FakeELWSTTS])
async def test_context_timeout_does_not_reuse_stream_ids(cls) -> None:
    """A context rebuilt after Pipecat times it out gets fresh stream ids."""
    s1 = "Your order 1234 is ready. "
    s2 = "Your parcel 5678 left today."
    frames = [
        LLMFullResponseStartFrame(),
        TextFrame(s1),
        TextFrame("Your "),
        SleepFrame(3.6),
        TextFrame(s2[5:]),
        LLMFullResponseEndFrame(),
    ]
    tts = cached(cls)(memcat=MemcatConfig(store=MemoryStore(), admit_after_calls=1))
    h = await hear(tts, frames)
    assert h.words == words_of(s1 + s2)
    sids = [c for c, _ in tts.requests]
    assert len(sids) == len(set(sids))


@pytest.mark.parametrize("variant", ["http", "ws-ctx", "ws-el", "ws-cursor"])
async def test_undecodable_entries_are_misses(variant, monkeypatch) -> None:
    """Opus entries in a process without PyAV are misses, in every mode."""
    store = MemoryStore()
    await prime(
        variant,
        lambda: MemcatConfig(store=store, admit_after_calls=1, codec="opus"),
        [A.strip(), C],
    )
    await memcat.aclose()
    monkeypatch.setattr(codec, "opus_available", lambda: False)
    tts = make(variant, MemcatConfig(store=store, codec="opus"))
    h = await hear(tts, turn(A + B + C))
    assert h.words == words_of(A + B + C)


@pytest.mark.parametrize("variant", ["ws-ctx", "ws-el"])
async def test_replay_error_sends_the_rest_live(variant, monkeypatch) -> None:
    """A clip that fails to replay is spoken live instead, in every mode."""
    store = MemoryStore()
    await prime(variant, lambda: MemcatConfig(store=store, admit_after_calls=1), [A.strip(), C])
    tts = make(variant, MemcatConfig(store=store))

    def boom(self, frame_ms=40):
        raise RuntimeError("bad clip")
        yield b""

    from memcat.core.lookup import Clip

    monkeypatch.setattr(Clip, "frames", boom)
    h = await hear(tts, turn(A + B + C))
    assert h.words == words_of(A + B + C)
    assert (await tts.memcat_stats())["enabled"] is False


async def test_provider_failure_is_not_spoken_twice() -> None:
    """A provider that fails after yielding audio is not called again with the same text."""

    class HalfFails(FakeHTTPTTS):
        async def run_tts(self, text, context_id):
            self.requests.append(text)
            pcm, _ = synth(text.split(",")[0], SR)
            yield TTSAudioRawFrame(pcm, SR, 1, context_id=context_id)
            raise RuntimeError("connection reset")

    plain = HalfFails()
    h0 = await hear(plain, turn(A + B + C))
    tts = cached(HalfFails)(memcat=MemcatConfig())
    h1 = await hear(tts, turn(A + B + C))
    assert h1.words == h0.words
    assert len(tts.requests) == len(plain.requests)


async def test_background_audio_service_is_never_replayed_around() -> None:
    """A non-WebSocket service delivering audio from a task keeps every live clause."""

    class Background(TTSService):
        def __init__(self, **kw):
            super().__init__(
                push_start_frame=True,
                push_stop_frames=True,
                sample_rate=SR,
                settings=_settings(),
                **kw,
            )

        async def run_tts(self, text, context_id):
            async def later():
                await asyncio.sleep(0.05)
                pcm, _ = synth(text, SR)
                for i in range(0, len(pcm), 1600):
                    await self.append_to_audio_context(
                        context_id,
                        TTSAudioRawFrame(pcm[i : i + 1600], SR, 1, context_id=context_id),
                    )

            self.create_task(later())
            yield None

    plain = Background()
    h0 = await hear(plain, turn(A + B + C) + [SleepFrame(1.0)])
    tts = cached(Background)(memcat=MemcatConfig(admit_after_calls=1))
    h1 = await hear(tts, turn(A + B + C) + [SleepFrame(1.0)])
    assert h1.words == h0.words == words_of(A + B + C)
    assert (await tts.memcat_stats())["enabled"] is False


async def test_render_twin_uses_current_settings() -> None:
    """Render twins use the service's current settings, not only its voice."""

    @dataclass
    class SpSettings(TTSSettings):
        speed: float | None = field(default_factory=lambda: NOT_GIVEN)

    class SpeedTTS(TTSService):
        def __init__(self, settings, **kw):
            super().__init__(
                push_start_frame=True,
                push_stop_frames=True,
                sample_rate=SR,
                settings=settings,
                **kw,
            )

        async def run_tts(self, text, context_id):
            t = text + (" FAST" if (self._settings.speed or 1) > 1 else " SLOW")
            pcm, _ = synth(t, SR)
            yield TTSAudioRawFrame(pcm, SR, 1, context_id=context_id)

    s = "Thanks for waiting today, and it should arrive soon."
    store = MemoryStore()

    def mk(speed: float):
        return cached(SpeedTTS)(
            memcat=MemcatConfig(store=store, admit_after_calls=1),
            settings=SpSettings(model="m", voice="alice", language="en", speed=speed),
        )

    for speed in (1.0, 1.5):
        t = mk(speed)
        await hear(t, [TTSSpeakFrame(s)])
        await get_runtime(t._memcat_cfg).worker.drain_all(10)
    h = await hear(mk(1.5), [TTSSpeakFrame(s)])
    assert "slow" not in h.words


async def test_each_service_keeps_its_own_admission_and_codec() -> None:
    """Services sharing a runtime each keep their own admission count and codec."""
    store = MemoryStore()
    first = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=store, admit_after_calls=1))
    await hear(first, [TTSSpeakFrame("warm the runtime up please")])
    rt = get_runtime(first._memcat_cfg)
    strict = Policy(admit_after_calls=5, lossless_only=True)
    sr = 24000
    pcm, _ = synth("hello there friend", sr)
    d = b"\x07" * 16
    rt.worker.observe("call-1", [ObserveItem(d, 5, captured=Audio(pcm, sr), policy=strict)])
    await rt.worker.drain_all(5)
    assert d not in rt.index  # 1 call of the 5 this service needs
    for i in range(2, 6):
        rt.worker.observe(f"call-{i}", [ObserveItem(d, 5, captured=Audio(pcm, sr), policy=strict)])
    await rt.worker.drain_all(5)
    (blob,) = await rt.store.get_many([d])
    assert blob is not None and codec.decode(blob).codec == codec.CODEC_PCM16


async def test_purge_reaches_other_processes() -> None:
    """Purge clears every process's L1 and index, not only the caller's."""
    store = MemoryStore()
    s = "How can I help you today?"
    a = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=store, admit_after_calls=1))
    await hear(a, [TTSSpeakFrame(s)])
    rt = get_runtime(a._memcat_cfg)
    await rt.worker.drain_all(5)
    digest = a._memcat_key().digest(s)
    assert rt.l1.get(digest) is not None
    # Another process: same store, its own runtime (simulated by a second index follower).
    from memcat.core.index import KeyIndex
    from memcat.core.lookup import L1Cache

    other_l1 = L1Cache(1 << 20)
    other_l1.put(digest, rt.l1.get(digest))  # type: ignore[arg-type]
    other = KeyIndex(
        store.bind("default"), on_purge=lambda ds: [other_l1.discard(x) for x in (ds or [digest])]
    )
    other.start()
    await asyncio.sleep(0.05)
    await a.memcat_purge(s)
    for _ in range(50):
        if other_l1.get(digest) is None:
            break
        await asyncio.sleep(0.01)
    assert other_l1.get(digest) is None and digest not in other
    await other.stop()


async def test_purge_beats_work_in_flight() -> None:
    """A capture queued before a purge does not store the purged clip again."""
    store = MemoryStore()
    a = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=store, admit_after_calls=1))
    await hear(a, [TTSSpeakFrame("warm the runtime up please")])
    rt = get_runtime(a._memcat_cfg)
    sr = 24000
    pcm, _ = synth("hello there friend", sr)
    d = b"\x09" * 16
    item = ObserveItem(d, 5, captured=Audio(pcm, sr))  # created now, before the purge
    await rt.purge([d])
    rt.worker.observe("call-1", [item])
    await rt.worker.drain_all(5)
    assert d not in rt.index and rt.l1.get(d) is None
    assert (await rt.store.get_many([d])) == [None]


def test_anchored_block_patterns_apply_to_each_clause() -> None:
    import re

    from memcat.core.chunker import split

    out = split(
        "Thanks for calling, your name is Jane Doe on file.",
        extra_blocks=[re.compile(r"^your name is\b", re.I)],
    )
    assert [(c.text, c.cacheable) for c in out][-1] == ("your name is Jane Doe on file.", False)


async def test_hits_refresh_with_the_services_own_ttl() -> None:
    store = MemoryStore()
    s = "How can I help you today?"
    first = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=store, admit_after_calls=1))
    await hear(first, [TTSSpeakFrame(s)])
    rt = get_runtime(first._memcat_cfg)
    await rt.worker.drain_all(5)
    touched: list[int] = []
    real = rt.store.touch

    async def spy(ds, ttl):
        touched.append(ttl)
        await real(ds, ttl)

    rt.store.touch = spy  # type: ignore[method-assign]
    rt.worker._touched_at.clear()
    short = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=store, ttl_s=120))
    await hear(short, [TTSSpeakFrame(s)])
    await rt.worker.drain_all(5)
    assert touched and touched[-1] == 120


async def test_ws_provider_error_reaches_the_pipeline() -> None:
    """In segmented mode a provider exception reaches the pipeline as an error."""
    from pipecat.frames.frames import ErrorFrame

    class Explodes(FakeWSTTS):
        async def run_tts(self, text, context_id):
            raise RuntimeError("provider exploded")
            yield None

    errors: list[str] = []
    tts = cached(Explodes)(memcat=MemcatConfig())
    orig = tts.push_error

    async def spy(*a, **k):
        errors.append(k.get("error_msg", ""))
        return await orig(*a, **k)

    tts.push_error = spy  # type: ignore[method-assign]
    await hear(tts, turn("Hello there, how are you today?"))
    assert errors and "exploded" not in errors[0]
    assert ErrorFrame
