"""HTTP mode: what is captured, what is resent after a failure, what is closed."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import numpy as np
import pytest
from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSSpeakFrame, TTSStartedFrame
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService

from memcat import MemcatConfig, MemoryStore, cached
from memcat.core import audio as audio_ops
from memcat.core import codec
from memcat.core.lookup import Clip
from memcat.core.runtime import get_runtime
from memcat.pipecat import mixin as mixin_mod
from tests.emulators.elevenlabs import ElevenLabsEmulator
from tests.fakes import SR, FakeHTTPTTS, synth, words_of
from tests.harness import hear, turn
from tests.util import drain, prime

A = "Thanks for waiting today, "
B = "your order 48213 has shipped, "
C = "and it should arrive soon."


async def _cached_clip(cfg: MemcatConfig, tts: Any, clause: str) -> Clip | None:
    rt = get_runtime(cfg)
    key = tts._memcat_key()
    [clip] = await rt.lookup.get_many([key.digest(clause)])
    return clip


async def test_wrong_rate_frame_stops_capture() -> None:
    """A live frame Memcat cannot capture (other sample rate) means the clip would
    have a hole: nothing is cached from that stream."""

    class _OneOddFrame(FakeHTTPTTS):
        async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
            pcm, _ = synth(text, SR)
            half = len(pcm) // 4 * 2
            yield TTSAudioRawFrame(pcm[:half], SR, 1, context_id=context_id)
            odd, _ = synth("hmm", 8000)
            yield TTSAudioRawFrame(odd, 8000, 1, context_id=context_id)
            yield TTSAudioRawFrame(pcm[half:], SR, 1, context_id=context_id)

    cfg = MemcatConfig(store=MemoryStore(), admit_after_calls=1, render=False)
    tts = cached(_OneOddFrame)(memcat=cfg)
    await hear(tts, [TTSSpeakFrame(C)])
    await drain(cfg)
    assert await _cached_clip(cfg, tts, C) is None


def test_rebase_words_never_mixes_clocks() -> None:
    """A provider clock 0.2 s behind Memcat's count of live audio: reading the first
    word as relative and the others as absolute would store garbage offsets."""
    base, lag = 1.5, 0.2
    raw = [(w, base - lag + off) for w, off in [("one", 0.05), ("two", 0.4), ("three", 0.7)]]
    assert mixin_mod._rebase_words(raw, base, 1.0, exact=False) is None
    # Relative per request, or cumulative from Memcat's count: both read whole.
    rel = [("one", 0.05), ("two", 0.4), ("three", 0.7)]
    assert mixin_mod._rebase_words(rel, base, 1.0, exact=False) == rel
    cum = [(w, base + t) for w, t in rel]
    got = mixin_mod._rebase_words(cum, base, 1.0, exact=False)
    assert got is not None and [round(t, 6) for _, t in got] == [0.05, 0.4, 0.7]


async def test_elevenlabs_http_clip_words_are_relative() -> None:
    """A live clause between cached ones is captured with its own word offsets, even
    when ElevenLabs' clock (summed alignment) differs from the audio already played."""
    x = "we packed it with great care, "
    y = "the driver is on the way now, "
    sentence = A + x + "and it should arrive soon, " + y + "is there anything else?"
    async with ElevenLabsEmulator() as emu, aiohttp.ClientSession() as session:
        store = MemoryStore()

        def make(cfg: MemcatConfig) -> Any:
            return cached(ElevenLabsHttpTTSService)(
                memcat=cfg,
                api_key="test",
                aiohttp_session=session,
                base_url=emu.http_url,
                sample_rate=SR,
                settings=ElevenLabsHttpTTSService.Settings(voice="v", model="eleven_flash_v2_5"),
            )

        def cfg() -> MemcatConfig:
            return MemcatConfig(store=store, admit_after_calls=1, render=False)

        c0 = cfg()
        await hear(
            make(c0),
            [
                TTSSpeakFrame(p)
                for p in [A.strip(), "and it should arrive soon,", "is there anything else?"]
            ],
        )
        await drain(c0)
        c1 = cfg()
        tts = make(c1)
        await hear(tts, turn(sentence))
        await drain(c1)
        for clause in (x, y):
            clip = await _cached_clip(c1, tts, clause)
            assert clip is not None, clause
            offs = [ms for _, ms in clip.words]
            assert [w for w, _ in clip.words] == clause.split()
            assert offs == sorted(offs) and 0 <= offs[0] < 200, offs
            assert offs[-1] < clip.duration_ms, (offs, clip.duration_ms)
        await mixin_mod.MemcatTTSMixin.memcat_stats(tts)  # still healthy


async def test_memcat_failure_mid_live_run_does_not_repeat_speech(monkeypatch) -> None:
    """Memcat failing while it trims a live run lets that run finish as is and says
    the rest live: nothing already sent to the provider is sent again."""
    store = MemoryStore()
    await prime("http", lambda: MemcatConfig(store=store, admit_after_calls=1), [A.strip()])
    tts = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=store))
    calls = [0]
    real = audio_ops.EdgeTrimmer.feed

    def flaky(self: Any, pcm: bytes) -> bytes:
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("trimmer bug")
        return real(self, pcm)

    monkeypatch.setattr(audio_ops.EdgeTrimmer, "feed", flaky)
    h = await hear(tts, turn(A + B + C))
    assert h.words == words_of(A + B + C)
    assert [r.strip() for r in tts.requests] == [(B + C).strip()]
    assert (await tts.memcat_stats())["enabled"] is False


class _Tracked(FakeHTTPTTS):
    closed: list[str]

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        pcm, _ = synth(text, SR)
        try:
            for i in range(0, len(pcm), 3200):
                yield TTSAudioRawFrame(pcm[i : i + 3200], SR, 1, context_id=context_id)
        finally:
            type(self).closed.append(text)


@pytest.mark.parametrize("with_runtime", [False, True])
async def test_provider_generator_is_closed_with_memcats(with_runtime: bool) -> None:
    """Closing Memcat's run_tts (an interruption) closes the provider's generator at
    once, releasing its HTTP response, instead of leaving it to the GC."""
    _Tracked.closed = []
    cfg = MemcatConfig(store=MemoryStore())
    tts: Any = cached(_Tracked)(memcat=cfg)
    tts._sample_rate = SR  # as after StartFrame
    if with_runtime:
        tts._memcat_rt = get_runtime(cfg)
    gen = tts.run_tts("Your order 48213 has shipped today.", "ctx")
    await gen.__anext__()
    await gen.aclose()
    assert _Tracked.closed == ["Your order 48213 has shipped today."]


async def test_live_loudness_sample_is_capped_and_rate_limited() -> None:
    sent: list[int] = []

    class _Worker:
        def live_loudness(self, _voice: str, pcm: bytes, _sr: int) -> None:
            sent.append(len(pcm))

    class _Rt:
        worker = _Worker()

    cfg = MemcatConfig(store=MemoryStore())
    tts: Any = cached(FakeHTTPTTS)(memcat=cfg)
    tts._sample_rate = SR  # as after StartFrame
    tts._memcat_rt = _Rt()
    key = tts._memcat_key()
    long = bytearray(SR * 2 * 20)  # 20 s
    tts._memcat_live_sample(key, long)
    tts._memcat_live_sample(key, long)
    assert sent == [int(mixin_mod._LOUDNESS_SAMPLE_S * 2 * SR)]


@pytest.mark.skipif(not codec.opus_available(), reason="needs PyAV")
async def test_opus_clip_decodes_off_the_loop_once(monkeypatch) -> None:
    pcm, _ = synth("thanks for waiting today", 16000)
    entry = codec.decode(
        codec.encode(
            codec.Entry(
                codec.CODEC_OPUS,
                16000,
                len(pcm) * 500 // 16000,
                None,
                (),
                codec._opus_encode(pcm, 16000),
            )
        )
    )
    clip = Clip.from_entry(entry)
    assert clip.pcm is None
    threads: list[int] = []
    real = codec.to_pcm16

    def spy(e: Any) -> bytes:
        threads.append(threading.get_ident())
        return real(e)

    monkeypatch.setattr(mixin_mod.audio_codec, "to_pcm16", spy)
    tts: Any = cached(FakeHTTPTTS)(memcat=MemcatConfig(store=MemoryStore(), gain_match=False))
    key = tts._memcat_key()
    a, b = await asyncio.gather(
        tts._memcat_clip_chunks(key, clip, b"d" * 16),
        tts._memcat_clip_chunks(key, clip, b"d" * 16),
    )
    assert a == b
    # Opus decoding is not bit-exact across decodes on every platform (x86 SIMD).
    got, want = np.frombuffer(b"".join(a), "<i2"), np.frombuffer(real(entry), "<i2")
    assert len(got) == len(want) and np.abs(got.astype(int) - want).max() <= 2
    assert len(threads) == 1 and threads[0] != threading.get_ident()


class _OpensOwnContext(FakeHTTPTTS):
    """Like NVIDIA's per-sentence mode: no ``push_start_frame``; run_tts opens the
    audio context and yields its TTSStartedFrame itself."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._push_start_frame = False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            yield TTSStartedFrame(context_id=context_id)
        async for f in super().run_tts(text, context_id):
            yield f


@pytest.mark.parametrize("text", [A.strip(), A + C])
async def test_sentence_opening_with_a_hit_opens_the_services_context(text: str) -> None:
    """A service that opens its own context gets it opened, with one TTSStartedFrame,
    when the sentence starts with a cached clause."""
    store = MemoryStore()

    def cfg() -> MemcatConfig:
        return MemcatConfig(store=store, admit_after_calls=1, render=False)

    await hear(cached(_OpensOwnContext)(memcat=cfg()), [TTSSpeakFrame(A.strip())])
    await drain(cfg())
    tts = cached(_OpensOwnContext)(memcat=cfg())
    h = await hear(tts, [TTSSpeakFrame(text)])
    assert (await tts.memcat_stats())["hits"] == 1
    assert h.words == words_of(text)
    assert h.started == 1 and h.stopped == 1
