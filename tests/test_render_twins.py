"""Render twins and the settings they render with."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import pytest
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService

from memcat import MemcatConfig, MemoryStore, cached
from memcat.core.runtime import get_runtime
from memcat.pipecat import render
from memcat.pipecat.render import RenderPool
from tests.emulators.elevenlabs import ElevenLabsEmulator
from tests.fakes import SR, FakeHTTPTTS, FakeWSTTS, _settings, decode, synth, words_of
from tests.harness import hear

TEXT = "Thanks for waiting today."


class _ErrorsHalfway(FakeHTTPTTS):
    """Yields the first word's audio, then reports an error."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        pcm, _ = synth(text.split()[0], self.sample_rate)
        yield TTSAudioRawFrame(pcm, self.sample_rate, 1, context_id=context_id)
        yield ErrorFrame(error="stream broke")


class _RaisesHalfway(FakeHTTPTTS):
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        pcm, _ = synth(text.split()[0], self.sample_rate)
        yield TTSAudioRawFrame(pcm, self.sample_rate, 1, context_id=context_id)
        raise RuntimeError("connection reset")


class _NeverDone(FakeWSTTS):
    """Sends part of the audio, then nothing: no "done", the context times out."""

    async def _fake_handle(self, sock: Any, msg: tuple[Any, ...]) -> None:
        if msg[0] == "text":
            await self._emit(msg[1], msg[2].split()[0])
        # the flush is never answered


async def _render(base: type, text: str = TEXT, **kwargs: Any) -> Any:
    pool = RenderPool(base, (), kwargs)
    try:
        fn = pool.render_fn("voice", text, _settings(), SR, lambda _n: None)
        return await fn()
    finally:
        await pool.close()


async def test_good_render_returns_audio() -> None:
    audio = await _render(FakeHTTPTTS)
    assert audio is not None and decode(audio.pcm) == words_of(TEXT)


@pytest.mark.parametrize("base", [_ErrorsHalfway, _RaisesHalfway])
async def test_failed_render_keeps_nothing(base: type) -> None:
    assert await _render(base) is None


async def test_render_cut_short_by_timeout_keeps_nothing() -> None:
    assert await _render(_NeverDone, stop_frame_timeout_s=0.3) is None


async def test_render_uses_settings_from_when_the_sentence_was_planned() -> None:
    """A voice change after a sentence is planned does not change what its clauses
    are rendered with: the digests were keyed under the old voice."""
    seen: list[str] = []

    class _Pool:
        def render_fn(self, _ident: str, _text: str, snap: Any, _sr: int, _cb: Any) -> Any:
            seen.append(snap.voice)
            return None

    cfg = MemcatConfig(store=MemoryStore(), render=True)
    tts: Any = cached(FakeHTTPTTS)(memcat=cfg)
    tts._memcat_rt = get_runtime(cfg)
    tts._memcat_pool = _Pool()
    plan = await tts._memcat_plan("Thanks for waiting today, is there anything else?")
    tts._settings.voice = "bob"  # TTSUpdateSettingsFrame, before the stream ends
    await tts._memcat_plan("Something else entirely, said in the new voice.")
    tts._memcat_observe_item(plan, 0)
    assert seen == ["alice"]


async def test_elevenlabs_twin_sends_the_snapshot_voice_settings() -> None:
    """ElevenLabs derives its voice_settings payload in __init__; a twin given other
    settings must send those, not the constructor's."""
    async with ElevenLabsEmulator() as emu, aiohttp.ClientSession() as session:
        kwargs = {
            "api_key": "test",
            "aiohttp_session": session,
            "base_url": emu.http_url,
            "sample_rate": SR,
            "settings": ElevenLabsHttpTTSService.Settings(
                voice="v", model="eleven_flash_v2_5", stability=0.5, similarity_boost=0.75
            ),
        }
        live: Any = cached(ElevenLabsHttpTTSService)(
            memcat=MemcatConfig(store=MemoryStore()), **kwargs
        )
        live._settings.stability = 0.9  # as after a TTSUpdateSettingsFrame
        snap = live._settings
        pool = RenderPool(ElevenLabsHttpTTSService, (), kwargs)
        try:
            audio = await pool.render_fn("v", TEXT, snap, SR, lambda _n: None)()
        finally:
            await pool.close()
        assert audio is not None
        assert emu.http_requests[-1]["json"]["voice_settings"]["stability"] == 0.9


class _Locator:
    """A constructor argument with no value equality (compared by identity)."""

    def __init__(self, name: str):
        self.name = name


class _Recording(FakeHTTPTTS):
    """Records the text each instance (live service or twin) is asked to speak."""

    log: list[tuple[str, str, Any]] = []

    def __init__(self, *, locator: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.locator = locator

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self.log.append((type(self).__name__, text, self.locator))
        async for f in super().run_tts(text, context_id):
            yield f


async def test_twin_renders_with_the_requesting_services_own_arguments() -> None:
    """Services built with different constructor arguments never share a pool, so a
    clause is rendered with exactly the arguments of the service that admitted it."""
    _Recording.log = []
    loc_a, loc_b = _Locator("a"), _Locator("b")
    pool_a = RenderPool.shared(_Recording, (), {"locator": loc_a})
    pool_b = RenderPool.shared(_Recording, (), {"locator": loc_b})
    assert pool_a is not pool_b
    assert RenderPool.shared(_Recording, (), {"locator": loc_a}) is pool_a
    assert await pool_a.render_fn("voice", TEXT, _settings(), SR, lambda _n: None)() is not None
    assert [loc for who, _, loc in _Recording.log if who.startswith("MemcatTwin")] == [loc_a]


async def test_pools_are_shared_by_equal_arguments() -> None:
    async with aiohttp.ClientSession() as s1, aiohttp.ClientSession() as s2:
        a = RenderPool.shared(FakeHTTPTTS, (), {"aiohttp_session": s1, "settings": _settings()})
        b = RenderPool.shared(FakeHTTPTTS, (), {"aiohttp_session": s2, "settings": _settings()})
        assert a is b  # a twin opens its own session, so the call's own does not matter


async def test_pool_is_dropped_once_unused() -> None:
    """The registry holds a pool only while a service uses it or its twins run."""
    pool = RenderPool.shared(FakeHTTPTTS, (), {"sample_rate": SR})
    pools = render._pools[asyncio.get_running_loop()]
    assert list(pools) == [pool]
    del pool
    gc.collect()
    assert not list(pools)


async def test_twin_does_not_reapply_text_transforms() -> None:
    """A clause reaches the twin already transformed by the live service."""
    _Recording.log = []

    async def tag(text: str, _type: Any) -> str:
        return text.replace("Metformin", "<phoneme>Metformin</phoneme>")

    cfg = MemcatConfig(store=MemoryStore(), admit_after_calls=1, render=True)
    tts = cached(_Recording)(memcat=cfg, text_transforms=[("*", tag)])
    await hear(tts, [TTSSpeakFrame("Please take your Metformin daily, and drink plenty of water.")])
    await get_runtime(cfg).worker.drain_all(20)
    rendered = [t for who, t, _ in _Recording.log if who.startswith("MemcatTwin")]
    assert sorted(rendered) == [
        "Please take your <phoneme>Metformin</phoneme> daily,",
        "and drink plenty of water.",
    ]
