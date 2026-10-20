"""Deepgram: audio still in flight when a Clear lands belongs to no stream."""

from __future__ import annotations

from pipecat.frames.frames import InterruptionFrame
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.tests.utils import SleepFrame

from memcat import MemcatConfig, MemoryStore, cached
from tests.emulators.deepgram import DeepgramEmulator
from tests.fakes import words_of
from tests.harness import turn
from tests.test_provider_deepgram import assert_closed, listen, ws

STALE = "zebra"


async def test_audio_before_cleared_is_not_the_next_turns() -> None:
    """After a barge-in, audio Deepgram sent before "Cleared" is dropped: it never
    plays in, nor is cached as part of, the next turn's stream."""
    emu = await DeepgramEmulator(clear_lag_s=0.3, stale_text=STALE).start()
    try:
        tts = ws(emu, MemcatConfig(store=MemoryStore(), admit_after_calls=1, render=False))
        first = "Thanks for your patience today, your order 5522 is running late. " * 2
        after = "Sure, is there anything else I can help with?"
        frames = [
            *turn(first),
            SleepFrame(0.05),
            InterruptionFrame(),
            SleepFrame(0.02),
            *turn(after),
            SleepFrame(0.5),
        ]
        h = await listen(tts, frames)
        assert STALE not in h.words
        want = words_of(after)
        assert h.words[-len(want) :] == want
        assert_closed(tts)
    finally:
        await emu.stop()
    assert not emu.errors, emu.errors


def test_disabled_service_reads_the_socket_unwrapped() -> None:
    """Disabled, the service reads its socket exactly as unwrapped Pipecat does."""
    sock = object()
    args = {"api_key": "test", "sample_rate": 16000}
    off = cached(DeepgramTTSService)(
        memcat=MemcatConfig(store=MemoryStore(), enabled=False), **args
    )
    off._websocket = sock
    assert off._get_websocket() is sock
    on = cached(DeepgramTTSService)(memcat=MemcatConfig(store=MemoryStore()), **args)
    on._websocket = sock
    assert on._get_websocket() is not sock
