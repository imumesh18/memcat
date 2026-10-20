"""Segmented streams: slow or lost provider streams, cancellation and interruptions."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pipecat.frames.frames import Frame, InterruptionFrame, TTSSpeakFrame, TTSUpdateSettingsFrame
from pipecat.services.settings import TTSSettings
from pipecat.tests.utils import SleepFrame

from memcat import MemcatConfig, MemoryStore, cached
from memcat.pipecat.contexts import CtxState, Segment, Stream
from tests.fakes import FakeELWSTTS, FakeWSTTS, words_of
from tests.harness import hear, turn
from tests.util import drain, make, prime

A = "Thanks for waiting today, "
B = "your order 48213 has shipped, "
C = "and it should arrive soon."
SWALLOW = "48213"


class _DropsDone(FakeWSTTS):
    """Cartesia-like, but the provider never confirms the stream carrying SWALLOW."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.texts: dict[str, str] = {}

    async def _fake_handle(self, sock: Any, msg: tuple[Any, ...]) -> None:
        if msg[0] == "text":
            self.texts[msg[1]] = self.texts.get(msg[1], "") + msg[2]
        if msg[0] == "flush" and SWALLOW in self.texts.get(msg[1], ""):
            return  # the "done" is lost
        await super()._fake_handle(sock, msg)


class _Mute(FakeELWSTTS):
    """ElevenLabs-like (one stream at a time), but never answers text with SWALLOW."""

    async def _fake_handle(self, sock: Any, msg: tuple[Any, ...]) -> None:
        if msg[0] == "text" and SWALLOW in msg[2]:
            return
        await super()._fake_handle(sock, msg)


@pytest.mark.parametrize("timeout_s", [0.5, 3.0])
async def test_cached_clause_after_unconfirmed_stream_is_heard(timeout_s: float) -> None:
    """A cached clause queued behind a live stream whose provider never confirms it
    still plays: Memcat keeps the context alive and closes the stream itself, after
    Pipecat's own silence timeout (whatever it is set to)."""
    store = MemoryStore()

    def cfg() -> MemcatConfig:
        return MemcatConfig(store=store, admit_after_calls=1, render=False)

    tts0 = cached(_DropsDone)(memcat=cfg())
    await hear(tts0, [TTSSpeakFrame(A.strip()), TTSSpeakFrame(C)])
    await drain(cfg())
    tts = cached(_DropsDone)(memcat=cfg(), stop_frame_timeout_s=timeout_s)
    t0 = time.monotonic()
    h = await hear(tts, turn(A + B + C))
    assert h.words == words_of(A + B + C)
    assert time.monotonic() - t0 < timeout_s + 3.0
    assert not tts._memcat_inflight and not tts._memcat_streams


async def test_mute_stream_frees_its_slot() -> None:
    """A stream that never produces audio is closed after the silence timeout and
    frees the single in-flight slot, so the next sentence is still spoken live."""
    tts = cached(_Mute)(
        memcat=MemcatConfig(store=MemoryStore(), render=False), stop_frame_timeout_s=0.5
    )
    after = "Shall I send the details by text message?"
    t0 = time.monotonic()
    h = await asyncio.wait_for(hear(tts, [*turn(B), TTSSpeakFrame(after)]), 20)
    assert h.words == words_of(after)
    assert time.monotonic() - t0 < 5.0
    assert not tts._memcat_inflight and not tts._memcat_waiting


async def test_cancel_during_drop_does_not_leak_inflight_slots() -> None:
    """A cancellation while the provider is told to drop streams still frees every
    stream's in-flight slot (with one slot, a leak would mute the rest of the call)."""
    tts: Any = cached(FakeELWSTTS)(memcat=MemcatConfig(store=MemoryStore()))
    st = CtxState("ctx")
    streams = []
    for n in range(3):
        s = Stream(sid=f"ctx-s{n}", cid="ctx", seg=Segment(), state="sent")
        st.streams.append(s)
        tts._memcat_streams[s.sid] = s
        tts._memcat_inflight.append(s)
        streams.append(s)
    entered = asyncio.Event()

    async def slow_cancel(_svc: Any, _sid: str) -> None:
        entered.set()
        await asyncio.sleep(10)

    tts._memcat_adapter.cancel_stream = slow_cancel
    task = asyncio.ensure_future(tts._memcat_drop_ctx(st, pump=False))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tts._memcat_inflight == []
    assert tts._memcat_streams == {}
    assert all(s.state == "done" for s in streams)


async def test_interruption_sends_no_queued_text() -> None:
    """Interrupting a turn whose later sentences wait for the single stream slot never
    sends their text to the provider afterwards (it would be billed for nothing)."""
    store = MemoryStore()
    s1 = "Your order 1234 is ready now. "
    s2 = "Your parcel 5678 left the depot today. "
    s3 = "Your refund 9012 was approved this morning."

    class _SlowClose(FakeELWSTTS):
        # Interrupting awaits real work (socket sends, metrics frames), so other tasks
        # run between one context's interruption and the next.
        async def on_audio_context_interrupted(self, context_id: str) -> None:
            await asyncio.sleep(0.01)
            await super().on_audio_context_interrupted(context_id)

        async def stop_all_metrics(self) -> None:
            await asyncio.sleep(0.01)
            await super().stop_all_metrics()

    tts = cached(_SlowClose)(
        memcat=MemcatConfig(store=store, render=False),
        delay=(0.2, 0.2),
        reuse_context_id_within_turn=False,
    )
    frames = [*turn(s1 + s2 + s3), SleepFrame(0.05), InterruptionFrame(), SleepFrame(0.6)]
    await hear(tts, frames)
    sent = " ".join(t for _, t in tts.requests)
    assert "9012" not in sent and "5678" not in sent


@pytest.mark.parametrize("variant", ["ws-ctx", "ws-el"])
async def test_late_done_is_quiet(variant: str) -> None:
    """A provider reply arriving for a stream Memcat already closed logs no warning."""
    from loguru import logger

    store = MemoryStore()
    await prime(variant, lambda: MemcatConfig(store=store, admit_after_calls=1), [A.strip()])
    tts = make(variant, MemcatConfig(store=store))
    warnings: list[str] = []
    sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        await hear(tts, turn(A + B + C))
        closed = list(tts._memcat_closed)
        assert closed
        for sid in closed:
            await tts.remove_audio_context(sid)
    finally:
        logger.remove(sink)
    assert not [w for w in warnings if "unable to remove" in w]


async def test_cancel_returns_promptly() -> None:
    """cancel() does not wait for the runtime to flush its queue."""
    tts: Any = cached(FakeWSTTS)(memcat=MemcatConfig(store=MemoryStore()))
    gate = asyncio.Event()

    class _Rt:
        async def release(self) -> None:
            await gate.wait()

    tts._memcat_rt = _Rt()
    t0 = time.monotonic()
    await tts._memcat_release(wait=False)
    assert time.monotonic() - t0 < 0.5
    gate.set()
    await asyncio.sleep(0)


async def test_stream_queued_across_a_voice_change_is_not_cached_under_the_old_voice() -> None:
    """A live run still queued for the single stream slot when the voice changes goes
    out in the new voice, so its audio must not be kept under the voice it was
    planned with: a later call in that voice would replay the wrong one."""
    store = MemoryStore()
    x, y = "Let me just pull that up, ", "and here is what I found."

    class _Voices(FakeELWSTTS):
        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.sent: list[tuple[str, str]] = []

        async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
            self.sent.append((text, self._settings.voice))
            async for f in super().run_tts(text, context_id):
                yield f

    def cfg() -> MemcatConfig:
        return MemcatConfig(store=store, admit_after_calls=1, render=False)

    await hear(cached(_Voices)(memcat=cfg()), [TTSSpeakFrame(A.strip())])
    await drain(cfg())
    tts = cached(_Voices)(memcat=cfg())
    await hear(tts, [*turn(x + A + y), TTSUpdateSettingsFrame(delta=TTSSettings(voice="bob"))])
    assert tts.sent[-1] == (y, "bob")
    await drain(cfg())
    later = cached(_Voices)(memcat=cfg())
    await hear(later, [TTSSpeakFrame(y)])
    assert later.sent == [(y, "alice")]
