"""``memcat serve`` keeps requests apart and fails fast.

A speaker serves one request at a time. Whatever ends a request early (a timeout, a
client that hangs up, a provider error) must not leave audio behind for the next one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from pipecat.frames.frames import Frame

from memcat import MemcatConfig, MemoryStore, cached
from memcat import serve as serve_mod
from memcat.serve import Server, Speaker
from tests.fakes import FakeHTTPTTS, FakeWSTTS, decode, words_of

LONG = " ".join(f"word{i}" for i in range(30)) + "."
SHORT = "Fine thanks."


class SlowTTS(FakeHTTPTTS):
    """Streams 100 ms chunks every 50 ms, so an utterance is still going when abandoned."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        async for f in super().run_tts(text, context_id):
            yield f
            await asyncio.sleep(0.05)


def _make(cls: type = SlowTTS, **kw: Any):
    store = MemoryStore()

    def make(rate: int = 16000) -> Any:
        return cached(cls)(memcat=MemcatConfig(store=store), sample_rate=rate, **kw)

    return make


async def _collect(sp: Speaker, text: str, call: str) -> bytes:
    got: list[bytes] = []

    async def write(chunk: bytes) -> None:
        got.append(chunk)

    await sp.speak(text, call, write)
    return b"".join(got)


@pytest.mark.parametrize("cls", [SlowTTS, FakeWSTTS])
async def test_client_disconnect_leaves_nothing_for_the_next_request(cls: type) -> None:
    sp = Speaker(_make(cls), 16000)
    await sp.start()
    try:
        writes = 0

        async def hang_up(_chunk: bytes) -> None:
            nonlocal writes
            writes += 1
            if writes == 2:
                raise ConnectionResetError("client went away")

        with pytest.raises(ConnectionResetError):
            await sp.speak(LONG, "call-a", hang_up)
        pcm = await _collect(sp, SHORT, "call-b")
        assert decode(pcm) == words_of(SHORT)
        pcm = await _collect(sp, SHORT, "call-c")
        assert decode(pcm) == words_of(SHORT)
    finally:
        await sp.stop()


async def test_timeout_leaves_nothing_for_the_next_request(monkeypatch) -> None:
    monkeypatch.setattr(serve_mod, "REQUEST_TIMEOUT_S", 0.5)
    sp = Speaker(_make(), 16000)
    await sp.start()
    try:
        with pytest.raises(TimeoutError):
            await _collect(sp, LONG, "call-a")
        pcm = await _collect(sp, SHORT, "call-b")
        assert decode(pcm) == words_of(SHORT)
    finally:
        await sp.stop()


async def test_timeout_mid_stream_then_next_request_over_http(monkeypatch) -> None:
    monkeypatch.setattr(serve_mod, "REQUEST_TIMEOUT_S", 0.5)
    server = Server(_make(), pool_size=1)
    async with TestClient(TestServer(server.app())) as client:
        r = await client.post("/v1/tts", json={"text": LONG, "sample_rate": 16000})
        assert r.status == 200
        cut = decode(await r.read())
        assert 0 < len(cut) < len(words_of(LONG))
        r = await client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000})
        assert r.status == 200 and decode(await r.read()) == words_of(SHORT)


async def test_timeout_before_audio_is_504(monkeypatch) -> None:
    monkeypatch.setattr(serve_mod, "REQUEST_TIMEOUT_S", 0.3)

    class Stalled(FakeHTTPTTS):
        async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
            await asyncio.sleep(5)
            yield None

    server = Server(_make(Stalled), pool_size=1)
    async with TestClient(TestServer(server.app())) as client:
        r = await client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000})
        assert r.status == 504 and (await r.json())["error"] == "timeout"


async def test_provider_error_is_a_fast_502_and_the_speaker_recovers() -> None:
    server = Server(_make(FakeHTTPTTS, fail_on="boom"), pool_size=1)
    async with TestClient(TestServer(server.app())) as client:
        t0 = time.monotonic()
        r = await client.post("/v1/tts", json={"text": "This will boom now.", "sample_rate": 16000})
        assert time.monotonic() - t0 < 5
        assert r.status == 502
        body = await r.json()
        assert body["error"] == "provider error" and "provider exploded" in body["detail"]
        r = await client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000})
        assert r.status == 200 and decode(await r.read()) == words_of(SHORT)


def test_busy_pool_spreads_waiters() -> None:
    server = Server(_make(), pool_size=2)
    taken = [server._acquire(16000) for _ in range(5)]
    pool = server._pools[16000]
    assert len(pool) == 2 and sorted(sp.load for sp in pool) == [2, 3]
    lighter = min(pool, key=lambda sp: sp.load)
    assert server._acquire(16000) is lighter
    for sp in taken:
        server._release(sp)
    server._release(lighter)
    assert [sp.load for sp in pool] == [0, 0]
    assert server._acquire(16000) is pool[0]  # idle speakers are reused before adding more
    assert len(pool) == 2


async def test_starting_a_speaker_does_not_block_other_rates(monkeypatch) -> None:
    gate = asyncio.Event()
    real_start = Speaker._start

    async def start(self: Speaker) -> None:
        if self.sample_rate == 24000:
            await gate.wait()
        await real_start(self)

    monkeypatch.setattr(Speaker, "_start", start)
    server = Server(_make(FakeHTTPTTS), pool_size=2)
    async with TestClient(TestServer(server.app())) as client:
        slow = asyncio.ensure_future(
            client.post("/v1/tts", json={"text": SHORT, "sample_rate": 24000})
        )
        await asyncio.sleep(0.05)
        r = await asyncio.wait_for(
            client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000}), 5
        )
        assert r.status == 200 and decode(await r.read()) == words_of(SHORT)
        assert not slow.done()
        gate.set()
        r = await slow
        assert r.status == 200 and decode(await r.read(), 24000) == words_of(SHORT)


async def test_speaker_that_fails_to_start_is_dropped(monkeypatch) -> None:
    calls = 0

    def make(rate: int) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("no credentials")
        return _make(FakeHTTPTTS)(rate)

    server = Server(make, pool_size=1)
    async with TestClient(TestServer(server.app())) as client:
        r = await client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000})
        assert r.status == 503 and (await r.json())["error"] == "speaker unavailable"
        assert r.headers["Retry-After"] == "5"
        r = await client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000})
        assert r.status == 200 and decode(await r.read()) == words_of(SHORT)


async def test_late_error_from_earlier_work_does_not_fail_a_healthy_request() -> None:
    from pipecat.frames.frames import ErrorFrame

    sp = Speaker(_make(), 16000)
    await sp.start()
    try:
        req = serve_mod._Request(999)
        sp._tap.request = req
        # An error arrives before this request's marker: it may be stale, so it is held.
        await sp._worker._call_event_handler("on_pipeline_error", ErrorFrame("stale failure"))
        await asyncio.sleep(0.05)  # handlers run as tasks
        assert req.queue.empty() and req.early_error is not None
        sp._tap.request = None
        # A real request on the same speaker still speaks normally.
        assert decode(await _collect(sp, SHORT, "c1")) == words_of(SHORT)
    finally:
        await sp.stop()


async def test_speaker_that_cannot_be_cleared_is_retired(monkeypatch) -> None:
    monkeypatch.setattr(serve_mod, "RESET_TIMEOUT_S", 0.05)
    sp = Speaker(_make(), 16000)
    await sp.start()
    real = sp._worker.queue_frame

    async def no_interrupt(frame: Any, *a: Any, **k: Any) -> None:
        if type(frame).__name__ == "InterruptionWorkerFrame":
            return  # the interruption never reaches the pipeline
        await real(frame, *a, **k)

    monkeypatch.setattr(sp._worker, "queue_frame", no_interrupt)

    async def hang_up(_chunk: bytes) -> None:
        raise ConnectionResetError

    with pytest.raises(ConnectionResetError):
        await sp.speak(LONG, "c1", hang_up)
    assert sp.dead  # never handed another request
    await sp.stop()
