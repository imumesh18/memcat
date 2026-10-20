"""Fake TTS services that make playback order audible.

Every word is a 60 ms tone whose frequency is unique to that word, followed by
20 ms of silence. Decoding the pipeline's output audio back into words gives the
exact order a listener would hear, whatever path each clause took.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import re
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService, WebsocketTTSService

from memcat.pipecat.adapters.providers import CartesiaAdapter, DeepgramAdapter, ElevenLabsAdapter

SR = 16000
TONE_MS = 60
GAP_MS = 20
EDGE_MS = 40
_FREQS: dict[str, float] = {}
_BASE_HZ = 300.0
_STEP_HZ = 31.25  # FFT bin width for a 32 ms window at 16 kHz, so peaks never collide


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def freq_for(word: str) -> float:
    w = _norm(word)
    f = _FREQS.get(w)
    if f is None:
        f = _BASE_HZ + _STEP_HZ * (len(_FREQS) + 1) * 2
        _FREQS[w] = f
    return f


def synth(text: str, sr: int = SR) -> tuple[bytes, list[tuple[str, float]]]:
    """Tone-per-word audio plus (word, start seconds) timestamps."""
    out: list[np.ndarray] = [np.zeros(sr * EDGE_MS // 1000)]
    words: list[tuple[str, float]] = []
    t = EDGE_MS / 1000
    n_tone = sr * TONE_MS // 1000
    n_gap = sr * GAP_MS // 1000
    for raw in text.split():
        if not _norm(raw):
            continue
        f = freq_for(raw)
        x = np.sin(2 * np.pi * f * np.arange(n_tone) / sr) * 9000
        words.append((raw, t))
        out.append(x)
        out.append(np.zeros(n_gap))
        t += (n_tone + n_gap) / sr
    out.append(np.zeros(sr * EDGE_MS // 1000))
    return np.concatenate(out).astype("<i2").tobytes(), words


def decode(pcm: bytes, sr: int = SR) -> list[str]:
    """Recover the spoken word sequence from tone audio."""
    return [w for w, _ in decode_timed(pcm, sr)]


def decode_timed(pcm: bytes, sr: int = SR) -> list[tuple[str, float]]:
    """Recover (word, onset seconds) from tone audio."""
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    win = sr // 200  # 5 ms
    n = len(x) // win
    if n == 0:
        return []
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
    loud = rms > 1500
    inv = {v: k for k, v in _FREQS.items()}
    freqs = np.array(sorted(inv))
    words: list[tuple[str, float]] = []
    i = 0
    while i < n:
        if not loud[i]:
            i += 1
            continue
        j = i
        while j < n and loud[j]:
            j += 1
        if (j - i) * 5 >= 25:
            seg = x[i * win : j * win]
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=sr))
            peak = float(np.argmax(spec))  # 1 Hz bins with n=sr
            k = int(np.argmin(np.abs(freqs - peak)))
            words.append((inv[float(freqs[k])], i * win / sr))
        i = j
    return words


def words_of(text: str) -> list[str]:
    return [_norm(w) for w in text.split() if _norm(w)]


def _settings() -> TTSSettings:
    return TTSSettings(model="fake-1", voice="alice", language="en")


class FakeHTTPTTS(TTSService):
    """Yields audio inline. With ``timestamps=True`` it reports word times the way
    ElevenLabs HTTP does: cumulative over the provider's own audio in a context."""

    def __init__(self, *, timestamps: bool = False, fail_on: str | None = None, **kwargs: Any):
        kwargs.setdefault("sample_rate", SR)
        kwargs.setdefault("settings", _settings())
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=not timestamps,
            **kwargs,
        )
        self.timestamps = timestamps
        self.fail_on = fail_on
        self.requests: list[str] = []
        self._cum = 0.0

    def can_generate_metrics(self) -> bool:
        return True

    async def push_frame(self, frame: Frame, direction: Any = None) -> None:  # type: ignore[override]
        if isinstance(frame, TTSStoppedFrame):
            self._cum = 0.0
        if direction is None:
            await super().push_frame(frame)
        else:
            await super().push_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self.requests.append(text)
        await self.start_tts_usage_metrics(text)
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("provider exploded")
        pcm, words = synth(text, self.sample_rate)
        await asyncio.sleep(0.005)
        if self.timestamps and words:
            await self.add_word_timestamps([(w, self._cum + t) for w, t in words], context_id)
        step = self.sample_rate // 10 * 2
        for i in range(0, len(pcm), step):
            yield TTSAudioRawFrame(pcm[i : i + step], self.sample_rate, 1, context_id=context_id)
        self._cum += len(pcm) / (2 * self.sample_rate)


class _FakeSocket:
    """A provider on the other end of a WebSocket: handles messages in order."""

    def __init__(self, owner: Any, delay: tuple[float, float]):
        self.owner = owner
        self.q: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue()
        self.delay = delay
        self.cancelled: set[str] = set()
        self.clock: dict[str, float] = {}
        self.log: list[tuple[Any, ...]] = []
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.task = asyncio.ensure_future(self.run())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with contextlib.suppress(BaseException):
                await self.task
            self.task = None

    async def run(self) -> None:
        while True:
            msg = await self.q.get()
            self.log.append(msg)
            try:
                await self.owner._fake_handle(self, msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # pragma: no cover - test diagnostics
                print("fake provider error", type(e).__name__, e)

    async def latency(self) -> None:
        await asyncio.sleep(random.uniform(*self.delay))


class _FakeWSBase(WebsocketTTSService):
    def __init__(self, *, delay: tuple[float, float] = (0.005, 0.03), **kwargs: Any):
        kwargs.setdefault("sample_rate", SR)
        kwargs.setdefault("settings", _settings())
        super().__init__(**kwargs)
        self.delay = delay
        self.sock: _FakeSocket | None = None
        self.requests: list[tuple[str, str]] = []

    def can_generate_metrics(self) -> bool:
        return True

    async def _connect(self) -> None:
        if self.sock is None:
            self.sock = _FakeSocket(self, self.delay)
            self.sock.start()

    async def _disconnect(self) -> None:
        if self.sock is not None:
            await self.sock.stop()
            self.sock = None

    async def _connect_websocket(self) -> None:  # pragma: no cover - unused
        return None

    async def _disconnect_websocket(self) -> None:  # pragma: no cover - unused
        return None

    async def _receive_messages(self) -> None:  # pragma: no cover - unused
        return None

    async def start(self, frame: Any) -> None:
        await super().start(frame)
        await self._connect()

    async def _emit(self, ctx: str, text: str) -> None:
        pcm, words = synth(text, self.sample_rate)
        base = self.sock.clock.get(ctx, 0.0) if self.sock else 0.0
        step = self.sample_rate // 20 * 2
        if not self.push_text:
            await self.add_word_timestamps([(w, base + t) for w, t in words], ctx)
        for i in range(0, len(pcm), step):
            await self.append_to_audio_context(
                ctx, TTSAudioRawFrame(pcm[i : i + step], self.sample_rate, 1, context_id=ctx)
            )
            await asyncio.sleep(0)
        if self.sock:
            self.sock.clock[ctx] = base + len(pcm) / (2 * self.sample_rate)

    push_text = False


class FakeWSTTS(_FakeWSBase):
    """Cartesia-like: every message carries a context id; flush ends a context."""

    __memcat_adapter__ = CartesiaAdapter

    def __init__(self, **kwargs: Any):
        super().__init__(push_start_frame=True, push_text_frames=False, **kwargs)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self.requests.append((context_id, text))
        await self.start_tts_usage_metrics(text)
        assert self.sock is not None
        await self.sock.q.put(("text", context_id, text))
        yield None

    async def flush_audio(self, context_id: str | None = None) -> None:
        if self.sock is not None and context_id:
            await self.sock.q.put(("flush", context_id))

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        if self.sock is not None:
            self.sock.cancelled.add(context_id)
            await self.sock.q.put(("cancel", context_id))
        await super().on_audio_context_interrupted(context_id)

    async def _fake_handle(self, sock: _FakeSocket, msg: tuple[Any, ...]) -> None:
        kind, ctx = msg[0], msg[1]
        if ctx in sock.cancelled:
            return
        if kind == "text":
            await sock.latency()
            if ctx in sock.cancelled or not self.audio_context_available(ctx):
                return
            await self._emit(ctx, msg[2])
        elif kind == "flush":
            await sock.latency()
            if self.audio_context_available(ctx):
                await self.append_to_audio_context(ctx, TTSStoppedFrame(context_id=ctx))
                await self.remove_audio_context(ctx)


class FakeCursorWSTTS(_FakeWSBase):
    """Deepgram-like: no context ids on the wire; ``Flush`` -> ``Flushed``."""

    __memcat_adapter__ = DeepgramAdapter
    push_text = True

    def __init__(self, **kwargs: Any):
        super().__init__(push_start_frame=True, **kwargs)
        self.flushes = 0

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self.requests.append((context_id, text))
        await self.start_tts_usage_metrics(text)
        assert self.sock is not None
        await self.sock.q.put(("speak", "", text))
        yield None

    async def flush_audio(self, context_id: str | None = None) -> None:
        self.flushes += 1
        if self.sock is not None:
            await self.sock.q.put(("flush", ""))

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        if self.sock is not None:
            # "Clear": drop everything queued on the socket.
            while not self.sock.q.empty():
                self.sock.q.get_nowait()
        await super().on_audio_context_interrupted(context_id)

    async def _fake_handle(self, sock: _FakeSocket, msg: tuple[Any, ...]) -> None:
        kind = msg[0]
        await sock.latency()
        ctx = self.get_active_audio_context_id()
        if ctx is None:
            return
        if kind == "speak":
            await self._emit(ctx, msg[2])
        elif kind == "flush":
            ctx = self.get_active_audio_context_id()
            if ctx and self.audio_context_available(ctx):
                await self.append_to_audio_context(ctx, TTSStoppedFrame(context_id=ctx))
                await self.remove_audio_context(ctx)


class FakeELWSTTS(_FakeWSBase):
    """ElevenLabs-like: run_tts opens its own context; close_context ends it."""

    __memcat_adapter__ = ElevenLabsAdapter

    def __init__(self, **kwargs: Any):
        super().__init__(push_text_frames=False, **kwargs)
        self.inits: list[str] = []

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)
            self.inits.append(context_id)
            assert self.sock is not None
            await self.sock.q.put(("init", context_id))
        self.requests.append((context_id, text))
        await self.start_tts_usage_metrics(text)
        assert self.sock is not None
        await self.sock.q.put(("text", context_id, text))
        yield None

    async def flush_audio(self, context_id: str | None = None) -> None:
        if self.sock is not None and context_id:
            await self.sock.q.put(("flush", context_id))

    async def _close_context(self, context_id: str) -> None:
        if self.sock is not None and context_id:
            await self.sock.q.put(("close", context_id))

    async def on_turn_context_completed(self) -> None:
        ctx = self._turn_context_id
        await super().on_turn_context_completed()
        if ctx:
            await self._close_context(ctx)

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        if self.sock is not None:
            self.sock.cancelled.add(context_id)
        await self._close_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def _fake_handle(self, sock: _FakeSocket, msg: tuple[Any, ...]) -> None:
        kind, ctx = msg[0], msg[1]
        if kind == "init":
            sock.clock[ctx] = 0.0
            return
        if ctx in sock.cancelled:
            return
        if kind == "text":
            if ctx not in sock.clock:
                sock.log.append(("text-before-init", ctx))
            await sock.latency()
            if self.audio_context_available(ctx):
                await self._emit(ctx, msg[2])
        elif kind == "close":
            await sock.latency()
            if self.audio_context_available(ctx):
                await self.append_to_audio_context(ctx, TTSStoppedFrame(context_id=ctx))
                await self.remove_audio_context(ctx)
