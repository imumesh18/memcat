"""A local Deepgram Aura TTS server, faithful to the wire protocol Pipecat 1.12 speaks.

WebSocket ``GET /v1/speak?model=..&encoding=linear16&sample_rate=..`` (header
``Authorization: Token <key>``):

* client -> server, JSON text: ``Speak {text}``, ``Flush``, ``Clear``, ``Close``.
* server -> client: binary linear16 audio; JSON ``Metadata`` (on connect),
  ``Flushed {sequence_id}``, ``Cleared {sequence_id}``, ``Warning {warn_code, warn_msg}``.

One socket handles its messages strictly in order: audio for text received before
a ``Flush`` is all sent before that ``Flushed``. ``Clear`` acts at once: pending
text and audio are dropped, then ``Cleared`` is sent. More than ``flush_limit``
``Flush`` messages in ``flush_window_s`` get a ``Warning`` and are not flushed.

HTTP ``POST /v1/speak?model=..&encoding=..&sample_rate=..&container=none`` with JSON
``{"text": ...}`` streams raw audio back.

Audio is the tone-per-word speech of :func:`tests.fakes.synth`, so a listener
can decode exactly which words were heard, in which order.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web

from tests.fakes import synth

CHUNK_MS = 50


@dataclass
class Conn:
    """One WebSocket connection and everything that happened on it."""

    params: dict[str, str]
    headers: dict[str, str]
    received: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    sent: list[tuple[float, str, Any]] = field(default_factory=list)  # (t, kind, payload)


class DeepgramEmulator:
    def __init__(
        self,
        *,
        generate_on_flush: bool = False,
        latency: tuple[float, float] = (0.005, 0.03),
        chunk_jitter: tuple[float, float] = (0.0, 0.004),
        flush_limit: int = 20,
        flush_window_s: float = 60.0,
        api_key: str = "test",
        clear_lag_s: float = 0.0,
        stale_text: str = "",
    ):
        self.generate_on_flush = generate_on_flush
        # Audio already on the wire when a Clear lands still arrives, before "Cleared":
        # after ``clear_lag_s``, ``stale_text`` is spoken, then "Cleared" is sent.
        # Messages sent meanwhile are read only after "Cleared", as on one socket.
        self.clear_lag_s = clear_lag_s
        self.stale_text = stale_text
        self.latency = latency
        self.chunk_jitter = chunk_jitter
        self.flush_limit = flush_limit
        self.flush_window_s = flush_window_s
        self.api_key = api_key
        self.conns: list[Conn] = []
        self.http_requests: list[dict[str, Any]] = []
        self.rejected_flushes: list[float] = []
        self.errors: list[str] = []
        self._runner: web.AppRunner | None = None
        self.port = 0

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> DeepgramEmulator:
        app = web.Application()
        app.router.add_get("/v1/speak", self._ws_handler)
        app.router.add_post("/v1/speak", self._http_handler)
        self._runner = web.AppRunner(app, handle_signals=False)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return self

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------- queries

    def messages(self, kind: str | None = None) -> list[dict[str, Any]]:
        out = [m for c in self.conns for _, m in c.received]
        return [m for m in out if kind is None or m.get("type") == kind]

    def speak_texts(self) -> list[str]:
        return [m["text"] for m in self.messages("Speak")]

    def flush_times(self) -> list[float]:
        return [t for c in self.conns for t, m in c.received if m.get("type") == "Flush"]

    def sent(self, kind: str) -> list[Any]:
        return [p for c in self.conns for _, k, p in c.sent if k == kind]

    def max_flushes_in_window(self) -> int:
        ts = sorted(self.flush_times())
        best, j = 0, 0
        for i, t in enumerate(ts):
            while ts[j] <= t - self.flush_window_s:
                j += 1
            best = max(best, i - j + 1)
        return best

    # ------------------------------------------------------------ handlers

    def _authorized(self, request: web.Request) -> bool:
        return request.headers.get("Authorization") == f"Token {self.api_key}"

    async def _ws_handler(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return web.Response(status=401, text="bad credentials")
        params = dict(request.query)
        if params.get("encoding", "linear16") != "linear16":
            return web.Response(status=400, text="emulator speaks linear16 only")
        try:
            sr = int(params["sample_rate"])
            assert sr > 0
        except Exception:
            return web.Response(status=400, text="sample_rate required")
        ws = web.WebSocketResponse(autoping=True)
        await ws.prepare(request)
        conn = Conn(params, dict(request.headers))
        self.conns.append(conn)
        sock = _Socket(self, conn, ws, sr)
        await sock.send_json(
            {
                "type": "Metadata",
                "request_id": f"req-{len(self.conns)}",
                "model_name": params.get("model", ""),
                "model_version": "emu",
                "model_uuid": "00000000-0000-0000-0000-000000000000",
            }
        )
        worker = asyncio.create_task(sock.work())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        self.errors.append(f"bad json {msg.data!r}")
                        continue
                    conn.received.append((time.monotonic(), data))
                    if await sock.on_message(data):
                        break
                elif msg.type == WSMsgType.BINARY:
                    self.errors.append("client sent binary")
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            worker.cancel()
            with contextlib.suppress(BaseException):
                await worker
            with contextlib.suppress(Exception):
                await ws.close()
        return ws

    async def _http_handler(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return web.Response(status=401, text="bad credentials")
        params = dict(request.query)
        body = await request.json()
        self.http_requests.append({"params": params, "body": body})
        try:
            sr = int(params["sample_rate"])
        except Exception:
            return web.Response(status=400, text="sample_rate required")
        if params.get("encoding", "linear16") != "linear16" or params.get("container") != "none":
            return web.Response(status=400, text="emulator streams raw linear16 only")
        await asyncio.sleep(random.uniform(*self.latency))
        pcm, _ = synth(body["text"], sr)
        resp = web.StreamResponse(status=200, headers={"Content-Type": "audio/l16"})
        await resp.prepare(request)
        step = sr * CHUNK_MS // 1000 * 2
        for i in range(0, len(pcm), step):
            await resp.write(pcm[i : i + step])
            await asyncio.sleep(random.uniform(*self.chunk_jitter))
        await resp.write_eof()
        return resp

    def _flush_allowed(self, times: deque[float], now: float) -> bool:
        while times and times[0] <= now - self.flush_window_s:
            times.popleft()
        if len(times) >= self.flush_limit:
            return False
        times.append(now)
        return True


class _Socket:
    """Server side of one connection: reader feeds an ordered work queue."""

    def __init__(self, emu: DeepgramEmulator, conn: Conn, ws: web.WebSocketResponse, sr: int):
        self.emu = emu
        self.conn = conn
        self.ws = ws
        self.sr = sr
        self.queue: asyncio.Queue[tuple[int, str, str]] = asyncio.Queue()
        self.epoch = 0  # bumped by Clear; stale work is dropped
        self.buffer: list[str] = []  # generate_on_flush: text waiting for a Flush
        self.flush_seq = 0
        self.clear_seq = 0
        self.flush_times: deque[float] = deque()  # the limit applies per connection
        self._send_lock = asyncio.Lock()

    async def send_json(self, obj: dict[str, Any]) -> None:
        async with self._send_lock:
            self.conn.sent.append((time.monotonic(), obj["type"], obj))
            await self.ws.send_str(json.dumps(obj))

    async def send_audio(self, chunk: bytes) -> None:
        async with self._send_lock:
            self.conn.sent.append((time.monotonic(), "audio", len(chunk)))
            await self.ws.send_bytes(chunk)

    async def on_message(self, data: dict[str, Any]) -> bool:
        """Handle one client message; True when the socket should close."""
        kind = data.get("type")
        if kind == "Speak":
            await self.queue.put((self.epoch, "speak", str(data.get("text", ""))))
        elif kind == "Flush":
            if self.emu._flush_allowed(self.flush_times, time.monotonic()):
                await self.queue.put((self.epoch, "flush", ""))
            else:
                self.emu.rejected_flushes.append(time.monotonic())
                await self.send_json(
                    {
                        "type": "Warning",
                        "warn_code": "FLUSH_LIMIT_EXCEEDED",
                        "warn_msg": f"Exceeded {self.emu.flush_limit} flushes per "
                        f"{self.emu.flush_window_s:g}s; Flush ignored",
                    }
                )
        elif kind == "Clear":
            self.epoch += 1
            self.buffer.clear()
            while not self.queue.empty():
                self.queue.get_nowait()
            if self.emu.clear_lag_s:
                await asyncio.sleep(self.emu.clear_lag_s)
                if self.emu.stale_text:
                    stale, _ = synth(self.emu.stale_text, self.sr)
                    await self.send_audio(stale)
            await self.send_json({"type": "Cleared", "sequence_id": self.clear_seq})
            self.clear_seq += 1
        elif kind == "Close":
            return True
        else:
            self.emu.errors.append(f"unknown message {data!r}")
        return False

    async def work(self) -> None:
        while True:
            epoch, kind, text = await self.queue.get()
            if epoch != self.epoch:
                continue
            try:
                if kind == "speak":
                    if self.emu.generate_on_flush:
                        self.buffer.append(text)
                    else:
                        await self._speak(epoch, text)
                elif kind == "flush":
                    if self.emu.generate_on_flush and self.buffer:
                        text, self.buffer = "".join(self.buffer), []
                        await self._speak(epoch, text)
                    if epoch == self.epoch:
                        await self.send_json({"type": "Flushed", "sequence_id": self.flush_seq})
                        self.flush_seq += 1
            except asyncio.CancelledError:
                raise
            except Exception as e:  # socket gone
                self.emu.errors.append(f"worker: {type(e).__name__}: {e}")
                return

    async def _speak(self, epoch: int, text: str) -> None:
        await asyncio.sleep(random.uniform(*self.emu.latency))
        pcm, _ = synth(text, self.sr)
        step = self.sr * CHUNK_MS // 1000 * 2
        for i in range(0, len(pcm), step):
            if epoch != self.epoch:
                return
            await self.send_audio(pcm[i : i + step])
            await asyncio.sleep(random.uniform(*self.emu.chunk_jitter))
