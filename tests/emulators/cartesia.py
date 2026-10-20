"""A local Cartesia TTS server, faithful to the wire protocol Pipecat 1.12 speaks.

WebSocket ``/tts/websocket`` (what ``CartesiaTTSService`` uses):

* The client sends JSON messages ``{"transcript", "continue", "context_id",
  "model_id", "voice", "output_format", "add_timestamps", ...}`` and
  ``{"context_id", "cancel": true}``.
* Per ``context_id`` the server synthesizes each transcript in arrival order and
  streams ``timestamps`` (word starts in seconds, relative to the start of the
  context) and ``chunk`` (base64 ``pcm_s16le``) messages. Different contexts on one
  socket are generated concurrently, so their messages interleave.
* With ``max_buffer_delay_ms == 0`` every transcript is flushed on its own, which
  real Cartesia acknowledges with ``flush_done``.
* ``continue: false`` closes the context: after its remaining audio, ``done``.
* ``cancel`` stops all further output for the context.

HTTP ``POST /tts/bytes`` (``CartesiaHttpTTSService``) returns raw PCM.

Audio is the tone-per-word audio of :mod:`tests.fakes`, so what a listener hears
can be decoded back into words.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web

from tests.fakes import TONE_MS, synth

__all__ = ["CartesiaEmulator"]


@dataclass
class _Ctx:
    cid: str
    q: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None
    clock_s: float = 0.0
    closed: bool = False  # continue=false received
    cancelled: bool = False
    done_sent: bool = False


class _Conn:
    def __init__(self, emu: CartesiaEmulator, ws: web.WebSocketResponse, n: int):
        self.emu = emu
        self.ws = ws
        self.n = n
        self.ctx: dict[str, _Ctx] = {}
        self.lock = asyncio.Lock()
        self.flush_id = 0

    async def send(self, msg: dict[str, Any]) -> None:
        if self.ws.closed:
            return
        async with self.lock:
            with contextlib.suppress(Exception):
                await self.ws.send_str(json.dumps(msg))
        self.emu.sent.append({"conn": self.n, **{k: v for k, v in msg.items() if k != "data"}})

    async def handle(self, msg: dict[str, Any]) -> None:
        cid = msg.get("context_id")
        if not cid:
            self.emu.violations.append(f"message without context_id: {msg}")
            return
        if msg.get("cancel"):
            c = self.ctx.get(cid)
            if c is None:
                return  # cancelling an unknown/finished context is a no-op
            c.cancelled = True
            if c.task is not None:
                c.task.cancel()
            return
        if "transcript" not in msg:
            self.emu.violations.append(f"unknown message: {msg}")
            return
        c = self.ctx.get(cid)
        if c is None:
            c = _Ctx(cid)
            self.ctx[cid] = c
            c.task = asyncio.ensure_future(self._run_ctx(c))
        if c.cancelled:
            self.emu.violations.append(f"transcript for cancelled context {cid}: {msg}")
            return
        if c.closed:
            self.emu.violations.append(f"transcript after continue=false on {cid}: {msg}")
            return
        if msg.get("continue") is False:
            c.closed = True
        c.q.put_nowait(msg)

    async def _run_ctx(self, c: _Ctx) -> None:
        try:
            while True:
                msg = await c.q.get()
                if msg is None:
                    return
                await self._generate(c, msg)
                if msg.get("continue") is False:
                    c.done_sent = True
                    await self.send(
                        {"type": "done", "done": True, "status_code": 206, "context_id": c.cid}
                    )
                    return
        except asyncio.CancelledError:
            pass

    async def _generate(self, c: _Ctx, msg: dict[str, Any]) -> None:
        text = msg.get("transcript") or ""
        fmt = msg.get("output_format") or {}
        sr = int(fmt.get("sample_rate") or 0)
        if fmt.get("encoding") != "pcm_s16le" or fmt.get("container") != "raw" or sr <= 0:
            self.emu.violations.append(f"unsupported output_format {fmt}")
            await self.send(
                {
                    "type": "error",
                    "context_id": c.cid,
                    "status_code": 400,
                    "done": True,
                    "title": "Invalid output format",
                    "message": f"bad output_format {fmt}",
                }
            )
            return
        if not text.strip():
            return
        await asyncio.sleep(random.uniform(*self.emu.first_byte_s))
        pcm, words = synth(text, sr)
        base = c.clock_s
        pending = list(words)
        pos = 0
        bps = 2 * sr
        while pos < len(pcm):
            ms = random.randint(*self.emu.chunk_ms)
            n = sr * ms // 1000 * 2
            chunk = pcm[pos : pos + n]
            end_s = (pos + len(chunk)) / bps
            if msg.get("add_timestamps") and pending and pending[0][1] < end_s:
                batch = []
                while pending and pending[0][1] < end_s:
                    batch.append(pending.pop(0))
                await self.send(
                    {
                        "type": "timestamps",
                        "done": False,
                        "status_code": 206,
                        "context_id": c.cid,
                        "word_timestamps": {
                            "words": [w for w, _ in batch],
                            "start": [round(base + t, 6) for _, t in batch],
                            "end": [round(base + t + TONE_MS / 1000, 6) for _, t in batch],
                        },
                    }
                )
            await self.send(
                {
                    "type": "chunk",
                    "data": base64.b64encode(chunk).decode(),
                    "done": False,
                    "status_code": 206,
                    "step_time": ms,
                    "context_id": c.cid,
                }
            )
            pos += len(chunk)
            await asyncio.sleep(random.uniform(*self.emu.chunk_gap_s))
        c.clock_s = base + len(pcm) / bps
        if msg.get("max_buffer_delay_ms") == 0:
            self.flush_id += 1
            await self.send(
                {
                    "type": "flush_done",
                    "done": False,
                    "flush_done": True,
                    "flush_id": self.flush_id,
                    "status_code": 206,
                    "context_id": c.cid,
                }
            )

    async def close(self) -> None:
        for c in self.ctx.values():
            if c.task is not None and not c.task.done():
                c.task.cancel()
                with contextlib.suppress(BaseException):
                    await c.task


class CartesiaEmulator:
    """``async with CartesiaEmulator() as emu:`` then point services at
    ``emu.ws_url`` / ``emu.http_url``."""

    def __init__(
        self,
        *,
        chunk_ms: tuple[int, int] = (20, 50),
        chunk_gap_s: tuple[float, float] = (0.0, 0.010),
        first_byte_s: tuple[float, float] = (0.005, 0.03),
    ):
        self.chunk_ms = chunk_ms
        self.chunk_gap_s = chunk_gap_s
        self.first_byte_s = first_byte_s
        #: every WebSocket message received, parsed, with ``_conn`` (connection number)
        self.received: list[dict[str, Any]] = []
        #: every server message sent (audio payload omitted)
        self.sent: list[dict[str, Any]] = []
        #: every HTTP /tts/bytes request body
        self.http_requests: list[dict[str, Any]] = []
        #: handshake headers per WebSocket connection
        self.handshakes: list[dict[str, str]] = []
        #: protocol misuse by the client
        self.violations: list[str] = []
        self._conns: list[_Conn] = []
        self._runner: web.AppRunner | None = None
        self.port = 0

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> CartesiaEmulator:
        app = web.Application()
        app.router.add_get("/tts/websocket", self._ws)
        app.router.add_post("/tts/bytes", self._bytes)
        self._runner = web.AppRunner(app, handle_signals=False)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        server = site._server
        assert server is not None
        self.port = server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return self

    async def stop(self) -> None:
        for c in self._conns:
            await c.close()
            with contextlib.suppress(Exception):
                await c.ws.close()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def __aenter__(self) -> CartesiaEmulator:
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/tts/websocket"

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ---------------------------------------------------------------- helpers

    def transcripts(self) -> list[tuple[str, str]]:
        """(context_id, transcript) of every non-empty transcript received."""
        return [
            (m["context_id"], m["transcript"])
            for m in self.received
            if m.get("transcript") and m["transcript"].strip()
        ]

    def contexts(self) -> list[str]:
        out: list[str] = []
        for m in self.received:
            cid = m.get("context_id")
            if cid and "transcript" in m and cid not in out:
                out.append(cid)
        return out

    def closed_contexts(self) -> set[str]:
        return {m["context_id"] for m in self.received if m.get("continue") is False}

    def cancelled_contexts(self) -> set[str]:
        return {m["context_id"] for m in self.received if m.get("cancel")}

    def done_contexts(self) -> set[str]:
        return {m["context_id"] for m in self.sent if m.get("type") == "done"}

    def reset(self) -> None:
        self.received.clear()
        self.sent.clear()
        self.http_requests.clear()
        self.violations.clear()

    # --------------------------------------------------------------- handlers

    async def _ws(self, request: web.Request) -> web.StreamResponse:
        if not request.headers.get("X-API-Key"):
            return web.Response(status=401, text="missing X-API-Key")
        if not request.headers.get("Cartesia-Version"):
            return web.Response(status=400, text="missing Cartesia-Version")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        conn = _Conn(self, ws, len(self._conns))
        self._conns.append(conn)
        self.handshakes.append(dict(request.headers))
        try:
            async for m in ws:
                if m.type != WSMsgType.TEXT:
                    continue
                try:
                    msg = json.loads(m.data)
                except ValueError:
                    self.violations.append(f"non-JSON message {m.data!r}")
                    continue
                self.received.append({"_conn": conn.n, **msg})
                await conn.handle(msg)
        finally:
            await conn.close()
        return ws

    async def _bytes(self, request: web.Request) -> web.StreamResponse:
        if not request.headers.get("X-API-Key"):
            return web.Response(status=401, text="missing X-API-Key")
        body = await request.json()
        self.http_requests.append(body)
        fmt = body.get("output_format") or {}
        sr = int(fmt.get("sample_rate") or 0)
        if fmt.get("encoding") != "pcm_s16le" or fmt.get("container") != "raw" or sr <= 0:
            return web.Response(status=400, text=f"bad output_format {fmt}")
        if not body.get("model_id") or not (body.get("voice") or {}).get("id"):
            return web.Response(status=400, text="model_id and voice.id are required")
        await asyncio.sleep(random.uniform(*self.first_byte_s))
        pcm, _ = synth(body.get("transcript") or "", sr)
        return web.Response(body=pcm, content_type="application/octet-stream")
