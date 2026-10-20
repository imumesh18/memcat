"""A local Sarvam (Bulbul) TTS server, as Pipecat 1.12's Sarvam services use it.

Read against ``pipecat/services/sarvam/tts.py``:

WebSocket ``{url}?model=<model>&send_completion_event=true`` with header
``api-subscription-key``. Client messages (JSON text frames)::

    {"type": "config", "data": {"speech_sample_rate": "16000", "min_buffer_size": 50, ...}}
    {"type": "text",   "data": {"text": "..."}}
    {"type": "flush"}
    {"type": "ping"}

Server messages::

    {"type": "audio", "data": {"request_id": ..., "content_type": "audio/wav",
                               "audio": <base64 linear16 PCM at speech_sample_rate>}}
    {"type": "event", "data": {"event_type": "final"}}   # after a flush, once audio is out
    {"type": "error", "data": {"message": "..."}}

HTTP ``POST {base_url}/text-to-speech`` -> ``{"request_id": ..., "audios": [<base64 WAV>]}``.

Audio is the tone-per-word audio of ``tests.fakes.synth``, so a listener's word
order can be decoded from the pipeline output.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import random
import uuid
import wave
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlparse

from aiohttp import web
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from tests.fakes import synth

CHUNK_MS = 50


@dataclass
class WSConn:
    """One client WebSocket: what it sent and what we generated for it."""

    peer_port: int
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    received: list[dict[str, Any]] = field(default_factory=list)
    generated: list[str] = field(default_factory=list)  # each text buffer synthesized
    finals: int = 0
    closed: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def texts(self) -> list[str]:
        return [m["data"]["text"] for m in self.received if m.get("type") == "text"]

    def types(self) -> list[str]:
        return [m.get("type", "?") for m in self.received]


class SarvamEmulator:
    def __init__(
        self,
        *,
        delay: tuple[float, float] = (0.002, 0.012),
        first_audio_delay: tuple[float, float] = (0.02, 0.06),
        api_key: str = "test",
    ):
        self.delay = delay
        self.first_audio_delay = first_audio_delay
        self.api_key = api_key
        self.conns: list[WSConn] = []
        self.http_requests: list[dict[str, Any]] = []
        self.http_headers: list[dict[str, str]] = []
        self._ws_server: Any = None
        self._runner: web.AppRunner | None = None
        self.ws_port = 0
        self.http_port = 0

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> SarvamEmulator:
        self._ws_server = await serve(self._ws_handler, "127.0.0.1", 0, max_size=2**22)
        self.ws_port = self._ws_server.sockets[0].getsockname()[1]
        app = web.Application()
        app.router.add_post("/text-to-speech", self._http_handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.http_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return self

    async def stop(self) -> None:
        if self._ws_server is not None:
            self._ws_server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._ws_server.wait_closed(), 2)
            self._ws_server = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.ws_port}/text-to-speech/ws"

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.http_port}"

    # ------------------------------------------------------------- inspection

    def all_received(self) -> list[dict[str, Any]]:
        return [m for c in self.conns for m in c.received]

    def texts(self, ports: set[int] | None = None) -> list[str]:
        """Text received, optionally only on connections from these client ports."""
        return [t for c in self.of(ports) for t in c.texts()]

    def of(self, ports: set[int] | None) -> list[WSConn]:
        return [c for c in self.conns if ports is None or c.peer_port in ports]

    def http_texts(self, client: str | None = None) -> list[str]:
        """Texts requested over HTTP, optionally only those with header ``x-client: client``."""
        return [
            r["text"]
            for r, h in zip(self.http_requests, self.http_headers, strict=True)
            if client is None or h.get("x-client") == client
        ]

    def clear_http(self) -> None:
        self.http_requests.clear()
        self.http_headers.clear()

    # -------------------------------------------------------------- WebSocket

    async def _ws_handler(self, ws: ServerConnection) -> None:
        req = ws.request
        assert req is not None
        u = urlparse(req.path)
        conn = WSConn(
            peer_port=ws.remote_address[1],
            path=u.path,
            query=parse_qs(u.query),
            headers={k.lower(): v for k, v in req.headers.raw_items()},
        )
        self.conns.append(conn)
        if conn.headers.get("api-subscription-key") != self.api_key:
            await ws.send(_err("Invalid API key"))
            await ws.close(4001, "unauthorized")
            return
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        worker = asyncio.create_task(self._ws_worker(ws, conn, queue))
        try:
            async for raw in ws:
                if not isinstance(raw, str):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send(_err("Invalid JSON"))
                    continue
                conn.received.append(msg)
                await queue.put(msg)
        except ConnectionClosed:
            pass
        finally:
            conn.closed = True
            worker.cancel()
            with contextlib.suppress(BaseException):
                await worker

    async def _ws_worker(
        self, ws: ServerConnection, conn: WSConn, queue: asyncio.Queue[dict[str, Any] | None]
    ) -> None:
        """Handles one socket's messages strictly in order."""
        buf = ""
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    return
                kind = msg.get("type")
                if kind == "config":
                    conn.config.update(msg.get("data") or {})
                elif kind == "ping":
                    continue
                elif kind == "text":
                    if not conn.config:
                        await ws.send(_err("Config must be sent before text"))
                        continue
                    piece = (msg.get("data") or {}).get("text", "")
                    # Assumed: Sarvam treats separate text messages as separate words.
                    if buf and piece and not buf[-1].isspace() and not piece[0].isspace():
                        buf += " "
                    buf += piece
                    if len(buf) >= int(conn.config.get("min_buffer_size") or 50):
                        text, buf = buf, ""
                        await self._generate(ws, conn, text)
                elif kind == "flush":
                    text, buf = buf, ""
                    if text.strip():
                        await self._generate(ws, conn, text)
                    if conn.query.get("send_completion_event") == ["true"]:
                        conn.finals += 1
                        final = {"type": "event", "data": {"event_type": "final"}}
                        await ws.send(json.dumps(final))
                else:
                    await ws.send(_err(f"Unknown message type: {kind}"))
        except ConnectionClosed:
            return

    async def _generate(self, ws: ServerConnection, conn: WSConn, text: str) -> None:
        conn.generated.append(text)
        sr = int(conn.config.get("speech_sample_rate") or 24000)
        pcm, _ = synth(text, sr)
        rid = uuid.uuid4().hex
        step = sr * CHUNK_MS // 1000 * 2
        await asyncio.sleep(random.uniform(*self.first_audio_delay))
        for i in range(0, len(pcm), step):
            data = {
                "request_id": rid,
                "content_type": "audio/wav",
                "audio": base64.b64encode(pcm[i : i + step]).decode(),
            }
            await ws.send(json.dumps({"type": "audio", "data": data}))
            await asyncio.sleep(random.uniform(*self.delay))

    # ------------------------------------------------------------------- HTTP

    async def _http_handler(self, request: web.Request) -> web.Response:
        if request.headers.get("api-subscription-key") != self.api_key:
            return web.json_response({"error": {"message": "Invalid API key"}}, status=403)
        body = await request.json()
        self.http_headers.append({k.lower(): v for k, v in request.headers.items()})
        self.http_requests.append(body)
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return web.json_response({"error": {"message": "text is required"}}, status=400)
        sr = int(body.get("speech_sample_rate") or 24000)
        pcm, _ = synth(text, sr)
        await asyncio.sleep(random.uniform(*self.first_audio_delay))
        return web.json_response(
            {"request_id": uuid.uuid4().hex, "audios": [base64.b64encode(_wav(pcm, sr)).decode()]}
        )


def _wav(pcm: bytes, sr: int) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    out = bio.getvalue()
    assert out[:4] == b"RIFF" and out[36:40] == b"data"  # the 44-byte header Pipecat strips
    return out


def _err(message: str) -> str:
    return json.dumps({"type": "error", "data": {"message": message}})
