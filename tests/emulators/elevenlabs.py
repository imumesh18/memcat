"""A local ElevenLabs: the multi-context WebSocket and the HTTP stream-with-timestamps API.

Speaks the wire protocol Pipecat 1.12's ``ElevenLabsTTSService`` and
``ElevenLabsHttpTTSService`` use (read against
``pipecat/services/elevenlabs/tts.py`` and ``tts_base.py``), with the tone-per-word
audio of :mod:`tests.fakes` so a test can hear which words played, in what order.

WebSocket (``/v1/text-to-speech/{voice}/multi-stream-input``)
    Client messages: context init ``{"text": " ", "context_id", "voice_settings"?}``,
    text ``{"text", "context_id"}``, ``{"context_id", "flush": true}``,
    ``{"context_id", "close_context": true}``, keepalive ``{"text": ""}`` (optionally
    with ``context_id``) and ``{"close_socket": true}``.
    Server messages: ``{"audio": b64, "alignment": {chars, charStartTimesMs,
    charDurationsMs}, "normalizedAlignment": ..., "contextId"}`` (alignment times are
    relative to the chunk they arrive with) and ``{"isFinal": true, "contextId"}``.
    Each context is generated in order on its own task, so contexts interleave on
    the socket the way they do on the real service. ``close_context`` lets the
    audio for text already received finish, then sends ``isFinal``.

HTTP (``POST /v1/text-to-speech/{voice}/stream/with-timestamps``)
    Newline-delimited JSON: ``{"audio_base64", "alignment": {characters,
    character_start_times_seconds, character_end_times_seconds},
    "normalized_alignment"}``, times relative to the start of the response.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web

from tests.fakes import EDGE_MS, GAP_MS, TONE_MS, _norm, synth

CHUNK_MS = 50


# ----------------------------------------------------------------- audio + alignment


@dataclass
class Chunk:
    pcm: bytes
    chars: list[str]
    starts_ms: list[int]  # relative to the chunk
    durs_ms: list[int]
    t0_ms: int  # chunk start, relative to the utterance


def render(text: str, sr: int, chunk_ms: int = CHUNK_MS) -> list[Chunk]:
    """Tone audio for ``text`` cut into ~``chunk_ms`` chunks, each with the
    character alignment of the audio it carries.

    Every character of ``text`` (spaces and punctuation included) gets a start time:
    a word's characters share its tone, the whitespace after it shares the gap, and
    characters with no audio of their own sit at zero width. Chunks are cut where a
    word starts, so the last character of a chunk ends exactly at the chunk's end
    and every chunk but the last completes at least one word: Pipecat advances its
    running word clock only on chunks that yield a complete word (as real
    ElevenLabs chunks, which span many words, always do).
    """
    pcm, _ = synth(text, sr)
    ms = sr // 1000
    n_samples = len(pcm) // 2
    tone, gap, edge = sr * TONE_MS // 1000, sr * GAP_MS // 1000, sr * EDGE_MS // 1000

    def q(x: int) -> int:  # quantize to whole milliseconds
        return x // ms * ms

    chars: list[str] = []
    starts: list[int] = []
    pos = edge
    pending_gap = 0
    for m in re.finditer(r"\S+|\s+", text):
        tok = m.group(0)
        if tok.isspace():
            span, pending_gap = pending_gap, 0
        elif _norm(tok):
            pos += pending_gap
            pending_gap = 0
            span = tone
        else:
            span = 0
        for k, ch in enumerate(tok):
            chars.append(ch)
            starts.append(pos + q(k * span // len(tok)))
        pos += span
        if not tok.isspace() and _norm(tok):
            pending_gap = gap
    pos += pending_gap + edge
    assert pos == n_samples, (pos, n_samples)

    min_len = sr * chunk_ms // 1000
    out: list[Chunk] = []
    cut, cur = 0, []  # cur: indexes of chars in the open chunk

    def emit(end: int) -> None:
        nonlocal cut
        rel = [(starts[i] - cut) // ms for i in cur]
        ends = [*(starts[i] for i in cur[1:]), end]
        durs = [(e - starts[i]) // ms for e, i in zip(ends, cur, strict=True)]
        out.append(Chunk(pcm[cut * 2 : end * 2], [chars[i] for i in cur], rel, durs, cut // ms))
        cut = end

    for i, s in enumerate(starts):
        word_start = not chars[i].isspace() and (i == 0 or chars[i - 1].isspace())
        if cur and word_start and s - cut >= min_len:
            emit(s)
            cur = []
        cur.append(i)
    emit(n_samples)
    return out


# ------------------------------------------------------------------------- server


@dataclass
class Received:
    conn: int
    t: float
    msg: dict[str, Any]


@dataclass
class _Ctx:
    cid: str
    conn: int
    queue: asyncio.Queue[tuple[str, Any]] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None
    init: dict[str, Any] | None = None
    texts: list[str] = field(default_factory=list)
    flushed: bool = False
    closed: bool = False
    extra_closes: int = 0
    final_at: float | None = None
    opened_at: float = 0.0
    anomalies: list[str] = field(default_factory=list)


class ElevenLabsEmulator:
    """Start with ``await emu.start()``; point services at ``ws_url`` / ``http_url``."""

    def __init__(
        self,
        *,
        latency: tuple[float, float] = (0.005, 0.03),
        jitter: tuple[float, float] = (0.0, 0.004),
        seed: int | None = None,
    ):
        self.latency = latency
        self.jitter = jitter
        self.rng = random.Random(seed)
        self.port = 0
        self._runner: web.AppRunner | None = None
        self._sockets: set[web.WebSocketResponse] = set()
        # WebSocket wire log
        self.connections: list[dict[str, Any]] = []
        self.received: list[Received] = []
        self.sent_finals: list[tuple[str, float]] = []
        self.contexts: dict[str, _Ctx] = {}
        self.errors: list[str] = []
        # HTTP wire log
        self.http_requests: list[dict[str, Any]] = []

    # ------------------------------------------------------------ lifecycle

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> ElevenLabsEmulator:
        app = web.Application()
        app.router.add_get("/v1/text-to-speech/{voice}/multi-stream-input", self._ws)
        app.router.add_post("/v1/text-to-speech/{voice}/stream/with-timestamps", self._http)
        self._runner = web.AppRunner(app, handle_signals=False, shutdown_timeout=1.0)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        server = site._server
        assert server is not None and server.sockets
        self.port = server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        for c in self.contexts.values():
            if c.task is not None:
                c.task.cancel()
                with contextlib.suppress(BaseException):
                    await c.task
        for ws in list(self._sockets):
            with contextlib.suppress(Exception):
                await ws.close()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def __aenter__(self) -> ElevenLabsEmulator:
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    def _sleep_latency(self) -> float:
        return self.rng.uniform(*self.latency)

    # ------------------------------------------------------------ WebSocket

    @staticmethod
    def _sr(output_format: str | None) -> int:
        m = re.fullmatch(r"pcm_(\d+)", output_format or "")
        if not m:
            raise web.HTTPBadRequest(text=f"unsupported output_format {output_format!r}")
        return int(m.group(1))

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        sr = self._sr(request.query.get("output_format"))
        conn = len(self.connections)
        self.connections.append(
            {
                "voice": request.match_info["voice"],
                "query": dict(request.query),
                "api_key": request.headers.get("xi-api-key"),
            }
        )
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        self._sockets.add(ws)
        lock = asyncio.Lock()

        async def send(obj: dict[str, Any]) -> None:
            async with lock:
                if not ws.closed:
                    await ws.send_str(json.dumps(obj))

        mine: list[_Ctx] = []
        try:
            async for raw in ws:
                if raw.type != WSMsgType.TEXT:
                    continue
                msg = json.loads(raw.data)
                self.received.append(Received(conn, time.perf_counter(), msg))
                if msg.get("close_socket"):
                    break
                cid = msg.get("context_id")
                if cid is None:
                    if msg.get("text") != "":
                        self.errors.append(f"message without context_id: {msg}")
                    continue  # context-less keepalive
                c = self.contexts.get(cid)
                if c is None:
                    if msg.get("flush") or msg.get("close_context"):
                        # Never opened: the real service has nothing to finish.
                        self.errors.append(
                            f"{next(iter(msg.keys() - {'context_id'}))} for unknown {cid}"
                        )
                        continue
                    c = _Ctx(cid, conn, opened_at=time.perf_counter())
                    self.contexts[cid] = c
                    mine.append(c)
                    if msg.get("text") != " ":
                        c.anomalies.append("first message was not the context init")
                    else:
                        c.init = msg
                        continue
                    c.task = asyncio.ensure_future(self._run_ctx(c, sr, send))
                if c.closed:
                    if msg.get("close_context"):
                        # Pipecat closes again on interruption after a turn-end close;
                        # the service ignores a second close.
                        c.extra_closes += 1
                    else:
                        c.anomalies.append(f"message after close_context: {msg}")
                    continue
                if c.task is None:
                    c.task = asyncio.ensure_future(self._run_ctx(c, sr, send))
                if msg.get("close_context"):
                    c.closed = True
                    await c.queue.put(("close", None))
                elif msg.get("flush"):
                    c.flushed = True
                    await c.queue.put(("flush", None))
                elif "text" in msg:
                    text = msg["text"]
                    if text == "":
                        continue  # keepalive for this context
                    c.texts.append(text)
                    await c.queue.put(("text", text))
        finally:
            for c in mine:
                if c.task is not None and not c.task.done():
                    c.task.cancel()
            self._sockets.discard(ws)
            await ws.close()
        return ws

    async def _run_ctx(self, c: _Ctx, sr: int, send: Any) -> None:
        while True:
            kind, text = await c.queue.get()
            if kind == "text":
                if not text.strip():
                    continue
                await asyncio.sleep(self._sleep_latency())
                for ch in render(text, sr):
                    align = {
                        "chars": ch.chars,
                        "charStartTimesMs": ch.starts_ms,
                        "charDurationsMs": ch.durs_ms,
                    }
                    await send(
                        {
                            "audio": base64.b64encode(ch.pcm).decode(),
                            "alignment": align,
                            "normalizedAlignment": align,
                            "contextId": c.cid,
                        }
                    )
                    await asyncio.sleep(self.rng.uniform(*self.jitter))
            elif kind == "close":
                await asyncio.sleep(self._sleep_latency() / 3)
                c.final_at = time.perf_counter()
                self.sent_finals.append((c.cid, c.final_at))
                await send({"isFinal": True, "contextId": c.cid})
                return
            # "flush": text is generated as it arrives, so there is nothing buffered

    # ----------------------------------------------------------------- HTTP

    async def _http(self, request: web.Request) -> web.StreamResponse:
        sr = self._sr(request.query.get("output_format"))
        body = await request.json()
        self.http_requests.append(
            {
                "voice": request.match_info["voice"],
                "query": dict(request.query),
                "api_key": request.headers.get("xi-api-key"),
                "json": body,
            }
        )
        resp = web.StreamResponse(status=200, headers={"Content-Type": "application/json"})
        await resp.prepare(request)
        await asyncio.sleep(self._sleep_latency())
        for ch in render(body["text"], sr):
            t0 = ch.t0_ms / 1000.0
            align = {
                "characters": ch.chars,
                "character_start_times_seconds": [t0 + s / 1000.0 for s in ch.starts_ms],
                "character_end_times_seconds": [
                    t0 + (s + d) / 1000.0 for s, d in zip(ch.starts_ms, ch.durs_ms, strict=True)
                ],
            }
            line = {
                "audio_base64": base64.b64encode(ch.pcm).decode(),
                "alignment": align,
                "normalized_alignment": align,
            }
            await resp.write(json.dumps(line).encode() + b"\n")
            await asyncio.sleep(self.rng.uniform(*self.jitter))
        await resp.write_eof()
        return resp

    # -------------------------------------------------------------- queries

    def ws_texts(self) -> list[tuple[str, str]]:
        """(context_id, text) for every non-init, non-keepalive text message, in order."""
        out = []
        for r in self.received:
            m = r.msg
            cid, text = m.get("context_id"), m.get("text")
            if cid is None or text in (None, ""):
                continue
            c = self.contexts.get(cid)
            if c is not None and c.init is m:
                continue
            out.append((cid, text))
        return out

    def messages_for(self, cid: str) -> list[dict[str, Any]]:
        return [r.msg for r in self.received if r.msg.get("context_id") == cid]
