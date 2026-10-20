"""Serve a recording back to a real Pipecat service, with the provider's timing.

WebSocket
    A connection binds to the next unused recorded connection with the same path
    and query (model, voice, format). Client messages must match the recording in
    order *per context*: Pipecat's random context ids are mapped to the recorded
    ones on first use, and any other difference is a mismatch. Keepalives
    (``{"text": ""}``) are timer driven and ignored on both sides.

    Each provider message leaves at its recorded delay after the later of (a) the
    last client message of its context that preceded it in the recording and
    (b) the previous provider message of that context. So time to first audio,
    gaps between chunks and slow stretches replay as recorded, measured from
    what the client actually did in this run. Contexts run independently, as they
    do on the provider.

HTTP
    A request binds to the first unused recorded request with the same method,
    path, query and JSON body. Its chunks replay with their recorded byte
    boundaries and offsets from the request's arrival.

``speed`` scales every delay: 1.0 as recorded, 0.0 as fast as possible, 2.0 a
provider twice as slow. ``problems`` lists every mismatch; a test that replays
cleanly leaves it empty.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web

from tests.replay.cassette import Cassette, HttpExchange, WsConn

_CTX_C = "context_id"  # client messages
_CTX_S = "contextId"  # provider messages


def _is_keepalive(msg: dict[str, Any]) -> bool:
    return msg.get("text") == "" and set(msg) <= {"text", _CTX_C}


def _sub(msg: dict[str, Any], key: str, cid: str | None) -> dict[str, Any]:
    if key in msg and cid is not None:
        return {**msg, key: cid}
    return msg


def _shape(msg: dict[str, Any]) -> dict[str, Any]:
    return _sub(msg, _CTX_C, "*")


@dataclass
class _Client:
    idx: int  # index in conn.events
    ctx: str | None
    msg: dict[str, Any]
    seen: asyncio.Event = field(default_factory=asyncio.Event)
    at: float = 0.0


@dataclass
class _Server:
    idx: int
    ctx: str | None
    data: str
    dep: _Client | None


class _ConnReplay:
    def __init__(self, player: Player, conn: WsConn, ws: web.WebSocketResponse, opened: float):
        self.p = player
        self.conn = conn
        self.ws = ws
        self.opened = opened
        self.rec2live: dict[str, str] = {}
        self.live2rec: dict[str, str] = {}
        self.clients: dict[str | None, list[_Client]] = {}
        self.next_client: dict[str | None, int] = {}
        self.tracks: dict[str | None, list[_Server]] = {}
        self.ctx_order: list[str] = []
        last_any: _Client | None = None
        last_by_ctx: dict[str | None, _Client] = {}
        for i, e in enumerate(conn.events):
            m = e.json()
            if e.dir == "c":
                if _is_keepalive(m):
                    continue
                ctx = m.get(_CTX_C)
                if ctx is not None and ctx not in self.clients:
                    self.ctx_order.append(ctx)
                c = _Client(i, ctx, m)
                self.clients.setdefault(ctx, []).append(c)
                last_any = last_by_ctx[ctx] = c
            else:
                ctx = m.get(_CTX_S)
                dep = last_by_ctx.get(ctx) if ctx is not None else last_any
                self.tracks.setdefault(ctx, []).append(_Server(i, ctx, e.data, dep))
        self.sent = 0
        self.total_server = sum(len(t) for t in self.tracks.values())

    # ------------------------------------------------------------ client side

    def _bind(self, live: str, msg: dict[str, Any]) -> str | None:
        for rec in self.ctx_order:
            if rec in self.rec2live:
                continue
            queue = self.clients[rec]
            k = self.next_client.get(rec, 0)
            if k < len(queue) and _sub(msg, _CTX_C, rec) == queue[k].msg:
                self.rec2live[rec] = live
                self.live2rec[live] = rec
                return rec
        return None

    def feed(self, msg: dict[str, Any]) -> str | None:
        """Match one live client message. Returns a problem, or None."""
        if _is_keepalive(msg):
            return None
        live = msg.get(_CTX_C)
        rec: str | None = None
        if live is not None:
            rec = self.live2rec.get(live) or self._bind(live, msg)
            if rec is None:
                return f"unexpected client message for a new context: {_shape(msg)!r:.300}"
        queue = self.clients.get(rec, [])
        k = self.next_client.get(rec, 0)
        if k >= len(queue):
            return f"client sent more than recorded: {_shape(msg)!r:.300}"
        want = queue[k]
        got = _sub(msg, _CTX_C, rec)
        if got != want.msg:
            return (
                f"client message differs: got {_shape(got)!r:.300}, "
                f"recorded {_shape(want.msg)!r:.300}"
            )
        self.next_client[rec] = k + 1
        want.at = time.monotonic()
        want.seen.set()
        return None

    def missing(self) -> list[str]:
        out = []
        for rec, queue in self.clients.items():
            for c in queue[self.next_client.get(rec, 0) :]:
                if c.msg.get("close_socket"):
                    continue
                out.append(f"client never sent: {_shape(c.msg)!r:.300}")
        return out

    # ------------------------------------------------------------ provider side

    async def run_track(self, track: list[_Server]) -> None:
        speed = self.p.speed
        events = self.conn.events
        prev: tuple[float, float] | None = None  # (live send time, recorded time)
        for s in track:
            t = events[s.idx].t
            target = self.opened + t * speed if s.dep is None and prev is None else 0.0
            if prev is not None:
                target = prev[0] + (t - prev[1]) * speed
            if s.dep is not None:
                await s.dep.seen.wait()
                target = max(target, s.dep.at + (t - events[s.dep.idx].t) * speed)
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.p.lateness.append(max(0.0, time.monotonic() - target))
            data = s.data
            if s.ctx is not None:
                live = self.rec2live.get(s.ctx)
                if live is not None:
                    data = json.dumps(_sub(json.loads(data), _CTX_S, live))
            if self.ws.closed:
                return
            await self.ws.send_str(data)
            self.sent += 1
            prev = (time.monotonic(), t)

    async def close_when_recorded(self, tracks: list[asyncio.Task[None]]) -> None:
        close = self.conn.close
        if close is None or close[1] != "s":
            return
        await asyncio.gather(*tracks, return_exceptions=True)
        # The provider closed after some client message (often close_socket): wait for it.
        events = self.conn.events
        before = [c for q in self.clients.values() for c in q if events[c.idx].t <= float(close[0])]
        if before:
            dep = max(before, key=lambda c: c.idx)
            await dep.seen.wait()
            delay = dep.at + (float(close[0]) - events[dep.idx].t) * self.p.speed - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        await self.ws.close(code=int(close[2]))


class Player:
    def __init__(self, cassette: Cassette, *, speed: float = 1.0):
        self.cassette = cassette
        self.speed = speed
        self.problems: list[str] = []
        self.unsent: int = 0
        self.lateness: list[float] = []  # per provider message: seconds past its recorded time
        self.port = 0
        self._ws_used: set[int] = set()
        self._http_used: set[int] = set()
        self._runner: web.AppRunner | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> Player:
        app = web.Application(client_max_size=16 << 20)
        app.router.add_route("*", "/{tail:.*}", self._route)
        self._runner = web.AppRunner(app, handle_signals=False, shutdown_timeout=1.0)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        server = site._server
        assert server is not None and server.sockets
        self.port = server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        if self._runner is not None:
            await self._runner.cleanup()

    async def __aenter__(self) -> Player:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def unused(self) -> list[str]:
        """Recorded connections and requests this run never made."""
        out = [
            f"websocket {c.path} never opened"
            for i, c in enumerate(self.cassette.ws)
            if i not in self._ws_used
        ]
        out += [
            f"{h.method} {h.path} never requested"
            for i, h in enumerate(self.cassette.http)
            if i not in self._http_used
        ]
        return out

    async def _route(self, request: web.Request) -> web.StreamResponse:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self._ws(request)
        return await self._http(request)

    # ------------------------------------------------------------ WebSocket

    async def _ws(self, request: web.Request) -> web.StreamResponse:
        cands = [
            i
            for i, c in enumerate(self.cassette.ws)
            if i not in self._ws_used and c.path == request.path_qs
        ]
        if not cands:
            self.problems.append(f"unrecorded websocket {request.path_qs}")
            return web.Response(status=404, text="not recorded")
        first = self.cassette.ws[cands[0]]
        if first.status != 101:
            self._ws_used.add(cands[0])
            await asyncio.sleep(first.handshake_s * self.speed)
            return web.Response(status=first.status)
        await asyncio.sleep(min(self.cassette.ws[i].handshake_s for i in cands) * self.speed)
        ws = web.WebSocketResponse(max_msg_size=0, autoping=True)
        await ws.prepare(request)
        opened = time.monotonic()
        rep: _ConnReplay | None = None
        pending: list[dict[str, Any]] = []
        tracks: list[asyncio.Task[None]] = []

        def bind(msg: dict[str, Any]) -> _ConnReplay | None:
            for i in cands:
                if i in self._ws_used:
                    continue
                conn = self.cassette.ws[i]
                firsts = [
                    e.json() for e in conn.events if e.dir == "c" and not _is_keepalive(e.json())
                ]
                if firsts and _shape(firsts[0]) == _shape(msg):
                    self._ws_used.add(i)
                    return _ConnReplay(self, conn, ws, opened)
            return None

        try:
            async for m in ws:
                if m.type != WSMsgType.TEXT:
                    continue
                try:
                    msg = json.loads(m.data)
                except ValueError:
                    self.problems.append(f"client sent non-JSON: {m.data!r:.200}")
                    break
                if rep is None:
                    if _is_keepalive(msg):
                        pending.append(msg)
                        continue
                    rep = bind(msg)
                    if rep is None:
                        self.problems.append(
                            f"no recorded connection starts with {_shape(msg)!r:.300}"
                        )
                        await ws.close(code=1008, message=b"not recorded")
                        break
                    for track in rep.tracks.values():
                        tracks.append(asyncio.ensure_future(rep.run_track(track)))
                    closer = asyncio.ensure_future(rep.close_when_recorded(list(tracks)))
                    tracks.append(closer)
                    self._tasks.update(tracks)
                problem = rep.feed(msg)
                if problem:
                    self.problems.append(problem)
                    await ws.close(code=1008, message=b"replay mismatch")
                    break
        finally:
            for t in tracks:
                t.cancel()
                with contextlib.suppress(BaseException):
                    await t
                self._tasks.discard(t)
            if rep is not None:
                self.problems.extend(rep.missing())
                self.unsent += rep.total_server - rep.sent
        return ws

    # ------------------------------------------------------------ HTTP

    async def _http(self, request: web.Request) -> web.StreamResponse:
        arrived = time.monotonic()
        raw = await request.read()
        try:
            body: Any = json.loads(raw) if raw else None
        except ValueError:
            body = raw.decode(errors="replace")
        query = sorted(request.query.items())
        ex: HttpExchange | None = None
        for i, h in enumerate(self.cassette.http):
            if i in self._http_used:
                continue
            if (h.method, h.path, sorted(h.query), h.body) == (
                request.method,
                request.path,
                query,
                body,
            ):
                self._http_used.add(i)
                ex = h
                break
        if ex is None:
            self.problems.append(
                f"unrecorded request {request.method} {request.path_qs} {body!r:.300}"
            )
            return web.Response(status=404, text="not recorded")
        resp = web.StreamResponse(status=ex.status, headers={"Content-Type": ex.content_type})
        await resp.prepare(request)
        for dt, chunk in ex.chunks:
            target = arrived + dt * self.speed
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.lateness.append(max(0.0, time.monotonic() - target))
            await resp.write(chunk)
        await resp.write_eof()
        return resp
