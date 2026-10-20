"""A local proxy that records a provider's real traffic.

Point the Pipecat service at ``ws_url`` / ``http_url`` with any dummy key. The
recorder forwards every request to the provider with the real key in the
``xi-api-key`` header (configurable), passes every message and chunk back
untouched, and logs each one with its time. The key lives only in this process:
it is never sent to the client, never logged, and ``save`` refuses to write a
file that contains it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import aiohttp
from aiohttp import WSMsgType, web

from tests.replay.cassette import Cassette, HttpExchange, WsConn, WsEvent

_HOP = {"host", "content-length", "transfer-encoding", "connection", "upgrade", "accept-encoding"}


class Recorder:
    def __init__(
        self,
        *,
        upstream_ws: str,
        upstream_http: str,
        api_key: str,
        key_header: str = "xi-api-key",
        meta: dict[str, Any] | None = None,
    ):
        self.upstream_ws = upstream_ws.rstrip("/")
        self.upstream_http = upstream_http.rstrip("/")
        self._key = api_key
        self._key_header = key_header
        self.cassette = Cassette(meta=dict(meta or {}))
        self.port = 0
        self._t0 = time.monotonic()
        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def secrets(self) -> tuple[str, ...]:
        return (self._key,)

    def _now(self) -> float:
        return time.monotonic() - self._t0

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> Recorder:
        self._session = aiohttp.ClientSession()  # store bodies as the client reads them
        app = web.Application(client_max_size=16 << 20)
        app.router.add_route("*", "/{tail:.*}", self._route)
        self._runner = web.AppRunner(app, handle_signals=False, shutdown_timeout=1.0)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        server = site._server
        assert server is not None and server.sockets
        self.port = server.sockets[0].getsockname()[1]
        self._t0 = time.monotonic()
        return self

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        if self._runner is not None:
            await self._runner.cleanup()
        if self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> Recorder:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def _headers(self, request: web.Request) -> dict[str, str]:
        h = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
        h.pop(self._key_header, None)
        for k in [k for k in h if k.lower() == self._key_header.lower()]:
            del h[k]
        h[self._key_header] = self._key
        return h

    async def _route(self, request: web.Request) -> web.StreamResponse:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self._ws(request)
        return await self._http(request)

    # ------------------------------------------------------------ WebSocket

    async def _ws(self, request: web.Request) -> web.StreamResponse:
        assert self._session is not None
        t_open = self._now()
        conn = WsConn(path=request.path_qs, t_open=t_open, handshake_s=0.0)
        self.cassette.ws.append(conn)
        start = time.monotonic()
        headers = self._headers(request)
        headers.pop("Sec-WebSocket-Key", None)
        headers = {k: v for k, v in headers.items() if not k.lower().startswith("sec-websocket")}
        try:
            up = await self._session.ws_connect(
                self.upstream_ws + request.path_qs,
                headers=headers,
                max_msg_size=0,
                autoping=True,
            )
        except aiohttp.WSServerHandshakeError as e:
            conn.handshake_s = time.monotonic() - start
            conn.status = e.status
            return web.Response(status=e.status, text=e.message or "")
        conn.handshake_s = time.monotonic() - start
        down = web.WebSocketResponse(max_msg_size=0, autoping=True)
        await down.prepare(request)
        opened = time.monotonic()

        def log(d: str, data: str) -> None:
            conn.events.append(WsEvent(time.monotonic() - opened, d, data))

        async def client_to_provider() -> None:
            async for msg in down:
                if msg.type == WSMsgType.TEXT:
                    log("c", msg.data)
                    await up.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    raise RuntimeError("binary client frames are not recorded")
            if conn.close is None:
                conn.close = (time.monotonic() - opened, "c", down.close_code or 1000)
            await up.close()

        async def provider_to_client() -> None:
            async for msg in up:
                if msg.type == WSMsgType.TEXT:
                    log("s", msg.data)
                    await down.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    raise RuntimeError("binary provider frames are not recorded")
            if conn.close is None:
                conn.close = (time.monotonic() - opened, "s", up.close_code or 1000)
            await down.close(code=up.close_code or 1000)

        tasks = [
            asyncio.ensure_future(client_to_provider()),
            asyncio.ensure_future(provider_to_client()),
        ]
        self._tasks.update(tasks)
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            # Let the other direction finish its close handshake.
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(asyncio.gather(*pending), 2.0)
            for t in done:
                if t.exception() is not None:
                    raise t.exception()  # type: ignore[misc]
        finally:
            for t in tasks:
                t.cancel()
                self._tasks.discard(t)
            await up.close()
        return down

    # ------------------------------------------------------------ HTTP

    async def _http(self, request: web.Request) -> web.StreamResponse:
        assert self._session is not None
        raw = await request.read()
        try:
            import json

            body: Any = json.loads(raw) if raw else None
        except ValueError:
            body = raw.decode(errors="replace")
        ex = HttpExchange(
            method=request.method,
            path=request.path,
            query=list(request.query.items()),
            body=body,
            t=self._now(),
        )
        self.cassette.http.append(ex)
        start = time.monotonic()
        async with self._session.request(
            request.method,
            self.upstream_http + request.path_qs,
            data=raw,
            headers=self._headers(request),
        ) as up:
            ex.status = up.status
            ex.content_type = up.headers.get("Content-Type", "")
            resp = web.StreamResponse(status=up.status, headers={"Content-Type": ex.content_type})
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                ex.chunks.append((time.monotonic() - start, chunk))
                await resp.write(chunk)
            await resp.write_eof()
            return resp
