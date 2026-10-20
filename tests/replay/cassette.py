"""The recording file: one gzipped JSON line per connection or request.

Line kinds:

``meta``  provider, scenario, take, voice, model, sample rate, versions, date.
``ws``    one WebSocket connection: ``path`` (with query), ``t_open`` (seconds
          since the recording started), ``handshake_s`` (time the provider took to
          accept), ``events`` (``[t, dir, data]``: seconds since open, ``"c"`` client
          or ``"s"`` server, the text message) and ``close`` (``[t, by, code]``).
``http``  one request: ``method``, ``path``, ``query``, ``body`` (JSON), ``t``,
          ``status``, ``content_type`` and ``chunks`` (``[dt, base64]``, seconds since
          the request arrived, raw bytes as the provider sent them).

No headers are stored, so no key can be.
"""

from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WsEvent:
    t: float
    dir: str  # "c" client -> provider, "s" provider -> client
    data: str

    def json(self) -> dict[str, Any]:
        try:
            v = json.loads(self.data)
        except ValueError:
            return {}
        return v if isinstance(v, dict) else {}


@dataclass
class WsConn:
    path: str
    t_open: float
    handshake_s: float
    events: list[WsEvent] = field(default_factory=list)
    close: tuple[float, str, int] | None = None
    status: int = 101  # a refused upgrade keeps the provider's HTTP status


@dataclass
class HttpExchange:
    method: str
    path: str
    query: list[tuple[str, str]]
    body: Any
    t: float
    status: int = 0
    content_type: str = ""
    chunks: list[tuple[float, bytes]] = field(default_factory=list)


@dataclass
class Cassette:
    meta: dict[str, Any] = field(default_factory=dict)
    ws: list[WsConn] = field(default_factory=list)
    http: list[HttpExchange] = field(default_factory=list)

    # ------------------------------------------------------------- views

    def server_audio(self, conn: int = 0, key: str = "audio") -> bytes:
        """All audio the provider sent on one WebSocket connection, in order."""
        out = bytearray()
        for e in self.ws[conn].events:
            if e.dir == "s" and (a := e.json().get(key)):
                out += base64.b64decode(a)
        return bytes(out)

    # ------------------------------------------------------------- file

    def dumps(self) -> bytes:
        lines = [{"kind": "meta", **self.meta}]
        for c in self.ws:
            lines.append(
                {
                    "kind": "ws",
                    "path": c.path,
                    "t_open": round(c.t_open, 4),
                    "handshake_s": round(c.handshake_s, 4),
                    "status": c.status,
                    "events": [[round(e.t, 4), e.dir, e.data] for e in c.events],
                    "close": list(c.close) if c.close else None,
                }
            )
        for h in self.http:
            lines.append(
                {
                    "kind": "http",
                    "method": h.method,
                    "path": h.path,
                    "query": h.query,
                    "body": h.body,
                    "t": round(h.t, 4),
                    "status": h.status,
                    "content_type": h.content_type,
                    "chunks": [[round(dt, 4), base64.b64encode(b).decode()] for dt, b in h.chunks],
                }
            )
        text = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n"
        return gzip.compress(text.encode(), mtime=0)

    def save(self, path: Path, *, secrets: tuple[str, ...] = ()) -> None:
        blob = self.dumps()
        text = gzip.decompress(blob).decode()
        for s in secrets:
            if s and s in text:
                raise RuntimeError("refusing to save a recording that contains a secret")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)

    @classmethod
    def load(cls, path: Path) -> Cassette:
        cas = cls()
        for line in gzip.decompress(path.read_bytes()).decode().splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            kind = x.pop("kind")
            if kind == "meta":
                cas.meta = x
            elif kind == "ws":
                cas.ws.append(
                    WsConn(
                        path=x["path"],
                        t_open=x["t_open"],
                        handshake_s=x["handshake_s"],
                        status=x.get("status", 101),
                        events=[WsEvent(t, d, data) for t, d, data in x["events"]],
                        close=tuple(x["close"]) if x.get("close") else None,  # type: ignore[arg-type]
                    )
                )
            elif kind == "http":
                cas.http.append(
                    HttpExchange(
                        method=x["method"],
                        path=x["path"],
                        query=[tuple(q) for q in x["query"]],  # type: ignore[misc]
                        body=x["body"],
                        t=x["t"],
                        status=x["status"],
                        content_type=x["content_type"],
                        chunks=[(dt, base64.b64decode(b)) for dt, b in x["chunks"]],
                    )
                )
        return cas
