"""``memcat serve`` is safe to expose: authenticated, bounded, and not billable by strangers."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from aiohttp import ClientTimeout
from aiohttp.test_utils import TestClient, TestServer

from memcat import MemcatConfig, MemoryStore, cached
from memcat.cli import main
from memcat.serve import Server, Speaker
from tests.fakes import FakeHTTPTTS, decode, words_of
from tests.test_serve_isolation import LONG, SHORT, SlowTTS

S = "Thank you for calling ABC Shop, how can I help you today?"


def _server(cls: type = FakeHTTPTTS, **kw: Any) -> tuple[Server, list[Any]]:
    store = MemoryStore()
    made: list[Any] = []

    def make(rate: int) -> Any:
        svc = cached(cls)(memcat=MemcatConfig(store=store, admit_after_calls=2), sample_rate=rate)
        made.append(svc)
        return svc

    return Server(make, **{"pool_size": 1, **kw}), made


def _texts(made: list[Any]) -> list[str]:
    return [t for m in made for t in m.requests]


async def _speaking(server: Server, rate: int = 16000) -> None:
    """Wait until a speaker for ``rate`` is busy with a request."""
    async with asyncio.timeout(10):
        while not any(sp.lock.locked() for sp in server._pools.get(rate, [])):
            await asyncio.sleep(0.01)


async def test_text_length_is_capped() -> None:
    server, made = _server(max_text_chars=20)
    async with TestClient(TestServer(server.app())) as client:
        r = await client.post("/v1/tts", json={"text": "x" * 21, "sample_rate": 16000})
        assert r.status == 413 and (await r.json())["maxChars"] == 20
        r = await client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000})
        assert r.status == 200 and decode(await r.read()) == words_of(SHORT)
    assert _texts(made) == [SHORT]


async def test_healthz_shows_counters_only_with_the_secret() -> None:
    server, _ = _server(secret="s3cret")
    async with TestClient(TestServer(server.app())) as client:
        h = {"X-VAPI-SECRET": "s3cret"}
        r = await client.post("/v1/tts", json={"text": S}, headers=h)
        assert r.status == 200
        await r.read()  # "served" counts a request once its audio has been sent
        assert await (await client.get("/healthz")).json() == {"ok": True}
        wrong = {"X-VAPI-SECRET": "nope"}
        assert await (await client.get("/healthz", headers=wrong)).json() == {"ok": True}
        full = await (await client.get("/healthz", headers=h)).json()
        assert full["served"] == 1 and "cache" in full


async def test_requests_without_a_call_id_cannot_force_admission() -> None:
    from memcat.core.runtime import get_runtime

    server, made = _server()
    async with TestClient(TestServer(server.app())) as client:
        for _ in range(3):  # admit_after_calls=2, but these are one anonymous call
            r = await client.post("/v1/tts", json={"text": S, "sample_rate": 16000})
            assert r.status == 200 and decode(await r.read()) == words_of(S)
            vapi = {"message": {"type": "voice-request", "text": S, "sampleRate": 16000}}
            r = await client.post("/vapi", json=vapi)
            assert r.status == 200 and decode(await r.read()) == words_of(S)
            await get_runtime(made[0]._memcat_cfg).worker.drain_all(10)
        assert len(_texts(made)) == 6  # every one was live: nothing was admitted
        # Two real calls still admit the line as usual.
        for call in ("a", "b", "c"):
            r = await client.post("/v1/tts", json={"text": S, "call_id": call})
            assert r.status == 200
            await get_runtime(made[0]._memcat_cfg).worker.drain_all(10)
        assert len(_texts(made)) == 8


async def test_full_queue_is_a_fast_503() -> None:
    server, _ = _server(SlowTTS, max_waiting=0)
    async with TestClient(TestServer(server.app())) as client:
        first = asyncio.ensure_future(
            client.post("/v1/tts", json={"text": LONG, "sample_rate": 16000})
        )
        await _speaking(server)
        r = await asyncio.wait_for(
            client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000}), 2
        )
        assert r.status == 503 and (await r.json())["error"] == "busy"
        assert r.headers["Retry-After"] == "1"
        r = await first
        assert r.status == 200 and await r.read()  # (the fake's tone vocabulary is shared)
        r = await client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000})  # room again
        assert r.status == 200


async def test_waiting_too_long_for_a_speaker_is_503_and_never_synthesized() -> None:
    server, made = _server(SlowTTS, queue_timeout_s=0.2)
    async with TestClient(TestServer(server.app())) as client:
        first = asyncio.ensure_future(
            client.post("/v1/tts", json={"text": LONG, "sample_rate": 16000})
        )
        await _speaking(server)
        r = await client.post("/v1/tts", json={"text": SHORT, "sample_rate": 16000})
        assert r.status == 503 and (await r.json())["error"] == "busy"
        assert (await first).status == 200
        await asyncio.sleep(0.2)
    assert SHORT not in _texts(made)


class RunAppServer(TestServer):
    """Like ``web.run_app``: a client hanging up does not cancel its handler."""

    async def _make_runner(self, handler_cancellation: bool = True, **kw: Any) -> Any:
        return await super()._make_runner(handler_cancellation=False, **kw)


async def test_client_gone_before_its_turn_is_never_synthesized() -> None:
    server, made = _server(SlowTTS)
    async with TestClient(RunAppServer(server.app())) as client:
        first = asyncio.ensure_future(
            client.post("/v1/tts", json={"text": LONG, "sample_rate": 16000})
        )
        await _speaking(server)
        with pytest.raises(asyncio.TimeoutError):
            await client.post(
                "/v1/tts",
                json={"text": SHORT, "sample_rate": 16000},
                timeout=ClientTimeout(total=0.2),
            )
        r = await first
        assert r.status == 200 and await r.read()  # (the fake's tone vocabulary is shared)
        await asyncio.sleep(0.5)  # the abandoned request's turn has come and gone
        r = await client.post("/v1/tts", json={"text": "All done.", "sample_rate": 16000})
        assert r.status == 200
    assert SHORT not in _texts(made) and "All done." in _texts(made)


async def test_speaker_skips_an_abandoned_request() -> None:
    from tests.test_serve_isolation import _make

    sp = Speaker(_make(FakeHTTPTTS), 16000)
    await sp.start()
    try:
        with pytest.raises(ConnectionResetError):
            await sp.speak(SHORT, "c1", lambda _b: None, abandoned=lambda: True)
        assert sp.service.requests == [] and not sp.lock.locked()
    finally:
        await sp.stop()


async def test_web_pages_cannot_spend_credits_on_a_local_server() -> None:
    server, made = _server()
    body = '{"text": "Hello from a web page.", "sample_rate": 16000}'
    async with TestClient(TestServer(server.app())) as client:
        # A cross-site "simple" request: no preflight, so the browser sends it as is.
        r = await client.post("/v1/tts", data=body, headers={"Content-Type": "text/plain"})
        assert r.status == 415
        page = {"Origin": "https://evil.example"}
        r = await client.post("/v1/tts", json={"text": S}, headers=page)
        assert r.status == 403
        rebound = {"Host": "evil.example:8080"}  # DNS rebinding: a page's own name
        r = await client.post("/v1/tts", json={"text": S}, headers=rebound)
        assert r.status == 403
        assert (await client.get("/healthz", headers=rebound)).status == 403
        for host in ("localhost:8080", "127.0.0.1", "[::1]:8080"):
            r = await client.post("/v1/tts", json={"text": SHORT}, headers={"Host": host})
            assert r.status == 200, host
            await r.read()
    assert _texts(made) == [SHORT] * 3


async def test_a_secret_or_explicit_opt_out_lifts_the_local_only_rule() -> None:
    page = {"Origin": "https://app.example", "Host": "tts.example"}
    for kw, headers in (
        ({"secret": "s3cret"}, {**page, "X-VAPI-SECRET": "s3cret"}),
        ({"insecure_no_auth": True}, page),
    ):
        server, _ = _server(**kw)
        async with TestClient(TestServer(server.app())) as client:
            r = await client.post("/v1/tts", json={"text": SHORT}, headers=headers)
            assert r.status == 200 and await r.read()
            r = await client.post("/v1/tts", data="{}", headers={**headers, "Content-Type": "x"})
            assert r.status == 415


async def test_a_client_that_stops_reading_frees_its_speaker(monkeypatch) -> None:
    import json
    import socket

    from memcat import serve

    monkeypatch.setattr(serve, "REQUEST_TIMEOUT_S", 1.0)
    server, _ = _server(max_text_chars=100_000)
    ts = TestServer(server.app())
    await ts.start_server()
    text = " ".join(f"word{i}" for i in range(6000)) + "."
    body = json.dumps({"text": text, "sample_rate": 48000}).encode()
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.setblocking(False)
    try:
        await asyncio.get_running_loop().sock_connect(sock, (ts.host, ts.port))
        head = b"POST /v1/tts HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        sock.sendall(head + b"Content-Length: %d\r\n\r\n" % len(body) + body)
        await _speaking(server, 48000)
        async with asyncio.timeout(10):  # never reads a byte
            while any(sp.lock.locked() or sp.load for sp in server._pools[48000]):
                await asyncio.sleep(0.05)
    finally:
        sock.close()
        await ts.close()


async def test_a_secret_must_be_a_string() -> None:
    with pytest.raises(ValueError, match="secret"):
        _server(secret=12345)


# ----------------------------------------------------------------- memcat serve (CLI)

CONF = '[service]\nclass = "tests.fakes.FakeHTTPTTS"\nsample_rate = 16000\n'


@pytest.fixture
def run_app(monkeypatch) -> list[dict[str, Any]]:
    from aiohttp import web

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(web, "run_app", lambda app, **kw: calls.append(kw))
    monkeypatch.delenv("MEMCAT_SERVE_SECRET", raising=False)
    return calls


def _serve(tmp_path, *argv: str, conf: str = CONF) -> int:
    p = tmp_path / "memcat.toml"
    p.write_text(conf)
    return main(["--config", str(p), "serve", *argv])


def test_serve_binds_loopback_by_default(tmp_path, run_app) -> None:
    assert _serve(tmp_path) == 0
    assert run_app[0]["host"] == "127.0.0.1" and run_app[0]["port"] == 8080


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "bot.example.com"])
def test_serve_refuses_a_public_host_without_a_secret(tmp_path, capsys, run_app, host) -> None:
    assert _serve(tmp_path, "--host", host) == 1
    err = capsys.readouterr().err
    assert "refusing to serve" in err and "MEMCAT_SERVE_SECRET" in err
    assert run_app == []


def test_serve_public_host_with_a_secret_or_explicit_opt_out(
    tmp_path, monkeypatch, run_app
) -> None:
    assert _serve(tmp_path, "--host", "0.0.0.0", "--insecure-no-auth") == 0
    assert _serve(tmp_path, "--host", "::1") == 0  # loopback needs no secret
    assert _serve(tmp_path, conf=CONF + '[serve]\nhost = "0.0.0.0"\nsecret = "abc"\n') == 0
    monkeypatch.setenv("MEMCAT_SERVE_SECRET", "abc")
    assert _serve(tmp_path, "--host", "0.0.0.0") == 0
    assert [c["host"] for c in run_app] == ["0.0.0.0", "::1", "0.0.0.0", "0.0.0.0"]


def test_serve_limits_come_from_config(tmp_path) -> None:
    import argparse

    from memcat.cli import _make_server, load_config

    p = tmp_path / "memcat.toml"
    p.write_text(CONF + "[serve]\nmax_text_chars = 500\nmax_waiting = 3\nqueue_timeout_s = 2.5\n")
    srv = _make_server(load_config(p), argparse.Namespace(pool_size=None))
    assert (srv._max_text_chars, srv._max_waiting, srv._queue_timeout) == (500, 3, 2.5)
    p.write_text(CONF + "[serve]\nmax_text_chars = 0\n")
    assert main(["--config", str(p), "serve"]) == 1


def test_serve_rejects_a_secret_that_is_not_a_string(tmp_path, capsys, run_app) -> None:
    assert _serve(tmp_path, conf=CONF + "[serve]\nsecret = 12345\n") == 1
    err = capsys.readouterr().err
    assert "[serve] secret must be a string" in err and "Traceback" not in err
    assert run_app == []


def test_serve_setup_errors_are_one_line(tmp_path, capsys, run_app) -> None:
    for extra in (
        '[memcat]\nnamespace = "bad ns"\n',
        '[store]\ntype = "redis"\nurl = "http://x"\n',
    ):
        assert _serve(tmp_path, conf=CONF + extra) == 1
        err = capsys.readouterr().err.strip()
        assert err.startswith("memcat: bad [memcat] or [store] option") and "\n" not in err
    assert run_app == []
