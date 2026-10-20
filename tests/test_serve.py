from __future__ import annotations

from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from memcat import MemcatConfig, MemoryStore, cached
from memcat.serve import Server
from tests.fakes import FakeHTTPTTS, FakeWSTTS, decode, words_of

S = "Thank you for calling ABC Shop, how can I help you today?"


def _server(cls, secret=None) -> tuple[Server, list[Any]]:
    store = MemoryStore()
    made: list[Any] = []

    def make(rate: int):
        svc = cached(cls)(memcat=MemcatConfig(store=store, admit_after_calls=2), sample_rate=rate)
        made.append(svc)
        return svc

    return Server(make, pool_size=2, secret=secret), made


def _vapi(text: str, rate: int = 16000, call: str = "c1") -> dict[str, Any]:
    return {
        "message": {
            "type": "voice-request",
            "text": text,
            "sampleRate": rate,
            "timestamp": 0,
            "call": {"id": call},
        }
    }


@pytest.mark.parametrize("cls", [FakeHTTPTTS, FakeWSTTS])
async def test_vapi_streams_pcm_and_caches_across_calls(cls) -> None:
    server, made = _server(cls)
    async with TestClient(TestServer(server.app())) as client:
        for call in ("call-a", "call-b", "call-c"):
            r = await client.post("/vapi", json=_vapi(S, call=call))
            assert r.status == 200 and r.headers["Content-Type"] == "application/octet-stream"
            pcm = await r.read()
            assert decode(pcm, 16000) == words_of(S)
            from memcat.core.runtime import get_runtime

            await get_runtime(made[0]._memcat_cfg).worker.drain_all(10)
        live = sum(len(m.requests) for m in made)
        assert live == 2  # admitted after 2 distinct calls; the third came from the cache
        health = await (await client.get("/healthz")).json()
        assert health["ok"] and health["served"] == 3


async def test_vapi_auth_and_validation() -> None:
    server, _ = _server(FakeHTTPTTS, secret="s3cret")
    async with TestClient(TestServer(server.app())) as client:
        assert (await client.post("/vapi", json=_vapi(S))).status == 401
        h = {"X-VAPI-SECRET": "s3cret"}
        assert (await client.post("/vapi", json=_vapi(S, rate=11025), headers=h)).status == 400
        assert (
            await client.post("/vapi", json={"message": {"type": "x"}}, headers=h)
        ).status == 400
        assert (await client.post("/vapi", json=_vapi("   "), headers=h)).status == 400
        r = await client.post("/v1/tts", json={"text": S, "sample_rate": 16000}, headers=h)
        assert r.status == 200 and decode(await r.read(), 16000) == words_of(S)


def test_cli_builds_server(tmp_path) -> None:
    import argparse

    from memcat.cli import _make_server, load_config

    p = tmp_path / "memcat.toml"
    p.write_text(
        '[service]\nclass = "tests.fakes.FakeHTTPTTS"\nsample_rate = 16000\n'
        '[serve]\npool_size = 3\nsecret = "abc"\n'
    )
    srv = _make_server(load_config(p), argparse.Namespace(pool_size=None))
    assert isinstance(srv, Server) and srv._pool_size == 3 and srv._secret == "abc"
    assert cached(FakeHTTPTTS)
