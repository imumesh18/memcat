"""Memcat with Pipecat's real ElevenLabs services, replaying recorded ElevenLabs traffic.

Normal runs replay ``tests/fixtures/elevenlabs/*.jsonl.gz`` (or
``$MEMCAT_FIXTURES_DIR/elevenlabs/``) with the provider's recorded timing and need no
key. Recordings are not committed, so a checkout without them skips these tests.

Record (or re-record) against the real API, about 2,000 characters:

    MEMCAT_RECORD=1 uv run pytest tests/test_replay_elevenlabs.py

The key comes from ``ELEVENLABS_API_KEY`` or ``.env.test`` (gitignored) and stays in
the recording proxy; Pipecat only ever sees a dummy key.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from importlib.metadata import version
from pathlib import Path

import aiohttp
import pytest

from tests.fakes import words_of
from tests.replay import Cassette, Player, Recorder, env
from tests.replay.scenarios import AFTER, MIXED, MODEL, SCENARIOS, Run, Target, run

FIXTURES = env.fixtures_dir("elevenlabs")
TAKES = 3  # each scenario is recorded this many times, for a spread of real latencies
SR = 16000
DEFAULT_VOICE = "EXAVITQu4vr4xnSDxMaL"  # "Sarah", a stock voice free accounts may use


def fixture(name: str, take: int) -> Path:
    return FIXTURES / f"{name}.take{take}.jsonl.gz"


async def _record(name: str, take: int) -> list[Run]:
    key = env.get("ELEVENLABS_API_KEY")
    if not key:
        pytest.fail("MEMCAT_RECORD=1 needs ELEVENLABS_API_KEY in the environment or .env.test")
    voice = env.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE
    meta = {
        "provider": "elevenlabs",
        "scenario": name,
        "take": take,
        "voice": voice,
        "model": MODEL,
        "sample_rate": SR,
        "pipecat": version("pipecat-ai"),
        "memcat": version("memcat"),
        "recorded": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    async with (
        Recorder(
            upstream_ws="wss://api.elevenlabs.io",
            upstream_http="https://api.elevenlabs.io",
            api_key=key,
            meta=meta,
        ) as rec,
        aiohttp.ClientSession() as s,
    ):
        runs = await run(name, Target(rec, voice, SR, s))
    rec.cassette.save(fixture(name, take), secrets=rec.secrets)
    return runs


async def _replay(name: str, take: int) -> tuple[list[Run], Player, Cassette]:
    path = fixture(name, take)
    if not path.exists():
        pytest.skip(f"no recording {path.name}: record with MEMCAT_RECORD=1")
    cas = Cassette.load(path)
    async with Player(cas, speed=env.speed()) as p, aiohttp.ClientSession() as s:
        runs = await run(name, Target(p, cas.meta["voice"], cas.meta["sample_rate"], s))
    return runs, p, cas


def _check(name: str, runs: list[Run]) -> None:
    """What the listener must get, whether recording or replaying."""
    last = runs[-1]
    if name != "error_bad_voice":
        for r in runs:
            # Real speech: Pipecat emits the words even when no audio came back.
            assert len(r.pcm) / 2 / SR >= 0.15 * len(r.words) > 0, (len(r.pcm), r.words)
    if name in ("plain_ws", "plain_http"):
        assert last.words == words_of(MIXED)
    elif name in ("memcat_ws_mixed", "memcat_http_mixed"):
        assert last.words == words_of(MIXED)
        assert last.hits == 3
        assert last.ttfa[0] < 0.05  # the opener is cached: no provider round trip
    elif name == "memcat_ws_render":
        assert runs[0].words == last.words == words_of(MIXED)
        assert last.hits >= 3
    elif name == "memcat_ws_two_turns":
        assert last.words == words_of(MIXED) + words_of(AFTER)
    elif name == "plain_ws_interrupt":
        assert last.words[-len(words_of(AFTER)) :] == words_of(AFTER)
    elif name == "memcat_ws_interrupt":
        want = words_of(AFTER)
        assert last.words[-len(want) :] == want
        assert len(last.words) < len(words_of(MIXED)) + len(want)


@pytest.mark.parametrize("take", range(TAKES))
@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_elevenlabs(name: str, take: int) -> None:
    if env.recording():
        _check(name, await _record(name, take))
        return
    runs, player, cas = await _replay(name, take)
    assert player.problems == [], player.problems
    assert player.unused() == []
    _check(name, runs)
    assert all(r.enabled for r in runs)  # no Memcat error switched it off
    if name == "plain_ws":
        # Without Memcat, the listener hears exactly what ElevenLabs sent.
        assert runs[0].pcm == cas.server_audio(0)


async def test_elevenlabs_stall_after_barge_in() -> None:
    """A real ElevenLabs stall, recorded by chance: after a barge-in the socket sent
    nothing for 3.3 s, past Pipecat's 3 s context timeout (1 run in 14). The turn
    after it is lost, as it would be without Memcat; Memcat must not hang, raise or
    switch itself off, and the cached opener before the barge-in still plays."""
    path = FIXTURES / "provider_stall.jsonl.gz"
    if not path.exists():
        pytest.skip("no stall recording (captured by chance, cannot be recorded on demand)")
    cas = Cassette.load(path)
    async with Player(cas, speed=env.speed()) as p, aiohttp.ClientSession() as s:
        t = Target(p, cas.meta["voice"], cas.meta["sample_rate"], s)
        runs = await asyncio.wait_for(run("memcat_ws_interrupt", t), 30)
    assert p.problems == [], p.problems
    last = runs[-1]
    assert last.ttfa and last.ttfa[0] < 0.05  # the cached opener played at once
    assert len(last.pcm) / 2 / SR > 0.5
    assert last.enabled
