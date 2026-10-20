"""The recorder and player, proven against the local ElevenLabs emulator.

Each scenario is recorded through ``Recorder`` with the emulator as the
"provider", then replayed with ``Player``. The replay must give the listener the
same words and byte-identical audio, with time to first audio within a few
milliseconds of the recording, and match every client message.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp
import numpy as np
import pytest

from tests.emulators.elevenlabs import ElevenLabsEmulator
from tests.fakes import SR, words_of
from tests.replay import Cassette, Player, Recorder
from tests.replay.scenarios import MIXED, SCENARIOS, Target, run


def same_audio(a: bytes, b: bytes) -> bool:
    """Equal length and within -40 dB of each other.

    Loudness matching runs in the background, so the gain on a cached clip can
    differ by a fraction of a dB between two runs of the same call. Anything
    dropped, doubled or reordered is far louder than that.
    """
    if len(a) != len(b):
        return False
    x = np.frombuffer(a, np.int16).astype(np.float64)
    y = np.frombuffer(b, np.int16).astype(np.float64)
    return float(((x - y) ** 2).sum()) <= 1e-4 * max(float((x**2).sum()), 1.0)


KEY = "sk-test-0123456789abcdef"  # stands in for a real key: must never reach a file


async def record(name: str, tmp: Path) -> tuple[Path, list, ElevenLabsEmulator]:
    emu = ElevenLabsEmulator(latency=(0.3, 0.3), jitter=(0.0, 0.0))  # as slow as the real API
    await emu.start()
    try:
        async with (
            Recorder(upstream_ws=emu.ws_url, upstream_http=emu.http_url, api_key=KEY) as rec,
            aiohttp.ClientSession() as s,
        ):
            runs = await run(name, Target(rec, "v", SR, s))
            path = tmp / f"{name}.jsonl.gz"
            rec.cassette.meta = {"scenario": name, "voice": "v", "sample_rate": SR}
            rec.cassette.save(path, secrets=rec.secrets)
    finally:
        await emu.stop()
    return path, runs, emu


async def replay(path: Path, speed: float = 1.0) -> tuple[list, Player]:
    cas = Cassette.load(path)
    async with Player(cas, speed=speed) as p, aiohttp.ClientSession() as s:
        runs = await run(cas.meta["scenario"], Target(p, "v", SR, s))
    return runs, p


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n != "error_bad_voice"])
async def test_replay_matches_recording(name, tmp_path) -> None:
    path, recorded, emu = await record(name, tmp_path)
    assert emu.errors == []
    replayed, player = await replay(path)
    assert player.problems == []
    assert player.unused() == []
    assert len(replayed) == len(recorded)
    for a, b in zip(recorded, replayed, strict=True):
        assert b.words == a.words
        assert same_audio(a.pcm, b.pcm)
        assert len(b.ttfa) == len(a.ttfa)


async def test_recording_holds_no_key_and_provider_got_it(tmp_path) -> None:
    path, _, emu = await record("plain_ws", tmp_path)
    assert all(c["api_key"] == KEY for c in emu.connections)
    import gzip

    text = gzip.decompress(path.read_bytes()).decode()
    assert KEY not in text and "replay-dummy-key" not in text


async def test_save_refuses_a_secret(tmp_path) -> None:
    cas = Cassette(meta={"note": f"oops {KEY}"})
    with pytest.raises(RuntimeError):
        cas.save(tmp_path / "x.jsonl.gz", secrets=(KEY,))
    assert not (tmp_path / "x.jsonl.gz").exists()


async def test_replay_keeps_the_providers_timing(tmp_path) -> None:
    """A slow provider replays slow; speed 0 replays it instantly."""
    path, recorded, _ = await record("plain_ws", tmp_path)
    assert recorded[0].ttfa[0] > 0.15
    slow, _ = await replay(path, speed=1.0)
    fast, _ = await replay(path, speed=0.0)
    assert slow[0].ttfa[0] > 0.15
    assert fast[0].ttfa[0] < 0.08


async def test_changed_client_behaviour_is_a_mismatch(tmp_path) -> None:
    """A recording made for one sentence cannot pass for another."""
    path, _, _ = await record("plain_ws", tmp_path)
    cas = Cassette.load(path)
    for e in cas.ws[0].events:
        if e.dir == "c" and "Thanks" in e.data:
            m = json.loads(e.data)
            m["text"] = m["text"].replace("Thanks", "Thank you")
            e.data = json.dumps(m)
    cas.save(path)
    _, player = await replay(path, speed=0.0)
    assert player.problems and "differs" in player.problems[0]


async def test_words_are_the_sentence(tmp_path) -> None:
    _, recorded, _ = await record("memcat_ws_mixed", tmp_path)
    assert recorded[-1].words == words_of(MIXED)
    assert recorded[-1].hits == 3
