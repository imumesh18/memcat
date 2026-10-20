"""What Memcat does with real ElevenLabs traffic, measured from the recordings.

Replays every recording in ``tests/fixtures/elevenlabs/`` (or
``$MEMCAT_FIXTURES_DIR/elevenlabs/``; make them with
``MEMCAT_RECORD=1 uv run pytest tests/test_replay_elevenlabs.py``) through Pipecat's real
ElevenLabs services at the provider's recorded speed, then:

* transcribes what the listener received with a local speech recognizer
  (faster-whisper) and scores it against the text that was sent, so a dropped,
  doubled or misordered clause shows up as word errors;
* reports time to first audio: ElevenLabs' own (from the recordings) and what
  the listener got with and without Memcat;
* writes each result as a WAV under ``bench/listen/elevenlabs/`` (gitignored).

No key and no API calls.

    uv run --group asr python bench/elevenlabs_replay.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import wave
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.replay import Cassette, Player, env  # noqa: E402
from tests.replay.scenarios import AFTER, MIXED, Target, run  # noqa: E402

FIXTURES = env.fixtures_dir("elevenlabs")
LISTEN = ROOT / "bench" / "listen" / "elevenlabs"
RESULTS = ROOT / "bench" / "results" / "elevenlabs_replay.json"

# The text the listener should hear in each scenario's last run.
EXPECTED = {
    "plain_ws": MIXED,
    "plain_http": MIXED,
    "memcat_ws_mixed": MIXED,
    "memcat_http_mixed": MIXED,
    "memcat_ws_render": MIXED,
    "memcat_ws_two_turns": MIXED + " " + AFTER,
}


def words(text: str) -> list[str]:
    """Lower-case words, digits and codes dropped: recognizers write numbers and
    spelled codes in too many ways to compare, and those clauses are always live."""
    out = []
    for w in re.findall(r"[a-z0-9']+", text.lower().replace("—", " ")):
        if not re.search(r"\d", w) and w not in {"zx", "qp"}:
            out.append(w)
    return out


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


def provider_latency(cas: Cassette) -> list[float]:
    """ElevenLabs' time from a stream's text to its first audio, per stream."""
    out = []
    for c in cas.ws:
        sent: dict[str, float] = {}
        first: dict[str, float] = {}
        for e in c.events:
            m = e.json()
            if e.dir == "c" and str(m.get("text", "")).strip():
                sent.setdefault(m["context_id"], e.t)
            elif e.dir == "s" and m.get("audio"):
                first.setdefault(m["contextId"], e.t)
        out += [first[k] - t for k, t in sent.items() if k in first]
    for h in cas.http:
        if h.chunks:
            out.append(h.chunks[0][0])
    return out


def write_wav(path: Path, pcm: bytes, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


async def replay(path: Path) -> tuple[Cassette, list, list[str]]:
    cas = Cassette.load(path)
    async with Player(cas, speed=1.0) as p, aiohttp.ClientSession() as s:
        runs = await run(
            cas.meta["scenario"], Target(p, cas.meta["voice"], cas.meta["sample_rate"], s)
        )
    return cas, runs, p.problems


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))] * 1000, 1) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small.en", help="faster-whisper model")
    args = ap.parse_args()
    if not any(FIXTURES.glob("*.take*.jsonl.gz")):
        sys.exit(
            f"no recordings in {FIXTURES}: record them with a key first\n"
            "    MEMCAT_RECORD=1 uv run pytest tests/test_replay_elevenlabs.py"
        )
    from faster_whisper import WhisperModel

    asr = WhisperModel(args.model, device="cpu", compute_type="int8")
    rows = []
    latency: dict[str, list[float]] = {"provider": [], "plain": [], "memcat_hit_opener": []}
    for path in sorted(FIXTURES.glob("*.take*.jsonl.gz")):
        cas, runs, problems = asyncio.run(replay(path))
        name, take = cas.meta["scenario"], cas.meta["take"]
        sr = cas.meta["sample_rate"]
        last = runs[-1]
        latency["provider"] += provider_latency(cas)
        if name in ("plain_ws", "plain_http") and last.ttfa:
            latency["plain"].append(last.ttfa[0])
        if name in ("memcat_ws_mixed", "memcat_http_mixed", "memcat_ws_two_turns") and last.ttfa:
            latency["memcat_hit_opener"].append(last.ttfa[0])
        write_wav(LISTEN / f"{name}.take{take}.wav", last.pcm, sr)
        row = {
            "scenario": name,
            "take": take,
            "replay_problems": len(problems),
            "audio_s": round(len(last.pcm) / 2 / sr, 2),
            "ttfa_ms": [round(x * 1000, 1) for x in last.ttfa],
            "hits": last.hits,
        }
        if name in EXPECTED:
            audio = np.frombuffer(last.pcm, np.int16).astype(np.float32) / 32768.0
            if sr != 16000:
                raise SystemExit("recordings are expected at 16 kHz")
            segs, _ = asr.transcribe(audio, language="en", beam_size=5, vad_filter=False)
            heard = " ".join(s.text for s in segs).strip()
            row["heard"] = heard
            row["wer"] = round(wer(words(EXPECTED[name]), words(heard)), 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = {
        "provider": "elevenlabs",
        "model": "eleven_flash_v2_5",
        "asr": args.model,
        "recordings": len(rows),
        "time_to_first_audio_ms": {
            k: {"n": len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9), "max": pct(v, 1.0)}
            for k, v in latency.items()
        },
        "wer": {
            "plain": statistics.mean(
                r["wer"] for r in rows if "wer" in r and r["scenario"].startswith("plain")
            ),
            "memcat": statistics.mean(
                r["wer"] for r in rows if "wer" in r and r["scenario"].startswith("memcat")
            ),
        },
        "runs": rows,
    }
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("time_to_first_audio_ms", "wer")}, indent=2))


if __name__ == "__main__":
    main()
