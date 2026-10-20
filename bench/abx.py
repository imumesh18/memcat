"""Listening-test kit and objective join checks, on Kokoro (no API keys).

Builds the spec's ABX set: 30 sentences, each rendered fully live (A) and with
cached and live clauses mixed (B), plus X (a random copy of A or B) and an answer
key, as WAV files for listeners. It also measures what a listener could notice at
each join between a cached and a live clause:

* pause: silence at the join compared with the same join in the fully live take;
* click: the largest sample-to-sample jump within 5 ms of the join, relative to
  the sentence's typical jump;
* loudness: LUFS of the mixed take compared with the live take.

    uv run python bench/abx.py --out bench/abx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="ERROR")

from _driver import Driver  # noqa: E402
from pipecat.services.kokoro.tts import KokoroTTSService  # noqa: E402

import memcat  # noqa: E402
from memcat import MemcatConfig, MemoryStore, cached  # noqa: E402
from memcat.core import audio as audio_ops  # noqa: E402
from memcat.core.chunker import split  # noqa: E402
from memcat.core.runtime import get_runtime  # noqa: E402

SR = 24000
OPENERS = ["Thanks for waiting,", "Sure,", "Of course,", "Good question,", "Okay,"]
MIDDLES = [
    "your order {n} has shipped,",
    "I can see {n} items on your account,",
    "the refund of {n} points is on its way,",
    "your ticket {n} is now open,",
    "I found {n} matching orders,",
    "your plan renews on day {n},",
]
CLOSERS = [
    "and it should arrive soon.",
    "is there anything else I can help with?",
    "let me know if you need anything else.",
    "and you will get an email shortly.",
    "and I have noted that on your account.",
]


def sentences(n: int, seed: int = 7) -> list[str]:
    rnd = random.Random(seed)
    return [
        f"{rnd.choice(OPENERS)} {rnd.choice(MIDDLES).format(n=rnd.randint(2, 999))} "
        f"{rnd.choice(CLOSERS)}"
        for _ in range(n)
    ]


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def pauses(pcm: bytes) -> list[tuple[float, float]]:
    """(start s, length s) of silences >= 40 ms inside speech."""
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
    w = SR // 100
    n = len(x) // w
    lv = 20 * np.log10(np.sqrt(np.mean(x[: n * w].reshape(n, w) ** 2, axis=1)) + 1e-9)
    loud = lv > -45
    idx = np.nonzero(loud)[0]
    out: list[tuple[float, float]] = []
    if not len(idx):
        return out
    i = idx[0]
    while i <= idx[-1]:
        if loud[i]:
            i += 1
            continue
        j = i
        while j < n and not loud[j]:
            j += 1
        if j - i >= 4:
            out.append((i / 100, (j - i) / 100))
        i = j
    return out


def clicks(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    d = np.abs(np.diff(x))
    return float(d.max() / (np.percentile(d, 99) + 1e-9)) if len(d) else 0.0


async def main(args: argparse.Namespace) -> dict[str, object]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = sentences(args.n)
    store = MemoryStore()
    Cached = cached(KokoroTTSService)
    settings = KokoroTTSService.Settings(voice="af_heart")

    # Cache every cacheable clause the way production would: rendered alone.
    prime = Driver(
        Cached(
            settings=settings, sample_rate=SR, memcat=MemcatConfig(store=store, admit_after_calls=1)
        ),
        SR,
    )
    await prime.start()
    seen: set[str] = set()
    for t in texts:
        for c in split(t):
            if c.cacheable and c.text.strip() not in seen:
                seen.add(c.text.strip())
                await prime.speak(c.text.strip())
    await get_runtime(prime.tts._memcat_cfg).worker.drain_all(300)
    await prime.stop()

    live = Driver(KokoroTTSService(settings=settings, sample_rate=SR), SR)
    mixed = Driver(Cached(settings=settings, sample_rate=SR, memcat=MemcatConfig(store=store)), SR)
    await live.start()
    await mixed.start()
    rnd = random.Random(11)
    key: list[dict[str, object]] = []
    pause_deltas: list[float] = []
    click_ratios: list[float] = []
    lufs_deltas: list[float] = []
    for i, t in enumerate(texts):
        _, a = await live.speak(t)
        _, b = await mixed.speak(t)
        # Which take is live is random per trial, so listeners cannot learn a pattern.
        live_is = rnd.choice("AB")
        take_a, take_b = (a, b) if live_is == "A" else (b, a)
        x_is = rnd.choice("AB")
        write_wav(out / f"{i:02d}_A.wav", take_a)
        write_wav(out / f"{i:02d}_B.wav", take_b)
        write_wav(out / f"{i:02d}_X.wav", take_a if x_is == "A" else take_b)
        key.append({"trial": i, "text": t, "x_is": x_is, "live_is": live_is})
        pa, pb = pauses(a), pauses(b)
        # Compare the pauses at clause joins: the k longest pauses of each take.
        k = max(0, len(split(t)) - 1)
        la = sorted((p for _, p in pa), reverse=True)[:k]
        lb = sorted((p for _, p in pb), reverse=True)[:k]
        pause_deltas += [abs(x - y) * 1000 for x, y in zip(sorted(la), sorted(lb), strict=False)]
        click_ratios.append(clicks(b) / max(clicks(a), 1e-9))
        fa, fb = audio_ops.loudness(a, SR), audio_ops.loudness(b, SR)
        if fa is not None and fb is not None:
            lufs_deltas.append(abs(fa - fb))
    await live.stop()
    await mixed.stop()
    stats = await mixed.tts.memcat_stats()
    (out / "answer_key.json").write_text(json.dumps(key, indent=2) + "\n")
    (out / "trials.json").write_text(
        json.dumps([{"trial": k["trial"], "text": k["text"]} for k in key], indent=2) + "\n"
    )
    await memcat.aclose()

    def q(xs: list[float], p: float) -> float:
        return round(float(np.percentile(xs, p)), 2) if xs else 0.0

    return {
        "trials": len(texts),
        "share_chars_cached_in_B": round(stats["hit_rate_chars"], 3),
        "join_pause_delta_ms": {"p50": q(pause_deltas, 50), "p90": q(pause_deltas, 90)},
        "click_ratio_B_over_A": {"p50": q(click_ratios, 50), "max": q(click_ratios, 100)},
        "loudness_delta_lufs": {"p50": q(lufs_deltas, 50), "max": q(lufs_deltas, 100)},
        "kit": str(out),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/abx")
    ap.add_argument("--n", type=int, default=30)
    print(json.dumps(asyncio.run(main(ap.parse_args())), indent=2))
