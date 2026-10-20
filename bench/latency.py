"""Time to first audio with Kokoro and a local Redis: hits, misses, and no Memcat.

uv run python bench/latency.py --redis redis://localhost:6379/0 --n 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

from _driver import Driver  # noqa: E402
from pipecat.services.kokoro.tts import KokoroTTSService  # noqa: E402

import memcat  # noqa: E402
from memcat import MemcatConfig, MemoryStore, RedisStore, cached  # noqa: E402
from memcat.core.runtime import get_runtime  # noqa: E402

PHRASES = [
    "Thanks for your patience,",
    "Thank you for calling, how can I help you today?",
    "Is there anything else I can help you with?",
    "Let me check that for you, one moment please.",
    "I can help you with that.",
    "Thanks for your patience while I look into this.",
    "Could you please confirm your full name?",
]


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def summary(xs: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(statistics.median(xs), 2),
        "p99_ms": round(pct(xs, 0.99), 2),
        "n": len(xs),
    }


async def main(args: argparse.Namespace) -> dict[str, object]:
    sr = args.sample_rate
    store = RedisStore(args.redis) if args.redis else MemoryStore()
    ns = "bench-latency"
    cfg = lambda **kw: MemcatConfig(store=store, namespace=ns, admit_after_calls=1, **kw)  # noqa: E731
    res: dict[str, object] = {"sample_rate": sr, "store": "redis" if args.redis else "memory"}

    plain = Driver(
        KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart"), sample_rate=sr), sr
    )
    await plain.start()
    base = [(await plain.speak(PHRASES[i % len(PHRASES)] + f" Ref {i}."))[0] for i in range(args.n)]
    await plain.stop()
    res["no_memcat"] = summary(base)

    Cached = cached(KokoroTTSService)
    warm = Driver(
        Cached(settings=KokoroTTSService.Settings(voice="af_heart"), sample_rate=sr, memcat=cfg()),
        sr,
    )
    await warm.start()
    await warm.tts.memcat_purge()
    for p in PHRASES:
        await warm.speak(p)
    await get_runtime(warm.tts._memcat_cfg).worker.drain_all(120)
    await warm.stop()

    # A fresh service and runtime: L2 (Redis or memory) on first use, then L1.
    await memcat.aclose()
    d = Driver(
        Cached(settings=KokoroTTSService.Settings(voice="af_heart"), sample_rate=sr, memcat=cfg()),
        sr,
    )
    await d.start()
    rt = get_runtime(d.tts._memcat_cfg)
    for _ in range(200):
        if rt.index.loaded:
            break
        await asyncio.sleep(0.01)
    l2 = [(await d.speak(p))[0] for p in PHRASES]
    hits = [(await d.speak(PHRASES[i % len(PHRASES)]))[0] for i in range(args.n)]
    # A cached opening clause, then a live one: the listener hears the cached clause
    # while the provider synthesizes the rest.
    misses = [
        (await d.speak(f"Thanks for your patience, order {i} is ready."))[0] for i in range(args.n)
    ]
    # Misses on a sentence with no cached clause at all, to compare with no Memcat.
    cold = [(await d.speak(f"Your reference number is {i}."))[0] for i in range(args.n)]
    base2 = []
    await d.stop()
    plain2 = Driver(
        KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart"), sample_rate=sr), sr
    )
    await plain2.start()
    base2 = [(await plain2.speak(f"Your reference number is {i}."))[0] for i in range(args.n)]
    await plain2.stop()
    res["hit_first_use_l2"] = summary(l2)
    res["hit_l1"] = summary(hits)
    res["cached_opener_then_live"] = summary(misses)
    res["miss_no_cacheable_clause"] = summary(cold)
    res["same_sentences_no_memcat"] = summary(base2)
    res["stats"] = await d.tts.memcat_stats()
    await memcat.aclose()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    out = asyncio.run(main(a))
    text = json.dumps(out, indent=2, default=str)
    print(text)
    if a.json:
        a.json.write_text(text + "\n")
