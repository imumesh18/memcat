"""Load and soak: many concurrent calls for minutes, checking nothing drifts.

Each call builds its own wrapped service and its own ``RedisStore(url)``, as bots do
in production, and speaks several LLM turns mixing reusable lines with order
numbers, through a fake provider with realistic latency. The run checks:

* every call heard exactly its text, in order (0 wrong calls);
* event-loop lag stays low (nothing blocks the loop);
* memory, asyncio tasks and Redis connections stop growing after warm-up;
* the cache fills (admission, captures, renders) while calls run.

    uv run python bench/soak.py --redis redis://localhost:6379/0 --concurrency 100 --minutes 5
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="ERROR")

import psutil  # noqa: E402

import memcat  # noqa: E402
from memcat import MemcatConfig, MemoryStore, RedisStore, cached  # noqa: E402
from memcat.core.runtime import get_runtime  # noqa: E402
from tests.fakes import FakeHTTPTTS, FakeWSTTS, words_of  # noqa: E402
from tests.harness import hear, turn  # noqa: E402

OPENERS = [
    "Thanks for calling ABC Shop, how can I help you today?",
    "Sure, I can help you with that.",
    "Thanks for waiting, I have your account here.",
]
MIDDLES = [
    "Your order {n} has shipped, and it should arrive soon.",
    "I can see {n} items on your account, is that right?",
    "Your refund of {n} points is on its way.",
    "Ticket {n} is open, and our team will call you back.",
]
CLOSERS = [
    "Is there anything else I can help with?",
    "Have a great day.",
    "Thanks for calling, goodbye.",
]


# The fake provider gives every distinct word its own tone, and tones must stay
# under the 8 kHz Nyquist limit, so order numbers come from a fixed pool. They
# still contain digits, so they are always spoken live.
NUMBERS = [str(n) for n in (17, 42, 108, 256, 311, 499, 512, 640, 777, 901, 1024, 2048)]


def script(rnd: random.Random) -> list[list[str]]:
    turns = [[rnd.choice(OPENERS)]]
    for _ in range(rnd.randint(2, 4)):
        turns.append([rnd.choice(MIDDLES).format(n=rnd.choice(NUMBERS)) + " "])
    turns.append([rnd.choice(CLOSERS)])
    return turns


async def lag_monitor(samples: list[float], stop: asyncio.Event) -> None:
    while not stop.is_set():
        t0 = time.perf_counter()
        await asyncio.sleep(0.05)
        samples.append((time.perf_counter() - t0 - 0.05) * 1000)


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * (len(s) - 1)))], 2) if s else 0.0


async def main(args: argparse.Namespace) -> dict[str, object]:
    proc = psutil.Process()
    rnd = random.Random(7)
    ns = f"soak{int(time.time())}"
    mem_store = MemoryStore()

    def config() -> MemcatConfig:
        store = RedisStore(args.redis) if args.redis else mem_store
        return MemcatConfig(store=store, namespace=ns, admit_after_calls=3, enabled=not args.plain)

    stop = asyncio.Event()
    lag: list[float] = []
    mon = asyncio.ensure_future(lag_monitor(lag, stop))
    deadline = time.monotonic() + args.minutes * 60
    done = wrong = 0
    call_secs: list[float] = []
    timeline: list[dict[str, float]] = []
    sem = asyncio.Semaphore(args.concurrency)
    kinds = [FakeWSTTS, FakeHTTPTTS]

    async def one_call(i: int) -> None:
        nonlocal done, wrong
        async with sem:
            cls = kinds[i % len(kinds)]
            turns = script(rnd)
            tts = (
                cached(cls)(memcat=config(), delay=(0.05, 0.25))
                if cls is FakeWSTTS
                else cached(cls)(memcat=config())
            )
            frames = [f for t in turns for f in turn(*t)]
            t0 = time.perf_counter()
            h = await hear(tts, frames)
            call_secs.append(time.perf_counter() - t0)
            want = words_of(" ".join(s for t in turns for s in t))
            done += 1
            if h.words != want:
                wrong += 1

    async def sampler() -> None:
        t_start = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(5)
            gc.collect()
            rt = None
            with contextlib.suppress(Exception):
                rt = get_runtime(config())
            timeline.append(
                {
                    "t_s": round(time.monotonic() - t_start),
                    "rss_mb": round(proc.memory_info().rss / 2**20, 1),
                    "tasks": len(asyncio.all_tasks()),
                    "calls_done": done,
                    "entries": len(rt.index) if rt else 0,
                    "connections": len(proc.net_connections()) if args.redis else 0,
                }
            )

    smp = asyncio.ensure_future(sampler())
    pending: set[asyncio.Task[None]] = set()
    i = 0
    while time.monotonic() < deadline:
        while len(pending) < args.concurrency * 2 and time.monotonic() < deadline:
            t = asyncio.ensure_future(one_call(i))
            pending.add(t)
            t.add_done_callback(pending.discard)
            i += 1
        await asyncio.sleep(0.05)
    await asyncio.gather(*pending, return_exceptions=True)
    stop.set()
    await mon
    smp.cancel()
    rt = get_runtime(config())
    metrics = rt.metrics.snapshot()
    await memcat.aclose()

    warm = [s for s in timeline if s["t_s"] >= 60] or timeline
    first, last = warm[0], warm[-1]
    return {
        "concurrency": args.concurrency,
        "minutes": args.minutes,
        "store": "redis" if args.redis else "memory",
        "calls": done,
        "calls_heard_wrong": wrong,
        "call_seconds_p50": round(statistics.median(call_secs), 2) if call_secs else 0,
        "loop_lag_ms": {"p50": pct(lag, 0.5), "p99": pct(lag, 0.99), "max": pct(lag, 1.0)},
        "hit_rate_chars": round(metrics["hit_rate_chars"], 3),
        "entries_stored": metrics["entries_stored"],
        "captures": metrics["captures"],
        "renders": metrics["renders"],
        "dropped_jobs": metrics["dropped_jobs"],
        "store_errors": metrics["store_errors"],
        "after_warmup": {
            "rss_mb": [first["rss_mb"], last["rss_mb"]],
            "tasks": [first["tasks"], last["tasks"]],
            "connections": [first["connections"], last["connections"]],
        },
        "timeline": timeline,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", default=None)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--plain", action="store_true", help="the same load with Memcat switched off")
    out = asyncio.run(main(ap.parse_args()))
    short = {k: v for k, v in out.items() if k != "timeline"}
    print(json.dumps(short, indent=2))
    if ap.parse_args().json:
        ap.parse_args().json.write_text(json.dumps(out, indent=2) + "\n")
