"""Hot-path step costs, checked against stored baselines (the nightly check fails past 3x).

uv run python bench/hotpath.py                 # print
uv run python bench/hotpath.py --check         # compare with bench/baseline.json
uv run python bench/hotpath.py --write-baseline
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memcat.core import codec  # noqa: E402
from memcat.core.breaker import Breaker  # noqa: E402
from memcat.core.chunker import split  # noqa: E402
from memcat.core.index import KeyIndex  # noqa: E402
from memcat.core.keys import Keyer  # noqa: E402
from memcat.core.lookup import Clip, L1Cache, Lookup  # noqa: E402
from memcat.core.metrics import Metrics  # noqa: E402
from memcat.core.store.memory import MemoryStore  # noqa: E402

BASELINE = Path(__file__).parent / "baseline.json"
SENTENCE = (
    "Thanks for waiting, I have pulled up your account and I can see the order, "
    "it shipped yesterday and should arrive within the next few business days."
)
GIVEN = {
    "model": "sonic-3",
    "voice": "a0e99841-438c-4a64-b679-ae501e7d6091",
    "language": "en",
    "generation_config": {"speed": 1.0, "volume": 1.0, "emotion": None},
}


def per_call(fn: Callable[[], object], n: int) -> float:
    for _ in range(max(10, n // 20)):
        fn()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6  # microseconds


def run() -> dict[str, float]:
    out: dict[str, float] = {}
    out["split_30_words_us"] = per_call(lambda: split(SENTENCE), 5000)
    keyer = Keyer("pipecat.services.cartesia.tts.CartesiaTTSService")
    out["key_context_memo_us"] = per_call(lambda: keyer.context(GIVEN, 8000), 20000)
    kc = keyer.context(GIVEN, 8000)
    out["key_digest_us"] = per_call(lambda: kc.digest("your order has shipped, "), 50000)
    out["key_build_and_hash_us"] = per_call(
        lambda: keyer.context(GIVEN, 8000).digest("your order has shipped, "), 20000
    )

    l1 = L1Cache(1 << 40)
    clip = Clip(8000, 1000, (), None, pcm=b"\0" * 16000, nbytes=16000)
    keys = [os.urandom(16) for _ in range(50_000)]
    for k in keys:
        l1.put(k, clip)
    probe = keys[25_000]
    out["l1_get_touch_50k_us"] = per_call(lambda: l1.get(probe), 200000)

    store = MemoryStore().bind("bench")
    idx = KeyIndex(store)
    for k in keys:
        idx.add(k)
    idx._loaded = True
    miss = os.urandom(16)
    out["index_check_us"] = per_call(lambda: idx.may_contain(miss), 200000)

    sr = 8000
    pcm = (np.sin(np.arange(sr * 3) / sr * 2 * np.pi * 220) * 8000).astype("<i2").tobytes()
    blob = codec.encode(
        codec.Entry(
            codec.CODEC_MULAW,
            sr,
            3000,
            -20.0,
            (("a", 0),),
            codec.encode_audio(pcm, sr, codec.CODEC_MULAW),
        )
    )
    out["decode_3s_mulaw_us"] = per_call(lambda: codec.to_pcm16(codec.decode(blob)), 2000)

    if codec.opus_available():
        sr2 = 24000
        pcm2 = (np.sin(np.arange(sr2 * 3) / sr2 * 2 * np.pi * 220) * 8000).astype("<i2").tobytes()
        blob2 = codec.encode(
            codec.Entry(
                codec.CODEC_OPUS,
                sr2,
                3000,
                None,
                (),
                codec.encode_audio(pcm2, sr2, codec.CODEC_OPUS),
            )
        )
        e2 = codec.decode(blob2)
        out["opus_decode_3s_us"] = per_call(lambda: codec.to_pcm16(e2), 20)

    # One whole sentence on a miss: split, key every clause, look them all up.
    lookup = Lookup(
        store,
        l1,
        idx,
        Breaker(),
        Metrics(),
        lambda _a, _b: None,
        lambda _: None,
        lambda _a, _b: None,
    )
    loop = asyncio.new_event_loop()

    def miss_sentence() -> None:
        clauses = split(SENTENCE)
        k = keyer.context(GIVEN, 8000)
        ds = [k.digest(c.text) if c.cacheable else None for c in clauses]
        coro = lookup.get_many(ds)
        with contextlib.suppress(StopIteration):
            coro.send(None)

    out["miss_sentence_total_us"] = per_call(miss_sentence, 5000)
    loop.close()
    return {k: round(v, 3) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--factor", type=float, default=3.0)
    args = ap.parse_args()
    res = run()
    print(json.dumps(res, indent=2))
    if args.write_baseline:
        BASELINE.write_text(json.dumps(res, indent=2) + "\n")
    if args.check:
        base = json.loads(BASELINE.read_text())
        bad = {k: (v, base[k]) for k, v in res.items() if k in base and v > base[k] * args.factor}
        if bad:
            print(f"slower than {args.factor}x baseline: {bad}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
