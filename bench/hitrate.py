"""Hit rate: the share of an agent's speech Memcat could serve from cache.

Replays real conversations, one dialogue per call, through Memcat's chunker, PII
rules and admission (a clause is cached only after k distinct calls said it). No
TTS is called; this measures the text, which is what bounds savings.

Corpora (downloaded on first use into ``bench/data``):

``abcd``  ABCD, 10,042 typed human customer-support chats (MIT) -- loosely scripted
          https://arxiv.org/abs/2104.00783
``hvb``   Harper Valley Bank, 1,446 spoken calls of human agents on a guided bank
          script (CC-BY-4.0) -- scripted, IVR-like
          https://github.com/cricketclub/gridspace-stanford-harper-valley
``tau2``  tau2-bench, GPT-4.1 agents following a written policy in the airline,
          retail and telecom domains (MIT) -- the closest public stand-in for an LLM
          voice agent; trial 0 only, so each scenario counts once
          https://github.com/sierra-research/tau2-bench

Two measures:

* ``share_cached``: characters served from cache / all characters (what TTS bills);
* ``responses_fully_cached``: agent turns whose every clause was a hit (the
  response-level count some vendors quote).

    uv run python bench/hitrate.py --corpus all
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memcat.core.chunker import Clause, is_cacheable, split  # noqa: E402
from memcat.core.keys import normalize_text  # noqa: E402

DATA = Path(__file__).parent / "data"
ABCD_URL = "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz"
HVB_TREE = (
    "https://api.github.com/repos/cricketclub/gridspace-stanford-harper-valley/git/trees/"
    "master?recursive=1"
)
HVB_RAW = "https://raw.githubusercontent.com/cricketclub/gridspace-stanford-harper-valley/master/"
TAU2_RAW = (
    "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/results/final/"
)
TAU2_FILES = [
    f"gpt-4.1-2025-04-14_{d}_default_gpt-4.1-2025-04-14_4trials.json"
    for d in ("airline", "retail", "telecom")
]
_SENT = re.compile(r"(?<=[.!?])\s+(?=\S)")
Calls = list[list[str]]


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", file=sys.stderr)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310
                data = r.read()
            break
        except OSError:
            time.sleep(2**attempt)
    else:
        with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310
            data = r.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def load_abcd() -> tuple[Calls, int]:
    path = DATA / "abcd_v1.1.json.gz"
    if not path.exists():
        _fetch(ABCD_URL, path)
    with gzip.open(path) as f:
        raw = json.load(f)
    calls: Calls = []
    for split_name in ("train", "dev", "test"):
        for convo in raw[split_name]:
            calls.append([t for who, t in convo["original"] if who == "agent" and t.strip()])
    return calls, len(raw["train"])


def load_hvb() -> Calls:
    d = DATA / "hvb"
    files = sorted(d.glob("*.json"))
    if len(files) < 1400:
        with urllib.request.urlopen(HVB_TREE, timeout=60) as r:  # noqa: S310
            tree = json.load(r)["tree"]
        paths = [t["path"] for t in tree if t["path"].startswith("data/transcript/")]

        def get(p: str) -> None:
            dest = d / Path(p).name
            if not dest.exists():
                _fetch(HVB_RAW + p, dest)

        with concurrent.futures.ThreadPoolExecutor(16) as ex:
            list(ex.map(get, paths))
        files = sorted(d.glob("*.json"))
    calls: Calls = []
    for f in files:
        segs = sorted(json.loads(f.read_text()), key=lambda s: s["start_ms"])
        turns = [
            s["human_transcript"].replace("[noise]", "").strip()
            for s in segs
            if s.get("speaker_role") == "agent"
        ]
        calls.append([t for t in turns if t])
    return calls


def load_tau2() -> dict[str, Calls]:
    out: dict[str, Calls] = {}
    for name in TAU2_FILES:
        path = DATA / "tau2" / name
        if not path.exists():
            _fetch(TAU2_RAW + name, path)
        sims = json.loads(path.read_text())["simulations"]
        domain = name.split("_")[1]
        out[domain] = [
            [
                m["content"]
                for m in s["messages"]
                if m.get("role") == "assistant" and (m.get("content") or "").strip()
            ]
            for s in sorted(sims, key=lambda s: s["task_id"])
            if s.get("trial", 0) == 0
        ]
    return out


def sentences(turn: str) -> list[str]:
    return [s for s in _SENT.split(turn.strip()) if s.strip()]


def units(turn: str, level: str, min_words: int) -> list[Clause]:
    out: list[Clause] = []
    for s in sentences(turn):
        if level == "sentence":
            out.append(Clause(s, is_cacheable(s)))
        else:
            out.extend(split(s, min_words=min_words))
    return out


def simulate(
    calls: Calls,
    level: str = "clause",
    k: int = 3,
    min_words: int = 3,
    warm: set[str] | None = None,
) -> dict[str, float]:
    """Replay calls in order. Hits are decided per sentence, as Memcat plans them.

    Net savings subtract what filling the cache costs: each admitted clause is
    synthesized once more by a render, unless it was spoken live as its own run
    (bounded by cached clauses or the sentence edges), when its live audio is
    captured for free. Warmed clauses are rendered once up front.
    """
    cache: set[str] = set(warm or ())
    seen: dict[str, set[int]] = defaultdict(set)
    total = cached = blocked = digit_keys = 0
    rendered = sum(len(w) for w in cache)
    captured = 0
    responses = full = partial = 0
    for call_id, turns in enumerate(calls):
        for turn in turns:
            hits = n_units = 0
            for sentence in sentences(turn):
                if level == "sentence":
                    clauses = [Clause(sentence, is_cacheable(sentence))]
                else:
                    clauses = split(sentence, min_words=min_words)
                keys = [normalize_text(c.text) if c.cacheable else "" for c in clauses]
                hit = [bool(x) and x in cache for x in keys]
                for i, c in enumerate(clauses):
                    n = len(c.text)
                    total += n
                    n_units += 1
                    if not c.cacheable:
                        blocked += n
                        continue
                    key = keys[i]
                    if not key:
                        continue
                    if hit[i]:
                        cached += n
                        hits += 1
                        continue
                    s = seen[key]
                    s.add(call_id)
                    if len(s) >= k and key not in cache:
                        cache.add(key)
                        del seen[key]
                        digit_keys += bool(re.search(r"\d", key))
                        alone = (i == 0 or hit[i - 1]) and (i == len(clauses) - 1 or hit[i + 1])
                        if alone:
                            captured += 1
                        else:
                            rendered += n
            if n_units:
                responses += 1
                full += hits == n_units
                partial += hits > 0
    return {
        "calls": len(calls),
        "chars_total": total,
        "share_cached": round(cached / total, 4) if total else 0.0,
        "share_saved_net": round((cached - rendered) / total, 4) if total else 0.0,
        "chars_rendered": rendered,
        "entries_captured_free": captured,
        "share_blocked_pii": round(blocked / total, 4) if total else 0.0,
        "responses_fully_cached": round(full / responses, 4) if responses else 0.0,
        "responses_with_a_hit": round(partial / responses, 4) if responses else 0.0,
        "entries": len(cache),
        "entries_with_digits": digit_keys,
    }


def top_clauses(calls: Calls, n: int, min_words: int) -> set[str]:
    counts: Counter[str] = Counter()
    for turns in calls:
        keys = {
            normalize_text(c.text)
            for t in turns
            for c in units(t, "clause", min_words)
            if c.cacheable
        }
        counts.update(x for x in keys if x)
    return {x for x, _ in counts.most_common(n)}


def run(corpus: str, k: int, min_words: int, warm_top: int, sweep: bool) -> dict[str, object]:
    res: dict[str, object] = {"k": k, "min_clause_words": min_words}
    if corpus in ("abcd", "all"):
        calls, n_train = load_abcd()
        test = calls[n_train:]
        warm = top_clauses(calls[:n_train], warm_top, min_words)
        res["abcd"] = {
            "sentence_level": simulate(calls, "sentence", k, min_words),
            "clause_level": simulate(calls, "clause", k, min_words),
            "test_split_cold": simulate(test, "clause", k, min_words),
            "test_split_warmed_from_train": simulate(test, "clause", k, min_words, warm),
        }
        if sweep:
            res["abcd"]["sweep_k"] = {  # type: ignore[index]
                str(x): simulate(calls, "clause", x, min_words)["share_cached"]
                for x in (2, 3, 4, 5)
            }
    if corpus in ("hvb", "all"):
        hvb = load_hvb()
        res["hvb"] = {"clause_level": simulate(hvb, "clause", k, min_words)}
    if corpus in ("tau2", "all"):
        res["tau2"] = {
            dom: {"clause_level": simulate(c, "clause", k, min_words)}
            for dom, c in load_tau2().items()
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["abcd", "hvb", "tau2", "all"], default="all")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--warm-top", type=int, default=500)
    ap.add_argument("--sweep", action="store_true", help="also sweep k = 2..5 on ABCD")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    out = json.dumps(run(args.corpus, args.k, args.min_words, args.warm_top, args.sweep), indent=2)
    print(out)
    if args.json:
        args.json.write_text(out + "\n")


if __name__ == "__main__":
    main()
