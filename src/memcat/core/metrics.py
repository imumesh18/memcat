"""Counters that never leave the process."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DEFAULT_PRICES_USD_PER_M", "Metrics", "price_for"]

# USD per 1M characters at the common pay-as-you-go or startup tier, read 18 Oct 2026.
# Keyed by (adapter, model prefix); the longest matching prefix wins; "" is the fallback.
DEFAULT_PRICES_USD_PER_M: dict[tuple[str, str], float] = {
    ("elevenlabs", "eleven_multilingual"): 80.0,
    ("elevenlabs", "eleven_v3"): 80.0,
    ("elevenlabs", "eleven_v4"): 80.0,
    ("elevenlabs", "eleven_flash"): 40.0,
    ("elevenlabs", "eleven_turbo"): 40.0,
    ("elevenlabs", "eleven_v3_conversational"): 40.0,
    ("elevenlabs", ""): 40.0,
    ("cartesia", ""): 39.0,
    ("sarvam", ""): 31.6,
    ("deepgram", ""): 30.0,
}


def price_for(
    adapter: str, model: str | None, overrides: Mapping[tuple[str, str], float] | None = None
) -> float:
    table = {**DEFAULT_PRICES_USD_PER_M, **(overrides or {})}
    m = (model or "").lower()
    best: tuple[int, float] | None = None
    for (a, prefix), price in table.items():
        if a == adapter and m.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), price)
    return best[1] if best else 0.0


_COUNTERS = (
    "hits",
    "misses",
    "chars_cached",
    "chars_live",
    "chars_rendered",
    "bytes_stored",
    "entries_stored",
    "store_errors",
    "dropped_jobs",
    "dropped_bytes",
    "breaker_open",
    "renders",
    "render_failures",
    "captures",
    "corrupt_entries",
    "fallbacks",
)


@dataclass
class Metrics:
    """Counters plus a bounded sample of time-to-first-audio on hits."""

    counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_COUNTERS, 0))
    usd_saved: float = 0.0
    _ttfa: deque[float] = field(default_factory=lambda: deque(maxlen=2048))

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def observe_ttfa(self, ms: float) -> None:
        self._ttfa.append(ms)

    def percentile(self, q: float) -> float | None:
        if not self._ttfa:
            return None
        s = sorted(self._ttfa)
        return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]

    def snapshot(self) -> dict[str, Any]:
        c = dict(self.counters)
        total = c["chars_cached"] + c["chars_live"]
        return {
            **c,
            "hit_rate_chars": (c["chars_cached"] / total) if total else 0.0,
            "usd_saved": round(self.usd_saved, 6),
            "replay_first_audio_ms_p50": self.percentile(0.5),
            "replay_first_audio_ms_p99": self.percentile(0.99),
        }
