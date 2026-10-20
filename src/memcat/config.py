"""User-facing configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from memcat.core.audio import CLAUSE_PAUSE_MS
from memcat.core.store.base import DEFAULT_TTL_S, Store
from memcat.core.store.memory import MemoryStore
from memcat.core.worker import Policy

__all__ = ["MemcatConfig", "default_store"]

_NS_RE = re.compile(r"[A-Za-z0-9_.\-]{1,64}")

_default_store: MemoryStore | None = None


def default_store() -> MemoryStore:
    """The process-wide :class:`MemoryStore` used when a config names no store.

    Every config without a store shares it, so calls share one runtime (one L1, one
    worker) and admission can count distinct calls. A fresh store per config would
    give every call its own runtime, and a clause would never reach
    ``admit_after_calls``.
    """
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store


@dataclass(frozen=True)
class MemcatConfig:
    """How Memcat caches one TTS service. Immutable once built.

    Attributes:
        store: Shared store (L2). L1 is always in-process. Defaults to one in-process
            :class:`MemoryStore` shared by every config that names no store.
        namespace: Tenant prefix on every key.
        admit_after_calls: Distinct calls that must say a clause before it is cached.
        min_clause_words: Shortest clause split out of a sentence.
        ttl_s: Entry lifetime, refreshed when served.
        l1_max_bytes: Bound on decoded clips held in process.
        codec: "auto" (mu-law at <= 8 kHz, Opus when installed, else PCM16), or a fixed codec.
        lossless_only: Never store lossy audio (for providers that watermark audio).
        render: Render admitted clauses in the background.
        warm: Phrases rendered at start, skipping admission.
        block_patterns: Extra regexes that are never cached.
        queue_size: Worker queue bound; overflow is dropped and counted.
        queue_max_bytes: Bound on captured audio waiting in the worker queue; captures
            past it are dropped and counted (``dropped_bytes``).
        enabled: Kill switch. When False the service behaves exactly as unwrapped.
        prices: Overrides for the USD-per-1M-characters table behind ``usd_saved``,
            keyed by (adapter, model prefix).
        gain_match: Match replayed loudness to the live voice (clamped to +/-3 dB).
        trim_joins: At each join between a cached and another clause, trim edge
            silence and set the pause Memcat expects there (see ``clause_pause_ms``).
        clause_pause_ms: The pause after a full clause at a comma; shorter after a
            one- or two-word opener, a little longer at ``;`` ``:`` and dashes.
        max_stream_flushes_per_min: Extra provider flushes Memcat may spend per minute to
            replay cached clauses after live audio. None uses the adapter's safe default.
    """

    store: Store = field(default_factory=default_store)
    namespace: str = "default"
    admit_after_calls: int = 3
    min_clause_words: int = 3
    ttl_s: int = DEFAULT_TTL_S
    l1_max_bytes: int = 268_435_456
    codec: Literal["auto", "mulaw", "pcm16", "opus"] = "auto"
    lossless_only: bool = False
    render: bool = True
    warm: Sequence[str] = ()
    block_patterns: Sequence[str] = ()
    queue_size: int = 1_000
    queue_max_bytes: int = 64 * 1024 * 1024
    enabled: bool = True
    prices: Mapping[tuple[str, str], float] = field(default_factory=dict)
    gain_match: bool = True
    trim_joins: bool = True
    clause_pause_ms: int = CLAUSE_PAUSE_MS
    max_stream_flushes_per_min: int | None = None
    compiled_blocks: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _NS_RE.fullmatch(self.namespace) or not self.namespace.strip("."):
            raise ValueError("namespace must be 1-64 characters of [A-Za-z0-9_.-], not only dots")
        if self.admit_after_calls < 1:
            raise ValueError("admit_after_calls must be >= 1")
        if self.min_clause_words < 1:
            raise ValueError("min_clause_words must be >= 1")
        if self.ttl_s < 60:
            raise ValueError("ttl_s must be at least 60 seconds")
        if self.codec not in ("auto", "mulaw", "pcm16", "opus"):
            raise ValueError(f"unknown codec {self.codec!r}")
        if self.queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        if self.queue_max_bytes < 1:
            raise ValueError("queue_max_bytes must be >= 1")
        object.__setattr__(self, "warm", tuple(self.warm))
        object.__setattr__(self, "block_patterns", tuple(self.block_patterns))
        object.__setattr__(
            self,
            "compiled_blocks",
            tuple(re.compile(p, re.IGNORECASE) for p in self.block_patterns),
        )

    def policy(self) -> Policy:
        """The per-service caching rules the background worker applies to this config."""
        return Policy(
            admit_after_calls=self.admit_after_calls,
            codec=self.codec,
            lossless_only=self.lossless_only,
            ttl_s=self.ttl_s,
            render=self.render,
        )
