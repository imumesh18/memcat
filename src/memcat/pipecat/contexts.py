"""Per-context state for the three replay modes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["CtxState", "HttpCtx", "LeadCtx", "Segment", "Stream"]


@dataclass(eq=False, slots=True)
class Segment:
    """A run of audio inside one Pipecat context: cached clips or one live stream.

    Items wait in ``items`` until every earlier segment has been released. Word
    timestamps inside are relative to the segment start; ``base_s`` (the audio
    already released into the context when this segment reached the head) turns
    them into context time.
    """

    items: list[Any] = field(default_factory=list)
    done: bool = False
    base_s: float | None = None


@dataclass(eq=False, slots=True)
class Stream:
    """One provider stream (a provider-side context) carrying live text."""

    sid: str
    cid: str
    seg: Segment
    registered: bool = False
    state: Literal["waiting", "sending", "sent", "done"] = "waiting"
    backlog: deque[str] = field(default_factory=deque)
    end_requested: bool = False
    ended: bool = False
    ended_at: float = 0.0
    started_at: float = 0.0  # when the stream began sending text to the provider
    watched: bool = False  # the end was requested and a watchdog keeps its context alive
    got_audio: bool = False
    last_item_at: float = 0.0
    clauses: list[tuple[bytes | None, str, bool]] = field(default_factory=list)
    capture: bool = True  # still eligible for caching (native format, under the length cap)
    pcm: bytearray = field(default_factory=bytearray)
    words: list[tuple[str, float]] = field(default_factory=list)
    error: bool = False
    plans: list[tuple[Any, list[int]]] = field(default_factory=list)
    trimmer: Any = None
    tail_trim: bool = False  # a cached clip of the same sentence follows this stream
    held: list[Any] = field(default_factory=list)


@dataclass(eq=False, slots=True)
class CtxState:
    """Segmented mode: the ordered segments of one Pipecat audio context."""

    cid: str
    segments: deque[Segment] = field(default_factory=deque)
    emitted_s: float = 0.0
    open_stream: Stream | None = None
    streams: list[Stream] = field(default_factory=list)
    input_done: bool = False
    finalized: bool = False
    releasing: bool = False
    base_flush_done: bool = False


@dataclass(eq=False, slots=True)
class HttpCtx:
    """HTTP mode: running totals for one Pipecat context."""

    emitted_s: float = 0.0
    cached_s: float = 0.0
    live_s: float = 0.0
    live_active: bool = False
    capture: bool = False
    capture_base_s: float = 0.0  # live audio before the captured run (generic word clock)
    word_clock: float | None = None  # the provider exposes its word clock (value at run start)
    trimmed_s: float = 0.0  # live audio removed by join trimming in this context
    trimmer: Any = None
    held: list[Any] = field(default_factory=list)
    pcm: bytearray = field(default_factory=bytearray)
    words: list[tuple[str, float]] = field(default_factory=list)


@dataclass(eq=False, slots=True)
class LeadCtx:
    """Leading mode: cached audio before the first live request of a context."""

    has_live: bool = False
    cached_s: float = 0.0
    finalized: bool = False
