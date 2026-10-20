"""Provider adapters: what Memcat needs to know about each TTS provider.

An adapter decides how Memcat may interleave cached and live audio:

``http``
    The provider yields audio inline from ``run_tts``. Cached and live runs mix in
    any order inside one generator.
``segmented``
    A WebSocket provider whose streams Memcat can open and close. Memcat keeps
    Pipecat's one audio context per turn, gives each live run of clauses its own
    provider stream, and releases cached and live segments into the context
    strictly in order, so cached clauses replay before, between and after live
    ones.
``leading``
    A WebSocket provider Memcat cannot segment safely. Cached clauses replay only
    until the first live request in a context; the rest is sent live.
``passthrough``
    Memcat stays out of the way entirely.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pipecat.services.tts_service import TTSService

from memcat.core.keys import SettingsNotSerializableError, canonical_value
from memcat.pipecat import compat

if TYPE_CHECKING:
    from memcat.pipecat.mixin import MemcatTTSMixin

__all__ = ["FlushBudget", "Mode", "ProviderAdapter", "Routing"]

Mode = Literal["http", "segmented", "leading", "passthrough"]
Routing = Literal["context", "cursor"]


class FlushBudget:
    """Sliding-window budget for provider flushes (Deepgram allows 20 per minute).

    Each audio context's first flush is what plain Pipecat sends anyway (its
    turn-end flush); those are always spent. A flush Memcat adds, so cached clauses
    can play after live ones, is spent only if no 60 s window it falls in would
    pass the limit, counting flushes already sent plus turn-end flushes still to
    come at the turn rate seen so far. Until ``warmup_s`` of history exists the
    rate is unknown, so no extra flush is spent.
    """

    def __init__(self, limit: int, window_s: float, warmup_s: float = 30.0):
        self._limit = limit
        self._window = window_s
        self._warmup = warmup_s
        self._spent: deque[tuple[float, bool]] = deque()  # (time, was_extra)
        self._t0: float | None = None

    def _now(self) -> float:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        while self._spent and self._spent[0][0] <= now - self._window:
            self._spent.popleft()
        return now

    def can_spend_extra(self) -> bool:
        now = self._now()
        assert self._t0 is not None
        elapsed = now - self._t0
        if elapsed < self._warmup:
            return False
        base = sum(1 for _, extra in self._spent if not extra)
        rate = base / min(self._window, elapsed)  # turn-end flushes per second
        times = [t for t, _ in self._spent]
        # Check the window ending at every moment one of today's flushes ages out.
        for tau in [0.0, *(t + self._window - now - 1e-6 for t in times), self._window]:
            tau = max(0.0, tau)
            still_in = sum(1 for t in times if t > now + tau - self._window)
            if still_in + 2 + rate * tau > self._limit:  # this flush + this turn's own
                return False
        return True

    def spend(self, extra: bool = False) -> None:
        now = self._now()
        self._spent.append((now, extra))


class ProviderAdapter:
    """Defaults for an unknown provider. Certified adapters override what differs."""

    name: str = "generic"
    certified: bool = False
    routing: Routing = "context"
    #: How many provider streams may be in flight at once.
    max_inflight: int = 4
    #: The provider's run_tts opens its own audio context (ElevenLabs).
    self_creates_context: bool = False
    #: Provider forbids altering its audio (watermarks), so only lossless codecs.
    watermarks_audio: bool = False
    #: Flush limit as (count, window seconds), or None.
    flush_limit: tuple[int, float] | None = None
    #: Constructor arguments that change the audio but are not runtime settings.
    identity_kwargs: tuple[str, ...] = ()
    #: Whether the WebSocket service can be segmented (certified adapters only).
    ws_segmentable: bool = False

    def mode_for(self, service: TTSService) -> Mode:
        if not compat.is_websocket(service):
            return "http"
        if self.ws_segmentable:
            return "segmented"
        if compat.push_start_frame(service):
            return "leading"
        return "passthrough"

    def flush_budget(self, override_per_min: int | None) -> FlushBudget | None:
        if self.flush_limit is None:
            if override_per_min is None:
                return None
            return FlushBudget(override_per_min, 60.0)
        limit, window = self.flush_limit
        if override_per_min is not None:
            limit = min(limit, override_per_min)
        return FlushBudget(limit, window)

    def init_identity(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.identity_kwargs:
            v = kwargs.get(name)
            if v is None:
                continue
            try:
                out[name] = canonical_value(v, name)
            except SettingsNotSerializableError:
                out[name] = repr(type(v))
        return out

    # --- stream control (segmented mode) -----------------------------------

    async def end_stream(self, service: MemcatTTSMixin, sid: str) -> None:
        """Tell the provider no more text is coming for this stream."""
        await service._memcat_super_flush(sid)

    async def cancel_stream(self, service: MemcatTTSMixin, sid: str) -> None:
        """Abandon a stream (interruption or its context timing out)."""
        await service._memcat_super_interrupted(sid)

    async def on_stream_done(self, service: MemcatTTSMixin, sid: str) -> None:
        """The provider finished a stream."""

    def pending_words(self, service: TTSService) -> list[tuple[str, float]]:
        """Words the provider buffered but never reported when a stream ended."""
        return []

    def wrap_socket(self, service: MemcatTTSMixin, ws: Any) -> Any:
        """The provider's WebSocket as its receive loop should see it (cursor routing
        providers use this to notice when a ``Clear`` has taken effect)."""
        return ws

    # --- HTTP word timestamps --------------------------------------------------

    def word_clock(self, service: TTSService) -> float | None:
        """The provider's own clock for word timestamps in the current context: what it
        will add to the next request's offsets. ``None`` when unknown."""
        return None

    # --- HTTP continuity -----------------------------------------------------

    def on_cached_text(self, service: TTSService, text: str) -> None:
        """A clause was served from cache; keep the provider's continuity state whole."""
