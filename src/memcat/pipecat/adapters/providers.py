"""Provider adapters: certified for Cartesia, ElevenLabs, Deepgram and Sarvam.
Fish Audio has an adapter but is not certified."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from pipecat.services.tts_service import TTSService

from memcat.pipecat.adapters.base import ProviderAdapter

if TYPE_CHECKING:
    from memcat.pipecat.mixin import MemcatTTSMixin

__all__ = [
    "CartesiaAdapter",
    "DeepgramAdapter",
    "ElevenLabsAdapter",
    "FishAudioAdapter",
    "SarvamAdapter",
    "select_adapter",
]


class CartesiaAdapter(ProviderAdapter):
    """Cartesia: messages carry ``context_id``; ``continue=False`` ends a context.

    ``flush_audio(id)`` sends the empty ``continue=False`` message, after which
    Cartesia replies ``done`` and Pipecat removes the context. There is no
    per-context state in the service, so any number of streams may overlap.
    """

    name = "cartesia"
    certified = True
    ws_segmentable = True
    identity_kwargs = ("encoding", "container", "cartesia_version")


class ElevenLabsAdapter(ProviderAdapter):
    """ElevenLabs multi-context WebSocket.

    ``run_tts`` creates its own provider context (sending voice settings), and the
    service keeps a single word clock that a new context resets, so streams run
    one at a time. A stream ends with ``flush`` then ``close_context``, which
    makes ``isFinal`` arrive right after the last audio.
    """

    name = "elevenlabs"
    certified = True
    max_inflight = 1
    self_creates_context = True
    ws_segmentable = True
    identity_kwargs = ("auto_mode", "enable_ssml_parsing", "pronunciation_dictionary_locators")

    async def end_stream(self, service: MemcatTTSMixin, sid: str) -> None:
        await service._memcat_super_flush(sid)
        await service._memcat_super_close_context(sid)

    async def on_stream_done(self, service: MemcatTTSMixin, sid: str) -> None:
        # Frees the per-connection bookkeeping (alignment and init-sent sets).
        await service._memcat_super_completed(sid)

    def pending_words(self, service: TTSService) -> list[tuple[str, float]]:
        # The WebSocket service reports a word only when the next one starts, and
        # never flushes the last at isFinal; a stream's clip would lose it.
        word = getattr(service, "_partial_word", "")
        start = getattr(service, "_partial_word_start_time", 0.0)
        if not word or not str(word).strip():
            return []
        service._partial_word = ""  # type: ignore[attr-defined]
        return [(str(word), float(start or 0.0))]

    def word_clock(self, service: TTSService) -> float | None:
        # HTTP service: word times are ``_cumulative_time`` (the summed alignment
        # length of earlier requests, not their audio length) plus the offset.
        v = getattr(service, "_cumulative_time", None)
        return float(v) if isinstance(v, (int, float)) else None

    def on_cached_text(self, service: TTSService, text: str) -> None:
        # HTTP service: keep previous_text whole so the next live request still
        # gets the clauses Memcat served from cache as context.
        prev = getattr(service, "_previous_text", None)
        if prev is None or not text.strip():
            return
        t = text.strip()
        service._previous_text = f"{prev} {t}" if prev else t  # type: ignore[attr-defined]


class DeepgramAdapter(ProviderAdapter):
    """Deepgram Aura WebSocket: no context ids on the wire.

    Audio and ``Flushed`` go to ``get_active_audio_context_id()``, so Memcat keeps
    one stream in flight and answers that call with it. Each stream ends with a
    ``Flush``; Deepgram allows 20 per minute, so extra flushes come from a budget
    that always leaves room for turn-end flushes.
    """

    name = "deepgram"
    certified = True
    routing = "cursor"
    max_inflight = 1
    ws_segmentable = True
    watermarks_audio = True
    flush_limit = (20, 60.0)
    identity_kwargs = ("encoding",)

    async def cancel_stream(self, service: MemcatTTSMixin, sid: str) -> None:
        # One "Clear" covers the socket; it is sent once per context by the mixin.
        return None

    def wrap_socket(self, service: MemcatTTSMixin, ws: Any) -> Any:
        # Audio still in flight when "Clear" was sent arrives before "Cleared" and
        # would be routed to whatever stream is current by then.
        return _ClearWatcher(ws, service._memcat_watching, service._memcat_on_cleared)


class _ClearWatcher:
    """A WebSocket whose receive loop reports each ``Cleared`` before Pipecat sees it."""

    def __init__(
        self, ws: Any, on_start: Callable[[Any], None], on_cleared: Callable[[], None]
    ) -> None:
        self._ws = ws
        self._on_start = on_start
        self._on_cleared = on_cleared

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        self._on_start(self._ws)
        async for message in self._ws:
            if isinstance(message, str) and "Cleared" in message:
                try:
                    kind = json.loads(message).get("type")
                except Exception:
                    kind = None
                if kind == "Cleared":
                    self._on_cleared()
            yield message


class SarvamAdapter(ProviderAdapter):
    """Sarvam Bulbul: WebSocket has no context ids, and whether each ``flush``
    yields its own ``final`` event is undocumented, so WebSocket runs in leading
    mode. The HTTP service gets full clause caching.
    """

    name = "sarvam"
    certified = True
    routing = "cursor"
    max_inflight = 1


class FishAudioAdapter(ProviderAdapter):
    """Fish Audio (S2, Drama 3) WebSocket: audio goes to the playing context and no
    event marks the end of a flushed request, so it runs in leading mode."""

    name = "fish"
    routing = "cursor"


_BY_MODULE: dict[str, type[ProviderAdapter]] = {
    "pipecat.services.cartesia.tts": CartesiaAdapter,
    "pipecat.services.elevenlabs.tts": ElevenLabsAdapter,
    "pipecat.services.deepgram.tts": DeepgramAdapter,
    "pipecat.services.sarvam.tts": SarvamAdapter,
    "pipecat.services.fish.tts": FishAudioAdapter,
}


def select_adapter(cls: type) -> ProviderAdapter:
    """Pick the adapter from the first class in the MRO that Pipecat ships.

    A service can name its own with a class attribute ``__memcat_adapter__``
    (a :class:`ProviderAdapter` subclass), which wins over module matching.
    """
    explicit = getattr(cls, "__memcat_adapter__", None)
    if isinstance(explicit, type) and issubclass(explicit, ProviderAdapter):
        return explicit()
    for klass in cls.__mro__:
        adapter = _BY_MODULE.get(klass.__module__)
        if adapter is not None:
            return adapter()
    return ProviderAdapter()
