"""Every touch of Pipecat internals lives here, so a Pipecat rename breaks one file.

Supports Pipecat 1.10 to 1.13.
"""

from __future__ import annotations

from typing import Any

from pipecat.services.tts_service import (
    TextAggregationMode,
    TTSService,
    WebsocketTTSService,
)

# The word-timestamp item routed through audio context queues: public from 1.13,
# private before.
try:
    from pipecat.utils.context.word_timestamp_tracker import WordTimestamp as WordEntry
except ImportError:  # pragma: no cover - Pipecat 1.10-1.12
    from pipecat.services.tts_service import (  # type: ignore[attr-defined,no-redef]
        _WordTimestampEntry as WordEntry,  # pyright: ignore[reportAttributeAccessIssue]
    )

__all__ = [
    "WordEntry",
    "clear_request_context",
    "context_queue_exists",
    "current_websocket",
    "disable_playback_pause",
    "drop_tts_span",
    "given_settings",
    "is_token_mode",
    "is_websocket",
    "is_word_entry",
    "keep_context_alive",
    "make_word_entry",
    "push_start_frame",
    "replace_settings",
    "settings_object",
    "stop_frame_timeout",
    "tts_span",
    "turn_context_id",
    "uses_word_timestamps",
    "word_entry_fields",
]


def is_websocket(service: TTSService) -> bool:
    return isinstance(service, WebsocketTTSService)


def is_token_mode(service: TTSService) -> bool:
    mode = getattr(service, "_text_aggregation_mode", TextAggregationMode.SENTENCE)
    return mode == TextAggregationMode.TOKEN


def given_settings(service: TTSService) -> dict[str, Any]:
    """Every set field of the service's current settings (voice, model, language, ...)."""
    return dict(service._settings.given_fields())


def settings_object(service: TTSService) -> Any:
    return service._settings


def replace_settings(service: TTSService, settings: Any) -> None:
    """Give a freshly built service other settings, and rebuild what its constructor
    derived from the old ones (ElevenLabs' ``voice_settings`` payload)."""
    svc: Any = service
    svc._settings = settings
    rebuild = getattr(svc, "_set_voice_settings", None)
    if callable(rebuild) and hasattr(svc, "_voice_settings"):
        svc._voice_settings = rebuild()


def stop_frame_timeout(service: TTSService) -> float:
    """Seconds of silence after which Pipecat ends an audio context."""
    v = getattr(service, "_stop_frame_timeout_s", None)
    return float(v) if isinstance(v, (int, float)) and v > 0 else 3.0


def keep_context_alive(service: TTSService, context_id: str) -> None:
    """Restart Pipecat's silence timeout for an audio context (no-op once it is gone).

    1.10-1.12 queue a keepalive marker on the context; 1.13 asks its
    ``AudioContextManager``. Both go through ``_refresh_audio_context``.
    """
    fn = getattr(service, "_refresh_audio_context", None)
    if callable(fn):
        fn(context_id)


def current_websocket(service: TTSService) -> Any:
    return getattr(service, "_websocket", None)


def uses_word_timestamps(service: TTSService) -> bool:
    """True when transcripts come from word timestamps (``push_text_frames=False``)."""
    return not bool(getattr(service, "_push_text_frames", True))


def push_start_frame(service: TTSService) -> bool:
    return bool(getattr(service, "_push_start_frame", False))


def turn_context_id(service: TTSService) -> str | None:
    return getattr(service, "_turn_context_id", None)


def context_queue_exists(service: TTSService, context_id: str) -> bool:
    """Whether Pipecat itself holds a playback queue for this id (ignores Memcat's view)."""
    manager = getattr(service, "_audio_context_manager", None)  # 1.13+
    if manager is not None:
        return context_id in manager.context_ids
    return context_id in getattr(service, "_audio_contexts", {})


def make_word_entry(word: str, timestamp: float, context_id: str, ifs: bool) -> Any:
    return WordEntry(
        word=word, timestamp=timestamp, context_id=context_id, includes_inter_frame_spaces=ifs
    )


def is_word_entry(item: Any) -> bool:
    return isinstance(item, WordEntry)


def word_entry_fields(item: Any) -> tuple[str, float, str, bool]:
    return item.word, item.timestamp, item.context_id, item.includes_inter_frame_spaces


def clear_request_context(service: Any) -> None:
    """Forget text a service sends as context with its next request (ElevenLabs HTTP's
    ``previous_text``). Pipecat clears it when a stop frame passes, which in a render
    twin may come after the next render has started."""
    if isinstance(getattr(service, "_previous_text", None), str):
        service._previous_text = ""


def tts_span(service: Any, context_id: str) -> Any:
    """The OpenTelemetry span Pipecat's tracing opened for this audio context, if any."""
    if not getattr(service, "_tracing_enabled", False):
        return None
    entry = getattr(service, "_tts_spans", {}).get(context_id)
    return entry.get("span") if isinstance(entry, dict) else None


def drop_tts_span(service: Any, context_id: str) -> None:
    """Forget the span tracing opened for a provider stream Memcat created inside a
    context (the context's own span carries the turn). Never ended, never exported."""
    for name in ("_tts_spans", "_tts_texts"):
        d = getattr(service, name, None)
        if isinstance(d, dict):
            d.pop(context_id, None)


def disable_playback_pause(service: Any) -> None:
    """Stop a service pausing input until playback ends (no transport will say so)."""
    if hasattr(service, "_pause_frame_processing"):
        service._pause_frame_processing = False
