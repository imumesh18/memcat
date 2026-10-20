"""``cached(cls)``: the one call users make."""

from __future__ import annotations

from typing import TypeVar

from pipecat.services.tts_service import TTSService

from memcat.pipecat.mixin import MemcatTTSMixin

__all__ = ["cached"]

T = TypeVar("T", bound=type[TTSService])

_classes: dict[type, type] = {}


def cached(cls: T) -> T:
    """Return a subclass of ``cls`` with Memcat first in its MRO.

    The subclass takes one extra keyword argument, ``memcat=MemcatConfig(...)``,
    and behaves exactly like ``cls`` when Memcat is disabled.
    """
    if not isinstance(cls, type) or not issubclass(cls, TTSService):
        raise TypeError("cached() expects a Pipecat TTSService subclass")
    if issubclass(cls, MemcatTTSMixin):
        return cls
    sub = _classes.get(cls)
    if sub is None:
        sub = type(
            f"Memcat{cls.__name__}",
            (MemcatTTSMixin, cls),
            {"_memcat_wrapped": cls, "__module__": cls.__module__, "__doc__": cls.__doc__},
        )
        _classes[cls] = sub
    return sub  # type: ignore[return-value]
