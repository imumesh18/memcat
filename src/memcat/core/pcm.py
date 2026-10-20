"""PCM16 helpers with no third-party dependency beyond what Pipecat installs."""

from __future__ import annotations

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # stdlib on 3.11/3.12, audioop-lts on 3.13+ (a Pipecat dependency)

__all__ = ["bytes_to_seconds", "lin2ulaw", "ulaw2lin"]


def lin2ulaw(pcm16: bytes) -> bytes:
    return audioop.lin2ulaw(pcm16, 2)


def ulaw2lin(ulaw: bytes) -> bytes:
    return audioop.ulaw2lin(ulaw, 2)


def bytes_to_seconds(n_bytes: int, sample_rate: int) -> float:
    """Duration of ``n_bytes`` of mono PCM16 at ``sample_rate`` Hz (0.0 when the rate is 0)."""
    return n_bytes / (2.0 * sample_rate) if sample_rate else 0.0
