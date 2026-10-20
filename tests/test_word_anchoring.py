"""Word offsets of a stored clip line up with its audio, whatever the provider's word clock."""

from __future__ import annotations

import numpy as np

from memcat.core import codec
from memcat.core.worker import Audio, _prepare


def _tone(sr: int, secs: float, lead: float) -> bytes:
    t = np.arange(int(sr * secs)) / sr
    x = np.sin(2 * np.pi * 300 * t) * 8000.0
    pad = np.zeros(int(sr * lead))
    return np.concatenate([pad, x, pad]).astype("<i2").tobytes()


def _words(words: tuple[tuple[str, float], ...]) -> tuple[tuple[str, int], ...]:
    sr = 16000
    blob, _ = _prepare(Audio(_tone(sr, 0.5, lead=0.2), sr, words), codec.CODEC_PCM16, 1)
    assert blob is not None
    return codec.decode(blob).words


def test_word_clock_in_step_with_the_clip() -> None:
    # Speech starts at 0.2 s; 30 ms of lead silence is kept.
    assert _words((("hello", 0.2), ("there", 0.45))) == (("hello", 30), ("there", 280))


def test_word_clock_running_ahead_is_reanchored() -> None:
    # Timestamps counted from an earlier request on the same connection (ElevenLabs).
    assert _words((("hello", 5.2), ("there", 5.45))) == (("hello", 30), ("there", 280))


def test_word_clock_running_behind_is_reanchored() -> None:
    assert _words((("hello", 0.0), ("there", 0.25))) == (("hello", 30), ("there", 280))


def test_small_drift_is_kept() -> None:
    assert _words((("hello", 0.25), ("there", 0.5))) == (("hello", 80), ("there", 330))


def test_noise_floor_keeps_the_providers_clock() -> None:
    """Room tone above the silence threshold: no leading silence is found, so the
    speech start is unknown and a first word at 0.3 s is trusted, not pulled to 0."""
    sr = 16000
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(int(sr * 0.3)) * 400).astype("<i2").tobytes()
    pcm = noise + _tone(sr, 0.5, lead=0.0)
    words = (("hello", 0.3), ("there", 0.55))
    blob, _ = _prepare(Audio(pcm, sr, words), codec.CODEC_PCM16, 1)
    assert blob is not None
    assert codec.decode(blob).words == (("hello", 300), ("there", 550))
