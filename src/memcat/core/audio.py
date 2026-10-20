"""Audio polish for stored clips: trim, fade, loudness, gain.

All functions take and return mono little-endian PCM16 bytes. They run in the
background worker (in a thread), never on the call's hot path, except
:func:`apply_gain`, which costs a few microseconds per 40 ms frame.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

__all__ = [
    "CLAUSE_PAUSE_MS",
    "EDGE_MS",
    "GAIN_CLAMP_DB",
    "SENTENCE_GAP_MS",
    "SHORT_PAUSE_MS",
    "EdgeTrimmer",
    "LoudnessTracker",
    "apply_gain",
    "fade_edges",
    "join_pause_ms",
    "loudness",
    "silence",
    "trim_silence",
]

GAIN_CLAMP_DB = 3.0


def _arr(pcm: bytes) -> np.ndarray:
    # A stray odd byte (a provider chunk cut mid-sample) is ignored, not an error.
    return np.frombuffer(pcm, dtype="<i2", count=len(pcm) // 2)


EDGE_MS = 30  # silence a stored clip keeps before its first and after its last word
SHORT_PAUSE_MS = 100  # after a one- or two-word opener ("Sure,")
SENTENCE_GAP_MS = 150  # after a sentence-final clip, before the next sentence
CLAUSE_PAUSE_MS = 320  # after a full clause at a comma (``MemcatConfig.clause_pause_ms``)


def join_pause_ms(text: str, clause_pause_ms: int = CLAUSE_PAUSE_MS) -> int:
    """The pause a listener expects after ``text`` when another clause follows.

    In live synthesis a short opener is followed by about 100 ms of pause and a full
    clause at a comma by roughly 300-400 ms (``bench/abx.py`` compares joins with
    live). A provider synthesizing a clause alone gets this backwards (openers end
    with long silence, full clauses with short), so Memcat stores clips tight and sets
    the pause at each join.
    """
    t = text.rstrip()
    if not t:
        return 0
    if t[-1] in ".!?\u2026":
        return SENTENCE_GAP_MS
    if len(t.split()) <= 2:
        return SHORT_PAUSE_MS
    if t[-1] in ";:\u2014\u2013-":
        return clause_pause_ms + 40
    return clause_pause_ms


def silence(sr: int, ms: float) -> bytes:
    n = max(0, int(sr * ms / 1000))
    return b"\x00\x00" * n


def trim_silence(
    pcm: bytes,
    sr: int,
    threshold_dbfs: float = -50.0,
    keep_ms: int = EDGE_MS,
) -> tuple[bytes, int]:
    """Trim leading and trailing silence, keeping ``keep_ms`` either side of the speech.

    Returns the trimmed audio and the number of samples cut from the start, so
    word offsets can be shifted to match.
    """
    x = _arr(pcm)
    if x.size == 0:
        return pcm, 0
    win = max(1, sr // 100)  # 10 ms windows
    n = x.size // win
    if n == 0:
        return pcm, 0
    frames = x[: n * win].astype(np.float32).reshape(n, win)
    rms = np.sqrt(np.mean(frames * frames, axis=1)) / 32768.0
    level = 20 * np.log10(np.maximum(rms, 1e-9))
    loud = np.nonzero(level > threshold_dbfs)[0]
    if loud.size == 0:
        return b"", 0
    keep = sr * keep_ms // 1000
    start = max(0, int(loud[0]) * win - keep)
    end = min(x.size, (int(loud[-1]) + 1) * win + keep)
    return x[start:end].tobytes(), start


def fade_edges(pcm: bytes, sr: int, ms: int = 10) -> bytes:
    """Raised-cosine fade in and out so joins between clips never click."""
    x = _arr(pcm).astype(np.float32)
    n = min(x.size // 2, sr * ms // 1000)
    if n <= 1:
        return pcm
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, n, dtype=np.float32))
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return np.clip(np.rint(x), -32768, 32767).astype("<i2").tobytes()


def loudness(pcm: bytes, sr: int) -> float | None:
    """Integrated loudness in LUFS (ITU BS.1770), or None for clips under 400 ms."""
    x = _arr(pcm)
    if x.size < sr * 0.4:
        return None
    try:
        import loudness as _ld  # shipped with Pipecat

        v = float(_ld.integrated_loudness(x.astype(np.float32) / 32768.0, int(sr)))
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v) or v < -70:
        return None
    return v


def apply_gain(pcm: bytes, gain_db: float) -> bytes:
    """Scale PCM16 by ``gain_db``, clamped to +/-3 dB, with hard clipping."""
    g = max(-GAIN_CLAMP_DB, min(GAIN_CLAMP_DB, gain_db))
    if abs(g) < 0.05:
        return pcm
    factor = 10.0 ** (g / 20.0)
    x = _arr(pcm).astype(np.float32) * factor
    return np.clip(np.rint(x), -32768, 32767).astype("<i2").tobytes()


class LoudnessTracker:
    """Running median of live-audio loudness per voice, for replay gain matching."""

    def __init__(self, window: int = 31):
        self._window = window
        self._samples: dict[str, list[float]] = {}

    def add(self, voice: str, lufs: float) -> None:
        buf = self._samples.setdefault(voice, [])
        buf.append(lufs)
        if len(buf) > self._window:
            del buf[0]

    def median(self, voice: str) -> float | None:
        buf = self._samples.get(voice)
        if not buf or len(buf) < 3:
            return None
        s = sorted(buf)
        return s[len(s) // 2]

    def gain_for(self, voice: str, clip_lufs: float | None) -> float:
        """Gain in dB to bring a clip to the live median, clamped to +/-3 dB."""
        if clip_lufs is None:
            return 0.0
        med = self.median(voice)
        if med is None:
            return 0.0
        return max(-GAIN_CLAMP_DB, min(GAIN_CLAMP_DB, med - clip_lufs))


class EdgeTrimmer:
    """Streaming trim of edge silence on live audio that meets a cached clip.

    A provider synthesizing a clause alone pads it with silence it would not put
    mid-sentence; at a join with cached audio that doubles the pause. Leading
    silence is cut to ``lead_keep_ms`` (when ``trim_lead``); trailing silence is
    held back while it is silence and cut to ``tail_keep_ms`` at :meth:`finish`.
    Speech is never delayed: only silent audio waits.
    """

    def __init__(
        self,
        sr: int,
        *,
        trim_lead: bool,
        lead_keep_ms: int = 30,
        tail_keep_ms: int | None = None,
        threshold_dbfs: float = -45.0,
    ):
        self._sr = sr
        self._win = max(2, sr // 100) * 2  # 10 ms in bytes
        self._thr = 32768.0 * 10 ** (threshold_dbfs / 20)
        self._lead = trim_lead
        self._lead_keep = sr * lead_keep_ms // 1000 * 2
        self._tail_keep = None if tail_keep_ms is None else sr * tail_keep_ms // 1000 * 2
        self._pending = bytearray()  # leading silence, or trailing silence held back
        self._partial = bytearray()  # less than one window, not yet classified
        self._seen_speech = False
        self.lead_trimmed_s = 0.0
        self.tail_trimmed_s = 0.0

    def feed(self, chunk: bytes) -> bytes:
        buf = bytes(self._partial) + chunk
        n = len(buf) // self._win * self._win
        self._partial = bytearray(buf[n:])
        if not n:
            return b""
        x = np.frombuffer(buf[:n], dtype="<i2").astype(np.float32).reshape(-1, self._win // 2)
        loud = np.sqrt(np.mean(x * x, axis=1)) > self._thr
        # Windows are classified at once; the state machine then runs once per run
        # of equally loud windows instead of once per 10 ms window.
        edges = [0, *(np.flatnonzero(np.diff(loud.astype(np.int8))) + 1).tolist(), len(loud)]
        out = bytearray()
        for a, b in itertools.pairwise(edges):
            seg = buf[a * self._win : b * self._win]
            if loud[a]:
                if not self._seen_speech:
                    self._seen_speech = True
                    if self._lead and len(self._pending) > self._lead_keep:
                        cut = len(self._pending) - self._lead_keep
                        self.lead_trimmed_s += cut / (2.0 * self._sr)
                        del self._pending[:cut]
                out += self._pending
                self._pending.clear()
                out += seg
            else:
                self._pending += seg
                passthrough = not self._lead and not self._seen_speech
                if passthrough or (self._tail_keep is None and self._seen_speech):
                    out += self._pending
                    self._pending.clear()
        return bytes(out)

    @property
    def seen_speech(self) -> bool:
        return self._seen_speech

    def finish(self, tail_keep_ms: int | None = None) -> bytes:
        """Return the audio still held. ``tail_keep_ms`` overrides the constructor's."""
        rest = self._pending + self._partial
        self._pending.clear()
        self._partial.clear()
        keep = self._tail_keep if tail_keep_ms is None else self._sr * tail_keep_ms // 1000 * 2
        if keep is not None and self._seen_speech and len(rest) > keep:
            cut = len(rest) - keep
            cut -= cut % 2
            self.tail_trimmed_s += cut / (2.0 * self._sr)
            rest = rest[: len(rest) - cut]
        return bytes(rest)
