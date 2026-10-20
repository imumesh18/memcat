"""The entry blob: a small, versioned, checksummed container for one clip.

Layout (little-endian), frozen at format version 1::

    0   4  magic "MCAT"
    4   1  format version (1)
    5   1  codec: 1 PCM16, 2 mu-law, 3 Opus
    6   1  flags: bit 0 rendered, bit 1 captured, bit 2 warmed
    7   1  reserved (0)
    8   4  sample rate, uint32
    12  4  duration in ms, uint32
    16  4  loudness in LUFS x 100, int32 (INT32_MIN = unknown)
    20  2  word count N, uint16
    22  4  audio length A in bytes, uint32
    26  .. N words: uint32 offset ms, uint8 length L, L bytes of UTF-8
    ..  A  audio: PCM16 samples, mu-law bytes, or Opus as (uint16 length, packet)*
    -4  4  CRC32 of everything before it

Parsing uses ``struct`` and checks every length against the buffer. Anything
malformed raises :class:`CorruptEntryError`, which callers treat as a miss.
"""

from __future__ import annotations

import functools
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from memcat.core import pcm

__all__ = [
    "CODEC_MULAW",
    "CODEC_OPUS",
    "CODEC_PCM16",
    "FLAG_CAPTURED",
    "FLAG_RENDERED",
    "FLAG_WARMED",
    "CorruptEntryError",
    "Entry",
    "decode",
    "encode",
    "opus_available",
    "to_pcm16",
]

MAGIC = b"MCAT"
FORMAT_VERSION = 1
CODEC_PCM16 = 1
CODEC_MULAW = 2
CODEC_OPUS = 3
FLAG_RENDERED = 1
FLAG_CAPTURED = 2
FLAG_WARMED = 4
LUFS_UNKNOWN = -(2**31)

_HEADER = struct.Struct("<4sBBBBIIiHI")  # 26 bytes
_WORD_HEAD = struct.Struct("<IB")
_U16 = struct.Struct("<H")
_CRC = struct.Struct("<I")
OPUS_RATES = (8000, 12000, 16000, 24000, 48000)


class CorruptEntryError(ValueError):
    """The blob is not a valid Memcat entry (wrong magic, version, length or CRC)."""


@dataclass(frozen=True, slots=True)
class Entry:
    """One stored clip."""

    codec: int
    sample_rate: int
    duration_ms: int
    lufs: float | None
    words: tuple[tuple[str, int], ...]  # (word, offset_ms from clause start)
    audio: bytes
    flags: int = 0
    _opus_packets: tuple[bytes, ...] | None = field(default=None, compare=False, repr=False)


def encode(entry: Entry) -> bytes:
    """Serialise an entry. Words longer than 255 UTF-8 bytes are truncated."""
    if len(entry.words) > 0xFFFF:
        raise ValueError("too many words")
    parts: list[bytes] = []
    for word, offset in entry.words:
        # Cut on a character boundary: a split UTF-8 sequence would fail to decode.
        b = word.encode("utf-8")[:255].decode("utf-8", "ignore").encode("utf-8")
        parts.append(_WORD_HEAD.pack(max(0, int(offset)), len(b)))
        parts.append(b)
    lufs = LUFS_UNKNOWN if entry.lufs is None else round(entry.lufs * 100)
    head = _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        entry.codec,
        entry.flags & 0xFF,
        0,
        entry.sample_rate,
        entry.duration_ms,
        lufs,
        len(entry.words),
        len(entry.audio),
    )
    body = head + b"".join(parts) + entry.audio
    return body + _CRC.pack(zlib.crc32(body))


def decode(blob: bytes | bytearray | memoryview) -> Entry:
    """Parse and validate a blob.

    Raises:
        CorruptEntryError: on any malformed input; never anything else.
    """
    try:
        return _decode(bytes(blob))
    except CorruptEntryError:
        raise
    except Exception as e:  # struct.error, UnicodeDecodeError, ...
        raise CorruptEntryError(str(e)) from None


def _decode(blob: bytes) -> Entry:
    n = len(blob)
    if n < _HEADER.size + _CRC.size:
        raise CorruptEntryError("short blob")
    (crc,) = _CRC.unpack_from(blob, n - 4)
    if zlib.crc32(memoryview(blob)[: n - 4]) != crc:
        raise CorruptEntryError("crc mismatch")
    magic, version, codec, flags, _reserved, sr, dur, lufs, nwords, alen = _HEADER.unpack_from(
        blob, 0
    )
    if magic != MAGIC:
        raise CorruptEntryError("bad magic")
    if version != FORMAT_VERSION:
        raise CorruptEntryError(f"unknown version {version}")
    if codec not in (CODEC_PCM16, CODEC_MULAW, CODEC_OPUS):
        raise CorruptEntryError(f"unknown codec {codec}")
    if not 1000 <= sr <= 384_000:
        raise CorruptEntryError("bad sample rate")
    pos = _HEADER.size
    end = n - 4
    words: list[tuple[str, int]] = []
    for _ in range(nwords):
        if pos + _WORD_HEAD.size > end:
            raise CorruptEntryError("truncated words")
        off, ln = _WORD_HEAD.unpack_from(blob, pos)
        pos += _WORD_HEAD.size
        if pos + ln > end:
            raise CorruptEntryError("truncated word")
        words.append((blob[pos : pos + ln].decode("utf-8"), off))
        pos += ln
    if pos + alen != end:
        raise CorruptEntryError("audio length mismatch")
    audio = blob[pos:end]
    packets: tuple[bytes, ...] | None = None
    if codec == CODEC_OPUS:
        packets = _split_packets(audio)
    elif codec == CODEC_PCM16 and alen % 2:
        raise CorruptEntryError("odd PCM16 length")
    return Entry(
        codec=codec,
        sample_rate=sr,
        duration_ms=dur,
        lufs=None if lufs == LUFS_UNKNOWN else lufs / 100.0,
        words=tuple(words),
        audio=audio,
        flags=flags,
        _opus_packets=packets,
    )


def _split_packets(audio: bytes) -> tuple[bytes, ...]:
    out: list[bytes] = []
    pos, n = 0, len(audio)
    while pos < n:
        if pos + 2 > n:
            raise CorruptEntryError("truncated opus length")
        (ln,) = _U16.unpack_from(audio, pos)
        pos += 2
        if ln == 0 or pos + ln > n:
            raise CorruptEntryError("bad opus packet")
        out.append(audio[pos : pos + ln])
        pos += ln
    return tuple(out)


# --- codecs ------------------------------------------------------------------


@functools.cache
def opus_available() -> bool:
    """Whether PyAV (``memcat[opus]``) can be imported; checked once per process."""
    try:
        import av  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except Exception:
        return False
    return True


def encode_audio(pcm16: bytes, sample_rate: int, codec: int) -> bytes:
    """Encode mono PCM16 with the given codec (runs off the hot path)."""
    if codec == CODEC_PCM16:
        return pcm16
    if codec == CODEC_MULAW:
        return pcm.lin2ulaw(pcm16)
    if codec == CODEC_OPUS:
        return _opus_encode(pcm16, sample_rate)
    raise ValueError(f"unknown codec {codec}")


def to_pcm16(entry: Entry) -> bytes:
    """Decode the whole clip to PCM16 at its native rate."""
    if entry.codec == CODEC_PCM16:
        return entry.audio
    if entry.codec == CODEC_MULAW:
        return pcm.ulaw2lin(entry.audio)
    return b"".join(_opus_decode_iter(entry))


def _opus_encode(pcm16: bytes, sample_rate: int) -> bytes:
    import av
    import numpy as np

    if sample_rate not in OPUS_RATES:
        raise ValueError(f"Opus needs one of {OPUS_RATES} Hz, got {sample_rate}")
    ctx: Any = av.CodecContext.create("libopus", "w")
    ctx.sample_rate = sample_rate
    ctx.layout = "mono"
    ctx.format = "s16"
    ctx.bit_rate = 32_000 if sample_rate >= 16000 else 24_000
    ctx.options = {"application": "voip", "frame_duration": "20", "vbr": "on"}
    ctx.open()
    frame_samples = sample_rate // 50  # 20 ms
    samples = np.frombuffer(pcm16, dtype="<i2")
    pad = (-len(samples)) % frame_samples
    if pad:
        samples = np.concatenate([samples, np.zeros(pad, dtype="<i2")])
    out = bytearray()
    pts = 0
    packets: list[Any] = []
    for i in range(0, len(samples), frame_samples):
        chunk = samples[i : i + frame_samples].reshape(1, -1)
        fr: Any = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
        fr.sample_rate = sample_rate
        fr.pts = pts
        pts += frame_samples
        packets.extend(ctx.encode(fr))
    packets.extend(ctx.encode(None))
    for p in packets:
        b = bytes(p)
        out += _U16.pack(len(b)) + b
    return bytes(out)


_OPUS_PRESKIP_48K = 312  # libopus look-ahead (6.5 ms) at the 48 kHz decode rate


def _opus_decode_iter(entry: Entry) -> Iterator[bytes]:
    """Decode packet by packet at 48 kHz, then resample to the clip's rate with soxr.

    FFmpeg's Opus decoders always output 48 kHz; soxr (a Pipecat dependency) brings
    it back to the native rate far faster than PyAV's own resampler.
    """
    import av
    import numpy as np
    import soxr

    packets = entry._opus_packets
    if packets is None:
        packets = _split_packets(entry.audio)
    ctx: Any = av.CodecContext.create("libopus", "r")
    ctx.layout = "mono"
    ctx.open()
    rs: Any = (
        None
        if entry.sample_rate == 48000
        else soxr.ResampleStream(48000, entry.sample_rate, 1, dtype="int16", quality="HQ")
    )
    total = entry.sample_rate * entry.duration_ms // 1000
    pre = _OPUS_PRESKIP_48K
    emitted = 0

    def finish(mono: Any) -> Any:
        nonlocal emitted
        if emitted + len(mono) > total:
            mono = mono[: max(0, total - emitted)]
        emitted += len(mono)
        return mono

    for raw in packets:
        for fr in ctx.decode(av.Packet(raw)):
            m = fr.to_ndarray().reshape(-1)
            if m.dtype != np.int16:
                m = np.clip(m * 32768.0, -32768, 32767).astype(np.int16)
            if pre:
                cut = min(pre, len(m))
                m = m[cut:]
                pre -= cut
            if rs is not None:
                m = rs.resample_chunk(m)
            m = finish(m)
            if len(m):
                yield m.astype("<i2").tobytes()
    if rs is not None:
        m = finish(rs.resample_chunk(np.zeros(0, dtype=np.int16), last=True))
        if len(m):
            yield m.astype("<i2").tobytes()
