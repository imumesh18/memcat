from __future__ import annotations

import contextlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memcat.core import codec
from memcat.core.codec import CorruptEntryError, Entry


def _pcm(sr: int, seconds: float) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2").tobytes()


@pytest.mark.parametrize("cod", [codec.CODEC_PCM16, codec.CODEC_MULAW, codec.CODEC_OPUS])
def test_roundtrip(cod: int) -> None:
    sr = 8000 if cod == codec.CODEC_MULAW else 24000
    pcm = _pcm(sr, 1.5)
    e = Entry(
        codec=cod,
        sample_rate=sr,
        duration_ms=1500,
        lufs=-23.4,
        words=(("héllo", 0), ("world", 700)),
        audio=codec.encode_audio(pcm, sr, cod),
        flags=3,
    )
    d = codec.decode(codec.encode(e))
    assert (d.codec, d.sample_rate, d.duration_ms, d.flags, d.words) == (cod, sr, 1500, 3, e.words)
    assert d.lufs == pytest.approx(-23.4)
    out = codec.to_pcm16(d)
    assert len(out) == len(pcm)


def test_storage_per_minute_telephony() -> None:
    sr = 8000
    pcm = _pcm(sr, 60)
    blob = codec.encode(
        Entry(codec.CODEC_MULAW, sr, 60_000, None, (), codec.encode_audio(pcm, sr, 2))
    )
    assert len(blob) <= 480 * 1024


def test_unknown_lufs_roundtrips() -> None:
    e = Entry(codec.CODEC_PCM16, 16000, 10, None, (), b"\x00\x00" * 160)
    assert codec.decode(codec.encode(e)).lufs is None


@given(st.binary(max_size=300))
@settings(max_examples=1500, deadline=None)
def test_corrupt_blob_is_a_miss(blob: bytes) -> None:
    with contextlib.suppress(CorruptEntryError):
        codec.decode(blob)


@given(st.integers(min_value=0, max_value=200), st.integers(min_value=0, max_value=255))
@settings(max_examples=400, deadline=None)
def test_bit_flips_are_caught(pos: int, flip: int) -> None:
    e = Entry(codec.CODEC_PCM16, 16000, 20, -20.0, (("a", 1),), b"\x01\x02" * 160)
    blob = bytearray(codec.encode(e))
    pos %= len(blob)
    if flip == 0:
        return
    blob[pos] ^= flip
    with pytest.raises(CorruptEntryError):
        codec.decode(bytes(blob))


def test_wrong_magic_and_version() -> None:
    blob = bytearray(codec.encode(Entry(codec.CODEC_PCM16, 16000, 1, None, (), b"\0\0")))
    assert blob[:4] == b"MCAT"
    for pos, val in ((0, ord("X")), (4, 9)):
        b = bytearray(blob)
        b[pos] = val
        import zlib

        b[-4:] = zlib.crc32(bytes(b[:-4])).to_bytes(4, "little")
        with pytest.raises(CorruptEntryError):
            codec.decode(bytes(b))
