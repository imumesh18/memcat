"""Bytes per cached minute of speech, by codec and sample rate (real Kokoro speech).

uv run python bench/storage.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

from _driver import Driver  # noqa: E402
from pipecat.services.kokoro.tts import KokoroTTSService  # noqa: E402

from memcat.core import codec  # noqa: E402
from memcat.core.worker import Audio, _prepare  # noqa: E402

TEXT = (
    "Thank you for calling Acme support, my name is Ava and I will be helping you today. "
    "I can see your account here, and I am happy to look into that order for you. "
    "Please give me a moment while I check the latest status with our shipping partner."
)


async def speech(sr: int) -> bytes:
    d = Driver(
        KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart"), sample_rate=sr), sr
    )
    await d.start()
    pcm = b""
    for sentence in TEXT.split(". "):
        _, part = await d.speak(sentence.strip().rstrip(".") + ".")
        pcm += part
    await d.stop()
    return pcm


def main() -> None:
    import numpy as np
    import soxr

    out: dict[str, dict[str, float]] = {}
    src = asyncio.run(speech(24000))
    for sr in (8000, 16000, 24000):
        x = np.frombuffer(src, dtype="<i2")
        pcm = src if sr == 24000 else soxr.resample(x, 24000, sr).astype("<i2").tobytes()
        minutes = len(pcm) / (2 * sr) / 60
        for name, cod in (
            ("pcm16", codec.CODEC_PCM16),
            ("mulaw", codec.CODEC_MULAW),
            ("opus", codec.CODEC_OPUS),
        ):
            if cod == codec.CODEC_OPUS and (
                not codec.opus_available() or sr not in codec.OPUS_RATES
            ):
                continue
            blob, _ = _prepare(Audio(pcm, sr), cod, 0)
            assert blob is not None
            out[f"{name}@{sr}"] = {"kb_per_min": round(len(blob) / 1024 / minutes, 1)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
