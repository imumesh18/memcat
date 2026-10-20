"""Provider keys for recording: the environment first, then ``.env.test`` (gitignored)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def get(name: str) -> str:
    if v := os.environ.get(name, "").strip():
        return v
    path = ROOT / ".env.test"
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == name:
                return v.strip().strip("'\"")
    return ""


def fixtures_dir(provider: str) -> Path:
    """Where recordings live: ``$MEMCAT_FIXTURES_DIR/<provider>``, default
    ``tests/fixtures/<provider>``. Recordings are never committed (gitignored)."""
    base = os.environ.get("MEMCAT_FIXTURES_DIR", "").strip()
    return (Path(base).expanduser() if base else ROOT / "tests" / "fixtures") / provider


def recording() -> bool:
    return os.environ.get("MEMCAT_RECORD") == "1"


def speed() -> float:
    return float(os.environ.get("MEMCAT_REPLAY_SPEED", "1"))
