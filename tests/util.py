from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pipecat.frames.frames import TTSSpeakFrame

from memcat import MemcatConfig, cached
from memcat.core.runtime import get_runtime
from tests.fakes import FakeCursorWSTTS, FakeELWSTTS, FakeHTTPTTS, FakeWSTTS
from tests.harness import hear

VARIANTS: dict[str, tuple[type, dict[str, Any]]] = {
    "http": (FakeHTTPTTS, {}),
    "http-ts": (FakeHTTPTTS, {"timestamps": True}),
    "ws-ctx": (FakeWSTTS, {}),
    "ws-cursor": (FakeCursorWSTTS, {}),
    "ws-el": (FakeELWSTTS, {}),
}
WS = ["ws-ctx", "ws-cursor", "ws-el"]
TIMESTAMPED = ["http-ts", "ws-ctx", "ws-el"]


def make(variant: str, cfg: MemcatConfig, *, seasoned: bool = True, **extra: Any) -> Any:
    """A wrapped fake. ``seasoned`` gives a flush budget two quiet minutes of history,
    as if the call had been going on a while (a fresh call spends no extra flush for
    its first 30 s)."""
    import time

    cls, kw = VARIANTS[variant]
    svc = cached(cls)(memcat=cfg, **{**kw, **extra})
    if seasoned and svc._memcat_budget is not None:
        svc._memcat_budget._t0 = time.monotonic() - 120
    return svc


async def drain(cfg: MemcatConfig) -> None:
    await get_runtime(cfg).worker.drain_all(20)


async def prime(variant: str, cfg_factory: Callable[[], MemcatConfig], phrases: list[str]) -> None:
    """Speak each phrase alone so its clauses get captured or rendered."""
    tts = make(variant, cfg_factory())
    await hear(tts, [TTSSpeakFrame(p) for p in phrases])
    await drain(tts._memcat_cfg)
