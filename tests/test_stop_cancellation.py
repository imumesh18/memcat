"""Stopping background work never swallows a cancellation of the caller."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from memcat.core.audio import LoudnessTracker
from memcat.core.breaker import Breaker
from memcat.core.index import KeyIndex
from memcat.core.lookup import L1Cache
from memcat.core.metrics import Metrics
from memcat.core.store.memory import MemoryStore
from memcat.core.worker import Worker, WorkerConfig


class _SlowToCancel:
    """A store whose subscription takes a moment to wind down when cancelled."""

    def __init__(self) -> None:
        self.inner = MemoryStore().bind("n")

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    async def subscribe_new(self) -> AsyncIterator[bytes]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)
            raise
        yield b""  # pragma: no cover


async def _cancel_while_stopping(stop) -> asyncio.Task[None]:
    task = asyncio.ensure_future(stop())
    await asyncio.sleep(0.05)  # inside stop(), waiting for the background tasks
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return task


async def test_index_stop_propagates_cancellation() -> None:
    idx = KeyIndex(_SlowToCancel())  # type: ignore[arg-type]
    idx.start()
    await asyncio.sleep(0.02)
    task = await _cancel_while_stopping(idx.stop)
    assert task.cancelled()


async def test_index_stop_completes_normally() -> None:
    idx = KeyIndex(_SlowToCancel())  # type: ignore[arg-type]
    idx.start()
    await asyncio.sleep(0.02)
    await idx.stop()  # the tasks' own cancellation does not escape


async def test_worker_stop_propagates_cancellation() -> None:
    store = _SlowToCancel()
    idx = KeyIndex(store)  # type: ignore[arg-type]
    w = Worker(
        store,  # type: ignore[arg-type]
        idx,
        L1Cache(1 << 20),
        Breaker(),
        Metrics(),
        LoudnessTracker(),
        WorkerConfig(drain_s=0.0),
    )
    w.start()
    gate = asyncio.Event()

    async def slow(*_a: object) -> None:
        try:
            await gate.wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)
            raise

    w._handle = slow  # type: ignore[method-assign]
    w.touch([bytes(16)])
    await asyncio.sleep(0.02)
    task = await _cancel_while_stopping(w.stop)
    assert task.cancelled()
