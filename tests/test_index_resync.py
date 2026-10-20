"""The key index relearns the store after its notification subscription drops."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import pytest

from memcat.core.index import KeyIndex
from memcat.core.store.base import RESYNC
from memcat.core.store.memory import MemoryStore
from memcat.core.store.redis import RedisStore


def d(i: int) -> bytes:
    return bytes([i]) * 16


async def _until(cond, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


class _Feed:
    """A MemoryStore view whose notifications the test drives by hand."""

    def __init__(self) -> None:
        self.inner = MemoryStore().bind("n")
        self.q: asyncio.Queue[bytes | Exception] = asyncio.Queue()
        self.subscriptions = 0

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    async def subscribe_new(self) -> AsyncIterator[bytes]:
        self.subscriptions += 1
        while True:
            item = await self.q.get()
            if isinstance(item, Exception):
                raise item
            yield item


async def test_resync_marker_rescans_the_store() -> None:
    store = _Feed()
    idx = KeyIndex(store, min_resync_interval_s=0)  # type: ignore[arg-type]
    idx.start()
    await _until(lambda: idx.loaded)
    # Stored while the subscription was down: the notification never arrives.
    await store.inner.put(d(1), b"x", 60)
    await asyncio.sleep(0.05)
    assert d(1) not in idx
    store.q.put_nowait(RESYNC)
    await _until(lambda: d(1) in idx)
    await idx.stop()


async def test_failed_subscription_rescans_on_resubscribe() -> None:
    store = _Feed()
    idx = KeyIndex(store, min_resync_interval_s=0)  # type: ignore[arg-type]
    idx.start()
    await _until(lambda: idx.loaded and store.subscriptions == 1)
    await store.inner.put(d(2), b"x", 60)
    store.q.put_nowait(ConnectionError("dropped"))
    await _until(lambda: d(2) in idx)
    assert store.subscriptions == 2
    await idx.stop()


async def test_resync_during_a_scan_scans_again() -> None:
    store = _Feed()
    gate = asyncio.Event()
    real_scan = store.inner.scan

    async def slow_scan() -> AsyncIterator[bytes]:
        found = [x async for x in real_scan()]  # what the store held when the scan began
        await gate.wait()
        for x in found:
            yield x

    store.scan = slow_scan  # type: ignore[method-assign]
    idx = KeyIndex(store, min_resync_interval_s=0)  # type: ignore[arg-type]
    idx.start()
    await asyncio.sleep(0.02)
    await store.inner.put(d(3), b"x", 60)  # after the first scan began: it misses this
    store.q.put_nowait(RESYNC)  # arrives while the first scan is still running
    await asyncio.sleep(0.02)
    gate.set()
    await _until(lambda: d(3) in idx)  # learned by the second pass
    await idx.stop()


def _redis_url() -> str:
    urls = [u for u in os.environ.get("MEMCAT_REDIS_URLS", "").split(",") if u]
    if not urls:
        pytest.skip("set MEMCAT_REDIS_URLS to run against Redis")
    return urls[0]


async def test_redis_reconnect_resyncs_the_index() -> None:
    url = _redis_url()
    root = RedisStore(url, op_timeout_ms=500)
    store = root.bind(f"t{uuid.uuid4().hex[:10]}")
    idx = KeyIndex(store, min_resync_interval_s=0)
    idx.start()
    try:
        await _until(lambda: idx.loaded)
        await asyncio.sleep(0.2)  # let the subscription settle
        await store.put(d(4), b"x", 60)  # its notification is lost
        assert d(4) not in idx
        import redis.asyncio as aioredis

        killer = aioredis.Redis.from_url(url)
        await killer.execute_command("CLIENT", "KILL", "TYPE", "pubsub")
        await killer.aclose()
        await _until(lambda: d(4) in idx, timeout=10.0)
    finally:
        await idx.stop()
        await store.delete(None)
        await root.aclose()


class _SlowScan(_Feed):
    """Scans yield one digest at a time, with a pause, and are counted."""

    def __init__(self, digests: list[bytes]) -> None:
        super().__init__()
        self.digests = digests
        self.scans = 0
        self.mid_scan = asyncio.Event()

    async def scan(self) -> AsyncIterator[bytes]:
        self.scans += 1
        for x in self.digests:
            yield x
            self.mid_scan.set()
            await asyncio.sleep(0.02)


async def test_purge_during_a_scan_is_not_undone() -> None:
    store = _SlowScan([d(1), d(2), d(3)])
    idx = KeyIndex(store, min_resync_interval_s=0)  # type: ignore[arg-type]
    idx.start()
    await store.mid_scan.wait()
    idx.discard(d(1))  # a purge arrives while the scan still holds d(1)
    await _until(lambda: idx.loaded)
    assert d(1) not in idx and d(2) in idx and d(3) in idx
    await idx.stop()


async def test_wipe_during_a_scan_rescans() -> None:
    store = _SlowScan([d(1), d(2)])
    idx = KeyIndex(store, min_resync_interval_s=0)  # type: ignore[arg-type]
    idx.start()
    await store.mid_scan.wait()
    store.digests = []  # the namespace is wiped while the first scan runs
    idx.clear()
    await _until(lambda: idx.loaded)
    assert store.scans == 2 and len(idx) == 0
    await idx.stop()


async def test_flapping_connection_does_not_rescan_back_to_back() -> None:
    store = _SlowScan([d(1)])
    idx = KeyIndex(store, min_resync_interval_s=0.5)  # type: ignore[arg-type]
    idx.start()
    await _until(lambda: idx.loaded)
    for _ in range(10):
        store.q.put_nowait(RESYNC)  # ten reconnects in quick succession
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.3)
    assert store.scans <= 2  # the first load plus at most one resync so far
    await _until(lambda: store.scans >= 2, timeout=2.0)  # the resync still happens
    await idx.stop()
