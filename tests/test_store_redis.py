"""Redis/Valkey store regressions: pool waits, per-operation budgets, lock ownership,
idle time, event loops. Needs MEMCAT_REDIS_URLS (comma-separated redis:// URLs)."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from urllib.parse import urlparse

import pytest

from memcat.core.store.base import StoreError
from memcat.core.store.redis import RedisStore

URLS = [u for u in os.environ.get("MEMCAT_REDIS_URLS", "").split(",") if u]
pytestmark = pytest.mark.skipif(not URLS, reason="set MEMCAT_REDIS_URLS")


def d(i: int) -> bytes:
    return bytes([i]) * 16


@pytest.fixture(params=URLS or ["unused"])
async def url(request):
    return request.param


def _ns() -> str:
    return f"t{uuid.uuid4().hex[:10]}"


async def _cleanup(root: RedisStore, ns: str) -> None:
    await root.bind(ns).delete(None)
    await root.aclose()


async def test_pool_waits_instead_of_failing(url: str) -> None:
    root = RedisStore(url, pool_size=2, op_timeout_ms=500)
    ns = _ns()
    b = root.bind(ns)
    # Two namespaces' subscriptions, as two runtimes' key indexes hold them.
    subs = [asyncio.ensure_future(root.bind(f"{ns}{x}").subscribe_new().__anext__()) for x in "ab"]
    await asyncio.sleep(0.3)
    try:
        res = await asyncio.gather(*(b.get_many([d(1)]) for _ in range(20)), return_exceptions=True)
        assert [r for r in res if isinstance(r, Exception)] == []
    finally:
        for t in subs:
            t.cancel()
        await asyncio.gather(*subs, return_exceptions=True)
        await _cleanup(root, ns)


class _SlowProxy:
    """TCP proxy that delays every server reply by ``delay`` seconds."""

    def __init__(self, url: str) -> None:
        u = urlparse(url)
        self.host, self.port = u.hostname or "localhost", u.port or 6379
        self.db = u.path or "/0"
        self.delay = 0.0
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> str:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        return f"redis://127.0.0.1:{port}{self.db}"

    async def _handle(self, cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        sr, sw = await asyncio.open_connection(self.host, self.port)

        async def pump(r: asyncio.StreamReader, w: asyncio.StreamWriter, slow: bool) -> None:
            try:
                while data := await r.read(65536):
                    if slow and self.delay:
                        await asyncio.sleep(self.delay)
                    w.write(data)
                    await w.drain()
            except Exception:
                pass
            finally:
                w.close()

        await asyncio.gather(pump(cr, sw, False), pump(sr, cw, True))

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()


async def test_longer_budgets_apply_beyond_the_read_budget(url: str) -> None:
    proxy = _SlowProxy(url)
    purl = await proxy.start()
    root = RedisStore(purl, op_timeout_ms=50)
    ns = _ns()
    b = root.bind(ns)
    try:
        assert await b.get_many([d(1)]) == [None]  # warm a pooled connection
        proxy.delay = 0.4  # over the old 250 ms socket timeout, under the 2 s write budget
        t0 = time.monotonic()
        with pytest.raises(StoreError):
            await b.get_many([d(1)])  # the hot-path read keeps its 50 ms budget
        assert time.monotonic() - t0 < 0.3
        # The cancelled read's connection is dropped (its reply was never read); open a
        # new one at full speed, since the proxy would also slow its handshake.
        proxy.delay = 0.0
        assert await b.get_many([d(1)]) == [None]
        proxy.delay = 0.4
        await b.put(d(2), b"x", 60)
        assert await b.try_lock(d(3), 60)
        await b.unlock(d(3))
        proxy.delay = 0.0
        assert await b.get_many([d(2)]) == [b"x"]
    finally:
        proxy.delay = 0.0
        await _cleanup(root, ns)
        await proxy.stop()


async def test_unlock_only_releases_own_lock_across_processes(url: str) -> None:
    ns = _ns()
    proc_a, proc_b = RedisStore(url, op_timeout_ms=500), RedisStore(url, op_timeout_ms=500)
    a, b = proc_a.bind(ns), proc_b.bind(ns)
    try:
        assert await a.try_lock(d(3), 1)
        await asyncio.sleep(1.2)  # A's render outlived its lock
        assert await b.try_lock(d(3), 60)
        await a.unlock(d(3))  # A finishes late
        assert not await a.try_lock(d(3), 60), "B's lock was released by A"
        await b.unlock(d(3))
        assert await a.try_lock(d(3), 60)
        await a.unlock(d(3))
    finally:
        await _cleanup(proc_a, ns)
        await proc_b.aclose()


async def test_unlock_only_releases_own_lock_within_a_process(url: str) -> None:
    root = RedisStore(url, op_timeout_ms=500)
    ns = _ns()
    s = root.bind(ns)
    first_unlock = asyncio.Event()

    async def slow_render() -> None:
        assert await s.try_lock(d(4), 1)
        await first_unlock.wait()
        await s.unlock(d(4))

    async def later_render() -> None:
        assert await s.try_lock(d(4), 60)
        first_unlock.set()
        await asyncio.sleep(0.2)
        assert not await s.try_lock(d(4), 60)
        await s.unlock(d(4))

    try:
        t = asyncio.ensure_future(slow_render())
        await asyncio.sleep(1.2)
        await later_render()
        await t
        assert await s.try_lock(d(4), 60)
    finally:
        await _cleanup(root, ns)


async def test_idle_seconds_reads_the_timestamp(url: str) -> None:
    import redis.asyncio as aioredis

    root = RedisStore(url, op_timeout_ms=500)
    ns = _ns()
    s = root.bind(ns)
    ctl = aioredis.Redis.from_url(url)
    try:
        await s.put(d(1), b"fresh", 3600)
        await s.put(d(2), b"old", 3600)
        await ctl.set(f"memcat:{ns}:t:{d(2).hex()}", int(time.time()) - 5000)
        fresh, old, gone = await s.idle_seconds([d(1), d(2), d(9)])
        assert fresh is not None and fresh < 5
        assert old is not None and 4990 < old < 5010
        assert gone is None
        await ctl.set(f"memcat:{ns}:t:{d(2).hex()}", 0)
        await s.touch([d(2)], 3600)  # served again
        (old,) = await s.idle_seconds([d(2)])
        assert old is not None and old < 5
        assert await s.delete([d(1), d(2)]) == 2
        assert await ctl.exists(f"memcat:{ns}:t:{d(1).hex()}", f"memcat:{ns}:t:{d(2).hex()}") == 0
    finally:
        await ctl.aclose()
        await _cleanup(root, ns)


async def test_publish_many(url: str) -> None:
    root = RedisStore(url, op_timeout_ms=500)
    ns = _ns()
    s = root.bind(ns)
    got: list[bytes] = []

    async def listen() -> None:
        async for x in s.subscribe_new():
            got.append(x)
            if len(got) == 600:
                return

    t = asyncio.ensure_future(listen())
    await asyncio.sleep(0.3)
    try:
        msgs = [b"\x01" + i.to_bytes(16, "big") for i in range(600)]
        await s.publish_many(msgs)
        await asyncio.wait_for(t, 5)
        assert got == msgs
    finally:
        t.cancel()
        await _cleanup(root, ns)


def test_store_survives_a_second_event_loop(url: str) -> None:
    root = RedisStore(url, op_timeout_ms=500)
    ns = _ns()
    s = root.bind(ns)

    async def first() -> None:
        await s.put(d(1), b"one", 60)
        assert await s.get_many([d(1)]) == [b"one"]  # leaves a client bound to this loop

    async def second() -> None:
        assert await s.get_many([d(1)]) == [b"one"]
        assert await s.try_lock(d(2), 60)
        await s.unlock(d(2))
        await _cleanup(root, ns)

    asyncio.run(first())
    asyncio.run(second())
    assert len(root._conns) <= 1  # the first loop's connections were let go


async def test_find_prefixes_one_scan_per_kind(url: str, monkeypatch) -> None:
    root = RedisStore(url, op_timeout_ms=500)
    ns = _ns()
    s = root.bind(ns)
    try:
        await s.put(b"\xaa\x01" + d(1)[2:], b"x", 60)
        await s.put(b"\xaa\x02" + d(2)[2:], b"x", 60)
        await s.put(b"\xbb\x01" + d(3)[2:], b"x", 60)
        await s.observe(b"\xcc\x01" + d(4)[2:], "c", 3)
        await root.bind(ns + "x").put(b"\xaa\x01" + d(5)[2:], b"x", 60)
        patterns: list[str] = []
        real = s._scan_keys

        def counting(match):
            patterns.append(match)
            return real(match)

        monkeypatch.setattr(s, "_scan_keys", counting)
        got = await s.find_prefixes([b"\xaa\x01", b"\xaa\x02", b"\xcc", b"\xdd\xee"])
        assert got == sorted(
            [b"\xaa\x01" + d(1)[2:], b"\xaa\x02" + d(2)[2:], b"\xcc\x01" + d(4)[2:]]
        )
        assert len(patterns) == 2
        assert await s.find_prefixes([b"\xbb\x01"]) == [b"\xbb\x01" + d(3)[2:]]
        assert await s.find_prefixes([]) == []
    finally:
        await root.bind(ns + "x").delete(None)
        await _cleanup(root, ns)


async def test_bulk_reads_get_the_bulk_budget(url: str) -> None:
    proxy = _SlowProxy(url)
    purl = await proxy.start()
    root = RedisStore(purl, op_timeout_ms=50)
    ns = _ns()
    b = root.bind(ns)
    try:
        await b.put(d(1), b"x", 60)
        proxy.delay = 0.3
        with pytest.raises(StoreError):
            await b.get_many([d(1)])
        proxy.delay = 0.0  # a fresh connection (the timed-out one is dropped) at full speed
        assert await b.get_many([d(1)]) == [b"x"]
        proxy.delay = 0.3
        assert await b.get_bulk([d(1), d(2)]) == [b"x", None]
    finally:
        proxy.delay = 0.0
        await _cleanup(root, ns)
        await proxy.stop()


async def test_touch_leaves_no_timestamp_for_a_purged_entry(url: str) -> None:
    root = RedisStore(url, op_timeout_ms=500)
    ns = _ns()
    b = root.bind(ns)
    try:
        await b.put(d(1), b"x", 60)
        await b.delete([d(1)])
        await b.touch([d(1), d(2)], 60)  # served from memory just as it was purged
        assert await root._conn().client.keys(f"memcat:{ns}:*") == []
        await b.put(d(3), b"y", 60)
        await b.touch([d(3)], 60)
        assert (await b.idle_seconds([d(3)]))[0] is not None
    finally:
        await _cleanup(root, ns)
