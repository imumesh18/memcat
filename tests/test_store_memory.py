"""MemoryStore regressions: full subscriber queues and the lock table."""

from __future__ import annotations

import asyncio

from memcat.core.store import memory
from memcat.core.store.base import RESYNC
from memcat.core.store.memory import MemoryStore


def d(i: int) -> bytes:
    return i.to_bytes(16, "big")


async def test_full_queue_resyncs_instead_of_dropping(monkeypatch) -> None:
    monkeypatch.setattr(memory, "_MAX_QUEUED", 5)
    s = MemoryStore().bind("n")
    it = s.subscribe_new()
    first = asyncio.ensure_future(it.__anext__())
    await asyncio.sleep(0)  # subscribed
    for i in range(12):
        await s.publish_new(d(i))
    got = [await first]
    while True:
        try:
            got.append(await asyncio.wait_for(it.__anext__(), 0.05))
        except TimeoutError:
            break
    await it.aclose()
    assert RESYNC in got
    assert got[-1] == d(11)  # the newest notification always arrives
    assert len(got) <= 7


async def test_unlock_only_releases_own_lock() -> None:
    s = MemoryStore().bind("n")
    release = asyncio.Event()

    async def slow() -> None:
        assert await s.try_lock(d(1), 0)  # expires at once: its render ran long
        await release.wait()
        await s.unlock(d(1))

    t = asyncio.ensure_future(slow())
    await asyncio.sleep(0)
    assert await s.try_lock(d(1), 60)
    release.set()
    await t
    assert not await s.try_lock(d(1), 60), "the late unlock released someone else's lock"
    await s.unlock(d(1))
    assert await s.try_lock(d(1), 60)


async def test_expired_locks_are_pruned() -> None:
    root = MemoryStore()
    s = root.bind("n")

    async def take(i: int) -> None:
        assert await s.try_lock(d(i), 0)  # taken and abandoned (expires at once)

    for i in range(5000):
        await asyncio.ensure_future(take(i))
    assert len(root._shared.locks) <= 2 * memory._LOCK_SWEEP
    assert len(s._tokens) <= 2 * memory._LOCK_SWEEP


async def test_find_prefixes() -> None:
    s = MemoryStore().bind("n")
    other = MemoryStore().bind("n")
    await s.put(b"\xaa" + d(1)[1:], b"x", 60)
    await s.put(b"\xbb" + d(2)[1:], b"x", 60)
    await s.observe(b"\xcc" + d(3)[1:], "c", 3)
    await other.put(b"\xaa" + d(4)[1:], b"x", 60)
    got = await s.find_prefixes([b"\xaa", b"\xcc"])
    assert got == sorted([b"\xaa" + d(1)[1:], b"\xcc" + d(3)[1:]])
    assert await s.find_prefixes([]) == []
    assert await s.find_prefixes([b"\xbb"]) == [b"\xbb" + d(2)[1:]]


def test_publish_from_another_thread_reaches_the_listener() -> None:
    import threading

    store = MemoryStore()
    got: list[bytes] = []
    ready = threading.Event()

    async def listen() -> None:
        sub = store.bind("x").subscribe_new()
        task = asyncio.ensure_future(sub.__anext__())
        await asyncio.sleep(0.05)  # subscribed
        ready.set()
        got.append(await asyncio.wait_for(task, 5))
        await sub.aclose()

    t = threading.Thread(target=lambda: asyncio.run(listen()))
    t.start()
    assert ready.wait(5)
    asyncio.run(store.bind("x").publish_new(b"\x01" * 16))
    t.join(5)
    assert got == [b"\x01" * 16]
