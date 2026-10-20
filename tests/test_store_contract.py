"""One contract, every backend: MemoryStore, Redis 7/8 and Valkey 8, S3 (moto).

Set MEMCAT_REDIS_URLS to a comma-separated list of redis:// URLs to include them.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from memcat.core.store.memory import MemoryStore
from memcat.core.store.redis import RedisStore


def _backends() -> list[str]:
    urls = [u for u in os.environ.get("MEMCAT_REDIS_URLS", "").split(",") if u]
    return ["memory", *urls, "s3"]


@pytest.fixture(params=_backends())
async def store(request):
    kind = request.param
    ns = f"t{uuid.uuid4().hex[:10]}"
    if kind == "memory":
        yield MemoryStore().bind(ns)
        return
    if kind == "s3":
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        from memcat.core.store.s3 import S3Store

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="memcat-test")
            yield S3Store("memcat-test", client=client).bind(ns)
        return
    root = RedisStore(kind, op_timeout_ms=500)
    b = root.bind(ns)
    yield b
    await b.delete(None)
    await root.aclose()


def d(i: int) -> bytes:
    return bytes([i]) * 16


async def test_put_get_touch_delete(store) -> None:
    assert await store.get_many([d(1), d(2)]) == [None, None]
    await store.put(d(1), b"one", 3600)
    await store.put(d(2), b"two", 3600)
    assert await store.get_many([d(1), d(3), d(2)]) == [b"one", None, b"two"]
    await store.touch([d(1), d(2)], 7200)
    got = sorted([x async for x in store.scan()])
    assert got == [d(1), d(2)]
    assert await store.delete([d(1)]) == 1
    assert await store.get_many([d(1)]) == [None]
    assert await store.delete(None) >= 1
    assert [x async for x in store.scan()] == []


async def test_admission_counts_distinct_calls(store) -> None:
    assert not await store.observe(d(5), "call-a", 3)
    assert not await store.observe(d(5), "call-a", 3)  # same call twice
    assert not await store.observe(d(5), "call-b", 3)
    assert await store.observe(d(5), "call-c", 3)
    # admitted keys are reset, so counting starts over
    assert not await store.observe(d(5), "call-d", 3)


async def test_forget_admission(store) -> None:
    await store.observe(d(6), "a", 3)
    await store.observe(d(6), "b", 3)
    await store.forget_admission([d(6)])
    assert not await store.observe(d(6), "c", 3)


async def test_lock(store) -> None:
    assert await store.try_lock(d(7), 60)
    assert not await store.try_lock(d(7), 60)
    await store.unlock(d(7))
    assert await store.try_lock(d(7), 60)


async def test_pubsub(store) -> None:
    if type(store).__name__ == "_BoundS3Store":
        pytest.skip("S3 has no notifications")
    got: list[bytes] = []

    async def listen() -> None:
        async for x in store.subscribe_new():
            got.append(x)
            return

    t = asyncio.ensure_future(listen())
    await asyncio.sleep(0.2)
    await store.publish_new(d(9))
    await asyncio.wait_for(t, 3)
    assert got == [d(9)]


async def test_namespaces_are_isolated() -> None:
    root = MemoryStore()
    a, b = root.bind("a"), root.bind("b")
    await a.put(d(1), b"x", 60)
    assert await b.get_many([d(1)]) == [None]
    assert await b.delete(None) == 0
    assert await a.get_many([d(1)]) == [b"x"]


async def test_memory_store_is_byte_bounded() -> None:
    s = MemoryStore(max_bytes=10).bind("n")
    await s.put(d(1), b"12345", 60)
    await s.put(d(2), b"12345", 60)
    await s.put(d(3), b"12345", 60)
    assert await s.get_many([d(1), d(2), d(3)]) == [None, b"12345", b"12345"]


def test_store_constructors_fail_fast(monkeypatch) -> None:
    from memcat.core.store import redis as redis_mod
    from memcat.core.store import s3 as s3_mod

    with pytest.raises(ValueError, match="redis://"):
        RedisStore("http://nope")
    # Without the extra installed, say so at startup, not as a failure on every call.
    monkeypatch.setattr(redis_mod, "find_spec", lambda _name: None)
    monkeypatch.setattr(s3_mod, "find_spec", lambda _name: None)
    with pytest.raises(ImportError, match=r"pip install 'memcat\[redis\]'"):
        RedisStore()
    with pytest.raises(ImportError, match=r"pip install 'memcat\[s3\]'"):
        s3_mod.S3Store("bucket")


async def test_redis_timeout_is_a_store_error() -> None:
    from memcat.core.store.base import StoreError

    s = RedisStore("redis://10.255.255.1:6379/0", op_timeout_ms=50).bind("n")
    with pytest.raises(StoreError):
        await s.get_many([d(1)])


async def test_redis_bulk_operations_fail_as_store_errors() -> None:
    from memcat.core.store.base import StoreError

    s = RedisStore("redis://10.255.255.1:6379/0", op_timeout_ms=50).bind("n")
    with pytest.raises(StoreError):
        _ = [x async for x in s.scan()]
    with pytest.raises(StoreError):
        await s.delete(None)
    with pytest.raises(StoreError):
        await s.forget_admission(None)


class _BrokenS3:
    """A boto3-shaped client whose every call fails, or never returns."""

    def __init__(self, hang: bool = False) -> None:
        self._hang = hang

    def _fail(self, **_kw: object) -> object:
        if self._hang:
            import time

            time.sleep(0.5)
        raise ConnectionError("endpoint unreachable")

    list_objects_v2 = delete_objects = get_object = put_object = head_object = _fail


@pytest.mark.parametrize("hang", [False, True])
async def test_s3_failures_are_store_errors(hang: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from memcat.core.store import s3
    from memcat.core.store.base import StoreError

    monkeypatch.setattr(s3, "_S3_TIMEOUT_S", 0.1)
    b = s3.S3Store("bucket", client=_BrokenS3(hang)).bind("n")
    with pytest.raises(StoreError):
        _ = [x async for x in b.scan()]
    with pytest.raises(StoreError):
        await b.delete([d(1)])
    with pytest.raises(StoreError):
        await b.delete(None)
    with pytest.raises(StoreError):
        await b.get_many([d(1)])
    if hang:
        with pytest.raises(StoreError):
            await b.idle_seconds([d(1)])
    else:
        assert await b.idle_seconds([d(1)]) == [None]


async def test_find_prefix_covers_entries_and_admission_counts(store) -> None:
    a1, a2, a3 = (
        b"\xaa\xbb\xcc\xdd" + d(1)[4:],
        b"\xaa\xbb\xcc\xdd" + d(2)[4:],
        b"\xaa\xbb\xcc\xdd" + d(3)[4:],
    )
    other = b"\x11\x22\x33\x44" + d(4)[4:]
    await store.put(a1, b"x", 3600)
    await store.put(a2, b"y", 3600)
    await store.put(other, b"z", 3600)
    await store.observe(a3, "call-a", 3)  # counted, not cached yet
    got = await store.find_prefixes([b"\xaa\xbb\xcc\xdd"])
    assert sorted(got) == sorted([a1, a2, a3])
