"""S3Store regressions: client creation, delete errors, identity, fan-out."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pytest

from memcat.core.store import s3
from memcat.core.store.base import StoreError

pytest.importorskip("moto")


def d(i: int) -> bytes:
    return i.to_bytes(16, "big")


@pytest.fixture
def aws(monkeypatch):
    import boto3
    from moto import mock_aws

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="memcat-test")
        yield


async def test_one_client_under_concurrent_first_use(aws, monkeypatch) -> None:
    import boto3

    created: list[object] = []
    real = boto3.client

    def counting(*a, **k):
        time.sleep(0.05)  # widen the race window
        c = real(*a, **k)
        created.append(c)
        return c

    monkeypatch.setattr(boto3, "client", counting)
    b = s3.S3Store("memcat-test").bind("ns")
    await asyncio.gather(*(b.get_many([d(i) for i in range(8)]) for _ in range(4)))
    assert len(created) == 1


class _PartialDelete:
    def delete_objects(self, Bucket, Delete):
        keys = [o["Key"] for o in Delete["Objects"]]
        return {"Errors": [{"Key": keys[0], "Code": "AccessDenied", "Message": "denied"}]}


async def test_delete_reports_per_key_errors() -> None:
    b = s3.S3Store("bucket", client=_PartialDelete()).bind("n")
    with pytest.raises(StoreError, match="1 of 3 objects not deleted.*AccessDenied"):
        await b.delete([d(1), d(2), d(3)])


async def test_delete_counts_and_deletes(aws) -> None:
    b = s3.S3Store("memcat-test").bind("n")
    await b.put(d(1), b"x", 60)
    await b.put(d(2), b"y", 60)
    assert await b.delete([d(1), d(2)]) == 2
    assert await b.get_many([d(1), d(2)]) == [None, None]


def test_identity_includes_region_endpoint_and_credentials() -> None:
    base = s3.S3Store("b", aws_access_key_id="AK", aws_secret_access_key="supersecret1")
    same = s3.S3Store("b", aws_access_key_id="AK", aws_secret_access_key="supersecret1")
    other_secret = s3.S3Store("b", aws_access_key_id="AK", aws_secret_access_key="supersecret2")
    other_region = s3.S3Store(
        "b", aws_access_key_id="AK", aws_secret_access_key="supersecret1", region_name="eu-west-1"
    )
    other_endpoint = s3.S3Store(
        "b",
        "memcat",
        "http://minio:9000",
        aws_access_key_id="AK",
        aws_secret_access_key="supersecret1",
    )
    assert base.identity() == same.identity()
    ids = {
        base.identity(),
        other_secret.identity(),
        other_region.identity(),
        other_endpoint.identity(),
    }
    assert len(ids) == 4
    assert "supersecret" not in repr(ids) and "'AK'" not in repr(ids)
    assert (
        s3.S3Store("b", client=object()).identity() != s3.S3Store("b", client=object()).identity()
    )


class _SlowGets:
    def __init__(self) -> None:
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def get_object(self, Bucket, Key):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        raise _NotFound()


class _NotFound(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


async def test_get_many_is_bounded_and_budgeted_per_request(monkeypatch) -> None:
    monkeypatch.setattr(s3, "_S3_TIMEOUT_S", 0.3)
    client = _SlowGets()
    b = s3.S3Store("bucket", client=client).bind("n")
    # 60 GETs of 50 ms, 10 at a time: about 0.3 s in all, each well within 0.3 s.
    assert await b.get_many([d(i) for i in range(60)]) == [None] * 60
    assert client.peak <= s3._CONCURRENCY


async def test_idle_seconds_uses_the_listing(aws) -> None:
    b = s3.S3Store("memcat-test").bind("n")
    for i in range(5):
        await b.put(d(i), b"x", 60)
    found = [x async for x in b.scan()]
    client = b._p._s3()
    heads: list[str] = []
    real = client.head_object

    def counting(**kw):
        heads.append(kw["Key"])
        return real(**kw)

    client.head_object = counting
    ages = await b.idle_seconds(found)
    assert heads == []
    assert all(a is not None and 0 <= a < 60 for a in ages)
    await b.put(d(0), b"rewritten", 60)  # LastModified changed: ask S3 again
    assert (await b.idle_seconds([d(0), d(99)]))[1] is None
    assert len(heads) == 2


def test_s3_store_survives_a_second_event_loop(aws) -> None:
    b = s3.S3Store("memcat-test").bind("n")
    asyncio.run(b.put(d(1), b"x", 60))
    assert asyncio.run(b.get_many([d(1)])) == [b"x"]


# --- purge events between processes -------------------------------------------------

PURGE_ONE, PURGE_ALL = b"\x00", b"\x01"


def _two_processes(poll: float = 0.05):
    a = s3.S3Store("memcat-test", events_poll_s=poll).bind("ns")
    b = s3.S3Store("memcat-test", events_poll_s=poll).bind("ns")
    return a, b


def _baseline_done(b, monkeypatch) -> asyncio.Event:
    """Set once ``b``'s subscription has taken its first listing."""
    done, real = asyncio.Event(), b._poll_events

    async def spy(*args, **kw):
        out = await real(*args, **kw)
        done.set()
        return out

    monkeypatch.setattr(b, "_poll_events", spy)
    return done


async def _collect(it, n: int, timeout: float) -> list[bytes]:
    got: list[bytes] = []

    async def run() -> None:
        async for m in it:
            got.append(m)
            if len(got) >= n:
                return

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(run(), timeout)
    return got


async def test_purge_reaches_another_process(aws, monkeypatch) -> None:
    a, b = _two_processes()
    await a.put(d(1), b"x", 60)
    await b.observe(d(1), "call-1", 3)
    await b.observe(d(1), "call-2", 3)
    baseline = _baseline_done(b, monkeypatch)
    sub = b.subscribe_new()
    first = asyncio.ensure_future(_collect(sub, 2, 5.0))
    await asyncio.wait_for(baseline.wait(), 5.0)
    await a.delete([d(1)])
    await a.publish_many([PURGE_ONE + d(1), d(2)])  # a new-entry notice is not written
    await a.publish_new(PURGE_ALL)
    got = await first
    assert got == [PURGE_ONE + d(1), PURGE_ALL]
    # b's process-local admission count for the purged digest was forgotten
    assert not await b.observe(d(1), "call-3", 3)


async def test_publisher_does_not_hear_itself_and_new_entries_cost_nothing(aws) -> None:
    a, _ = _two_processes()
    sub = a.subscribe_new()
    pending = asyncio.ensure_future(_collect(sub, 1, 0.5))
    await asyncio.sleep(0.1)
    await a.publish_new(d(3))
    assert [o async for o in a._list_raw(a._events)] == []
    await a.publish_new(PURGE_ONE + d(3))
    assert await pending == []


async def test_event_written_during_the_first_listing_is_delivered(aws, monkeypatch) -> None:
    a, b = _two_processes()
    real = b._poll_events
    calls = {"n": 0}

    async def write_then_list(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            await a.publish_many([PURGE_ONE + d(7)])
        return await real(*args, **kw)

    monkeypatch.setattr(b, "_poll_events", write_then_list)
    assert await _collect(b.subscribe_new(), 1, 5.0) == [PURGE_ONE + d(7)]


async def test_no_replay_of_old_events(aws) -> None:
    a, b = _two_processes()
    await a.publish_many([PURGE_ONE + d(1)])
    got = await _collect(b.subscribe_new(), 1, 0.4)
    assert got == []


async def test_events_are_not_entries(aws) -> None:
    a, b = _two_processes()
    await a.put(d(1), b"x", 60)
    await a.publish_many([PURGE_ONE + d(9)])
    assert [x async for x in b.scan()] == [d(1)]
    assert await b.find_prefixes([b"\x00", b"\x5f"]) == [d(1)]
    assert await b.delete(None) == 1
    assert [o async for o in a._list_raw(a._dir)] == []  # purge --all clears events too


async def test_old_events_are_cleaned_up(aws, monkeypatch) -> None:
    a, _ = _two_processes()
    await a.publish_many([PURGE_ONE + d(1)])
    monkeypatch.setattr(s3, "_EVENT_RETENTION_S", 0.0)
    await a.publish_many([PURGE_ONE + d(2)])
    keys = [o["Key"] async for o in a._list_raw(a._events)]
    assert len(keys) <= 1


async def test_poll_failure_recovers(aws, monkeypatch) -> None:
    a, b = _two_processes()
    baseline = _baseline_done(b, monkeypatch)
    sub = b.subscribe_new()
    pending = asyncio.ensure_future(_collect(sub, 2, 5.0))
    await asyncio.wait_for(baseline.wait(), 5.0)
    real = b._list_raw
    calls = {"n": 0}

    def failing(*args, **kw):
        calls["n"] += 1
        raise StoreError("listing down")

    monkeypatch.setattr(b, "_list_raw", failing)
    await a.publish_many([PURGE_ONE + d(5)])  # written while b cannot list
    monkeypatch.setattr(s3, "_EVENT_RETENTION_S", 0.0)  # any gap counts as too long
    await asyncio.sleep(0.3)
    monkeypatch.setattr(b, "_list_raw", real)
    got = await pending
    assert calls["n"] >= 1
    assert got == [s3.RESYNC, PURGE_ONE + d(5)]


async def test_find_prefixes_one_listing_each(aws) -> None:
    b = s3.S3Store("memcat-test").bind("ns")
    for i, tag in enumerate([b"\xaa\x01", b"\xaa\x02", b"\xbb\x01"]):
        await b.put(tag + bytes([i]) * 14, b"x", 60)
    await b.observe(b"\xcc" * 16, "c", 3)
    got = await b.find_prefixes([b"\xaa", b"\xaa\x01", b"\xcc"])
    assert got == sorted(
        [b"\xaa\x01" + bytes([0]) * 14, b"\xaa\x02" + bytes([1]) * 14, b"\xcc" * 16]
    )
    assert await b.find_prefixes([b"\xbb"]) == [b"\xbb\x01" + bytes([2]) * 14]
