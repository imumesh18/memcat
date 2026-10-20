"""Core robustness: breaker recovery, runtime lifecycle, codec, worker and index edge cases."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from memcat import MemcatConfig, MemoryStore
from memcat.core import audio, codec
from memcat.core import index as index_mod
from memcat.core import runtime as runtime_mod
from memcat.core.audio import LoudnessTracker
from memcat.core.breaker import Breaker
from memcat.core.codec import Entry
from memcat.core.index import BloomFilter, KeyIndex
from memcat.core.keys import Keyer
from memcat.core.lookup import L1Cache, Lookup
from memcat.core.metrics import Metrics
from memcat.core.runtime import get_runtime
from memcat.core.store.base import RESYNC, StoreError
from memcat.core.worker import Audio, ObserveItem, Policy, Tombstones, Worker, WorkerConfig


def d(i: int) -> bytes:
    return i.to_bytes(16, "big")


def tone(sr: int, secs: float, amp: float = 8000.0) -> bytes:
    t = np.arange(int(sr * secs)) / sr
    return (np.sin(2 * np.pi * 300 * t) * amp).astype("<i2").tobytes()


def _worker(store=None, policy: Policy | None = None, **cfg):
    store = store or MemoryStore().bind("n")
    idx = KeyIndex(store)
    idx._loaded = True
    l1 = L1Cache(1 << 30)
    m = Metrics()
    b = Breaker()
    w = Worker(store, idx, l1, b, m, LoudnessTracker(), WorkerConfig(policy or Policy(), **cfg))
    return w, store, idx, l1, m


def _nop(*_a: object) -> None:
    pass


async def _until(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


# --- breaker ---------------------------------------------------------------------


async def test_slow_store_read_is_a_miss_within_the_budget() -> None:
    """A store read on the call path that outlasts its budget (an S3 GET at its tail,
    a custom store with no timeout) is a miss and a store error, not a stall."""

    class Store:
        async def get_many(self, ds):
            await asyncio.sleep(5)
            return [None] * len(ds)

    b, m = Breaker(threshold=1, min_error_ratio=0.0), Metrics()
    store: Any = Store()
    lk = Lookup(store, L1Cache(1 << 20), KeyIndex(store), b, m, _nop, _nop, _nop)
    t0 = time.monotonic()
    assert await lk.get_many([d(1)]) == [None]
    assert time.monotonic() - t0 < 1.0
    assert m.snapshot()["store_errors"] == 1
    assert b.is_open


async def test_store_with_its_own_read_budget_gets_that_budget() -> None:
    """``RedisStore(op_timeout_ms=300)`` is honoured on the call path, not cut to 100 ms."""

    class Store:
        read_timeout_s = 0.3

        async def get_many(self, ds):
            await asyncio.sleep(0.15)
            return [None] * len(ds)

    m = Metrics()
    store: Any = Store()
    lk = Lookup(store, L1Cache(1 << 20), KeyIndex(store), Breaker(), m, _nop, _nop, _nop)
    assert await lk.get_many([d(1)]) == [None]
    assert m.snapshot()["store_errors"] == 0


async def test_cancelled_probe_does_not_keep_the_breaker_open() -> None:
    now = [0.0]
    b = Breaker(threshold=1, cooldown_s=1, clock=lambda: now[0])
    gate = asyncio.Event()

    class Store:
        async def get_many(self, ds):
            await gate.wait()  # the probe hangs until a barge-in cancels it
            return [None] * len(ds)

    idx = KeyIndex(Store())  # type: ignore[arg-type]
    store: Any = Store()
    lk = Lookup(store, L1Cache(1 << 20), idx, b, Metrics(), _nop, _nop, _nop)
    b.failure()
    assert b.is_open
    now[0] = 2.0
    probe = asyncio.ensure_future(lk.get_many([d(1)]))
    await asyncio.sleep(0.01)
    probe.cancel()
    await asyncio.gather(probe, return_exceptions=True)
    assert b.allow()  # the next request probes; it is not refused forever
    b.success()
    assert not b.is_open


def test_probe_that_never_reports_is_given_up_after_its_deadline() -> None:
    now = [0.0]
    b = Breaker(threshold=1, cooldown_s=1, clock=lambda: now[0], probe_timeout_s=5)
    b.failure()
    now[0] = 2.0
    assert b.allow()  # the probe goes out and never reports back
    now[0] = 6.0
    assert not b.allow()  # still within its deadline
    now[0] = 7.5
    assert b.allow()  # given up: a new probe
    b.success()
    assert not b.is_open


# --- default store and runtime lifecycle -----------------------------------------


async def test_default_configs_share_one_store_and_runtime() -> None:
    a, b = MemcatConfig(), MemcatConfig()
    assert a.store is b.store
    assert get_runtime(a) is get_runtime(b)
    assert MemcatConfig(store=MemoryStore()).store is not a.store


async def test_idle_runtime_leaves_the_registry(monkeypatch) -> None:
    monkeypatch.setattr(runtime_mod, "_IDLE_STOP_S", 0.01)
    cfg = MemcatConfig(store=MemoryStore())
    rt = get_runtime(cfg)
    rt.acquire()
    await rt.release()
    await _until(lambda: get_runtime(cfg) is not rt)
    assert not rt.worker._tasks and rt.index._follow_task is None


async def test_call_arriving_during_idle_stop_keeps_a_working_runtime(monkeypatch) -> None:
    monkeypatch.setattr(runtime_mod, "_IDLE_STOP_S", 0.01)
    cfg = MemcatConfig(store=MemoryStore())
    rt = get_runtime(cfg)
    rt.acquire()
    real_stop = rt.worker.stop

    async def stop_then_a_call_arrives() -> None:
        await real_stop()
        rt.acquire()  # a call holding rt starts while the index is still being stopped

    rt.worker.stop = stop_then_a_call_arrives  # type: ignore[method-assign]
    await rt.release()
    await asyncio.sleep(0.2)
    assert rt._refs == 1
    assert rt.index._follow_task is not None and not rt.index._follow_task.done()
    assert rt.worker._tasks and not any(t.done() for t in rt.worker._tasks)
    assert get_runtime(cfg) is rt  # registered again for the next call


# --- codec ------------------------------------------------------------------------


def test_long_multibyte_word_is_cut_on_a_character_boundary() -> None:
    w = "é" * 200  # 400 UTF-8 bytes; a cut at byte 255 splits a character
    blob = codec.encode(Entry(codec.CODEC_PCM16, 16000, 10, None, ((w, 0),), b"\x00\x00" * 160))
    e = codec.decode(blob)  # a cut inside a character would raise CorruptEntryError
    assert e.words[0][0] == "é" * 127


# --- observe batches --------------------------------------------------------------


def test_odd_length_pcm_is_tolerated() -> None:
    sr = 16000
    pcm = tone(sr, 0.5) + b"\x01"
    out, _ = audio.trim_silence(pcm, sr)
    assert out and len(out) % 2 == 0
    assert audio.loudness(pcm, sr) is not None
    assert len(audio.apply_gain(pcm, 2.0)) == len(pcm) - 1


async def test_one_bad_item_does_not_drop_the_batch() -> None:
    w, _, idx, _, m = _worker(policy=Policy(admit_after_calls=1))
    w.start()
    sr = 16000
    w.observe(
        "c1",
        [
            ObserveItem(d(1), 10, captured=Audio(tone(sr, 0.4) + b"\x00", sr)),  # odd length
            ObserveItem(d(2), 10, captured=Audio(tone(sr, 0.4), sr, ((None, "x"),))),  # type: ignore[arg-type]
            ObserveItem(d(3), 10, captured=Audio(tone(sr, 0.4), sr)),
        ],
    )
    await w.drain_all(5)
    assert d(1) in idx and d(2) not in idx and d(3) in idx
    assert m.counters["captures"] == 2
    await w.stop()


async def test_silent_render_is_marked_failed_and_not_paid_for_again() -> None:
    w, _, idx, _, m = _worker(policy=Policy(admit_after_calls=1))
    w.start()
    renders = {"n": 0}

    async def quiet() -> Audio:
        renders["n"] += 1
        return Audio(b"\x00\x00" * 16000, 16000)

    w.observe("c1", [ObserveItem(d(1), 10, render=quiet)])
    await w.drain_all(5)
    w.observe("c2", [ObserveItem(d(1), 10, render=quiet)])
    await w.drain_all(5)
    assert renders["n"] == 1 and d(1) not in idx
    assert m.counters["render_failures"] == 1
    await w.stop()


async def test_observes_past_the_byte_bound_are_dropped_and_counted() -> None:
    w, *_, m = _worker(policy=Policy(admit_after_calls=1), queue_max_bytes=100_000)
    sr = 16000
    clip = tone(sr, 2.0)  # 64 KB
    w.observe("c1", [ObserveItem(d(1), 10, captured=Audio(clip, sr))])
    w.observe("c1", [ObserveItem(d(2), 10, captured=Audio(clip, sr))])  # over the bound
    assert m.counters["dropped_jobs"] == 1 and m.counters["dropped_bytes"] == len(clip)
    w.start()
    await w.drain_all(5)
    assert w._queued_bytes == 0
    w.observe("c1", [ObserveItem(d(3), 10, captured=Audio(clip, sr))])  # room again
    assert m.counters["dropped_jobs"] == 1
    await w.stop()


# --- touch ------------------------------------------------------------------------


async def test_failed_touch_is_retried_on_the_next_hit() -> None:
    w, store, *_ = _worker(touch_every_s=3600)
    calls: list[list[bytes]] = []
    fail = [True]

    async def touch(ds, ttl):
        calls.append(list(ds))
        if fail[0]:
            raise StoreError("down")

    store.touch = touch  # type: ignore[method-assign]
    w.start()
    w.touch([d(1)])
    await w.drain_all(5)
    fail[0] = False
    w.touch([d(1)])  # the first refresh failed: not throttled
    await w.drain_all(5)
    w.touch([d(1)])  # this one worked: throttled for an hour
    await w.drain_all(5)
    assert calls == [[d(1)], [d(1)]]
    await w.stop()


# --- tombstones -------------------------------------------------------------------


def test_tombstones_mark_is_constant_time(monkeypatch) -> None:
    from memcat.core import worker as worker_mod

    monkeypatch.setattr(worker_mod, "_MAX_TOMBSTONES", 1000)
    t = Tombstones()
    t0 = time.perf_counter()
    for i in range(50_000):  # all inside the window: O(n) pruning would rescan each time
        t.mark([d(i)])
    assert time.perf_counter() - t0 < 2.0
    assert t.blocks(d(0), 0.0) and t.blocks(d(49_999), 0.0)


def test_tombstones_expire_oldest_first(monkeypatch) -> None:
    from memcat.core import worker as worker_mod

    monkeypatch.setattr(worker_mod, "_MAX_TOMBSTONES", 2)
    now = [1000.0]
    monkeypatch.setattr(worker_mod.time, "monotonic", lambda: now[0])
    t = Tombstones()
    t.mark([d(1), d(2)])
    now[0] += Tombstones.WINDOW_S + 1
    t.mark([d(3)])
    assert len(t) == 1 and t.blocks(d(3), 0.0) and not t.blocks(d(1), 0.0)


# --- settings memo ----------------------------------------------------------------


@dataclasses.dataclass
class _Voice:
    name: str
    stability: float = dataclasses.field(repr=False, default=0.5)


def test_settings_memo_sees_fields_hidden_from_repr() -> None:
    k = Keyer("svc.Cls")
    a = k.context({"voice": "v", "extra": _Voice("x", 0.5)}, 16000).digest("hello there")
    b = k.context({"voice": "v", "extra": _Voice("x", 0.9)}, 16000).digest("hello there")
    assert a != b


def test_settings_memo_sees_in_place_changes() -> None:
    k = Keyer("svc.Cls")
    opts = {"speed": 1.0}
    a = k.context({"voice": "v", "opts": opts}, 16000).digest("hello there")
    opts["speed"] = 1.2
    b = k.context({"voice": "v", "opts": opts}, 16000).digest("hello there")
    assert a != b
    assert k.context({"voice": "v", "opts": opts}, 16000) is k.context(
        {"voice": "v", "opts": dict(opts)}, 16000
    )  # still memoised when nothing changed


# --- key index: resync and Bloom --------------------------------------------------


class _Feed:
    """A MemoryStore view whose notifications the test drives by hand."""

    def __init__(self) -> None:
        self.inner = MemoryStore().bind("n")
        self.q: asyncio.Queue[bytes | Exception] = asyncio.Queue()

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    async def subscribe_new(self) -> AsyncIterator[bytes]:
        while True:
            item = await self.q.get()
            if isinstance(item, Exception):
                raise item
            yield item


async def test_resync_forgets_entries_purged_during_an_outage() -> None:
    store = _Feed()
    l1 = L1Cache(1 << 20)
    evicted: list[bytes] = []

    def on_resync() -> None:
        for x in l1.digests():
            if x not in idx:
                evicted.append(x)
                l1.discard(x)

    idx = KeyIndex(store, min_resync_interval_s=0, on_resync=on_resync)  # type: ignore[arg-type]
    await store.inner.put(d(1), b"x", 60)
    await store.inner.put(d(2), b"x", 60)
    idx.start()
    await _until(lambda: idx.loaded and d(1) in idx)
    from memcat.core.lookup import Clip

    l1.put(d(1), Clip(16000, 10, (), None, pcm=b"\x00\x00", nbytes=2))
    await store.inner.delete([d(1)])  # purged elsewhere; the notice is lost
    store.q.put_nowait(RESYNC)
    await _until(lambda: d(1) not in idx)
    assert d(2) in idx and l1.get(d(1)) is None and evicted == [d(1)]
    await idx.stop()


async def test_runtime_resync_evicts_l1() -> None:
    cfg = MemcatConfig(store=MemoryStore(), namespace="rs")
    rt = get_runtime(cfg)
    from memcat.core.lookup import Clip

    await rt.store.put(d(5), b"x", 60)
    rt.index.add(d(5))
    rt.l1.put(d(5), Clip(16000, 10, (), None, pcm=b"\x00\x00", nbytes=2))
    await rt.store.delete([d(5)])
    rt.index._min_resync_s = 0
    rt.index.resync()
    await _until(lambda: rt.l1.get(d(5)) is None)
    assert d(5) not in rt.index and rt.worker.tomb.blocks(d(5), 0.0)
    await rt.stop()


async def test_subscription_backoff_resets_after_a_healthy_connection(monkeypatch) -> None:
    monkeypatch.setattr(index_mod, "_FOLLOW_HEALTHY_S", 0.0)
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(s: float, *a):
        if s >= 0.5:
            delays.append(s)
            s = 0.001
        return await real_sleep(s, *a)

    store = _Feed()
    idx = KeyIndex(store, min_resync_interval_s=0)  # type: ignore[arg-type]
    monkeypatch.setattr(index_mod.asyncio, "sleep", fake_sleep)
    idx.start()
    for _ in range(5):
        store.q.put_nowait(ConnectionError("dropped"))
    await _until(lambda: len(delays) >= 5)
    monkeypatch.setattr(index_mod.asyncio, "sleep", real_sleep)
    await idx.stop()
    assert all(x == 0.5 for x in delays[:5])


def test_bloom_bulk_build_matches_single_adds_and_counts_distinct() -> None:
    keys = [os.urandom(16) for _ in range(5000)]
    bulk = BloomFilter.from_digests(set(keys), 100_000)
    one = BloomFilter(100_000)
    for k in keys:
        one.add(k)
        one.add(k)  # a re-add is not counted twice
    assert bulk._bits == one._bits
    assert bulk.count == 5000 and one.count >= 4990


async def test_bloom_conversion_runs_off_the_loop_and_keeps_changes() -> None:
    idx = KeyIndex(MemoryStore().bind("n"), bloom_after=1000)
    idx._loaded = True
    keys = [os.urandom(16) for _ in range(1001)]
    for k in keys:
        idx.add(k)
    assert idx.is_bloom and idx._frozen is not None  # building in a thread
    late = os.urandom(16)
    idx.add(late)  # added while the filter is built
    idx.discard(keys[0])  # removed while the filter is built
    assert all(k in idx for k in keys[1:]) and late in idx and keys[0] not in idx
    await _until(lambda: idx._bloom is not None)
    assert idx._frozen is None
    assert all(k in idx for k in keys[1:]) and late in idx and keys[0] not in idx
    assert len(idx) == 1002


async def test_bloom_resync_builds_in_a_thread_and_replaces() -> None:
    store = _Feed()
    keys = [os.urandom(16) for _ in range(300)]
    for k in keys:
        await store.inner.put(k, b"x", 60)
    idx = KeyIndex(store, bloom_after=100, min_resync_interval_s=0)  # type: ignore[arg-type]
    idx.start()
    await _until(lambda: idx.loaded)
    assert idx._bloom is not None and len(idx) == 300 and all(k in idx for k in keys)
    await store.inner.delete(keys[:250])
    store.q.put_nowait(RESYNC)
    await _until(lambda: idx._bloom is None)  # back under bloom_after: a plain set again
    assert len(idx) == 50 and keys[0] not in idx and keys[-1] in idx
    await idx.stop()


async def test_purge_publishes_in_one_batch_when_the_store_can() -> None:
    rt = get_runtime(MemcatConfig(store=MemoryStore(), namespace="pm"))
    sent: list[list[bytes]] = []

    async def publish_many(msgs: list[bytes]) -> None:
        sent.append(list(msgs))

    rt.store.publish_many = publish_many  # type: ignore[attr-defined]
    await rt.purge([d(1), d(2)])
    assert sent == [[index_mod.PURGE_ONE + d(1), index_mod.PURGE_ONE + d(2)]]


async def test_concurrent_hits_decode_a_clip_once(monkeypatch) -> None:
    w, *_ = _worker()
    w.start()
    calls = 0
    real = codec.to_pcm16

    def counting(e: Entry) -> bytes:
        nonlocal calls
        calls += 1
        return real(e)

    monkeypatch.setattr(codec, "to_pcm16", counting)
    e = Entry(codec.CODEC_PCM16, 16000, 10, None, (), b"\x00\x00" * 160)
    for _ in range(5):
        w.promote(d(7), e)
    await w.drain_all(5)
    assert calls == 1
    w.promote(d(7), e)  # a later hit, after the first decode finished, may decode again
    await w.drain_all(5)
    assert calls == 2
    await w.stop()


def test_finished_event_loops_are_released() -> None:
    from memcat.core.runtime import _registry

    store = MemoryStore()

    async def one_call() -> None:
        get_runtime(MemcatConfig(store=store, namespace="loops"))

    for _ in range(4):
        asyncio.run(one_call())
    asyncio.run(one_call())
    assert sum(1 for lp in _registry if lp.is_closed()) <= 1


async def test_purge_whose_notification_fails_does_not_report_success(monkeypatch) -> None:
    import pytest

    from memcat import PurgeNotifyError

    rt = get_runtime(MemcatConfig(store=MemoryStore(), namespace="notify"))
    await rt.store.put(d(9), b"x", 3600)
    calls = 0

    async def broken(messages) -> None:
        nonlocal calls
        calls += 1
        raise ConnectionError("pub/sub down")

    monkeypatch.setattr(rt.store, "publish_many", broken, raising=False)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    with pytest.raises(PurgeNotifyError) as err:
        await rt.purge([d(9)])
    assert err.value.deleted == 1 and calls == 3
    assert await rt.store.get_many([d(9)]) == [None]  # deleted all the same


_real_sleep = asyncio.sleep


async def _no_sleep(_s: float) -> None:
    await _real_sleep(0)
