"""Unit tests for the core pieces the pipeline tests reach only indirectly."""

from __future__ import annotations

import asyncio
import dataclasses

import numpy as np
import pytest

from memcat import MemcatConfig
from memcat.core import audio, codec
from memcat.core.audio import LoudnessTracker
from memcat.core.breaker import Breaker
from memcat.core.index import KeyIndex
from memcat.core.lookup import Clip, L1Cache, Lookup
from memcat.core.metrics import Metrics, price_for
from memcat.core.store.base import StoreError
from memcat.core.store.memory import MemoryStore
from memcat.core.worker import Audio, ObserveItem, Policy, Worker, WorkerConfig, _prepare


def tone(sr: int, secs: float, amp: float = 8000.0, lead: float = 0.0) -> bytes:
    t = np.arange(int(sr * secs)) / sr
    x = np.sin(2 * np.pi * 300 * t) * amp
    pad = np.zeros(int(sr * lead))
    return np.concatenate([pad, x, pad]).astype("<i2").tobytes()


def d(i: int) -> bytes:
    return bytes([i]) * 16


# --- audio ---------------------------------------------------------------------


def test_trim_and_tail() -> None:
    sr = 16000
    pcm = tone(sr, 0.5, lead=0.3)
    out, cut = audio.trim_silence(pcm, sr, keep_ms=30)
    assert cut == int(sr * 0.27)
    assert len(out) // 2 == pytest.approx(sr * (0.5 + 0.03 + 0.03), abs=sr * 0.011)
    assert audio.trim_silence(b"\0\0" * 1600, sr) == (b"", 0)
    assert audio.trim_silence(b"", sr) == (b"", 0)
    assert audio.join_pause_ms("Sure, ") == audio.SHORT_PAUSE_MS
    assert audio.join_pause_ms("Thanks for waiting, ") == 320
    assert audio.join_pause_ms("I can help with that; ", 300) == 340
    assert audio.join_pause_ms("Have a great day.") == audio.SENTENCE_GAP_MS
    assert audio.join_pause_ms("") == 0
    assert len(audio.silence(16000, 100)) == 3200


def test_fade_gain_loudness() -> None:
    sr = 16000
    pcm = tone(sr, 1.0)
    faded = np.frombuffer(audio.fade_edges(pcm, sr), dtype="<i2")
    assert abs(int(faded[0])) < 5 and abs(int(faded[-1])) < 100
    assert audio.fade_edges(b"\x01\x00", sr) == b"\x01\x00"
    loud = audio.loudness(pcm, sr)
    assert loud is not None and -30 < loud < 0
    assert audio.loudness(tone(sr, 0.2), sr) is None
    up = np.frombuffer(audio.apply_gain(pcm, 10.0), dtype="<i2")  # clamped to +3 dB
    ratio = np.abs(up).max() / np.abs(np.frombuffer(pcm, dtype="<i2")).max()
    assert ratio == pytest.approx(10 ** (3 / 20), rel=0.01)
    assert audio.apply_gain(pcm, 0.01) == pcm


def test_loudness_tracker() -> None:
    lt = LoudnessTracker(window=5)
    assert lt.gain_for("v", -20.0) == 0.0
    for v in (-18.0, -18.5, -17.5, -18.0, -40.0, -18.0):
        lt.add("v", v)
    assert lt.median("v") == -18.0
    assert lt.gain_for("v", -20.0) == pytest.approx(2.0)
    assert lt.gain_for("v", -30.0) == 3.0
    assert lt.gain_for("v", None) == 0.0


# --- L1 and lookup ---------------------------------------------------------------


def test_l1_bounds_and_discard() -> None:
    l1 = L1Cache(1000)
    c = Clip(8000, 10, (), None, pcm=b"\0" * 400, nbytes=400)
    l1.put(d(1), c)
    l1.put(d(2), c)
    assert len(l1) == 1 and l1.get(d(1)) is None and l1.get(d(2)) is c
    l1.put(d(3), Clip(8000, 10, (), None, pcm=b"\0" * 5000, nbytes=5000))  # too big
    assert l1.get(d(3)) is None
    l1.discard(d(2))
    assert len(l1) == 0 and l1.size_bytes == 0
    l1.put(d(4), c)
    l1.clear()
    assert l1.size_bytes == 0


def _entry(cod: int, sr: int = 16000) -> bytes:
    pcm = tone(sr, 0.5)
    return codec.encode(
        codec.Entry(cod, sr, 500, -20.0, (("hi", 0),), codec.encode_audio(pcm, sr, cod))
    )


class _Calls:
    def __init__(self) -> None:
        self.hits: list[list[bytes]] = []
        self.corrupt: list[bytes] = []
        self.decoded: list[bytes] = []


def _lookup(store, *, loaded: bool = True):
    calls = _Calls()
    idx = KeyIndex(store)
    idx._loaded = loaded
    l1 = L1Cache(1 << 30)
    lk = Lookup(
        store,
        l1,
        idx,
        Breaker(),
        Metrics(),
        lambda ds, _ttl: calls.hits.append(ds),
        calls.corrupt.append,
        lambda dd, e: calls.decoded.append(dd),
    )
    return lk, idx, l1, calls


async def test_lookup_l2_paths() -> None:
    store = MemoryStore().bind("n")
    await store.put(d(1), _entry(codec.CODEC_PCM16), 60)
    await store.put(d(2), _entry(codec.CODEC_OPUS), 60)
    await store.put(d(3), b"garbage", 60)
    lk, idx, l1, calls = _lookup(store, loaded=False)
    res = await lk.get_many([d(1), None, d(2), d(3), d(4), d(1)])
    assert res[0] is not None and res[0].pcm is not None
    assert res[1] is None and res[3] is None and res[4] is None
    assert res[2] is not None and res[2].entry is not None  # Opus, decoded later
    assert len(codec.to_pcm16(res[2].entry)) == 16000 * 2 // 2
    assert calls.corrupt == [d(3)] and calls.decoded == [d(2)]
    assert l1.get(d(1)) is not None and d(1) in idx and d(2) in idx
    # Second time: L1 serves d(1) without the store.
    res2 = await lk.get_many([d(1)])
    assert res2[0] is res[0]
    assert calls.hits[-1] == [d(1)]


async def test_lookup_never_asks_store_for_unknown_digest() -> None:
    class Spy(MemoryStore):
        pass

    store = MemoryStore().bind("n")
    asked: list[list[bytes]] = []
    real = store.get_many

    async def spy(ds):
        asked.append(list(ds))
        return await real(ds)

    store.get_many = spy  # type: ignore[method-assign]
    lk, idx, _, _ = _lookup(store, loaded=True)
    assert await lk.get_many([d(9)]) == [None]
    assert asked == []
    idx.add(d(9))
    assert await lk.get_many([d(9)]) == [None]
    assert asked == [[d(9)]] and d(9) not in idx  # expired entries leave the index


async def test_lookup_store_errors_trip_breaker() -> None:
    store = MemoryStore().bind("n")

    async def boom(ds):
        raise StoreError("down")

    store.get_many = boom  # type: ignore[method-assign]
    lk, _, _, _ = _lookup(store, loaded=False)
    for _ in range(5):
        assert await lk.get_many([d(1)]) == [None]
    assert lk._breaker.is_open
    assert lk._metrics.counters["store_errors"] == 5


# --- worker --------------------------------------------------------------------


_POLICY_FIELDS = ("admit_after_calls", "codec", "lossless_only", "ttl_s", "render")


def _worker(store=None, **cfg):
    store = store or MemoryStore().bind("n")
    idx = KeyIndex(store)
    idx._loaded = True
    l1 = L1Cache(1 << 30)
    m = Metrics()
    pol = Policy(**{k: cfg.pop(k) for k in _POLICY_FIELDS if k in cfg})
    w = Worker(store, idx, l1, Breaker(), m, LoudnessTracker(), WorkerConfig(pol, **cfg))
    return w, store, idx, l1, m


def test_choose_codec() -> None:
    w, *_ = _worker()
    assert w.choose_codec(8000) == codec.CODEC_MULAW
    assert w.choose_codec(24000) == codec.CODEC_OPUS
    assert w.choose_codec(22050) == codec.CODEC_PCM16  # not an Opus rate
    lossless = Policy(lossless_only=True)
    assert w.choose_codec(8000, lossless) == codec.CODEC_PCM16
    assert w.choose_codec(24000, lossless) == codec.CODEC_PCM16
    for mode in ("pcm16", "mulaw", "opus"):
        pol = Policy(codec=mode, lossless_only=True)
        assert w.choose_codec(24000, pol) == codec.CODEC_PCM16  # lossless_only wins
    assert w.choose_codec(24000, Policy(codec="mulaw")) == codec.CODEC_MULAW
    w.cfg.policy = Policy(codec="mulaw")  # the default for jobs without a policy
    assert w.choose_codec(24000) == codec.CODEC_MULAW


def test_prepare_shifts_words_and_rejects_silence() -> None:
    sr = 16000
    pcm = tone(sr, 0.5, lead=0.2)
    blob, clip = _prepare(Audio(pcm, sr, (("hello", 0.2), ("there", 0.45))), codec.CODEC_PCM16, 1)
    assert blob is not None and clip is not None
    e = codec.decode(blob)
    assert e.words[0] == ("hello", 30) and e.words[1][1] == 280
    assert _prepare(Audio(b"\0\0" * 8000, sr), codec.CODEC_PCM16, 1) == (None, None)


async def test_observe_admit_capture_render_and_retry() -> None:
    w, store, idx, l1, m = _worker(admit_after_calls=2, render_retry_s=3600)
    w.start()
    sr = 16000
    renders = {"n": 0}

    async def good():
        renders["n"] += 1
        return Audio(tone(sr, 0.4), sr, (("x", 0.0),))

    async def bad():
        renders["n"] += 1
        return None

    w.observe(
        "call-1",
        [
            ObserveItem(d(1), 10, render=good),
            ObserveItem(d(2), 10, render=bad),
            ObserveItem(d(3), 10, captured=Audio(tone(sr, 0.4), sr)),
        ],
    )
    w.observe(
        "call-2",
        [
            ObserveItem(d(1), 10, render=good),
            ObserveItem(d(2), 10, render=bad),
            ObserveItem(d(3), 10, captured=Audio(tone(sr, 0.4), sr)),
        ],
    )
    await w.drain_all(5)
    assert d(1) in idx and d(3) in idx and d(2) not in idx
    assert m.counters["renders"] == 1 and m.counters["render_failures"] == 1
    assert m.counters["captures"] == 1
    # A failed digest is not retried within the hour, even when admitted again.
    w.observe("call-3", [ObserveItem(d(2), 10, render=bad)])
    w.observe("call-4", [ObserveItem(d(2), 10, render=bad)])
    await w.drain_all(5)
    assert renders["n"] == 2
    # Already stored: never observed again.
    w.observe("call-5", [ObserveItem(d(1), 10, render=good)])
    await w.drain_all(5)
    assert renders["n"] == 2
    await w.stop()


async def test_render_lock_held_elsewhere_skips() -> None:
    w, store, idx, _, m = _worker(admit_after_calls=1)
    assert await store.try_lock(d(1), 60)
    w.start()
    called = []

    async def r():
        called.append(1)
        return None

    w.observe("c", [ObserveItem(d(1), 1, render=r)])
    await w.drain_all(5)
    assert called == [] and m.counters["renders"] == 0
    await w.stop()


async def test_render_skips_when_store_already_has_it() -> None:
    w, store, idx, _, _ = _worker(admit_after_calls=1)
    idx._loaded = False
    await store.put(d(1), _entry(codec.CODEC_PCM16), 60)
    w.start()
    called = []

    async def r():
        called.append(1)
        return None

    w.warm(d(1), r)
    await w.drain_all(5)
    assert called == [] and d(1) in idx
    await w.stop()


async def test_queue_overflow_is_dropped_and_counted() -> None:
    w, *_rest, m = _worker(queue_size=2)
    for i in range(5):
        w.observe("c", [ObserveItem(d(i), 1)])
    assert m.counters["dropped_jobs"] == 3


async def test_touch_throttled_and_promote_and_loudness() -> None:
    w, store, idx, l1, m = _worker(touch_every_s=3600)
    touched: list[list[bytes]] = []
    real = store.touch

    async def spy(ds, ttl):
        touched.append(list(ds))
        await real(ds, ttl)

    store.touch = spy  # type: ignore[method-assign]
    w.start()
    w.touch([d(1), d(2)])
    w.touch([d(1), d(2), d(3)])
    e = codec.decode(_entry(codec.CODEC_OPUS))
    w.promote(d(7), e)
    sr = 16000
    w.live_loudness("voice", tone(sr, 3.0), sr)
    w.live_loudness("voice", tone(sr, 3.0), sr)  # throttled
    await w.drain_all(5)
    assert touched == [[d(1), d(2)], [d(3)]]
    clip = l1.get(d(7))
    assert clip is not None and clip.pcm is not None and len(clip.pcm) == 16000
    assert w._loudness._samples["voice"] and len(w._loudness._samples["voice"]) == 1
    w.delete_corrupt(d(7))
    await w.drain_all(5)
    assert l1.get(d(7)) is None
    await w.stop()


async def test_observe_store_errors_counted() -> None:
    store = MemoryStore().bind("n")

    async def boom(*a, **k):
        raise StoreError("x")

    store.observe = boom  # type: ignore[method-assign]
    w, _, _, _, m = _worker(store, admit_after_calls=3)
    w.start()
    w.observe("c", [ObserveItem(d(1), 1)])
    await w.drain_all(5)
    assert m.counters["store_errors"] == 1
    await w.stop()


# --- config, metrics, index ------------------------------------------------------


def test_config_validation() -> None:
    for kw in (
        {"namespace": "bad space"},
        {"admit_after_calls": 0},
        {"min_clause_words": 0},
        {"ttl_s": 5},
        {"codec": "flac"},
        {"queue_size": 0},
    ):
        with pytest.raises(ValueError):
            MemcatConfig(**kw)  # type: ignore[arg-type]
    for ns in ("ok\n", ".", "..", "x" * 65):
        with pytest.raises(ValueError):
            MemcatConfig(namespace=ns)
    assert MemcatConfig(namespace="tenant.a-1_b").namespace == "tenant.a-1_b"
    c = MemcatConfig(block_patterns=[r"acme"])
    assert c.compiled_blocks[0].search("ACME")


def test_config_is_immutable() -> None:
    """Fields are checked and patterns compiled once, so a later change could only be
    silently ignored: none is allowed."""
    c = MemcatConfig(block_patterns=[r"acme"], warm=["Hello there, friend."])
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.namespace = "bad space"  # type: ignore[misc]
    assert c.block_patterns == ("acme",) and c.warm == ("Hello there, friend.",)
    c2 = dataclasses.replace(c, block_patterns=[r"widget"])
    assert c2.compiled_blocks[0].search("WIDGET") and not c2.compiled_blocks[0].search("acme")


def test_prices() -> None:
    assert price_for("elevenlabs", "eleven_flash_v2_5") == 40.0
    assert price_for("elevenlabs", "eleven_multilingual_v2") == 80.0
    assert price_for("cartesia", "sonic-3") == 39.0
    assert price_for("generic", "x") == 0.0
    assert price_for("cartesia", None, {("cartesia", ""): 10.0}) == 10.0
    m = Metrics()
    assert m.percentile(0.5) is None
    for v in range(100):
        m.observe_ttfa(float(v))
    assert m.snapshot()["replay_first_audio_ms_p50"] == pytest.approx(50, abs=1)


async def test_index_follows_new_entries() -> None:
    root = MemoryStore()
    a, b = root.bind("n"), root.bind("n")
    await a.put(d(1), b"x", 60)
    idx = KeyIndex(b)
    idx.start()
    for _ in range(100):
        if idx.loaded:
            break
        await asyncio.sleep(0.01)
    assert d(1) in idx and len(idx) == 1
    await a.publish_new(d(2))
    for _ in range(100):
        if d(2) in idx:
            break
        await asyncio.sleep(0.01)
    assert d(2) in idx
    await idx.stop()


def test_index_switches_to_bloom_filter() -> None:
    import os

    from memcat.core.index import BloomFilter

    store = MemoryStore().bind("n")
    idx = KeyIndex(store, bloom_after=1000)
    idx._loaded = True
    keys = [os.urandom(16) for _ in range(5000)]
    for k in keys:
        idx.add(k)
    assert idx.is_bloom and len(idx) == 5000
    assert all(k in idx for k in keys)
    fp = sum(os.urandom(16) in idx for _ in range(20000)) / 20000
    assert fp < 0.02
    idx.discard(keys[0])
    assert keys[0] not in idx and keys[1] in idx
    idx.add(keys[0])
    assert keys[0] in idx
    bf = BloomFilter(1_000_000)
    assert bf.nbytes < 1.3 * 1024 * 1024


async def test_render_lock_errors_feed_the_breaker() -> None:
    store = MemoryStore().bind("n")
    calls = {"lock": 0}

    async def boom(*a, **k):
        calls["lock"] += 1
        raise StoreError("down")

    store.try_lock = boom  # type: ignore[method-assign]
    w, _, _, _, m = _worker(store, admit_after_calls=1)
    w.start()
    rendered = []

    async def r():
        rendered.append(1)
        return None

    for i in range(10):
        w.warm(d(i), r)
    await w.drain_all(5)
    # The breaker opens after five failures; later renders are skipped without a store call.
    assert w._breaker.is_open
    assert calls["lock"] == 5 and m.counters["store_errors"] == 5
    assert rendered == []
    await w.stop()


async def test_failure_logs_keep_error_text_out_of_warnings() -> None:
    from loguru import logger

    records: list[tuple[str, str]] = []
    sink = logger.add(lambda msg: records.append((msg.record["level"].name, msg.record["message"])))
    try:
        w, *_ = _worker(admit_after_calls=1)
        w.start()

        async def r():
            raise RuntimeError("could not say: your balance is low")

        w.warm(d(1), r)
        await w.drain_all(5)
        await w.stop()
    finally:
        logger.remove(sink)
    warnings = [m for lvl, m in records if lvl == "WARNING"]
    assert warnings and all("balance" not in m for m in warnings)
    assert any("RuntimeError" in m for m in warnings)
    assert any("balance" in m for lvl, m in records if lvl == "DEBUG")
