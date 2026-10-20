"""Purging a text removes it in every voice, setting and sample rate, everywhere."""

from __future__ import annotations

import asyncio
import time

from memcat import MemcatConfig, MemoryStore
from memcat.core.keys import Keyer, text_tag
from memcat.core.runtime import get_runtime

TEXT = "Thanks for waiting today Margaret Hamilton"
OTHER = "is there anything else I can help with?"


def _ctx(voice: str, rate: int):
    return Keyer("pipecat.services.cartesia.tts.CartesiaTTSService").context(
        {"model": "sonic", "voice": voice, "language": "en"}, rate
    )


def test_every_variant_of_a_text_shares_its_tag() -> None:
    a = _ctx("v1", 8000).digest(TEXT)
    b = _ctx("v2", 24000).digest(TEXT)
    assert a != b
    assert a[:4] == b[:4] == text_tag(TEXT)
    assert _ctx("v1", 8000).digest(OTHER)[:4] != a[:4]


async def test_purge_texts_reaches_every_voice_and_admission_count() -> None:
    store = MemoryStore()
    rt = get_runtime(MemcatConfig(store=store, namespace="p"))
    bound = rt.store
    v1, v2, v3 = _ctx("v1", 8000), _ctx("v2", 16000), _ctx("v3", 24000)
    await bound.put(v1.digest(TEXT), b"a", 3600)
    await bound.put(v2.digest(TEXT), b"b", 3600)
    await bound.observe(v3.digest(TEXT), "call-1", 3)  # heard once, not cached
    keep = v1.digest(OTHER)
    await bound.put(keep, b"c", 3600)

    assert await rt.purge_texts([TEXT]) == 2
    assert await bound.get_many([v1.digest(TEXT), v2.digest(TEXT), keep]) == [None, None, b"c"]
    assert await bound.find_prefixes([text_tag(TEXT)]) == []  # admission count gone too


async def test_purge_matches_the_clause_whatever_punctuation_followed_it() -> None:
    rt = get_runtime(MemcatConfig(store=MemoryStore(), namespace="punct"))
    v = _ctx("v1", 8000)
    for clause in ("Hi John, ", "Hi John."):
        await rt.store.put(v.digest(clause), b"a", 3600)
    assert await rt.purge_texts(["Hi John"]) == 2


async def test_a_write_landing_after_another_process_purged_it_is_deleted_again() -> None:
    from memcat.core import codec
    from memcat.core.runtime import Runtime
    from memcat.core.worker import Audio, Policy
    from tests.test_core_robustness import tone

    store = MemoryStore()
    cfg = MemcatConfig(store=store, namespace="race")
    purger, writer = Runtime(store, cfg), Runtime(store, cfg)  # two processes
    for rt in (purger, writer):
        rt.acquire()
    async with asyncio.timeout(5):
        while not (purger.index.loaded and writer.index.loaded):
            await asyncio.sleep(0.01)
    d = _ctx("v1", 16000).digest(TEXT)
    try:
        # The purger deletes first; the writer's put of the same clause lands just
        # after, and before the purge notification reaches the writer.
        real_delete = purger.store.delete
        await real_delete([d])
        stored = await writer.worker._store_audio(
            d, Audio(tone(16000, 0.5), 16000), codec.FLAG_RENDERED, Policy(), time.monotonic()
        )
        assert stored == "stored" and await store.bind("race").get_many([d]) != [None]

        async def deleted_already(_ds):
            return 0

        purger.store.delete = deleted_already  # type: ignore[method-assign]
        await purger.purge([d])  # now only notifies
        async with asyncio.timeout(5):
            while await store.bind("race").get_many([d]) != [None]:
                await asyncio.sleep(0.01)
        assert d not in writer.index and writer.l1.get(d) is None
    finally:
        for rt in (purger, writer):
            await rt.stop()
