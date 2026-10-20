"""Per-process shared state: one runtime per (store, namespace, event loop).

Bots usually build a new TTS service, and often a new ``RedisStore(url)``, for
every call. Sharing the runtime means the key index is loaded once, the L1
cache is warm for every call, one worker serves them all, and the connection
pool is reused.
"""

from __future__ import annotations

import asyncio
import contextlib
import weakref
from collections.abc import Hashable, Sequence

from loguru import logger

from memcat.config import MemcatConfig
from memcat.core.audio import LoudnessTracker
from memcat.core.breaker import Breaker
from memcat.core.chunker import split
from memcat.core.index import PURGE_ALL, PURGE_ONE, KeyIndex
from memcat.core.keys import TEXT_TAG_BYTES, text_tag
from memcat.core.lookup import L1Cache, Lookup
from memcat.core.metrics import Metrics
from memcat.core.store.base import BoundStore, PurgeNotifyError, Store
from memcat.core.worker import Worker, WorkerConfig

__all__ = ["Runtime", "aclose_all", "get_runtime"]

_IDLE_STOP_S = 300.0


class Runtime:
    """The store view, key index, L1 cache, breaker, metrics and worker that every
    call on one event loop shares for one (store, namespace).

    The first config to create it supplies the process-wide settings (L1 size,
    queue size, default policy); each service still sends its own policy with
    every job. :meth:`acquire` and :meth:`release` count the calls using it, and
    it stops its background tasks after five idle minutes.
    """

    def __init__(self, store: Store, cfg: MemcatConfig):
        self.store_root = store
        self.store: BoundStore = store.bind(cfg.namespace)
        self.metrics = Metrics()
        self.breaker = Breaker(on_open=lambda: self.metrics.inc("breaker_open"))
        self.l1 = L1Cache(cfg.l1_max_bytes)
        self.index = KeyIndex(
            self.store, on_purge=self._purged_elsewhere, on_resync=self._evict_missing
        )
        self.loudness = LoudnessTracker()
        self.worker = Worker(
            self.store,
            self.index,
            self.l1,
            self.breaker,
            self.metrics,
            self.loudness,
            WorkerConfig(
                policy=cfg.policy(),
                queue_size=cfg.queue_size,
                queue_max_bytes=cfg.queue_max_bytes,
            ),
        )
        self.lookup = Lookup(
            self.store,
            self.l1,
            self.index,
            self.breaker,
            self.metrics,
            on_hits=self.worker.touch,
            on_corrupt=self.worker.delete_corrupt,
            on_decoded=self.worker.promote,
            purged=self.worker.tomb.blocks,
        )
        self._refs = 0
        self._idle_task: asyncio.Task[None] | None = None
        # Where get_runtime registered this runtime (the loop held weakly), or None.
        self._key: tuple[weakref.ref[asyncio.AbstractEventLoop], Hashable] | None = None

    def acquire(self) -> None:
        self._refs += 1
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None
        self.index.start()
        self.worker.start()
        # Held by a call after an idle stop took it out of the registry: back in, unless
        # a newer runtime has the slot (then this one serves its calls and stops idle).
        if self._key is not None:
            loop_ref, key = self._key
            loop = loop_ref()
            if loop is not None:
                _registry.setdefault(loop, {}).setdefault(key, self)

    async def release(self) -> None:
        self._refs = max(0, self._refs - 1)
        if self._refs:
            return  # other calls keep the worker busy; this call's teardown never waits
        await self.worker.drain()
        if self._refs == 0 and self._idle_task is None:
            self._idle_task = asyncio.ensure_future(self._stop_when_idle())

    async def _stop_when_idle(self) -> None:
        await asyncio.sleep(_IDLE_STOP_S)
        if self._refs:
            return
        # Leave the registry first, so a call arriving while the tasks wind down gets a
        # fresh runtime instead of one being stopped (and the registry never grows
        # with runtimes nobody uses).
        self._unregister()
        await self.stop()
        if self._refs:
            # A call that already held this runtime acquired it while it stopped; its
            # start() calls were no-ops on tasks this stop then cancelled.
            self.index.start()
            self.worker.start()

    async def stop(self) -> None:
        if self._idle_task is not None and self._idle_task is not asyncio.current_task():
            self._idle_task.cancel()
        self._idle_task = None
        await self.worker.stop()
        await self.index.stop()

    async def purge(self, digests: list[bytes] | None) -> int:
        """Delete entries everywhere: the store, this process, and (through the store's
        notifications) every other process's L1 and key index."""
        n = await self.store.delete(digests)
        self.forget(digests)
        msgs = [PURGE_ALL] if digests is None else [PURGE_ONE + d for d in digests]
        for attempt in range(3):
            try:
                many = getattr(self.store, "publish_many", None)
                if many is not None:  # one pipelined round trip where the store offers it
                    await many(msgs)
                else:
                    for m in msgs:
                        await self.store.publish_new(m)
                return n
            except Exception as e:
                logger.warning(f"memcat: purge notification failed ({type(e).__name__})")
                if attempt == 2:
                    # The audio left the store, but other processes may still hold it
                    # in memory: an erasure must not report success.
                    raise PurgeNotifyError(n, e) from e
                await asyncio.sleep(0.5 * (attempt + 1))
        return n  # unreachable; for the type checker

    async def purge_texts(self, texts: Sequence[str]) -> int:
        """Purge every cached variant of each text (any voice, settings or sample
        rate), with its admission counts, everywhere."""
        tags = {text_tag(v) for t in texts if t.strip() for v in _purge_variants(t)}
        found: set[bytes] = set()
        find_all = getattr(self.store, "find_prefixes", None)
        if find_all is not None:  # one pass over the store for every tag
            found.update(await find_all(sorted(tags)))
        else:  # a custom store without prefix lookup: one full scan
            async for d in self.store.scan():
                if d[:TEXT_TAG_BYTES] in tags:
                    found.add(d)
        found.update(d for d in self.l1.digests() if d[:TEXT_TAG_BYTES] in tags)
        return await self.purge(sorted(found)) if found else 0

    async def purge_text(self, text: str, cfg: MemcatConfig) -> int:
        """Purge ``text`` and each clause ``cfg`` splits it into (see :meth:`purge_texts`)."""
        clauses = split(text, min_words=cfg.min_clause_words, extra_blocks=cfg.compiled_blocks)
        return await self.purge_texts([text, *(c.text for c in clauses)])

    def _unregister(self) -> None:
        if self._key is None:
            return
        loop_ref, key = self._key
        loop = loop_ref()
        per_loop = _registry.get(loop) if loop is not None else None
        if per_loop is not None and per_loop.get(key) is self:
            del per_loop[key]
            if not per_loop and loop is not None:
                del _registry[loop]

    def _evict_missing(self) -> None:
        """After a rescan of the store: drop what this process still holds that the
        store no longer has (purged while notifications were lost)."""
        gone = [d for d in self.l1.digests() if d not in self.index]
        if gone:
            self.worker.tomb.mark(gone)
            for d in gone:
                self.l1.discard(d)

    def _purged_elsewhere(self, digests: list[bytes] | None) -> None:
        self.forget(digests)
        self.worker.purged_elsewhere(digests)

    def forget(self, digests: list[bytes] | None) -> None:
        self.worker.tomb.mark(digests)
        if digests is None:
            self.index.clear()
            self.l1.clear()
        else:
            for d in digests:
                self.index.discard(d)
                self.l1.discard(d)


_TRAILING = ("", ",", ".", "?", "!", ";", ":", " —", " –", " -", "—", "…")


def _purge_variants(text: str) -> list[str]:
    """``text`` as given, and with each ending a clause can be cached with: the
    splitter keeps a clause's punctuation, so "Hi John" was stored as "Hi John," or
    "Hi John." depending on what followed it."""
    base = text.strip().rstrip(".,;:!?…—–- ")
    return [text, *(base + end for end in _TRAILING)] if base else [text]


_registry: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Hashable, Runtime]] = (
    weakref.WeakKeyDictionary()
)


def get_runtime(cfg: MemcatConfig) -> Runtime:
    """Return the runtime for this store and namespace on the running loop."""
    loop = asyncio.get_running_loop()
    if loop not in _registry:
        # Runtimes hold their loop (tasks, queues), so a finished loop (one per
        # asyncio.run()) never leaves the weak registry by itself.
        for old in [lp for lp in _registry if lp.is_closed()]:
            del _registry[old]
    per_loop = _registry.setdefault(loop, {})
    key = (cfg.store.identity(), cfg.namespace)
    rt = per_loop.get(key)
    if rt is None:
        rt = Runtime(cfg.store, cfg)
        rt._key = (weakref.ref(loop), key)
        per_loop[key] = rt
    return rt


async def aclose_all() -> None:
    """Stop every runtime on the running loop (tests, scripts, graceful exit)."""
    loop = asyncio.get_running_loop()
    per_loop = _registry.pop(loop, {})
    for rt in per_loop.values():
        rt._key = None  # closed for good: a late acquire must not register it again
        with contextlib.suppress(Exception):
            await rt.stop()
        closer = getattr(rt.store_root, "aclose", None)
        if closer is not None:
            with contextlib.suppress(Exception):
                await closer()
