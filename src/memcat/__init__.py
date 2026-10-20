"""Memcat: a clause-level text-to-speech cache for Pipecat.

from memcat import cached, MemcatConfig, RedisStore

CachedCartesia = cached(CartesiaTTSService)
tts = CachedCartesia(api_key=..., voice_id=..., memcat=MemcatConfig(store=RedisStore(url)))
"""

from importlib import metadata as _metadata

from memcat.config import MemcatConfig
from memcat.core.runtime import aclose_all as _aclose_runtimes
from memcat.core.store.base import BoundStore, PurgeNotifyError, Store, StoreError
from memcat.core.store.memory import MemoryStore
from memcat.core.store.redis import RedisStore
from memcat.core.store.s3 import S3Store
from memcat.pipecat.mixin import MemcatTTSMixin
from memcat.pipecat.wrap import cached

try:
    __version__ = _metadata.version("memcat")
except _metadata.PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "BoundStore",
    "MemcatConfig",
    "MemcatTTSMixin",
    "MemoryStore",
    "PurgeNotifyError",
    "RedisStore",
    "S3Store",
    "Store",
    "StoreError",
    "__version__",
    "aclose",
    "cached",
]


async def aclose() -> None:
    """Stop Memcat's background work on the running event loop and close its clients."""
    from memcat.pipecat.render import aclose_render_pools

    await _aclose_runtimes()  # workers first, so no render starts a new twin
    await aclose_render_pools()
