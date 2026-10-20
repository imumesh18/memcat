"""The storage contract every backend implements."""

from __future__ import annotations

from collections.abc import AsyncIterator, Hashable, Sequence
from typing import Protocol, runtime_checkable

__all__ = ["DEFAULT_TTL_S", "RESYNC", "BoundStore", "Store", "StoreError"]

DEFAULT_TTL_S = 30 * 24 * 3600  # entry lifetime (refreshed when served): 30 days

# Yielded by ``subscribe_new`` after it reconnects: notifications sent while it was
# disconnected are lost, so the listener should re-read the store.
RESYNC = b"\x02"


class StoreError(Exception):
    """A store operation failed or timed out. Callers treat it as a miss."""


class PurgeNotifyError(StoreError):
    """A purge deleted its entries from the store, but other processes could not be
    told, so they may keep playing the audio from memory. Retry the purge (it is
    idempotent) or restart the other processes."""

    def __init__(self, deleted: int, cause: BaseException):
        super().__init__(
            f"deleted {deleted} entries, but other processes were not notified "
            f"({type(cause).__name__}); retry the purge or restart them"
        )
        self.deleted = deleted


@runtime_checkable
class BoundStore(Protocol):
    """A store bound to one namespace. All digests are 16-byte blake2b values."""

    namespace: str

    async def get_many(self, digests: Sequence[bytes]) -> list[bytes | None]:
        """Fetch entry blobs in one round trip; missing digests come back as None."""
        ...

    async def put(self, digest: bytes, blob: bytes, ttl_s: int) -> None: ...

    async def touch(self, digests: Sequence[bytes], ttl_s: int) -> None:
        """Refresh the TTL of entries that were just served."""
        ...

    async def delete(self, digests: Sequence[bytes] | None = None) -> int:
        """Delete entries (and their admission state). None deletes the whole namespace."""
        ...

    def scan(self) -> AsyncIterator[bytes]:
        """Iterate every digest stored in this namespace."""
        ...

    async def publish_new(self, digest: bytes) -> None:
        """Tell other processes a new entry exists."""
        ...

    def subscribe_new(self) -> AsyncIterator[bytes]:
        """Yield notifications other processes publish: 16-byte digests of new entries,
        purge markers, and :data:`RESYNC` after a reconnect that may have lost some."""
        ...

    async def observe(self, digest: bytes, call_id: str, k: int) -> bool:
        """Record that ``call_id`` missed this clause; True once k distinct calls have."""
        ...

    async def forget_admission(self, digests: Sequence[bytes] | None = None) -> None:
        """Drop admission counts for these digests (None: all in the namespace)."""
        ...

    async def try_lock(self, digest: bytes, ttl_s: int) -> bool:
        """Take the render lock for a digest; False when another process holds it."""
        ...

    async def unlock(self, digest: bytes) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class Store(Protocol):
    """An unbound store, as users construct it (``RedisStore(url=...)``)."""

    def identity(self) -> Hashable:
        """A value equal for stores that reach the same data, so calls can share one runtime."""
        ...

    def bind(self, namespace: str) -> BoundStore:
        """Return a view of this store limited to one namespace."""
        ...
