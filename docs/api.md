---
title: Python API
description: Everything the memcat package exports, the stores and their arguments, and the methods on a wrapped service.
sidebar:
  order: 7
---

# Python API

Everything below is importable from `memcat`:

```python
from memcat import (
    cached, MemcatConfig, MemoryStore, RedisStore, S3Store,
    Store, BoundStore, StoreError, PurgeNotifyError, MemcatTTSMixin, aclose,
    __version__,
)
```

## `cached(cls)`

Returns a subclass of `cls`, a Pipecat `TTSService` subclass, with Memcat first in
its method resolution order. The subclass is named `Memcat` plus the class name
(`MemcatCartesiaTTSService`) and takes one extra keyword argument,
`memcat=MemcatConfig(...)`. Every other argument goes to `cls` unchanged.

- Calling `cached()` twice on the same class returns the same subclass.
- Calling it on a class that is already wrapped returns that class.
- Anything that is not a `TTSService` subclass raises `TypeError`.

Call it once, at import time, and build a new instance per call.

## Methods on a wrapped service

### `memcat_stats()`

```python
stats = await tts.memcat_stats()   # dict
```

This service's counters, plus a `process` block once the service has started. See
[Metrics](./operations.md#metrics).

### `memcat_purge()`

```python
deleted = await tts.memcat_purge("Thanks for waiting.")   # int
```

Deletes the cached audio and admission counts for the text and each of its
clauses, in every voice, setting and sample rate, in the store and in every running
process, and returns the number of entries deleted. On S3, other processes drop it
within `events_poll_s` (see [Erasure](./privacy.md#erasure)). With no argument, it
deletes the whole namespace.

Raises `StoreError` if the store fails, and `PurgeNotifyError` (a `StoreError`
with a `deleted` count) if the entries were deleted but other processes could not
be told after three tries. Purging again is safe.

## `MemcatConfig`

A dataclass with the fields listed in [Configuration](./configuration.md).
`MemcatConfig()` with no `store` uses one `MemoryStore` shared by the whole process,
so calls in one process count toward admission together.

## Stores

A store holds the encoded audio (L2). Every process also keeps decoded clips in
memory (L1, bounded by `l1_max_bytes`).

### `MemoryStore`

```python
MemoryStore(max_bytes=512 * 1024 * 1024, max_admission_keys=100_000)
```

A store inside one process, for a single process, tests and development. Every
`MemcatConfig` without a `store` shares one process-wide `MemoryStore`; build your
own only to keep a cache apart.
`max_bytes` bounds the stored bytes; the least recently used entries go first.
`max_admission_keys` bounds the clauses being counted for admission.

### `RedisStore`

```python
RedisStore(url="redis://localhost:6379/0", *, pool_size=10, op_timeout_ms=50, client=None)
```

A shared store on Redis 6+ or Valkey 7+ (`pip install "memcat[redis]"`; without it
the constructor raises `ImportError`). Tested on
Redis 7 and 8 and Valkey 8. This is the store for live calls across several
processes.

| Argument | Meaning |
| --- | --- |
| `url` | `redis://` or `rediss://` URL. |
| `pool_size` | Maximum command connections per event loop. A command waits for a free one within its own time budget. |
| `op_timeout_ms` | Budget of the cache read on the call path (one `MGET`); a slower read counts as a miss and as a store error. |
| `client` | An existing `redis.asyncio.Redis` to use instead of `url`, on its own event loop only; it also carries the subscription. |

The store keeps one blocking connection pool per event loop. Pub/sub runs on its
own connection, never one from the pool. Each operation has its own time budget:
`op_timeout_ms` for the `MGET` on the call path, 2 s for writes (entries, TTL
refreshes, admission counts, locks, notifications), 5 s for each page of a scan,
each batch of deletes, each batch of purge notifications and each bulk read
(`stats`, `export`, the check before a render), and 10 s for each batch of
`purge --older-than` reads.

Keys, where `{ns}` is the namespace and `{hex}` the 32-character digest:

| Key | Type | Holds |
| --- | --- | --- |
| `memcat:{ns}:e:{hex}` | string | The entry, its TTL refreshed when served. |
| `memcat:{ns}:t:{hex}` | string | Unix time the entry was last written or served (for `purge --older-than`); same TTL as the entry. |
| `memcat:{ns}:a:{hex}` | HyperLogLog | Distinct calls that missed the clause; deleted once it is admitted. |
| `memcat:{ns}:lock:{hex}` | string | The render lock, 60 s. It holds a random token of its owner and is released by compare-and-delete, so a render that outlived its lock never releases another process's. |
| `memcat:{ns}:new` | pub/sub channel | New entries and purges, so other processes stay in sync. |

### `S3Store`

```python
S3Store(bucket, prefix="memcat", endpoint_url=None, *, client=None, events_poll_s=10.0, **client_kwargs)
```

An object-storage store for S3, Cloudflare R2 or MinIO (`pip install "memcat[s3]"`;
without it the constructor raises `ImportError`),
for durable copies of a namespace: the target of `memcat warm`, `export` and
`import`. At most 10 requests run at once per store, and each may take up to 10 s
before it fails (time spent waiting for a slot does not count). On the call path a
read gets 100 ms and is otherwise a miss, so it is not meant as the store a live
call reads from. A delete that S3 rejects for some objects
raises `StoreError` naming the first failure.

| Argument | Meaning |
| --- | --- |
| `bucket` | Bucket name. |
| `prefix` | Key prefix inside the bucket. Objects live at `{prefix}/{namespace}/{hex}`. |
| `endpoint_url` | A custom endpoint, for R2 or MinIO. |
| `client` | An existing boto3 S3 client. |
| `events_poll_s` | How often each process checks for purges made by other processes (one LIST request per poll). |
| `**client_kwargs` | Passed to `boto3.client("s3", ...)`, e.g. `region_name`. |

Objects have no TTL; bucket lifecycle rules decide how long they live. New entries
are not announced to other processes; purges are, through small event objects under
`{prefix}/{namespace}/_events/` that each process polls (see
[Erasure](./privacy.md#erasure)). Admission counts and render locks are kept in each
process.

### Your own store

`Store` and `BoundStore` are the protocols a store implements, and `StoreError` is
the exception a store raises when an operation fails or times out; Memcat treats it
as a miss. A `Store` has `identity()` (equal for stores that reach the same data)
and `bind(namespace)`, which returns a `BoundStore`. A `BoundStore` has
`get_many`, `put`, `touch`, `delete`, `scan`, `publish_new`, `subscribe_new`,
`observe`, `forget_admission`, `try_lock`, `unlock` and `close`.

These are optional; Memcat uses them when a `BoundStore` has them:

- `find_prefixes(prefixes: Sequence[bytes]) -> list[bytes]` returns the digests of
  entries and admission counts that start with any of `prefixes`, in one pass.
  `purge --text` and `memcat_purge(text)` use it to find every variant of a text and
  its clauses. Without it, a purge scans the namespace's entries once, and admission
  counts of a text that was never cached are left to expire.
- `publish_many(messages: Sequence[bytes]) -> None` publishes several notifications
  at once (a bulk purge); without it Memcat calls `publish_new` once per message.
- `get_bulk(digests: Sequence[bytes]) -> list[bytes | None]` reads entries off the
  call path (`stats`, `export`, the check before a render) with a longer budget;
  without it Memcat uses `get_many`.
- `read_timeout_s: float`, the store's own budget for `get_many` on the call path;
  without it a call-path read gets 100 ms.

Digests are 16 bytes: a 4-byte tag of the clause text, then 12 bytes of BLAKE2b
over the full key. See
[`core/store/base.py`](https://github.com/imumesh18/memcat/blob/main/src/memcat/core/store/base.py)
for the exact signatures, and the three built-in stores for examples.

## `MemcatTTSMixin`

The class `cached()` puts first in the method resolution order.
`isinstance(tts, MemcatTTSMixin)` tells whether a service is wrapped.

## `aclose()`

```python
await memcat.aclose()
```

Stops Memcat's background work on the running event loop and closes its clients,
including the render twins. Call it when a process shuts down cleanly.

## `__version__`

The installed version of the package, such as `"1.0.0"`.
