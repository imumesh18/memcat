---
title: Configuration
description: Every MemcatConfig field, its default and what it does, plus what goes into a cache key.
sidebar:
  order: 3
---

# Configuration

`MemcatConfig` is passed to the wrapped service as `memcat=`. Every field has a safe
default; most deployments set only `store` and `namespace`. A config cannot be
changed after it is built: use `dataclasses.replace(cfg, ...)` for a variant.

```python
from memcat import MemcatConfig, RedisStore

MemcatConfig(
    store=RedisStore("redis://..."),   # default: one MemoryStore per process; RedisStore, S3Store
    namespace="acme-support",          # tenant prefix on every key
    admit_after_calls=3,               # distinct calls before a clause is cached
    warm=["Thank you for calling ABC Shop, how can I help you today?"],  # cached at start
    block_patterns=[r"\bACME-\w+"],    # extra text that must never be cached
    ttl_s=30 * 24 * 3600,              # refreshed when served
    codec="auto",                      # mu-law at 8 kHz, Opus when installed, else PCM16
    enabled=True,                      # kill switch: False means a pure pass-through
)
```

## Fields

| Field | Default | What it does |
| --- | --- | --- |
| `store` | one `MemoryStore` per process | Shared store (L2). Every config without a store shares the same process-wide `MemoryStore`, so calls in one process count toward admission together. `RedisStore(url)` shares the cache across bot processes and is the store for live calls; `S3Store(bucket)` (S3, R2, MinIO) is a durable store for `warm`, `export` and `import`, too slow for the live path. L1 is always in process. See [Stores](./api.md#stores). |
| `namespace` | `"default"` | Prefix on every key. Use one per tenant, brand or voice catalogue; namespaces never share audio. |
| `admit_after_calls` | `3` | A clause is cached only after this many distinct calls said it. This is the main privacy guard for things the text rules cannot see, such as names; it counts calls, not callers (see [Admission](./privacy.md#admission)). |
| `min_clause_words` | `3` | Shortest clause split out of a sentence. A one- or two-word opener ("Sure, ") is split off anyway when the rest is a full clause. |
| `ttl_s` | 30 days | Entry lifetime, refreshed when served (at most once an hour per entry). |
| `l1_max_bytes` | 256 MB | Bound on decoded clips kept in memory per process (shared by every call; the first service to start sets it). |
| `codec` | `"auto"` | `auto` stores mu-law at 8 kHz or below, Opus when `memcat[opus]` is installed and the rate is one Opus supports (8, 12, 16, 24 or 48 kHz), else PCM16. Or choose `pcm16`, `mulaw` or `opus`; `opus` falls back to PCM16 at other rates or without `memcat[opus]`. |
| `lossless_only` | `False` | Never store lossy audio. Forced on for Deepgram, whose [terms](https://deepgram.com/terms) (section 2.4) forbid altering watermarks in its output. |
| `render` | `True` | Render admitted clauses in the background with a second instance of the same service. Off: only clauses captured from live audio are stored. |
| `warm` | `()` | Phrases rendered at start, skipping admission (still subject to the privacy rules). |
| `block_patterns` | `()` | Extra regular expressions (case-insensitive) that make a clause uncacheable. |
| `queue_size` | `1000` | Bound on background jobs; overflow is dropped and counted (`dropped_jobs`), never awaited. |
| `queue_max_bytes` | 64 MB | Bound on the audio held by queued background jobs (captured clips and loudness samples); jobs past it are dropped and counted (`dropped_jobs`, `dropped_bytes`). |
| `enabled` | `True` | Kill switch. `False` makes the wrapped service a pure pass-through. |
| `prices` | built in | USD per 1M characters behind `usd_saved`, keyed by `(adapter, model prefix)`. Your entries are merged over the [built-in table](./providers.md#pricing-table). |
| `gain_match` | `True` | Match replayed loudness to the live voice, within ±3 dB. |
| `trim_joins` | `True` | Set the pause at each join with a cached clip the way live speech does. |
| `clause_pause_ms` | `320` | Pause after a full clause at a comma (about 100 ms after a short opener, +40 ms at `;` `:` dashes). |
| `max_stream_flushes_per_min` | adapter default | Cap on the extra flushes Memcat may spend on WebSocket providers. For Deepgram it can only lower the built-in limit of 20 per 60 s. When no flush is left, a cached clause after a live one is spoken live instead. |

## Validation

`MemcatConfig(...)` raises `ValueError` when:

- `namespace` is not 1 to 64 characters of letters, digits, `_`, `.` or `-`, or is
  only dots;
- `admit_after_calls`, `min_clause_words`, `queue_size` or `queue_max_bytes` is below 1;
- `ttl_s` is below 60 seconds;
- `codec` is not one of `auto`, `mulaw`, `pcm16` or `opus`.

An invalid regular expression in `block_patterns` raises `re.error`.

## What is in a key

A key is 16 bytes: a 4-byte tag of the normalised clause text alone (so a purge
finds every variant of a text), then 12 bytes of BLAKE2b over canonical JSON of the
wrapped service class, `model`, `voice`, `language`, every other runtime setting
(speed, emotion, stability...), constructor options that change audio (e.g.
Cartesia `encoding`), the output sample rate, and the normalised clause text (NFKC,
straight quotes, collapsed spaces; the capital of a common opening word such as
"Sure" is folded). Secrets, timeouts and per-session ids never enter a key: a
setting is left out when its name is a secret or id name (`api_key`, `token`,
`auth_token`, `password`, `session_id`, `request_id`, `user_id` and the like) or
contains `timeout` or `retries`. `model`, `voice` and `language` are always kept. A
`TTSUpdateSettingsFrame` changes the key from the next sentence on.

So a change of voice, model, speed or sample rate starts a new set of entries. In
Redis and in memory the old ones expire after `ttl_s`; on S3, the bucket's lifecycle
rules decide.

## Services that need their own adapter

A service class can declare how Memcat should treat it:

```python
from memcat.pipecat.adapters.providers import CartesiaAdapter

class MyCartesia(CartesiaTTSService):
    __memcat_adapter__ = CartesiaAdapter
```

Subclasses of the certified services pick their adapter automatically. See
[Providers](./providers.md).

## In `memcat.toml`

The CLI reads the same fields from the `[memcat]` table of `memcat.toml`, except
`store` (it comes from `[store]`) and `prices`. See [Operations](./operations.md#memcattoml).
