---
title: Getting started
description: Install Memcat, wrap a Pipecat TTS service, and check that lines come from the cache.
sidebar:
  order: 1
---

# Getting started

## Requirements

- Python 3.11, 3.12 or 3.13.
- `pipecat-ai` 1.10–1.13.

## Install

```bash
uv add memcat                      # or: pip install memcat
uv add "memcat[redis,opus]"        # shared cache across bot processes, compact audio
```

Extras:

| Extra | Adds |
| --- | --- |
| `redis` | `RedisStore` (Redis 6+ or Valkey 7+) |
| `opus` | Opus storage (PyAV) |
| `s3` | `S3Store` (boto3) |
| `all` | all of the above |

## Wrap a service

```python
from pipecat.services.cartesia.tts import CartesiaTTSService
from memcat import cached, MemcatConfig, RedisStore

CachedCartesia = cached(CartesiaTTSService)   # once, at import time

tts = CachedCartesia(
    api_key=CARTESIA_API_KEY,
    voice_id=VOICE_ID,
    memcat=MemcatConfig(store=RedisStore(url=REDIS_URL), namespace="acme-support"),
)
```

`cached(cls)` returns a subclass of `cls` that takes one extra keyword argument,
`memcat=`. Use `tts` exactly as you used the unwrapped service. Nothing else in the
pipeline changes. Without `memcat=`, the service uses `MemcatConfig()`: one
`MemoryStore` shared by the whole process and the `"default"` namespace.

Build a new service for every call, as you would without Memcat. Calls in one
process that use the same store and namespace share one key index, one in-memory
cache, one background worker and one connection pool.

`MemoryStore` keeps the cache inside one process. Use `RedisStore` when several bot
processes should share it. See [Configuration](./configuration.md) for every option.

## Hear it with no API keys

[`examples/kokoro_local.py`](https://github.com/imumesh18/memcat/blob/main/examples/kokoro_local.py)
speaks three lines twice with Kokoro, a local voice. The second time, the reusable
clauses come from the cache. From a clone of the repository:

```bash
git clone https://github.com/imumesh18/memcat
cd memcat
uv run examples/kokoro_local.py
```

In your own project, install Kokoro first (`uv add "pipecat-ai[kokoro]" memcat`),
then copy the script and run it.

For each call it prints the time to first audio of every line and the `hits`,
`misses` and `hit_rate_chars` counters. It then writes the second call's audio to
`memcat_demo.wav`. The demo sets `admit_after_calls=1`, so a clause is cached after
one call instead of the default three. The clause "Your order 48213 has shipped"
holds a digit, so it is spoken live both times.

## Check that it works

```python
stats = await tts.memcat_stats()
```

`memcat_stats()` returns this service's counters plus a `process` block for
everything in the process that shares its store and namespace. Look at:

- `enabled`: `True` when Memcat is active for this service. If it is `False`, see
  [Troubleshooting](./troubleshooting.md#memcat-turned-itself-off).
- `hits` and `misses`: clauses served from the cache and clauses spoken live.
- `hit_rate_chars`: the share of characters served from the cache.
- `usd_saved`: what the cached characters would have cost at list price.
- `adapter` and `mode`: how Memcat handles this provider (see [Providers](./providers.md)).

Expect no hits at first. A clause is cached only after it has been heard in 3
distinct calls (`admit_after_calls`), and then it is stored in the background. To
have common lines cached from the start, list them in `warm` (see
[Configuration](./configuration.md)) or render them with `memcat warm` (see
[Operations](./operations.md)).

Every metric is listed in [Operations](./operations.md#metrics).

## Shut down cleanly

```python
import memcat

await memcat.aclose()
```

`memcat.aclose()` stops Memcat's background work on the running event loop and
closes its clients. Call it when a process shuts down cleanly.

## Next steps

- [How it works](./how-it-works.md) explains what happens to each sentence.
- [Privacy](./privacy.md) lists what is never cached.
- [Operations](./operations.md) covers the `memcat` CLI and `memcat serve` for hosted platforms such as Vapi.
- [Examples](https://github.com/imumesh18/memcat/tree/main/examples) has a Twilio phone bot and a WebRTC browser bot.
