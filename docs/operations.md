---
title: Operations
description: The memcat command line, memcat serve for hosted voice platforms, metrics, tracing and running Memcat in production.
sidebar:
  order: 6
---

# Operations

```bash
memcat warm  --config memcat.toml --from calls.jsonl --top 500   # render your most repeated clauses
memcat stats --config memcat.toml                                # entries, minutes, codecs
memcat purge --config memcat.toml --text "..."                   # erase a text everywhere
memcat export --config memcat.toml --to backup/ ; memcat import --config memcat.toml --from backup/
memcat serve --config memcat.toml --port 8080                    # TTS endpoint for Vapi and other platforms
```

## `memcat.toml`

Every CLI command reads one file, `memcat.toml` in the current directory unless
`--config PATH` says otherwise:

```toml
[service]
class = "pipecat.services.cartesia.tts.CartesiaTTSService"
sample_rate = 8000                 # the rate your pipeline plays; part of every key

[service.args]                     # constructor arguments; ${ENV} and ${ENV:-default} work
api_key = "${CARTESIA_API_KEY}"
voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"

[store]
type = "redis"                     # redis | s3 (memory only for serve and warm --dry-run)
url = "${REDIS_URL}"

[memcat]                            # MemcatConfig fields (store comes from [store];
namespace = "acme-support"         # prices cannot be set here)
```

`--config` goes before or after the command (`memcat --config memcat.toml stats` or
`memcat stats --config memcat.toml`). `stats`, `export`, `import` and every form of
`purge` use only `[store]` and `[memcat]`, so they need no provider package or API
key. `warm` (also `warm --from ... --dry-run`, which splits clauses with the
service's settings) and `serve` build the service from `[service]`; `serve` builds one per
requested sample rate and ignores `[service] sample_rate`.

Every command except `serve` and `warm --dry-run` needs a shared store: a `[store]`
of type `redis` or `s3`. A `memory` store would be empty and gone when the command
exits, so those commands refuse it, and refuse a `memcat.toml` with no `[store]`.

A missing or invalid config, an unset `${ENV}`, a missing store or a bad input line
prints `memcat: <reason>` and exits with status 1; a usage error exits with status 2.
`MEMCAT_LOG=DEBUG` turns on Memcat's logs (the default level is `WARNING`).

`[store]` takes `type` plus the store's own arguments (see [Stores](./api.md#stores)):

| `type` | Other keys |
| --- | --- |
| `memory` (default) | none |
| `redis` | `url` (default `redis://localhost:6379/0`), `pool_size`, `op_timeout_ms` |
| `s3` | `bucket` (required), `prefix`, `endpoint_url`, `events_poll_s` (default 10), and any other keyword argument for `boto3.client("s3", ...)`, such as `region_name` |

Every command except `serve` prints its result as JSON on standard output.

## Commands

| Command | Does |
| --- | --- |
| `memcat warm --from calls.jsonl --top 500` | Ranks clauses by the number of distinct calls that said them and renders the top N (`--min-calls 3`, `--dry-run` to preview). `calls.jsonl` holds one call per line: `{"call_id": "...", "agent_turns": ["..."]}`. |
| `memcat warm --phrase "..."` | Renders the given phrases (repeatable). `--timeout 600` bounds a warm run in seconds. |
| `memcat stats` | Entries, audio minutes, KB per minute, codecs, how entries were made (captured, rendered, warmed). |
| `memcat purge --text "..."` | Deletes the text and each of its clauses in every voice, setting and sample rate, with their admission counts, in the store and in every running process (on S3, see [Erasure](./privacy.md#erasure)). |
| `memcat purge --older-than 30d` | Deletes entries idle this long (Redis: since last written or served; S3: since written, as S3 keeps no access time). Takes a positive age in `s`, `m`, `h` or `d`, e.g. `12h`. |
| `memcat purge --all` | Deletes the namespace. On a terminal it asks `[y/N]` (on standard error); otherwise it needs `--yes`. |
| `memcat export --to DIR` / `memcat import --from DIR` | Copies entries as `<digest>.mcat` files plus a manifest; import validates every entry. |

`purge` takes exactly one of `--text`, `--older-than` and `--all`.

What each command prints:

| Command | Keys |
| --- | --- |
| `warm` | `phrases`, `stored`, `render_failures`, `store_errors`, `chars_rendered`, `seconds`, all for this run only; exits 1 if store errors left nothing stored |
| `warm --dry-run` | `would_render`, and `top`: up to 20 `[clause, calls]` pairs |
| `stats` | `namespace`, `entries`, `bytes`, `audio_minutes`, `kb_per_minute`, `codecs`, `origin`, `corrupt` |
| `purge` | `deleted` |
| `export` | `exported`, `to` |
| `import` | `imported`, `skipped` (files that are not valid entries) |

`stats` counts the entries it could read: an entry listed by the scan but expired or
deleted before it was read is left out.

`export` writes `manifest.json` with the namespace, the format version and the
exported digests, and refuses a directory that already holds an export. `import`
loads only the entries the manifest lists. It refuses an export whose manifest
names another namespace, or one with no manifest, unless you pass `--force`; it then
writes into the namespace in `memcat.toml`. A missing import directory is an error.
An export is a copy of cached audio: a purge does not reach it, so delete old
exports too.

## `memcat serve`

```toml
[serve]
host = "127.0.0.1"            # default; any other address needs a secret
port = 8080
pool_size = 4                 # pipelines per sample rate
secret = "${VAPI_SECRET}"     # or MEMCAT_SERVE_SECRET; checked against X-VAPI-SECRET
max_text_chars = 2000
max_waiting = 16              # per sample rate; default 4 × pool_size
queue_timeout_s = 10
```

`--host`, `--port` and `--pool-size` override the file. When a secret is set,
every `POST` endpoint requires a matching `X-VAPI-SECRET` header. Listening on
anything but loopback requires a secret; `--insecure-no-auth` overrides this when
something in front of Memcat authenticates requests. Without a secret (and without
`--insecure-no-auth`), requests that carry an `Origin` header or a `Host` other than
loopback are refused, so a web page open in a browser cannot use the server. Every
`POST` body must be `Content-Type: application/json`. Without the secret,
`GET /healthz` returns only `{"ok": true}`.

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /vapi` | Vapi `voice-request`: `message.text`, `message.sampleRate`, `message.call.id` | `application/octet-stream`, mono 16-bit little-endian PCM at the requested rate, streamed |
| `POST /v1/tts` | `{"text", "sample_rate", "call_id"}` (`sample_rate` defaults to 24000) | the same |
| `GET /healthz` | | `{"ok", "served", "rejected_busy", "speakers"}`, plus `"cache"` once a pipeline has started; only `{"ok": true}` without the secret |

The platform's call id is the call Memcat counts for admission. Requests without
one all count as a single call, so repeating a line without a call id cannot push
it into the cache.

Errors are JSON `{"error": ...}`: 400 bad request, empty text or unsupported rate,
401 wrong secret, 403 `Origin` or non-loopback `Host` without a secret, 413 text
longer than `max_text_chars`, 415 body not JSON, 502 provider error or no
audio (before any audio was sent), 503 no pipeline available (`Retry-After: 5`), or busy
(queue full or waited longer than `queue_timeout_s`; `Retry-After: 1`), 504 synthesis timed out
(after 30 s). Once audio is streaming, a failure ends the stream early, and so does
a client that stops reading until the 30 s deadline passes. Request
bodies are limited to 1 MB.

Rates: 8000, 16000, 22050, 24000, 44100, 48000 Hz. Each pipeline speaks one request
at a time; up to `max_waiting` more requests per sample rate wait for a free one. A
request whose client disconnects before its turn is dropped without synthesizing
anything. Point Vapi's `voice.server.url` at `https://<host>/vapi`.

`speakers` in `/healthz` is the number of pipelines per sample rate,
`rejected_busy` the number of requests turned away as busy, and `cache` is the
`process` block of `memcat_stats()` described below.

## Metrics

`await tts.memcat_stats()` returns this service's counters plus a `process` block
for everything sharing its store and namespace. Nothing leaves the process.

Counted per service and per process:

| Key | Meaning |
| --- | --- |
| `hits` | Clauses played from the cache. |
| `misses` | Clauses spoken live, cacheable or not. |
| `chars_cached` | Characters played from the cache. |
| `chars_live` | Characters sent to the provider by the call. |
| `hit_rate_chars` | `chars_cached / (chars_cached + chars_live)`. |
| `usd_saved` | `chars_cached` priced with the [pricing table](./providers.md#pricing-table). |
| `chars_rendered` | Characters the background render twin synthesized (the cost of filling the cache). |
| `replay_first_audio_ms_p50`, `replay_first_audio_ms_p99` | Time to first cached audio of a sentence, over the last 2,048 such sentences; `null` before the first. |

Counted per process, by the background worker and the store (they stay 0 in the
service block):

| Key | Meaning |
| --- | --- |
| `entries_stored`, `bytes_stored` | Entries written to the store, and their size. |
| `captures` | Entries stored from live audio, at no extra cost. |
| `renders`, `render_failures` | Clauses rendered by the render twin, and renders that failed. |
| `store_errors` | Store calls that failed or timed out; each one is a miss. |
| `dropped_jobs` | Background jobs dropped because the queue was full (`queue_size`) or held too much audio (`queue_max_bytes`). |
| `dropped_bytes` | Audio bytes in the jobs dropped past `queue_max_bytes`. |
| `breaker_open` | Times the store breaker opened. In the `process` block, this key is instead `true` while the breaker is open. |
| `corrupt_entries` | Entries that failed to decode; each is deleted and counted as a miss. |

`fallbacks` (times Memcat turned itself off for this service) counts in the service
block only. The service block also has `mode`, `adapter` and `enabled`. Only in the `process` block:
`index_keys` (digests in the in-process key index), `index_loaded` (whether the
first scan of the store has finished), `l1_entries` and `l1_bytes` (decoded clips in
memory).

With Pipecat tracing enabled, Memcat adds `memcat.hit`, `memcat.chars_cached`,
`memcat.chars_live`, `memcat.adapter` and `memcat.mode` to Pipecat's TTS spans.

## Running it well

- Keep Redis in the same zone as the bots: the cache read on the call path (one
  `MGET`) gets `op_timeout_ms`, 50 ms by default (100 ms on other stores).
  Background writes get 2 s, and bulk work (scans, deletes, purge notifications,
  bulk reads) 5 s per batch.
- One `RedisStore(url)` per call is fine: calls in a process share one key index,
  L1, worker and connection pool per URL and namespace. A command waits for a free
  pooled connection within its own time budget.
- When at least 5 store calls fail within 10 s and they are at least half of the
  calls in that window, the breaker opens for 30 s and only L1 is used; then one
  probe call decides whether to close it. A probe that has not reported back after
  15 s is given up and the next request probes again.
- On S3, purges reach other processes within `events_poll_s` (default 10 s); see
  [Erasure](./privacy.md#erasure).
- Call `await memcat.aclose()` when shutting a process down cleanly.
