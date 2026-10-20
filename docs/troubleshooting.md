---
title: Troubleshooting
description: Why a clause is not cached, why Memcat turned itself off, and other common questions.
sidebar:
  order: 8
---

# Troubleshooting

## Nothing is cached yet

A clause is cached only after it has been heard in `admit_after_calls` (default 3)
distinct calls. A call is one run of the service's pipeline: each time the service
starts, it gets a new call id. Saying the same line three times in one call does not
admit it.

Once admitted, the clause is stored in the background: at once if a live run
synthesized it alone, otherwise after the render twin has synthesized it. Watch
`entries_stored`, `captures` and `renders` in the `process` block of
`await tts.memcat_stats()`.

To have lines cached from the first call, list them in `MemcatConfig(warm=[...])`
or render them with `memcat warm` (see [Operations](./operations.md#commands)).

## A clause is never cached

Check, in order:

1. **Privacy rules.** A clause with any digit or numeral, a currency symbol, an `@`,
   a URL or domain (written or spoken), two number words in a row, a number word next
   to a month, a money word or "am"/"pm", a spelled-out code, or a long all-caps word
   is never cached. Neither is a clause that matches one of your `block_patterns`, or
   a clause holding a number word next to a clause blocked for a number. See
   [Privacy](./privacy.md#text-rules).
2. **Splitting.** Memcat splits only at commas, semicolons and colons followed by a
   space, and at dashes. A piece shorter than `min_clause_words` (default 3) is
   merged into a neighbouring cacheable piece when there is one, so it is cached as
   part of that clause, not alone. A one- or two-word opener such as "Sure, " is the
   exception: it stays a clause of its own. Markup such as SSML tags is never split.
3. **The key.** The voice, model, language, every other setting and the output
   sample rate are part of the key. Audio cached for one voice or rate is not used
   for another. See [What is in a key](./configuration.md#what-is-in-a-key).
4. **`render=False`.** Then only clauses that a live run synthesized alone are
   stored; a clause always spoken next to others is never stored.
5. **The provider.** On Sarvam, Fish Audio and other WebSocket services in leading
   mode, cached clauses play only before the first live clause of a turn's audio
   context; the rest is spoken live. See
   [Providers](./providers.md).

To see which clauses of real transcripts would be cached, run
`memcat warm --from calls.jsonl --dry-run`.

## Memcat turned itself off

`await tts.memcat_stats()` shows `"enabled": false`, and one warning starting with
`memcat:` is in the log. The service keeps working as if Memcat were not installed.
The warning says why:

| Warning | Cause |
| --- | --- |
| `is a WebSocket service Memcat cannot replay into safely; caching is off for it` | A WebSocket service without a segmented adapter that does not push its own start frames (`passthrough` mode). |
| `TOKEN aggregation mode is not supported; caching is off` | The service uses Pipecat's `TOKEN` text aggregation. |
| `disabled for this service: settings cannot be keyed` | A setting has no canonical form, so no key can be built. |
| `disabled for this service: audio arrives outside run_tts; caching is off` | An HTTP or local service delivers audio from a background task instead of returning it. |
| `disabled for this service: could not start`, `planning failed`, or `replay failed; the rest is sent live` | An error inside Memcat. The rest of the sentence was spoken live. |

`enabled=False` in `MemcatConfig` also turns it off, without a warning. The
details of an error are logged at DEBUG: set `MEMCAT_LOG=DEBUG` for the CLI, or let
your bot's `loguru` sink show DEBUG messages.

## Hits in one process but not in another

`MemoryStore` keeps the cache inside one process. Use `RedisStore` to share it
across bot processes. For the same reason, every CLI command except `serve` and
`warm --dry-run` refuses a `memcat.toml` with no `[store]` or with
`[store] type = "memory"`.

Each `namespace` is a separate cache. Make sure the bots and the CLI use the same
one.

## `store_errors` or `breaker_open` keep rising

Store calls that fail or run out of time count as store errors, and the clauses
they were reading are spoken live. On Redis the cache read on the call path gets
`op_timeout_ms` (50 ms by default), and on any other store 100 ms; background
writes get 2 s. When too many store calls fail, the breaker opens and Memcat uses
only the in-memory cache for 30 s (see
[Running it well](./operations.md#running-it-well)). Keep Redis in the same zone as
the bots, or raise `RedisStore(op_timeout_ms=...)`.

## `dropped_jobs` is above 0

The background queue was full (`queue_size`, default 1000) or held too much audio
(`queue_max_bytes`, default 64 MB; `dropped_bytes` counts the audio dropped), so
some admission counts, captures, loudness samples or writes were skipped. The calls
were not affected. Raise `queue_size` or `queue_max_bytes` if it keeps happening.

## Opus entries are missed

A process without PyAV (`memcat[opus]`) cannot decode Opus entries, so it treats them
as misses and leaves them in the store for processes that can. Install the extra in
every process that shares the store, or use `codec="mulaw"` or `"pcm16"`.

## Deepgram audio is stored as PCM16

Deepgram's terms forbid altering the watermarks in its audio, so Memcat forces
`lossless_only` for Deepgram and never stores lossy audio for it.

## A failed render is not tried again

A render that fails, times out or ends before the provider finishes the clause
stores nothing and is not retried for an hour. The clause is spoken live meanwhile.

## A provider stream was closed

The warning `memcat: a provider stream went quiet without finishing; closing it`
(or `sent no audio`, or `never started`) means a WebSocket provider did not finish a
stream Memcat opened for live clauses. Memcat closes it once it has been quiet for
Pipecat's `stop_frame_timeout_s` (3 s by default) after it was asked to end, or
after 30 s if it never got going, plays the clauses queued behind it, and never
caches its audio. Memcat stays on. It is logged once per service.

## The CLI exits with status 1

The message after `memcat:` names the problem: a missing or invalid `memcat.toml`,
an unset `${ENV}` variable, a bad `[memcat]`, `[store]` or `[serve]` option, a
service that cannot be built, or a bad input line. Also:

- Every command except `serve` and `warm --dry-run` needs a Redis or S3 `[store]`.
- `purge --all` without `--yes`, when standard input is not a terminal (or `N` at
  the prompt).
- `import` from a directory that does not exist, or from an export of another
  namespace or with no `manifest.json` (pass `--force` to import it anyway).
- `export` into a directory that already holds an export.
- `warm stored nothing: N store errors`: the store did not answer in time.
- A `[serve] secret` that is not a string: quote it in `memcat.toml`.
- `serve` on an address other than loopback with no secret: set `[serve] secret` or
  `MEMCAT_SERVE_SECRET`, or pass `--insecure-no-auth` if something in front of
  Memcat authenticates requests.
