---
title: Memcat
description: A clause-level text-to-speech cache for Pipecat voice bots.
sidebar:
  order: 0
---

# Memcat

Memcat is a text-to-speech cache for [Pipecat](https://github.com/pipecat-ai/pipecat)
voice bots. When your agent says a line it has said before (or part of one), Memcat
replays the stored audio instead of paying the provider to synthesize it again, and
the audio starts in milliseconds.

You wrap your existing Pipecat TTS service with one call. Nothing else in the
pipeline changes.

## Measured

- 54.0% of speech characters served from cache on scripted support calls (Harper
  Valley Bank), 32.4% on typed support chats (ABCD).
- First audio in 2–13 ms when a sentence opens with a cached clause, against 346 ms
  without Memcat, on recorded ElevenLabs traffic.
- No measurable time added on a miss, and no cache key containing a digit on any
  corpus.

[Benchmarks](./benchmarks.md) has every number and how it was measured.

## In short

- **Clauses, not just sentences.** Each sentence is split at commas, semicolons,
  colons and dashes, so a reusable clause is cached even when the rest of its
  sentence is not.
- **Privacy first.** A clause with any digit, a number in words, an email, a URL,
  money, a date, a time or a spelled-out code is never cached, and a clause is
  cached only after it has been heard in 3 distinct calls.
- **Exact order.** Cached and live clauses play in the order they were written, with
  correct transcripts and word timestamps.
- **Fails open.** If the store, a codec or Memcat itself fails, the sentence goes to
  the provider as if Memcat were not installed.

## Pages

- [Getting started](./getting-started.md): install, wrap a service, check that it works.
- [How it works](./how-it-works.md): clauses, ordering, joins, filling the cache, failures.
- [Configuration](./configuration.md): every `MemcatConfig` field and what goes into a cache key.
- [Providers](./providers.md): which Pipecat services are supported, and how.
- [Privacy](./privacy.md): what is never cached and how to erase what is.
- [Operations](./operations.md): the `memcat` CLI, `memcat serve`, metrics and tracing.
- [Python API](./api.md): `cached()`, the stores and the methods on a wrapped service.
- [Troubleshooting](./troubleshooting.md): why a clause is not cached, and other common questions.
- [Benchmarks](./benchmarks.md): how every measured number was produced.

## License

BSD 2-Clause, the same as Pipecat. Keep one cache (or one `namespace`) per customer
so audio is never shared across customers, and check your provider's terms for
anything else. The source is on [GitHub](https://github.com/imumesh18/memcat).
