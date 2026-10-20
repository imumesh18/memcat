---
title: How it works
description: What Memcat does with each sentence, how cached and live audio stay in order, and what happens when something fails.
sidebar:
  order: 2
---

# How it works

1. **Clauses, not just sentences.** Each sentence Pipecat hands to the TTS is split
   at commas, semicolons, colons and dashes. "Thanks for waiting, your order 48213
   has shipped, and it should arrive soon." is three clauses; the first and last are
   reusable, the middle one is spoken live.
2. **Privacy first.** A clause with any digit, a number in words, an email, a URL,
   money, a date, a time or a spelled-out code is never cached. A clause is cached
   only after it has been heard in 3 distinct calls, so a line said in one call never
   enters the cache.
   See [Privacy](./privacy.md).
3. **One lookup per sentence.** An in-process index answers misses with no network
   call; hits come from memory or one Redis `MGET`.
4. **Exact order on every provider.** On HTTP services cached and live audio
   interleave inside one generator. On WebSocket services Memcat keeps Pipecat's
   audio context and gives each run of live clauses its own provider stream,
   releasing segments strictly in order, so cached clauses play before, between and
   after live ones. Transcripts and word timestamps stay correct.
5. **Joins matched to live speech.** Clips are stored tight and Memcat sets the pause
   at each join the way live speech does (short after "Sure,", longer after a full
   clause), matches loudness, and fades edges.
6. **Fill in the background.** A clause admitted from a live run that synthesized
   it alone is stored as is, for free; others are rendered once by a background
   copy of the same service. Nothing on the call ever waits for the cache.

If the store, a codec or Memcat itself fails, the sentence goes to the provider as
if Memcat were not installed: Memcat can lose savings, but the call goes on.

## The path of one sentence

```text
run_tts(sentence) ──► split into clauses ──► key each clause ──► L1 / key index / one MGET
                                                                     │
          ┌──────────────────────── runs of hits and misses ─────────┘
          ▼
   hits: replay 40 ms frames + word timestamps         misses: provider (live)
          │                                                    │
          └──────────────► Pipecat's audio context, strictly in order ◄──┘
                                       │
        background worker: admission (k calls) → capture or render → encode → store
```

Opus clips from the store are decoded in a worker thread, off the event loop, so a
hit never stalls the pipeline.

## Ordering on WebSocket providers

Pipecat plays one audio context per turn. Memcat keeps that context and treats its
contents as a queue of segments: a run of cached clips, or a run of live clauses
sent to the provider as a separate stream (a provider-side context with its own id).
Only the segment at the head of the queue writes into Pipecat's context; later ones
wait in memory. A live stream ends with the provider's flush, and its completion
message releases the next segment. Word timestamps are moved onto the context's
timeline as each segment reaches the head.

Pipecat ends an audio context after `stop_frame_timeout_s` (3 s by default) without
new audio. While cached clauses wait behind a slow live stream, Memcat keeps the
context alive, so they still play after it, and applies that timeout to the stream
itself instead.

With no cached clauses the provider sees one stream per turn, the same traffic
plain Pipecat sends.

## Joins

A provider synthesizing a clause alone pads it with silence it would not put
mid-sentence, and ends a short opener with a long pause and a long clause with a
short one: the opposite of live speech. Memcat stores clips with 30 ms of silence
on each side and, at each join with a cached clip, trims the live audio's edge
silence and inserts the pause live speech has there (about 100 ms after "Sure,",
320 ms after a full clause). Measured on Kokoro over 30 sentences
(`bench/abx.py`), the pause at a join is within 45 ms of the fully live take at the
median (121 ms at the 90th percentile), click energy at the join is the same as live
at the median (0.99x, at most 1.54x), and loudness is within 0.62 LU. Real
ElevenLabs audio with cached and live clauses mixed transcribes to the same words
as the sentence sent (`bench/elevenlabs_replay.py`).

## Filling the cache

Every missed cacheable clause is counted for admission in the background. When a
live run (or a provider stream) carried exactly one clause, its audio is kept beside
that count and stored the moment the clause is admitted, at no extra cost. Other
admitted clauses are rendered once by a render twin: a second instance of the same
provider class, built from the same constructor arguments and settings, running in
its own pipeline, and shared by every call in the process built with equal
arguments. It skips `text_transforms` and `text_filters`, which the live service
already applied to the clause text.

When several processes share a store, a render lock (held for up to 60 s) lets only
one of them render a given clause. A render that errors, times out or ends before
the provider finishes the clause stores nothing, and the clause is not rendered
again for an hour.

Set `render=False` to store only clauses captured from live audio. Phrases in
`warm` (and in `memcat warm`) are split into clauses like any sentence; each
cacheable clause that is not already stored skips admission and is rendered when the
service starts. Clauses the privacy rules block are skipped.

Every new entry is announced to the other processes on the same Redis store, so
their key indexes learn about it without a scan.

## Failure handling

| Failure | What happens |
| --- | --- |
| Store slow or down | Miss. The cache read on the call path times out after 50 ms on Redis (`RedisStore(op_timeout_ms=...)`; background writes get 2 s) and after 100 ms on any other store. When at least 5 store calls fail within 10 s and they are at least half of the calls in that window, only L1 is used for 30 s; then a single probe call decides whether the store is used again (a probe with no answer after 15 s is replaced by the next request). |
| Corrupt or unknown entry | Miss; the entry is deleted. |
| Opus entry in a process without PyAV | Miss (the entry stays for processes that can decode it). |
| Provider error | Propagates exactly as without Memcat. |
| Error in Memcat | The rest of the sentence is spoken live, Memcat turns itself off for that service, and one warning is logged with the error's type (details at DEBUG). |
| Provider never finishes a stream | Closed once it has been quiet for `stop_frame_timeout_s` (3 s by default) after it was asked to end, or after 30 s if it never gets going; one warning is logged and its audio is never cached. The clauses queued behind it still play. |
| Worker queue full, or holding more than `queue_max_bytes` of audio | Job dropped and counted (`dropped_jobs`, `dropped_bytes`). |
