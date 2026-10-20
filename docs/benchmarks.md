---
title: Benchmarks
description: How every measured number was produced - hit rate, latency, recorded ElevenLabs traffic, hot path, storage, listening and load.
sidebar:
  order: 9
---

# Benchmarks

All results below were measured on 20 Oct 2026 on an Apple M-series laptop with
Python 3.13, Pipecat 1.13.0, Kokoro (ONNX, CPU) and a local Redis 8. Raw outputs are
in [`bench/results/`](https://github.com/imumesh18/memcat/tree/main/bench/results).

[`bench/`](https://github.com/imumesh18/memcat/tree/main/bench) reproduces every number: `hitrate.py` (ABCD, Harper
Valley Bank, τ²-bench), `elevenlabs_replay.py` (recorded ElevenLabs traffic),
`latency.py` (Kokoro + Redis), `storage.py`, `hotpath.py` (run nightly, which fails
at 3x its baseline), `soak.py` (100 concurrent calls) and `abx.py`, which builds a
30-sentence listening test and measures each join against live speech. The "Redis
killed mid-call" result is
[`tests/test_store_outage.py`](https://github.com/imumesh18/memcat/blob/main/tests/test_store_outage.py).

## Hit rate (`bench/hitrate.py`)

| Corpus | Calls | Characters from cache | Net of fill cost | Responses fully cached | Blocked as personal data |
| --- | --- | --- | --- | --- | --- |
| Harper Valley Bank: spoken, scripted bank calls (CC-BY-4.0) | 1,446 | 54.0% | 54.0% | 59.0% | 15.1% |
| ABCD: typed support chats (MIT), clause level | 10,042 | 32.4% | 31.9% | 38.5% | 5.7% |
| ABCD, sentence level only | 10,042 | 25.9% | 25.9% | 34.8% | 6.8% |
| τ²-bench telecom: GPT-4.1 policy agent (MIT) | 114 | 16.8% | 14.9% | 9.1% | 13.8% |
| τ²-bench retail | 114 | 12.3% | 11.4% | 17.8% | 26.9% |
| τ²-bench airline | 50 | 5.9% | 5.2% | 14.2% | 33.2% |

"Net of fill cost" subtracts the one extra synthesis each cached clause costs when
it is rendered; a clause first spoken live on its own is captured from that audio
for free, which is most of them (3,340 of 4,235 entries on ABCD, all 311 on Harper
Valley Bank).

Admission is k = 3 distinct calls. Clause-level splitting adds 6.5 points over
whole sentences on ABCD. The τ² corpora are small (50–114 calls), so admission
barely warms up, and their agents put an id or a price in most sentences.

## Latency (`bench/latency.py`, Kokoro + local Redis, 30 utterances)

| Case | First audio p50 | p99 |
| --- | --- | --- |
| No Memcat | 745 ms | 894 ms |
| Hit from L1 | 1.0 ms | 1.2 ms |
| Hit, first fetch from Redis | 10 ms | 12 ms |
| Cached opening clause, then live | 1.4 ms | 1.8 ms |
| Miss, nothing cacheable | 488 ms | 561 ms |
| Same sentences without Memcat | 495 ms | 560 ms |

## Recorded ElevenLabs traffic (`bench/elevenlabs_replay.py`)

Recordings of the ElevenLabs API (`eleven_flash_v2_5`, stock voice "Sarah",
16 kHz PCM), 9 scenarios recorded 3 times each,
replayed through Pipecat's `ElevenLabsTTSService` and `ElevenLabsHttpTTSService`
with every provider message sent at its recorded delay. What the listener received
was transcribed with a local speech recognizer (faster-whisper `small.en`). The
recordings are not in the repository; to reproduce, record your own with an
ElevenLabs key (`MEMCAT_RECORD=1 uv run pytest tests/test_replay_elevenlabs.py`,
about 2,000 characters, within the free plan) and run the benchmark on them.

| Measure | Result |
| --- | --- |
| ElevenLabs time to first audio per stream (93 streams) | 227 ms median, 292 ms p90, 843 ms worst |
| First audio, plain Pipecat (6 runs, WebSocket and HTTP) | 346 ms median |
| First audio, sentence opening with a cached clause (9 runs) | 2.4 ms median over HTTP, 13 ms over WebSocket, 16 ms worst |
| Word error, cached and live clauses mixed (12 runs) | 0.3% (one "is" for "has" in a live clause) |
| Word error, plain Pipecat (6 runs) | 0.0% |
| Replay mismatches (client messages differing from the recording) | 0 |

Digits and spelled codes are left out of the word comparison, since recognizers
write them in many forms; those clauses are always live. One further recording
caught ElevenLabs sending nothing for 3.3 s after a barge-in; Memcat's behaviour
there is covered by `test_elevenlabs_stall_after_barge_in`.

## Hot path (`bench/hotpath.py`, nightly check fails at 3x baseline)

Split a 30-word sentence (with every privacy rule) 36 µs · build and hash a key
10 µs · L1 get 0.08 µs · key index check 0.10 µs · decode 3 s of mu-law 11 µs ·
decode 3 s of Opus 10 ms · whole miss sentence (split, key, look up) 63 µs.

## Storage (`bench/storage.py`, Kokoro speech)

mu-law at 8 kHz 464 KB/min · Opus at 16 kHz 232 KB/min · Opus at 24 kHz 236 KB/min ·
PCM16 at 24 kHz 2,783 KB/min.

## Listening (`bench/abx.py`)

Builds 30 ABX trials (fully live vs cached and live mixed, 60% of characters from
cache) as WAV files with an answer key, for human listeners. Objective checks on the
same set: pause at joins within 45 ms of live at the median (121 ms at p90), click
energy 0.99x live, loudness within 0.62 LU.

## Load and soak (`bench/soak.py`)

100 concurrent calls for 5 minutes in one process, each with its own wrapped
service and `RedisStore`, through fake providers with 50–250 ms latency (no
real-time pacing, so the process runs flat out):

| Measure | With Memcat | Memcat switched off (`--plain`) |
| --- | --- | --- |
| Calls completed | 18,293 | 20,822 |
| Calls heard wrong (any word missing or out of order) | 0 | 0 |
| Characters served from cache | 50.7% | 0% |
| Memory after warm-up | 232.3 MB → 233.1 MB over 4 minutes | 193.6 MB → 195.8 MB |
| asyncio tasks after warm-up | 1,523 → 1,369 (no growth) | 1,517 → 1,491 |
| Redis connections | 4 after warm-up | 0 |
| Store errors | 60 (all cache lookups past their 50 ms budget while the event loop was saturated; those clauses were spoken live) | 0 |
| Dropped background jobs | 0 | 0 |

With the process running flat out, Memcat costs about 2 ms of event-loop time per
call (12% fewer calls in the same 5 minutes) and about 40 MB of memory for the
cache and index. The fake providers do no work, so
this is the most Memcat's overhead can be relative to a call; with real providers
the time goes to waiting on the network. Under this load the event loop runs 33 ms
behind at the median (15 ms without Memcat), so a few lookups miss the 50 ms
`op_timeout_ms` budget: Memcat then speaks the clause live rather than delay first
audio.
