---
title: Providers
description: Which Pipecat TTS services Memcat supports, where cached clauses can play in a sentence, and the pricing table behind usd_saved.
sidebar:
  order: 4
---

# Providers

Memcat needs to know how a provider delivers audio to replay cached clauses in the
right place. Each certified provider has an adapter, tested with its Pipecat service
class against a local emulator of that provider's wire protocol
([`tests/emulators/`](https://github.com/imumesh18/memcat/tree/main/tests/emulators)).

ElevenLabs is also tested against recordings of its API, replayed with their
original timing
([`tests/test_replay_elevenlabs.py`](https://github.com/imumesh18/memcat/blob/main/tests/test_replay_elevenlabs.py));
the results are in [Benchmarks](./benchmarks.md). The recordings are not in the
repository: [CONTRIBUTING.md](https://github.com/imumesh18/memcat/blob/main/CONTRIBUTING.md#recorded-provider-traffic)
explains how to make your own. The other providers have not yet been tested against
their live APIs.

## Summary

| Provider | WebSocket | HTTP |
| --- | --- | --- |
| Cartesia | cached clauses anywhere in a sentence | cached clauses anywhere |
| ElevenLabs | cached clauses anywhere | cached clauses anywhere, `previous_text` kept |
| Deepgram Aura | cached clauses anywhere, within Deepgram's 20-flush/minute limit | cached clauses anywhere |
| Sarvam | cached clauses before the first live one | cached clauses anywhere |
| Any other Pipecat TTS service | cached clauses before the first live one (if it pushes its own start frames) | cached clauses anywhere (if it returns audio inline) |

## Adapters

The adapter is picked from the module of the first class in the service's MRO that
Pipecat ships, so subclasses of a certified service get its adapter. A service can
name one explicitly with a class attribute `__memcat_adapter__` set to a
`ProviderAdapter` subclass (the class, not an instance); that wins over module
matching (see [Configuration](./configuration.md#services-that-need-their-own-adapter)).
`await tts.memcat_stats()` reports the chosen `adapter` and `mode`.

| Provider | Service | Mode | Notes |
| --- | --- | --- | --- |
| Cartesia | `CartesiaTTSService` | segmented | One provider context per live run, ended with `continue: false`; up to 4 in flight. |
| Cartesia | `CartesiaHttpTTSService` | http | |
| ElevenLabs | `ElevenLabsTTSService` | segmented | The service opens each stream itself, so voice settings are always sent; one stream at a time (shared word clock); ends with flush + `close_context`. |
| ElevenLabs | `ElevenLabsHttpTTSService` | http | `previous_text` includes cached clauses, in order. |
| Deepgram | `DeepgramTTSService` | segmented | No context ids on the wire: one stream in flight, attributed by Memcat. Extra flushes stay within Deepgram's [limit of 20 per 60 s](https://developers.deepgram.com/docs/tts-ws-flush); none in a call's first 30 s. Lossless storage only. |
| Deepgram | `DeepgramHttpTTSService` | http | |
| Sarvam | `SarvamTTSService` | leading | Cached clauses before the first live one in a context. |
| Sarvam | `SarvamHttpTTSService` | http | |
| Fish Audio | `FishAudioTTSService` | leading | Not certified: no emulator test yet. |
| other | WebSocket with `push_start_frame=True` | leading | |
| other | WebSocket otherwise | passthrough | |
| other | HTTP / local (e.g. Kokoro) | http | The first sentence checks the service returns audio inline; if not, Memcat turns itself off for it. |

**segmented**: cached clauses play anywhere in a sentence, before, between or after
live ones. **leading**: cached clauses play until the first live clause of a
context; the rest goes live. **http**: cached and live clauses interleave freely.
**passthrough**: Memcat is off for the service and logs one warning; every
sentence goes to the provider unchanged.

Memcat also turns itself off, with one warning, for a service that uses Pipecat's
`TOKEN` text aggregation mode, which Memcat does not support.

## Pricing table

`usd_saved` uses USD per 1M characters at pay-as-you-go or startup tiers, read on
18 Oct 2026 from each provider's pricing page: ElevenLabs Multilingual/v3/v4 80,
Flash/Turbo/v3 Conversational 40, any other ElevenLabs model 40; Cartesia Sonic 39;
Sarvam Bulbul v3 31.6; Deepgram Aura-2 30. Prices are keyed by adapter and model-name
prefix, and the longest matching prefix wins. Override with
`MemcatConfig(prices={("cartesia", ""): 30.0})`.

## Terms

As of October 2026, none of the certified providers' terms that we read forbid
caching and replaying your own output. The limits we found are on sharing audio
across customers, on free-tier audio (for example, [ElevenLabs free-plan audio](https://elevenlabs.io/docs/help-center/legal/can-i-publish-the-content-i-generate-on-the-platform)
is non-commercial and needs attribution) and on benchmarking. Keep one cache or
`namespace` per customer, and check your own plan's terms. This is not legal advice.
