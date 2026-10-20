# Changelog

## 1.0.0

First release. Supports Pipecat 1.10 through 1.13 and Python 3.11 to 3.13.

- `cached(cls)` wraps any Pipecat `TTSService`; `MemcatConfig`; `MemoryStore`,
  `RedisStore` (tested on Redis 7 and 8 and Valkey 8), `S3Store`.
- Clause-level caching with admission after k distinct calls and privacy rules run
  on NFKC text: any digit or numeral in any script and any currency symbol; written
  and spoken emails, URLs and domains; numbers in words in English and Hindi (two in
  a row or joined by "then"/"aur", or one next to a month, money word or am/pm),
  including numbers split across clauses; spelled-out codes, NATO spellings and
  long all-caps codes.
- Exact playback order on HTTP and WebSocket providers, including cached clauses
  after live ones on Cartesia, ElevenLabs and Deepgram; leading-clause replay on
  Sarvam and other WebSocket services.
- Correct transcripts and word timestamps; one TTSStartedFrame/TTSStoppedFrame per
  context; Pipecat usage metrics count billed characters only.
- Natural joins: tight clips, per-join pauses, loudness matching, edge fades.
- Free capture of single-clause live runs; background render twins for the rest.
- Codecs: PCM16, mu-law, Opus (PyAV, decoded at 48 kHz then resampled with soxr).
- Shared per-process runtime: key index, L1, worker and connection pool per store
  and namespace.
- Circuit breaker, stream watchdog, Deepgram flush budget; every failure falls back
  to the provider.
- Erasure everywhere: `memcat purge --text` and `memcat_purge(text)` delete a text
  and each of its clauses in every voice, setting and sample rate, with admission
  counts, in the store and in every running process (over Redis pub/sub, or within
  `events_poll_s` through S3 event objects). A purge that cannot notify other
  processes raises `PurgeNotifyError` instead of reporting success.
- CLI: `memcat warm | stats | purge | export | import | serve`. Cache commands need a
  Redis or S3 store; `purge --all` asks for confirmation; `import` checks the
  export's namespace.
- `memcat serve`: TTS endpoint for hosted platforms (Vapi custom voice), streamed
  PCM. Listens on 127.0.0.1 by default and needs a secret on any other address;
  bounded text length, queue and wait (413, 503 with `Retry-After`); requests
  without a call id count as one call toward admission.
- Bloom-filter key index past 1M keys: a fixed 12 MB filter sized for 10M keys at a
  1% false-positive rate (a set of 1M keys takes about 91 MB).
- OpenTelemetry attributes on Pipecat's TTS spans; `memcat_stats()`.
- Benchmarks in `bench/`: hit rate, latency, storage, hot path, load, recorded
  ElevenLabs traffic and a listening-test kit.
- Every certified provider tested against a local emulator of its wire protocol;
  ElevenLabs also against recordings of its API (`tests/replay/`; record your own,
  see CONTRIBUTING.md).
