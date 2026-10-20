# Memcat

**A text-to-speech cache for [Pipecat](https://github.com/pipecat-ai/pipecat) voice bots.**
When your agent says a line it has said before (or part of one), Memcat replays the
stored audio instead of paying the provider to synthesize it again, and the audio
starts in milliseconds.

| | Measured |
| --- | --- |
| Speech served from cache, scripted support calls ([Harper Valley Bank](https://github.com/cricketclub/gridspace-stanford-harper-valley), 1,446 calls) | **54.0% of characters, 59.0% of responses** |
| Same, typed support chats ([ABCD](https://arxiv.org/abs/2104.00783), 10,042 chats) | 32.4% of characters, 38.5% of responses |
| First audio of a sentence that opens with a cached clause, recorded ElevenLabs traffic | **2–13 ms** with Memcat vs 346 ms without (median, HTTP and WebSocket) |
| What a listener hears, recorded ElevenLabs traffic, cached and live clauses mixed | the same words in the same order: 0.3% word error vs 0.0% without Memcat, by speech recognition |
| First audio on a hit, local Kokoro voice | 1.0 ms from memory, 10 ms from Redis (vs 745 ms live) |
| Added time on a miss | none measurable: 488 ms vs 495 ms without Memcat (median, within run-to-run noise) |
| Storage per cached minute | 464 KB at 8 kHz (mu-law), 236 KB at 24 kHz (Opus) |
| Cache keys containing a digit (order numbers, prices, dates, phone numbers) | **0** on all corpora |
| Redis killed mid-call | every turn spoken, no error reaches the pipeline |

Hit rates come from replaying public dialogue corpora through Memcat's own chunker,
privacy rules and admission policy (`bench/hitrate.py`); no TTS was called. Net of
what filling the cache costs, savings are within 2 points of these (54.0% and 31.9%).
LLM agents whose every sentence carries an order number or a price cache less: 6–17%
of characters on [τ²-bench](https://github.com/sierra-research/tau2-bench) traces,
where those clauses stay live. The ElevenLabs rows replay traffic recorded from the
ElevenLabs API (Flash v2.5) through Pipecat's ElevenLabs services
(`bench/elevenlabs_replay.py`). The recordings are not in the repository; anyone with
an ElevenLabs key can make their own and rerun it.

## Install

```bash
uv add memcat                      # or: pip install memcat
uv add "memcat[redis,opus]"        # shared cache across bot processes, compact audio
```

Python 3.11–3.13 and `pipecat-ai` 1.10–1.13. Extras: `redis` (`RedisStore`), `opus`
(Opus storage), `s3` (`S3Store`), `all`.

## Use

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

Use `tts` exactly as you used the unwrapped service. Nothing else in the pipeline
changes. `await tts.memcat_stats()` reports hits, characters saved and dollars saved.

No API keys at hand? From a clone, `uv run examples/kokoro_local.py` speaks three lines twice
with a local voice; the second time they come from the cache.

## How it works

1. **Clauses, not just sentences.** Each sentence Pipecat hands to the TTS is split
   at commas, semicolons, colons and dashes. "Thanks for waiting, your order 48213
   has shipped, and it should arrive soon." is three clauses; the first and last are
   reusable, the middle one is spoken live.
2. **Privacy first.** A clause with any digit, a number in words, an email, a URL,
   money, a date, a time or a spelled-out code is never cached. A clause is cached
   only after it has been heard in 3 distinct calls, so a line said in one call never
   enters the cache.
3. **One lookup per sentence.** An in-process index answers misses with no network
   call; hits come from memory or one Redis `MGET`.
4. **Exact order on every provider.** On HTTP services cached and live audio
   interleave inside one generator. On WebSocket services Memcat keeps Pipecat's
   audio context and gives each run of live clauses its own provider stream,
   releasing segments strictly in order, so cached clauses play before, between and
   after live ones. Transcripts and word timestamps stay correct.
5. **Joins matched to live speech.** Clips are stored tight and Memcat sets the pause
   at each join the way live speech does (short after "Sure,", longer after a full
   clause), matches loudness, and fades edges. Against live synthesis of the same
   sentences, pauses differ by 45 ms at the median and loudness by at most 0.62 LU
   (`bench/abx.py`).
6. **Fill in the background.** A clause admitted from a live run that synthesized
   it alone is stored as is, for free; others are rendered once by a background
   copy of the same service. Nothing on the call ever waits for the cache.

If the store, a codec or Memcat itself fails, the sentence goes to the provider as
if Memcat were not installed: Memcat can lose savings, but the call goes on.

## Providers

| Provider | WebSocket | HTTP |
| --- | --- | --- |
| Cartesia | cached clauses anywhere in a sentence | cached clauses anywhere |
| ElevenLabs | cached clauses anywhere | cached clauses anywhere, `previous_text` kept |
| Deepgram Aura | cached clauses anywhere, within Deepgram's 20-flush/minute limit | cached clauses anywhere |
| Sarvam | cached clauses before the first live one | cached clauses anywhere |
| Any other Pipecat TTS service | cached clauses before the first live one (if it pushes its own start frames) | cached clauses anywhere (if it returns audio inline) |

Every listed provider is tested with its Pipecat service class against a local
emulator of the provider's wire protocol (`tests/emulators/`). ElevenLabs is also
tested against recordings of its API, replayed with their original timing
(`tests/test_replay_elevenlabs.py`). To record your own, see
[CONTRIBUTING.md](https://github.com/imumesh18/memcat/blob/main/CONTRIBUTING.md).

## Configure

```python
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

All fields are described in [docs/configuration.md](https://github.com/imumesh18/memcat/blob/main/docs/configuration.md).

## Operate

```bash
memcat warm  --config memcat.toml --from calls.jsonl --top 500   # render your most repeated clauses
memcat stats --config memcat.toml                                # entries, minutes, codecs
memcat purge --config memcat.toml --text "..."                   # erase a text everywhere
memcat export --config memcat.toml --to backup/ ; memcat import --config memcat.toml --from backup/
memcat serve --config memcat.toml --port 8080                    # TTS endpoint for Vapi and other platforms
```

`memcat serve` puts the cache in front of hosted platforms that call a custom TTS
URL (Vapi's custom voice): it answers with raw 16-bit PCM, streamed, and counts
admission per platform call. It listens on 127.0.0.1 unless you set a secret (or
pass `--insecure-no-auth` behind your own authentication).

Every command except `serve` and `warm --dry-run` needs a Redis or S3 `[store]`.
`purge --text` erases the text and each of its clauses in every voice, setting and
sample rate; `purge --all` asks for confirmation (or `--yes`).

See [docs/operations.md](https://github.com/imumesh18/memcat/blob/main/docs/operations.md) for `memcat.toml`, metrics and OpenTelemetry.

## Benchmarks

`bench/` reproduces every measured number above: `hitrate.py` (ABCD, Harper Valley
Bank, τ²-bench), `elevenlabs_replay.py` (recorded ElevenLabs traffic), `latency.py`
(Kokoro + Redis), `storage.py`, `hotpath.py` (run nightly, which fails at 3x
its baseline), `soak.py` (100 concurrent calls) and `abx.py`, which builds a
30-sentence listening test and measures each join against live speech. The Redis
row is `tests/test_store_outage.py`. Results live in `bench/results/`.

## License

BSD 2-Clause, the same as Pipecat. Keep one cache (or one `namespace`) per
customer so audio is never shared across customers, and check your provider's
terms for anything else.
