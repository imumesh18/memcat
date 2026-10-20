---
title: Privacy
description: The text rules, admission and erasure that keep callers' details out of the cache.
sidebar:
  order: 5
---

# Privacy

Memcat caches what your agent says, never what callers say. Three layers keep a
caller's details out of the cache: text rules, admission and erasure.

## Text rules

Every rule reads the text after Unicode NFKC normalisation, so "①②", "⁴⁵" and "＄"
count as the digits and symbol they stand for, and invisible formatting characters
(zero-width spaces and joiners, soft hyphens) count as spaces, so "nine\u200b eight"
is still two number words. A clause is never cached if it contains:

- **A number written as a number.** Any digit in any script, any other Unicode
  numeric character ("Ⅷ", "½") or any currency symbol (Unicode category Sc: `$`,
  `€`, `₹`...). Two or more numerals written as letters in a row ("三四", "七百")
  also block.
- **An address.** An `@`, `://`, `www.`, or any written `x.yz` pattern (a letter or
  digit, a dot, two letters: "acme.ca", "shop.example.xyz"). Emails and domains
  spoken aloud block too: "dot" before a common top-level domain ("example dot
  com"), "dot" shortly after "at" ("john (at) gmail (dot) xyz"), "at" before a
  common mail provider ("priya at gmail") and "at the rate".
- **A number in words.** Two number words in a row, whatever separates them
  ("forty two", "nine; eight"), or joined by "and", "then", "point", "dash",
  "hyphen", "slash" or "dot" ("two hundred and fifty", "four, then two"). One
  number word next to a month ("May five", "five of June"), next to a money word
  ("twenty dollars", "fifty grand", "five rupees"), or followed by "am", "pm" or
  "o'clock" ("nine pm"). An ordinal next to a month ("March fifth", "the fifth of
  June"), and a tens word before an ordinal ("twenty-first"). A single number word
  on its own stays cacheable: "one moment", "a second" and "first," are among the
  most repeated lines there are.
- **A spelled-out code.** Three or more single letters or number words in a row
  ("Q X R", "A nine B", "B, then K, then L"), unless they are all "a", "I" or "o";
  two NATO alphabet words in a row ("Bravo Kilo", "Bravo, then Kilo"); a letter
  followed by "as in" or by "for" and a NATO word ("B as in Bravo", "B for Bravo");
  an all-caps word of 5 or more letters ("QXRTZ"), or of 4 letters with no vowel
  ("BXTR"). Short acronyms such as "OK", "ID", "PIN", "SMS" and "FAQ", and
  four-letter ones with a vowel such as "ASAP", stay cacheable.
- **A match for your own `block_patterns`** (case-insensitive regular expressions).

Number words are recognised in English and in Hindi, Devanagari and romanised: every
number from one to a hundred, "sau", "hazaar", "lakh", "crore", the joiners "aur"
and "phir", Hindi money words such as "rupaye", and month names in Devanagari. In
other languages, digits, numerals and currency symbols still block, but numbers
spelled out in words are not recognised: add `block_patterns` for them, and rely on
admission.

A clause with personal data never takes its neighbours down with it: in "12345, got
that." the tail is still cacheable. Numbers and codes are the exception, because
they can be said in pieces. A rule that matches across a clause boundary blocks
every clause it spans: in "Your code is Q, X, R, thanks for waiting." only "thanks
for waiting." is cacheable. A clause holding a number or NATO word next to a clause
blocked for a number is blocked too, in a chain: in "Call nine eight, seven six five
four, three two." every clause is spoken live.

## Admission

A clause is cached only after it has been heard in `admit_after_calls` (default 3)
distinct calls. This is the layer that covers what no rule can recognise, such as a
caller's name or street.

Admission counts calls, not callers: Memcat cannot tell who is on the line, so three
calls from the same person count as three. A bot that calls the same person again
(reminders, collections, retries) or a caller who rings back can get a line such as
"Am I speaking with Priya Sharma?" cached. If your agent says names, addresses or
other personal details in fixed templates, add `block_patterns` for those templates
(`r"\bspeaking with\b"`, `r"\bship(ped)? to\b"`).

On Redis, admission counts are kept for every cacheable clause heard in at least one
call, keyed by the clause digest, and expire 30 days after the clause was last
heard (on S3 and in a `MemoryStore` they stay in process memory). A digest is a hash
of the clause text with no secret in it, so someone who can read the store and
guesses a text can check whether it was said. Treat the store as holding your
agent's speech and restrict access to it.

## Erasure

`memcat purge --text "..."` (or `await tts.memcat_purge(text)`) deletes the text and
each of its clauses in every voice, setting and sample rate, together with their
admission counts. Every cache key starts with a 4-byte tag of its clause text, so
Memcat finds every variant by that prefix: on Redis with a `SCAN` of the entry and
admission keys, on S3 with one prefix listing, in a `MemoryStore` by filtering. A
different text that happens to share the tag loses its audio too, which costs one
re-render. A clause is stored with the punctuation that followed it ("Hi John,"),
so a purge also covers the text with each common ending (`,` `.` `?` `!` `;` `:`
`—` `…`). `memcat purge --text` reads only `[store]` and `[memcat]` from
`memcat.toml`, so it needs no provider package or API key.

If the store deletes the entries but other processes cannot be told (Redis
unreachable for the notification, after three tries), the purge raises
`PurgeNotifyError` and `memcat purge` exits with status 1, so an erasure is never
reported done while another process may still play the audio. Purging again is
safe.

The process that purges drops the audio from its own memory at once. When other
running processes stop serving it depends on the store:

- **Redis.** The purge is announced on the store's channel, and every subscribed
  process drops the audio from its memory and its key index at once. A process
  whose subscription was down rescans the store when it reconnects and drops what
  is gone.
- **S3.** The purge is written as a small event object under
  `{prefix}/{namespace}/_events/`. Every other process lists that prefix every
  `events_poll_s` seconds (default 10; one LIST request per poll), then drops the
  audio from its memory and key index and forgets its admission counts. A purge
  therefore takes effect everywhere within about `events_poll_s` seconds; host
  clocks may differ by up to 60 s. Later purges delete event objects older than a
  day; a bucket lifecycle rule expiring `{prefix}/*/_events/` after one day keeps
  that prefix small.

`await tts.memcat_purge()` with no text deletes the whole namespace, like
`memcat purge --all`.

## Checking your own transcripts

To see which clauses of your own transcripts would be cached, run
`memcat warm --from calls.jsonl --dry-run` (see [Operations](./operations.md#commands)):
it lists the most repeated cacheable clauses without rendering anything.

## What Memcat itself keeps

Memcat never logs clause text or cache digests at INFO or above, never uses
`pickle` or `eval` on stored data (entries are parsed with `struct` and every length is checked), and
sends no telemetry.

Measured: on ABCD, Harper Valley Bank and τ²-bench (11,766 conversations), zero
cache keys hold a digit; a Hypothesis test over generated numbers, codes, emails,
URLs, money and dates finds none either.
