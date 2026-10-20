# Contributing

```bash
uv sync
uv run pytest -q                       # ~3 minutes; no network or API keys
MEMCAT_REDIS_URLS=redis://localhost:6379/0 uv run pytest -q   # include Redis/Valkey
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

`MEMCAT_TEST_LOG=DEBUG` shows Memcat's and Pipecat's logs during tests (default WARNING).
`tests/test_chaos.py` and `tests/test_store_outage.py` start their own Redis and skip
when `redis-server` is not on `PATH`. To test another supported Pipecat version:
`uv pip install "pipecat-ai[kokoro]==1.10.0" && uv run --no-sync pytest -q`.

- Every touch of Pipecat internals goes through `src/memcat/pipecat/compat.py`.
- A new provider adapter needs an emulator in `tests/emulators/` and a
  `tests/test_provider_<name>.py` that drives the Pipecat service class.
- Never weaken a rule in `docs/privacy.md` to make a test pass.

## Docs

`docs/` is the documentation site's source: the docs hub fetches that folder, and
nothing outside it, at a release tag. Each page starts with front matter holding
`title`, a one-line `description` and `sidebar.order` (its place in the sidebar;
`index.md` always comes first), followed by an `# h1` equal to the title. Link to
other pages with relative paths (`./configuration.md`), and to anything outside
`docs/` with a full GitHub URL (`https://github.com/imumesh18/memcat/blob/main/...`).
Keep pages as `.md`. The hub does not fetch images, so an image needs a full URL.

`tests/test_docs.py` checks this in every test run, CI included
(`uv run pytest -q tests/test_docs.py` runs it alone). Outside code blocks and inline
code, it fails when:

- `docs/` holds anything but `.md` files directly in it (no subfolders, no images);
- a page has no front matter, or it lacks `title`, `description` or a whole-number
  `sidebar.order`; two pages share an order; or `index.md` does not have the lowest;
- a page does not have exactly one `# h1`, or its h1 differs from `title`;
- a relative link leaves `docs/` (`../`), uses a site path (`/...`), or points to a
  page that does not exist, or to a heading anchor (`./page.md#anchor` or `#anchor`)
  that the page does not have;
- a `https://github.com/imumesh18/memcat/blob/main/...` or `.../tree/main/...` link
  names a path that does not exist in the repository.

## Recorded provider traffic

`tests/test_replay_elevenlabs.py` runs Pipecat's ElevenLabs services
against recordings of the ElevenLabs API. Each recording holds every
WebSocket message and HTTP chunk with the time it arrived, and replay sends each
one back at its recorded delay after the client message that caused it. Every
client message must match the recording, with Pipecat's context ids remapped,
so any change in what Memcat or Pipecat sends fails the test.

The recordings contain provider audio, so they are not in the repository
(`tests/fixtures/` is gitignored) and a fresh checkout skips these tests. To run
them, record your own once. A free ElevenLabs account is enough: recording uses
about 2,000 of its 10,000 monthly characters. Keep the recordings on your machine:
free-plan audio is licensed for non-commercial use only.

```bash
cp .env.test.example .env.test         # gitignored; add ELEVENLABS_API_KEY
MEMCAT_RECORD=1 uv run pytest tests/test_replay_elevenlabs.py   # record, ~2,000 characters
uv run pytest tests/test_replay_elevenlabs.py                  # replay, no key, no API calls
uv run --group asr python bench/elevenlabs_replay.py           # transcribe and time it
```

Recordings go to `tests/fixtures/elevenlabs/`, or to
`$MEMCAT_FIXTURES_DIR/elevenlabs/` when that is set. Re-record after a deliberate
change to what Memcat sends, or after a Pipecat upgrade, when replay reports a
mismatch.

The key stays inside the recording proxy (`tests/replay/recorder.py`): Pipecat
gets a dummy key, no headers are stored, and saving fails if the key appears
anywhere in a recording. `tests/test_replay_machinery.py` proves the recorder
and player against the local emulator and runs everywhere, CI included.
`MEMCAT_REPLAY_SPEED=0` replays without delays; `2` replays a provider twice as
slow.

## Releasing

1. Set `version` in `pyproject.toml` and add its section to `CHANGELOG.md`.
2. Merge to `main`, then tag: `git tag v1.0.1 && git push origin v1.0.1`.
3. The release workflow runs CI, builds, publishes to PyPI (trusted publishing,
   environment `pypi`) and creates the GitHub release.
