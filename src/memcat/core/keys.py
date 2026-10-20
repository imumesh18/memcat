"""Cache keys: canonical JSON of what shapes the audio, hashed to 16 bytes.

A key covers the service class, every setting that can change the audio, the
output sample rate and the normalised clause text. It never includes the codec
or how the entry was produced, so one clause has exactly one key.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

__all__ = [
    "KEY_SCHEMA",
    "KeyContext",
    "Keyer",
    "SettingsNotSerializableError",
    "canonical_settings",
    "canonical_value",
    "normalize_text",
    "text_tag",
]

KEY_SCHEMA = "s1"

# Secrets, per-user and per-request ids, timeouts and retries stay out of the key, at
# any nesting depth. Secrets and ids are matched by exact (lowercased) name: a
# substring would also drop settings that change the audio (Neuphonic ``ipa_key``,
# ``max_tokens``). ``model``, ``voice`` and ``language`` at the top level are always kept.
_EXCLUDED_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "api_secret",
        "apikey",
        "auth",
        "auth_token",
        "authorization",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "client_reference_id",
        "credentials",
        "password",
        "request_id",
        "safety_identifier",
        "secret",
        "secret_key",
        "session_id",
        "token",
        "user_id",
    }
)
_EXCLUDED_NAME_PARTS = ("timeout", "retries")
_PRIMARY = {"model": "m", "voice": "v", "language": "l"}

_QUOTES = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "«": '"',
        "»": '"',
    }
)
_WS = re.compile(r"\s+")


class SettingsNotSerializableError(ValueError):
    """A setting value has no stable canonical form, so no safe key exists."""


# Words whose capital at the start of a clause never changes how they are said.
# Folding them merges "Sure, ..." with "sure, ..." and "How can I help?" with the
# "how can I help?" clause split out of "Hi, how can I help?". Anything else keeps
# its case: "Polish" and "polish", "Nice" and "nice" sound different.
_FOLD_WORDS = frozenset(
    [
        "a",
        "about",
        "absolutely",
        "actually",
        "after",
        "all",
        "also",
        "alright",
        "although",
        "an",
        "and",
        "any",
        "anything",
        "are",
        "as",
        "at",
        "awesome",
        "be",
        "because",
        "before",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "done",
        "each",
        "either",
        "even",
        "every",
        "everything",
        "excellent",
        "fantastic",
        "feel",
        "few",
        "first",
        "for",
        "from",
        "good",
        "got",
        "great",
        "had",
        "has",
        "have",
        "having",
        "he",
        "hello",
        "her",
        "here",
        "hey",
        "hi",
        "his",
        "hmm",
        "how",
        "however",
        "i'm",
        "if",
        "in",
        "is",
        "it",
        "it's",
        "its",
        "just",
        "kindly",
        "let",
        "let's",
        "like",
        "lovely",
        "may",
        "maybe",
        "me",
        "might",
        "more",
        "most",
        "much",
        "my",
        "no",
        "nope",
        "not",
        "now",
        "of",
        "oh",
        "ok",
        "okay",
        "on",
        "once",
        "one",
        "only",
        "or",
        "our",
        "perfect",
        "perhaps",
        "please",
        "right",
        "she",
        "should",
        "since",
        "so",
        "some",
        "something",
        "sorry",
        "still",
        "sure",
        "thank",
        "thanks",
        "that",
        "that's",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "though",
        "to",
        "too",
        "unfortunately",
        "until",
        "up",
        "us",
        "very",
        "wait",
        "was",
        "we",
        "we'll",
        "we're",
        "welcome",
        "well",
        "were",
        "what",
        "what's",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "wow",
        "yeah",
        "yes",
        "yep",
        "yet",
        "you",
        "you'll",
        "you're",
        "your",
    ]
)
_FIRST_WORD = re.compile(r"^([A-Z][a-z']*)(?=[\s,.;:!?]|$)")


def normalize_text(text: str) -> str:
    """NFKC, straight quotes, collapsed whitespace, stripped; case and punctuation kept,
    except the capital of a common opening word (see ``_FOLD_WORDS``)."""
    t = _WS.sub(" ", unicodedata.normalize("NFKC", text).translate(_QUOTES)).strip()
    m = _FIRST_WORD.match(t)
    if m and m.group(1).lower() in _FOLD_WORDS:
        t = t[0].lower() + t[1:]
    return t


def _excluded(name: str) -> bool:
    low = name.lower()
    return low in _EXCLUDED_NAMES or any(part in low for part in _EXCLUDED_NAME_PARTS)


def _canon(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return _canon(value.value, path)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise SettingsNotSerializableError(f"{path}: non-finite float")
        r = round(value, 6)
        return int(r) if r.is_integer() else r
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            ks = str(k)
            if _excluded(ks):
                continue
            out[ks] = _canon(v, f"{path}.{ks}")
        return out
    if isinstance(value, (list, tuple)):
        return [_canon(v, f"{path}[]") for v in value]
    if isinstance(value, (set, frozenset)):
        items = [_canon(v, f"{path}{{}}") for v in value]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
    dump = getattr(value, "model_dump", None)
    if callable(dump):  # pydantic v2 models
        return _canon(dump(), path)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canon(dataclasses.asdict(value), path)
    raise SettingsNotSerializableError(f"{path}: {type(value).__name__} has no canonical form")


def canonical_value(value: Any, path: str = "value") -> Any:
    """The canonical (JSON-ready) form of one setting value."""
    return _canon(value, path)


def canonical_settings(given: Mapping[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Split given settings into (model, voice, language, others), all canonical.
    Secrets, ids, timeouts and retries are left out (see ``_EXCLUDED_NAMES``)."""
    m = _canon(given.get("model"), "model")
    v = _canon(given.get("voice"), "voice")
    lang = _canon(given.get("language"), "language")
    others: dict[str, Any] = {}
    for name, value in given.items():
        if name in _PRIMARY or _excluded(name):
            continue
        others[name] = _canon(value, name)
    return m, v, lang, others


@dataclasses.dataclass(frozen=True, slots=True)
class KeyContext:
    """Everything but the text, pre-serialised: a digest is one hash call."""

    head: bytes
    tail: bytes
    voice_id: str

    @property
    def identity(self) -> str:
        """Every setting but the text, hashed: one render twin per distinct voice setup."""
        return hashlib.blake2b(self.head + self.tail, digest_size=12).hexdigest()

    def digest(self, text: str) -> bytes:
        """16 bytes: :func:`text_tag` of ``text``, then 12 bytes of blake2b over the
        canonical key (text, voice, settings, sample rate)."""
        t = json.dumps(normalize_text(text), ensure_ascii=True).encode("ascii")
        return text_tag(text) + hashlib.blake2b(self.head + t + self.tail, digest_size=12).digest()

    def canonical(self, text: str) -> str:
        """The exact canonical JSON hashed by :meth:`digest`, for tests and debugging."""
        t = json.dumps(normalize_text(text), ensure_ascii=True)
        return self.head.decode("ascii") + t + self.tail.decode("ascii")


TEXT_TAG_BYTES = 4


def text_tag(text: str) -> bytes:
    """The first bytes of every digest for ``text``, whatever the voice, settings or
    sample rate: purging a text finds every variant by this prefix. A collision
    with another text only deletes that text's audio too (a cache miss)."""
    t = json.dumps(normalize_text(text), ensure_ascii=True).encode("ascii")
    return hashlib.blake2b(t, digest_size=TEXT_TAG_BYTES, person=b"memcat.text").digest()


class Keyer:
    """Builds :class:`KeyContext` objects for one wrapped service.

    Settings are read fresh for every sentence, so a runtime
    ``TTSUpdateSettingsFrame`` changes the key from the next sentence on. The
    key context is memoised on the canonical form of the settings (the same form
    that is hashed), never on ``repr``: a value whose repr hides a field, or two
    values with the same repr, cannot reuse a stale key.
    """

    def __init__(self, service_path: str, extra: Mapping[str, Any] | None = None):
        self._service_path = service_path
        self._extra = _canon(dict(extra or {}), "init")
        self._memo_canon: str | None = None
        self._memo_rate = -1
        self._memo: KeyContext | None = None

    def context(self, given: Mapping[str, Any], sample_rate: int) -> KeyContext:
        """Return the key context for these settings and output rate.

        Raises:
            SettingsNotSerializableError: when a setting has no canonical form.
        """
        m, v, lang, others = canonical_settings(given)
        rep = json.dumps(
            [m, v, lang, others], sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        if self._memo is not None and rep == self._memo_canon and sample_rate == self._memo_rate:
            return self._memo
        if self._extra:
            others = {**others, "__init__": self._extra}
        fields: dict[str, Any] = {
            "l": lang,
            "m": m,
            "o": others,
            "p": self._service_path,
            "r": int(sample_rate),
            "s": KEY_SCHEMA,
            "t": "\x00",
            "v": v,
        }
        full = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        marker = '"t":"\\u0000"'
        idx = full.index(marker) + len('"t":')
        head = full[:idx].encode("ascii")
        tail = full[idx + len('"\\u0000"') :].encode("ascii")
        ctx = KeyContext(
            head=head, tail=tail, voice_id=json.dumps([m, v, lang, self._service_path])
        )
        self._memo_canon, self._memo_rate, self._memo = rep, sample_rate, ctx
        return ctx
