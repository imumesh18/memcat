"""Memcat may lose savings but must never lose a call."""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pipecat.frames.frames import MetricsFrame, TTSSpeakFrame
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.services.tts_service import TextAggregationMode

from memcat import MemcatConfig, MemoryStore, RedisStore, cached
from memcat.core.chunker import split
from memcat.core.runtime import get_runtime
from tests.fakes import FakeHTTPTTS, FakeWSTTS, words_of
from tests.harness import hear, turn
from tests.util import VARIANTS, drain, make, prime

S = "Thanks for waiting today, your order 48213 has shipped, and it should arrive soon."


@pytest.mark.parametrize("variant", list(VARIANTS))
async def test_redis_down_bot_keeps_talking(variant) -> None:
    store = RedisStore("redis://10.255.255.1:6379/0", op_timeout_ms=50)
    cfg = MemcatConfig(store=store, admit_after_calls=1)
    tts = make(variant, cfg)
    frames = []
    for _ in range(4):
        frames += turn(S)
    h = await hear(tts, frames)
    assert h.words == words_of(S) * 4
    rt = get_runtime(cfg)
    m = rt.metrics.snapshot()
    assert m["store_errors"] >= 1
    assert m["misses"] >= 12


async def test_breaker_opens_and_l1_still_serves() -> None:
    store = RedisStore("redis://10.255.255.1:6379/0", op_timeout_ms=30)
    cfg = MemcatConfig(store=store)
    tts = cached(FakeHTTPTTS)(memcat=cfg)
    await hear(tts, turn(S) * 8)
    rt = get_runtime(cfg)
    assert rt.breaker.is_open
    assert rt.metrics.snapshot()["store_errors"] <= 6  # then the breaker stops asking


@pytest.mark.parametrize("variant", ["http", "ws-ctx"])
async def test_token_mode_passthrough(variant) -> None:
    cls, kw = VARIANTS[variant]
    tts = cached(cls)(memcat=MemcatConfig(), text_aggregation_mode=TextAggregationMode.TOKEN, **kw)
    plain = cls(text_aggregation_mode=TextAggregationMode.TOKEN, **kw)
    h0 = await hear(plain, turn("Thanks ", "for ", "waiting, ", "friend."))
    h1 = await hear(tts, turn("Thanks ", "for ", "waiting, ", "friend."))
    assert h1.words == h0.words
    assert (await tts.memcat_stats())["enabled"] is False


def _billed(h) -> int:
    n = 0
    for f in h.frames:
        if isinstance(f, MetricsFrame):
            for d in f.data:
                if isinstance(d, TTSUsageMetricsData):
                    n += d.value
    return n


@pytest.mark.parametrize("variant", list(VARIANTS))
async def test_usage_metrics_count_billed_only(variant) -> None:
    store = MemoryStore()
    cf = lambda: MemcatConfig(store=store, admit_after_calls=1)  # noqa: E731
    await prime(variant, cf, ["Thanks for waiting today,", "and it should arrive soon."])
    tts = make(variant, cf())
    h = await hear(tts, turn(S), metrics=True)
    assert h.words == words_of(S)
    live = "your order 48213 has shipped, "
    assert _billed(h) == len(live)


@pytest.mark.parametrize("variant", ["http", "ws-ctx", "ws-cursor", "ws-el"])
async def test_exception_in_memcat_sends_live(variant, monkeypatch) -> None:
    cfg = MemcatConfig(store=MemoryStore())
    tts = make(variant, cfg)

    async def boom(*a, **k):
        raise RuntimeError("bug in memcat")

    h0 = await hear(tts, [TTSSpeakFrame("warm up the runtime please")])
    monkeypatch.setattr(get_runtime(cfg).lookup, "get_many", boom)
    h = await hear(tts, turn(S + " ", "Is there anything else?"))
    assert h.words == words_of(S + " Is there anything else?")
    assert (await tts.memcat_stats())["enabled"] is False
    assert h0.words


async def test_provider_error_propagates_like_plain_pipecat() -> None:
    plain = FakeHTTPTTS(fail_on="48213")
    tts = cached(FakeHTTPTTS)(memcat=MemcatConfig(), fail_on="48213")
    h0 = await hear(plain, turn(S))
    h1 = await hear(tts, turn(S))
    errs0 = [type(f).__name__ for f in h0.frames if "Error" in type(f).__name__]
    errs1 = [type(f).__name__ for f in h1.frames if "Error" in type(f).__name__]
    assert errs0 == errs1


PII = st.one_of(
    st.integers(min_value=100, max_value=10**9).map(str),
    st.lists(st.integers(0, 9), min_size=3, max_size=6).map(lambda d: " ".join(map(str, d))),
    st.from_regex(r"[a-z]{3,8}@[a-z]{3,8}\.(com|org|in)", fullmatch=True),
    st.from_regex(r"https?://[a-z]{3,8}\.(com|io)/[a-z0-9]{1,6}", fullmatch=True),
    st.from_regex(r"[A-Z]{2}[0-9]{2}[A-Z]{2}", fullmatch=True),
    st.integers(1, 999).map(lambda n: f"${n}"),
    st.integers(1, 999).map(lambda n: f"{n} dollars"),
    st.sampled_from(["March", "July", "December"]).flatmap(
        lambda m: st.integers(1, 28).map(lambda d: f"{m} {d}")
    ),
    st.integers(1, 28).map(lambda d: f"{d}th of May"),
    st.lists(
        st.sampled_from(["four", "five", "six", "seven", "nine", "zero"]), min_size=3, max_size=6
    ).map(" ".join),
)
TEMPLATES = [
    "Thanks for waiting, your reference is {}, and I have noted it down.",
    "Sure thing, I can see {} on your account; is that right?",
    "{}, got that, let me check.",
    "I have sent it to {} — please check your inbox.",
    "Your appointment is on {}, at the main office.",
]
_LEAK = re.compile(r"\d|@|https?://|www\.|[$€£₹]|\bdollars?\b", re.IGNORECASE)
_NUMWORD = {"four", "five", "six", "seven", "nine", "zero"}


@given(st.sampled_from(TEMPLATES), PII)
@settings(max_examples=500, deadline=None)
def test_pii_never_cached(template: str, pii: str) -> None:
    sentence = template.format(pii)
    for c in split(sentence):
        if c.cacheable:
            assert pii not in c.text, (sentence, c.text)
            assert not _LEAK.search(c.text), (sentence, c.text)
            toks = re.findall(r"[a-z]+", c.text.lower())
            run = best = 0
            for t in toks:
                run = run + 1 if t in _NUMWORD else 0
                best = max(best, run)
            assert best < 3, (sentence, c.text)


async def test_pii_never_reaches_the_store() -> None:
    store = MemoryStore()
    stored: list[str] = []
    phrases = [
        t.format(p)
        for t, p in zip(
            TEMPLATES, ["4471", "jane@acme.com", "AB12CD", "March 3", "$42"], strict=True
        )
    ]
    for i in range(3):  # admitted after 3 calls, and still never stored
        cfg = MemcatConfig(store=store, admit_after_calls=3)
        tts = cached(FakeWSTTS)(memcat=cfg)
        await hear(tts, [TTSSpeakFrame(p) for p in phrases])
        await drain(cfg)
    rt = get_runtime(cfg)
    key = tts._memcat_key()
    allowed = {key.digest(c.text) for p in phrases for c in split(p) if c.cacheable}
    for d in rt.index._keys:
        assert d in allowed
    del stored
