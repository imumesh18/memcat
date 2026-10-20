from __future__ import annotations

import re
import time

from hypothesis import given, settings
from hypothesis import strategies as st

from memcat.core.chunker import is_cacheable, split


@given(st.text(max_size=400))
@settings(max_examples=600, deadline=None)
def test_chunker_concat_identity(text: str) -> None:
    assert "".join(c.text for c in split(text)) == text


@given(
    st.lists(
        st.sampled_from(
            [
                "Hello there",
                "my friend",
                ",",
                ";",
                ":",
                " — ",
                " - ",
                "–",
                " ",
                "12345",
                "ok",
                "<spell>a, b</spell>",
                '{"word":"x","pronounce":"y"}',
            ]
        ),
        max_size=30,
    )
)
@settings(max_examples=400, deadline=None)
def test_chunker_concat_identity_structured(parts: list[str]) -> None:
    text = "".join(parts)
    assert "".join(c.text for c in split(text)) == text


def test_splits_at_commas_and_dashes() -> None:
    out = [c.text for c in split("Thanks for waiting, your order has shipped — it arrives soon.")]
    assert out == ["Thanks for waiting, ", "your order has shipped — ", "it arrives soon."]


def test_short_pieces_merge() -> None:
    assert [c.text for c in split("Well then, of course, I can help with that today.")] == [
        "Well then, ",
        "of course, I can help with that today.",
    ]
    assert len(split("Hi, there.")) == 1


def test_short_opener_stands_alone() -> None:
    assert [c.text for c in split("Hello, how can I help you today?")] == [
        "Hello, ",
        "how can I help you today?",
    ]
    assert [c.text for c in split("Red, green, blue, or yellow?")][1:] == [
        "green, blue, or yellow?"
    ]


def test_pii_never_pulls_neighbours_down() -> None:
    out = split("12345, got that.")
    assert [(c.text, c.cacheable) for c in out] == [("12345, ", False), ("got that.", True)]


def test_markup_is_never_split() -> None:
    t = "Say <spell>a, b, c</spell> for me, please, my good friend."
    assert all("<spell>" not in c.text or "</spell>" in c.text for c in split(t))
    t2 = 'Take {"word":"Metformin","pronounce":"mɛt"} daily, with plenty of water.'
    assert any('{"word"' in c.text and '"}' in c.text for c in split(t2))


BLOCKED = [
    "Your order 12345 shipped",
    "Your PIN is 4 2 9",
    "call 555-0199 now",
    "It is four five six seven",
    "write to jane@acme.com",
    "write to jane at acme dot com",
    "see https://acme.example/x",
    "visit acme.com today",
    "that is $20 total",
    "that will be twenty dollars",
    "Rs. 500 is due",
    "on March 5",
    "on the fifth of June",
    "on 5th May",
    "the 21st of the month",
    "meet at 10:30",
    "code AB12CD please",
    "spell K 7 X",
    "₹300",
    "100 euros",
]
ALLOWED = [
    "Thanks for calling, how can I help you today?",
    "Is there anything else I can help with?",
    "I can help you with that.",
    "May I have your name, please?",
    "One moment while I look that up.",
    "I will be there in a moment.",
    "It takes three to five business days.",
    "First, let me check your account.",
]


def test_merged_pieces_are_checked_again() -> None:
    out = split("My number is nine, eight, seven, six, five.")
    assert not any(c.cacheable and "seven" in c.text for c in out)
    out = split("The code is A, B, C, D, E, thanks for waiting.")
    assert not any(c.cacheable and "B, C" in c.text for c in out)
    assert split("Yes, I can, and I will help you today.")[-1].cacheable


def test_block_rules() -> None:
    for t in BLOCKED:
        assert not is_cacheable(t), t
    for t in ALLOWED:
        assert is_cacheable(t), t


def test_extra_block_patterns() -> None:
    pat = [re.compile(r"\bacme\b", re.IGNORECASE)]
    assert not is_cacheable("Welcome to Acme support", pat)
    assert is_cacheable("Welcome to our support line", pat)


def test_split_is_fast() -> None:
    t = (
        "Thanks for waiting, I have pulled up your account and I can see the order, "
        "it shipped yesterday and should arrive within three days."
    )
    for _ in range(200):
        split(t)
    n = 5000
    t0 = time.perf_counter()
    for _ in range(n):
        split(t)
    per = (time.perf_counter() - t0) / n
    assert per < 400e-6, f"{per * 1e6:.1f} us"  # ~40 us on a laptop; CI runners are slower


def test_hyphenated_number_words_block() -> None:
    for text in (
        "It costs twenty-five dollars",
        "Due May twenty-five",
        "See you on May twenty-first",
        "the twenty-first of May works",
        "call nine-eight-seven six",
    ):
        assert not is_cacheable(text), text
    assert is_cacheable("A well-known two-step process")


def test_word_lists_are_normalised() -> None:
    import unicodedata

    from memcat.core import chunker

    # Rules compare NFKC, lowercased tokens; a list entry in any other form never matches.
    for words in (
        chunker._NUM_LANG,
        chunker._MONTH_SET,
        chunker._UNIT_SET,
        chunker._ORDINAL_SET,
        chunker._NATO_SET,
        chunker._TLD_SET,
    ):
        for w in words:
            assert w == unicodedata.normalize("NFKC", w).lower(), w
            assert [m.group() for m in chunker._TOKEN_RE.finditer(w)] == [w], w
