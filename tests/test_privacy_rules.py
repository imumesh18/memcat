"""Privacy rules on hard cases (Unicode, spoken numbers, codes, addresses) and time bounds."""

from __future__ import annotations

import random
import re
import time

import pytest

from memcat.core.chunker import _protected, is_cacheable, split

# Each of these carries a number, code or address a naive rule set misses.
BLOCKED = [
    # Unicode digits and symbols (NFKC)
    "Your PIN is ①②③④.",
    "SSN ends with ⁴⁵⁶⁷.",
    "Pay ＄ fifty.",
    "Your code is ⅷ today",
    "आपका पिन ९८७६ है",
    # number words joined by separators or each other
    "nine; eight; seven; six",
    "nine / eight / seven",
    "It is two hundred and fifty",
    "that is forty two",
    "born in nineteen ninety",
    "nine point five",
    "five oh three",
    "the twenty first works",
    "at nine pm",
    "at nine o'clock",
    "nine a.m. works",
    # money
    "about fifty grand",
    "costs fifty ¢",
    "pay ₺ today",
    "pay ₫ today",
    "pay ₴ today",
    "pay ₪ today",
    "pay ₿ today",
    # spelled codes
    "the code is (Q X R)",
    "the code is Q/X/R",
    "the code is 'Q' 'X' 'R'",
    "that is Alpha Bravo Charlie",
    "that is Bravo Kilo",
    "B as in boy",
    "A nine B",
    "your code is QXRTZP",
    "your code is BXTR",
    "your code is ABEKU",
    # emails and domains spoken or written
    "go to example dot com",
    "email john (at) gmail (dot) com",
    "write to jane at acme dot xyz",
    "see acme.xyz now",
    "see acme.ca now",
    "see shop.example.co.uk now",
    # Hindi number words
    "आपका नंबर नौ आठ सात छह है",
    "पांच सौ रुपये",
    "aapka number nau aath saat chhe hai",
    "do sau pachaas",
    "paanch rupaye",
    "Aapka OTP hai chaubis pachchis untis.",
    "आपका ओटीपी है चौबीस पच्चीस उनतीस।",
    "Aapki EMI pachchis sau hai.",
    "आपकी जन्मतिथि चौबीस मार्च है।",
    # number words and codes joined by "then", "aur", "phir"
    "Your OTP is four, then two, then seven, then one.",
    "Your code is nine. Then eight. Then seven.",
    "Aapka code hai ek aur do aur teen.",
    "chaar, phir do, phir saat.",
    "That's Bravo, then Kilo, then Lima.",
    "It's B, then K, then L.",
    # spoken emails without the domain ending
    "Your email on file is priya sharma at gmail.",
    "Your email is priya at the rate gmail.",
]

# Common support lines that must stay cacheable: the hit rate depends on them.
ALLOWED = [
    "Thanks for waiting today,",
    "Is there anything else I can help with?",
    "Sure,",
    "One moment please.",
    "Oh, one moment please.",
    "Just one second, please.",
    "I can help with that.",
    "First, may I have your name?",
    "Is anyone else on the line?",
    "OK, your ID has been verified.",
    "Please do NOT share your PIN.",
    "We'll send an SMS shortly, see our FAQ.",
    "That is ASAP.",
    "I'm a bit confused, I'm sorry.",
    "It's a one-time code.",
    "I can do one more thing for you.",
    "Let me double check that.",
    "One or two business days.",
    "It takes three to five business days.",
    "I'm a hundred percent sure.",
    "Plan A or B works.",
    "That is, e.g. a refund.",
    "I am here to help.",
    "aapke saath baat karke accha laga",
    "मैं आपकी मदद कर सकता हूँ",
    "Let me check that, then I'll get back to you.",
    "Press one, then wait.",
    "Meet me at the station.",
]


@pytest.mark.parametrize("text", BLOCKED)
def test_probe_is_blocked(text: str) -> None:
    assert not is_cacheable(text), text
    assert not any(c.cacheable for c in split(text)), split(text)


@pytest.mark.parametrize("text", ALLOWED)
def test_support_line_stays_cacheable(text: str) -> None:
    assert is_cacheable(text), text


@pytest.mark.parametrize(
    "text",
    [
        "Call nine eight, seven six five four, three two.",
        "Call nine, seven six five four, three.",
        "Your PIN is 4471, nine, thanks.",
        "nine; eight; seven; six",
        "Sure, nine, eight, seven, six.",
        "It is two hundred, and fifty.",
    ],
)
def test_number_pieces_are_not_cached(text: str) -> None:
    for c in split(text):
        if c.cacheable:
            toks = re.findall(r"[a-z]+", c.text.lower())
            assert not {"nine", "eight", "three", "two", "fifty", "hundred"} & set(toks), c


def test_number_neighbours_spare_other_clauses() -> None:
    out = split("Your PIN is 4471, thanks for waiting today.")
    assert out[-1] == split("x, thanks for waiting today.")[-1]
    assert out[-1].cacheable
    out = split("I sent it to jane@acme.com, one moment please.")
    assert out[-1].cacheable  # blocked for an email, not a number: does not spread


# --- linear time -------------------------------------------------------------

ADVERSARIAL = [
    "a." * 16000,
    "a-" * 16000,
    "a " * 16000,
    "<" * 20000,
    "<<" * 10000,
    "<a>" * 10000,
    "<a1>" * 2000 + "".join(f"<t{i}>" for i in range(3000)),
    "{" * 20000,
    "[" * 20000,
    " " * 30000 + "x",
    "x" + " " * 30000 + "-",
    "a, " * 10000,
    "jane " * 10000 + "at",
    "one " * 10000,
    "Q " * 10000 + "QQ",
    "a.b" * 10000,
    "<a:" + "a:" * 10000 + ">",
    "nine dot " * 5000,
    "①" * 20000,
]


@pytest.mark.parametrize("text", ADVERSARIAL, ids=lambda t: repr(t[:8]))
def test_adversarial_text_is_fast(text: str) -> None:
    t0 = time.perf_counter()
    out = split(text)
    is_cacheable(text)
    elapsed = time.perf_counter() - t0
    assert "".join(c.text for c in out) == text
    assert elapsed < 0.5, f"{elapsed:.2f}s for {len(text)} chars"  # quadratic took 1-16 s


def test_time_grows_linearly() -> None:
    def cost(n: int) -> float:
        text = "a." * n + " <<" * n + " <a>" * n
        t0 = time.perf_counter()
        split(text)
        return time.perf_counter() - t0

    cost(1000)
    small, big = min(cost(2000) for _ in range(3)), min(cost(16000) for _ in range(3))
    assert big < small * 8 * 4, (small, big)  # 8x the input; quadratic would be ~64x


# --- markup scanner matches the regex it replaced -----------------------------

_OLD_PROTECT = re.compile(
    r"<([A-Za-z][\w:-]*)\b[^<>]*>.*?</\1\s*>|<<.*?>>|<[^<>]*>|\{[^{}]*\}|\[[^\[\]]*\]", re.DOTALL
)


def test_protected_spans_match_old_regex() -> None:
    parts = ["<", ">", "</", "a", "b", "ab", " ", "{", "}", "[", "]", "<<", ">>", "-", ":"]
    parts += ["\n", "x", "<a>", "</a>", "<b x='1'>", "</b >", "<a:x>", "<spell>", "</spell>"]
    rng = random.Random(7)
    for _ in range(20000):
        t = "".join(rng.choices(parts, k=rng.randint(0, 14)))
        assert _protected(t) == [(m.start(), m.end()) for m in _OLD_PROTECT.finditer(t)], t


@pytest.mark.parametrize(
    "text", ["注文番号は三四五六です", "您的验证码是七八九", "金額は三百円です"]
)
def test_cjk_numerals_block(text: str) -> None:
    assert not is_cacheable(text)


@pytest.mark.parametrize("text", ["一つお待ちください", "ご注文ありがとうございます"])
def test_single_cjk_numeral_stays_cacheable(text: str) -> None:
    assert is_cacheable(text)


@pytest.mark.parametrize(
    "text",
    [
        "Your PIN is nine​ eight​ seven​ six.",
        "Your code is Q​X​R.",
        "Your PIN is nine⁠ eight⁠ seven.",
        "B for Bravo, K for Kilo.",
    ],
)
def test_hidden_characters_and_letter_for_word_spelling_block(text: str) -> None:
    assert not is_cacheable(text)


def test_spelled_code_cut_into_clauses_blocks_every_piece() -> None:
    got = [(c.text, c.cacheable) for c in split("Your code is Q, X, R, thanks for waiting.")]
    assert got == [("Your code is Q, X, R, ", False), ("thanks for waiting.", True)]
    assert not any(c.cacheable for c in split("B for Bravo, K for Kilo.") if "Kilo" in c.text)


@pytest.mark.parametrize(
    "text", ["A for effort, thanks.", "I for one agree.", "Thanks​ for waiting"]
)
def test_ordinary_phrases_stay_cacheable(text: str) -> None:
    assert is_cacheable(text)
