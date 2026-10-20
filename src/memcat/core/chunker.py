"""Split a sentence into clauses and mark the ones that may never be cached.

The splitter is lossless: ``"".join(c.text for c in split(t)) == t`` for every
input, so the exact text the bot was asked to say is always what reaches the
listener, whether a clause comes from the cache or from the provider.

Every regex here runs in linear time: the text comes from an LLM and can be anything.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["Clause", "is_cacheable", "split"]


@dataclass(frozen=True, slots=True)
class Clause:
    """One clause of a sentence.

    Attributes:
        text: Exact substring of the sentence, trailing punctuation and space included.
        cacheable: False when the clause may carry personal data and must never be stored.
    """

    text: str
    cacheable: bool


# --- split points -----------------------------------------------------------
#
# After ", ; :" followed by whitespace, and after a spaced dash ("a — b", "a - b")
# or an unspaced em/en dash ("a—b"). The lookbehind starts a spaced dash only at
# the first space of a run, so a long run of spaces is scanned once, not once per
# space. Markup that providers read as instructions (SSML/Cartesia tags, Deepgram's
# inline pronunciation objects) is never split; see ``_protected``.
_SPLIT_RE = re.compile(r"[,;:](?=\s)\s*|(?<!\s)\s+[—–-]\s+|[—–]")
_OPENERS_RE = re.compile(r"[<{\[]")
_OPEN_TAG_RE = re.compile(r"<([A-Za-z][\w:-]*+)[^<>]*>")
_CLOSE_TAG_RE = re.compile(r"</([A-Za-z][\w:-]*+)\s*>")
_ANGLE_RE = re.compile(r"<[^<>]*>")
_BRACE_RE = re.compile(r"\{[^{}]*\}")
_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")

# --- hard-block rules -------------------------------------------------------
#
# Every rule runs on the NFKC form of the text, so "①②", "⁴⁵" and "＄" count as
# the digits and symbol they stand for. Any digit (any script), any other numeric
# character and any currency symbol (Unicode category Sc) blocks a clause. The word
# rules below catch the same things spoken aloud (English, and Hindi in Devanagari
# or Latin letters). They work on a token list in one pass, with no regex that can
# backtrack, so time stays linear in the text whatever an LLM emits.
#
# A false positive costs one live render; a false negative caches a caller's data.
# The rules lean that way, but leave single number words alone: "one moment",
# "a second", "first," and "anyone" are some of the most repeated lines there are.


def _word_list(words: str) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFKC", w) for w in words.split())


_EN_NUM_LIST = _word_list(
    "zero oh one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
    "sixty seventy eighty ninety hundred thousand million billion double triple nought "
    "nil"
)
# Hindi number words, romanised and in Devanagari (Sarvam bots speak both).
_HI_NUM_LIST = _word_list(
    "shunya shoonya ek do teen char chaar paanch panch chhe chhah chah saat aath nau das "
    "gyarah gyaarah barah baarah terah chaudah pandrah solah satrah atharah unnis bees "
    "tees chalis chaalis pachas pachaas saath sattar assi nabbe sau hazaar hazar lakh "
    "crore dedh dhai sava saade saadhe "
    "ikkis baais bais teis chaubis chaubees pachchis pachis pachees chhabbis sattais "
    "atthais athais untis ikattis iktees battis taintis tetis chautis paintis chhattis "
    "saintis adtis artis untalis unchalis iktalis bayalis taintalis chavalis chauvalis "
    "paintalis chhiyalis saintalis artalis adtalis unchas ikyavan bavan tirpan chauvan "
    "pachpan chhappan sattavan atthavan unsath iksath basath tirsath chaunsath painsath "
    "chhiyasath sadsath adsath unhattar ikhattar bahattar tihattar chauhattar pachhattar "
    "chhihattar satattar athattar unasi ikyasi bayasi tirasi chaurasi pachasi chhiyasi "
    "sattasi atthasi navasi ikyanve banve tiranve chauranve pachanve chhiyanve sattanve "
    "atthanve ninyanve "
    "शून्य एक दो तीन चार पांच पाँच छह छः छे सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह "
    "सोलह सत्रह अठारह उन्नीस बीस तीस चालीस पचास साठ सत्तर अस्सी नब्बे सौ हज़ार हजार "
    "लाख करोड़ डेढ़ ढाई सवा साढ़े "
    "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस इकतीस बत्तीस तैंतीस "
    "चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस इकतालीस बयालीस तैंतालीस चौवालीस "
    "पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास इक्यावन बावन तिरपन चौवन पचपन छप्पन "
    "सत्तावन अट्ठावन उनसठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी इक्यासी "
    "बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी इक्यानबे बानबे तिरानबे "
    "चौरानबे पचानबे छियानबे सत्तानबे अट्ठानबे निन्यानबे"
)
_TENS = frozenset(_word_list("twenty thirty forty fifty sixty seventy eighty ninety"))
_MONTH_LIST = _word_list(
    "jan january feb february mar march apr april may jun june jul july aug august sep "
    "sept september oct october nov november dec december "
    "जनवरी फ़रवरी फरवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर सितम्बर अक्टूबर अक्तूबर "
    "नवंबर नवम्बर दिसंबर दिसम्बर"
)
# Money and time units: a number word right next to one of these is an amount or a time.
_UNIT_LIST = _word_list(
    "dollar dollars buck bucks grand cent cents euro euros pound pounds pence quid rupee "
    "rupees rupaye rupaiye rupaya paise paisa yen yuan franc francs peso pesos reais real "
    "rand dirham dirhams riyal riyals won usd eur gbp inr jpy cad aud rs lakh lakhs "
    "crore crores रुपये रुपए रुपया रुपयों पैसे"
)
_TIME_LIST = _word_list("am pm o'clock oclock")  # only after the number: "nine pm"
_ORDINAL_LIST = _word_list(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
    "twentieth thirtieth"
)
_NATO_LIST = _word_list(
    "alpha alfa bravo charlie delta echo foxtrot golf hotel india juliet juliett kilo "
    "lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "whisky xray yankee zulu"
)
_TLD_LIST = _word_list(
    "com org net io ai co in uk us de fr app dev me info biz gov edu ca au nz xyz tv ly gg "
    "jp cn ru br es it nl eu online site store tech"
)
# Words that join the digits of one spoken number: "two hundred and fifty", "nine point five",
# "four, then two".
_JOINERS = frozenset(_word_list("and point dash hyphen slash dot then aur phir fir और फिर"))
_MAIL_SET = frozenset(
    _word_list("gmail googlemail yahoo ymail outlook hotmail icloud rediffmail protonmail aol")
)

# Number words map to a language, and two of them only make a number in the same
# language: romanised Hindi "do" and "teen" are also English words ("do one thing").
_NUM_LANG = {w: "en" for w in _EN_NUM_LIST} | {w: "hi" for w in _HI_NUM_LIST}
_MONTH_SET = frozenset(_MONTH_LIST)
_UNIT_SET = frozenset(_UNIT_LIST)
_DATE_OR_UNIT = _MONTH_SET | _UNIT_SET
_AFTER_NUMBER = _DATE_OR_UNIT | frozenset(_TIME_LIST)
_ORDINAL_SET = frozenset(_ORDINAL_LIST)
_NATO_SET = frozenset(_NATO_LIST)
_TLD_SET = frozenset(_TLD_LIST)
_SINGLE_WORDS = frozenset("aio")  # letters that are words; a run of only these is prose
_VOWELS = frozenset("AEIOUY")

_OK, _NUMBER, _OTHER = 0, 1, 2  # why a text is blocked; NUMBER spreads to neighbours

_HAS_DIGIT = re.compile(r"\d")
_SYMBOLS_RE = re.compile(r"[@$]|://|\bwww\.", re.IGNORECASE)
# A dot between a letter or digit and two letters: "acme.ca", "shop.example.xyz".
# Also catches "Mr.Smith"; a false positive is fine. No quantifier can backtrack.
_DOMAIN_RE = re.compile(r"[^\W_]\.[^\W\d_]{2}")
# A token is a run of anything but whitespace and punctuation; apostrophes stay inside
# ("I'm") and are trimmed from the ends ("'Q'"). Devanagari vowel signs are kept.
_TOKEN_RE = re.compile(
    r"[^\s!-&(-/:-@\[-`{-~¡-¿‐-‗‚-‧‰-⁞"
    r"।॥　-〿]+"
)
_PLAIN_GAP = re.compile(r"[\s.\-]*")  # "May 5", "Mar. fifth", "twenty-first"
_APOSTROPHES = "'‘’"
_LATIN_ABBREV = re.compile(r"\b(?:e\.g|i\.e)\.", re.IGNORECASE)  # not letters spelled out


def _word_count(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())


def is_cacheable(text: str, extra_blocks: Sequence[re.Pattern[str]] = ()) -> bool:
    """Return False when ``text`` matches a hard-block rule or an extra pattern."""
    return _verdict(text, extra_blocks) == _OK


def _normalized(text: str) -> str:
    """NFKC, without invisible format characters (zero-width spaces, joiners, soft
    hyphens) that would otherwise hide "nine\u200b eight" from the word rules."""
    if text.isascii():
        return text
    t = unicodedata.normalize("NFKC", text)
    return "".join(" " if unicodedata.category(ch) == "Cf" else ch for ch in t)


def _verdict(text: str, extra_blocks: Sequence[re.Pattern[str]] = ()) -> int:
    t = _normalized(text)
    if _HAS_DIGIT.search(t) or "$" in t:
        return _NUMBER
    if not text.isascii():  # before NFKC, which turns "ⅷ" into letters
        for ch in set(text):
            if ch > "\x7f":
                cat = unicodedata.category(ch)
                if cat[0] == "N" or cat == "Sc":  # ⅷ, ௰, ¢, ₺, ₪ ...
                    return _NUMBER
        # Numerals written as letters (三, 七, 百 ...): two in a row read as a number.
        run = 0
        for ch in text:
            run = run + 1 if ch > "\x7f" and unicodedata.numeric(ch, None) is not None else 0
            if run >= 2:
                return _NUMBER
    if _SYMBOLS_RE.search(t) or _DOMAIN_RE.search(t):
        return _OTHER
    why = _word_rules(_LATIN_ABBREV.sub("eg", t))
    if why:
        return why
    return _OTHER if any(p.search(text) for p in extra_blocks) else _OK


def _tokens(t: str) -> list[tuple[str, str, bool]]:
    """(token, lowercased, gap before it is plain) for each word of ``t``."""
    out: list[tuple[str, str, bool]] = []
    prev = 0
    for m in _TOKEN_RE.finditer(t):
        raw = m.group().strip(_APOSTROPHES)
        if raw:
            out.append((raw, raw.lower(), _PLAIN_GAP.fullmatch(t, prev, m.start()) is not None))
            prev = m.end()
    return out


def _has_code_word(text: str) -> bool:
    t = _normalized(text)
    return any(w in _NUM_LANG or w in _NATO_SET for _, w, _ in _tokens(t))


def _word_rules(t: str) -> int:
    toks = _tokens(t)
    words = [w for _, w, _ in toks]
    n = len(words)
    run = 0  # consecutive spelled-code units: single letters, number words, NATO words
    run_signal = False  # the run holds something other than "a", "I", "o"
    for i, (raw, w, plain) in enumerate(toks):
        prev = words[i - 1] if i else ""
        nxt = words[i + 1] if i + 1 < n else ""
        nxt_plain = i + 1 < n and toks[i + 1][2]
        # All-caps codes: "QXRTZP", "ABCDE", "BXTR". Short acronyms ("OK", "ID", "PIN",
        # "SMS", "FAQ") and four-letter ones with a vowel ("ASAP", "FDIC") stay cacheable.
        if (
            len(raw) >= 4
            and raw.isascii()
            and raw.isalpha()
            and raw.isupper()
            and (len(raw) >= 5 or _VOWELS.isdisjoint(raw))
        ):
            return _NUMBER
        lang = _NUM_LANG.get(w)
        if lang:
            # Two number words in a row make a number, whatever separates them:
            # "forty two", "nineteen ninety", "nine; eight", "two hundred and fifty".
            # Interjection "Oh," before a number word is not a zero: "Oh, one moment".
            if _NUM_LANG.get(prev) == lang and (prev != "oh" or plain):
                return _NUMBER
            if prev in _JOINERS and i >= 2 and _NUM_LANG.get(words[i - 2]) == lang:
                return _NUMBER
            # One number word next to a month or unit: "twenty dollars", "May five",
            # "nine pm", "five of June", "twenty-first".
            if (plain and prev in _DATE_OR_UNIT) or (nxt_plain and nxt in _AFTER_NUMBER):
                return _NUMBER
            if nxt == "of" and i + 2 < n and words[i + 2] in _MONTH_SET:
                return _NUMBER
            if w in _TENS and nxt_plain and nxt in _ORDINAL_SET:
                return _NUMBER
        if w in _ORDINAL_SET and (
            (nxt_plain and nxt in _MONTH_SET)
            or (plain and prev in _MONTH_SET)
            or (nxt == "of" and i + 2 < n and words[i + 2] in _MONTH_SET)
            or (prev == "the" and i >= 2 and words[i - 2] in _MONTH_SET)
        ):
            return _NUMBER
        # Spoken emails and domains: "example dot com", "john (at) gmail (dot) xyz",
        # "priya at gmail", "priya at the rate gmail".
        if w == "dot" and (nxt in _TLD_SET or "at" in words[max(0, i - 3) : i]):
            return _OTHER
        if w == "at" and (nxt in _MAIL_SET or words[i + 1 : i + 3] == ["the", "rate"]):
            return _OTHER
        # Spelled codes: "Q X R", "(Q X R)", "Q/X/R", "A nine B", "Bravo Kilo",
        # "B as in Bravo", "B for Bravo". Letters that are words ("I", "a") alone are prose.
        single = len(w) == 1 and w.isalpha()
        if single and nxt == "as" and i + 2 < n and words[i + 2] == "in":
            return _NUMBER
        if single and nxt == "for" and i + 2 < n and words[i + 2] in _NATO_SET:
            return _NUMBER  # "B for Bravo"
        if single or lang or w in _NATO_SET:
            if w in _NATO_SET and (
                prev in _NATO_SET or (prev in _JOINERS and i >= 2 and words[i - 2] in _NATO_SET)
            ):
                return _NUMBER
            run += 1
            run_signal = run_signal or w not in _SINGLE_WORDS
            if run >= 3 and run_signal:
                return _NUMBER
        elif w not in _JOINERS:  # "B, then K, then L" is still one code
            run, run_signal = 0, False
    return _OK


def split(
    text: str,
    *,
    min_words: int = 3,
    extra_blocks: Sequence[re.Pattern[str]] = (),
) -> list[Clause]:
    """Split ``text`` into clauses at commas, semicolons, colons and dashes.

    Pieces shorter than ``min_words`` merge into a neighbour of the same kind
    (cacheable or not). A piece is never merged across that boundary, so a
    clause holding personal data never takes a reusable clause down with it:
    in ``"12345, got that."`` the tail ``"got that."`` stays cacheable. The one
    exception is a number cut in pieces: a piece holding a number word next to a
    piece blocked for a number is blocked too ("Call nine, eight seven six, five").
    """
    if not text:
        return []
    pieces = _pieces(text)
    whole = _verdict(text, extra_blocks)
    if whole == _OK or len(pieces) == 1:
        # Built-in rules match locally, so a sentence that is clean as a whole is
        # clean in every piece. A user pattern may be anchored (^...) or need
        # context, so with those each final clause is checked again.
        return _merge(
            [[p, whole] for p in pieces], min_words, extra_blocks, recheck=bool(extra_blocks)
        )
    marked: list[list[str | int]] = [[p, _verdict(p, extra_blocks)] for p in pieces]
    # A rule can match across a cut ("nine; eight", "Q, X, R"): every piece it spans is
    # blocked. Spelled codes need three units, so windows of three pieces are checked
    # as well as pairs.
    for size in (2, 3):
        for i in range(len(marked) - size + 1):
            window = marked[i : i + size]
            if all(g[1] == _OK for g in window):
                why = _verdict("".join(str(g[0]) for g in window))
                if why:
                    for g in window:
                        g[1] = why
    _spread_numbers(marked)
    return _merge(marked, min_words, extra_blocks)


def _spread_numbers(groups: list[list[str | int]]) -> None:
    """Block each clean group holding a number or NATO word next to a group blocked for a number.

    "Call nine eight, seven six five four, three two." is one number said in three
    pieces; each piece alone may look harmless. Two passes cover chains both ways.
    """
    n = len(groups)
    for i in (*range(1, n), *range(n - 2, -1, -1)):
        if groups[i][1] != _OK or not _has_code_word(str(groups[i][0])):
            continue
        if (i and groups[i - 1][1] == _NUMBER) or (i + 1 < n and groups[i + 1][1] == _NUMBER):
            groups[i][1] = _NUMBER


def _join_why(a: str | int, b: str | int) -> int:
    """Reason for two joined pieces: a number wins, so it keeps spreading."""
    return _NUMBER if _NUMBER in (a, b) else max(int(a), int(b))


def _protected(text: str) -> list[tuple[int, int]]:
    """Spans of markup, left to right, as ``finditer`` over these alternatives would give:
    ``<tag ...>...</tag>``, ``<<...>>``, ``<...>``, ``{...}``, ``[...]``.

    Hand-rolled so that unclosed openers cost one scan in total, not one per opener.
    """
    spans: list[tuple[int, int]] = []
    closes: dict[str, tuple[list[int], list[int]]] | None = None
    next_gg = -2  # cached position of the next ">>" (-1: none left, -2: not searched)
    i = 0
    while (m := _OPENERS_RE.search(text, i)) is not None:
        s = m.start()
        end = -1
        c = text[s]
        if c == "<":
            om = _OPEN_TAG_RE.match(text, s)
            if om:
                if closes is None:
                    closes = {}
                    for cm in _CLOSE_TAG_RE.finditer(text):
                        starts, ends = closes.setdefault(cm.group(1), ([], []))
                        starts.append(cm.start())
                        ends.append(cm.end())
                for name in _tag_names(text, om):
                    found = closes.get(name)
                    if found and (k := bisect_left(found[0], om.end())) < len(found[0]):
                        end = found[1][k]
                        break
            if end < 0 and text.startswith("<<", s):
                if next_gg == -2 or 0 <= next_gg < s + 2:
                    next_gg = text.find(">>", s + 2)
                if next_gg >= 0:
                    end = next_gg + 2
            if end < 0 and (am := _ANGLE_RE.match(text, s)):
                end = am.end()
        elif (bm := (_BRACE_RE if c == "{" else _BRACKET_RE).match(text, s)) is not None:
            end = bm.end()
        if end >= 0:
            spans.append((s, end))
            i = end
        else:
            i = s + 1
    return spans


def _tag_names(text: str, om: re.Match[str]) -> list[str]:
    """Names ``<([A-Za-z][\\w:-]*)\\b`` could capture at this tag, longest first, as the
    regex would backtrack through them looking for a closing tag ("<a:x>" tries
    "a:x", "a:", "a"). Real tag names are short; past 64 characters only the full
    name is tried, to keep the scan linear."""
    name = om.group(1)
    if len(name) > 64:
        return [name.rstrip(":-")]
    after = text[om.end(1) : om.end(1) + 1]
    chars = name + after
    return [
        name[:k]
        for k in range(len(name), 0, -1)
        if _is_word(chars[k - 1]) != (k < len(chars) and _is_word(chars[k]))
    ]


def _is_word(c: str) -> bool:
    return c == "_" or c.isalnum()


def _pieces(text: str) -> list[str]:
    protected = _protected(text) if _OPENERS_RE.search(text) else []
    ends = [e for _, e in protected]
    cuts: list[int] = []
    for m in _SPLIT_RE.finditer(text):
        end = m.end()
        if end >= len(text) or not text[end:].strip():
            continue
        # Spans are sorted and disjoint, and a split match holds no opener, so only
        # the first span ending after the match start can contain either end of it.
        k = bisect_right(ends, m.start())
        if k < len(protected):
            s, e = protected[k]
            if s < m.start() < e or s < end < e:
                continue
        cuts.append(end)
    if not cuts:
        return [text]
    out: list[str] = []
    prev = 0
    for c in cuts:
        if c > prev:
            out.append(text[prev:c])
            prev = c
    out.append(text[prev:])
    return out


def _merge(
    marked: list[list[str | int]],
    min_words: int,
    extra_blocks: Sequence[re.Pattern[str]] = (),
    recheck: bool = True,
) -> list[Clause]:
    # A short opener ("Hello, ", "Sure, ", "Thanks! ") is a natural unit of speech
    # and the most repeated clause there is, so it stands alone whatever its length.
    lead = 0
    if (
        len(marked) > 1
        and marked[0][1] == _OK
        and _word_count(str(marked[0][0])) <= 2
        and _word_count("".join(str(t) for t, _ in marked[1:])) >= min_words
    ):
        lead = 1
    # 1. Join neighbours of the same kind until each reaches min_words.
    groups: list[list[str | int]] = [[t, why] for t, why in marked[:lead]]
    for piece, why in marked[lead:]:
        if (
            len(groups) > lead
            and (groups[-1][1] == _OK) == (why == _OK)
            and _word_count(str(groups[-1][0])) < min_words
        ):
            groups[-1] = [str(groups[-1][0]) + str(piece), _join_why(groups[-1][1], why)]
        else:
            groups.append([piece, why])
    # 2. A short trailing group folds back into the previous group of its kind.
    merged: list[list[str | int]] = []
    for g in groups:
        if (
            len(merged) > lead
            and (merged[-1][1] == _OK) == (g[1] == _OK)
            and (_word_count(str(g[0])) < min_words or _word_count(str(merged[-1][0])) < min_words)
        ):
            merged[-1] = [str(merged[-1][0]) + str(g[0]), _join_why(merged[-1][1], g[1])]
        else:
            merged.append(g)
    # 3. A rule can match across pieces that were checked one by one ("nine, eight,
    #    seven, six"), so every merged clause is checked again as a whole, and a
    #    number word next to a newly blocked number is blocked with it.
    if recheck:
        for g in merged:
            if g[1] == _OK:
                g[1] = _verdict(str(g[0]), extra_blocks)
        _spread_numbers(merged)
    # 4. Uncacheable neighbours never need to be apart: they are sent live together.
    out: list[Clause] = []
    for t, why in merged:
        if out and not out[-1].cacheable and why != _OK:
            out[-1] = Clause(out[-1].text + str(t), False)
        else:
            out.append(Clause(str(t), why == _OK))
    return out
