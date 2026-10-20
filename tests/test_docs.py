"""The docs check CONTRIBUTING.md promises: ``docs/`` stays publishable by the docs hub.

The hub fetches ``docs/`` and nothing else, so every page needs its front matter,
links between pages must stay inside the folder, and anything outside it must be a
full GitHub URL.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPO_URL = "https://github.com/imumesh18/memcat/"
REPO_FILE = re.compile(r"https://github\.com/imumesh18/memcat/(?:blob|tree)/main/([^#?]+)")

_FENCE = re.compile(r"^\s*(```|~~~)")
_INLINE_CODE = re.compile(r"`[^`]*`")
_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

PAGES = sorted(DOCS.glob("*.md"))


def _split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """The front matter as a flat dict (nested keys joined by dots) and the body."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, text
    meta: dict[str, object] = {}
    parent = ""
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indented = raw[0] in " \t"
        key, sep, value = raw.strip().partition(":")
        assert sep, f"front matter line without a key: {raw!r}"
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not indented:
            parent = key if not value else ""
            if value:
                meta[key] = value
        else:
            assert parent, f"indented front matter line outside a mapping: {raw!r}"
            meta[f"{parent}.{key}"] = value
    return meta, "\n".join(lines[end + 1 :])


def _prose_lines(body: str) -> list[str]:
    """Body lines outside fenced code blocks, with inline code removed."""
    out: list[str] = []
    fence: str | None = None
    for line in body.split("\n"):
        m = _FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif m.group(1) == fence:
                fence = None
            continue
        if fence is None:
            out.append(_INLINE_CODE.sub("", line))
    return out


def _slug(heading: str) -> str:
    """GitHub-style anchor: lowercase, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links keep their text
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    _, body = _split_front_matter(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    fence: str | None = None
    for line in body.split("\n"):
        m = _FENCE.match(line)
        if m:
            fence = m.group(1) if fence is None else (None if m.group(1) == fence else fence)
            continue
        if fence is None and (h := re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)):
            out.add(_slug(h.group(1)))
    return out


def _links(path: Path) -> list[str]:
    _, body = _split_front_matter(path.read_text(encoding="utf-8"))
    return [m.group(1) for line in _prose_lines(body) for m in _LINK.finditer(line)]


def test_docs_folder_holds_only_markdown_pages() -> None:
    assert PAGES, "docs/ has no pages"
    others = sorted(
        str(p.relative_to(DOCS)) for p in DOCS.rglob("*") if p.is_file() and p.suffix != ".md"
    )
    nested = sorted(str(p.relative_to(DOCS)) for p in DOCS.rglob("*.md") if p.parent != DOCS)
    assert not others, f"docs/ may hold only .md pages (the hub fetches nothing else): {others}"
    assert not nested, f"docs/ pages must sit directly in docs/: {nested}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_front_matter(page: Path) -> None:
    meta, _ = _split_front_matter(page.read_text(encoding="utf-8"))
    assert meta, f"{page.name}: no front matter (--- ... --- at the top)"
    for key in ("title", "description", "sidebar.order"):
        assert str(meta.get(key, "")).strip(), f"{page.name}: front matter has no {key}"
    assert "\n" not in str(meta["description"]), f"{page.name}: description is one line"
    assert re.fullmatch(r"\d+", str(meta["sidebar.order"])), (
        f"{page.name}: sidebar.order must be a whole number"
    )


def test_sidebar_orders_are_unique_and_index_comes_first() -> None:
    orders: dict[int, str] = {}
    for page in PAGES:
        meta, _ = _split_front_matter(page.read_text(encoding="utf-8"))
        order = int(str(meta.get("sidebar.order", "-1")))
        assert order not in orders, f"{page.name} and {orders[order]} share sidebar.order {order}"
        orders[order] = page.name
    assert orders[min(orders)] == "index.md", "index.md must have the lowest sidebar.order"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_one_h1_equal_to_title(page: Path) -> None:
    meta, body = _split_front_matter(page.read_text(encoding="utf-8"))
    h1s = [line[2:].strip() for line in _prose_lines(body) if line.startswith("# ")]
    assert len(h1s) == 1, f"{page.name}: expected exactly one h1, found {h1s}"
    assert h1s[0] == meta.get("title"), f"{page.name}: h1 {h1s[0]!r} != title {meta.get('title')!r}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_links(page: Path) -> None:
    problems: list[str] = []
    for target in _links(page):
        if target.startswith("#"):
            if target[1:] not in _anchors(page):
                problems.append(f"{target}: no such heading on this page")
            continue
        if _SCHEME.match(target) or target.startswith("//"):
            m = REPO_FILE.match(target)
            if m and not (ROOT / m.group(1).rstrip("/")).exists():
                problems.append(f"{target}: {m.group(1)} does not exist in the repository")
            continue
        path, _, anchor = target.partition("#")
        if path.startswith("/"):
            problems.append(f"{target}: use a relative link (./page.md), not a site path")
            continue
        dest = (page.parent / path).resolve()
        if dest.parent != DOCS:
            problems.append(f"{target}: leaves docs/; use a full {REPO_URL}blob/main/... URL")
            continue
        if dest.suffix != ".md" or not dest.is_file():
            problems.append(f"{target}: no such page in docs/")
            continue
        if anchor and anchor not in _anchors(dest):
            problems.append(f"{target}: {dest.name} has no heading #{anchor}")
    assert not problems, f"{page.name}:\n  " + "\n  ".join(problems)
