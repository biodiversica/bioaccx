"""The documentation tree: generated blocks, links, and translation stamps.

``docs/`` is plain Markdown, so nothing renders it and nothing would notice a
link that points at a heading that moved, a generated table that no longer
matches the code, or a translation left behind by an edit to the English page.
These tests are what notices.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import gen_docs  # noqa: E402

PAGES = sorted(ROOT.joinpath("docs").rglob("*.md")) + sorted(ROOT.glob("README*.md"))

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6}) (.+?)\s*$", re.M)
FENCE = re.compile(r"^```.*?^```", re.M | re.S)


def slug(text: str) -> str:
    """GitHub's heading anchor for *text*."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def anchors_of(path: Path) -> set[str]:
    body = FENCE.sub("", path.read_text(encoding="utf-8"))
    return {slug(m.group(2)) for m in HEADING.finditer(body)}


def links_of(path: Path) -> list[str]:
    body = FENCE.sub("", path.read_text(encoding="utf-8"))
    return [t for t in LINK.findall(body)
            if not t.startswith(("http://", "https://", "mailto:"))]


def ids(paths):
    return [str(p.relative_to(ROOT)) for p in paths]


class TestGeneratedBlocks:
    def test_generated_blocks_match_the_code(self):
        """`tools/gen_docs.py` output is committed; it must not be stale."""
        stale = [p for p in PAGES
                 if gen_docs.BLOCK.search(p.read_text(encoding="utf-8"))
                 and gen_docs.process(p)[0] != p.read_text(encoding="utf-8")]
        assert not stale, (f"out of date, run 'python tools/gen_docs.py': "
                           f"{ids(stale)}")

    def test_every_config_key_reaches_the_reference(self):
        """A new config field must show up in the reference, in both languages."""
        from bioaccx.gui.schema import build_schema

        names = set()
        for section in build_schema()["sections"]:
            names |= {f["name"] for f in section["fields"] if f["widget"] != "nested"}
        for lang in ("en", "pt-BR"):
            page = ROOT / f"docs/{lang}/config-reference.md"
            text = page.read_text(encoding="utf-8")
            missing = sorted(n for n in names if f"| `{n}` |" not in text)
            assert not missing, f"{page.name} ({lang}) is missing: {missing}"


class TestLinks:
    @pytest.mark.parametrize("page", PAGES, ids=ids(PAGES))
    def test_relative_links_resolve(self, page):
        for target in links_of(page):
            file_part, _, anchor = target.partition("#")
            dest = (page.parent / file_part).resolve() if file_part else page
            assert dest.exists(), f"{page.name} links to a missing {file_part}"
            if anchor and dest.suffix == ".md":
                assert anchor in anchors_of(dest), \
                    f"{page.name} links to {file_part}#{anchor}, which has no such heading"


class TestTranslations:
    def test_translations_are_not_behind_their_source(self):
        """Each translated page stamps the English page it was written from."""
        behind = gen_docs.stale_translations()
        assert not behind, "\n".join(
            f"{p.relative_to(ROOT)}: {why}" for p, why in behind)
