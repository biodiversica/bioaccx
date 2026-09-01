"""Checks on the browser assets.

There is no bundler and no type checker in front of this code, so a renamed
element id or a stale import path would only show up as a blank panel in
someone's browser. These read the files and catch that here instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "bioaccx" / "gui" / "web"
HTML = (WEB / "index.html").read_text()
CSS = (WEB / "styles.css").read_text()
SCRIPTS = {path.name: path.read_text() for path in WEB.glob("*.js")}


def _ids_in_html() -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', HTML))


def _ids_used_by(script: str) -> set[str]:
    """Element ids the script looks up, via `$("x")` or getElementById("x")."""
    return set(re.findall(r'(?:\$|getElementById)\(\s*["\']([A-Za-z0-9_-]+)["\']\s*\)',
                          script))


class TestHiddenElementsStayHidden:
    """The regression that stacked the editor and the explorer on one screen.

    `hidden` comes from the user-agent stylesheet, and an author rule setting
    `display` beats it on origin — so `.split { display: grid }` kept
    `<main hidden>` on screen. Every view switch in this UI is the `hidden`
    attribute, so the override has to be there.
    """

    def test_the_stylesheet_forces_hidden_to_win(self):
        assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important",
                         CSS), "styles.css must force [hidden] to override display"

    @pytest.mark.parametrize("element_id", ["explorer", "run-progress"])
    def test_elements_that_set_display_are_still_hideable(self, element_id):
        """These carry a class with its own `display`, so they need the override."""
        assert re.search(rf'id="{element_id}"[^>]*\bhidden\b|'
                         rf'\bhidden\b[^>]*id="{element_id}"', HTML)

    def test_the_two_views_start_with_only_one_visible(self):
        editor = re.search(r'<main[^>]*id="editor"[^>]*>', HTML).group(0)
        explorer = re.search(r'<main[^>]*id="explorer"[^>]*>', HTML).group(0)
        assert "hidden" not in editor
        assert "hidden" in explorer


class TestScriptsMatchTheMarkup:
    @pytest.mark.parametrize("name", sorted(SCRIPTS))
    def test_every_element_the_script_looks_up_exists(self, name):
        missing = _ids_used_by(SCRIPTS[name]) - _ids_in_html()
        assert not missing, f"{name} looks up ids that are not in index.html: {missing}"

    def test_every_module_import_resolves(self):
        for name, source in SCRIPTS.items():
            for target in re.findall(r'from\s+["\']/static/([^"\']+)["\']', source):
                assert (WEB / target).exists(), f"{name} imports missing {target}"

    def test_the_page_loads_the_entry_point(self):
        assert '<script type="module" src="/static/app.js">' in HTML

    def test_ids_are_unique(self):
        ids = re.findall(r'\bid="([^"]+)"', HTML)
        duplicates = {name for name in ids if ids.count(name) > 1}
        assert not duplicates, f"duplicate ids in index.html: {duplicates}"


class TestStylesheetIsSelfConsistent:
    def test_braces_balance(self):
        assert CSS.count("{") == CSS.count("}")

    def test_every_colour_comes_from_a_token(self):
        """A literal hex outside :root would not follow the dark theme."""
        body = CSS.split("}", 1)[1] if CSS.startswith(":root") else CSS
        # Token definitions live in :root and the dark-scheme block; everything
        # after that should reference var(--…) instead of naming a colour.
        after_tokens = body.split("* { box-sizing: border-box; }", 1)[-1]
        literals = re.findall(r":\s*(#[0-9a-fA-F]{3,8})\s*[;}]", after_tokens)
        assert not literals, f"hard-coded colours outside the token block: {literals}"
