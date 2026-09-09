#!/usr/bin/env python3
"""Render the generated blocks of the documentation from the code itself.

Two references in ``docs/`` are restatements of things the code already knows:
the config keys (the same dataclasses ``bioaccx.gui.schema`` builds the browser
form from) and the CLI (the same Typer app ``bioaccx --help`` prints). Writing
them by hand means two sources of truth and, in practice, a documentation that
drifts behind the code.

So they are generated. Each page keeps its hand-written prose and marks the
generated part with a pair of comments::

    <!-- generated:config-reference lang=pt-BR — edit config.py / the locale, not this -->
    ...replaced by this script...
    <!-- /generated:config-reference -->

Only what is between the markers is rewritten, so the prose around them is safe
to edit.

    python tools/gen_docs.py            # rewrite every generated block
    python tools/gen_docs.py --check    # exit 1 if a block is out of date

The Portuguese config reference costs nothing extra: the field labels and help
texts are the ones ``bioaccx/gui/locales/pt-BR.json`` already carries for the
GUI, so translating a field for the form translates it for the docs too.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bioaccx.gui import i18n                      # noqa: E402
from bioaccx.gui.schema import build_schema       # noqa: E402

DOCS = ROOT / "docs"
LANGS = ("en", "pt-BR")

# Chrome for the generated tables. Everything that names a config key comes
# from the code or the locale; only these few words are written here.
STRINGS = {
    "en": {
        "key": "Key", "default": "Default", "description": "Description",
        "required": "required", "secret": "secret", "run_level": "run level",
        "secret_note": "never echoed back to the browser",
        "run_level_note": "cannot be overridden per source",
        "when": "Only when",
        "option": "Option", "command": "Command", "usage": "Usage",
        "no_help": "—",
    },
    "pt-BR": {
        "key": "Chave", "default": "Padrão", "description": "Descrição",
        "required": "obrigatório", "secret": "secreto", "run_level": "nível da execução",
        "secret_note": "nunca devolvido ao navegador",
        "run_level_note": "não pode ser sobrescrito por fonte",
        "when": "Somente quando",
        "option": "Opção", "command": "Comando", "usage": "Uso",
        "no_help": "—",
    },
}


# ── values ────────────────────────────────────────────────────────────────

def fmt_default(field: dict) -> str:
    """A config default as it would be written in YAML."""
    if field.get("required"):
        return ""
    value = field.get("default")
    if dataclasses.is_dataclass(value):
        return "`{}`"
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, (list, tuple)):
        return f"`{json.dumps(list(value))}`" if value else "`[]`"
    if isinstance(value, str):
        return f"`{value}`" if value else '`""`'
    return f"`{value}`"


def cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


# ── config reference ──────────────────────────────────────────────────────

def _translate(bundle: dict, path: str, field: dict) -> str:
    entry = bundle.get("fields", {}).get(path, {})
    return entry.get("help") or field.get("help", "")


def _rows(fields: list[dict], bundle: dict, s: dict) -> list[str]:
    rows = []
    for f in fields:
        if f.get("widget") == "nested":
            continue          # rendered as its own table below
        help_text = _translate(bundle, f["path"], f)
        badges = []
        if f.get("secret"):
            badges.append(f"**{s['secret']}** ({s['secret_note']})")
        if f.get("run_level"):
            badges.append(f"**{s['run_level']}** ({s['run_level_note']})")
        desc = " — ".join([t for t in [cell(help_text)] + badges if t]) or s["no_help"]
        default = fmt_default(f) or f"*{s['required']}*"
        rows.append(f"| `{f['name']}` | {default} | {desc} |")
    return rows


def _table(title: str, help_text: str, fields: list[dict], bundle: dict, s: dict,
           level: int = 2) -> str:
    rows = _rows(fields, bundle, s)
    if not rows:
        return ""
    head = "#" * level
    out = [f"{head} {title}", ""]
    if help_text:
        out += [help_text, ""]
    out += [f"| {s['key']} | {s['default']} | {s['description']} |", "|---|---|---|"]
    out += rows
    out.append("")
    return "\n".join(out)


def render_config(lang: str) -> str:
    schema = build_schema()
    bundle = i18n.bundle(lang)
    s = STRINGS[lang]
    parts = []

    for section in schema["sections"]:
        name = section["name"]
        loc = bundle.get("sections", {}).get(name, {})
        title = f"`{name}`"
        if loc.get("title"):
            title += f" — {loc['title']}"
        parts.append(_table(title, loc.get("help", section.get("help", "")),
                            section["fields"], bundle, s))

        for group in section.get("groups", []):
            gname = loc.get("groups", {}).get(group["name"], group["title"])
            when = group.get("visible_when")
            help_text = ""
            if when:
                values = " / ".join(f"`{v}`" for v in when["in"])
                help_text = f"*{s['when']} `{when['path']}` = {values}.*"
            parts.append(_table(f"`{name}.{group['name']}` — {gname}", help_text,
                                group["fields"], bundle, s, level=3))

        # Nested blocks (e.g. dataset.augmentation) belong to the section their
        # fields are pathed under.
        for key, block in schema.get("nested", {}).items():
            fields = block.get("fields", [])
            if not fields or not fields[0]["path"].startswith(f"{name}."):
                continue
            prefix = fields[0]["path"].rsplit(".", 1)[0]
            gtitle = bundle.get("nested", {}).get(key, {}).get("title", block.get("title", key))
            parts.append(_table(f"`{prefix}` — {gtitle}", "", fields, bundle, s, level=3))

    return "\n".join(p for p in parts if p).rstrip()


# ── CLI reference ─────────────────────────────────────────────────────────

def render_cli(lang: str = "en") -> str:
    import click
    import typer.main

    from bioaccx.cli import app as typer_app

    s = STRINGS[lang]
    group = typer.main.get_command(typer_app)
    ctx = click.Context(group, info_name="bioaccx")
    parts = []

    # Declaration order, not alphabetical: cli.py lists the commands in the
    # order a run uses them, and that is the useful order to read them in.
    names = list(getattr(group, "commands", {})) or group.list_commands(ctx)
    for cmd_name in names:
        cmd = group.get_command(ctx, cmd_name)
        if cmd is None or cmd.hidden:
            continue
        sub = click.Context(cmd, info_name=f"bioaccx {cmd_name}", parent=ctx)
        usage = cmd.collect_usage_pieces(sub)
        parts.append(f"### `bioaccx {cmd_name}`\n")
        parts.append(f"```\nbioaccx {cmd_name} {' '.join(usage)}\n```\n")
        if cmd.help:
            parts.append(inspect_help(cmd.help) + "\n")

        rows = []
        for param in cmd.get_params(sub):
            if isinstance(param, click.Argument):
                names = f"`{param.make_metavar(sub) if _takes_ctx(param) else param.make_metavar()}`"
            else:
                if param.name == "help":
                    continue
                names = ", ".join(f"`{o}`" for o in param.opts + param.secondary_opts)
            help_text = cell(getattr(param, "help", "") or "")
            default = ""
            if not isinstance(param, click.Argument) and param.default not in (None, False):
                default = f"`{param.default}`"
            rows.append(f"| {names} | {default} | {help_text or s['no_help']} |")
        if rows:
            parts.append(f"| {s['option']} | {s['default']} | {s['description']} |")
            parts.append("|---|---|---|")
            parts += rows
            parts.append("")

    return "\n".join(parts).rstrip()


def _takes_ctx(param) -> bool:
    """click >= 8.2 requires a context for ``make_metavar``; 8.1 does not."""
    import inspect as _inspect
    return bool(_inspect.signature(param.make_metavar).parameters)


def inspect_help(text: str) -> str:
    """A command's docstring as a paragraph: rewrapped, escape sequences gone."""
    return re.sub(r"\s*\n\s*", " ", text.replace("\b", "")).strip()


# ── block replacement ─────────────────────────────────────────────────────

BLOCK = re.compile(
    r"(?P<open><!-- generated:(?P<name>[\w-]+)[^>]*-->\n)"
    r".*?"
    r"(?P<close><!-- /generated:(?P=name) -->)",
    re.S,
)

RENDERERS = {"config-reference": render_config, "cli": render_cli}


def lang_of(path: Path) -> str:
    """The language a page is written in, from ``docs/<lang>/``."""
    return path.parent.name if path.parent.parent == DOCS else "en"


def process(path: Path) -> tuple[str, list[str]]:
    text = path.read_text(encoding="utf-8")
    names: list[str] = []

    def repl(m: re.Match) -> str:
        name = m.group("name")
        render = RENDERERS.get(name)
        if render is None:
            raise SystemExit(f"{path}: unknown generated block {name!r}")
        names.append(name)
        return f"{m.group('open')}\n{render(lang_of(path))}\n\n{m.group('close')}"

    return BLOCK.sub(repl, text), names


# ── translations ──────────────────────────────────────────────────────────

STAMP = re.compile(r"<!-- translated-from: (?P<src>[\w/.\-]+)@(?P<hash>\w+) -->")
PENDING = "pending"


def digest(path: Path) -> str:
    """Short content hash of a page, as stamped into its translations."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def translated_pages() -> list[Path]:
    """Every page that is a translation of another one, stamp or not.

    That is ``docs/<lang>/*.md`` for every language but English, plus the
    ``README.<lang>.md`` files. ``docs/README.md`` is an index, not a
    translation, so it stays out.
    """
    pages = [p for p in sorted(DOCS.glob("*/*.md")) if p.parent.name != "en"]
    return pages + sorted(ROOT.glob("README.*.md"))


def stale_translations() -> list[tuple[Path, str]]:
    """Translated pages whose English source has moved on since (or that are
    missing the stamp that would let anyone tell)."""
    out = []
    for path in translated_pages():
        text = path.read_text(encoding="utf-8")
        match = STAMP.search(text)
        if match is None:
            if BLOCK.search(text):
                continue           # generated: it is rebuilt, never translated
            out.append((path, "no translated-from stamp"))
            continue
        source = ROOT / match["src"]
        if not source.exists():
            out.append((path, f"source is gone: {match['src']}"))
        elif match["hash"] == PENDING:
            out.append((path, "stamp never taken (run --stamp)"))
        elif match["hash"] != digest(source):
            out.append((path, f"{match['src']} changed since this was translated"))
    return out


def stamp_translations() -> list[Path]:
    """Record the current English content in every translation's stamp."""
    touched = []
    for path in translated_pages():
        text = path.read_text(encoding="utf-8")
        match = STAMP.search(text)
        if match is None:
            continue
        source = ROOT / match["src"]
        if not source.exists():
            continue
        new = f"<!-- translated-from: {match['src']}@{digest(source)} -->"
        if new != match.group(0):
            path.write_text(text.replace(match.group(0), new, 1), encoding="utf-8")
            touched.append(path.relative_to(ROOT))
    return touched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if a block or a translation stamp "
                         "is out of date")
    ap.add_argument("--stamp", action="store_true",
                    help="record the current English pages in the translations' "
                         "translated-from stamps")
    args = ap.parse_args()

    if args.stamp:
        for path in stamp_translations():
            print(f"stamped {path}")
        return 0

    stale = []
    for path in sorted(DOCS.rglob("*.md")):
        new, names = process(path)
        if not names:
            continue
        if new == path.read_text(encoding="utf-8"):
            continue
        stale.append(path.relative_to(ROOT))
        if not args.check:
            path.write_text(new, encoding="utf-8")

    if args.check:
        behind = stale_translations()
        if stale:
            print("out of date (run: python tools/gen_docs.py):", file=sys.stderr)
            for p in stale:
                print(f"  {p}", file=sys.stderr)
        if behind:
            print("translations behind their source:", file=sys.stderr)
            for p, why in behind:
                print(f"  {p.relative_to(ROOT)}: {why}", file=sys.stderr)
        return 1 if (stale or behind) else 0

    for p in stale:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
