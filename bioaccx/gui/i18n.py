"""Interface languages for the GUI.

Two ship with the editor — ``en`` and ``pt-BR`` — one JSON file each under
``locales/``. Dropping another JSON in that folder is all a third takes.

A locale file only has to carry what it translates: :func:`bundle` merges it
over English, so a partial translation shows English for the rest rather than
a missing key. That is also why ``en.json`` holds no ``sections`` or ``fields``
block — the English labels and help text are generated from the config
dataclasses (see :mod:`bioaccx.gui.schema`), and copying them here would be a
second source of truth to keep in step.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LOCALE_DIR = Path(__file__).parent / "locales"
DEFAULT_LANG = "en"


@lru_cache(maxsize=None)
def available() -> tuple[str, ...]:
    """Language codes with a locale file, the default first.

    >>> available()[0]
    'en'
    """
    codes = sorted(p.stem for p in LOCALE_DIR.glob("*.json"))
    if DEFAULT_LANG in codes:
        codes.remove(DEFAULT_LANG)
        codes.insert(0, DEFAULT_LANG)
    return tuple(codes)


def _read(lang: str) -> dict:
    path = LOCALE_DIR / f"{lang}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _merge(base: dict, over: dict) -> dict:
    """*over* wins, key by key, recursing into nested objects."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=None)
def bundle(lang: str) -> dict:
    """The translation bundle for *lang*, over English.

    >>> bundle("pt-BR")["bar"]["save"]
    'Salvar'
    >>> bundle("pt-BR")["meta"]["name"]
    'Português (Brasil)'
    """
    english = _read(DEFAULT_LANG)
    return english if lang == DEFAULT_LANG else _merge(english, _read(lang))


def normalize(lang: str | None) -> str:
    """Best available match for a requested language tag.

    >>> normalize("pt"), normalize("PT-br"), normalize("de"), normalize(None)
    ('pt-BR', 'pt-BR', 'en', 'en')
    """
    if not lang:
        return DEFAULT_LANG
    codes = available()
    if lang in codes:
        return lang
    lower = lang.lower()
    for code in codes:
        if code.lower() == lower:
            return code
    base = lower.split("-")[0]
    for code in codes:
        if code.lower().split("-")[0] == base:
            return code
    return DEFAULT_LANG


def language_names() -> list[dict[str, str]]:
    """``[{code, name}]`` for the language picker, in each language's own name.

    >>> language_names()[0]
    {'code': 'en', 'name': 'English'}
    """
    return [
        {"code": code, "name": bundle(code).get("meta", {}).get("name", code)}
        for code in available()
    ]
