"""Tests for the GUI's interface languages.

A locale file is data the browser trusts blindly: a key that is missing shows
the key itself, and a key that no longer exists in the schema translates
nothing at all. Neither breaks anything loudly, so they are caught here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bioaccx.gui import i18n
from bioaccx.gui.schema import build_schema

WEB = Path(__file__).resolve().parents[2] / "bioaccx" / "gui" / "web"
HTML = (WEB / "index.html").read_text()
SCRIPTS = {path.name: path.read_text() for path in WEB.glob("*.js")}


def _raw(lang: str) -> dict:
    return json.loads((i18n.LOCALE_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def _flatten(data: dict, prefix: str = "") -> set[str]:
    """Every dotted key in a bundle that leads to a string."""
    keys = set()
    for name, value in data.items():
        path = f"{prefix}{name}"
        if isinstance(value, dict):
            keys |= _flatten(value, f"{path}.")
        elif isinstance(value, str):
            keys.add(path)
    return keys


def _schema_field_paths() -> set[str]:
    schema = build_schema()
    paths = set()
    for section in schema["sections"]:
        paths |= {f["path"] for f in section["fields"]}
        for group in section.get("groups", []):
            paths |= {f["path"] for f in group["fields"]}
    for nested in schema["nested"].values():
        paths |= {f["path"] for f in nested["fields"]}
    return paths


class TestLookup:
    def test_both_languages_ship(self):
        assert set(i18n.available()) == {"en", "pt-BR"}
        assert i18n.available()[0] == "en", "the default language leads the picker"

    @pytest.mark.parametrize("asked,expected", [
        ("pt-BR", "pt-BR"), ("pt", "pt-BR"), ("PT-br", "pt-BR"), ("pt-PT", "pt-BR"),
        ("en", "en"), ("de", "en"), ("", "en"), (None, "en"),
    ])
    def test_normalize(self, asked, expected):
        assert i18n.normalize(asked) == expected

    def test_language_names_are_in_their_own_language(self):
        names = {entry["code"]: entry["name"] for entry in i18n.language_names()}
        assert names == {"en": "English", "pt-BR": "Português (Brasil)"}

    def test_a_bundle_is_merged_over_english(self):
        """A locale only has to carry what it translates."""
        bundle = i18n.bundle("pt-BR")
        assert bundle["bar"]["save"] == "Salvar"
        # Not translated in pt-BR.json — the acronym is the same word.
        assert bundle["metrics"]["col_auprc"] == _raw("en")["metrics"]["col_auprc"]

    def test_an_unknown_language_reads_as_english(self):
        assert i18n.bundle("en") == _raw("en")


class TestTranslationsMatchTheApplication:
    """The keys the front end asks for, and the fields it names."""

    def test_every_locale_key_exists_in_english(self):
        """English is the source; anything else is a typo or a stale key."""
        english = _flatten(_raw("en")) | {"meta.name", "meta.code"}
        for lang in i18n.available():
            if lang == i18n.DEFAULT_LANG:
                continue
            extra = {key for key in _flatten(_raw(lang))
                     if not key.startswith(("fields.", "sections.", "nested."))}
            assert extra <= english, f"{lang}.json has keys English does not: {extra - english}"

    def test_markup_only_asks_for_keys_english_has(self):
        english = _flatten(_raw("en"))
        asked = set(re.findall(r'data-i18n(?:-title|-ph|-aria|-alt)?="([^"]+)"', HTML))
        assert asked <= english, f"index.html asks for missing keys: {asked - english}"

    @pytest.mark.parametrize("name", sorted(SCRIPTS))
    def test_scripts_only_ask_for_keys_english_has(self, name):
        """Only the literal `t("…")` keys; the computed ones are covered below."""
        english = _flatten(_raw("en"))
        asked = set(re.findall(r'\bt\(\s*"([a-z_0-9.]+)"', SCRIPTS[name]))
        assert asked <= english, f"{name} asks for missing keys: {asked - english}"

    @pytest.mark.parametrize("prefix,keys", [
        ("run_state", {"done", "failed", "cancelled"}),
        ("validation", {"backbone", "label_mode", "classifier", "sources", "output_dir"}),
        ("theme", {"system", "light", "dark"}),
    ])
    def test_computed_key_families_are_complete(self, prefix, keys):
        """Keys the front end builds as `t(`{prefix}.${value}`)`."""
        english = _raw("en")[prefix]
        assert keys <= set(english), f"{prefix} is missing {keys - set(english)}"

    def test_metric_columns_and_facts_are_named(self):
        english = _raw("en")["metrics"]
        columns = re.search(r"const COLUMNS = \[(.*?)\];",
                            SCRIPTS["explorer.js"], re.S).group(1)
        for key in re.findall(r'"([a-z_0-9]+)"', columns):
            assert f"col_{key}" in english, f"no metrics.col_{key} for the {key} column"

    def test_field_help_keys_name_real_config_fields(self):
        """A renamed config key must not leave a translation pointing nowhere."""
        paths = _schema_field_paths()
        for lang in i18n.available():
            translated = set(_raw(lang).get("fields", {}))
            assert translated <= paths, \
                f"{lang}.json translates fields that no longer exist: {translated - paths}"

    def test_pt_br_labels_every_field(self):
        translated = _raw("pt-BR")["fields"]
        missing = sorted(path for path in _schema_field_paths()
                         if "label" not in translated.get(path, {}))
        assert not missing, f"untranslated field labels: {missing}"

    def test_pt_br_translates_every_help_text(self):
        schema = build_schema()
        with_help = set()
        for section in schema["sections"]:
            for field in section["fields"]:
                if field.get("help"):
                    with_help.add(field["path"])
            for group in section.get("groups", []):
                with_help |= {f["path"] for f in group["fields"] if f.get("help")}
        for nested in schema["nested"].values():
            with_help |= {f["path"] for f in nested["fields"] if f.get("help")}

        translated = {path for path, entry in _raw("pt-BR")["fields"].items()
                      if "help" in entry}
        assert with_help <= translated, \
            f"untranslated help text: {sorted(with_help - translated)}"

    def test_the_sources_block_reuses_the_field_labels(self):
        """Its four rows are built by hand, so they can drift from the fields."""
        pt = _raw("pt-BR")
        for row, path in [("source_data_dir", "dataset.data_dir"),
                          ("source_label_mode", "dataset.label_mode"),
                          ("source_table_file", "dataset.table_file"),
                          ("source_augmentation", "dataset.augmentation")]:
            assert pt["form"][row] == pt["fields"][path]["label"], \
                f"form.{row} disagrees with the label of {path}"

    def test_pt_br_names_every_section(self):
        schema = build_schema()
        translated = _raw("pt-BR")["sections"]
        for section in schema["sections"]:
            assert section["name"] in translated
            assert {"title", "help"} <= set(translated[section["name"]])
            for group in section.get("groups", []):
                assert group["name"] in translated[section["name"]]["groups"]

    def test_placeholders_survive_translation(self):
        """A translated string must keep the {names} the caller substitutes."""
        english = _raw("en")
        for lang in i18n.available():
            bundle = i18n.bundle(lang)
            for key in _flatten(english):
                node_en, node_lang = english, bundle
                for part in key.split("."):
                    node_en, node_lang = node_en[part], node_lang[part]
                assert (set(re.findall(r"\{(\w+)\}", node_en))
                        == set(re.findall(r"\{(\w+)\}", node_lang))), \
                    f"{lang}.json changed the placeholders of {key}"
