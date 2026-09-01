"""Tests for the form schema generated from the config dataclasses.

The point of generating the schema rather than writing it is that it cannot
drift from ``config.py``. These tests are what makes that true: a field added
with a type the front end has no widget for fails here rather than rendering as
a broken control in the browser.
"""
from __future__ import annotations

import dataclasses

import pytest

from bioaccx.config import (
    AugmentationConfig,
    DatasetConfig,
    FoundationModelConfig,
    KerasConfig,
    OutputConfig,
    SklearnConfig,
    UmapConfig,
)
from bioaccx.gui.schema import build_schema, known_widgets


@pytest.fixture(scope="module")
def schema():
    return build_schema()


def _all_fields(schema) -> list[dict]:
    out = []
    for section in schema["sections"]:
        out.extend(section["fields"])
        for group in section.get("groups", []):
            out.extend(group["fields"])
    for nested in schema["nested"].values():
        out.extend(nested["fields"])
    return out


class TestCoverage:
    def test_every_field_has_a_widget_the_front_end_implements(self, schema):
        unknown = {f["widget"] for f in _all_fields(schema)} - known_widgets()
        assert not unknown, f"no widget implemented for: {unknown}"

    def test_no_field_falls_back_to_raw(self, schema):
        """A `raw` widget means a type the mapping did not recognise."""
        raw = [f["path"] for f in _all_fields(schema) if f["widget"] == "raw"]
        assert not raw, f"unmapped types fell back to a raw YAML box: {raw}"

    @pytest.mark.parametrize("cls,section", [
        (FoundationModelConfig, "foundation_model"),
        (DatasetConfig, "dataset"),
        (KerasConfig, "training.keras"),
        (SklearnConfig, "training.sklearn"),
        (OutputConfig, "output"),
        (UmapConfig, "umap"),
        (AugmentationConfig, "dataset.augmentation"),
    ])
    def test_every_public_dataclass_field_is_offered(self, schema, cls, section):
        paths = {f["path"] for f in _all_fields(schema)}
        for f in dataclasses.fields(cls):
            if f.name.startswith("_"):
                continue
            assert f"{section}.{f.name}" in paths

    def test_private_fields_are_not_offered(self, schema):
        assert not [f for f in _all_fields(schema) if f["name"].startswith("_")]


class TestDescriptors:
    def _find(self, schema, path):
        return next(f for f in _all_fields(schema) if f["path"] == path)

    def test_literal_becomes_a_select_with_choices(self, schema):
        field = self._find(schema, "training.keras.upsampling_mode")
        assert field["widget"] == "select"
        assert field["choices"] == ["repeat", "mean", "linear", "smote"]

    def test_bool_becomes_a_checkbox(self, schema):
        assert self._find(schema, "umap.enabled")["widget"] == "checkbox"

    def test_numbers_carry_their_default(self, schema):
        field = self._find(schema, "training.keras.dropout")
        assert field["widget"] == "number"
        assert field["default"] == 0.25

    def test_list_becomes_tags(self, schema):
        assert self._find(schema, "output.exclude_labels")["widget"] == "tags"

    def test_union_of_str_and_list_becomes_the_paths_widget(self, schema):
        assert self._find(schema, "dataset.data_dir")["widget"] == "paths"

    def test_scalar_or_pair_becomes_the_freq_widget(self, schema):
        assert self._find(schema, "dataset.filter_freq")["widget"] == "freq"

    def test_dict_of_lists_becomes_the_groups_widget(self, schema):
        assert self._find(schema, "training.keras.label_groups")["widget"] == "groups"

    def test_optional_nested_dataclass(self, schema):
        field = self._find(schema, "dataset.augmentation")
        assert field["widget"] == "nested"
        assert field["dataclass"] == "AugmentationConfig"

    def test_a_field_without_a_default_is_required(self, schema):
        assert self._find(schema, "foundation_model.name")["required"] is True

    def test_a_field_with_a_default_is_not_required(self, schema):
        assert self._find(schema, "foundation_model.version")["required"] is False

    def test_help_is_read_from_the_source_comments(self, schema):
        assert "ext_table_file" in self._find(schema, "dataset.data_dir")["help"]

    def test_defaults_from_a_factory_are_evaluated(self, schema):
        assert self._find(schema, "dataset.audio_extensions")["default"] == [
            "wav", "flac", "mp3", "ogg"]


class TestSafetyFlags:
    def _find(self, schema, path):
        return next(f for f in _all_fields(schema) if f["path"] == path)

    def test_credentials_are_marked_secret(self, schema):
        assert self._find(schema, "dataset.xc_api_key")["secret"] is True

    def test_ordinary_fields_are_not_secret(self, schema):
        assert self._find(schema, "dataset.test_ratio")["secret"] is False

    def test_secret_set_matches_the_metadata_redaction_list(self, schema):
        from bioaccx.report import _SECRET_DATASET_FIELDS
        flagged = {f["name"] for f in _all_fields(schema) if f["secret"]}
        assert flagged == set(_SECRET_DATASET_FIELDS)

    def test_run_level_dataset_fields_are_flagged(self, schema):
        from bioaccx.config import RUN_LEVEL_DATASET_FIELDS
        flagged = {f["name"] for f in _all_fields(schema)
                   if f["path"].startswith("dataset.") and f.get("run_level")}
        assert flagged == set(RUN_LEVEL_DATASET_FIELDS)


class TestSyntheticKeys:
    """Keys the loader consumes before the dataclasses exist."""

    def test_registry_id_leads_the_foundation_model_section(self, schema):
        first = schema["sections"][0]["fields"][0]
        assert first["name"] == "registry_id"
        assert first["widget"] == "registry"

    def test_registry_choices_carry_defaults_to_apply(self, schema):
        choices = schema["sections"][0]["fields"][0]["choices"]
        ids = [c["id"] for c in choices]
        assert "0xbb00" in ids
        birdnet = next(c for c in choices if c["id"] == "0xbb00")
        assert birdnet["defaults"]["embedding_size"] == 1024
        assert birdnet["description"]

    def test_sources_is_offered_on_the_dataset_section(self, schema):
        dataset = next(s for s in schema["sections"] if s["name"] == "dataset")
        sources = next(f for f in dataset["fields"] if f["name"] == "sources")
        assert sources["widget"] == "sources"

    def test_registry_id_is_not_a_dataclass_field(self):
        """If it ever becomes one, the synthetic descriptor must be removed."""
        names = {f.name for f in dataclasses.fields(FoundationModelConfig)}
        assert "registry_id" not in names
