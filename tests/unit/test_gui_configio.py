"""Tests for comment-preserving config read/write.

The editor's whole value rests on being able to open an existing config, change
two fields and save it back without destroying the documentation around them —
so that is what these check, against the real ``example_config.yaml``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bioaccx.gui import configio

EXAMPLE = Path(__file__).resolve().parents[2] / "example_config.yaml"


class TestRoundTrip:
    def test_saving_an_untouched_config_changes_nothing(self):
        original = EXAMPLE.read_text()
        assert configio.dumps(configio.load(EXAMPLE)) == original

    def test_comments_survive_an_edit(self):
        original = EXAMPLE.read_text()
        doc = configio.load(EXAMPLE)
        configio.apply_edits(doc, {"output.model_name": "edited"})
        out = configio.dumps(doc)

        def comment_lines(text):
            return [line.strip() for line in text.splitlines()
                    if line.strip().startswith("#")]

        assert comment_lines(out) == comment_lines(original)
        assert "edited" in out

    def test_explicit_null_is_not_rewritten(self):
        """`output_activation: null` is documented; a save must not blank it."""
        doc = configio.loads("training:\n  keras:\n    output_activation: null\n")
        assert "output_activation: null" in configio.dumps(doc)

    def test_edited_config_still_loads_through_the_real_loader(self, tmp_path):
        from bioaccx.config import load_config
        doc = configio.load(EXAMPLE)
        configio.apply_edits(doc, {"output.model_name": "roundtripped",
                                   "training.keras.epochs": 7})
        target = tmp_path / "edited.yaml"
        configio.save(target, doc)
        cfg = load_config(str(target))
        assert cfg.output.model_name == "roundtripped"
        assert cfg.training.keras.epochs == 7

    def test_json_configs_round_trip_as_json(self, tmp_path):
        target = tmp_path / "cfg.json"
        configio.save(target, {"output": {"model_name": "x"}})
        assert configio.load(target) == {"output": {"model_name": "x"}}
        assert target.read_text().lstrip().startswith("{")


class TestApplyEdits:
    def test_sets_a_nested_value(self):
        assert configio.apply_edits({}, {"a.b.c": 1}) == {"a": {"b": {"c": 1}}}

    def test_none_removes_the_key_rather_than_writing_null(self):
        doc = {"output": {"model_name": "x", "model_version": "1"}}
        configio.apply_edits(doc, {"output.model_name": None})
        assert doc == {"output": {"model_version": "1"}}

    def test_removing_an_absent_key_is_harmless(self):
        assert configio.apply_edits({}, {"a.b": None}) == {"a": {}}

    def test_existing_siblings_are_left_alone(self):
        doc = configio.loads("output:\n  # keep me\n  model_name: a\n  model_version: '1'\n")
        configio.apply_edits(doc, {"output.model_version": "2"})
        out = configio.dumps(doc)
        assert "# keep me" in out and "model_name: a" in out and "model_version: '2'" in out

    def test_a_scalar_in_the_path_is_replaced_by_a_mapping(self):
        doc = {"dataset": "not-a-mapping"}
        configio.apply_edits(doc, {"dataset.overlap": 0.5})
        assert doc == {"dataset": {"overlap": 0.5}}


class TestRedact:
    def test_a_set_secret_becomes_a_presence_marker(self):
        doc = {"dataset": {"xc_api_key": "hunter2", "test_ratio": 0.2}}
        out = configio.redact(doc, ["dataset.xc_api_key"])
        assert out["dataset"]["xc_api_key"] == "__SET__"
        assert out["dataset"]["test_ratio"] == 0.2

    def test_an_unset_secret_is_left_alone(self):
        assert configio.redact({"dataset": {}}, ["dataset.xc_api_key"]) == {"dataset": {}}

    def test_the_original_document_is_not_modified(self):
        doc = {"dataset": {"xc_api_key": "hunter2"}}
        configio.redact(doc, ["dataset.xc_api_key"])
        assert doc["dataset"]["xc_api_key"] == "hunter2"

    def test_a_missing_section_is_harmless(self):
        assert configio.redact({}, ["dataset.xc_api_key"]) == {}


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(configio.ConfigIOError, match="no such config"):
            configio.load(tmp_path / "nope.yaml")

    def test_malformed_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("a:\n  - [unclosed\n")
        with pytest.raises(configio.ConfigIOError, match="not valid YAML"):
            configio.load(bad)

    def test_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{nope")
        with pytest.raises(configio.ConfigIOError, match="not valid JSON"):
            configio.load(bad)

    def test_refuses_to_write_an_unknown_suffix(self, tmp_path):
        with pytest.raises(configio.ConfigIOError, match="must end in"):
            configio.save(tmp_path / "cfg.txt", {})

    def test_creates_missing_parent_directories(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "cfg.yaml"
        configio.save(target, {"a": 1})
        assert target.exists()
