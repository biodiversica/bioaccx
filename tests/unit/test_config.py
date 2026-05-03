"""Unit tests for bioaccx.config."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioaccx.config import (
    BioaccxConfig,
    DatasetConfig,
    FoundationModelConfig,
    KerasConfig,
    OutputConfig,
    SklearnConfig,
    TrainingConfig,
    load_config,
    _parse_config,
)


def _minimal_dict(overrides: dict | None = None) -> dict:
    base = {
        "foundation_model": {
            "name": "birdnet",
            "version": "2.4",
            "format": "onnx",
            "source": "local",
            "path": "/tmp/model.onnx",
            "sample_rate": 48000,
            "window_seconds": 3.0,
            "input_name": "INPUT",
            "output_name": "embedding",
            "embedding_size": 1024,
        },
        "dataset": {"data_dir": "/tmp/data"},
    }
    if overrides:
        base.update(overrides)
    return base


class TestFoundationModelConfig:
    def test_get_window_samples_from_seconds(self):
        fm = FoundationModelConfig(name="x", sample_rate=48_000, window_seconds=3.0)
        assert fm.get_window_samples() == 144_000

    def test_get_window_samples_from_samples(self):
        fm = FoundationModelConfig(name="x", sample_rate=48_000, window_samples=12_000)
        assert fm.get_window_samples() == 12_000

    def test_window_samples_takes_priority(self):
        fm = FoundationModelConfig(name="x", sample_rate=48_000,
                                   window_samples=100, window_seconds=3.0)
        assert fm.get_window_samples() == 100

    def test_missing_window_raises(self):
        fm = FoundationModelConfig(name="x", sample_rate=48_000)
        with pytest.raises(ValueError, match="window"):
            fm.get_window_samples()

    def test_defaults(self):
        fm = FoundationModelConfig(name="x")
        assert fm.format == "onnx"
        assert fm.source == "local"
        assert fm.sample_rate == 48_000
        assert fm.embedding_size == 1024


class TestOutputConfig:
    def test_output_dir_property(self):
        cfg = BioaccxConfig(
            foundation_model=FoundationModelConfig(name="x", sample_rate=48_000, window_seconds=1.0),
            dataset=DatasetConfig(data_dir="/tmp"),
            output=OutputConfig(output_path="/models", model_name="cls", model_version="2.0"),
        )
        assert cfg.output_dir == Path("/models/cls_v2.0")

    def test_exclude_labels_default_empty(self):
        out = OutputConfig()
        assert out.exclude_labels == []

    def test_export_flags_default_false(self):
        out = OutputConfig()
        assert out.export_dataset is False
        assert out.export_embeddings is False


class TestParseConfig:
    def test_parse_minimal_config(self):
        cfg = _parse_config(_minimal_dict())
        assert cfg.foundation_model.name == "birdnet"
        assert cfg.dataset.data_dir == "/tmp/data"

    def test_parse_unknown_keys_silently_ignored(self):
        d = _minimal_dict()
        d["foundation_model"]["unknown_key"] = "value"
        cfg = _parse_config(d)
        assert cfg.foundation_model.name == "birdnet"

    def test_training_defaults(self):
        cfg = _parse_config(_minimal_dict())
        assert cfg.training.classifier == "keras"
        assert isinstance(cfg.training.keras, KerasConfig)
        assert isinstance(cfg.training.sklearn, SklearnConfig)

    def test_keras_config_parsed(self):
        d = _minimal_dict({"training": {"classifier": "keras",
                                         "keras": {"epochs": 10, "hidden_units": 128}}})
        cfg = _parse_config(d)
        assert cfg.training.keras.epochs == 10
        assert cfg.training.keras.hidden_units == 128

    def test_sklearn_config_parsed(self):
        d = _minimal_dict({"training": {"classifier": "sklearn",
                                         "sklearn": {"C": 2.0, "max_iter": 500}}})
        cfg = _parse_config(d)
        assert cfg.training.sklearn.C == 2.0
        assert cfg.training.sklearn.max_iter == 500

    def test_output_config_parsed(self):
        d = _minimal_dict({"output": {"model_name": "mymodel", "model_version": "1.1",
                                       "exclude_labels": ["noise", "background"]}})
        cfg = _parse_config(d)
        assert cfg.output.model_name == "mymodel"
        assert "noise" in cfg.output.exclude_labels

    def test_dataset_label_mode_defaults(self):
        cfg = _parse_config(_minimal_dict())
        assert cfg.dataset.label_mode == "subfolders"
        assert cfg.dataset.test_ratio == 0.2

    def test_dataset_overlap_default(self):
        cfg = _parse_config(_minimal_dict())
        assert cfg.dataset.overlap == 0.0


class TestLoadConfigFromFile:
    def test_load_from_json(self, tmp_path):
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps(_minimal_dict()))
        cfg = load_config(cfg_file)
        assert cfg.foundation_model.name == "birdnet"

    def test_load_from_yaml(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            "foundation_model:\n"
            "  name: perch\n"
            "  version: '2.0'\n"
            "  format: onnx\n"
            "  source: local\n"
            "  path: /tmp/model.onnx\n"
            "  sample_rate: 32000\n"
            "  window_seconds: 5.0\n"
            "  input_name: inputs\n"
            "  output_name: embedding\n"
            "  embedding_size: 1536\n"
            "dataset:\n"
            "  data_dir: /tmp/data\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.foundation_model.name == "perch"
        assert cfg.foundation_model.sample_rate == 32_000
        assert cfg.foundation_model.embedding_size == 1536
