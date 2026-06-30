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
        assert fm.data_type == "FP32"


class TestOutputConfig:
    def test_output_dir_property_unknown_model(self):
        cfg = BioaccxConfig(
            foundation_model=FoundationModelConfig(name="x", sample_rate=48_000, window_seconds=1.0),
            dataset=DatasetConfig(data_dir="/tmp"),
            output=OutputConfig(output_path="/models", model_name="cls", model_version="2.0"),
        )
        assert cfg.output_dir == Path("/models/cls_0xffff_v2.0")

    def test_output_dir_property_known_model(self):
        cfg = BioaccxConfig(
            foundation_model=FoundationModelConfig(
                name="birdnet", version="2.4", data_type="FP32",
                sample_rate=48_000, window_seconds=3.0,
            ),
            dataset=DatasetConfig(data_dir="/tmp"),
            output=OutputConfig(output_path="/models", model_name="cls", model_version="1.0"),
        )
        assert cfg.output_dir == Path("/models/cls_0xbb00_v1.0")

    def test_model_stem_known_model(self):
        cfg = BioaccxConfig(
            foundation_model=FoundationModelConfig(
                name="perch", version="2.0", data_type="FP32",
                sample_rate=32_000, window_seconds=5.0,
            ),
            output=OutputConfig(model_name="my_cls", model_version="3.0"),
        )
        assert cfg.model_stem == "my_cls_0xbb10_v3.0"

    def test_model_stem_unknown_model_uses_fallback(self):
        cfg = BioaccxConfig(
            foundation_model=FoundationModelConfig(name="custom", version="9.9"),
            output=OutputConfig(model_name="head", model_version="1.0"),
        )
        assert cfg.model_stem == "head_0xffff_v1.0"

    def test_exclude_labels_default_empty(self):
        out = OutputConfig()
        assert out.exclude_labels == []

    def test_export_flags_default_false(self):
        out = OutputConfig()
        assert out.export_dataset is False
        assert out.export_embeddings is False

    def test_embeddings_format_default_npy(self):
        out = OutputConfig()
        assert out.embeddings_format == "npy"

    def test_embeddings_format_sqlite_parsed(self):
        d = _minimal_dict({"output": {"embeddings_format": "sqlite"}})
        cfg = _parse_config(d)
        assert cfg.output.embeddings_format == "sqlite"

    def test_embeddings_format_npy_explicit(self):
        d = _minimal_dict({"output": {"embeddings_format": "npy"}})
        cfg = _parse_config(d)
        assert cfg.output.embeddings_format == "npy"

    def test_data_types_default_unset(self):
        out = OutputConfig()
        assert out.data_types is None


class TestOutputDataTypes:
    def test_defaults_to_foundation_data_type(self):
        cfg = _parse_config(_minimal_dict())
        # foundation model defaults to FP32; no output.data_types set
        assert cfg.output_data_types == ["FP32"]

    def test_default_follows_foundation_int8(self):
        d = _minimal_dict()
        d["foundation_model"]["data_type"] = "INT8"
        cfg = _parse_config(d)
        assert cfg.output_data_types == ["INT8"]

    def test_explicit_list_controls_output(self):
        d = _minimal_dict({"output": {"data_types": ["FP32", "FP16", "INT8"]}})
        cfg = _parse_config(d)
        assert cfg.output_data_types == ["FP32", "FP16", "INT8"]

    def test_normalizes_case_and_dedupes(self):
        d = _minimal_dict({"output": {"data_types": ["fp16", "FP16", "int8"]}})
        cfg = _parse_config(d)
        assert cfg.output_data_types == ["FP16", "INT8"]

    def test_invalid_precision_raises(self):
        d = _minimal_dict({"output": {"data_types": ["FP64"]}})
        cfg = _parse_config(d)
        with pytest.raises(ValueError, match="Unknown output data_type"):
            _ = cfg.output_data_types


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


class TestRegistryResolution:
    def test_registry_id_fills_defaults(self):
        d = {"foundation_model": {"registry_id": "0xbb00"}, "dataset": {"data_dir": "/tmp"}}
        cfg = _parse_config(d)
        fm = cfg.foundation_model
        assert fm.name == "birdnet"
        assert fm.version == "2.4"
        assert fm.source == "huggingface"
        assert fm.hf_repo == "biodiversica/BirdNET-onnx-backbone"
        assert fm.sample_rate == 48000
        assert fm.embedding_size == 1024

    def test_registry_id_user_fields_override_defaults(self):
        d = {
            "foundation_model": {
                "registry_id": "0xbb00",
                "source": "local",
                "path": "/tmp/model.onnx",
            },
            "dataset": {"data_dir": "/tmp"},
        }
        cfg = _parse_config(d)
        assert cfg.foundation_model.source == "local"
        assert cfg.foundation_model.path == "/tmp/model.onnx"
        # registry defaults still fill non-overridden fields
        assert cfg.foundation_model.sample_rate == 48000

    def test_auto_lookup_by_name_version(self):
        d = {
            "foundation_model": {
                "name": "birdnet",
                "version": "2.4",
                "window_seconds": 3.0,
            },
            "dataset": {"data_dir": "/tmp"},
        }
        cfg = _parse_config(d)
        assert cfg.foundation_model.source == "huggingface"
        assert cfg.foundation_model.hf_repo == "biodiversica/BirdNET-onnx-backbone"

    def test_auto_lookup_user_overrides_registry_source(self):
        d = {
            "foundation_model": {
                "name": "perch",
                "version": "2.0",
                "source": "local",
                "path": "/tmp/perch.onnx",
            },
            "dataset": {"data_dir": "/tmp"},
        }
        cfg = _parse_config(d)
        assert cfg.foundation_model.source == "local"
        assert cfg.foundation_model.path == "/tmp/perch.onnx"
        assert cfg.foundation_model.embedding_size == 1536

    def test_tflite_registry_id_sets_offset(self):
        d = {
            "foundation_model": {
                "registry_id": "0xbb02",
                "path": "/tmp/birdnet.tflite",
            },
            "dataset": {"data_dir": "/tmp"},
        }
        cfg = _parse_config(d)
        assert cfg.foundation_model.tflite_output_tensor_offset == -1
        assert cfg.foundation_model.format == "tflite"

    def test_unknown_registry_id_raises(self):
        d = {"foundation_model": {"registry_id": "0xdead"}, "dataset": {"data_dir": "/tmp"}}
        with pytest.raises(ValueError, match="registry_id"):
            _parse_config(d)

    def test_model_stem_uses_registry_id(self):
        d = {"foundation_model": {"registry_id": "0xbb00"}, "dataset": {"data_dir": "/tmp"}}
        cfg = _parse_config(d)
        assert cfg.model_stem.startswith("custom_classifier_0xbb00_")


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


class TestDatasetSources:
    """Multi-source dataset blocks via ``dataset.sources``."""

    def _dict_with_sources(self, sources, ds_extra=None):
        ds = {"test_ratio": 0.25, "random_seed": 7, "embedding_workers": 8}
        if ds_extra:
            ds.update(ds_extra)
        ds["sources"] = sources
        return _minimal_dict({"dataset": ds})

    def test_no_sources_keeps_single_block(self):
        cfg = _parse_config(_minimal_dict())
        assert cfg._dataset_blocks is None
        assert len(cfg.dataset_blocks) == 1
        assert cfg.dataset_blocks[0] is cfg.dataset

    def test_sources_build_one_block_each(self):
        cfg = _parse_config(self._dict_with_sources([
            {"data_dir": "/a", "label_mode": "subfolders"},
            {"data_dir": "/b", "label_mode": "file_per_label"},
        ]))
        blocks = cfg.dataset_blocks
        assert len(blocks) == 2
        assert blocks[0].data_dir == "/a"
        assert blocks[0].label_mode == "subfolders"
        assert blocks[1].data_dir == "/b"
        assert blocks[1].label_mode == "file_per_label"

    def test_blocks_inherit_top_level_fields(self):
        cfg = _parse_config(self._dict_with_sources(
            [{"data_dir": "/a"}, {"data_dir": "/b", "overlap": 0.5}],
            ds_extra={"overlap": 0.25, "min_anchor_fraction": 0.3},
        ))
        b0, b1 = cfg.dataset_blocks
        # inherited from the top-level block
        assert b0.overlap == 0.25
        assert b0.min_anchor_fraction == 0.3
        assert b1.min_anchor_fraction == 0.3
        # per-source override wins for non-run-level fields
        assert b1.overlap == 0.5
        # run-level fields inherited everywhere
        assert b0.embedding_workers == 8 and b1.embedding_workers == 8

    def test_run_level_fields_not_overridable_per_source(self):
        cfg = _parse_config(self._dict_with_sources([
            {"data_dir": "/a", "test_ratio": 0.9, "random_seed": 999,
             "embedding_workers": 1},
        ]))
        b = cfg.dataset_blocks[0]
        # run-level overrides are ignored; values come from the top-level block
        assert b.test_ratio == 0.25
        assert b.random_seed == 7
        assert b.embedding_workers == 8
        # the run-level config also exposes them
        assert cfg.dataset.test_ratio == 0.25

    def test_augmentation_inherited_when_not_overridden(self):
        cfg = _parse_config(self._dict_with_sources(
            [{"data_dir": "/a"}, {"data_dir": "/b"}],
            ds_extra={"augmentation": {"augmentation_dir": "noise",
                                       "snr_levels": [10, 20]}},
        ))
        for b in cfg.dataset_blocks:
            assert b.augmentation is not None
            assert b.augmentation.snr_levels == [10, 20]

    def test_augmentation_null_opts_out(self):
        cfg = _parse_config(self._dict_with_sources(
            [{"data_dir": "/a", "augmentation": None}, {"data_dir": "/b"}],
            ds_extra={"augmentation": {"augmentation_dir": "noise",
                                       "snr_levels": [10]}},
        ))
        assert cfg.dataset_blocks[0].augmentation is None
        assert cfg.dataset_blocks[1].augmentation is not None

    def test_per_source_augmentation_override(self):
        cfg = _parse_config(self._dict_with_sources([
            {"data_dir": "/a"},
            {"data_dir": "/b", "augmentation": {"augmentation_dir": "n",
                                                "snr_levels": [3]}},
        ]))
        assert cfg.dataset_blocks[0].augmentation is None
        assert cfg.dataset_blocks[1].augmentation.snr_levels == [3]

    def test_ssh_is_per_block(self):
        cfg = _parse_config(self._dict_with_sources([
            {"data_dir": "/local", "label_mode": "subfolders"},
            {"data_dir": "/remote", "ssh_host": "host.example",
             "ssh_user": "me", "ssh_port": 2222},
        ]))
        assert cfg.dataset_blocks[0].ssh_host is None
        assert cfg.dataset_blocks[1].ssh_host == "host.example"
        assert cfg.dataset_blocks[1].ssh_user == "me"
        assert cfg.dataset_blocks[1].ssh_port == 2222

    def test_append_inherited_for_dedup(self):
        cfg = _parse_config(self._dict_with_sources(
            [{"data_dir": "/a"}, {"data_dir": "/b"}],
            ds_extra={"append_dataset_path": "/existing"},
        ))
        # run-level append target is exposed on the run-level config…
        assert cfg.dataset.append_dataset_path == "/existing"
        # …and inherited by each block so it can de-duplicate.
        for b in cfg.dataset_blocks:
            assert b.append_dataset_path == "/existing"

    def test_sources_must_be_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            _parse_config(_minimal_dict({"dataset": {"sources": {"data_dir": "/a"}}}))

    def test_source_entry_must_be_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _parse_config(_minimal_dict({"dataset": {"sources": ["/a"]}}))


class TestAugmentationLabels:
    def test_labels_only_is_valid(self):
        d = _minimal_dict({"dataset": {
            "data_dir": "/tmp/data",
            "augmentation": {"augmentation_labels": ["noise"], "snr_levels": [10]},
        }})
        cfg = _parse_config(d)
        assert cfg.dataset.augmentation.augmentation_dir is None
        assert cfg.dataset.augmentation.augmentation_labels == ["noise"]

    def test_dir_and_labels_combine(self):
        d = _minimal_dict({"dataset": {
            "data_dir": "/tmp/data",
            "augmentation": {"augmentation_dir": "noise", "snr_levels": [10],
                             "augmentation_labels": ["rain", "wind"]},
        }})
        cfg = _parse_config(d)
        assert cfg.dataset.augmentation.augmentation_dir == "noise"
        assert cfg.dataset.augmentation.augmentation_labels == ["rain", "wind"]

    def test_neither_dir_nor_labels_raises(self):
        d = _minimal_dict({"dataset": {
            "data_dir": "/tmp/data",
            "augmentation": {"snr_levels": [10]},
        }})
        with pytest.raises(ValueError, match="augmentation_dir"):
            _parse_config(d)
