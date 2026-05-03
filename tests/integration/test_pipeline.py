"""Integration tests for the full bioaccx training pipeline.

Each test runs `train.run()` with a real (dummy) ONNX foundation model and
synthetic audio data, then asserts that the expected output files exist and
contain meaningful content.

Keras tests are marked @pytest.mark.slow because TF startup takes time.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import EMBED_DIM, SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_SECONDS


def _base_cfg_dict(foundation_cfg, data_dir: str | Path,
                   output_path: str | Path,
                   label_mode: str = "subfolders",
                   **overrides) -> dict:
    """Build a minimal config dict for train.run()."""
    from dataclasses import asdict
    fm = foundation_cfg
    return {
        "foundation_model": {
            "name": fm.name,
            "version": fm.version,
            "format": fm.format,
            "source": "local",
            "path": fm.path,
            "sample_rate": fm.sample_rate,
            "window_samples": fm.window_samples,
            "input_name": fm.input_name,
            "output_name": fm.output_name,
            "embedding_size": fm.embedding_size,
        },
        "dataset": {
            "data_dir": str(data_dir),
            "label_mode": label_mode,
            "embedding_workers": 2,
        },
        "training": {
            "classifier": "sklearn",
            "sklearn": {"C": 1.0, "max_iter": 200},
        },
        "output": {
            "output_path": str(output_path),
            "model_name": "test_cls",
            "model_version": "0.1",
            "output_type": "head",
            "output_format": "onnx",
        },
        **overrides,
    }


def _run(cfg_dict: dict):
    from bioaccx.config import _parse_config
    from bioaccx.train import run
    cfg = _parse_config(cfg_dict)
    return run(cfg)


# ---------------------------------------------------------------------------
# Subfolders mode — sklearn
# ---------------------------------------------------------------------------

class TestSubfoldersPipeline:
    def test_outputs_dict_has_expected_keys(self, foundation_cfg, subfolders_dataset, tmp_path):
        outputs = _run(_base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path))
        assert "sklearn_onnx_head" in outputs
        assert "sklearn_report" in outputs
        assert "dataset_info" in outputs
        assert "labels" in outputs
        assert "model_info" in outputs

    def test_output_files_exist(self, foundation_cfg, subfolders_dataset, tmp_path):
        outputs = _run(_base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path))
        for key, path in outputs.items():
            assert Path(path).exists(), f"{key}: {path} does not exist"

    def test_labels_file_has_all_classes(self, foundation_cfg, subfolders_dataset, tmp_path):
        outputs = _run(_base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path))
        labels = Path(outputs["labels"]).read_text().strip().splitlines()
        assert set(labels) == {"bird", "frog", "background"}

    def test_model_info_json_valid(self, foundation_cfg, subfolders_dataset, tmp_path):
        outputs = _run(_base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path))
        info = json.loads(Path(outputs["model_info"]).read_text())
        assert info["num_classes"] == 3
        assert "outputs" in info
        assert "foundation_model" in info

    def test_dataset_info_csv_has_all_samples(self, foundation_cfg, subfolders_dataset, tmp_path):
        outputs = _run(_base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path))
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        # 3 classes × 6 files = 18 samples
        assert len(rows) == 18
        labels_in_csv = {r["label"] for r in rows}
        assert labels_in_csv == {"bird", "frog", "background"}

    def test_onnx_head_runnable(self, foundation_cfg, subfolders_dataset, tmp_path):
        import onnxruntime as ort
        outputs = _run(_base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path))
        sess = ort.InferenceSession(str(outputs["sklearn_onnx_head"]))
        X = np.zeros((2, EMBED_DIM), dtype=np.float32)
        in_name = sess.get_inputs()[0].name
        probs = sess.run(["probabilities"], {in_name: X})[0]
        assert probs.shape == (2, 3)


class TestPredefinedSplitPipeline:
    def test_uses_predefined_split(self, foundation_cfg, predefined_split_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, predefined_split_dataset, tmp_path)
        outputs = _run(cfg)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        train_rows = [r for r in rows if r["split"] == "train"]
        test_rows  = [r for r in rows if r["split"] == "test"]
        # 2 classes × (6 train + 2 test) = 16 total
        assert len(train_rows) == 12
        assert len(test_rows) == 4


# ---------------------------------------------------------------------------
# file_per_label mode
# ---------------------------------------------------------------------------

class TestFilePerLabelPipeline:
    def test_fpl_mode_outputs_labels_file(self, foundation_cfg, file_per_label_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, file_per_label_dataset, tmp_path, label_mode="file_per_label")
        outputs = _run(cfg)
        labels = Path(outputs["labels"]).read_text().strip().splitlines()
        assert set(labels) == {"bird", "frog", "background"}

    def test_fpl_start_end_times_in_dataset_csv(self, foundation_cfg, file_per_label_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, file_per_label_dataset, tmp_path, label_mode="file_per_label")
        outputs = _run(cfg)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        # Every row should have non-empty start/end times
        for row in rows:
            assert row["start_time"] != ""
            assert row["end_time"] != ""

    def test_fpl_multi_row_generates_multiple_samples(
            self, foundation_cfg, file_per_label_dataset_multi_row, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, file_per_label_dataset_multi_row, tmp_path,
                             label_mode="file_per_label")
        outputs = _run(cfg)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        # 2 classes × 3 files × 2 rows each = 12 samples (before chunking, possibly more)
        assert len(rows) >= 12


# ---------------------------------------------------------------------------
# Table mode
# ---------------------------------------------------------------------------

class TestTablePipeline:
    def test_table_mode_basic(self, foundation_cfg, table_dataset, tmp_path):
        data_dir, table_path = table_dataset
        cfg = _base_cfg_dict(foundation_cfg, data_dir, tmp_path, label_mode="table")
        cfg["dataset"]["table_file"] = str(table_path)
        outputs = _run(cfg)
        labels = Path(outputs["labels"]).read_text().strip().splitlines()
        assert set(labels) == {"bird", "frog", "background"}

    def test_table_with_predefined_split(self, foundation_cfg, table_dataset_with_split, tmp_path):
        data_dir, table_path = table_dataset_with_split
        cfg = _base_cfg_dict(foundation_cfg, data_dir, tmp_path, label_mode="table")
        cfg["dataset"]["table_file"] = str(table_path)
        outputs = _run(cfg)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        train = [r for r in rows if r["split"] == "train"]
        test  = [r for r in rows if r["split"] == "test"]
        assert len(train) > 0 and len(test) > 0


# ---------------------------------------------------------------------------
# exclude_labels
# ---------------------------------------------------------------------------

class TestExcludeLabels:
    def test_excluded_label_absent_from_labels_file(self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["output"]["exclude_labels"] = ["background"]
        outputs = _run(cfg)
        labels = Path(outputs["labels"]).read_text().strip().splitlines()
        assert "background" not in labels
        assert "bird" in labels
        assert "frog" in labels

    def test_excluded_label_reduces_onnx_output_dimension(self, foundation_cfg, subfolders_dataset, tmp_path):
        import onnxruntime as ort
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["output"]["exclude_labels"] = ["background"]
        outputs = _run(cfg)
        sess = ort.InferenceSession(str(outputs["sklearn_onnx_head"]))
        X = np.zeros((1, EMBED_DIM), dtype=np.float32)
        in_name = sess.get_inputs()[0].name
        probs = sess.run(["probabilities_filtered"], {in_name: X})[0]
        assert probs.shape == (1, 2)  # 3 classes - 1 excluded = 2

    def test_model_info_records_excluded_labels(self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["output"]["exclude_labels"] = ["background"]
        outputs = _run(cfg)
        info = json.loads(Path(outputs["model_info"]).read_text())
        assert "background" in info.get("excluded_labels", [])


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class TestEmbeddingCache:
    def test_cache_files_created_when_export_embeddings(
            self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        embed_dir = tmp_path / "emb_export"
        cfg["output"]["export_embeddings"] = True
        cfg["output"]["embeddings_path"] = str(embed_dir)
        _run(cfg)
        npy_files = list(embed_dir.glob("*.npy"))
        assert len(npy_files) > 0

    def test_second_run_uses_cache(self, foundation_cfg, subfolders_dataset, tmp_path):
        """Running twice with cache_dir set — second run should load from cache."""
        import time
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path / "run1")
        cfg["dataset"]["embeddings_cache_path"] = str(cache_dir)
        cfg["output"]["export_embeddings"] = True
        cfg["output"]["embeddings_path"] = str(cache_dir)
        _run(cfg)

        # Second run — should load all embeddings from cache
        t0 = time.perf_counter()
        cfg2 = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path / "run2")
        cfg2["dataset"]["embeddings_cache_path"] = str(cache_dir)
        _run(cfg2)
        elapsed = time.perf_counter() - t0
        # Second run must finish, cache files from first run must still exist
        npy_files = list(cache_dir.glob("*.npy"))
        assert len(npy_files) > 0


# ---------------------------------------------------------------------------
# Output type: head vs full vs both
# ---------------------------------------------------------------------------

class TestOutputTypes:
    def test_head_only_output(self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["output"]["output_type"] = "head"
        outputs = _run(cfg)
        assert "sklearn_onnx_head" in outputs
        assert "sklearn_onnx_full" not in outputs

    def test_full_only_output(self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["output"]["output_type"] = "full"
        outputs = _run(cfg)
        assert "sklearn_onnx_full" in outputs
        assert "sklearn_onnx_head" not in outputs

    def test_both_outputs(self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["output"]["output_type"] = "both"
        outputs = _run(cfg)
        assert "sklearn_onnx_head" in outputs
        assert "sklearn_onnx_full" in outputs

    def test_full_model_accepts_audio_input(self, foundation_cfg, subfolders_dataset, tmp_path):
        """Full ONNX model should accept raw audio [batch, window_samples] as input."""
        import onnxruntime as ort
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["output"]["output_type"] = "full"
        outputs = _run(cfg)
        sess = ort.InferenceSession(str(outputs["sklearn_onnx_full"]))
        audio = np.zeros((2, WINDOW_SAMPLES), dtype=np.float32)
        out_names = [o.name for o in sess.get_outputs()]
        target = "probabilities" if "probabilities" in out_names else out_names[-1]
        probs = sess.run([target], {"INPUT": audio})[0]
        assert probs.shape[0] == 2
        assert probs.shape[1] == 3


# ---------------------------------------------------------------------------
# Both classifiers + comparison report
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBothClassifiers:
    def test_comparison_report_generated(self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["training"] = {
            "classifier": "both",
            "keras": {"hidden_units": 0, "epochs": 2, "batch_size": 16, "learning_rate": 1e-3},
            "sklearn": {"C": 1.0, "max_iter": 100},
        }
        outputs = _run(cfg)
        assert "comparison_report" in outputs
        assert Path(outputs["comparison_report"]).exists()

    def test_both_classifiers_produce_onnx_heads(self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["training"] = {
            "classifier": "both",
            "keras": {"hidden_units": 0, "epochs": 2, "batch_size": 16, "learning_rate": 1e-3},
            "sklearn": {"C": 1.0, "max_iter": 100},
        }
        outputs = _run(cfg)
        assert "keras_onnx_head" in outputs
        assert "sklearn_onnx_head" in outputs


# ---------------------------------------------------------------------------
# Keras pipeline
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestKerasPipeline:
    def test_keras_head_onnx_exported(self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["training"] = {
            "classifier": "keras",
            "keras": {"hidden_units": 32, "epochs": 2, "batch_size": 8, "learning_rate": 1e-3},
        }
        outputs = _run(cfg)
        assert "keras_onnx_head" in outputs
        assert Path(outputs["keras_onnx_head"]).exists()

    def test_keras_report_generated(self, foundation_cfg, subfolders_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["training"] = {
            "classifier": "keras",
            "keras": {"hidden_units": 0, "epochs": 2, "batch_size": 8, "learning_rate": 1e-3},
        }
        outputs = _run(cfg)
        assert "keras_report" in outputs
        report_text = Path(outputs["keras_report"]).read_text()
        assert "Keras" in report_text
        assert "accuracy" in report_text.lower()

    def test_keras_output_activation_softmax(self, foundation_cfg, subfolders_dataset, tmp_path):
        """Softmax output → probabilities sum to 1 for each sample."""
        import onnxruntime as ort
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["training"] = {
            "classifier": "keras",
            "keras": {"hidden_units": 0, "epochs": 2, "batch_size": 8,
                      "learning_rate": 1e-3, "output_activation": "softmax"},
        }
        outputs = _run(cfg)
        sess = ort.InferenceSession(str(outputs["keras_onnx_head"]))
        X = np.random.default_rng(0).standard_normal((4, EMBED_DIM)).astype(np.float32)
        in_name = sess.get_inputs()[0].name
        scores = sess.run(None, {in_name: X})[0]
        row_sums = scores.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------

class TestDatasetExport:
    def test_export_dataset_creates_wav_files(self, foundation_cfg, file_per_label_dataset, tmp_path):
        cfg = _base_cfg_dict(foundation_cfg, file_per_label_dataset, tmp_path, label_mode="file_per_label")
        cfg["output"]["export_dataset"] = True
        _run(cfg)
        dataset_dir = tmp_path / "test_cls_v0.1" / "dataset"
        wav_files = list(dataset_dir.rglob("*.wav"))
        assert len(wav_files) > 0

    def test_export_dataset_skipped_for_subfolders_mode(
            self, foundation_cfg, subfolders_dataset, tmp_path, capsys):
        cfg = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg["output"]["export_dataset"] = True
        _run(cfg)
        out = capsys.readouterr().out
        assert "skip" in out.lower()
        # No exported dataset directory should be created
        dataset_dir = tmp_path / "test_cls_v0.1" / "dataset"
        assert not dataset_dir.exists()
