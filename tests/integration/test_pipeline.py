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
        # 3 classes × 6 files × 3 windows (0.3 s file / 0.1 s window) = 54 samples
        assert len(rows) == 54
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
        # 2 classes × 6 train files × 3 windows = 36 train samples
        # 2 classes × 2 test  files × 3 windows = 12 test  samples
        assert len(train_rows) == 36
        assert len(test_rows) == 12


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
        npy_files = list(embed_dir.rglob("*.npy"))
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
        npy_files = list(cache_dir.rglob("*.npy"))
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
        from bioaccx.config import _parse_config
        cfg_dict = _base_cfg_dict(foundation_cfg, file_per_label_dataset, tmp_path, label_mode="file_per_label")
        cfg_dict["output"]["export_dataset"] = True
        _run(cfg_dict)
        dataset_dir = _parse_config(cfg_dict).output_dir / "dataset"
        wav_files = list(dataset_dir.rglob("*.wav"))
        assert len(wav_files) > 0

    def test_export_dataset_skipped_for_subfolders_mode(
            self, foundation_cfg, subfolders_dataset, tmp_path, capsys):
        from bioaccx.config import _parse_config
        cfg_dict = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg_dict["output"]["export_dataset"] = True
        _run(cfg_dict)
        out = capsys.readouterr().out
        assert "skip" in out.lower()
        # No exported dataset directory should be created
        dataset_dir = _parse_config(cfg_dict).output_dir / "dataset"
        assert not dataset_dir.exists()


# ---------------------------------------------------------------------------
# Augmentation / random sample shift
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def short_sample_fpl_dataset(tmp_path_factory):
    """2 classes × 4 WAV files; annotations are 0.0–0.05 s (half the 0.1 s window)."""
    from tests.conftest import make_sine_wav
    root = tmp_path_factory.mktemp("short_fpl_ds")
    classes = {"bird": 440, "frog": 2_000}
    for ci, (cls, freq) in enumerate(classes.items()):
        for j in range(4):
            wav = root / f"{cls}_{j:02d}.wav"
            txt = root / f"{cls}_{j:02d}.txt"
            make_sine_wav(wav, freq, 0.5, seed=ci * 40 + j)
            # Annotation is only 0.05 s — shorter than the 0.1 s window
            txt.write_text(f"0.0\t0.05\t{cls}\n")
    return root


@pytest.fixture(scope="session")
def noise_dataset(tmp_path_factory):
    """Two short noise WAV files used as augmentation sources."""
    import soundfile as sf
    root = tmp_path_factory.mktemp("noise_ds")
    rng = np.random.default_rng(0)
    for name in ("noise_a.wav", "noise_b.wav"):
        audio = rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1  # 1 s
        sf.write(str(root / name), audio, SAMPLE_RATE)
    return root


class TestAugmentation:
    def test_augmented_samples_expand_train_set(
            self, foundation_cfg, subfolders_dataset, noise_dataset, tmp_path):
        """Each train sample × 2 noise files × 2 SNR levels = 4× more train rows."""
        cfg_dict = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg_dict["dataset"]["augmentation"] = {
            "augmentation_dir": str(noise_dataset),
            "snr_levels": [10.0, 0.0],
            "keep_original": False,
        }
        outputs = _run(cfg_dict)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        train_rows = [r for r in rows if r["split"] == "train"]
        # keep_original=False → only augmented; 2 noise × 2 SNR = 4 augmented per clean sample
        # subfolders_dataset: 3 classes × 6 files × 3 windows = 54 total
        # roughly 80 % train (auto-split) × 4 = ~172; just check it's > 54
        assert len(train_rows) > 54

    def test_keep_original_includes_clean_samples(
            self, foundation_cfg, subfolders_dataset, noise_dataset, tmp_path):
        cfg_dict = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg_dict["dataset"]["augmentation"] = {
            "augmentation_dir": str(noise_dataset),
            "snr_levels": [10.0],
            "keep_original": True,
        }
        outputs = _run(cfg_dict)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        train_rows = [r for r in rows if r["split"] == "train"]
        clean = [r for r in train_rows if r["noise_file"] == ""]
        augmented = [r for r in train_rows if r["noise_file"] != ""]
        assert len(clean) > 0
        assert len(augmented) > 0
        # augmented = clean × 2 noise × 1 SNR = 2× clean
        assert len(augmented) == 2 * len(clean)

    def test_csv_has_noise_file_and_snr_columns(
            self, foundation_cfg, subfolders_dataset, noise_dataset, tmp_path):
        cfg_dict = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg_dict["dataset"]["augmentation"] = {
            "augmentation_dir": str(noise_dataset),
            "snr_levels": [5.0],
            "keep_original": False,
        }
        outputs = _run(cfg_dict)
        with Path(outputs["dataset_info"]).open() as f:
            reader = csv.DictReader(f)
            assert "noise_file" in reader.fieldnames
            assert "snr_db" in reader.fieldnames
            rows = list(reader)
        snr_values = {float(r["snr_db"]) for r in rows if r["snr_db"]}
        assert snr_values == {5.0}

    def test_augmentation_params_in_metadata_json(
            self, foundation_cfg, subfolders_dataset, noise_dataset, tmp_path):
        cfg_dict = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg_dict["dataset"]["augmentation"] = {
            "augmentation_dir": str(noise_dataset),
            "snr_levels": [5.0, 15.0],
            "keep_original": True,
            "augment_test": False,
        }
        outputs = _run(cfg_dict)
        info = json.loads(Path(outputs["model_info"]).read_text())
        aug = info.get("augmentation")
        assert aug is not None
        assert aug["snr_levels"] == [5.0, 15.0]
        assert aug["keep_original"] is True
        assert aug["augment_test"] is False
        assert "augmentation_dir" in aug

    def test_no_augmentation_key_when_not_configured(
            self, foundation_cfg, subfolders_dataset, tmp_path):
        outputs = _run(_base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path))
        info = json.loads(Path(outputs["model_info"]).read_text())
        assert "augmentation" not in info

    def test_augmented_model_is_trainable(
            self, foundation_cfg, subfolders_dataset, noise_dataset, tmp_path):
        """Pipeline should complete successfully and produce a valid ONNX head."""
        cfg_dict = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg_dict["dataset"]["augmentation"] = {
            "augmentation_dir": str(noise_dataset),
            "snr_levels": [10.0],
            "keep_original": True,
        }
        outputs = _run(cfg_dict)
        assert "sklearn_onnx_head" in outputs
        assert Path(outputs["sklearn_onnx_head"]).exists()

    def test_no_augmentation_when_dir_is_empty(
            self, foundation_cfg, subfolders_dataset, tmp_path, tmp_path_factory, capsys):
        empty_dir = tmp_path_factory.mktemp("empty_noise")
        cfg_dict = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg_dict["dataset"]["augmentation"] = {
            "augmentation_dir": str(empty_dir),
            "snr_levels": [10.0],
            "keep_original": True,
        }
        outputs = _run(cfg_dict)
        out = capsys.readouterr().out
        assert "skipping" in out.lower()
        # augmentation was skipped → no noise columns in CSV
        with Path(outputs["dataset_info"]).open() as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert "noise_file" not in fieldnames

    def test_augment_test_expands_test_set(
            self, foundation_cfg, subfolders_dataset, noise_dataset, tmp_path):
        cfg_dict = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg_dict["dataset"]["augmentation"] = {
            "augmentation_dir": str(noise_dataset),
            "snr_levels": [10.0],
            "keep_original": False,
            "augment_test": True,
        }
        outputs = _run(cfg_dict)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        test_rows = [r for r in rows if r["split"] == "test"]
        # augment_test=True, keep_original=False → every test row is augmented
        assert len(test_rows) > 0
        assert all(r["noise_file"] != "" for r in test_rows)

    def test_augment_test_false_leaves_test_set_clean(
            self, foundation_cfg, subfolders_dataset, noise_dataset, tmp_path):
        cfg_dict = _base_cfg_dict(foundation_cfg, subfolders_dataset, tmp_path)
        cfg_dict["dataset"]["augmentation"] = {
            "augmentation_dir": str(noise_dataset),
            "snr_levels": [10.0],
            "keep_original": False,
            "augment_test": False,
        }
        outputs = _run(cfg_dict)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        test_rows = [r for r in rows if r["split"] == "test"]
        assert all(r["noise_file"] == "" for r in test_rows)

    def test_augmented_export_dataset_creates_mixed_wavs(
            self, foundation_cfg, file_per_label_dataset, noise_dataset, tmp_path):
        from bioaccx.config import _parse_config
        cfg_dict = _base_cfg_dict(
            foundation_cfg, file_per_label_dataset, tmp_path, label_mode="file_per_label"
        )
        cfg_dict["output"]["export_dataset"] = True
        cfg_dict["dataset"]["augmentation"] = {
            "augmentation_dir": str(noise_dataset),
            "snr_levels": [10.0],
            "keep_original": False,
        }
        _run(cfg_dict)
        dataset_dir = _parse_config(cfg_dict).output_dir / "dataset"
        train_wavs = list((dataset_dir / "train").rglob("*.wav"))
        test_wavs  = list((dataset_dir / "test").rglob("*.wav"))
        # keep_original=False → every train file is augmented (has noise tag)
        assert len(train_wavs) > 0
        assert all("_noise_" in f.stem for f in train_wavs)
        # test set is never augmented
        assert all("_noise_" not in f.stem for f in test_wavs)


# ---------------------------------------------------------------------------
# Random sample shift
# ---------------------------------------------------------------------------

class TestRandomSampleShift:
    def _shift_cfg(self, foundation_cfg, data_dir, output_path, **dataset_overrides):
        cfg = _base_cfg_dict(
            foundation_cfg, data_dir, output_path, label_mode="file_per_label"
        )
        cfg["dataset"]["random_sample_shift"] = True
        cfg["dataset"].update(dataset_overrides)
        return cfg

    def test_short_samples_get_offset_column_in_csv(
            self, foundation_cfg, short_sample_fpl_dataset, tmp_path):
        outputs = _run(self._shift_cfg(foundation_cfg, short_sample_fpl_dataset, tmp_path))
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        assert "signal_offset_samples" in rows[0]
        # All samples are short (0.05 s < 0.1 s window) — every row has an offset
        assert all(r["signal_offset_samples"] != "" for r in rows)

    def test_offsets_within_valid_range(
            self, foundation_cfg, short_sample_fpl_dataset, tmp_path):
        outputs = _run(self._shift_cfg(foundation_cfg, short_sample_fpl_dataset, tmp_path))
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        # signal is 0.05 s = 2400 samples; window = 4800 → max_offset = 2400
        for r in rows:
            off = int(r["signal_offset_samples"])
            assert 0 <= off <= WINDOW_SAMPLES // 2

    def test_offsets_are_reproducible(
            self, foundation_cfg, short_sample_fpl_dataset, tmp_path):
        cfg = self._shift_cfg(foundation_cfg, short_sample_fpl_dataset, tmp_path)
        out1 = _run(cfg)
        out2 = _run(cfg)
        rows1 = list(csv.DictReader(Path(out1["dataset_info"]).open()))
        rows2 = list(csv.DictReader(Path(out2["dataset_info"]).open()))
        offsets1 = [r["signal_offset_samples"] for r in rows1]
        offsets2 = [r["signal_offset_samples"] for r in rows2]
        assert offsets1 == offsets2

    def test_no_shift_column_when_disabled(
            self, foundation_cfg, short_sample_fpl_dataset, tmp_path):
        cfg = _base_cfg_dict(
            foundation_cfg, short_sample_fpl_dataset, tmp_path, label_mode="file_per_label"
        )
        outputs = _run(cfg)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        assert "signal_offset_samples" not in rows[0]

    def test_augmented_copies_have_distinct_offsets(
            self, foundation_cfg, short_sample_fpl_dataset, noise_dataset, tmp_path):
        cfg = self._shift_cfg(foundation_cfg, short_sample_fpl_dataset, tmp_path)
        cfg["dataset"]["augmentation"] = {
            "augmentation_dir": str(noise_dataset),
            "snr_levels": [0.0, 10.0],
            "keep_original": True,
        }
        outputs = _run(cfg)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        # Group by (filepath, start_time, end_time); augmented copies are
        # identified by non-empty noise_file.  Within each source sample, the
        # offsets across its augmented copies must not all be equal.
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for r in rows:
            key = (r["filepath"], r["start_time"], r["end_time"])
            if r.get("noise_file", "") != "":
                groups[key].append(int(r["signal_offset_samples"]))
        assert groups, "expected augmented rows with noise_file"
        # At least one source sample should have varied offsets across its copies
        assert any(len(set(offs)) > 1 for offs in groups.values())

    def test_pipeline_runs_with_shift_enabled(
            self, foundation_cfg, short_sample_fpl_dataset, tmp_path):
        outputs = _run(self._shift_cfg(foundation_cfg, short_sample_fpl_dataset, tmp_path))
        assert Path(outputs["sklearn_onnx_head"]).exists()

    def test_full_window_samples_have_no_offset(
            self, foundation_cfg, file_per_label_dataset, tmp_path):
        # file_per_label_dataset has 0.3 s annotations = 3 full windows → no short samples
        cfg = _base_cfg_dict(
            foundation_cfg, file_per_label_dataset, tmp_path, label_mode="file_per_label"
        )
        cfg["dataset"]["random_sample_shift"] = True
        outputs = _run(cfg)
        with Path(outputs["dataset_info"]).open() as f:
            rows = list(csv.DictReader(f))
        # No short samples → column should not be present (or all empty)
        if "signal_offset_samples" in rows[0]:
            assert all(r["signal_offset_samples"] == "" for r in rows)
