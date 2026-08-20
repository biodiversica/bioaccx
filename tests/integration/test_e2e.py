"""End-to-end tests using synthetic audio with distinct spectral content.

The DFT-based ONNX embedder extracts phase-invariant spectral energy features,
so classes with different fundamental frequencies (bird/frog/background) produce
well-separated embeddings.  Both classifiers are expected to achieve ≥ 80%
accuracy on the test set, and their ONNX head models are compared directly.

All tests here are marked @pytest.mark.slow (TF startup + training).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import (
    CLASS_FREQS,
    EMBED_DIM,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    make_sine_wav,
)


# ---------------------------------------------------------------------------
# Dataset fixture — more files per class for reliable accuracy
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e_dataset(tmp_path_factory):
    """3 classes × 20 WAV files, each 0.3 s, organized in subfolders.

    Frequencies chosen so the DFT embedder cleanly separates them:
      bird       → 440 Hz
      frog       → 2 000 Hz
      background → white noise
    """
    root = tmp_path_factory.mktemp("e2e_ds")
    for ci, (cls, freq) in enumerate(CLASS_FREQS.items()):
        d = root / cls
        d.mkdir()
        for j in range(20):
            make_sine_wav(d / f"{cls}_{j:03d}.wav", freq, 0.3,
                          noise_std=0.05, seed=ci * 200 + j)
    return root


# ---------------------------------------------------------------------------
# Pipeline runner helper
# ---------------------------------------------------------------------------

def _run_pipeline(dft_foundation_cfg, data_dir: Path, output_path: Path,
                  classifier: str, output_type: str = "head",
                  output_activation: str | None = None,
                  exclude_labels: list[str] | None = None,
                  output_format: str = "onnx",
                  data_types: list[str] | None = None) -> dict[str, str]:
    from bioaccx.config import _parse_config
    from bioaccx.train import run

    fm = dft_foundation_cfg
    cfg_dict = {
        "foundation_model": {
            "name": fm.name, "version": fm.version,
            "format": fm.format, "source": "local", "path": fm.path,
            "sample_rate": fm.sample_rate, "window_samples": fm.window_samples,
            "input_name": fm.input_name, "output_name": fm.output_name,
            "embedding_size": fm.embedding_size,
        },
        "dataset": {
            "data_dir": str(data_dir),
            "label_mode": "subfolders",
            "embedding_workers": 2,
            "test_ratio": 0.25,
            "random_seed": 42,
        },
        "training": {
            "classifier": classifier,
            "keras": {
                "hidden_units": 32,
                "epochs": 80,
                "batch_size": 8,
                "learning_rate": 3e-3,
                **({"output_activation": output_activation} if output_activation else {}),
            },
            "sklearn": {"C": 1.0, "max_iter": 500},
        },
        "output": {
            "output_path": str(output_path),
            "model_name": f"e2e_{classifier}",
            "model_version": "0.1",
            "output_type": output_type,
            "output_format": output_format,
            "exclude_labels": exclude_labels or [],
            **({"data_types": data_types} if data_types else {}),
        },
    }
    cfg = _parse_config(cfg_dict)
    return run(cfg)


def _onnx_accuracy(onnx_path: Path, X: np.ndarray, y: np.ndarray,
                   output_tensor: str = "probabilities") -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    target = output_tensor if output_tensor in out_names else out_names[-1]
    probs = sess.run([target], {in_name: X})[0]
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == y))


def _load_test_embeddings(model_info_path: Path, dataset_csv_path: Path,
                           dft_foundation_cfg) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Re-extract embeddings for the test split using the DFT embedder.

    Samples whose label is not in the model's output label set (e.g. excluded
    labels) are silently skipped so the returned y indices are valid for the
    exported model.
    """
    from bioaccx.embedder import load_embedder

    with dataset_csv_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

    info = json.loads(model_info_path.read_text())
    label_names = info["labels"]
    label_to_idx = {n: i for i, n in enumerate(label_names)}

    embedder = load_embedder(dft_foundation_cfg)
    X_list, y_list = [], []
    for row in rows:
        if row["label"] not in label_to_idx:
            continue
        fp = Path(row["filepath"])
        start = float(row["start_time"]) if row["start_time"] else None
        end   = float(row["end_time"])   if row["end_time"]   else None
        emb = embedder.embed_file(fp, start, end)
        X_list.append(emb)
        y_list.append(label_to_idx[row["label"]])
    return np.stack(X_list).astype(np.float32), np.array(y_list), label_names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestE2ESklearnAccuracy:
    def test_sklearn_achieves_high_accuracy(
            self, dft_foundation_cfg, e2e_dataset, tmp_path):
        outputs = _run_pipeline(dft_foundation_cfg, e2e_dataset,
                                tmp_path, classifier="sklearn")
        X_test, y_test, _ = _load_test_embeddings(
            Path(outputs["model_info"]),
            Path(outputs["dataset_list"]),
            dft_foundation_cfg,
        )
        acc = _onnx_accuracy(Path(outputs["sklearn_onnx_head_fp32"]), X_test, y_test)
        assert acc >= 0.80, f"sklearn accuracy too low: {acc:.3f}"

    def test_sklearn_report_content(
            self, dft_foundation_cfg, e2e_dataset, tmp_path):
        outputs = _run_pipeline(dft_foundation_cfg, e2e_dataset,
                                tmp_path, classifier="sklearn")
        report = Path(outputs["sklearn_report"]).read_text()
        assert "bird" in report
        assert "frog" in report
        assert "background" in report
        assert "precision" in report.lower()


@pytest.mark.slow
class TestE2EKerasAccuracy:
    def test_keras_achieves_high_accuracy(
            self, dft_foundation_cfg, e2e_dataset, tmp_path):
        outputs = _run_pipeline(dft_foundation_cfg, e2e_dataset,
                                tmp_path, classifier="keras")
        X_test, y_test, _ = _load_test_embeddings(
            Path(outputs["model_info"]),
            Path(outputs["dataset_list"]),
            dft_foundation_cfg,
        )
        acc = _onnx_accuracy(Path(outputs["keras_onnx_head_fp32"]), X_test, y_test)
        assert acc >= 0.80, f"keras accuracy too low: {acc:.3f}"


@pytest.mark.slow
class TestE2EComparison:
    """Both classifiers trained together; compare metrics and agreement."""

    @pytest.fixture(scope="class")
    def both_outputs(self, dft_foundation_cfg, e2e_dataset, tmp_path_factory):
        out = tmp_path_factory.mktemp("e2e_both")
        return _run_pipeline(
            dft_foundation_cfg, e2e_dataset, out, classifier="both"
        )

    def test_comparison_report_contains_both_classifiers(self, both_outputs):
        report = Path(both_outputs["comparison_report"]).read_text()
        assert "keras" in report.lower()
        assert "sklearn" in report.lower()

    def test_comparison_report_contains_accuracy_line(self, both_outputs):
        report = Path(both_outputs["comparison_report"]).read_text()
        assert "accuracy" in report.lower()

    def test_classifiers_agree_above_chance(self, dft_foundation_cfg, both_outputs):
        """The two classifiers should agree on more than chance (~1/3 for 3 classes)."""
        import onnxruntime as ort
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, EMBED_DIM)).astype(np.float32)

        def _predict(path):
            sess = ort.InferenceSession(str(path))
            in_name = sess.get_inputs()[0].name
            out_names = [o.name for o in sess.get_outputs()]
            target = "probabilities" if "probabilities" in out_names else out_names[-1]
            probs = sess.run([target], {in_name: X})[0]
            return np.argmax(probs, axis=1)

        preds_keras   = _predict(both_outputs["keras_onnx_head_fp32"])
        preds_sklearn = _predict(both_outputs["sklearn_onnx_head_fp32"])
        agreement = np.mean(preds_keras == preds_sklearn)
        assert agreement > 0.5, f"Classifier agreement too low: {agreement:.3f}"


@pytest.mark.slow
class TestE2EOutputPrecision:
    """Multiple output precisions produce separately-tagged, runnable files."""

    @pytest.fixture(scope="class")
    def prec_outputs(self, dft_foundation_cfg, e2e_dataset, tmp_path_factory):
        out = tmp_path_factory.mktemp("e2e_prec")
        return _run_pipeline(
            dft_foundation_cfg, e2e_dataset, out, classifier="both",
            output_type="head", output_format="both",
            data_types=["FP32", "FP16", "INT8"],
        )

    def test_keras_precision_files_written(self, prec_outputs):
        # ONNX and TFLite keras heads exist for every requested precision.
        for dt in ("fp32", "fp16", "int8"):
            assert Path(prec_outputs[f"keras_onnx_head_{dt}"]).exists()
            assert Path(prec_outputs[f"keras_tflite_head_{dt}"]).exists()

    def test_sklearn_only_fp32(self, prec_outputs):
        # sklearn ONNX is FP32-only; FP16/INT8 keys are absent.
        assert Path(prec_outputs["sklearn_onnx_head_fp32"]).exists()
        assert "sklearn_onnx_head_fp16" not in prec_outputs
        assert "sklearn_onnx_head_int8" not in prec_outputs

    def test_metadata_lists_precisions(self, prec_outputs):
        info = json.loads(Path(prec_outputs["model_info"]).read_text())
        assert info["output_data_types"] == ["FP32", "FP16", "INT8"]

    def test_fp16_and_int8_onnx_runnable(self, prec_outputs, dft_foundation_cfg):
        X_test, y_test, _ = _load_test_embeddings(
            Path(prec_outputs["model_info"]),
            Path(prec_outputs["dataset_list"]),
            dft_foundation_cfg,
        )
        # FP16/INT8 keras heads keep float32 I/O and stay accurate.
        for dt in ("fp16", "int8"):
            acc = _onnx_accuracy(Path(prec_outputs[f"keras_onnx_head_{dt}"]),
                                 X_test, y_test)
            assert acc >= 0.80, f"{dt} accuracy too low: {acc:.3f}"


@pytest.mark.slow
class TestE2EFullModel:
    """Test that head and full ONNX models produce consistent predictions."""

    def test_head_and_full_model_predictions_match(
            self, dft_foundation_cfg, e2e_dataset, tmp_path):
        import onnxruntime as ort
        outputs = _run_pipeline(
            dft_foundation_cfg, e2e_dataset, tmp_path,
            classifier="sklearn", output_type="both",
        )
        X_test, y_test, _ = _load_test_embeddings(
            Path(outputs["model_info"]),
            Path(outputs["dataset_list"]),
            dft_foundation_cfg,
        )

        sess_head = ort.InferenceSession(str(outputs["sklearn_onnx_head_fp32"]))
        in_name_h = sess_head.get_inputs()[0].name
        probs_head = sess_head.run(["probabilities"], {in_name_h: X_test})[0]

        with Path(outputs["dataset_list"]).open() as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

        from bioaccx.audio import load_mono, to_fixed_length
        sess_full = ort.InferenceSession(str(outputs["sklearn_onnx_full_fp32"]))
        in_name_f = sess_full.get_inputs()[0].name
        out_names_full = [o.name for o in sess_full.get_outputs()]
        prob_out = "probabilities" if "probabilities" in out_names_full else out_names_full[-1]
        probs_full_list = []
        for row in rows:
            audio = load_mono(Path(row["filepath"]), SAMPLE_RATE,
                              offset=float(row["start_time"]) if row["start_time"] else 0.0)
            audio = to_fixed_length(audio, WINDOW_SAMPLES)
            out = sess_full.run([prob_out], {in_name_f: audio[np.newaxis, :]})[0]
            probs_full_list.append(out[0])
        probs_full = np.stack(probs_full_list)

        preds_head = np.argmax(probs_head, axis=1)
        preds_full = np.argmax(probs_full, axis=1)
        agreement = np.mean(preds_head == preds_full)
        assert agreement == pytest.approx(1.0, abs=0.02), \
            f"Head vs full model agreement: {agreement:.3f}"


@pytest.mark.slow
class TestE2EExcludeLabels:
    """Exclude 'background' from output; classifier must still train on it."""

    def test_remaining_classes_still_accurate(
            self, dft_foundation_cfg, e2e_dataset, tmp_path):
        outputs = _run_pipeline(
            dft_foundation_cfg, e2e_dataset, tmp_path,
            classifier="sklearn", exclude_labels=["background"],
        )
        X_test, y_test, _ = _load_test_embeddings(
            Path(outputs["model_info"]),
            Path(outputs["dataset_list"]),
            dft_foundation_cfg,
        )
        acc = _onnx_accuracy(Path(outputs["sklearn_onnx_head_fp32"]), X_test, y_test,
                              output_tensor="probabilities_filtered")
        assert acc >= 0.80, f"accuracy after exclusion: {acc:.3f}"
