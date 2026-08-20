"""End-to-end tests for the CLI-driven model surgery modes (no config file).

``--merge`` and ``--extract-head`` accept their few inputs as arguments; these
tests drive them exactly as a user would and check both the plumbing (where
files land) and the numerics (the produced model behaves like its source).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bioaccx.cli import main
from tests.conftest import EMBED_DIM, WINDOW_SAMPLES
from tests.unit.test_extract_head import _embedding_of, _make_full_like_tflite


N_CLASSES = 4


@pytest.fixture
def head_onnx(tmp_path) -> Path:
    """A small ONNX classifier head over the dummy backbone's embedding."""
    import tensorflow as tf
    from bioaccx.exporters.onnx_exporter import export_onnx

    rng = np.random.default_rng(0)
    inp = tf.keras.layers.Input(shape=(EMBED_DIM,), name="embedding")
    out = tf.keras.layers.Dense(N_CLASSES, name="scores")(inp)
    model = tf.keras.Model(inp, out)
    W, b = model.get_layer("scores").get_weights()
    model.get_layer("scores").set_weights([W, rng.standard_normal(b.shape).astype(np.float32)])

    path = tmp_path / "my_head.onnx"
    export_onnx(model, "keras", EMBED_DIM, path)
    return path


@pytest.fixture
def full_tflite(tmp_path) -> Path:
    """A BirdNET-Analyzer-shaped model: backbone-ish frontend + dense head."""
    path = tmp_path / "custom_model.tflite"
    _make_full_like_tflite(path, EMBED_DIM, N_CLASSES, hidden=8)
    return path


class TestMergeCli:
    def test_merged_model_matches_backbone_then_head(
            self, dummy_onnx_model, head_onnx, tmp_path):
        import onnxruntime as ort

        out = tmp_path / "merged.onnx"
        main(["--merge", str(head_onnx), "--backbone", str(dummy_onnx_model), "-o", str(out)])
        assert out.exists()

        audio = np.random.default_rng(1).standard_normal((2, WINDOW_SAMPLES)).astype(np.float32)
        backbone = ort.InferenceSession(str(dummy_onnx_model))
        head = ort.InferenceSession(str(head_onnx))
        merged = ort.InferenceSession(str(out))

        emb = backbone.run(None, {backbone.get_inputs()[0].name: audio})[0]
        expected = head.run(None, {head.get_inputs()[0].name: emb})[0]
        got = merged.run(None, {merged.get_inputs()[0].name: audio})[0]

        assert got.shape == (2, N_CLASSES)
        assert got == pytest.approx(expected, abs=1e-4)

    def test_default_output_next_to_head(self, dummy_onnx_model, head_onnx, tmp_path):
        main(["--merge", str(head_onnx), "--backbone", str(dummy_onnx_model)])
        assert (tmp_path / "my_head_full.onnx").exists()

    def test_local_backbone_keeps_its_input_name(
            self, dummy_onnx_model, head_onnx, tmp_path):
        """Nothing in the CLI knows the tensor name, so it must come from the file."""
        import onnxruntime as ort
        out = tmp_path / "merged.onnx"
        main(["--merge", str(head_onnx), "--backbone", str(dummy_onnx_model), "-o", str(out)])
        assert ort.InferenceSession(str(out)).get_inputs()[0].name == "INPUT"


class TestExtractHeadCli:
    def _labels_file(self, model: Path) -> Path:
        path = model.with_name(f"{model.stem}_Labels.txt")
        path.write_text("".join(f"Genus sp{i}_Class{i}\n" for i in range(N_CLASSES)))
        return path

    def test_writes_head_directory_next_to_source(self, full_tflite, tmp_path):
        self._labels_file(full_tflite)
        main(["--extract-head", str(full_tflite), "--embed-dim", str(EMBED_DIM)])

        out_dir = tmp_path / "custom_model_head"
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "custom_model_extract_head.json",
            "custom_model_head.tflite",
            "custom_model_head_fp32.onnx",
            "custom_model_labels.txt",
        ]
        labels = (out_dir / "custom_model_labels.txt").read_text().split()
        assert labels == [f"Class{i}" for i in range(N_CLASSES)]

    def test_extracted_head_matches_the_full_model(self, full_tflite, tmp_path):
        """The ONNX head reproduces the source model's own output."""
        import onnxruntime as ort

        main(["--extract-head", str(full_tflite), "--embed-dim", str(EMBED_DIM),
              "-o", str(tmp_path / "head"), "--format", "onnx"])
        head_path = tmp_path / "head" / "custom_model_head_fp32.onnx"

        x = np.random.default_rng(2).standard_normal((1, EMBED_DIM)).astype(np.float32)
        full_out, embedding = _embedding_of(None, full_tflite, EMBED_DIM, x)

        sess = ort.InferenceSession(str(head_path))
        head_out = sess.run(None, {sess.get_inputs()[0].name: embedding})[0]
        assert head_out == pytest.approx(full_out, abs=1e-4)

    def test_format_onnx_skips_the_tflite_head(self, full_tflite, tmp_path):
        main(["--extract-head", str(full_tflite), "--embed-dim", str(EMBED_DIM),
              "--format", "onnx"])
        out_dir = tmp_path / "custom_model_head"
        assert not (out_dir / "custom_model_head.tflite").exists()
        assert (out_dir / "custom_model_head_fp32.onnx").exists()

    def test_extracts_from_a_full_onnx_model(self, dummy_onnx_model, head_onnx, tmp_path):
        """head → merge → extract must give back the original head."""
        import onnxruntime as ort

        full = tmp_path / "merged.onnx"
        main(["--merge", str(head_onnx), "--backbone", str(dummy_onnx_model), "-o", str(full)])
        main(["--extract-head", str(full), "--embed-dim", str(EMBED_DIM), "--format", "onnx"])

        extracted = tmp_path / "merged_head" / "merged_head_fp32.onnx"
        assert extracted.exists()

        x = np.random.default_rng(3).standard_normal((5, EMBED_DIM)).astype(np.float32)
        def _run(path):
            sess = ort.InferenceSession(str(path))
            return sess.run(None, {sess.get_inputs()[0].name: x})[0]
        assert _run(extracted) == pytest.approx(_run(head_onnx), abs=1e-4)

    def test_onnx_source_head_reads_the_embedding(
            self, dummy_onnx_model, head_onnx, tmp_path):
        """The extracted head must start exactly at the backbone's output.

        A head that swallowed a trailing backbone op would still agree with the
        full model, but would no longer accept a real embedding — so the check
        runs the backbone and feeds its output to both heads.
        """
        import onnxruntime as ort

        full = tmp_path / "merged.onnx"
        main(["--merge", str(head_onnx), "--backbone", str(dummy_onnx_model), "-o", str(full)])
        main(["--extract-head", str(full), "--embed-dim", str(EMBED_DIM), "--format", "onnx"])

        audio = np.random.default_rng(4).standard_normal((3, WINDOW_SAMPLES)).astype(np.float32)
        backbone = ort.InferenceSession(str(dummy_onnx_model))
        embedding = backbone.run(None, {backbone.get_inputs()[0].name: audio})[0]

        def _run(path, x):
            sess = ort.InferenceSession(str(path))
            return sess.run(None, {sess.get_inputs()[0].name: x})[0]
        extracted = tmp_path / "merged_head" / "merged_head_fp32.onnx"
        full_out = _run(full, audio)
        assert _run(extracted, embedding) == pytest.approx(full_out, abs=1e-4)

    def test_metadata_records_source_and_labels(self, full_tflite, tmp_path):
        main(["--extract-head", str(full_tflite), "--embed-dim", str(EMBED_DIM)])
        info = json.loads(
            (tmp_path / "custom_model_head" / "custom_model_extract_head.json").read_text())
        assert info["source_model"] == str(full_tflite)
        assert info["embedding_size"] == EMBED_DIM
        assert info["n_classes"] == N_CLASSES
        assert info["backbone"] == "custom"      # no version to report from --embed-dim
