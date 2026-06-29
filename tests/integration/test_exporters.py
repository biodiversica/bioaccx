"""Integration tests for ONNX and TFLite exporters.

These tests build minimal Keras / sklearn models directly (no full pipeline)
and verify that the exporter functions produce valid, runnable models.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sklearn_pipe(n_classes: int = 3, embed_dim: int = EMBED_DIM):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, embed_dim)).astype(np.float32)
    y = np.tile(np.arange(n_classes), 60 // n_classes + 1)[:60]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=100))])
    pipe.fit(X, y)
    return pipe, X


def _make_keras_model(n_classes: int = 3, embed_dim: int = EMBED_DIM):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, embed_dim)).astype(np.float32)
    y = np.tile(np.arange(n_classes), 60 // n_classes + 1)[:60]
    y_oh = tf.keras.utils.to_categorical(y, n_classes).astype(np.float32)

    inp = tf.keras.Input(shape=(embed_dim,), name="embedding")
    out = tf.keras.layers.Dense(n_classes, name="scores")(inp)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y_oh, epochs=1, verbose=0)
    return model, X


# ---------------------------------------------------------------------------
# ONNX exporter
# ---------------------------------------------------------------------------

class TestONNXExporter:
    def test_sklearn_head_export(self, tmp_path):
        from bioaccx.exporters.onnx_exporter import export_onnx
        import onnxruntime as ort

        pipe, X = _make_sklearn_pipe()
        out_path = tmp_path / "sklearn_head.onnx"
        export_onnx(pipe, "sklearn", EMBED_DIM, out_path)

        assert out_path.exists()
        sess = ort.InferenceSession(str(out_path))
        out_names = [o.name for o in sess.get_outputs()]
        assert "probabilities" in out_names

    def test_sklearn_inference_shape(self, tmp_path):
        from bioaccx.exporters.onnx_exporter import export_onnx
        import onnxruntime as ort

        n_classes = 3
        pipe, X = _make_sklearn_pipe(n_classes)
        out_path = tmp_path / "sklearn_head.onnx"
        export_onnx(pipe, "sklearn", EMBED_DIM, out_path)

        sess = ort.InferenceSession(str(out_path))
        probs = sess.run(["probabilities"], {"float_input": X[:5]})[0]
        assert probs.shape == (5, n_classes)

    def test_keras_head_export(self, tmp_path):
        from bioaccx.exporters.onnx_exporter import export_onnx
        import onnxruntime as ort

        model, _ = _make_keras_model()
        out_path = tmp_path / "keras_head.onnx"
        export_onnx(model, "keras", EMBED_DIM, out_path)

        assert out_path.exists()
        sess = ort.InferenceSession(str(out_path))
        assert len(sess.get_outputs()) == 1

    def test_keras_inference_shape(self, tmp_path):
        from bioaccx.exporters.onnx_exporter import export_onnx
        import onnxruntime as ort

        n_classes = 3
        model, X = _make_keras_model(n_classes)
        out_path = tmp_path / "keras_head.onnx"
        export_onnx(model, "keras", EMBED_DIM, out_path)

        sess = ort.InferenceSession(str(out_path))
        in_name = sess.get_inputs()[0].name
        scores = sess.run(None, {in_name: X[:4]})[0]
        assert scores.shape == (4, n_classes)

    def test_filter_output_labels_keras(self, tmp_path):
        """keep_indices=[0, 2] on a 3-class model → 2 output columns."""
        from bioaccx.exporters.onnx_exporter import export_onnx
        import onnxruntime as ort

        model, X = _make_keras_model(n_classes=3)
        out_path = tmp_path / "keras_filtered.onnx"
        export_onnx(model, "keras", EMBED_DIM, out_path, keep_indices=[0, 2])

        sess = ort.InferenceSession(str(out_path))
        in_name = sess.get_inputs()[0].name
        scores = sess.run(None, {in_name: X[:4]})[0]
        assert scores.shape == (4, 2)

    def test_filter_output_labels_sklearn(self, tmp_path):
        """keep_indices=[1] on a 3-class sklearn model → probabilities shape (N, 1)."""
        from bioaccx.exporters.onnx_exporter import export_onnx
        import onnxruntime as ort

        pipe, X = _make_sklearn_pipe(n_classes=3)
        out_path = tmp_path / "sklearn_filtered.onnx"
        export_onnx(pipe, "sklearn", EMBED_DIM, out_path, keep_indices=[1])

        sess = ort.InferenceSession(str(out_path))
        in_name = sess.get_inputs()[0].name
        probs = sess.run(None, {in_name: X[:5]})[0]
        assert probs.shape == (5, 1)

    def test_full_model_export_merges_graphs(self, tmp_path, dummy_onnx_model):
        """Export full model (foundation + keras head) and run inference."""
        from bioaccx.exporters.onnx_exporter import export_onnx
        import onnxruntime as ort
        from tests.conftest import WINDOW_SAMPLES, SAMPLE_RATE

        model, _ = _make_keras_model(n_classes=3)
        out_path = tmp_path / "full_model.onnx"
        export_onnx(
            model, "keras", EMBED_DIM, out_path,
            foundation_onnx_path=dummy_onnx_model,
            foundation_input_name="INPUT",
            output_type="full",
        )

        assert out_path.exists()
        sess = ort.InferenceSession(str(out_path))
        # Input is raw audio [batch, window_samples]
        audio_batch = np.zeros((2, WINDOW_SAMPLES), dtype=np.float32)
        result = sess.run(None, {"INPUT": audio_batch})
        assert result[0].shape == (2, 3)

    def test_full_model_without_foundation_raises(self, tmp_path):
        from bioaccx.exporters.onnx_exporter import export_onnx

        model, _ = _make_keras_model()
        with pytest.raises(ValueError, match="foundation_onnx_path"):
            export_onnx(model, "keras", EMBED_DIM, tmp_path / "out.onnx", output_type="full")

    def test_keras_head_fp16_runs(self, tmp_path):
        """FP16 ONNX head keeps float32 I/O (keep_io_types) and runs with correct shape."""
        from bioaccx.exporters.onnx_exporter import export_onnx
        import onnxruntime as ort

        n_classes = 3
        model, X = _make_keras_model(n_classes)
        out_path = tmp_path / "keras_head_fp16.onnx"
        export_onnx(model, "keras", EMBED_DIM, out_path, data_type="FP16")

        assert out_path.exists()
        sess = ort.InferenceSession(str(out_path))
        in_name = sess.get_inputs()[0].name
        scores = sess.run(None, {in_name: X[:4]})[0]
        assert scores.shape == (4, n_classes)

    def test_keras_head_int8_runs(self, tmp_path):
        """INT8 (dynamic) ONNX head inserts integer-matmul ops and runs correctly."""
        from bioaccx.exporters.onnx_exporter import export_onnx
        import onnx
        import onnxruntime as ort

        n_classes = 3
        model, X = _make_keras_model(n_classes)
        int8_path = tmp_path / "keras_head_int8.onnx"
        export_onnx(model, "keras", EMBED_DIM, int8_path, data_type="INT8")

        assert int8_path.exists()
        # Dynamic quantization rewrites MatMul/Gemm into integer ops.
        op_types = {n.op_type for n in onnx.load(str(int8_path)).graph.node}
        assert op_types & {"MatMulInteger", "DynamicQuantizeLinear", "QLinearMatMul"}
        sess = ort.InferenceSession(str(int8_path))
        in_name = sess.get_inputs()[0].name
        scores = sess.run(None, {in_name: X[:4]})[0]
        assert scores.shape == (4, n_classes)


# ---------------------------------------------------------------------------
# TFLite exporter
# ---------------------------------------------------------------------------

class TestTFLiteExporter:
    def test_keras_head_tflite_export(self, tmp_path):
        from bioaccx.exporters.tflite_exporter import export_tflite
        import tensorflow as tf

        model, X = _make_keras_model(n_classes=3)
        out_path = tmp_path / "head.tflite"
        result = export_tflite(model, "keras", EMBED_DIM, out_path)

        assert result is not None
        assert result.exists()
        assert result.stat().st_size > 0

    def test_tflite_head_inference(self, tmp_path):
        from bioaccx.exporters.tflite_exporter import export_tflite
        import tensorflow as tf

        n_classes = 3
        model, X = _make_keras_model(n_classes)
        out_path = tmp_path / "head.tflite"
        export_tflite(model, "keras", EMBED_DIM, out_path)

        interp = tf.lite.Interpreter(model_path=str(out_path))
        interp.allocate_tensors()
        inp_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]
        interp.set_tensor(inp_idx, X[:1])
        interp.invoke()
        scores = interp.get_tensor(out_idx)
        assert scores.shape == (1, n_classes)

    def test_sklearn_tflite_returns_none(self, tmp_path, capsys):
        from bioaccx.exporters.tflite_exporter import export_tflite
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # sklearn + TFLite is unsupported — should return None, not raise
        pipe = Pipeline([("s", StandardScaler())])
        result = export_tflite(pipe, "sklearn", EMBED_DIM, tmp_path / "out.tflite")
        assert result is None
        assert "skip" in capsys.readouterr().out.lower()

    def test_full_tflite_skipped_without_savedmodel(self, tmp_path, capsys):
        from bioaccx.exporters.tflite_exporter import export_tflite

        model, _ = _make_keras_model()
        result = export_tflite(
            model, "keras", EMBED_DIM, tmp_path / "full.tflite",
            foundation_path=None,
            output_type="full",
        )
        assert result is None
        out = capsys.readouterr().out
        assert "skip" in out.lower() or "protobuf" in out.lower()

    @pytest.mark.parametrize("data_type", ["FP32", "FP16", "INT8"])
    def test_head_tflite_precision_runs(self, tmp_path, data_type):
        """Each output precision produces a runnable TFLite head with correct shape."""
        from bioaccx.exporters.tflite_exporter import export_tflite
        import tensorflow as tf

        n_classes = 3
        model, X = _make_keras_model(n_classes)
        out_path = tmp_path / f"head_{data_type.lower()}.tflite"
        result = export_tflite(model, "keras", EMBED_DIM, out_path, data_type=data_type)

        assert result is not None and result.exists()
        interp = tf.lite.Interpreter(model_path=str(out_path))
        interp.allocate_tensors()
        inp_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]
        interp.set_tensor(inp_idx, X[:1])
        interp.invoke()
        scores = interp.get_tensor(out_idx)
        assert scores.shape == (1, n_classes)

    def test_keras_filtered_tflite_output_shape(self, tmp_path):
        from bioaccx.exporters.tflite_exporter import export_tflite
        import tensorflow as tf

        n_classes = 4
        model, X = _make_keras_model(n_classes)
        out_path = tmp_path / "filtered.tflite"
        export_tflite(model, "keras", EMBED_DIM, out_path, keep_indices=[0, 2])

        interp = tf.lite.Interpreter(model_path=str(out_path))
        interp.allocate_tensors()
        inp_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]
        interp.set_tensor(inp_idx, X[:1])
        interp.invoke()
        scores = interp.get_tensor(out_idx)
        assert scores.shape == (1, 2)  # only 2 of the 4 classes kept
