"""Exporting a softmax-trained head with exclude_labels.

The exporters append a Gather that drops the excluded columns.  For a flat
softmax head that Gather must run on probabilities, not logits, or the excluded
classes' share of the softmax is lost.  These tests run the same export
preparation as ``train.run`` (``_prepare_keras_export``) on a small 4-class
head and check every export path against the Keras probabilities.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.conftest import EMBED_DIM, WINDOW_SAMPLES

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

N_CLASSES = 4
KEEP = [0, 1, 3]            # class 2 excluded


def _head(output_activation):
    """A 4-class head laid out like train_keras builds it (weights, no training)."""
    import tensorflow as tf

    inp = tf.keras.Input(shape=(EMBED_DIM,), name="embedding")
    scores = tf.keras.layers.Dense(N_CLASSES, name="scores")
    x = scores(inp)
    if output_activation == "grouped_softmax":
        from bioaccx.trainers.grouped import grouped_softmax_layer
        x = grouped_softmax_layer([(0, 2), (2, 4)], name="output_activation")(x)
    elif output_activation is not None:
        x = tf.keras.layers.Activation(output_activation, name="output_activation")(x)
    model = tf.keras.Model(inp, x)
    rng = np.random.default_rng(7)
    scores.set_weights([
        (rng.standard_normal((EMBED_DIM, N_CLASSES)) * 2.0).astype(np.float32),
        rng.standard_normal(N_CLASSES).astype(np.float32),
    ])
    return model


def _logits(model, X):
    import tensorflow as tf
    return tf.keras.Model(model.input, model.get_layer("scores").output).predict(X, verbose=0)


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _embeddings(n=16):
    return np.random.default_rng(3).standard_normal((n, EMBED_DIM)).astype(np.float32)


def _prepare(model, output_activation, *, export_logits=False, grouped=False, keep=KEEP):
    from bioaccx.train import _prepare_keras_export
    return _prepare_keras_export(model, output_activation, export_logits, grouped,
                                 keep, N_CLASSES)


def _run_tflite(path, X):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path))
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.resize_tensor_input(inp["index"], X.shape)
    interp.allocate_tensors()
    interp.set_tensor(inp["index"], X.astype(np.float32))
    interp.invoke()
    return interp.get_tensor(out["index"])


def _run_onnx(path, X):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    return sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[0]


@pytest.fixture(scope="module")
def tflite_backbone(tmp_path_factory):
    """A tiny TFLite 'backbone' (audio window → embedding) for full-model export."""
    import tensorflow as tf
    inp = tf.keras.Input(shape=(WINDOW_SAMPLES,), name="audio")
    model = tf.keras.Model(inp, tf.keras.layers.Dense(EMBED_DIM, name="embed")(inp))
    rng = np.random.default_rng(5)
    model.get_layer("embed").set_weights([
        (rng.standard_normal((WINDOW_SAMPLES, EMBED_DIM)) * 0.05).astype(np.float32),
        np.zeros(EMBED_DIM, np.float32),
    ])
    path = tmp_path_factory.mktemp("backbone") / "backbone.tflite"
    path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
    return path, model


# ---------------------------------------------------------------------------
# Softmax heads: exported kept columns == trained softmax probabilities
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("activation", [None, "softmax"])
class TestSoftmaxHeadWithExcludedClass:
    def test_tflite_head_emits_trained_probabilities(self, activation, tmp_path):
        from bioaccx.exporters.tflite_exporter import export_tflite
        model, X = _head(activation), _embeddings()
        export_model, _ = _prepare(model, activation)
        path = export_tflite(export_model, "keras", EMBED_DIM, tmp_path / "h.tflite",
                             keep_indices=KEEP)
        expected = _softmax(_logits(model, X))[:, KEEP]
        np.testing.assert_allclose(_run_tflite(path, X), expected, atol=1e-5)

    def test_onnx_head_emits_trained_probabilities(self, activation, tmp_path):
        from bioaccx.exporters.onnx_exporter import export_onnx
        model, X = _head(activation), _embeddings()
        export_model, _ = _prepare(model, activation)
        path = export_onnx(export_model, "keras", EMBED_DIM, tmp_path / "h.onnx",
                           keep_indices=KEEP)
        expected = _softmax(_logits(model, X))[:, KEEP]
        np.testing.assert_allclose(_run_onnx(path, X), expected, atol=1e-5)

    def test_tflite_full_emits_trained_probabilities(self, activation, tmp_path,
                                                     tflite_backbone):
        from bioaccx.exporters.tflite_exporter import export_tflite
        backbone_path, backbone = tflite_backbone
        model = _head(activation)
        export_model, _ = _prepare(model, activation)
        path = export_tflite(export_model, "keras", EMBED_DIM, tmp_path / "f.tflite",
                             foundation_path=backbone_path, output_type="full",
                             keep_indices=KEEP)
        audio = np.random.default_rng(4).standard_normal((4, WINDOW_SAMPLES)).astype(np.float32)
        emb = backbone.predict(audio, verbose=0)
        expected = _softmax(_logits(model, emb))[:, KEEP]
        np.testing.assert_allclose(_run_tflite(path, audio), expected, atol=1e-4)

    def test_onnx_full_emits_trained_probabilities(self, activation, tmp_path,
                                                   dummy_onnx_model):
        from bioaccx.exporters.onnx_exporter import export_onnx
        model = _head(activation)
        export_model, _ = _prepare(model, activation)
        path = export_onnx(export_model, "keras", EMBED_DIM, tmp_path / "f.onnx",
                           foundation_onnx_path=dummy_onnx_model, foundation_input_name="INPUT",
                           output_type="full", keep_indices=KEEP)
        audio = np.random.default_rng(4).standard_normal((4, WINDOW_SAMPLES)).astype(np.float32)
        emb = _run_onnx(dummy_onnx_model, audio)
        expected = _softmax(_logits(model, emb))[:, KEEP]
        np.testing.assert_allclose(_run_onnx(path, audio), expected, atol=1e-5)

    def test_metadata_says_probabilities(self, activation):
        _, meta = _prepare(_head(activation), activation)
        assert meta["scores"] == "probabilities"
        assert meta["activation"] == "identity"


@pytest.mark.slow
class TestSoftmaxExportDecisions:
    def test_softmax_export_logits_with_excluded_class_is_rejected(self):
        with pytest.raises(ValueError, match="export_logits cannot be combined"):
            _prepare(_head("softmax"), "softmax", export_logits=True)

    def test_softmax_export_logits_without_exclusion_still_emits_logits(self):
        model = _head("softmax")
        export_model, meta = _prepare(model, "softmax", export_logits=True, keep=None)
        X = _embeddings()
        np.testing.assert_allclose(export_model.predict(X, verbose=0), _logits(model, X),
                                   atol=1e-5)
        assert meta == {"scores": "logits", "activation": "softmax"}

    def test_null_head_without_exclusion_keeps_emitting_logits(self):
        model = _head(None)
        export_model, meta = _prepare(model, None, keep=None)
        assert export_model is model
        assert meta == {"scores": "logits", "activation": "softmax"}

    def test_insertion_prints_a_note(self, capsys):
        _prepare(_head(None), None)
        assert "inserting a softmax over all 4 trained classes" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Sigmoid and grouped heads: unchanged
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestOtherHeadsUnchanged:
    def test_sigmoid_head_is_not_wrapped(self, tmp_path):
        from bioaccx.exporters.tflite_exporter import export_tflite
        model, X = _head("sigmoid"), _embeddings()
        export_model, meta = _prepare(model, "sigmoid")
        assert export_model is model
        assert meta == {"scores": "probabilities", "activation": "identity"}
        path = export_tflite(export_model, "keras", EMBED_DIM, tmp_path / "s.tflite",
                             keep_indices=KEEP)
        expected = 1.0 / (1.0 + np.exp(-_logits(model, X)[:, KEEP]))
        np.testing.assert_allclose(_run_tflite(path, X), expected, atol=1e-5)

    def test_sigmoid_export_logits_gathers_logits(self):
        model = _head("sigmoid")
        export_model, meta = _prepare(model, "sigmoid", export_logits=True)
        X = _embeddings()
        np.testing.assert_allclose(export_model.predict(X, verbose=0), _logits(model, X),
                                   atol=1e-5)
        assert meta == {"scores": "logits", "activation": "sigmoid"}

    def test_grouped_head_is_not_wrapped(self, tmp_path):
        from bioaccx.exporters.onnx_exporter import export_onnx
        model, X = _head("grouped_softmax"), _embeddings()
        export_model, meta = _prepare(model, "grouped_softmax", grouped=True)
        assert export_model is model
        assert meta == {"scores": "probabilities", "activation": "identity"}
        path = export_onnx(export_model, "keras", EMBED_DIM, tmp_path / "g.onnx",
                           keep_indices=KEEP)
        z = _logits(model, X)
        expected = np.concatenate([_softmax(z[:, 0:2]), _softmax(z[:, 2:4])], axis=1)[:, KEEP]
        np.testing.assert_allclose(_run_onnx(path, X), expected, atol=1e-5)
