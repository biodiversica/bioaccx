"""Unit tests for bioaccx.trainers.keras_trainer."""
from __future__ import annotations

import numpy as np
import pytest

from bioaccx.config import KerasConfig
from bioaccx.trainers.keras_trainer import strip_output_activation, train_keras


def _head(embed_dim: int = 8, n_classes: int = 3, activation: str | None = "sigmoid"):
    """Build a head with the same layer names train_keras uses."""
    import tensorflow as tf
    inp = tf.keras.Input(shape=(embed_dim,), name="embedding")
    x = tf.keras.layers.Dense(n_classes, activation=None, name="scores")(inp)
    if activation:
        x = tf.keras.layers.Activation(activation, name="output_activation")(x)
    return tf.keras.Model(inp, x)


def _toy_data(n: int = 60, embed_dim: int = 8, n_classes: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    X = rng.normal(size=(n, embed_dim)).astype(np.float32)
    X[np.arange(n), y % embed_dim] += 4.0     # make the classes separable
    return X, y


class TestStripOutputActivation:
    def test_sigmoid_head_becomes_logits(self):
        model = _head(activation="sigmoid")
        stripped = strip_output_activation(model)
        assert stripped is not model
        assert [layer.name for layer in stripped.layers][-1] == "scores"

    def test_stripped_output_is_the_inverse_sigmoid(self):
        from scipy.special import expit
        model = _head(activation="sigmoid")
        X = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
        assert expit(strip_output_activation(model).predict(X, verbose=0)) == pytest.approx(
            model.predict(X, verbose=0), abs=1e-5
        )

    def test_softmax_head_can_also_be_stripped(self):
        model = _head(activation="softmax")
        stripped = strip_output_activation(model)
        X = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)
        logits = stripped.predict(X, verbose=0)
        probs = model.predict(X, verbose=0)
        # argmax is unchanged; the raw values are not probabilities any more.
        assert np.array_equal(logits.argmax(1), probs.argmax(1))
        assert not np.allclose(logits.sum(1), 1.0)

    def test_linear_head_is_returned_unchanged(self):
        model = _head(activation=None)
        assert strip_output_activation(model) is model

    def test_weights_are_shared_not_copied(self):
        model = _head(activation="sigmoid")
        stripped = strip_output_activation(model)
        assert stripped.get_layer("scores") is model.get_layer("scores")

    def test_model_without_scores_layer_is_returned_unchanged(self):
        import tensorflow as tf
        inp = tf.keras.Input(shape=(4,))
        model = tf.keras.Model(inp, tf.keras.layers.Dense(2, name="other")(inp))
        assert strip_output_activation(model) is model


class TestExportLogitsReporting:
    def test_recorded_when_activation_present(self):
        X, y = _toy_data()
        cfg = KerasConfig(hidden_units=0, epochs=1, output_activation="sigmoid",
                          export_logits=True)
        model = train_keras(X, y, X[:20], y[:20], ["a", "b", "c"], cfg, seed=1)
        assert model._report_params["export_logits"] is True

    def test_not_recorded_when_head_is_already_linear(self):
        """export_logits is a no-op without an activation layer to strip."""
        X, y = _toy_data()
        cfg = KerasConfig(hidden_units=0, epochs=1, output_activation=None,
                          export_logits=True)
        model = train_keras(X, y, X[:20], y[:20], ["a", "b", "c"], cfg, seed=1)
        assert model._report_params["export_logits"] is False
        assert strip_output_activation(model) is model
