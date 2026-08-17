"""Unit tests for bioaccx.embedder (TFLite graph trimming)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bioaccx.config import FoundationModelConfig
from bioaccx.embedder import TFLiteEmbedder, _TRIM_CACHE, _trimmed_tflite_bytes

EMBED_DIM = 4
WINDOW = 8


@pytest.fixture(scope="module")
def multi_output_tflite(tmp_path_factory) -> Path:
    """A tflite model with an embedding output plus a much larger extra output.

    Stands in for the real backbones that ship a classifier head alongside the
    embedding (BirdNET, Perch v2).  The embedding must end up as the *first*
    tflite output, which is what the embedder resolves to; the converter does
    not necessarily preserve the Keras output order, so both orders are tried.
    """
    import tensorflow as tf
    tf.random.set_seed(0)

    def _build(swap: bool) -> bytes:
        inp = tf.keras.Input(shape=(WINDOW,), name="audio")
        embedding = tf.keras.layers.Dense(EMBED_DIM, activation="tanh", name="embedding")(inp)
        extra = tf.keras.layers.Dense(64, activation="relu", name="extra")(embedding)
        outputs = [extra, embedding] if swap else [embedding, extra]
        return tf.lite.TFLiteConverter.from_keras_model(tf.keras.Model(inp, outputs)).convert()

    for swap in (False, True):
        data = _build(swap)
        interp = tf.lite.Interpreter(model_content=data)
        interp.allocate_tensors()
        details = interp.get_output_details()
        if len(details) == 2 and int(details[0]["shape"][-1]) == EMBED_DIM:
            path = tmp_path_factory.mktemp("tflite") / "multi.tflite"
            path.write_bytes(data)
            return path
    pytest.skip("could not build a tflite whose first output is the embedding")


def _cfg(path: Path, trim: bool, embedding_size: int = EMBED_DIM) -> FoundationModelConfig:
    return FoundationModelConfig(
        name="test", format="tflite", source="local", path=str(path),
        sample_rate=WINDOW, window_samples=WINDOW, input_name="audio",
        embedding_size=embedding_size, tflite_trim_to_embedding=trim,
    )


@pytest.fixture
def audio() -> np.ndarray:
    return np.random.default_rng(0).normal(size=WINDOW).astype(np.float32)


class TestTFLiteTrimming:
    def test_trimming_does_not_change_the_embedding(self, multi_output_tflite, audio):
        full = TFLiteEmbedder(_cfg(multi_output_tflite, trim=False), multi_output_tflite)
        trimmed = TFLiteEmbedder(_cfg(multi_output_tflite, trim=True), multi_output_tflite)
        assert trimmed.embed(audio) == pytest.approx(full.embed(audio))

    def test_trimmed_graph_has_a_single_output(self, multi_output_tflite):
        full = TFLiteEmbedder(_cfg(multi_output_tflite, trim=False), multi_output_tflite)
        trimmed = TFLiteEmbedder(_cfg(multi_output_tflite, trim=True), multi_output_tflite)
        assert len(full._interp.get_output_details()) == 2
        assert len(trimmed._interp.get_output_details()) == 1

    def test_embedding_shape_is_preserved(self, multi_output_tflite, audio):
        emb = TFLiteEmbedder(_cfg(multi_output_tflite, trim=True), multi_output_tflite)
        assert emb.embed(audio).shape == (EMBED_DIM,)

    def test_falls_back_when_trimmed_output_does_not_match_embedding_size(
            self, multi_output_tflite, capsys):
        """A wrong embedding_size must not silently produce a mis-wired graph."""
        emb = TFLiteEmbedder(_cfg(multi_output_tflite, trim=True, embedding_size=999),
                             multi_output_tflite)
        assert len(emb._interp.get_output_details()) == 2   # untrimmed model kept
        assert "using the model as shipped" in capsys.readouterr().out

    def test_trim_is_cached_across_embedders(self, multi_output_tflite):
        _TRIM_CACHE.clear()
        for _ in range(3):
            TFLiteEmbedder(_cfg(multi_output_tflite, trim=True), multi_output_tflite)
        assert len(_TRIM_CACHE) == 1

    def test_failed_trim_returns_none_instead_of_raising(self, multi_output_tflite, capsys):
        assert _trimmed_tflite_bytes(multi_output_tflite, 99999) is None
        assert "could not trim" in capsys.readouterr().out

    def test_single_output_model_is_left_alone(self, tmp_path, audio):
        """Nothing to gain, so the graph is used as shipped."""
        import tensorflow as tf
        inp = tf.keras.Input(shape=(WINDOW,), name="audio")
        model = tf.keras.Model(inp, tf.keras.layers.Dense(EMBED_DIM, name="embedding")(inp))
        path = tmp_path / "single.tflite"
        path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())

        _TRIM_CACHE.clear()
        emb = TFLiteEmbedder(_cfg(path, trim=True), path)
        assert emb.embed(audio).shape == (EMBED_DIM,)
        assert _TRIM_CACHE == {}
