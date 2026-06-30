"""Unit tests for extracting a classifier head from a full TFLite model."""
import numpy as np
import pytest

from bioaccx.extract_head import (
    build_keras_head,
    extract_tflite_head,
    read_labels_file,
    slice_tflite_head,
)


def _make_full_like_tflite(path, embed_dim, n_classes, hidden=None, seed=0):
    """Build a tiny 'full-like' model (frontend → head) and save as TFLite.

    A relu ``frontend`` of width ``embed_dim`` stands in for the backbone's
    embedding output. With ``hidden`` set, the head is an MLP (hidden relu layer
    + linear classifier); otherwise it is a single linear classifier.
    """
    import tensorflow as tf

    tf.random.set_seed(seed)
    inp = tf.keras.Input(shape=(embed_dim,), name="emb")
    x = tf.keras.layers.Dense(embed_dim, activation="relu", name="frontend")(inp)
    if hidden:
        x = tf.keras.layers.Dense(hidden, activation="relu", name="hidden")(x)
    out = tf.keras.layers.Dense(n_classes, activation=None, name="scores")(x)
    model = tf.keras.Model(inp, out)
    data = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    path.write_bytes(data)
    return model


def _frontend_of(model):
    import tensorflow as tf
    return tf.keras.Model(model.input, model.get_layer("frontend").output)


class TestExtractLinearHead:
    def test_single_layer_shape_and_activation(self, tmp_path):
        embed_dim, n_classes = 8, 3
        _make_full_like_tflite(tmp_path / "full.tflite", embed_dim, n_classes)
        layers = extract_tflite_head(tmp_path / "full.tflite", embed_dim)
        assert len(layers) == 1
        assert layers[0].W.shape == (n_classes, embed_dim)
        assert layers[0].activation is None

    def test_rebuilt_head_matches_full_model(self, tmp_path):
        embed_dim, n_classes = 16, 4
        model = _make_full_like_tflite(tmp_path / "full.tflite", embed_dim, n_classes)
        layers = extract_tflite_head(tmp_path / "full.tflite", embed_dim)
        head = build_keras_head(layers, embed_dim)
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((5, embed_dim)).astype(np.float32)
        full_logits = model.predict(emb, verbose=0)
        head_logits = head.predict(_frontend_of(model).predict(emb, verbose=0), verbose=0)
        assert np.allclose(full_logits, head_logits, atol=1e-4)


class TestExtractMlpHead:
    def test_chain_layers_and_activations(self, tmp_path):
        embed_dim, hidden, n_classes = 8, 6, 3
        _make_full_like_tflite(tmp_path / "full.tflite", embed_dim, n_classes, hidden=hidden)
        layers = extract_tflite_head(tmp_path / "full.tflite", embed_dim)
        assert len(layers) == 2
        assert layers[0].W.shape == (hidden, embed_dim)
        assert layers[0].activation == "relu"
        assert layers[1].W.shape == (n_classes, hidden)
        assert layers[1].activation is None

    def test_rebuilt_mlp_head_matches_full_model(self, tmp_path):
        embed_dim, hidden, n_classes = 12, 7, 4
        model = _make_full_like_tflite(tmp_path / "full.tflite", embed_dim, n_classes, hidden=hidden)
        layers = extract_tflite_head(tmp_path / "full.tflite", embed_dim)
        head = build_keras_head(layers, embed_dim)
        rng = np.random.default_rng(1)
        emb = rng.standard_normal((6, embed_dim)).astype(np.float32)
        full_logits = model.predict(emb, verbose=0)
        head_logits = head.predict(_frontend_of(model).predict(emb, verbose=0), verbose=0)
        assert np.allclose(full_logits, head_logits, atol=1e-4)


def _embedding_of(model, full_tflite, embed_dim, x):
    """Run the full TFLite model and return (output, embedding) for input x."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(full_tflite), experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    it.resize_tensor_input(inp["index"], x.shape)
    it.allocate_tensors()
    it.set_tensor(inp["index"], x)
    it.invoke()
    out = it.get_tensor(outp["index"])
    emb_idx = [d["index"] for d in it.get_tensor_details()
               if list(d["shape"]) == list(x.shape[:-1]) + [embed_dim] and "frontend" in d["name"].lower()][0]
    return out, it.get_tensor(emb_idx).astype("float32")


class TestSliceTfliteHead:
    @pytest.mark.parametrize("hidden", [None, 6])
    def test_sliced_head_is_bit_exact(self, tmp_path, hidden):
        import tensorflow as tf
        embed_dim, n_classes = 16, 3
        model = _make_full_like_tflite(tmp_path / "full.tflite", embed_dim, n_classes, hidden=hidden)
        slice_tflite_head(tmp_path / "full.tflite", embed_dim, tmp_path / "head.tflite")

        x = np.random.default_rng(0).standard_normal((1, embed_dim)).astype(np.float32)
        full_out, emb = _embedding_of(model, tmp_path / "full.tflite", embed_dim, x)

        h = tf.lite.Interpreter(model_path=str(tmp_path / "head.tflite"))
        h.allocate_tensors()
        hi, ho = h.get_input_details()[0], h.get_output_details()[0]
        h.set_tensor(hi["index"], emb)
        h.invoke()
        head_out = h.get_tensor(ho["index"])
        # Slicing copies the original ops/weights verbatim → identical bits.
        assert np.array_equal(full_out, head_out)
        # I/O tensors are renamed for a clean API (labels only — values unchanged).
        assert hi["name"] == "embedding"
        assert ho["name"] == "scores"


class TestReadLabelsFile:
    def test_strips_scientific_prefix(self, tmp_path):
        p = tmp_path / "labels.txt"
        p.write_text("Genus species_Common name\nOther sp_Other\n", encoding="utf-8")
        assert read_labels_file(p) == ["Common name", "Other"]

    def test_plain_labels_and_blank_lines(self, tmp_path):
        p = tmp_path / "labels.txt"
        p.write_text("alpha\n\nbeta\n", encoding="utf-8")
        assert read_labels_file(p) == ["alpha", "beta"]
