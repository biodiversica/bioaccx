"""Unit tests for bioaccx.convert_head (ONNX ↔ TFLite head conversion)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bioaccx.convert_head import (
    assert_tflite_is_head,
    convert_head,
    default_output_path,
    onnx_head_to_layers,
    verify_heads,
)

EMBED = 12
CLASSES = 3


def _keras_head(embed=EMBED, hidden=6, n_classes=CLASSES, normalize=False, activation=None):
    """A head shaped like a bioaccx-trained one, with non-zero biases."""
    import tensorflow as tf
    from tensorflow.keras.layers import Activation, Dense, Input, Normalization

    rng = np.random.default_rng(0)
    inp = Input(shape=(embed,), name="embedding")
    x = inp
    if normalize:
        norm = Normalization(axis=-1)
        norm.adapt(rng.standard_normal((64, embed)).astype(np.float32))
        x = norm(x)
    if hidden:
        x = Dense(hidden, activation="relu", name="hidden")(x)
    x = Dense(n_classes, activation=None, name="scores")(x)
    if activation:
        x = Activation(activation, name="output_activation")(x)
    model = tf.keras.Model(inp, x)
    for layer in model.layers:
        if isinstance(layer, Dense):
            W, b = layer.get_weights()
            layer.set_weights([W, rng.standard_normal(b.shape).astype(np.float32)])
    return model


def _write_pair(tmp_path: Path, stem: str, keep_indices=None, **kw) -> tuple[Path, Path]:
    """Export the same head as ONNX and TFLite; return (onnx_path, tflite_path)."""
    from bioaccx.exporters.onnx_exporter import export_onnx
    from bioaccx.exporters.tflite_exporter import export_tflite

    model = _keras_head(**kw)
    onnx_path = tmp_path / f"{stem}.onnx"
    tflite_path = tmp_path / f"{stem}.tflite"
    export_onnx(model, "keras", EMBED, onnx_path, keep_indices=keep_indices)
    export_tflite(model, "keras", EMBED, tflite_path, keep_indices=keep_indices)
    return onnx_path, tflite_path


def _raw_onnx(path: Path, ops: list[str], branch: bool = False) -> Path:
    """Hand-build a small ONNX graph from *ops* to exercise the parser's limits."""
    import onnx
    from onnx import TensorProto, helper
    from onnx import numpy_helper as nph

    W = np.eye(EMBED, CLASSES, dtype=np.float32)
    nodes, inits, cur = [], [nph.from_array(W, name="W")], "embedding"
    for i, op in enumerate(ops):
        out = f"t{i}"
        inputs = [cur, "W"] if op == "MatMul" else [cur]
        nodes.append(helper.make_node(op, inputs=inputs, outputs=[out]))
        cur = out
    if branch:
        # A second consumer of the graph input makes the graph non-linear.
        nodes.append(helper.make_node("Relu", inputs=["embedding"], outputs=["side"]))
    dim = CLASSES if "MatMul" in ops else EMBED
    graph = helper.make_graph(
        nodes, "head",
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, EMBED])],
        [helper.make_tensor_value_info(cur, TensorProto.FLOAT, [None, dim])],
        initializer=inits,
    )
    model = helper.make_model(graph)
    model.ir_version = 8
    model.opset_import[0].domain, model.opset_import[0].version = "", 13
    onnx.save(model, str(path))
    return path


# TFLite leaves very small weight tensors in float32, so a head that is actually
# quantized needs a kernel above the converter's threshold (~1k elements).
INT8_EMBED = 256
INT8_CLASSES = 8


def _int8_head(tmp_path: Path) -> Path:
    """Export a head large enough that INT8 conversion really quantizes it."""
    from bioaccx.exporters.tflite_exporter import export_tflite
    path = tmp_path / "int8_head.tflite"
    model = _keras_head(embed=INT8_EMBED, hidden=0, n_classes=INT8_CLASSES)
    export_tflite(model, "keras", INT8_EMBED, path, data_type="INT8")
    return path


class TestDefaultOutputPath:
    def test_onnx_to_tflite(self):
        assert default_output_path(Path("/m/head.onnx")) == Path("/m/head.tflite")

    def test_tflite_to_onnx(self):
        assert default_output_path(Path("/m/head.tflite")) == Path("/m/head.onnx")

    def test_rejects_other_suffixes(self):
        with pytest.raises(ValueError, match="expected a .onnx or .tflite head"):
            default_output_path(Path("/m/head.pb"))


class TestOnnxHeadToLayers:
    def test_single_dense(self, tmp_path):
        onnx_path, _ = _write_pair(tmp_path, "plain", hidden=0)
        layers, embed_dim, keep = onnx_head_to_layers(onnx_path)
        assert embed_dim == EMBED
        assert keep is None
        assert len(layers) == 1
        assert layers[0].W.shape == (CLASSES, EMBED)   # [out, in]
        assert layers[0].activation is None

    def test_hidden_layer_and_activation(self, tmp_path):
        onnx_path, _ = _write_pair(tmp_path, "deep", hidden=6, activation="softmax")
        layers, _, _ = onnx_head_to_layers(onnx_path)
        assert [l.activation for l in layers] == ["relu", "softmax"]
        assert [int(l.W.shape[0]) for l in layers] == [6, CLASSES]

    def test_normalization_folded_into_first_layer(self, tmp_path):
        onnx_path, _ = _write_pair(tmp_path, "norm", normalize=True, hidden=0)
        layers, _, _ = onnx_head_to_layers(onnx_path)
        # Folding keeps the chain dense-only: still one layer, same shape.
        assert len(layers) == 1
        assert layers[0].W.shape == (CLASSES, EMBED)

    def test_gather_returned_not_folded(self, tmp_path):
        onnx_path, _ = _write_pair(tmp_path, "excl", hidden=0, activation="softmax",
                                   keep_indices=[0, 2])
        layers, _, keep = onnx_head_to_layers(onnx_path)
        assert keep == [0, 2]
        # The dense layer still emits every class; the filter stays a separate step.
        assert int(layers[-1].W.shape[0]) == CLASSES

    def test_rejects_unsupported_op(self, tmp_path):
        path = _raw_onnx(tmp_path / "exp.onnx", ["MatMul", "Exp"])
        with pytest.raises(ValueError, match="Unsupported ONNX op 'Exp'"):
            onnx_head_to_layers(path)

    def test_rejects_activation_before_dense(self, tmp_path):
        path = _raw_onnx(tmp_path / "pre_act.onnx", ["Relu", "MatMul"])
        with pytest.raises(ValueError, match="before any dense layer"):
            onnx_head_to_layers(path)

    def test_quantized_head_gets_targeted_error(self, tmp_path):
        from bioaccx.exporters.onnx_exporter import export_onnx
        path = tmp_path / "q.onnx"
        export_onnx(_keras_head(hidden=0), "keras", EMBED, path, data_type="INT8")
        with pytest.raises(ValueError, match="quantized ONNX head"):
            onnx_head_to_layers(path)

    def test_rejects_branching_graph(self, tmp_path):
        path = _raw_onnx(tmp_path / "branch.onnx", ["MatMul"], branch=True)
        with pytest.raises(ValueError, match="not a simple chain"):
            onnx_head_to_layers(path)


class TestTfliteIsQuantized:
    def test_fp32_head(self, tmp_path):
        from bioaccx.convert_head import tflite_is_quantized
        _, path = _write_pair(tmp_path, "fp32")
        assert tflite_is_quantized(path) is False

    def test_int8_head(self, tmp_path):
        from bioaccx.convert_head import tflite_is_quantized
        assert tflite_is_quantized(_int8_head(tmp_path)) is True


class TestFindOnnxEmbeddingTensor:
    def _full_graph(self, path: Path, extra_tail_op: str | None = None,
                    const_shape=(), normalize=False) -> Path:
        """A 'backbone' (one dense layer) followed by a head, as raw ONNX.

        *extra_tail_op* appends an elementwise backbone op just before the head,
        which must NOT be pulled into it; *normalize* prepends the head's own
        per-feature normalization, which must be.
        """
        import onnx
        from onnx import TensorProto, helper
        from onnx import numpy_helper as nph

        rng = np.random.default_rng(0)
        backbone_W = rng.standard_normal((5, EMBED)).astype(np.float32)
        head_W = rng.standard_normal((EMBED, CLASSES)).astype(np.float32)
        inits = [nph.from_array(backbone_W, name="bW"), nph.from_array(head_W, name="hW")]
        nodes = [helper.make_node("MatMul", ["x", "bW"], ["backbone_out"])]
        cur = "backbone_out"
        if extra_tail_op:
            inits.append(nph.from_array(
                rng.standard_normal(const_shape).astype(np.float32) + 2.0, name="bc"))
            nodes.append(helper.make_node(extra_tail_op, [cur, "bc"], ["embedding"]))
            cur = "embedding"
        if normalize:
            inits.append(nph.from_array(
                rng.standard_normal((1, EMBED)).astype(np.float32), name="mean"))
            nodes.append(helper.make_node("Sub", [cur, "mean"], ["normed"]))
            cur = "normed"
        nodes.append(helper.make_node("MatMul", [cur, "hW"], ["scores"]))
        graph = helper.make_graph(
            nodes, "full",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 5])],
            [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, CLASSES])],
            initializer=inits,
        )
        model = helper.make_model(graph)
        model.ir_version = 8
        model.opset_import[0].domain, model.opset_import[0].version = "", 13
        onnx.save(model, str(path))
        return path

    def test_plain_boundary(self, tmp_path):
        from bioaccx.convert_head import find_onnx_embedding_tensor
        path = self._full_graph(tmp_path / "full.onnx")
        assert find_onnx_embedding_tensor(path, EMBED) == "backbone_out"

    def test_head_normalization_is_included(self, tmp_path):
        from bioaccx.convert_head import find_onnx_embedding_tensor
        path = self._full_graph(tmp_path / "norm.onnx", normalize=True)
        assert find_onnx_embedding_tensor(path, EMBED) == "backbone_out"

    def test_scalar_backbone_op_is_not_swallowed(self, tmp_path):
        """A mean-pool divisor belongs to the backbone, not the head."""
        from bioaccx.convert_head import find_onnx_embedding_tensor
        path = self._full_graph(tmp_path / "pool.onnx", extra_tail_op="Div",
                                const_shape=(), normalize=True)
        assert find_onnx_embedding_tensor(path, EMBED) == "embedding"

    def test_spatial_backbone_op_is_not_swallowed(self, tmp_path):
        """A 4-D BatchNorm constant is not per-feature head normalization."""
        from bioaccx.convert_head import find_onnx_embedding_tensor
        path = self._full_graph(tmp_path / "bn.onnx", extra_tail_op="Mul",
                                const_shape=(1, 1, 1, EMBED))
        assert find_onnx_embedding_tensor(path, EMBED) == "embedding"

    def test_unknown_embed_dim_raises(self, tmp_path):
        from bioaccx.convert_head import find_onnx_embedding_tensor
        path = self._full_graph(tmp_path / "full.onnx")
        with pytest.raises(ValueError, match="No dense layer with an input size of 99"):
            find_onnx_embedding_tensor(path, 99)


class TestAssertTfliteIsHead:
    def test_accepts_head(self, tmp_path):
        _, tflite_path = _write_pair(tmp_path, "head", normalize=True)
        assert_tflite_is_head(tflite_path)   # does not raise

    def test_rejects_non_head(self, tmp_path):
        """A convolutional graph is not a dense head."""
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(8, 8, 1)),
            tf.keras.layers.Conv2D(2, 3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
        ])
        path = tmp_path / "conv.tflite"
        path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
        with pytest.raises(ValueError, match="not a classifier head"):
            assert_tflite_is_head(path)


class TestConvertHead:
    @pytest.mark.parametrize("kw", [
        dict(hidden=0),
        dict(hidden=6, activation="softmax"),
        dict(hidden=6, normalize=True, activation="sigmoid"),
        dict(normalize=True, hidden=0),
    ])
    def test_onnx_to_tflite_matches_source(self, tmp_path, kw):
        onnx_path, _ = _write_pair(tmp_path, "h", **kw)
        out = convert_head(onnx_path)
        assert out == tmp_path / "h.tflite"
        diff, magnitude = verify_heads(onnx_path, out, EMBED)
        assert diff <= 1e-4 * max(1.0, magnitude)

    @pytest.mark.parametrize("kw", [
        dict(hidden=0),
        dict(hidden=6, activation="softmax"),
        dict(hidden=6, normalize=True, activation="sigmoid"),
    ])
    def test_tflite_to_onnx_matches_source(self, tmp_path, kw):
        _, tflite_path = _write_pair(tmp_path, "h", **kw)
        out = convert_head(tflite_path)
        assert out == tmp_path / "h.onnx"
        diff, magnitude = verify_heads(tflite_path, out, EMBED)
        assert diff <= 1e-4 * max(1.0, magnitude)

    def test_int8_tflite_converts_to_onnx(self, tmp_path, capsys):
        """A quantized TFLite head still converts — with the re-quantization flagged."""
        out = convert_head(_int8_head(tmp_path))
        assert out.exists()
        assert "quantized" in capsys.readouterr().out

    def test_excluded_labels_preserved(self, tmp_path):
        """The output filter survives the conversion, columns and values alike."""
        onnx_path, _ = _write_pair(tmp_path, "excl", activation="softmax",
                                   keep_indices=[0, 2])
        out = convert_head(onnx_path)
        diff, magnitude = verify_heads(onnx_path, out, EMBED)
        assert diff <= 1e-4 * max(1.0, magnitude)

        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        x = np.zeros((1, EMBED), np.float32)
        assert sess.run(None, {sess.get_inputs()[0].name: x})[0].shape == (1, 2)

    def test_round_trip_onnx_tflite_onnx(self, tmp_path):
        onnx_path, _ = _write_pair(tmp_path, "rt", hidden=6, normalize=True,
                                   activation="softmax")
        tflite_path = convert_head(onnx_path, tmp_path / "rt_mid.tflite")
        back = convert_head(tflite_path, tmp_path / "rt_back.onnx")
        diff, magnitude = verify_heads(onnx_path, back, EMBED)
        assert diff <= 1e-4 * max(1.0, magnitude)

    def test_explicit_output_path(self, tmp_path):
        onnx_path, _ = _write_pair(tmp_path, "h", hidden=0)
        out = convert_head(onnx_path, tmp_path / "sub" / "custom.tflite")
        assert out.exists() and out.name == "custom.tflite"

    def test_missing_source(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert_head(tmp_path / "nope.onnx")

    def test_unknown_suffix(self, tmp_path):
        src = tmp_path / "head.pb"
        src.write_bytes(b"")
        with pytest.raises(ValueError, match="expected a .onnx or .tflite head"):
            convert_head(src)

    def test_rejects_same_source_and_destination(self, tmp_path):
        onnx_path, _ = _write_pair(tmp_path, "h", hidden=0)
        with pytest.raises(ValueError, match="must differ"):
            convert_head(onnx_path, onnx_path)
