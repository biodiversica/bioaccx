"""Extract a classifier head from a full BirdNET-Analyzer TFLite model.

BirdNET-Analyzer exports custom models as a single TFLite file that bundles the
BirdNET backbone (raw audio → embedding) with a small classifier head
(embedding → class scores). The head is the trailing chain of ``FULLY_CONNECTED``
ops; their weights, biases and fused activations are stored in the flatbuffer —
so the head can be recovered without converting (or even running) the backbone.

This module reads the flatbuffer directly and rebuilds an equivalent Keras head
(``embedding`` → ``scores``), single- or multi-layer, that can be re-exported
through the normal bioaccx ONNX/TFLite pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class HeadLayer:
    """One dense layer of the extracted head, ordered embedding → output."""
    W: np.ndarray            # weight, TFLite layout [out_features, in_features]
    b: np.ndarray            # bias, [out_features]
    activation: Optional[str]  # Keras activation name (None = linear / logits)


# TFLite fused-activation codes → Keras activation names.
_ACTIVATIONS = {0: None, 1: "relu", 3: "relu6", 4: "tanh"}
# Standalone activation ops that may trail the final dense layer.
def _standalone_activations():
    from tensorflow.lite.python import schema_py_generated as schema
    B = schema.BuiltinOperator
    return {B.LOGISTIC: "sigmoid", B.SOFTMAX: "softmax", B.RELU: "relu", B.TANH: "tanh"}


def extract_tflite_head(tflite_path: Path, embed_dim: int) -> list[HeadLayer]:
    """Recover the classifier head from a full TFLite model as a layer chain.

    The head is the chain of ``FULLY_CONNECTED`` ops that ends at the model's
    output tensor. The chain is walked backwards from the output until a dense
    layer is reached whose input dimension equals ``embed_dim`` (the embedding
    boundary), so backbone/frontend dense ops are never included. Each layer's
    fused activation (and a single trailing standalone activation op, e.g. a
    sigmoid/softmax) is captured.

    Parameters
    ----------
    tflite_path:
        Path to the full BirdNET-Analyzer ``.tflite`` model.
    embed_dim:
        Backbone embedding dimensionality (e.g. 1024 for BirdNET 2.4). Marks the
        boundary between backbone and head.

    Returns
    -------
    list[HeadLayer]
        Dense layers ordered embedding → output. The last layer's
        ``W.shape[0]`` is the number of output classes.

    Raises
    ------
    ValueError
        When the head cannot be located, an unsupported op interrupts the chain,
        or weights are not float32 (e.g. a quantized model).
    """
    try:
        from tensorflow.lite.python import schema_py_generated as schema
    except ImportError as exc:  # pragma: no cover - TF ships this module
        raise RuntimeError(
            "Reading the TFLite flatbuffer requires tensorflow's schema module."
        ) from exc

    B = schema.BuiltinOperator
    FC = B.FULLY_CONNECTED
    PASSTHROUGH = {B.RESHAPE, B.SQUEEZE}
    standalone = _standalone_activations()

    buf = tflite_path.read_bytes()
    model = schema.Model.GetRootAsModel(buf, 0)
    sg = model.Subgraphs(0)

    def tshape(i: int) -> list[int]:
        return list(sg.Tensors(i).ShapeAsNumpy())

    def tvalue(i: int) -> Optional[np.ndarray]:
        t = sg.Tensors(i)
        if t.Type() != schema.TensorType.FLOAT32:
            raise ValueError(
                f"{tflite_path.name}: head tensor {i} is not float32 "
                f"(quantized heads are not supported)."
            )
        b = model.Buffers(t.Buffer())
        if b.DataLength() == 0:
            return None
        return np.frombuffer(b.DataAsNumpy().tobytes(), dtype=np.float32).reshape(tshape(i))

    # Map each tensor index to the operator that produces it, plus per-op metadata.
    producer: dict[int, tuple] = {}
    for k in range(sg.OperatorsLength()):
        op = sg.Operators(k)
        code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        fused = None
        if op.BuiltinOptionsType() == schema.BuiltinOptions.FullyConnectedOptions:
            o = schema.FullyConnectedOptions()
            o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            fused = o.FusedActivationFunction()
        for j in range(op.OutputsLength()):
            producer[op.Outputs(j)] = (code, op, fused)

    out_idx = sg.Outputs(0)
    layers: list[HeadLayer] = []
    cur = out_idx
    pending_act: Optional[str] = None  # a standalone activation seen below an FC

    for _ in range(64):  # generous bound; heads are tiny
        if cur not in producer:
            raise ValueError(
                f"{tflite_path.name}: no producer for tensor {cur} while tracing "
                f"the head — could not locate the classifier."
            )
        code, op, fused = producer[cur]
        if code == FC:
            W = tvalue(op.Inputs(1))
            if W is None or W.ndim != 2:
                raise ValueError(f"{tflite_path.name}: head dense layer has no 2-D weight.")
            bidx = op.Inputs(2) if op.InputsLength() > 2 else -1
            b = tvalue(bidx) if bidx >= 0 else None
            if b is None:
                b = np.zeros(W.shape[0], dtype=np.float32)
            act = _ACTIVATIONS.get(fused, "UNKNOWN")
            if act == "UNKNOWN":
                raise ValueError(
                    f"{tflite_path.name}: unsupported fused activation {fused} in head."
                )
            # A standalone trailing activation overrides a linear fused op.
            if act is None and pending_act is not None:
                act = pending_act
            pending_act = None
            layers.append(HeadLayer(W=W, b=b, activation=act))
            act_input = op.Inputs(0)
            if tshape(act_input)[-1] == embed_dim:
                break
            cur = act_input
        elif code in PASSTHROUGH:
            cur = op.Inputs(0)
        elif code in standalone:
            pending_act = standalone[code]
            cur = op.Inputs(0)
        else:
            raise ValueError(
                f"{tflite_path.name}: unexpected op (code {code}) between the "
                f"embedding and the output — the head is not a plain dense chain."
            )
    else:
        raise ValueError(f"{tflite_path.name}: head chain did not terminate at the embedding.")

    layers.reverse()  # embedding → output order
    if layers[0].W.shape[1] != embed_dim:
        raise ValueError(
            f"{tflite_path.name}: first head layer input {layers[0].W.shape[1]} "
            f"does not match embedding size {embed_dim}."
        )
    return layers


def build_keras_head(layers: list[HeadLayer], embed_dim: int):
    """Rebuild a Keras head (``embedding`` → ``scores``) from extracted layers.

    The returned model matches the structure of a bioaccx-trained head (input
    named ``"embedding"``, final ``Dense`` named ``"scores"``), so it can be
    passed straight to ``export_onnx`` / ``export_tflite``.
    """
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model

    inp = Input(shape=(embed_dim,), name="embedding")
    x = inp
    built: list[tuple] = []
    for i, layer in enumerate(layers):
        name = "scores" if i == len(layers) - 1 else f"hidden_{i}"
        dense = Dense(int(layer.W.shape[0]), activation=layer.activation, name=name)
        x = dense(x)
        built.append((dense, layer))
    model = Model(inp, x)
    for dense, layer in built:
        # Keras Dense kernel is [in, out]; TFLite weight is [out, in] → transpose.
        dense.set_weights([layer.W.T.astype(np.float32), layer.b.astype(np.float32)])
    return model


def _find_embedding_idx(model, sg, embed_dim: int) -> int:
    """Return the tensor index of the embedding (input to the first head dense).

    Walks the ``FULLY_CONNECTED`` chain backwards from the subgraph output until
    a dense layer whose input feature dimension equals ``embed_dim`` is reached.
    """
    from tensorflow.lite.python import schema_py_generated as schema

    B = schema.BuiltinOperator
    FC = B.FULLY_CONNECTED
    PASSTHROUGH = {B.RESHAPE, B.SQUEEZE}
    standalone = set(_standalone_activations())

    producer = {}
    for k, op in enumerate(sg.operators):
        for o in op.outputs:
            producer[o] = k

    cur = sg.outputs[0]
    for _ in range(64):
        if cur not in producer:
            raise ValueError("could not locate the embedding tensor for the head.")
        op = sg.operators[producer[cur]]
        code = model.operatorCodes[op.opcodeIndex].builtinCode
        if code == FC:
            ain = op.inputs[0]
            if list(sg.tensors[ain].shape)[-1] == embed_dim:
                return ain
            cur = ain
        elif code in PASSTHROUGH or code in standalone:
            cur = op.inputs[0]
        else:
            raise ValueError(
                f"unexpected op (code {code}) between the embedding and the output."
            )
    raise ValueError("head chain did not terminate at the embedding.")


def slice_tflite_head(
    tflite_path: Path,
    embed_dim: int,
    out_path: Path,
    input_name: str = "embedding",
    output_name: str = "scores",
) -> Path:
    """Write a bit-exact head-only TFLite by slicing the original flatbuffer.

    Rather than rebuilding and re-converting the head (which incurs float32
    rounding), this copies the original head operators and their weight buffers
    verbatim into a new single-subgraph TFLite model whose input is the
    embedding tensor. The result reproduces the source model's head computation
    exactly (same ops, same buffers, same precision — including quantized heads).

    The head's input/output tensors are renamed to *input_name* / *output_name*
    for a clean API (matching the ONNX head). Names are labels only, so this does
    not affect the computed values.

    Parameters
    ----------
    tflite_path:
        Path to the full BirdNET-Analyzer ``.tflite`` model.
    embed_dim:
        Backbone embedding dimensionality (marks the head boundary).
    out_path:
        Destination ``.tflite`` file for the head.
    input_name, output_name:
        Names assigned to the head's input (embedding) and output tensors.
    """
    import flatbuffers
    from tensorflow.lite.python import schema_py_generated as schema

    buf = tflite_path.read_bytes()
    model = schema.ModelT.InitFromPackedBuf(buf, 0)
    sg = model.subgraphs[0]

    emb_idx = _find_embedding_idx(model, sg, embed_dim)
    out_idx = sg.outputs[0]

    def is_const(i: int) -> bool:
        b = model.buffers[sg.tensors[i].buffer]
        return b.data is not None and len(b.data) > 0

    # Backward reachability from the output, stopping at the embedding (a new
    # input) and at constants (weights/biases, kept as leaf tensors).
    producer = {}
    for k, op in enumerate(sg.operators):
        for o in op.outputs:
            producer[o] = k

    head_ops: set[int] = set()
    frontier = [out_idx]
    while frontier:
        t = frontier.pop()
        if t == emb_idx or is_const(t):
            continue
        if t not in producer:
            raise ValueError(
                f"{tflite_path.name}: tensor {t} in the head has no producer and "
                f"is not a constant — cannot slice."
            )
        k = producer[t]
        if k in head_ops:
            continue
        head_ops.add(k)
        for i in sg.operators[k].inputs:
            if i >= 0:
                frontier.append(i)
    head_ops_sorted = sorted(head_ops)  # original order is topological

    # Collect the tensors, buffers and opcodes the head ops reference, and build
    # old→new index maps so the sliced subgraph is self-contained.
    used_tensors: list[int] = []
    for k in head_ops_sorted:
        for i in list(sg.operators[k].inputs) + list(sg.operators[k].outputs):
            if i >= 0 and i not in used_tensors:
                used_tensors.append(i)
    if emb_idx not in used_tensors:
        used_tensors.append(emb_idx)
    tmap = {old: new for new, old in enumerate(used_tensors)}

    # Buffer 0 is empty by TFLite convention; referenced buffers follow.
    used_buffers: list[int] = []
    for old in used_tensors:
        b = sg.tensors[old].buffer
        if b not in used_buffers:
            used_buffers.append(b)
    bmap = {old: new + 1 for new, old in enumerate(used_buffers)}

    used_opcodes: list[int] = []
    for k in head_ops_sorted:
        oc = sg.operators[k].opcodeIndex
        if oc not in used_opcodes:
            used_opcodes.append(oc)
    ocmap = {old: new for new, old in enumerate(used_opcodes)}

    new_model = schema.ModelT()
    new_model.version = model.version
    new_model.description = model.description
    new_model.operatorCodes = [model.operatorCodes[oc] for oc in used_opcodes]
    empty = schema.BufferT()
    new_model.buffers = [empty] + [model.buffers[b] for b in used_buffers]

    new_sg = schema.SubGraphT()
    new_sg.name = b"head"
    new_sg.tensors = []
    for old in used_tensors:
        t = sg.tensors[old]
        t.buffer = bmap[t.buffer]
        new_sg.tensors.append(t)
    new_sg.inputs = [tmap[emb_idx]]
    new_sg.outputs = [tmap[out_idx]]
    # Rename the head's I/O tensors for a clean API (labels only — values unchanged).
    new_sg.tensors[tmap[emb_idx]].name = input_name.encode()
    new_sg.tensors[tmap[out_idx]].name = output_name.encode()
    new_sg.operators = []
    for k in head_ops_sorted:
        op = sg.operators[k]
        op.opcodeIndex = ocmap[op.opcodeIndex]
        op.inputs = [tmap[i] if i >= 0 else -1 for i in op.inputs]
        op.outputs = [tmap[o] for o in op.outputs]
        new_sg.operators.append(op)
    new_model.subgraphs = [new_sg]

    builder = flatbuffers.Builder(0)
    builder.Finish(new_model.Pack(builder), file_identifier=b"TFL3")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(builder.Output()))
    return out_path


# Suffixes a bioaccx export appends to the model stem; stripping them recovers
# the stem its *_labels.txt was written under.
_EXPORT_SUFFIX_RE = re.compile(r"_(keras_|sklearn_)?(full|head)(_fp32|_fp16|_int8)?$")


def find_labels_file(src: Path, explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the class-label file for the model at *src*.

    Checks, in order: an explicitly configured path, BirdNET-Analyzer's sibling
    ``<model>_Labels.txt``, the same name lower-cased, and — for a bioaccx
    export such as ``X_keras_full.onnx`` — the ``X_labels.txt`` written next to
    it by the training run. Returns None when nothing matches.
    """
    if explicit:
        return Path(explicit)
    stems = [src.stem]
    trimmed = _EXPORT_SUFFIX_RE.sub("", src.stem)
    if trimmed != src.stem:
        stems.append(trimmed)
    for stem in stems:
        for name in (f"{stem}_Labels.txt", f"{stem}_labels.txt"):
            candidate = src.with_name(name)
            if candidate.exists():
                return candidate
    return None


def read_labels_file(path: Path) -> list[str]:
    """Read one label per line from a BirdNET-Analyzer ``_Labels.txt`` file.

    BirdNET labels are formatted ``scientificName_commonName``; the common name
    (after the last underscore) is used as the display label when present.
    """
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(line.rsplit("_", 1)[-1] if "_" in line else line)
    return labels


# ---------------------------------------------------------------------------
# ONNX source models
# ---------------------------------------------------------------------------

def extract_onnx_head(
    onnx_path: Path, embed_dim: int
) -> tuple[list[HeadLayer], Optional[list[int]], str]:
    """Recover the classifier head of a full ONNX model as a layer chain.

    The ONNX counterpart of :func:`extract_tflite_head`: the graph is walked
    backwards to the embedding boundary, then the tail from there is parsed into
    dense layers. Any embedding normalization is folded into the first layer, so
    the result is a plain dense chain like the TFLite one.

    Returns ``(layers, output_filter, embedding_tensor)``. *output_filter* holds
    the class indices the source model already restricts its output to (an
    ``exclude_labels`` export), or None. *embedding_tensor* names the graph
    tensor the head reads, which :func:`verify_onnx_head` uses to check the
    extraction against the source model.
    """
    from bioaccx.convert_head import find_onnx_embedding_tensor, onnx_head_to_layers

    boundary = find_onnx_embedding_tensor(onnx_path, embed_dim)
    layers, _, output_filter = onnx_head_to_layers(
        onnx_path, embed_dim=embed_dim, start_tensor=boundary
    )
    return layers, output_filter, boundary


def verify_onnx_head(
    full_path: Path, head_path: Path, embedding_tensor: str, seed: int = 0
) -> tuple[float, float]:
    """Compare an extracted ONNX head against the model it came from.

    The source model is run once on random input with its embedding tensor
    exposed as an extra output; the head is then run on that embedding and the
    two outputs are compared. Returns ``(max_abs_difference, output_magnitude)``.
    """
    import tempfile

    import onnx
    import onnxruntime as ort

    model = onnx.load(str(full_path))
    if not any(o.name == embedding_tensor for o in model.graph.output):
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(embedding_tensor))

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        probe_path = Path(tmp.name)
    try:
        onnx.save(model, str(probe_path))
        sess = ort.InferenceSession(str(probe_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        x = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
        outputs = sess.run(None, {inp.name: x})
        names = [o.name for o in sess.get_outputs()]
        full_out = outputs[names.index(model.graph.output[0].name)]
        embedding = np.asarray(outputs[names.index(embedding_tensor)], dtype=np.float32)
    finally:
        probe_path.unlink(missing_ok=True)

    head = ort.InferenceSession(str(head_path), providers=["CPUExecutionProvider"])
    head_out = head.run(None, {head.get_inputs()[0].name: embedding})[0]
    return float(np.max(np.abs(head_out - full_out))), float(np.max(np.abs(full_out)))
