"""Convert a standalone classifier head between ONNX and TFLite.

A classifier head is a small dense chain (``embedding`` → class scores), so it
can be moved between the two formats without the backbone, a config file, or
retraining:

``tflite → onnx``
    Handed to tf2onnx directly — a dense-only graph converts cleanly (the full
    model cannot, because the backbone uses ops such as RFFT2D that ONNX lacks).

``onnx → tflite``
    There is no general ONNX→TFLite converter in the dependency set, so the
    graph is read the way :mod:`bioaccx.extract_head` reads a TFLite flatbuffer:
    the dense chain (with its activations, and any embedding normalization
    folded into the first layer) is recovered as weights, rebuilt as an
    equivalent Keras head, and converted with TFLiteConverter. Anything outside
    that op vocabulary raises rather than producing a subtly different model.

Both directions verify the result by running the source and the converted model
on the same random embeddings and reporting the largest output difference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from bioaccx.extract_head import HeadLayer, build_keras_head


# Output difference tolerated between source and converted head before the
# verification step warns, relative to the size of the source outputs. Weights
# are copied verbatim, so anything above float32 rounding noise means the graph
# was not reproduced faithfully. Scaling by the output magnitude keeps the check
# meaningful for both probabilities (|y| <= 1) and raw logits (|y| in the tens).
VERIFY_TOLERANCE = 1e-4

# ONNX activation ops → Keras activation names.
_ONNX_ACTIVATIONS = {"Relu": "relu", "Sigmoid": "sigmoid", "Softmax": "softmax", "Tanh": "tanh"}
# Ops that pass their input through unchanged.
_ONNX_PASSTHROUGH = {"Identity", "Cast"}

_SUFFIXES = {".onnx": ".tflite", ".tflite": ".onnx"}

# TFLite builtin ops a classifier head may contain: the dense chain, its
# activations, the embedding normalization, the exclude_labels gather, and
# shape/precision bookkeeping. Anything else means the file is not a head —
# a full model brings CONV_2D, RFFT2D and hundreds of others.
_TFLITE_HEAD_OPS = frozenset({
    "FULLY_CONNECTED", "RESHAPE", "SQUEEZE", "GATHER", "CAST",
    "SOFTMAX", "LOGISTIC", "RELU", "RELU6", "RELU_N1_TO_1", "TANH",
    "ADD", "SUB", "MUL", "DIV", "DEQUANTIZE", "QUANTIZE",
})


def tflite_is_quantized(tflite_path: Path) -> bool:
    """True when the flatbuffer stores any tensor at integer precision."""
    from tensorflow.lite.python import schema_py_generated as schema

    types = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    model = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel(tflite_path.read_bytes(), 0)
    )
    return any(
        types.get(tensor.type, "").startswith(("INT", "UINT"))
        and types.get(tensor.type) != "INT32"          # index/shape tensors
        for subgraph in model.subgraphs for tensor in subgraph.tensors
    )


def assert_tflite_is_head(tflite_path: Path) -> None:
    """Raise ValueError unless the flatbuffer holds only classifier-head ops.

    tf2onnx will happily convert a whole BirdNET model, producing a large graph
    that is not a head at all, so the op vocabulary is checked first — the
    mirror of the whitelist :func:`onnx_head_to_layers` applies going the other
    way.
    """
    from tensorflow.lite.python import schema_py_generated as schema

    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    model = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel(tflite_path.read_bytes(), 0)
    )
    subgraph = model.subgraphs[0]
    found = {
        names.get(model.operatorCodes[op.opcodeIndex].builtinCode, "UNKNOWN")
        for op in subgraph.operators
    }
    unsupported = sorted(found - _TFLITE_HEAD_OPS)
    if unsupported:
        raise ValueError(
            f"'{tflite_path.name}' is not a classifier head: it contains "
            f"{len(subgraph.operators)} ops including {', '.join(unsupported[:5])}"
            f"{'…' if len(unsupported) > 5 else ''}. Recover the head from a full "
            f"model with --extract_head first."
        )


def default_output_path(src: Path) -> Path:
    """Return the sibling path of *src* carrying the other format's suffix."""
    try:
        return src.with_suffix(_SUFFIXES[src.suffix.lower()])
    except KeyError:
        raise ValueError(
            f"Cannot convert '{src.name}': expected a .onnx or .tflite head, "
            f"got '{src.suffix or 'no suffix'}'."
        ) from None


# ---------------------------------------------------------------------------
# ONNX head → dense layer chain
# ---------------------------------------------------------------------------

def find_onnx_embedding_tensor(onnx_path: Path, embed_dim: int) -> str:
    """Return the tensor that feeds the classifier head of a full ONNX model.

    The mirror of :func:`bioaccx.extract_head._find_embedding_idx` for ONNX: the
    graph is walked backwards from its output until a dense layer is reached
    whose *input* size is ``embed_dim`` — the boundary between backbone and
    head. Any embedding normalization sitting in front of that layer
    (``Sub``/``Mul``/``Div``/``Add`` with constants) belongs to the head too, so
    the walk continues past it and returns the tensor before it.

    Raises ValueError when no such layer exists (wrong ``embed_dim``, or a model
    whose head is not a dense chain).
    """
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    producers = {out: node for node in graph.node for out in node.output}

    def data_input(node) -> str:
        """The node input carrying activations (as opposed to a constant)."""
        for name in node.input:
            if name not in inits:
                return name
        raise ValueError(f"Node '{node.name or node.op_type}' has no non-constant input.")

    def kernel(node):
        """The node's constant 2-D weight in [in, out] layout, or None."""
        for name in node.input[1:]:
            W = inits.get(name)
            if W is not None and W.ndim == 2:
                if node.op_type == "Gemm" and any(
                        a.name == "transB" and a.i for a in node.attribute):
                    return W.T
                return W
        return None

    cur = graph.output[0].name
    while True:
        node = producers.get(cur)
        if node is None:
            raise ValueError(
                f"No dense layer with an input size of {embed_dim} was found in "
                f"'{onnx_path.name}' — check the backbone / --embed-dim."
            )
        if node.op_type in ("MatMul", "Gemm") and kernel(node) is not None:
            boundary = data_input(node)
            if kernel(node).shape[0] == embed_dim:
                # Walk past the head's own input normalization, if any. Only a
                # per-feature constant counts: the backbone's tail carries
                # elementwise ops too (a scalar mean-pool divisor, 4-D BatchNorm
                # channels), and swallowing one would leave a head that no longer
                # accepts the backbone's embedding.
                while True:
                    prev = producers.get(boundary)
                    if prev is None or prev.op_type not in ("Sub", "Mul", "Div", "Add"):
                        break
                    consts = [inits.get(name) for name in prev.input[1:]]
                    if not all(
                        c is not None and c.ndim <= 2 and c.size == embed_dim
                        for c in consts
                    ):
                        break
                    boundary = data_input(prev)
                return boundary
            cur = boundary
        else:
            cur = data_input(node)


def onnx_head_to_layers(
    onnx_path: Path,
    embed_dim: Optional[int] = None,
    start_tensor: Optional[str] = None,
) -> tuple[list[HeadLayer], int, Optional[list[int]]]:
    """Recover an ONNX head as (dense layers, embedding size, output filter).

    By default the whole graph is treated as the head, walked from its input to
    its output. Pass *start_tensor* (with *embed_dim*) to parse only the tail of
    a larger graph — that is how a head is extracted from a full model, with
    :func:`find_onnx_embedding_tensor` locating the boundary.

    Recognised ops:

    ``MatMul`` / ``Gemm`` (+ following ``Add``)
        One dense layer; the ``Add`` supplies the bias when present (a
        zero bias is folded away by the exporter, so it may be missing).
    ``Sub`` / ``Mul`` / ``Div`` with a constant
        Embedding normalization. Folded into the first dense layer — exactly
        equivalent, and it keeps the rebuilt head a plain dense chain.
    ``Relu`` / ``Sigmoid`` / ``Softmax`` / ``Tanh``
        Activation of the preceding dense layer.
    ``Gather`` with constant indices
        The ``exclude_labels`` output filter. Returned separately rather than
        folded in, because with a softmax the filter must stay *after* the
        activation to preserve the model's exact outputs.
    ``Identity`` / ``Cast``
        Passthrough.

    Raises ValueError on any other op, on a branching graph, or on weights that
    are not float32 (a quantized head cannot be rebuilt this way).
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    if start_tensor is None:
        if len(graph.input) != 1 or len(graph.output) != 1:
            raise ValueError(
                f"Expected a head with one input and one output, got "
                f"{len(graph.input)} / {len(graph.output)} — is this a full model?"
            )
        dims = graph.input[0].type.tensor_type.shape.dim
        embed_dim = dims[-1].dim_value if len(dims) else 0
        if embed_dim <= 0:
            raise ValueError(
                f"Could not determine the embedding size from the graph input "
                f"'{graph.input[0].name}' (its last dimension is not fixed)."
            )
    elif not embed_dim:
        raise ValueError("embed_dim is required when parsing from start_tensor.")

    # Every node that consumes a given tensor, so the chain can be walked forward.
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def const(name: str) -> Optional[np.ndarray]:
        arr = inits.get(name)
        if arr is None:
            return None
        if arr.dtype != np.float32 and not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"Head weight '{name}' has dtype {arr.dtype}; only float32 heads "
                f"can be converted (a quantized head is not supported)."
            )
        return arr

    layers: list[HeadLayer] = []
    keep_indices: Optional[list[int]] = None
    # Pending input affine (normalization): x → x * scale + shift, folded into
    # the first dense layer. Kept as scalars until a constant sets them.
    scale: np.ndarray | float = 1.0
    shift: np.ndarray | float = 0.0

    def add_dense(kernel: np.ndarray, bias: Optional[np.ndarray]) -> None:
        """Append a dense layer, folding any pending input affine into the first."""
        nonlocal scale, shift
        W = kernel.astype(np.float32)                       # [in, out]
        b = (np.zeros(W.shape[1], np.float32) if bias is None
             else bias.astype(np.float32).reshape(-1))
        if not layers and (np.ndim(scale) or scale != 1.0 or np.ndim(shift) or shift != 0.0):
            # y = (x * s + t) @ W + b = x @ (W * s[:, None]) + (t @ W + b)
            s = np.broadcast_to(np.asarray(scale, np.float32).reshape(-1), (W.shape[0],))
            t = np.broadcast_to(np.asarray(shift, np.float32).reshape(-1), (W.shape[0],))
            b = (t @ W + b).astype(np.float32)
            W = (W * s[:, None]).astype(np.float32)
            scale, shift = 1.0, 0.0
        # HeadLayer stores the TFLite layout [out, in]; build_keras_head transposes back.
        layers.append(HeadLayer(W=W.T.copy(), b=b, activation=None))

    cur = start_tensor if start_tensor is not None else graph.input[0].name
    out_name = graph.output[0].name
    seen = 0
    while cur != out_name:
        nodes = consumers.get(cur, [])
        if len(nodes) != 1:
            raise ValueError(
                f"Head graph is not a simple chain: tensor '{cur}' has "
                f"{len(nodes)} consumer(s); only linear dense heads are supported. "
                f"A full model (backbone + head) cannot be converted this way — "
                f"recover its head first (--extract_head, or train with output_type: head)."
            )
        node = nodes[0]
        seen += 1
        if seen > len(graph.node):  # pragma: no cover - guards a malformed cyclic graph
            raise ValueError("Head graph appears to contain a cycle.")
        op = node.op_type

        if op in _ONNX_PASSTHROUGH:
            pass
        elif op == "MatMul":
            W = next((const(n) for n in node.input if n != cur and const(n) is not None), None)
            if W is None or W.ndim != 2:
                raise ValueError(f"MatMul '{node.name or op}' has no constant 2-D weight.")
            add_dense(W, None)
        elif op == "Gemm":
            attrs = {a.name: a for a in node.attribute}
            if attrs.get("transA") is not None and attrs["transA"].i:
                raise ValueError("Gemm with transA=1 is not supported in a head graph.")
            W = const(node.input[1])
            if W is None:
                raise ValueError(f"Gemm '{node.name or op}' has a non-constant weight.")
            if attrs.get("transB") is not None and attrs["transB"].i:
                W = W.T
            bias = const(node.input[2]) if len(node.input) > 2 else None
            add_dense(W, bias)
        elif op == "Add":
            c = next((const(n) for n in node.input if n != cur and const(n) is not None), None)
            if c is None:
                raise ValueError(f"Add '{node.name or op}' has no constant operand.")
            if layers and layers[-1].activation is None and np.prod(c.shape) == len(layers[-1].b):
                # Bias of the dense layer just recorded.
                layers[-1].b = layers[-1].b + c.astype(np.float32).reshape(-1)
            elif not layers:
                shift = np.asarray(shift, np.float32) + c.astype(np.float32)
            else:
                raise ValueError("Add after an activation is not supported in a head graph.")
        elif op in ("Sub", "Mul", "Div"):
            c = next((const(n) for n in node.input if n != cur and const(n) is not None), None)
            if c is None:
                raise ValueError(f"{op} '{node.name or op}' has no constant operand.")
            if layers:
                raise ValueError(
                    f"'{op}' after the first dense layer is not supported in a head graph."
                )
            c = c.astype(np.float32)
            if op == "Sub":
                shift = np.asarray(shift, np.float32) - c
            else:
                factor = c if op == "Mul" else 1.0 / c
                scale = np.asarray(scale, np.float32) * factor
                shift = np.asarray(shift, np.float32) * factor
        elif op in _ONNX_ACTIVATIONS:
            if not layers:
                raise ValueError(f"Activation '{op}' before any dense layer is not supported.")
            if layers[-1].activation is not None:
                raise ValueError("Two consecutive activations are not supported in a head graph.")
            layers[-1].activation = _ONNX_ACTIVATIONS[op]
        elif op == "Gather":
            idx = const(node.input[1])
            if idx is None:
                raise ValueError("Gather with non-constant indices is not supported.")
            keep_indices = [int(i) for i in np.asarray(idx).reshape(-1)]
        elif "Quantize" in op:
            raise ValueError(
                f"'{onnx_path.name}' is a quantized ONNX head (op '{op}'), which cannot be "
                f"rebuilt as TFLite. Convert the FP32 head instead and re-export it at the "
                f"precision you need."
            )
        else:
            raise ValueError(
                f"Unsupported ONNX op '{op}' in the head graph. Only dense chains "
                f"(MatMul/Gemm/Add), activations, normalization and an output "
                f"Gather can be converted to TFLite."
            )
        cur = node.output[0]

    if not layers:
        raise ValueError("No dense layer found — this does not look like a classifier head.")
    return layers, embed_dim, keep_indices


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _run_onnx(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])


def _run_tflite(path: Path, x: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path))
    inp = interp.get_input_details()[0]
    interp.resize_tensor_input(inp["index"], list(x.shape))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interp.invoke()
    return np.asarray(interp.get_tensor(interp.get_output_details()[0]["index"]))


def _run_head(path: Path, x: np.ndarray) -> np.ndarray:
    return _run_onnx(path, x) if path.suffix.lower() == ".onnx" else _run_tflite(path, x)


def verify_heads(src: Path, dst: Path, embed_dim: int, n: int = 8,
                 seed: int = 0) -> tuple[float, float]:
    """Run both heads on identical random embeddings.

    Returns ``(max_abs_difference, output_magnitude)``, the latter being the
    largest absolute source output — the scale the difference is judged against.
    """
    x = np.random.default_rng(seed).standard_normal((n, embed_dim)).astype(np.float32)
    a, b = _run_head(src, x), _run_head(dst, x)
    if a.shape != b.shape:
        raise ValueError(
            f"Converted head has output shape {b.shape}, source has {a.shape}."
        )
    return float(np.max(np.abs(a - b))), float(np.max(np.abs(a)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def convert_head(
    src: Path,
    dst: Path | None = None,
    data_type: str = "FP32",
    verify: bool = True,
) -> Path:
    """Convert a classifier head between ONNX and TFLite, returning the new path.

    The direction is taken from the file suffixes: a ``.tflite`` source is
    converted to ONNX and an ``.onnx`` source to TFLite. *dst* defaults to the
    source path with the other suffix.
    """
    from bioaccx.exporters.onnx_exporter import export_tflite_head_to_onnx
    from bioaccx.exporters.tflite_exporter import _export_head_tflite, _slice_keras_output

    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Classifier head not found: {src}")
    suffix = src.suffix.lower()
    if suffix not in _SUFFIXES:
        raise ValueError(
            f"Cannot convert '{src.name}': expected a .onnx or .tflite head, "
            f"got '{src.suffix or 'no suffix'}'."
        )
    dst = Path(dst) if dst is not None else default_output_path(src)
    if dst.resolve() == src.resolve():
        raise ValueError("Output path must differ from the source head.")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"bioaccx — convert classifier head ({suffix.lstrip('.')} → {dst.suffix.lstrip('.')})")
    print(f"Source : {src}")
    print(f"Output : {dst}")
    print(f"{'='*62}\n")

    quantized_source = False
    if suffix == ".tflite":
        import onnx
        assert_tflite_is_head(src)
        quantized_source = tflite_is_quantized(src)
        if quantized_source:
            print("  Source head is quantized — the ONNX copy re-quantizes it "
                  "(see the verification below).")
        try:
            export_tflite_head_to_onnx(src, dst, data_type=data_type)
        except Exception as exc:
            raise ValueError(
                f"Could not convert '{src.name}' to ONNX: {exc}\n"
                f"Only a classifier head converts cleanly — a full model's backbone uses "
                f"ops ONNX lacks (RFFT2D and friends). Recover the head first with "
                f"--extract_head."
            ) from exc
        dims = onnx.load(str(dst)).graph.input[0].type.tensor_type.shape.dim
        embed_dim = dims[-1].dim_value if len(dims) else 0
        print(f"  ONNX head → {dst} ({data_type.upper()})")
    else:
        layers, embed_dim, keep_indices = onnx_head_to_layers(src)
        arch = " → ".join([str(embed_dim)] + [str(int(l.W.shape[0])) for l in layers])
        acts = [l.activation or "linear" for l in layers]
        print(f"  Recovered head: {arch}  ({len(layers)} dense layer(s), activations={acts})")
        model = build_keras_head(layers, embed_dim)
        if keep_indices is not None:
            print(f"  Output filter: {len(keep_indices)} of "
                  f"{int(layers[-1].W.shape[0])} classes kept (applied after the activation)")
            model = _slice_keras_output(model, keep_indices)
        _export_head_tflite(model, dst, data_type)

    if verify and embed_dim:
        diff, magnitude = verify_heads(src, dst, embed_dim)
        tolerance = VERIFY_TOLERANCE * max(1.0, magnitude)
        ok = diff <= tolerance
        label = "OK" if ok else ("QUANTIZATION DRIFT" if quantized_source else "MISMATCH")
        print(f"  Verification: max output difference = {diff:.2e} "
              f"(outputs up to {magnitude:.2e})  [{label}]")
        if not ok and quantized_source:
            print("  Note: the source head is quantized, and the conversion re-quantizes it, "
                  "so its outputs shift slightly. Convert an FP32 head for an exact copy.")
        elif not ok:
            print("  Warning: the converted head does not reproduce the source outputs.")

    print(f"\nDone. Converted head saved to: {dst}")
    return dst
