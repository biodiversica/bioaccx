"""ONNX export for head-only and full (foundation + classifier) models."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import onnx


def export_onnx(
    classifier,                     # trained Keras model or sklearn Pipeline
    classifier_type: Literal["keras", "sklearn"],
    embed_dim: int,
    out_path: Path,
    # For full-model export only
    foundation_onnx_path: Path | None = None,
    foundation_input_name: str = "input",
    output_type: Literal["head", "full"] = "head",
    keep_indices: list[int] | None = None,  # output label filter (None = keep all)
    data_type: str = "FP32",                # output precision: FP32 | FP16 | INT8
) -> Path:
    """Convert a classifier to ONNX and optionally merge with a foundation backbone.

    Parameters
    ----------
    classifier:
        A trained Keras Model or a sklearn Pipeline with a 'clf' step.
    classifier_type:
        ``"keras"`` or ``"sklearn"``.  Determines which conversion backend is used.
    embed_dim:
        Dimensionality of the embedding vector (= foundation model output size).
    out_path:
        Destination .onnx file.  Parent directories are created automatically.
    foundation_onnx_path:
        Required when output_type == 'full'.  Path to the backbone ONNX file.
    foundation_input_name:
        Name of the foundation model's raw-audio input tensor.  Used to rename
        the merged graph's input for a clean API.
    output_type:
        ``"head"`` writes the classifier alone; ``"full"`` prepends the
        foundation backbone via _merge_onnx.
    keep_indices:
        When set, appends a Gather node that selects only the listed output
        indices (used to exclude background/noise labels from the export).
    data_type:
        Output precision applied after the model is written: ``"FP32"`` (no
        conversion), ``"FP16"`` (float16 weights, float32 I/O preserved), or
        ``"INT8"`` (dynamic/weight-only quantization).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if output_type == "head":
        _convert(classifier, classifier_type, embed_dim, out_path)
        if keep_indices is not None:
            _filter_onnx_outputs(out_path, keep_indices, classifier_type)
        _apply_onnx_precision(out_path, data_type)
        return out_path

    # full: convert head to a temp file, then merge; temp file is always cleaned up.
    if foundation_onnx_path is None:
        raise ValueError("foundation_onnx_path required for output_type='full'")
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _convert(classifier, classifier_type, embed_dim, tmp_path, verbose=False)
        if keep_indices is not None:
            _filter_onnx_outputs(tmp_path, keep_indices, classifier_type)
        _merge_onnx(foundation_onnx_path, tmp_path, out_path, foundation_input_name)
        _apply_onnx_precision(out_path, data_type)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path


def _apply_onnx_precision(path: Path, data_type: str) -> None:
    """Convert the ONNX model at *path* in place to the requested precision.

    ``FP32`` is a no-op. ``FP16`` casts weights to float16 while keeping the
    graph's input/output tensors float32 (``keep_io_types=True``) so the model
    stays drop-in for float32 callers. ``INT8`` applies dynamic/weight-only
    quantization (no calibration dataset). The conversion libraries are imported
    lazily so a base install without them still works for FP32 exports.
    """
    dt = data_type.upper()
    if dt == "FP32":
        return
    if dt == "FP16":
        try:
            from onnxconverter_common import float16  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "FP16 ONNX export requires 'onnxconverter-common'. "
                "Install it with 'pip install onnxconverter-common' "
                "(or 'uv sync')."
            ) from exc
        model = onnx.load(str(path))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, str(path))
        print(f"  ONNX precision    → FP16 ({path.name})")
    elif dt == "INT8":
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "INT8 ONNX export requires 'onnxruntime' and 'sympy' "
                "(sympy is a transitive requirement of onnxruntime.quantization). "
                "Install them with 'pip install onnxruntime sympy' (or 'uv sync')."
            ) from exc
        quantize_dynamic(str(path), str(path), weight_type=QuantType.QInt8)
        print(f"  ONNX precision    → INT8 ({path.name})")
    else:
        raise ValueError(f"Unknown ONNX data_type: {data_type!r}")


def _convert(classifier, classifier_type: str, embed_dim: int, out_path: Path, *, verbose: bool = True) -> None:
    """Dispatch to the backend-specific converter based on *classifier_type*."""
    if classifier_type == "keras":
        _keras_to_onnx(classifier, embed_dim, out_path, verbose=verbose)
    elif classifier_type == "sklearn":
        _sklearn_to_onnx(classifier, embed_dim, out_path, verbose=verbose)
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type!r}")


def _keras_to_onnx(model, embed_dim: int, out_path: Path, *, verbose: bool = True) -> None:
    """Convert a Keras model to ONNX using tf2onnx.

    opset=13 is chosen as a stable minimum that covers all operators used by the
    dense classifier and is broadly supported by deployment runtimes.  The input
    tensor is named ``"embedding"`` to match the head's expected input when merging.
    """
    import tensorflow as tf
    import tf2onnx

    input_sig = [tf.TensorSpec([None, embed_dim], tf.float32, name="embedding")]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=13)
    onnx.save(onnx_model, str(out_path))
    if verbose:
        print(f"  Keras ONNX head  → {out_path}")


def _sklearn_to_onnx(pipe, embed_dim: int, out_path: Path, *, verbose: bool = True) -> None:
    """Convert a sklearn Pipeline to ONNX using skl2onnx.

    ``zipmap=False`` makes the probability output a plain float array [batch, N]
    instead of a list of dicts, which is easier to post-process in inference runtimes.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, embed_dim]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type, options={"zipmap": False})
    onnx.save(onnx_model, str(out_path))
    if verbose:
        print(f"  Sklearn ONNX head → {out_path}")


def _merge_onnx(
    foundation_path: Path,
    head_path: Path,
    out_path: Path,
    foundation_input_name: str,
) -> None:
    """Merge foundation model and classifier head into a single ONNX graph.

    Uses manual graph stitching rather than onnx.compose so that mismatched
    IR versions and opset domains (e.g. ai.onnx.ml from sklearn) are handled
    gracefully.
    """
    foundation = onnx.load(str(foundation_path))
    head = onnx.load(str(head_path))

    foundation_out = foundation.graph.output[0].name
    foundation_out_shape = [
        d.dim_value for d in foundation.graph.output[0].type.tensor_type.shape.dim
    ]
    head_in = head.graph.input[0].name

    # Some transformer backbones produce (batch, seq_len, embed); we need
    # (batch, embed) for the dense head, so insert a ReduceMean over axis 1.
    pool_out = foundation_out
    extra_nodes: list = []
    extra_inits: list = []
    if len(foundation_out_shape) == 3:
        pool_out = foundation_out + "_pooled"
        extra_nodes.append(
            onnx.helper.make_node(
                "ReduceMean",
                inputs=[foundation_out],
                outputs=[pool_out],
                axes=[1],
                keepdims=0,
            )
        )

    # Rewire: replace every occurrence of head's input tensor name with
    # the (possibly pooled) foundation output so the two graphs connect.
    head_nodes = list(head.graph.node)
    for node in head_nodes:
        for i, inp in enumerate(node.input):
            if inp == head_in:
                node.input[i] = pool_out

    all_nodes = list(foundation.graph.node) + extra_nodes + head_nodes
    all_inits = list(foundation.graph.initializer) + extra_inits + list(head.graph.initializer)

    # Graph input comes from foundation; rename to the canonical name if needed.
    graph_inputs = list(foundation.graph.input)
    orig_input_name = graph_inputs[0].name if graph_inputs else foundation_input_name
    if orig_input_name != foundation_input_name:
        graph_inputs[0].name = foundation_input_name
        # Propagate the rename through all node references so the graph is consistent.
        for node in all_nodes:
            for i, inp in enumerate(node.input):
                if inp == orig_input_name:
                    node.input[i] = foundation_input_name

    # Graph output comes from the head.
    graph_outputs = list(head.graph.output)

    new_graph = onnx.helper.make_graph(
        all_nodes,
        "full_model",
        graph_inputs,
        graph_outputs,
        initializer=all_inits,
    )

    new_model = onnx.helper.make_model(new_graph)
    # IR version must be at least as high as either constituent model.
    new_model.ir_version = max(foundation.ir_version, head.ir_version)

    # Merge opsets: keep the higher version per domain; include all domains
    # from both models (e.g. ai.onnx.ml from sklearn).
    domains: dict[str, int] = {}
    for op in list(foundation.opset_import) + list(head.opset_import):
        domains[op.domain] = max(domains.get(op.domain, 0), op.version)

    del new_model.opset_import[:]
    for domain, version in domains.items():
        entry = new_model.opset_import.add()
        entry.domain = domain
        entry.version = version

    onnx.save(new_model, str(out_path))
    print(f"  ONNX full model   → {out_path}")


def _rename_graph_tensor(graph, old: str, new: str) -> None:
    """Rename every reference to tensor *old* → *new* across a graph in place."""
    if old == new:
        return
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.name == old:
            vi.name = new
    for node in graph.node:
        node.input[:] = [new if t == old else t for t in node.input]
        node.output[:] = [new if t == old else t for t in node.output]


def _tflite_head_to_onnx(
    tflite_path: Path,
    out_path: Path,
    opset: int = 13,
    input_name: str | None = None,
    output_name: str | None = None,
) -> None:
    """Convert a TFLite classifier head to ONNX using tf2onnx.

    Used by run_merge() (with default TFLite tensor names) and by the
    --extract_head ONNX path. When *input_name* / *output_name* are given, the
    graph's single input / output tensors are renamed for a clean API (tf2onnx
    otherwise carries over TFLite's verbose internal tensor names).
    """
    import tf2onnx

    model_proto, _ = tf2onnx.convert.from_tflite(str(tflite_path), opset=opset)
    graph = model_proto.graph
    if input_name and graph.input:
        _rename_graph_tensor(graph, graph.input[0].name, input_name)
    if output_name and graph.output:
        _rename_graph_tensor(graph, graph.output[0].name, output_name)
    onnx.save(model_proto, str(out_path))


def export_tflite_head_to_onnx(
    tflite_path: Path,
    out_path: Path,
    keep_indices: list[int] | None = None,
    data_type: str = "FP32",
    input_name: str = "embedding",
    output_name: str = "scores",
) -> Path:
    """Export a sliced TFLite head to ONNX (rename I/O, filter outputs, set precision).

    The head is converted directly from its TFLite form — since a sliced head
    contains only dense/activation ops, tf2onnx handles it cleanly (unlike the
    full model, whose backbone uses ops such as RFFT2D that ONNX lacks).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _tflite_head_to_onnx(tflite_path, out_path, input_name=input_name, output_name=output_name)
    if keep_indices is not None:
        _filter_onnx_outputs(out_path, keep_indices, "keras")
    _apply_onnx_precision(out_path, data_type)
    return out_path


def _filter_onnx_outputs(model_path: Path, keep_indices: list[int], classifier_type: str) -> None:
    """Append a Gather node that selects only keep_indices from the score/probability output.

    The model is updated in-place on disk.
    For keras: the single output tensor (logits/scores) is gathered along axis=-1.
    For sklearn: the 'probabilities' tensor is gathered; the integer 'label' output is dropped.
    """
    import numpy as np

    model = onnx.load(str(model_path))
    graph = model.graph

    # Identify the score output to filter
    if classifier_type == "sklearn":
        score_out = next((o for o in graph.output if o.name == "probabilities"), None)
        if score_out is None:
            return
        # Drop the integer label output — it's meaningless after index remapping
        label_out = next((o for o in graph.output if o.name == "label"), None)
        if label_out is not None:
            graph.output.remove(label_out)
    else:
        score_out = graph.output[0]

    original_name = score_out.name
    gathered_name = original_name + "_filtered"

    # Add keep_indices as a constant initializer
    indices_tensor = onnx.numpy_helper.from_array(
        np.array(keep_indices, dtype=np.int64), name="__keep_indices"
    )
    graph.initializer.append(indices_tensor)

    # Gather node: axis=1 for [batch, N] → [batch, K]
    gather_node = onnx.helper.make_node(
        "Gather",
        inputs=[original_name, "__keep_indices"],
        outputs=[gathered_name],
        axis=1,
    )
    graph.node.append(gather_node)

    # Replace the output descriptor
    graph.output.remove(score_out)
    graph.output.append(
        onnx.helper.make_tensor_value_info(
            gathered_name, onnx.TensorProto.FLOAT, [None, len(keep_indices)]
        )
    )

    onnx.save(model, str(model_path))
