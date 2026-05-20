"""TFLite export for Keras classifiers models."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def export_tflite(
    classifier,
    classifier_type: Literal["keras", "sklearn"],
    embed_dim: int,
    out_path: Path,
    # For full-model export only
    foundation_path: Path | None = None,
    foundation_input_name: str = "input",
    output_type: Literal["head", "full"] = "head",
    keep_indices: list[int] | None = None,
    tflite_output_tensor_offset: int = 0,
) -> Path | None:
    """Convert Keras classifier (and optionally full pipeline) to TFLite.

    Returns None if TFLite export is not supported for the given configuration
    (sklearn classifiers are not supported).

    Parameters
    ----------
    classifier:
        Trained Keras Model.
    classifier_type:
        ``"keras"`` or ``"sklearn"``.  sklearn causes an early return of None.
    embed_dim:
        Embedding vector dimensionality.
    out_path:
        Destination .tflite file.
    foundation_path:
        Path to the backbone .tflite file (required for output_type='full').
    tflite_output_tensor_offset:
        Offset applied to the backbone's declared output tensor index to reach
        the embedding tensor.  Matches FoundationModelConfig.tflite_output_tensor_offset.
    keep_indices:
        When set, a Gather layer is prepended to slice the output to only the
        desired class indices before converting.
    """
    if classifier_type == "sklearn":
        print("  [tflite] Skipping: TFLite export is not supported for sklearn classifiers.")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _slice_keras_output(classifier, keep_indices) if keep_indices is not None else classifier

    if output_type == "full":
        if foundation_path is None or not str(foundation_path).endswith(".tflite"):
            print("  [tflite] Skipping full model: foundation_path must be a .tflite file.")
            return None
        return _export_full_tflite(
            model, foundation_path, out_path, tflite_output_tensor_offset
        )

    return _export_head_tflite(model, out_path)


def _slice_keras_output(model, keep_indices: list[int]):
    """Wrap *model* with a Lambda layer that gathers only *keep_indices* outputs."""
    import tensorflow as tf
    inp = model.input
    sliced = tf.keras.layers.Lambda(
        lambda x: tf.gather(x, keep_indices, axis=-1),
        name="output_filter",
    )(model.output)
    sliced_model = tf.keras.Model(inp, sliced)
    sliced_model._report_history = getattr(model, "_report_history", {})
    sliced_model._report_params  = getattr(model, "_report_params", {})
    return sliced_model


def _export_head_tflite(model, out_path: Path) -> Path:
    """Convert a Keras model to a TFLite flatbuffer."""
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"  Keras TFLite head → {out_path}")
    return out_path


def _export_full_tflite(
    classifier,
    foundation_path: Path,
    out_path: Path,
    tflite_output_tensor_offset: int,
) -> Path:
    """Merge a tflite backbone with a Keras classifier head into a single tflite.

    Strategy:
    1. Trim the backbone to expose only the embedding tensor as output, removing
       the original classifier head ops, tensors, and weight buffers.
    2. Convert the Keras head to tflite bytes.
    3. Merge the two flatbuffers: remap head tensor/buffer/opcode indices,
       wire backbone embedding output → head input, combine operator lists.
    """
    from tensorflow.lite.python.util import flatbuffer_utils

    backbone_bytes = foundation_path.read_bytes()
    bb_sg = flatbuffer_utils.read_model_from_bytearray(backbone_bytes).subgraphs[0]
    embed_idx = bb_sg.outputs[0] + tflite_output_tensor_offset
    backbone_bytes = _trim_tflite_to_output(backbone_bytes, embed_idx)
    head_bytes = _keras_to_tflite_bytes(classifier)
    merged = _merge_tflite_models(backbone_bytes, head_bytes)
    out_path.write_bytes(merged)
    print(f"  TFLite full model → {out_path}")
    return out_path


def _trim_tflite_to_output(tflite_bytes: bytes, keep_output_idx: int) -> bytes:
    """Return a new tflite that computes only *keep_output_idx* as its output.

    Does a backward BFS from *keep_output_idx* to identify every op and tensor
    needed to produce it, then rebuilds the model dropping everything else
    (including their buffer data) and re-indexes all references.
    """
    from tensorflow.lite.python.util import flatbuffer_utils

    model = flatbuffer_utils.read_model_from_bytearray(tflite_bytes)
    sg = model.subgraphs[0]

    # BFS backward: collect tensor indices needed to compute keep_output_idx
    needed_tensors: set[int] = set()
    queue = [keep_output_idx]
    while queue:
        t = queue.pop()
        if t < 0 or t in needed_tensors:
            continue
        needed_tensors.add(t)
        for op in sg.operators:
            if t in op.outputs:
                queue.extend(inp for inp in op.inputs if inp >= 0)

    # Live ops: any op that produces at least one needed tensor
    live_op_idxs = [
        i for i, op in enumerate(sg.operators)
        if any(t in needed_tensors for t in op.outputs if t >= 0)
    ]

    # An op must run as a whole: add all outputs of live ops to needed_tensors
    # so tensor_remap covers them (prevents KeyError for multi-output ops).
    for i in live_op_idxs:
        for t in sg.operators[i].outputs:
            if t >= 0:
                needed_tensors.add(t)

    # Tensor and buffer re-indexing
    live_tensor_idxs = sorted(needed_tensors)
    tensor_remap = {old: new for new, old in enumerate(live_tensor_idxs)}

    live_buf_idxs = sorted({0} | {sg.tensors[t].buffer for t in live_tensor_idxs})
    buf_remap = {old: new for new, old in enumerate(live_buf_idxs)}

    # Rebuild model in-place
    new_model = copy.deepcopy(model)
    new_sg = new_model.subgraphs[0]

    new_sg.tensors = []
    for t_idx in live_tensor_idxs:
        t = copy.deepcopy(sg.tensors[t_idx])
        t.buffer = buf_remap[t.buffer]
        new_sg.tensors.append(t)

    new_sg.operators = []
    for op_idx in live_op_idxs:
        op = copy.deepcopy(sg.operators[op_idx])
        op.inputs  = [tensor_remap[t] if t >= 0 else t for t in op.inputs]
        op.outputs = [tensor_remap[t] if t >= 0 else t for t in op.outputs]
        new_sg.operators.append(op)

    new_sg.inputs  = [tensor_remap[t] for t in sg.inputs if t in tensor_remap]
    new_sg.outputs = [tensor_remap[keep_output_idx]]

    new_model.buffers = [copy.deepcopy(model.buffers[b]) for b in live_buf_idxs]

    return bytes(flatbuffer_utils.convert_object_to_bytearray(new_model))


def _keras_to_tflite_bytes(model) -> bytes:
    """Convert a Keras model to raw TFLite flatbuffer bytes."""
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return bytes(converter.convert())


def _merge_tflite_models(backbone_bytes: bytes, head_bytes: bytes) -> bytes:
    """Merge two TFLite flatbuffers: backbone (→ embedding) + head (embedding →).

    The backbone's single output tensor is wired to the head's single input
    tensor.  All tensor/buffer/opcode indices in the head are remapped to avoid
    collision with the backbone's index space.
    """
    from tensorflow.lite.python.util import flatbuffer_utils

    bb = flatbuffer_utils.read_model_from_bytearray(backbone_bytes)
    hd = flatbuffer_utils.read_model_from_bytearray(head_bytes)

    bb_sg = bb.subgraphs[0]
    hd_sg = hd.subgraphs[0]

    bb_embed_idx = bb_sg.outputs[0]
    hd_input_idx = hd_sg.inputs[0]

    n_bb_tensors = len(bb_sg.tensors)
    n_bb_buffers = len(bb.buffers)
    n_bb_opcodes = len(bb.operatorCodes)

    # --- Merge operator codes ---
    # Reuse backbone opcodes with matching builtin_code; append new ones.
    hd_opcode_map: dict[int, int] = {}
    for i, hd_oc in enumerate(hd.operatorCodes):
        match = next(
            (j for j, bb_oc in enumerate(bb.operatorCodes)
             if bb_oc.builtinCode == hd_oc.builtinCode
             and bb_oc.customCode == hd_oc.customCode),
            None,
        )
        if match is not None:
            hd_opcode_map[i] = match
        else:
            hd_opcode_map[i] = len(bb.operatorCodes)
            bb.operatorCodes.append(hd_oc)

    # --- Merge buffers ---
    # Buffer 0 in both models is the empty sentinel; map head's 0 → backbone's 0.
    # All other head buffers are appended after backbone's buffers.
    hd_buf_map: dict[int, int] = {0: 0}
    for i in range(1, len(hd.buffers)):
        hd_buf_map[i] = n_bb_buffers + i - 1
        bb.buffers.append(hd.buffers[i])

    # --- Merge tensors ---
    # Head's input tensor → backbone's embedding output tensor (no new tensor added).
    # All other head tensors are appended after backbone's tensors.
    hd_tensor_map: dict[int, int] = {}
    next_new_tensor = n_bb_tensors
    for i, tensor in enumerate(hd_sg.tensors):
        if i == hd_input_idx:
            hd_tensor_map[i] = bb_embed_idx
        else:
            hd_tensor_map[i] = next_new_tensor
            next_new_tensor += 1
            t = copy.deepcopy(tensor)
            t.buffer = hd_buf_map.get(tensor.buffer, n_bb_buffers + tensor.buffer)
            bb_sg.tensors.append(t)

    # --- Append head operators ---
    for op in hd_sg.operators:
        new_op = copy.deepcopy(op)
        new_op.inputs  = [hd_tensor_map[t] if t >= 0 else t for t in op.inputs]
        new_op.outputs = [hd_tensor_map[t] if t >= 0 else t for t in op.outputs]
        new_op.opcodeIndex = hd_opcode_map[op.opcodeIndex]
        bb_sg.operators.append(new_op)

    # --- Update model outputs to head's outputs ---
    bb_sg.outputs = [hd_tensor_map[o] for o in hd_sg.outputs]

    return bytes(flatbuffer_utils.convert_object_to_bytearray(bb))
