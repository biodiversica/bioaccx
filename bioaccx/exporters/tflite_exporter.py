"""TFLite export for Keras classifiers and full (SavedModel + Keras) models."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def export_tflite(
    classifier,                         # trained Keras model (sklearn not supported)
    classifier_type: Literal["keras", "sklearn"],
    embed_dim: int,
    out_path: Path,
    # For full-model export only
    foundation_savedmodel_path: Path | None = None,
    foundation_input_name: str = "input",
    output_type: Literal["head", "full"] = "head",
    keep_indices: list[int] | None = None,
) -> Path | None:
    """Convert Keras classifier (and optionally full pipeline) to TFLite.

    Returns None if TFLite export is not supported for the given configuration
    (e.g. sklearn + tflite or non-protobuf foundation with output_type='full').
    """
    if classifier_type == "sklearn":
        print(
            "  [tflite] Skipping: TFLite export is not supported for sklearn classifiers."
        )
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap classifier to slice output if needed
    model = _slice_keras_output(classifier, keep_indices) if keep_indices is not None else classifier

    if output_type == "full":
        if foundation_savedmodel_path is None:
            print(
                "  [tflite] Skipping full model: foundation must be a TF SavedModel "
                "(protobuf) for TFLite full-model export."
            )
            return None
        return _export_full_tflite(
            model, foundation_savedmodel_path, embed_dim, out_path, foundation_input_name
        )

    return _export_head_tflite(model, embed_dim, out_path)


def _slice_keras_output(model, keep_indices: list[int]):
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


def _export_head_tflite(model, embed_dim: int, out_path: Path) -> Path:
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
    embed_dim: int,
    out_path: Path,
    foundation_input_name: str,
) -> Path:
    """Combine TF SavedModel foundation + Keras classifier → TFLite."""
    import tensorflow as tf

    pb_model = tf.saved_model.load(str(foundation_path))
    sigs = list(pb_model.signatures.keys())
    embed_sig = "embeddings" if "embeddings" in sigs else sigs[0]

    class FullModel(tf.Module):
        def __init__(self):
            super().__init__()
            self.foundation = pb_model
            self.head = classifier

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name=foundation_input_name)
        ])
        def __call__(self, x):
            emb_out = self.foundation.signatures[embed_sig](x)
            emb = list(emb_out.values())[0]
            if len(emb.shape) == 3:
                emb = tf.reduce_mean(emb, axis=1)
            return self.head(emb)

    full = FullModel()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [full.__call__.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"  TFLite full model → {out_path}")
    return out_path
