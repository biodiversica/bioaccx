# Outputs

## Output directory structure

All outputs are written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/`.

The `foundation_model_id` is a 16-bit hexadecimal identifier resolved from the registry (see [Foundation model registry](registry.md#foundation-model-registry)). Exported model files also carry a precision suffix (`_fp32`, `_fp16` or `_int8`; see [Output precision](#output-precision-fp32--fp16--int8)). For example, a classifier trained on BirdNET 2.4 FP32 ONNX (`0xbb00`) produces:

```
custom_models/
  my_classifier_0xbb00_v1.0/
    my_classifier_0xbb00_v1.0_labels.txt              # output class names, one per line
    my_classifier_0xbb00_v1.0_metadata.json           # full metadata JSON
    my_classifier_0xbb00_v1.0_dataset_list.csv        # per-sample split/label summary
    my_classifier_0xbb00_v1.0_dataset_metadata.json   # how the dataset was built + per-label counts
    my_classifier_0xbb00_v1.0_keras_head_fp32.onnx    # Keras head only (embedding input)
    my_classifier_0xbb00_v1.0_keras_full_fp32.onnx    # Keras + backbone merged (audio input)
    my_classifier_0xbb00_v1.0_keras_head_fp32.tflite  # Keras head only (TFLite)
    my_classifier_0xbb00_v1.0_keras_full_fp32.tflite  # Keras + backbone merged (TFLite)
    my_classifier_0xbb00_v1.0_sklearn_head_fp32.onnx  # sklearn head only
    my_classifier_0xbb00_v1.0_keras_report.txt        # training history + metrics
    my_classifier_0xbb00_v1.0_evaluation.csv          # per-class evaluation (BirdNET-compatible)
    my_classifier_0xbb00_v1.0_sklearn_report.txt    # sklearn metrics
    my_classifier_0xbb00_v1.0_comparison_report.txt # side-by-side comparison
    0xbb00_embeddings.db                            # exported embeddings (SQLite, optional)
    my_classifier_0xbb00_v1.0_umap.csv              # UMAP coordinates + label, split, key, cluster
    my_classifier_0xbb00_v1.0_umap.png              # UMAP scatter-plot, coloured by label
    my_classifier_0xbb00_v1.0_clusters.png          # same projection by KMeans cluster, with NMI
    dataset/                                        # exported chunked WAV files (optional)
      train/
        crow/
        robin/
      test/
        crow/
        robin/
```

The three UMAP files are written by `bioaccx embeddings` with `umap.enabled`, not
by a training run (see [Computing the embedding database and UMAP](embeddings-umap.md#computing-the-embedding-database-and-umap-no-training));
they are written to the same directory, under the same stem.

### Dataset metadata JSON

Every run that builds a dataset (`bioaccx train`, `bioaccx dataset` and `bioaccx embeddings`) writes
`[stem]_dataset_metadata.json` next to the per-sample `[stem]_dataset_list.csv`. Whereas the CSV
lists one row per sample, the JSON records **how the dataset was produced and what it contains**:

| Key | Contents |
|---|---|
| `foundation_model` | The window for which the samples were chunked (`sample_rate`, `window_samples`, `window_seconds`) and the identity of the model |
| `sources` | One entry per dataset source (each `dataset.sources` block, or the single `dataset` block): `data_dir`, `label_mode`, table paths, `overlap`, filter and speed preprocessing, the `augmentation` block, and so on |
| `run` | Run-level settings shared by all sources: `test_ratio`, `random_seed`, `append_dataset_path`, `audio_extensions`, caches |
| `totals` | `n_train`, `n_test`, `n_total`, `num_labels`, `n_source_files`, and `n_augmented` / `n_appended` when applicable |
| `labels` | Per label: `train`, `test` and `total` sample counts (plus `train_augmented` / `test_augmented` when augmentation was used) |

Only parameters that actually affected the dataset are written: fields left unset, or left at a
default that had no effect, are omitted. `xc_api_key` is always redacted, so the file can be
shared together with the dataset.

```json
{
  "created_at": "2026-08-20T15:26:45",
  "model_stem": "my_classifier_0xbb00_v1.0",
  "foundation_model": {
    "name": "birdnet", "version": "2.4", "data_type": "FP32", "format": "onnx",
    "sample_rate": 48000, "window_samples": 144000, "window_seconds": 3.0
  },
  "sources": [
    {
      "data_dir": "/data/recordings", "label_mode": "subfolders",
      "overlap": 0.5, "speed": 1.0,
      "augmentation": {
        "snr_levels": [0, 10], "augmentation_dir": "/data/noise",
        "keep_original": true, "augment_test": false, "skip_labels": ["background"]
      }
    }
  ],
  "run": {
    "audio_extensions": ["wav", "flac", "mp3", "ogg"],
    "embedding_workers": 4, "test_ratio": 0.2, "random_seed": 42
  },
  "totals": {
    "num_labels": 2, "n_train": 240, "n_test": 60,
    "n_total": 300, "n_source_files": 100, "n_augmented": 160
  },
  "labels": {
    "crow":  {"train": 120, "test": 30, "total": 150, "train_augmented": 80, "test_augmented": 0},
    "robin": {"train": 120, "test": 30, "total": 150, "train_augmented": 80, "test_augmented": 0}
  }
}
```

### Per-class evaluation

The Keras report ends with a **Per-class Evaluation** table, also written separately as
`*_evaluation.csv`, with the same columns and layout as the `*_evaluation.csv` of
BirdNET-Analyzer, so that the two can be compared or read by the same tools:

| Column | Meaning |
|---|---|
| `Class` | Label name; the first row is `OVERALL (Macro-avg)`, followed by `OVERALL (Macro-avg, included)` when `exclude_labels` removes a class |
| `Precision (0.5)` / `Recall (0.5)` / `F1 Score (0.5)` | Metrics at the fixed 0.5 threshold |
| `Precision (opt)` / `Recall (opt)` / `F1 Score (opt)` | Metrics at the per-class F1-optimal threshold |
| `AUPRC` / `AUROC` | Threshold-free area under the precision-recall and ROC curves |
| `Optimal Threshold` | Threshold that maximises the F1 of the class, searched over 0.10–0.85 in steps of 0.05 |
| `True/False Positives`, `True/False Negatives` | Confusion counts **at the optimal threshold** |
| `Samples` / `Percentage (%)` | Test samples of the class, and their proportion of the test set |

Each class is scored one-vs-rest: the positives are the test samples of the class, and the score
is the output column of the class. Scores are taken directly from the model when the head ends
in `sigmoid` or `softmax`, and a softmax is applied when the head outputs raw logits
(`output_activation: null`), so a reported threshold is the value to apply to the output of the
exported model. The macro-average row averages the per-class values. `AUPRC` and `AUROC` are
undefined for a class without test samples (shown as `—` in the report and left empty in the
CSV), and such classes are excluded from the average.

When `exclude_labels` is set, the listed classes are still used for training and evaluation, but
the exported model does not output them. The `OVERALL (Macro-avg, included)` row averages only the
classes kept by the exported model, and therefore measures the model that is actually deployed.
The report summary, like the sklearn report, includes the corresponding
`Macro F1 (included, …)` line.

### What the exported model outputs

`output_activation` is an actual layer of the trained graph, so with the default
`export_logits: false` it is **part of every export**: the head and the merged full model, in both
ONNX and TFLite. `sigmoid` or `softmax` scores are therefore output by the exported model directly,
and no activation needs to be applied at inference time:

| `output_activation` | `export_logits` | Exported model emits |
|---|---|---|
| `null` (default) | (n/a) | Raw logits of a head trained with softmax; a softmax over all classes must be applied. Cannot be combined with `exclude_labels` (see below) |
| `sigmoid` | `false` | Per-class probabilities in `(0, 1)`, independent of each other |
| `softmax` | `false` | A probability distribution over the classes (sums to 1) |
| `grouped_softmax` | `false` | A softmax per group; see [Grouped softmax](configuration.md#grouped-softmax) |
| `sigmoid` / `softmax` | `true` | Raw logits (see below) |

`_metadata.json` records both values under `keras_classifier`, and the resulting output under
`exported_output.keras`: `scores` is `probabilities` or `logits`, and `activation` is the
activation that a consumer must apply to the output (`identity` when the output already contains
probabilities). In auricularia, `activation: identity` is the correct setting for probability
output.

> **`softmax` + `exclude_labels`:** the label filter is a gather applied *after* the activation, so
> the softmax is still computed over every trained class before the excluded columns are removed.
> The retained columns keep their calibrated values but no longer sum to 1: a window that was
> mostly `background` yields low scores for all classes, which is usually the intended behaviour.
> `sigmoid` is preferable when each remaining column must be independent, and a
> [grouped softmax](configuration.md#grouped-softmax) when the removed probability mass must remain
> recoverable.

> **`null` + `exclude_labels` is rejected**, and so is `softmax` + `export_logits` + `exclude_labels`.
> Both train the same softmax head and export its raw logits. Only the differences between these
> logits are meaningful, so a consumer must apply a softmax over *every* trained class; the gather,
> however, would already have removed the excluded classes, transferring their share to the
> retained ones. `bioaccx train` rejects the combination before loading any data. The output must
> be chosen explicitly: `softmax` for the trained probabilities (the softmax is applied before the
> gather), or `sigmoid` with `export_logits: true` for per-class logits suitable for a
> BirdNET-style sigmoid and threshold. (Between 2026-09-30 and this change, a `null` head with
> `exclude_labels` had a softmax inserted before the gather without notice and emitted
> probabilities; its `_metadata.json` records `softmax_before_filter: true`.)

### Training with an activation, exporting logits

`keras.export_logits: true` removes the final activation layer from the exported head while
leaving training unchanged, as BirdNET-Analyzer does with `classifier.pop()` before saving.
Combined with `output_activation: sigmoid`, it reproduces the BirdNET arrangement:

```yaml
training:
  keras:
    output_activation: sigmoid   # trains with sigmoid + binary cross-entropy
    export_logits: true          # exported head emits raw logits
```

The modified model shares its weights with the trained one: nothing is retrained, and
`sigmoid(exported_logits)` reproduces the output of the trained model exactly. Both the ONNX and
the TFLite heads, as well as the merged full models, are exported from the modified graph. Label
filtering with `exclude_labels` still applies to a `sigmoid` head, whose classes are independent,
but is rejected for a `softmax` or grouped head. The setting has no effect when
`output_activation` is `null`, since there is no activation to remove.

The per-class evaluation is computed on the *trained* model, so its thresholds are expressed as
probabilities, whereas the exported head emits logits; the activation must be applied to the head
output before comparing it with them. The report includes a reminder to this effect when
`export_logits` is enabled, and `_metadata.json` records both `output_activation` and
`export_logits` under `keras_classifier` for use by inference code.

---

## Output precision (FP32 / FP16 / INT8)

`output.data_types` sets the precisions of the exported model and classifier. The list **fully determines** which precisions are written; when omitted, it defaults to the `data_type` of the foundation model (usually `FP32`). Each requested precision produces a **separate** file, tagged with the precision in its name:

```yaml
output:
  data_types: [FP32, FP16, INT8]   # any subset; omit to use the data_type of the foundation model
```

For each model variant, this writes files such as `..._keras_head_fp32.onnx`, `..._keras_head_fp16.onnx` and `..._keras_head_int8.tflite`. The selected precisions are also recorded in `_metadata.json` under `output_data_types`.

How each precision is produced:

| Precision | ONNX | TFLite |
|-----------|------|--------|
| `FP32` | exported unchanged (float32) | float32 (no weight quantisation) |
| `FP16` | weights cast to float16, float32 I/O preserved | float16 weights |
| `INT8` | dynamic (weight-only) quantisation | dynamic-range int8 weights |

`INT8` uses **dynamic (weight-only)** quantisation: the weights are stored as int8, while the activations remain floating-point, so **no calibration dataset is required**.

Notes:
- **Full models:** for ONNX, the selected precision applies to the *entire* merged graph (backbone and head). For TFLite, only the classifier head is converted; the merged backbone keeps the precision it has on disk.
- **scikit-learn heads** are exported in `FP32` only, because their `ai.onnx.ml` operators cannot be quantised; other requested precisions are skipped with a note.
- ONNX FP16 and INT8 require `onnxconverter-common`, `onnxruntime` and `sympy`, which are included in the default installation.

---

## Excluding labels from the exported model

`exclude_labels` removes classes (for example, background noise) from the final model output while still using them during training. This improves training stability without adding unwanted classes to the inference output.

```yaml
output:
  exclude_labels: [background, noise, unknown]
```

The excluded classes are present in the training data and in the internal classifier, but the output tensor of the exported ONNX or TFLite model contains scores only for the remaining classes. The `_labels.txt` file reflects the final order of the output labels.

The filter is a gather appended after the output activation of the classifier, so with `output_activation: softmax` the remaining columns are probabilities computed over *all* trained classes and no longer sum to 1; see [What the exported model outputs](#what-the-exported-model-outputs).
