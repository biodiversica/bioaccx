# Outputs

## Output directory structure

All outputs are written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/`.

The `foundation_model_id` is a 16-bit hex identifier resolved from the registry (see [Foundation model registry](registry.md#foundation-model-registry)).  Exported model files also carry a precision suffix (`_fp32`, `_fp16`, or `_int8`; see [Output precision](#output-precision-fp32--fp16--int8)).  For example, a classifier trained on BirdNET 2.4 FP32 ONNX (`0xbb00`) would produce:

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
by a training run (see [Computing the embedding database + UMAP](embeddings-umap.md#computing-the-embedding-database--umap-no-training));
they land in the same directory, under the same stem.

### Dataset metadata JSON

Every run that builds a dataset (training, `--dataset`, `--embeddings`) writes
`[stem]_dataset_metadata.json` next to the per-sample `[stem]_dataset_list.csv`. While the CSV
lists one row per sample, the JSON records **how the dataset was produced and what it contains**:

| Key | Contents |
|---|---|
| `foundation_model` | The window the samples were chunked for: `sample_rate`, `window_samples`, `window_seconds`, plus model identity |
| `sources` | One entry per dataset source (each `dataset.sources` block, or the single `dataset` block): `data_dir`, `label_mode`, table paths, `overlap`, filter/speed preprocessing, the `augmentation` block, … |
| `run` | Run-level settings shared by all sources: `test_ratio`, `random_seed`, `append_dataset_path`, `audio_extensions`, caches |
| `totals` | `n_train`, `n_test`, `n_total`, `num_labels`, `n_source_files`, and `n_augmented` / `n_appended` when applicable |
| `labels` | Per label: `train`, `test` and `total` sample counts (plus `train_augmented` / `test_augmented` when augmentation was used) |

Only parameters that actually shaped the dataset are written: fields left unset, or left at a
default that had no effect, are omitted. `xc_api_key` is always redacted, so the file can be
shared alongside the dataset.

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
`*_evaluation.csv` with the same columns and layout as BirdNET-Analyzer's `*_evaluation.csv`,
so the two can be compared or loaded by the same tooling:

| Column | Meaning |
|---|---|
| `Class` | Label name; the first row is `OVERALL (Macro-avg)`, followed by `OVERALL (Macro-avg, included)` when `exclude_labels` removes a class |
| `Precision (0.5)` / `Recall (0.5)` / `F1 Score (0.5)` | Metrics at the fixed 0.5 threshold |
| `Precision (opt)` / `Recall (opt)` / `F1 Score (opt)` | Metrics at the per-class F1-optimal threshold |
| `AUPRC` / `AUROC` | Threshold-free area under the precision-recall and ROC curves |
| `Optimal Threshold` | Threshold maximising that class' F1, searched over 0.10–0.85 in steps of 0.05 |
| `True/False Positives`, `True/False Negatives` | Confusion counts **at the optimal threshold** |
| `Samples` / `Percentage (%)` | Test samples of that class, and their share of the test set |

Each class is scored one-vs-rest: positives are the test samples of that class and the score
is that class' output column. Scores are taken straight from the model when the head ends in
`sigmoid` or `softmax`, and softmaxed when the head outputs raw logits
(`output_activation: null`) — so a reported threshold is the value you would apply to the
exported model's output. The macro-average row averages the per-class values; `AUPRC`/`AUROC`
are undefined for a class with no test samples (shown as `—` in the report, empty in the CSV)
and such classes are skipped in that average.

When `exclude_labels` is set, the classes it names are still trained on and evaluated, but the
exported model does not output them. The `OVERALL (Macro-avg, included)` row averages only the
classes the exported model keeps, so it measures the model you actually deploy. The report
summary adds the matching `Macro F1 (included, …)` line, and so does the sklearn report.

### What the exported model outputs

`output_activation` is a real layer in the trained graph, so with the default
`export_logits: false` it is **part of every export** — the head, the merged full model, ONNX and
TFLite alike. `sigmoid` or `softmax` scores therefore come out of the exported model ready to use,
with no activation to apply at inference time:

| `output_activation` | `export_logits` | Exported model emits |
|---|---|---|
| `null` (default) | (n/a) | Raw logits — apply a softmax (the training loss's). With `exclude_labels`: softmax probabilities, see below |
| `sigmoid` | `false` | Per-class probabilities in `(0, 1)`, independent of each other |
| `softmax` | `false` | A probability distribution over the classes (sums to 1) |
| `grouped_softmax` | `false` | A softmax per group; see [Grouped softmax](configuration.md#grouped-softmax) |
| `sigmoid` / `softmax` | `true` | Raw logits — see below |

`_metadata.json` records both values under `keras_classifier`, and the resulting output under
`exported_output.keras`: `scores` is `probabilities` or `logits`, and `activation` is what a
consumer must apply to the output (`identity` when it already holds probabilities). In
auricularia, `activation: identity` is the right setting for probability output.

> **`softmax` + `exclude_labels`:** the label filter is a gather applied *after* the activation, so
> the softmax is still computed over every trained class and the excluded columns are then dropped.
> The retained columns keep their calibrated values but no longer sum to 1 — a window that was
> mostly `background` yields small scores across the board, which is usually the point. Use
> `sigmoid` if you need each remaining column to stand on its own, or a
> [grouped softmax](configuration.md#grouped-softmax) if you need the dropped mass to stay recoverable.

> **`null` + `exclude_labels`:** a `null` head is trained with a softmax that lives only in the
> loss, so gathering its logits would lose the excluded classes' share of the softmax and no
> consumer could rebuild the probabilities the report describes. Instead, the export inserts a
> softmax over every trained class *before* the gather: the file emits the trained probabilities
> for the kept columns (they do not sum to 1; the remainder is `P(excluded)`), and a note is
> printed at export time. `softmax` + `export_logits` + `exclude_labels` is rejected for the same
> reason — the file cannot be both logits and missing the excluded classes.

### Training with an activation, exporting logits

`keras.export_logits: true` strips the final activation layer from the exported head while
leaving training untouched — the same thing BirdNET-Analyzer does with `classifier.pop()`
before saving. Combined with `output_activation: sigmoid` it gives you BirdNET's arrangement:

```yaml
training:
  keras:
    output_activation: sigmoid   # trains with sigmoid + binary cross-entropy
    export_logits: true          # exported head emits raw logits
```

The stripped model shares weights with the trained one — nothing is retrained, and
`sigmoid(exported_logits)` reproduces the trained model's output exactly. Both the ONNX and
TFLite heads (and the merged full models) are exported from the stripped graph; label
filtering via `exclude_labels` still applies to a `sigmoid` head, whose classes are independent;
it is rejected for a `softmax` (or grouped) head. The setting is a no-op when
`output_activation` is `null`, since there is no activation to remove.

Note that the per-class evaluation is computed on the *trained* model, so its thresholds are
in probability space while the exported head emits logits — apply the activation to the head
output before comparing against them. The report prints a reminder to that effect when
`export_logits` is on, and `_metadata.json` records both `output_activation` and
`export_logits` under `keras_classifier` for inference code to read.

---

## Output precision (FP32 / FP16 / INT8)

Use `output.data_types` to choose the precision(s) of the exported model/classifier. The list **fully controls** which precisions are written; when omitted it defaults to the foundation model's own `data_type` (usually `FP32`). Each requested precision produces a **separate** file, tagged with the precision in its name:

```yaml
output:
  data_types: [FP32, FP16, INT8]   # any subset; omit to use the foundation model's data_type
```

This writes, for each model variant, files like `..._keras_head_fp32.onnx`, `..._keras_head_fp16.onnx`, and `..._keras_head_int8.tflite`. The chosen precisions are also recorded in `_metadata.json` under `output_data_types`.

How each precision is produced:

| Precision | ONNX | TFLite |
|-----------|------|--------|
| `FP32` | exported as-is (float32) | true float32 (no weight quantization) |
| `FP16` | weights cast to float16, float32 I/O preserved | float16 weights |
| `INT8` | dynamic/weight-only quantization | dynamic-range int8 weights |

`INT8` uses **dynamic / weight-only** quantization — weights are stored as int8 while activations stay float — so **no calibration dataset is required**.

Notes:
- **Full models**: for ONNX, the chosen precision applies to the *entire* merged graph (backbone + head). For TFLite, only the classifier-head portion is converted; the merged backbone keeps its on-disk precision.
- **sklearn heads** are exported at `FP32` only (their `ai.onnx.ml` operators are not quantizable); other requested precisions are skipped with a note.
- ONNX FP16/INT8 require `onnxconverter-common` and `onnxruntime`+`sympy`, which ship with the default install.

---

## Excluding labels from the exported model

Use `exclude_labels` to remove classes (e.g. background noise) from the final model output while still using them during training. This improves training stability without polluting the inference output.

```yaml
output:
  exclude_labels: [background, noise, unknown]
```

The excluded classes are present in the training data and the internal classifier, but the exported ONNX/TFLite model's output tensor only contains scores for the remaining classes. The `_labels.txt` file reflects the final output label order.

The filter is a gather appended after the classifier's output activation, so with `output_activation: softmax` the remaining columns are probabilities computed over *all* trained classes and no longer sum to 1 — see [What the exported model outputs](#what-the-exported-model-outputs).
