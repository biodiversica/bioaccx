# Configuration

All parameters live in a single YAML (or JSON) file. This page walks through it
section by section, with worked examples; every individual key, its default and
its description live in the [config reference](config-reference.md).

## `foundation_model`

### Registry shorthand (recommended)

Every known foundation model has a registry entry with a 16-bit hex ID.  Set `registry_id` to load all defaults in one line and download the model automatically from HuggingFace:

```yaml
foundation_model:
  registry_id: "0xbb00"   # BirdNET 2.4 ONNX — downloads from HuggingFace
```

Any field you add alongside `registry_id` overrides the registry default, so switching to a local copy requires only two lines:

```yaml
foundation_model:
  registry_id: "0xbb00"
  source: local
  path: /path/to/birdnet_backbone.onnx
```

Run `bioaccx --registry` to browse all available IDs, descriptions, and source URLs.

### Fully explicit config

You can also specify every field manually.  This is equivalent to the `registry_id` shorthand and remains fully supported:

```yaml
foundation_model:
  name: birdnet               # display name (used in reports and output filenames)
  version: "2.4"              # display version (used in reports and output filenames)
  data_type: FP32             # weight precision — part of the registry key

  # Model format on disk
  format: onnx                # onnx | tflite | protobuf (TF SavedModel)

  # --- Local file ---
  source: local
  path: /path/to/model.onnx

  # --- OR from HuggingFace Hub ---
  # source: huggingface
  # hf_repo: biodiversica/birdnet-headless
  # hf_filename: birdnet_headless.onnx   # optional; defaults to model.onnx
  # hf_revision: main                    # branch / tag / commit (optional)

  # --- OR from Kaggle ---
  # source: kaggle
  # kaggle_handle: google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu
  # kaggle_filename: perch_v2_backbone.onnx   # specific file within the downloaded dir

  # Audio preprocessing
  sample_rate: 48000
  window_seconds: 3.0         # duration of each input window
  # window_samples: 144000    # alternative: exact sample count (takes priority)

  # Tensor names (check your model's input/output node names)
  input_name: INPUT
  output_name: embedding
  embedding_size: 1024        # dimensionality of the embedding vector

  # ONNX-specific
  # onnx_providers: [CUDAExecutionProvider, CPUExecutionProvider]
  # onnx_batch_size: 1        # values > 1 enable GPU batching (ONNX only)

  # TFLite-specific: offset from the declared output tensor index to the
  # embedding tensor. Use -1 for models like BirdNET tflite where the
  # classifier head is the first declared output and the embedding sits
  # one tensor slot before it.
  # tflite_output_tensor_offset: -1
```

Every `foundation_model` key, with its default and its description, is in the [config reference](config-reference.md#foundation_model).

### Using the full BirdNET TFLite model

The BirdNET TFLite files contain a classifier head that outputs 6522 bird species. The embedding lives one tensor slot before the classifier output. With the registry shorthand, `tflite_output_tensor_offset: -1` is set automatically:

```yaml
foundation_model:
  registry_id: "0xbb02"
```

Equivalently, fully explicit and also if you want to supply a local path:

```yaml
foundation_model:
  name: birdnet
  version: "2.4"
  format: tflite
  source: local
  path: /path/to/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite
  sample_rate: 48000
  window_seconds: 3.0
  input_name: INPUT
  embedding_size: 1024
  tflite_output_tensor_offset: -1
```

When exporting a full TFLite model with this backbone, the original classifier head ops and weight tensors are removed from the merged output — only the backbone computation up to the embedding is retained, followed by your new classifier head.

### Using the full Perch v2 TFLite model

`0xbb12` is the Perch v2 TFLite. Unlike BirdNET's, its embedding is already the **first** of its four outputs (embedding, spatial_embedding, spectrogram, and a 14795-class label head), so no offset is needed:

```yaml
foundation_model:
  registry_id: "0xbb12"
```

Its embeddings match the `0xbb10` ONNX backbone to within 5e-7, so a classifier trained on either is equivalent. Choose `0xbb12` when the deliverable has to be a TFLite full model, since `output_type: full` can only merge a TFLite head with a TFLite backbone. The merged export is ~43 MB — the 14795-class head and the auxiliary branches are dropped by the trim.

### TFLite graph trimming

Models that bundle a classifier head or extra outputs compute all of it on every window, even though only the embedding is used. When `tflite_trim_to_embedding` is enabled (the default), the graph is trimmed to the embedding tensor once at load time and every later window runs the smaller graph. Measured on this machine:

| Model | As shipped | Trimmed | |
|---|---|---|---|
| Perch v2 (`0xbb12`) | 1.60 s/window | 0.197 s/window | 8.2× |
| BirdNET 2.4 (`0xbb02`) | 0.060 s/window | 0.041 s/window | 1.5× |

Embeddings are bit-identical either way. The cost is a one-off trim at load (~13 s for Perch, <1 s for BirdNET), cached in-process so the worker threads created by `embedding_workers` share it. Trimming also removes the need for `experimental_preserve_all_tensors` on offset models like `0xbb02`, which otherwise keeps every intermediate tensor in memory. If a trim fails or does not yield a single output of the expected `embedding_size`, bioaccx warns and falls back to the model as shipped; set `tflite_trim_to_embedding: false` to skip it entirely.

---

## `dataset`

```yaml
dataset:
  data_dir: /path/to/audio_dataset

  # How labels are organised in data_dir
  label_mode: subfolders      # subfolders | file_per_label | table

  # For label_mode: table only
  # table_file: /path/to/annotations.csv
  # filename_col: filename
  # label_col: label
  # start_col: start_time
  # end_col: end_time
  # split_col: split          # optional; values: train | test

  audio_extensions: [wav, flac, mp3, ogg]
  overlap: 0.0                # window overlap for file_per_label / table (0.0–1.0)
  embedding_workers: 4        # parallel workers for embedding extraction
  embeddings_cache_path: null # pre-computed embeddings cache path (optional)
  embeddings_format: sqlite   # sqlite | npy
  test_ratio: 0.2             # fraction of data for test set
  random_seed: 42

  # Optional: mix each training sample with background noise at multiple SNRs
  # augmentation:
  #   augmentation_dir: /path/to/noise_files
  #   snr_levels: [0, 10, 20]   # dB (power-based); one copy per level per noise file
  #   keep_original: true        # also keep the clean sample
  #   augment_test: false        # apply the same expansion to the test set
```

Every `dataset` key, with its default and its description, is in the [config reference](config-reference.md#dataset).

---

## `training`

```yaml
training:
  classifier: keras           # keras | sklearn | both

  keras:
    hidden_units: 256         # hidden layer size; 0 = single linear layer
    dropout: 0.25
    epochs: 50
    batch_size: 32
    learning_rate: 0.0001
    output_activation: null   # null (logits) | sigmoid | softmax | grouped_softmax
    export_logits: false      # train with the activation, export without it
    normalize_embeddings: true

    # Optional: grouped softmax — exclusive within a group, independent across
    # groups (e.g. several call types per species). See "Grouped softmax" below.
    # label_groups:
    #   BOABIS: [BOABIS1, BOABIS2, BOABIS3]
    #   DENMIN: [DENMIN1, DENMIN2, DENMIN3]

    # Optional: focal loss (replaces cross-entropy)
    # focal_loss: false
    # focal_loss_gamma: 2.0
    # focal_loss_alpha: 0.25

    # Optional: label smoothing
    # label_smoothing: false
    # label_smoothing_alpha: 0.1

    # Optional: mixup augmentation on training embeddings
    # mixup: false
    # mixup_ratio: 0.25
    # mixup_alpha: 0.2

    # Optional: minority class upsampling
    # upsampling_ratio: 0.0
    # upsampling_mode: repeat   # repeat | mean | linear | smote

  sklearn:
    C: 1.0
    max_iter: 2000
    solver: lbfgs
```

Every `training` key, with its default and its description, is in the [config reference](config-reference.md#training).

### Keras training details

- **Metrics**: AUPRC (area under precision-recall curve) and AUROC (area under ROC curve) are computed on both the training and validation sets every epoch, alongside accuracy.
- **LR schedule**: linear warmup for the first `max(3, epochs // 10)` epochs, then cosine decay to 10% of the peak LR.
- **Early stopping**: monitors `val_loss` with patience `max(5, epochs // 10)` and restores the best weights.
- **Output activation notes**:
  - `null` (default): raw logits; use `softmax` at inference time for class probabilities.
  - `sigmoid`: per-class binary probability; suitable for multi-label problems.
  - `softmax`: normalised class probabilities; use when the model should output probabilities directly.
  - `grouped_softmax`: softmax within each group of `label_groups`, groups independent — see below.
- **`normalize_embeddings`**: a `Normalization` layer is fitted on the training embeddings and baked into the exported model. This is applied before any Dense layer. Set to `false` to match the BirdNET-Analyzer architecture (backbone → FC → logits directly).

### Grouped softmax

A flat `softmax` makes *all* classes compete (they sum to 1, so two classes can never both score high), while `sigmoid` makes them all independent (nothing prevents two mutually exclusive classes from firing together). `grouped_softmax` sits between the two: a softmax is applied **within each group**, and the groups are independent of one another.

The motivating case is several call types per species — one species emits one call at a time, but two species can easily overlap in the same window:

```yaml
training:
  keras:
    output_activation: grouped_softmax
    label_groups:
      BOABIS: [BOABIS1, BOABIS2, BOABIS3]
      BOALEP: [BOALEP1, BOALEP2]
      DENMIN: [DENMIN1, DENMIN2, DENMIN3]

output:
  exclude_labels: [BOABIS_none, BOALEP_none, DENMIN_none]
```

**Output layout.** Each group contributes its members plus a synthetic `<group>_none` column, so the head emits `sum(len(members) + 1)` values — 11 for the example above. `labels.txt` lists them in model order:

```
BOABIS1  BOABIS2  BOABIS3  BOABIS_none | BOALEP1  BOALEP2  BOALEP_none | DENMIN1  DENMIN2  DENMIN3  DENMIN_none
```

Each group sums to 1, so two calls of one species can never both exceed 0.5 — that is arithmetic, not a tuned threshold — while all three species may fire at once.

**Background labels.** Training labels not named in any group get no output column. Instead they supply the `none` target for *every* group, so a background clip teaches all groups "absent" at once. Nothing changes in the `dataset` section: labels are still discovered from the data, and a background folder added later is picked up automatically. The trainer prints the resulting split at startup.

**Dropping the `none` columns.** Listing them in `exclude_labels` gives a head that emits only the call columns. This is lossless: the per-group softmax runs *before* the gather, so the remaining values are still calibrated probabilities and `P(none)` is recoverable as `1 - sum(that group's remaining columns)`. Combining `exclude_labels` with `export_logits` is rejected, because that would gather raw logits and leave the consumer unable to reconstruct the per-group softmax.

**Group of one.** A single-member group is simply an independent binary detector — useful for a species with one known call, or to promote a background label to a real output.

**Reporting.** The training report evaluates each group as its own single-label problem and summarises with *mean group accuracy* and *all-groups accuracy* (every group simultaneously correct). Macro F1/precision/recall are omitted, as they are not defined across independent groups.

**Not compatible with** `focal_loss`, `label_smoothing`, or `upsampling_ratio`, all of which assume one-hot targets; the trainer raises rather than silently misbehaving.

### Pre-training data transforms (applied in order)

1. **Upsampling** — minority class copies are generated before any other transform.
2. **Mixup** — pairs of positive samples are blended with a random coefficient.
3. **Label smoothing** — positive targets are reduced by `alpha` and the mass redistributed to all classes.

---

## `output`

```yaml
output:
  output_path: ./custom_models
  model_name: my_classifier
  model_version: "1.0"

  output_type: head           # head | full | both
  output_format: onnx         # onnx | tflite | both

  exclude_labels: []          # labels to remove from the exported output
  export_dataset: false       # write chunked WAV files to output_dir/dataset/
  export_embeddings: true    # save embeddings alongside the model
  embeddings_format: sqlite   # sqlite | npy
  # embeddings_path: /path/to/save/embeddings   # default: output_dir
```

Every `output` key, with its default and its description, is in the [config reference](config-reference.md#output).

### Full model export

| Backbone format | `output_format: onnx` | `output_format: tflite` |
|---|---|---|
| `onnx` | Full ONNX model ✓ | Head-only TFLite ✓ |
| `tflite` | Head-only ONNX ✓ | Full TFLite model ✓ |
| `protobuf` | Head-only ONNX ✓ | Full TFLite model ✓ |

When exporting a full TFLite model from a TFLite backbone that has a classifier head (e.g. BirdNET tflite), the original head ops and their weight tensors are stripped from the merged model — producing the same file size as a purpose-built backbone-only model.
