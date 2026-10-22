# Configuration

All parameters are defined in a single YAML (or JSON) file. This page describes it
section by section, with worked examples; every individual key, with its default
and description, is listed in the [config reference](config-reference.md).

## `foundation_model`

### Registry shorthand (recommended)

Every known foundation model has a registry entry with a 16-bit hexadecimal ID. Setting `registry_id` loads all defaults in one line and downloads the model automatically from Hugging Face:

```yaml
foundation_model:
  registry_id: "0xbb00"   # BirdNET 2.4 ONNX, downloaded from Hugging Face
```

Any field added alongside `registry_id` overrides the registry default, so switching to a local copy requires only two additional lines:

```yaml
foundation_model:
  registry_id: "0xbb00"
  source: local
  path: /path/to/birdnet_backbone.onnx
```

`bioaccx registry` lists all available IDs, with their descriptions and source URLs.

### Fully explicit configuration

Every field can also be specified manually. This is equivalent to the `registry_id` shorthand and remains fully supported:

```yaml
foundation_model:
  name: birdnet               # display name (used in reports and output filenames)
  version: "2.4"              # display version (used in reports and output filenames)
  data_type: FP32             # weight precision; part of the registry key

  # Model format on disk
  format: onnx                # onnx | tflite | protobuf (TF SavedModel)

  # --- Local file ---
  source: local
  path: /path/to/model.onnx

  # --- OR from the Hugging Face Hub ---
  # source: huggingface
  # hf_repo: biodiversica/birdnet-headless
  # hf_filename: birdnet_headless.onnx   # optional; defaults to model.onnx
  # hf_revision: main                    # branch, tag or commit (optional)

  # --- OR from Kaggle ---
  # source: kaggle
  # kaggle_handle: google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu
  # kaggle_filename: perch_v2_backbone.onnx   # file within the downloaded directory

  # Audio preprocessing
  sample_rate: 48000
  window_seconds: 3.0         # duration of each input window
  # window_samples: 144000    # alternative: exact sample count (takes priority)

  # Tensor names (must match the input and output node names of the model)
  input_name: INPUT
  output_name: embedding
  embedding_size: 1024        # dimensionality of the embedding vector

  # ONNX-specific
  # onnx_providers: [CUDAExecutionProvider, CPUExecutionProvider]
  # onnx_batch_size: 1        # values above 1 enable GPU batching (ONNX only)

  # TFLite only: offset from the declared output tensor index to the
  # embedding tensor. Use -1 for models such as the BirdNET TFLite model, in
  # which the classifier head is the first declared output and the embedding
  # is one tensor position before it.
  # tflite_output_tensor_offset: -1
```

Every `foundation_model` key, with its default and description, is listed in the [config reference](config-reference.md#foundation_model).

### TFLite backbones

`0xbb02` (BirdNET 2.4) and `0xbb12` (Perch v2) are TFLite backbones from which the classifier head and the auxiliary outputs of Perch have been removed, so the embedding is the only output of the model and no offset is needed:

```yaml
foundation_model:
  registry_id: "0xbb12"
```

Their embeddings are bit-identical to those of the original full TFLite models and match the ONNX backbones (`0xbb00`, `0xbb10`) to within float32 rounding, so classifiers trained on either format are equivalent. A TFLite backbone should be chosen when the deliverable must be a full TFLite model, since `output_type: full` can merge a TFLite head only with a TFLite backbone. The merged export is about 25 MB for BirdNET and about 43 MB for Perch.

### Using a full BirdNET TFLite model

A BirdNET TFLite file that still contains its classifier head (such as `BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`) outputs the scores of 6522 species, and the embedding is located one tensor position before that output. Set `path` to the file and set the offset:

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

When a full TFLite model is exported with this backbone, the operations and weight tensors of the original classifier head are removed from the merged output: only the backbone computation up to the embedding is retained, followed by the new classifier head.

An offset is rejected for a model whose only output is already the embedding. On such a backbone, it would select a different tensor, which may have the same width and would therefore not cause an error, but would produce incorrect embeddings.

### TFLite graph trimming

Models that include a classifier head or additional outputs (full models from other sources, as opposed to the registered backbones) compute all of them for every window, even though only the embedding is used. When `tflite_trim_to_embedding` is enabled (the default), the graph is trimmed to the embedding tensor once at load time, and every subsequent window runs the smaller graph. The following timings were measured with the original full models:

| Model | As shipped | Trimmed | |
|---|---|---|---|
| Perch v2 (`perch_v2.tflite`) | 1.60 s/window | 0.197 s/window | 8.2× |
| BirdNET 2.4 (`BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`) | 0.060 s/window | 0.041 s/window | 1.5× |

The embeddings are bit-identical in both cases. The cost is a single trimming step at load time (about 13 s for Perch and under 1 s for BirdNET), cached within the process so that the worker threads created by `embedding_workers` share it. Trimming also removes the need for `experimental_preserve_all_tensors` on models that use an offset, such as the full BirdNET model, which would otherwise keep every intermediate tensor in memory. A backbone with a single output has nothing to trim and is used unchanged. If trimming fails or does not produce a single output of the expected `embedding_size`, bioaccx issues a warning and uses the model unchanged; `tflite_trim_to_embedding: false` disables trimming entirely.

---

## `dataset`

```yaml
dataset:
  data_dir: /path/to/audio_dataset

  # How labels are organised in data_dir
  label_mode: subfolders      # subfolders | file_per_label | table

  # Only for label_mode: table
  # table_file: /path/to/annotations.csv
  # filename_col: filename
  # label_col: label
  # start_col: start_time
  # end_col: end_time
  # split_col: split          # optional; values: train | test

  audio_extensions: [wav, flac, mp3, ogg]
  overlap: 0.0                # window overlap for file_per_label and table (0.0–1.0)
  embedding_workers: 4        # parallel workers for embedding extraction
  embeddings_cache_path: null # pre-computed embeddings cache path (optional)
  embeddings_format: sqlite   # sqlite | npy
  test_ratio: 0.2             # fraction of the data used for the test set
  random_seed: 42

  # Optional: mix each training sample with background noise at several SNRs
  # augmentation:
  #   augmentation_dir: /path/to/noise_files
  #   snr_levels: [0, 10, 20]   # dB (power-based); one copy per level per noise file
  #   keep_original: true        # also keep the clean sample
  #   augment_test: false        # apply the same expansion to the test set
```

Every `dataset` key, with its default and description, is listed in the [config reference](config-reference.md#dataset).

---

## `training`

```yaml
training:
  classifier: keras           # keras | sklearn | both

  keras:
    hidden_units: 256         # hidden layer size; 0 means a single linear layer
    dropout: 0.25
    epochs: 50
    batch_size: 32
    learning_rate: 0.0001
    output_activation: null   # null (softmax-trained, exports logits) | sigmoid | softmax | grouped_softmax
    export_logits: false      # train with the activation, export without it
    normalize_embeddings: true

    # Optional: grouped softmax, exclusive within a group and independent across
    # groups (for example, several call types per species). See "Grouped softmax" below.
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

Every `training` key, with its default and description, is listed in the [config reference](config-reference.md#training).

### Keras training details

- **Metrics:** AUPRC (area under the precision-recall curve) and AUROC (area under the ROC curve) are computed on both the training and the validation sets at every epoch, together with accuracy.
- **Learning-rate schedule:** a linear warm-up for the first `max(3, epochs // 10)` epochs, followed by cosine decay to 10% of the peak learning rate.
- **Early stopping:** monitors `val_loss` with a patience of `max(5, epochs // 10)` epochs and restores the best weights.
- **Output activation:**
  - `null` (default): trained as with `softmax` (the classes compete), but raw logits are exported, so a softmax over all classes must be applied at inference time. Cannot be combined with `exclude_labels`; see [What the exported model outputs](outputs.md#what-the-exported-model-outputs).
  - `sigmoid`: a binary probability per class; suitable for multi-label problems.
  - `softmax`: normalised class probabilities; appropriate when the model should output probabilities directly.
  - `grouped_softmax`: a softmax within each group of `label_groups`, with the groups independent of each other; see below.
- **`normalize_embeddings`:** a `Normalization` layer is fitted on the training embeddings and included in the exported model, before any Dense layer. Set it to `false` to match the BirdNET-Analyzer architecture (backbone → fully connected layer → logits).

### Grouped softmax

A flat `softmax` makes *all* classes compete (their scores sum to 1, so two classes can never both score highly), whereas `sigmoid` makes them all independent (nothing prevents two mutually exclusive classes from being predicted together). `grouped_softmax` lies between the two: a softmax is applied **within each group**, and the groups are independent of one another.

The typical use case is several call types per species: one species emits one call at a time, but two species can easily overlap in the same window:

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

**Output layout.** Each group contributes its members and a synthetic `<group>_none` column, so the head emits `sum(len(members) + 1)` values, which is 11 in the example above. `labels.txt` lists them in the order of the model:

```
BOABIS1  BOABIS2  BOABIS3  BOABIS_none | BOALEP1  BOALEP2  BOALEP_none | DENMIN1  DENMIN2  DENMIN3  DENMIN_none
```

Each group sums to 1, so two calls of one species can never both exceed 0.5 (a mathematical property, not a tuned threshold), while all three species may be predicted at the same time.

**Background labels.** Training labels that are not named in any group receive no output column. Instead, they supply the `none` target for *every* group, so that a background clip trains all groups on "absent" at the same time. The `dataset` section is unaffected: labels are still discovered from the data, and a background folder added later is included automatically. The trainer prints the resulting assignment at startup.

**Removing the `none` columns.** Listing them in `exclude_labels` produces a head that emits only the call columns. No information is lost: the per-group softmax is applied *before* the gather, so the remaining values are still calibrated probabilities, and `P(none)` can be recovered as `1 - sum(remaining columns of the group)`. Combining `exclude_labels` with `export_logits` is rejected, because the gather would then select raw logits, and the consumer could not reconstruct the per-group softmax.

**Single-member groups.** A group with one member is an independent binary detector, which is useful for a species with a single known call or for turning a background label into an actual output.

**Reporting.** The training report evaluates each group as a separate single-label problem and summarises the results with the *mean group accuracy* and the *all-groups accuracy* (all groups correct simultaneously). Macro F1, precision and recall are omitted, as they are not defined across independent groups.

**Not compatible with** `focal_loss`, `label_smoothing` or `upsampling_ratio`, all of which assume one-hot targets; the trainer raises an error rather than producing incorrect results.

### Audio mixup (multi-label sigmoid heads)

Datasets built from separated sounds contain one class per window, so a sigmoid head never encounters two simultaneous calls during training, although overlaps are common in field recordings. `training.audio_mixup` synthesises such overlaps: it sums the audio of training windows from different classes, embeds the mixture with the foundation model and trains on the union of their labels.

```yaml
training:
  keras:
    output_activation: sigmoid     # required: a mix carries several labels
  audio_mixup:
    labels: []                     # only these labels are mixed (empty = all)
    ratio: 0.5                     # mixes = ratio × mixable train windows (or n_mixes: 2000)
    max_sources: 2                 # 3 allows three-source mixes, in the proportion set by p_three_sources
    snr_db: [-6, 6]                # added source level relative to the first
    pairing: balanced              # select classes uniformly, then a clip (uniform: select clips)
    exclusive_groups:              # never mixed with each other
      BOABIS: [BOABIS1, BOABIS2, BOABIS3]
    background_labels: [ambiente, aves, insecta]
    background_target: drop        # drop | include
    test_mix_ratio: 0.0            # >0: a separate mixed test set, as a fraction of the test windows
```

**When mixing takes place.** Mixing takes place after the train/test split and before embedding, using training windows only, so that no test clip contributes to a training mix. It reuses the loading, resampling and windowing of ordinary samples, at the sample rate and window length of the foundation model. Each added source is scaled so that its power is `snr_db` (drawn uniformly from the range) above or below that of the first source, and the sum is scaled down if it would clip. A noise-augmented copy contributes the clean window from which it was cut, never its noise, and a window is counted once regardless of how many copies it has, so mixing behaves identically with `keep_original: false`. SSH sources and precomputed `.npy` windows are not used as sources.

**Selecting what is mixed.** `labels` restricts mixing to the listed labels. Windows of any other label are never used as a source but are still trained on as real samples, so the head continues to learn them from real data (the Training Set table of the report shows 0 mixes for them). A real multi-label window is used only if all its labels are listed, and background labels intended for mixing must also be listed. An empty list means every label.

**Pairing rules.** A mix never contains the same label twice or two labels of the same `exclusive_groups` group, contains at most one background label, and always contains at least one non-background source. With `background_target: drop`, a background source adds no label (a mix of a species and `ambiente` is labelled with the species only, like the real species clips, which also contain ambient sound); with `include`, it adds its own label.

**Targets and caching.** The target of a mix is the union of the labels of its sources. Its embedding is cached under a key derived from its sources, windows and levels, so a repeated run with the same seed (by default `dataset.random_seed`) reuses it. Every mix is listed in `_dataset_list.csv` (its sources in `mix_sources`, their levels in `mix_snr_db` and its labels in `labels`), and `export_dataset` writes the rendered mixes to `dataset/mixes/`, outside the label folders.

**Checking the mixes.** When the real windows have already been exported, `output.export_mixes: true` (with `export_dataset: false`) writes only the mixes, to `dataset/mixes/train/` (and to `dataset/mixes/test/` for `test_mix_ratio`). `bioaccx dataset` also applies this setting, so the mixes can be listened to before training: with the same configuration and seed, it draws exactly the mixes that a training run would draw. Each file is named after its mix key, which is the `filepath` of its row in `_dataset_list.csv`, where its labels, sources and levels are recorded. (`bioaccx dataset --no-split` draws from every window, so its mixes differ from those of a training run.)

**Evaluation.** The test set consists of real, unmixed audio. With `test_mix_ratio`, a separate set of mixes is built from test windows only (`test_mix_ratio` times the number of test windows that may be mixed, drawn with the same rules as the training mixes) and reported in a separate section (and in `_evaluation_mixed.csv`), never pooled with the real clips. A class that appears in no test mix (such as a dropped background label) keeps its row, so that false positives are shown, but is excluded from the macro averages of that section. The report begins with a *Training Set* table that gives, for each class, the real training windows, the mixes whose target contains it, the mixes into which it was summed, its mix share (the synthetic proportion of its positive training samples), its main mix partners and its real test windows, since the scores that follow depend on this composition. The same figures are written to both metadata files and shown in the GUI. The report states how many training samples are synthetic, the `_metadata.json` file of the model records the `audio_mixup` settings with the numbers of real and synthetic samples, and for a multi-label head the report gives the subset accuracy (all labels of a window correct) and the Hamming loss; the one-vs-rest table scores each class against every window that contains it.

**Overlapping annotations.** Independently of mixup, annotations of different classes that produce the same window (same file, start and end) are merged into one multi-label window. A sigmoid head trains on it with a multi-hot target; any other head, including the scikit-learn head, receives it once per label, as before. `export_dataset` copies such a window into each of its label folders, and reading the export back merges the copies again (see [append](dataset-modes.md)).

Audio mixup requires `output_activation: sigmoid` and raises an error for `softmax`, `null` or `grouped_softmax`, whose classes compete. With `classifier: both`, the scikit-learn head trains on the real windows only.

### Pre-training data transforms (applied in order)

1. **Upsampling:** copies of minority-class samples are generated before any other transform.
2. **Mixup:** pairs of positive samples are blended with a random coefficient.
3. **Label smoothing:** positive targets are reduced by `alpha`, and the difference is redistributed to all classes.

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
  export_embeddings: true     # save the embeddings alongside the model
  embeddings_format: sqlite   # sqlite | npy
  # embeddings_path: /path/to/save/embeddings   # default: output_dir
```

Every `output` key, with its default and description, is listed in the [config reference](config-reference.md#output).

### Full model export

| Backbone format | `output_format: onnx` | `output_format: tflite` |
|---|---|---|
| `onnx` | Full ONNX model ✓ | Head-only TFLite ✓ |
| `tflite` | Head-only ONNX ✓ | Full TFLite model ✓ |
| `protobuf` | Head-only ONNX ✓ | Full TFLite model ✓ |

When a full TFLite model is exported from a TFLite backbone that contains a classifier head (for example, the BirdNET TFLite model), the operations and weight tensors of the original head are removed from the merged model, which therefore has the same file size as a dedicated backbone-only model.
