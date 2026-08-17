# bioaccx

**BIOAcoustic Custom Classifier eXchange** — a Python CLI library for training and sharing custom bioacoustic classifiers on top of pre-trained foundation models such as [BirdNET](https://birdnet.cornell.edu/) and [Perch](https://www.kaggle.com/models/google/bird-vocalization-classifier).

The core idea is a clean separation between the **backbone** and the **classifier head**:

- The backbone (foundation model) is a large, general-purpose audio encoder shared by the community.
- The head is a small, task-specific classifier trained on your labeled data — just a few kilobytes.

This separation enables two complementary workflows:

**Share lightweight heads.** Because the head is tiny compared to the backbone, you can publish and distribute your custom classifier without bundling the full model. Anyone who already has the same backbone can load your head and run inference immediately.

**Deploy as a single file.** When you need a self-contained model — for edge devices, cloud APIs, or third-party tools — bioaccx can merge the backbone and head into one ONNX or TFLite file with a single audio input and class-score output.

**Skip re-embedding when you already have embeddings.** If you or your team have already run the backbone and saved the resulting embedding vectors, you can point bioaccx at that cache and train or evaluate the head directly — no audio processing, no GPU time, no waiting.

**Build datasets from multiple sources in one config.** bioaccx can assemble training data from local recordings, [iNaturalist](https://www.inaturalist.org/) observations, [Xeno-canto](https://xeno-canto.org/) recordings, and [Arbimon](https://arbimon.org/) projects — all mixed in a single table or combined with a local directory. Remote audio is downloaded and cached automatically, so subsequent runs skip the network entirely.

bioaccx handles the full pipeline from raw audio to exported model, driven by a single config file.

---

## Contents

- [How it works](#how-it-works)
- [Installation](#installation)
  - [GPU acceleration](#gpu-acceleration-onnx-runtime--cuda)
- [Quick start](#quick-start)
- [Configuration reference](#configuration-reference)
  - [`foundation_model`](#foundation_model)
  - [`dataset`](#dataset)
  - [`training`](#training)
  - [`output`](#output)
- [Dataset modes](#dataset-modes)
  - [`subfolders`](#subfolders)
  - [`file_per_label`](#file_per_label)
  - [`table`](#table)
- [Remote sound sources](#remote-sound-sources)
  - [iNaturalist and Xeno-canto](#inaturalist-and-xeno-canto)
  - [Arbimon](#arbimon)
- [Appending to an existing dataset](#appending-to-an-existing-dataset)
- [Combining multiple sources in one run](#combining-multiple-sources-in-one-run)
- [Augmentation and windowing](#augmentation-and-windowing)
  - [Noise augmentation](#noise-augmentation)
  - [Random sample shift](#random-sample-shift)
  - [Windowing and overlap](#windowing-and-overlap)
- [Embedding cache](#embedding-cache)
- [Excluding labels from the exported model](#excluding-labels-from-the-exported-model)
- [Output directory structure](#output-directory-structure)
- [Merging a pre-existing head into a full model](#merging-a-pre-existing-head-into-a-full-model)
- [Foundation model registry](#foundation-model-registry)
  - [Browsing the registry](#browsing-the-registry)
  - [Using a registry ID in config](#using-a-registry-id-in-config)
  - [Supported foundation models](#supported-foundation-models)
  - [Adding a new model](#adding-a-new-model)
- [Python API](#python-api)
- [CLI reference](#cli-reference)

---

## How it works

```
Audio files  →  Foundation model (backbone)  →  Embeddings
                                                      ↓
                                       Keras or sklearn classifier head
                                                      ↓
                             ┌────────────────────────────────────────┐
                             │  backbone + head merged  │  head only  │
                             │   (single-file deploy)   │ (shareable) │
                             └────────────────────────────────────────┘
```

1. **Foundation model** — a backbone version of a bioacoustic model extracts rich feature vectors from raw audio windows. Supported formats: ONNX, TFLite, TF SavedModel (protobuf).
2. **Classifier head** — a small Keras MLP or sklearn LogisticRegression is trained on top of those embeddings using your labeled data.
3. **Export** — choose between:
   - **Full model** (`output_type: full`): merges backbone + head into a single ONNX or TFLite graph. Drop-in inference with no external dependencies.
   - **Head only** (`output_type: head`): exports just the classifier. Tiny file, easy to share; requires the backbone at inference time.
   - **Both** (`output_type: both`): produces both variants.

---

## Installation

```bash
pip install bioaccx
```

**Optional extras** (install only what you need):

| Extra | When needed |
|---|---|
| `tensorflow-cpu` / `tensorflow` | Keras classifier, TFLite or protobuf foundation models |
| `tf2onnx` | Exporting Keras head to ONNX |
| `scikit-learn` + `skl2onnx` | sklearn classifier |
| `huggingface-hub` | Downloading foundation models from HuggingFace Hub |
| `kaggle` | Downloading foundation models from Kaggle |
| `bioaccx[umap]` (`umap-learn` + `matplotlib`) | UMAP projection + plot for `--embeddings` (when `umap.enabled: true`) |

### GPU acceleration (ONNX Runtime + CUDA)

To run the ONNX backbone on a GPU, set `onnx_providers` in the config:

```yaml
foundation_model:
  onnx_providers: [CUDAExecutionProvider, CPUExecutionProvider]
```

ONNX Runtime's CUDA provider requires several NVIDIA libraries that are **not** bundled with the `onnxruntime-gpu` package and must be installed separately.  The `apt` packages are the simplest route if you have sudo access:

```bash
sudo apt install libcurand-12 libcufft-12 libcudart-12
```

If you are working in a virtualenv without system-level access, install the pip equivalents into the same environment as bioaccx:

```bash
uv pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
# or: pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
```

These packages install the `.so` files under `site-packages/nvidia/*/lib/`, but ONNX Runtime loads them via `dlopen` before Python's import machinery runs, so they are invisible to the dynamic linker by default.  The fix is a `sitecustomize.py` file that preloads them at interpreter startup.  Create it at:

```
<venv>/lib/python3.x/site-packages/sitecustomize.py
```

with the following content:

```python
import ctypes, pathlib

_nvidia_base = pathlib.Path(__file__).parent / "nvidia"
for lib in _nvidia_base.glob("*/lib/lib*.so.*"):
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
```

This loads every nvidia `.so` into the process with `RTLD_GLOBAL` so that subsequent `dlopen` calls from ONNX Runtime can resolve them.  No changes to `LD_LIBRARY_PATH` or the shell environment are needed.

To verify the CUDA provider is active after setup:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## Quick start

```bash
# 0. Browse available foundation models and their registry IDs
bioaccx --registry

# 1. Copy and edit the example config
cp example_config.yaml my_config.yaml

# 2. Validate config without running training
bioaccx my_config.yaml --validate

# 3. Export the dataset as chunked WAV files (no model needed)
bioaccx my_config.yaml --dataset

# 4. Compute the embedding database + UMAP (no training)
bioaccx my_config.yaml --embeddings

# 5. Train and export
bioaccx my_config.yaml

# 6. Merge an existing classifier head with a backbone (without training)
bioaccx my_config.yaml --merge

# 7. Extract the head from a full custom BirdNET-Analyzer TFLite model (without training)
bioaccx my_config.yaml --extract_head
```

---

## Configuration reference

All parameters live in a single YAML (or JSON) file. Below is the full reference with defaults and descriptions.

### `foundation_model`

#### Registry shorthand (recommended)

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

#### Fully explicit config

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

| Parameter | Default | Description |
|---|---|---|
| `registry_id` | `null` | Hex registry ID (e.g. `"0xbb00"`). Fills all defaults; any co-specified field overrides the registry value |
| `name` | required | Model name used in reports and output filenames |
| `version` | `"unknown"` | Model version string |
| `data_type` | `"FP32"` | Weight precision (e.g. `FP32`, `INT8`) |
| `format` | `onnx` | File format: `onnx`, `tflite`, or `protobuf` |
| `source` | `local` | Where to load from: `local`, `huggingface`, or `kaggle` |
| `path` | `null` | Path to local model file or directory |
| `hf_repo` | `null` | HuggingFace repo ID, e.g. `biodiversica/BirdNET-onnx-backbone` |
| `hf_filename` | `null` | Filename within HF repo |
| `hf_revision` | `null` | Branch, tag, or commit hash |
| `kaggle_handle` | `null` | Kaggle model handle, e.g. `google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu` |
| `kaggle_filename` | `null` | Specific file within the downloaded Kaggle directory |
| `sample_rate` | `48000` | Expected audio sample rate in Hz |
| `window_seconds` | `null` | Input window duration in seconds |
| `window_samples` | `null` | Input window in samples (takes priority over `window_seconds`) |
| `input_name` | `"input"` | Name of the model's input tensor |
| `output_name` | `"embedding"` | Name of the model's output tensor |
| `embedding_size` | `1024` | Embedding vector dimensionality |
| `onnx_providers` | `null` | ONNX Runtime execution providers, e.g. `[CUDAExecutionProvider, CPUExecutionProvider]`. Defaults to ORT's own priority when `null` |
| `onnx_batch_size` | `1` | Number of audio windows per inference call. Values > 1 enable GPU batch mode (ONNX only) |
| `tflite_output_tensor_offset` | `0` | Offset added to the TFLite model's declared output tensor index to reach the embedding. Use `-1` for full BirdNET tflite models (see below) |
| `tflite_trim_to_embedding` | `true` | Trim a TFLite graph down to the embedding tensor once at load time when it computes more than that (bundled classifier head, auxiliary outputs). Set `false` to run the model exactly as shipped |

#### Using the full BirdNET TFLite model

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

#### Using the full Perch v2 TFLite model

`0xbb12` is the Perch v2 TFLite. Unlike BirdNET's, its embedding is already the **first** of its four outputs (embedding, spatial_embedding, spectrogram, and a 14795-class label head), so no offset is needed:

```yaml
foundation_model:
  registry_id: "0xbb12"
```

Its embeddings match the `0xbb10` ONNX backbone to within 5e-7, so a classifier trained on either is equivalent. Choose `0xbb12` when the deliverable has to be a TFLite full model, since `output_type: full` can only merge a TFLite head with a TFLite backbone. The merged export is ~43 MB — the 14795-class head and the auxiliary branches are dropped by the trim.

#### TFLite graph trimming

Models that bundle a classifier head or extra outputs compute all of it on every window, even though only the embedding is used. When `tflite_trim_to_embedding` is enabled (the default), the graph is trimmed to the embedding tensor once at load time and every later window runs the smaller graph. Measured on this machine:

| Model | As shipped | Trimmed | |
|---|---|---|---|
| Perch v2 (`0xbb12`) | 1.60 s/window | 0.197 s/window | 8.2× |
| BirdNET 2.4 (`0xbb02`) | 0.060 s/window | 0.041 s/window | 1.5× |

Embeddings are bit-identical either way. The cost is a one-off trim at load (~13 s for Perch, <1 s for BirdNET), cached in-process so the worker threads created by `embedding_workers` share it. Trimming also removes the need for `experimental_preserve_all_tensors` on offset models like `0xbb02`, which otherwise keeps every intermediate tensor in memory. If a trim fails or does not yield a single output of the expected `embedding_size`, bioaccx warns and falls back to the model as shipped; set `tflite_trim_to_embedding: false` to skip it entirely.

---

### `dataset`

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

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | required | Root directory of the audio dataset |
| `sources` | `null` | List of per-source blocks combined in one run; each inherits the top-level `dataset` fields and overrides them (see [Combining multiple sources in one run](#combining-multiple-sources-in-one-run)) |
| `label_mode` | `subfolders` | Dataset layout mode (see below) |
| `table_file` | `null` | Path to CSV/TSV annotation table (table mode only) |
| `filename_col` | `filename` | Column name for audio file paths in table |
| `label_col` | `label` | Column name for class labels in table |
| `start_col` | `start_time` | Column name for segment start time (seconds) |
| `end_col` | `end_time` | Column name for segment end time (seconds) |
| `split_col` | `split` | Optional column with predefined `train`/`test` split |
| `audio_extensions` | `[wav,flac,mp3,ogg]` | Accepted audio file extensions (case-insensitive) |
| `overlap` | `0.0` | Fractional overlap between consecutive windows (0.0–1.0) |
| `embedding_workers` | `4` | Parallel threads for embedding extraction (capped to CPU count) |
| `embeddings_cache_path` | `null` | Path to a pre-computed embeddings cache (`.db` for SQLite, directory for `.npy`) |
| `embeddings_format` | `sqlite` | Storage format for exported/cached embeddings: `sqlite` (single `.db` file) or `npy` (one file per sample) |
| `append_dataset_path` | `null` | Path to an existing exported dataset to append new samples to |
| `ext_table_file` | `null` | CSV/TSV of remote audio sources (iNaturalist, Xeno-canto, Arbimon, or local mixed) |
| `obs_id_col` | `observation_id` | Column name for iNaturalist observation IDs |
| `sound_index_col` | `sound_index` | Column name for iNaturalist sound index (0-based) |
| `xc_id_col` | `xc_id` | Column name for Xeno-canto recording IDs |
| `ext_cache_dir` | `~/.cache/bioaccx/ext` | Local cache for all downloaded remote audio and metadata |
| `xc_api_key` | `null` | Xeno-canto API v3 key — enables scientific name lookup for XC rows; audio downloads without it |
| `arbimon_credentials_path` | `null` | Path to the rfcx persisted credentials file; required for any Arbimon row |
| `arbimon_stream_id_col` | `stream_id` | Column name for Arbimon stream/site IDs |
| `arbimon_date_col` | `date` | Column name for the local recording date |
| `arbimon_time_col` | `time` | Column name for the local recording start time |
| `arbimon_utc_offset_col` | `utc_offset` | Column name for the UTC offset |
| `test_ratio` | `0.2` | Proportion of data held out for the test set |
| `random_seed` | `42` | Random seed for reproducible splits and noise offsets |
| `random_sample_shift` | `false` | Randomise signal placement within padded windows for short samples |
| `augmentation.augmentation_dir` | `null` | Directory containing noise WAV files for augmentation. Optional when `augmentation_labels` is set |
| `augmentation.augmentation_labels` | `null` | Labels of the dataset being created whose audio is used as additional noise sources (combined with `augmentation_dir`). These labels are not augmented themselves but remain trainable classes |
| `augmentation.snr_levels` | `null` | List of SNR values in dB; one augmented copy is produced per noise file per level |
| `augmentation.keep_original` | `true` | Also include the clean (unaugmented) sample alongside augmented copies |
| `augmentation.augment_test` | `false` | Apply the same augmentation to the test set |
| `augmentation.concatenate_augmentation_dir` | `false` | Concatenate all files in `augmentation_dir` into a single noise source. Sibling Audacity `.txt` label files are used to select segments; files without a label file are included in full |
| `augmentation.random_augmentation_dir` | `false` | Shuffle noise files once (using `random_seed`) and assign one file per *(sample, SNR)* pair round-robin. Produces one noise condition per sample (like `concatenate_augmentation_dir`) while drawing from individual files |

---

### `training`

```yaml
training:
  classifier: keras           # keras | sklearn | both

  keras:
    hidden_units: 256         # hidden layer size; 0 = single linear layer
    dropout: 0.25
    epochs: 50
    batch_size: 32
    learning_rate: 0.0001
    output_activation: null   # null (logits) | sigmoid | softmax
    export_logits: false      # train with the activation, export without it
    normalize_embeddings: true

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

| Parameter | Default | Description |
|---|---|---|
| `classifier` | `keras` | Which classifier(s) to train: `keras`, `sklearn`, or `both` |
| `keras.hidden_units` | `256` | Units in the hidden Dense layer; `0` = no hidden layer (single linear classifier) |
| `keras.dropout` | `0.25` | Dropout rate applied before each Dense layer |
| `keras.epochs` | `50` | Maximum training epochs (early stopping may halt earlier) |
| `keras.batch_size` | `32` | Mini-batch size |
| `keras.learning_rate` | `0.0001` | Adam optimizer peak learning rate (with cosine decay + linear warmup) |
| `keras.output_activation` | `null` | Output activation: `null` (logits), `sigmoid`, or `softmax` |
| `keras.export_logits` | `false` | Strip the activation layer before export, so the head trains with `output_activation` but emits raw logits (BirdNET-Analyzer's `classifier.pop()`). No-op when `output_activation` is `null` |
| `keras.normalize_embeddings` | `true` | Apply Z-score normalization (mean/std adapted on training embeddings) as the first layer |
| `keras.focal_loss` | `false` | Replace cross-entropy with focal loss — helps with class imbalance |
| `keras.focal_loss_gamma` | `2.0` | Focal loss focusing parameter γ |
| `keras.focal_loss_alpha` | `0.25` | Focal loss class balance parameter α |
| `keras.label_smoothing` | `false` | Smooth one-hot targets before training — reduces overconfidence |
| `keras.label_smoothing_alpha` | `0.1` | Amount subtracted from positive labels and redistributed to negatives |
| `keras.mixup` | `false` | Apply mixup augmentation to training embeddings |
| `keras.mixup_ratio` | `0.25` | Fraction of positive training samples to mix |
| `keras.mixup_alpha` | `0.2` | Beta distribution parameter for the mixing coefficient |
| `keras.upsampling_ratio` | `0.0` | Upsample minority classes to at least this fraction of the majority class count (`0` = disabled) |
| `keras.upsampling_mode` | `repeat` | Upsampling strategy: `repeat` (random duplication), `mean` (pairwise mean), `linear` (random linear interpolation), `smote` (k-NN interpolation) |
| `sklearn.C` | `1.0` | Regularisation strength (LogisticRegression) |
| `sklearn.max_iter` | `2000` | Maximum iterations for the solver |
| `sklearn.solver` | `lbfgs` | Solver algorithm |

#### Keras training details

- **Metrics**: AUPRC (area under precision-recall curve) and AUROC (area under ROC curve) are computed on both the training and validation sets every epoch, alongside accuracy.
- **LR schedule**: linear warmup for the first `max(3, epochs // 10)` epochs, then cosine decay to 10% of the peak LR.
- **Early stopping**: monitors `val_loss` with patience `max(5, epochs // 10)` and restores the best weights.
- **Output activation notes**:
  - `null` (default): raw logits; use `softmax` at inference time for class probabilities.
  - `sigmoid`: per-class binary probability; suitable for multi-label problems.
  - `softmax`: normalised class probabilities; use when the model should output probabilities directly.
- **`normalize_embeddings`**: a `Normalization` layer is fitted on the training embeddings and baked into the exported model. This is applied before any Dense layer. Set to `false` to match the BirdNET-Analyzer architecture (backbone → FC → logits directly).

#### Pre-training data transforms (applied in order)

1. **Upsampling** — minority class copies are generated before any other transform.
2. **Mixup** — pairs of positive samples are blended with a random coefficient.
3. **Label smoothing** — positive targets are reduced by `alpha` and the mass redistributed to all classes.

---

### `output`

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

| Parameter | Default | Description |
|---|---|---|
| `output_path` | `./outputs` | Parent directory for all output |
| `model_name` | `custom_classifier` | Used in filenames and the output subdirectory |
| `model_version` | `"1.0"` | Version string used in filenames |
| `output_type` | `head` | `head` = classifier only; `full` = foundation + classifier merged; `both` = save both |
| `output_format` | `onnx` | `onnx`, `tflite`, or `both` |
| `data_types` | `null` | Output precision(s) — any subset of `[FP32, FP16, INT8]`. `null` = use the foundation model's `data_type`. See [Output precision](#output-precision-fp32--fp16--int8) |
| `exclude_labels` | `[]` | Labels to omit from the exported model output (still used during training) |
| `export_dataset` | `false` | Export chunked audio as WAV files in label subfolders |
| `export_embeddings` | `false` | Save extracted embeddings |
| `embeddings_format` | `sqlite` | Embedding storage format: `sqlite` (single `.db` file named by registry ID) or `npy` (one file per sample) |
| `embeddings_path` | `null` | Custom path for exported embeddings |
| `embeddings_overwrite` | `false` | In `--embeddings` mode, recompute and overwrite an existing store instead of reusing it (only UMAP is rebuilt on reuse) |
| `head_path` | `null` | Path to an existing classifier head (ONNX or TFLite) for use with `--merge` |
| `extract_from` | `null` | Path to a full BirdNET-Analyzer `.tflite` model for use with `--extract_head` |
| `labels_file` | `null` | Optional class-label file for `--extract_head`; defaults to the sibling `<model>_Labels.txt` |

#### Full model export

| Backbone format | `output_format: onnx` | `output_format: tflite` |
|---|---|---|
| `onnx` | Full ONNX model ✓ | Head-only TFLite ✓ |
| `tflite` | Head-only ONNX ✓ | Full TFLite model ✓ |
| `protobuf` | Head-only ONNX ✓ | Full TFLite model ✓ |

When exporting a full TFLite model from a TFLite backbone that has a classifier head (e.g. BirdNET tflite), the original head ops and their weight tensors are stripped from the merged model — producing the same file size as a purpose-built backbone-only model.

---

## Dataset modes

### `subfolders`

The simplest layout: one subdirectory per class.

```
data_dir/
  crow/
    recording1.wav
    recording2.flac
  robin/
    song_a.wav
  background/
    noise1.wav
```

If the top-level contains `train/` and `test/` subdirectories, the predefined split is used:

```
data_dir/
  train/
    crow/
      recording1.wav
    robin/
      song_a.wav
  test/
    crow/
      recording2.wav
```

### `file_per_label`

Each audio file has a paired annotation file with the same stem:

```
data_dir/
  soundscape_01.wav
  soundscape_01.txt
  soundscape_02.wav
  soundscape_02.txt
```

The `.txt` file contains tab- or comma-separated rows with `start_time`, `end_time`, and `label` (no header required):

```
0.0    3.0    crow
5.5    8.5    robin
12.0   15.0   crow
```

Audio files without a matching annotation file are silently skipped.

### `table`

A single CSV or TSV file maps audio segments to labels:

```yaml
dataset:
  label_mode: table
  table_file: /path/to/annotations.csv
  filename_col: filename       # default
  label_col: label             # default
  start_col: start_time        # default
  end_col: end_time            # default
  split_col: split             # optional; values: train / test
```

Example CSV:

```csv
filename,label,start_time,end_time,split
soundscapes/rec01.wav,crow,0.0,3.0,train
soundscapes/rec01.wav,robin,5.5,8.5,train
soundscapes/rec02.wav,crow,1.0,4.0,test
```

---

## Remote sound sources

Use `ext_table_file` to include audio from [iNaturalist](https://www.inaturalist.org/), [Xeno-canto](https://xeno-canto.org/), and/or [Arbimon](https://arbimon.org/) alongside (or instead of) local files.

```yaml
dataset:
  data_dir: /path/to/local_recordings   # optional; omit for remote-only
  ext_table_file: /path/to/observations.csv
```

**Row dispatch** — each row is identified by the first non-empty ID column:

| Priority | Column | Source |
|---|---|---|
| 1 | `observation_id` | iNaturalist observation |
| 2 | `xc_id` | Xeno-canto recording (`12345` or `XC12345`) |
| 3 | `stream_id` | Arbimon 1-minute recording (also requires `date`, `time`, `utc_offset`) |
| 4 | `filename` | Local audio file |

**All columns:**

| Column | Notes |
|---|---|
| `observation_id` | iNaturalist observation ID |
| `sound_index` | 0-based sound index within the observation; defaults to `0` (iNaturalist only) |
| `xc_id` | Xeno-canto recording ID — numeric or with `XC` prefix |
| `stream_id` | Arbimon stream/site ID (requires `date`, `time`, `utc_offset` columns) |
| `date` | Local recording date for Arbimon rows: `YYYY-MM-DD` |
| `time` | Local recording start time for Arbimon rows: `HH:MM` or `HH:MM:SS` |
| `utc_offset` | UTC offset in hours for Arbimon rows (e.g. `-3`, `+5.5`, `UTC-3`, `UTC+5:30`) |
| `filename` | Path to a local audio file (absolute or relative to `data_dir`) |
| `label` | Falls back to the taxon name / stream ID for remote rows if empty |
| `start_time` / `end_time` | Seconds; same partial-time rules as local table mode |
| `split` | `train` or `test`; auto-split if empty |

**Mixed table** — all source types can coexist in one file:

```csv
filename,observation_id,sound_index,xc_id,stream_id,date,time,utc_offset,label,start_time,end_time
/data/rec.wav,,,,,,,,cicada,0.0,3.0
,12345678,0,,,,,,,1.0,6.0
,,,98765,,,,, Turdus merula,,,train
,,,,abc123,2023-07-14,06:00,-3,Guira guira,10.0,30.0
```

**Caching** — audio files and metadata are cached locally on first download; subsequent runs skip the network entirely.

```yaml
dataset:
  ext_cache_dir: /path/to/cache   # default: ~/.cache/bioaccx/ext
```

### iNaturalist and Xeno-canto

Rows are identified by `observation_id` (iNaturalist) or `xc_id` (Xeno-canto). Labels fall back to the taxon scientific name fetched from the respective API when the `label` column is empty.

**Xeno-canto API key** — Xeno-canto uses API v3, which requires a personal key for metadata queries (scientific name lookup). Without a key, audio is still downloaded directly but the label falls back to `"xc_<id>"` unless you set it explicitly in the table. Register at [xeno-canto.org/explore/api](https://xeno-canto.org/explore/api).

```yaml
dataset:
  xc_api_key: YOUR_KEY_HERE   # optional; enables scientific name lookup for XC rows
```

### Arbimon

Each Arbimon row identifies a specific 1-minute recording by its stream (site) ID and local timestamp. The label falls back to `stream_id` when the `label` column is empty.

**Required columns:**

| Column | Default name | Description |
|---|---|---|
| `stream_id` | `stream_id` | Arbimon recording site / stream ID |
| `date` | `date` | Local recording date, ISO format: `YYYY-MM-DD` |
| `time` | `time` | Local recording start time: `HH:MM` or `HH:MM:SS` |
| `utc_offset` | `utc_offset` | UTC offset of the local time in hours (e.g. `-3`, `+5.5`, `UTC-3`, `UTC+5:30`) |

**Config:**

```yaml
dataset:
  ext_table_file: /path/to/observations.csv
  arbimon_credentials_path: /path/to/.rfcx_credentials   # required
  # arbimon_stream_id_col: stream_id   # column name overrides (optional)
  # arbimon_date_col: date
  # arbimon_time_col: time
  # arbimon_utc_offset_col: utc_offset
```

**Authentication** — Arbimon uses the rfcx SDK for downloads. Authenticate once to create a credentials file:

```python
import rfcx
client = rfcx.Client()
client.authenticate(persisted_credentials_path="/path/to/.rfcx_credentials")
```

This opens a browser URL for device authorisation and saves a token to disk. All subsequent runs load the token from that file without any user interaction.

**Installing the rfcx SDK** — the SDK is not on PyPI; install it directly from the GitHub release:

```bash
pip install https://github.com/rfcx/rfcx-sdk-python/releases/download/0.3.1/rfcx-0.3.1-py3-none-any.whl
```

**Caching** — downloaded audio is stored under `<ext_cache_dir>/arbimon/<stream_id>/`. A lightweight sentinel file is written for each downloaded minute so that repeated runs skip the network entirely and locate the audio file without re-scanning the directory.

---

## Appending to an existing dataset

Use `append_dataset_path` to grow an existing dataset incrementally without reprocessing samples that are already there.

```yaml
dataset:
  data_dir: /path/to/new_recordings
  label_mode: file_per_label
  append_dataset_path: /path/to/existing_exported_dataset
```

**How it works:**

1. All `data_dir` paths are scanned and chunked into `AudioSample` objects as usual.
2. The existing dataset at `append_dataset_path` is loaded (must be in the `subfolders` layout with `train/` and `test/` subdirectories — the format produced by `export_dataset: true` or `--dataset`).
3. Each new sample is compared against the existing dataset by matching its would-be export filename (`{stem}_{start:.3f}_{end:.3f}`). Duplicates are dropped.
4. The merged set (existing + new unique samples) is used for the rest of the pipeline.

**Split assignment:** existing samples keep their original `train`/`test` assignments. New samples are stratified auto-split using `test_ratio` and `random_seed`.

**Typical workflow:**

```bash
# First run — train on initial data and export the dataset
bioaccx config_v1.yaml   # with export_dataset: true

# Later — add new recordings and retrain on the full merged set
bioaccx config_v2.yaml   # with append_dataset_path pointing to the first export
```

---

## Combining multiple sources in one run

A single `dataset` block describes one source with one set of options. When your training data is heterogeneous — for example, one directory laid out as `subfolders`, another as `file_per_label`, and a third you want to noise-augment but the others not — you can declare a list of **source blocks** under `dataset.sources` instead of chaining several `--dataset` runs together with `append_dataset_path`.

Each source is loaded, split, and augmented independently, then all sources are concatenated into the final train/test sets. Splitting per source preserves each source's class proportions in both subsets, and augmentation (or any per-source preprocessing) applies only to the sources that request it.

```yaml
dataset:
  # Run-level settings — defined once, shared by every source
  test_ratio: 0.2
  random_seed: 42
  embedding_workers: 8

  sources:
    - data_dir: /data/curated          # class subfolders, no augmentation
      label_mode: subfolders

    - data_dir: /data/soundscapes      # Audacity-style annotations
      label_mode: file_per_label

    - data_dir: /data/rare_species     # augment ONLY this source
      label_mode: subfolders
      augmentation:
        augmentation_dir: /data/noise
        snr_levels: [20, 10, 3]

    - data_dir: /remote/recordings     # a remote source over SSH
      label_mode: subfolders
      ssh_host: host.example
      ssh_user: me
```

### Inheritance and overrides

Each source **inherits** the top-level `dataset` fields and overrides them with its own. So you can set common options (e.g. an `overlap` or a `filter`) once at the top level and let every source pick them up, while varying `data_dir`, `label_mode`, augmentation, etc. per source.

To **opt a source out** of an inherited augmentation, set `augmentation: null` on that source:

```yaml
dataset:
  augmentation:                  # applied to every source by default…
    augmentation_dir: /data/noise
    snr_levels: [10, 20]
  sources:
    - data_dir: /data/a
    - data_dir: /data/b
      augmentation: null         # …but not this one
```

### Run-level vs. per-source fields

A few fields describe the **whole run** and are taken only from the top-level `dataset` block — setting them inside a source has no effect, keeping the split, seeding, append target, and credentials unambiguous across the run:

| Run-level (top-level only) | Per-source (overridable) |
|---|---|
| `test_ratio`, `random_seed` | `data_dir`, `label_mode`, `table_file` / `ext_table_file` |
| `append_dataset_path` | `overlap`, `min_anchor_fraction` |
| `embedding_workers`, `embeddings_cache_path` | `filter`, `filter_freq`, `filter_order`, `speed` |
| `ext_cache_dir`, `audio_extensions` | `augmentation`, `random_sample_shift` |
| `xc_api_key`, `arbimon_credentials_path` | `ssh_host` / `ssh_user` / `ssh_port` / `ssh_key_path`, label/column names |

Because **SSH settings are per-source**, you can freely mix local and remote sources — even sources on different SSH hosts — in one run.

`append_dataset_path` stays run-level: the existing exported dataset is loaded once and merged in (keeping its predefined `train`/`test` split), while each source still de-duplicates its new samples against it. This works the same across the full pipeline, `--dataset`, and `--embeddings`.

> Configs without a `sources:` key behave exactly as before — this is a purely additive feature.

---

## Augmentation and windowing

### Noise augmentation

Use `dataset.augmentation` to expand the training set by mixing each sample with background noise at one or more signal-to-noise ratios.

```yaml
dataset:
  augmentation:
    augmentation_dir: /path/to/noise_files   # WAV files used as noise sources
    snr_levels: [0, 10, 20]                  # dB (power-based)
    keep_original: true                      # also keep the unaugmented sample
    augment_test: false                      # default: test set is not augmented
    concatenate_augmentation_dir: false      # see below
```

**How it works:**

- For every combination of *(training sample, noise file, SNR level)* one augmented copy is produced.
- A window-length chunk is extracted from the noise file at a **random start position** derived from a stable hash of `(random_seed, sample path, times, noise filename, SNR)` — results are fully reproducible across runs regardless of iteration order.
- If the noise file is shorter than the model's input window, the extracted chunk is zero-padded.
- Augmented samples are embedded by mixing signal and noise at inference time; no intermediate audio files are written unless `export_dataset: true` is also set.
- When `export_dataset: true`, augmented WAV files are written alongside clean ones with a `_noise_<stem>_t<offset>_snr<value>` suffix in the filename (`<offset>` is the noise start position in seconds, so multiple windows of the same noise file get distinct names).
- The dataset list CSV gains two extra columns — `noise_file` and `snr_db` — for every run that uses augmentation (empty for clean samples).
- Augmentation parameters are recorded in the `_metadata.json` output file under an `"augmentation"` key.

**Dataset expansion factor:**

```
total train samples = clean_train × (N_noise_files × N_snr_levels + keep_original)
```

For example, 100 clean train samples with 3 noise files, SNR levels `[0, 10, 20]`, and `keep_original: true` → 100 × (3×3 + 1) = **1000 train samples**.

By default, `augment_test: false` keeps the test set clean for unbiased evaluation. Set it to `true` when you specifically want to measure model robustness under noise conditions.

#### Concatenated noise source

Setting `concatenate_augmentation_dir: true` treats the entire contents of `augmentation_dir` as a single noise source instead of one noise file per augmented copy.

```yaml
dataset:
  augmentation:
    augmentation_dir: /path/to/noise_files
    snr_levels: [0, 10, 20]
    concatenate_augmentation_dir: true
```

- All audio files in `augmentation_dir` are scanned recursively.
- For each file, if a sibling Audacity label file (`.txt`) with the same stem exists, only the labeled segments are used — regardless of label text. This lets you mark clean background passages in long field recordings and exclude everything else.
- Files without a label file (or with an empty one) are included in full.
- All selected segments are concatenated into a single in-memory mono array at the model's sample rate. No intermediate audio file is exported.
- This concatenated array is used as the sole noise source, so the expansion factor becomes `clean_train × 1 × N_snr_levels` instead of `clean_train × N_files × N_snr_levels`.

#### Random noise selection

Setting `random_augmentation_dir: true` keeps a single noise condition per augmented sample (like `concatenate_augmentation_dir`) but draws the noise from individual files rather than a merged track. Noise files are shuffled once using `random_seed` and then assigned round-robin to each *(sample, SNR)* pair.

```yaml
dataset:
  augmentation:
    augmentation_dir: /path/to/noise_files
    snr_levels: [0, 10, 20]
    random_augmentation_dir: true
```

- The noise files in `augmentation_dir` are sorted and then shuffled with a single pass using `random_seed`.
- Each *(training sample, SNR level)* pair is assigned the next file in the shuffled list. When the list is exhausted it wraps around, so no file is repeated before all others have been used once.
- The noise start offset within the chosen file is still derived from a stable hash of `(random_seed, sample path, times, noise filename, SNR)` — results are fully reproducible across runs.
- The expansion factor is `clean_train × 1 × N_snr_levels` (same as the concatenated option), not `clean_train × N_files × N_snr_levels`.

**When to use each option:**

| Option | Expansion factor | Noise source |
|---|---|---|
| default | `× N_files × N_SNR` | one copy per noise file |
| `concatenate_augmentation_dir: true` | `× N_SNR` | all files merged into one track |
| `random_augmentation_dir: true` | `× N_SNR` | one randomly assigned file per sample |

Use `random_augmentation_dir` when you have many noise files and want to avoid the large dataset expansion of the default mode while still exposing the model to varied individual noise recordings rather than a single blended track.

#### Labels as noise sources

Instead of (or in addition to) pointing at an external `augmentation_dir`, you can nominate one or more **labels of the dataset being created** as noise sources via `augmentation_labels`:

```yaml
dataset:
  augmentation:
    augmentation_labels: [rain, wind]   # labels whose audio becomes noise
    augmentation_dir: /path/to/noise    # optional — combined with the labels
    snr_levels: [0, 10, 20]
```

- The audio of every sample carrying a listed label is added to the noise pool, **alongside** any files in `augmentation_dir` (the two are combined — set either or both).
- Those labels are **never augmented themselves** (they are treated like `skip_labels`) but they **remain trainable classes** in the dataset — their clean samples are kept as usual.
- Labels are resolved **across the whole dataset**: when using multiple `sources`, a noise label may live in a different source than the one whose `augmentation` block references it (e.g. a `noise` source folder feeding augmentation for a target-species source).
- The noise pool is drawn per split: train augmentation uses the train-side samples of those labels, so test audio never leaks into the training noise.
- For `table` / `file_per_label` sources, only the annotated segment of each noise-label sample is used as a noise source; for `subfolders` sources the whole file is used.
- All three modes apply: by default each noise-label sample is one noise source (per-file expansion); `random_augmentation_dir` and `concatenate_augmentation_dir` collapse the pool (including the label audio) to a single condition per sample exactly as with `augmentation_dir`.

This is convenient when your dataset already contains background/non-target classes (rain, wind, traffic, silence) that you want to reuse as realistic noise for the target classes without maintaining a separate noise directory.

### Random sample shift

Short samples (those whose audio duration is less than the foundation model window) are normally placed at the start of the padded window. Setting `random_sample_shift: true` randomises this placement:

```yaml
dataset:
  random_sample_shift: true   # default: false
```

**How it works:**

- Each short sample is assigned a random integer offset in `[0, window_samples − signal_samples]` that determines where the signal starts within the zero-padded window.
- The offset is derived from a stable hash of `(random_seed, sample path, start/end times, noise_path, SNR)` so results are **fully reproducible** across runs.
- When augmentation is also active, each augmented copy of a sample gets a **different offset** because the noise file and SNR are part of the hash key.
- The dataset list CSV gains a `signal_offset_samples` column when `random_sample_shift` is enabled (empty for full-window samples).
- The shift is applied both at embedding time and when `export_dataset: true` is set.

### Windowing and overlap

For `file_per_label` and `table` modes, each annotated segment is divided into fixed-length windows matching the foundation model's input size.

- **Long segments** → multiple windows stepped by `window × (1 − overlap)`
- **Short segments** → a single window zero-padded to the required length

With `overlap: 0.5` and a 3 s window on a 9 s segment:

```
Window 1: 0.0 – 3.0 s
Window 2: 1.5 – 4.5 s
Window 3: 3.0 – 6.0 s
Window 4: 4.5 – 7.5 s
Window 5: 6.0 – 9.0 s
```

---

## Embedding cache

Computing embeddings is the slowest step. Use `embeddings_cache_path` to store and reuse them across runs.

### SQLite cache (default)

```yaml
dataset:
  embeddings_cache_path: /path/to/cache/0xbb00_embeddings.db
  embeddings_format: sqlite
```

All embeddings for a run are stored in a single `.db` file. The filename conventionally encodes the foundation model registry ID (e.g. `0xbb00_embeddings.db`) — bioaccx warns and discards the cache if the ID in the filename doesn't match the configured backbone.

### NPY cache

```yaml
dataset:
  embeddings_cache_path: /path/to/cache_dir
  embeddings_format: npy
```

Each embedding is stored as an individual `.npy` file. Filenames encode the audio file stem, start time, and end time, so different chunking or overlap settings produce separate cache entries.

### Exporting embeddings

To save computed embeddings as a pipeline output (in addition to or instead of caching them):

```yaml
output:
  export_embeddings: true
  embeddings_format: sqlite   # sqlite | npy
  # embeddings_path: /custom/export/path   # default: output_dir
```

---

## Excluding labels from the exported model

Use `exclude_labels` to remove classes (e.g. background noise) from the final model output while still using them during training. This improves training stability without polluting the inference output.

```yaml
output:
  exclude_labels: [background, noise, unknown]
```

The excluded classes are present in the training data and the internal classifier, but the exported ONNX/TFLite model's output tensor only contains scores for the remaining classes. The `_labels.txt` file reflects the final output label order.

---

## Extracting a head from a BirdNET-Analyzer model

Custom models trained with [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) are distributed as a single TFLite file bundling the BirdNET backbone with a small classifier head. `--extract_head` recovers just that head and re-exports it as a lightweight head-only model — **without converting or running the backbone**:

```yaml
foundation_model:
  registry_id: "0xbb02"          # the backbone the head was trained on (defines embed_dim)
output:
  output_path: ./custom_models
  model_name: MyCustomModel
  model_version: "0.0"
  output_format: both            # onnx, tflite, or both
  extract_from: /path/to/MyCustomModel.tflite   # the full BirdNET-Analyzer model
  # labels_file: /path/to/labels.txt   # optional; defaults to the sibling *_Labels.txt
```

```bash
bioaccx my_config.yaml --extract_head
```

**How it works:**

- The classifier head is the trailing chain of `FULLY_CONNECTED` ops ending at the model output. The flatbuffer is read directly to recover each dense layer's weight (`[out, in]`), bias, and fused activation; the chain is walked back from the output until it reaches the layer whose input is the embedding (`embed_dim`), so backbone/frontend ops are never included. The backbone is never decoded.
- Both **single linear heads** (BirdNET-Analyzer *Hidden units = 0*) and **multi-layer (MLP) heads** are supported.
- **TFLite head — bit-exact.** When `output_format` includes `tflite`, the head is produced by **slicing the original flatbuffer**: the head operators and their weight buffers are copied verbatim into a new single-input TFLite model rooted at the embedding tensor, with its I/O tensors renamed to `embedding` → `scores` (labels only — values unchanged). It reproduces the source model's head output *exactly* (`0.0` difference, same precision — quantized heads included). It is written as `<stem>_head.tflite` (no precision suffix — it inherits the source precision).
- **ONNX head.** The sliced head is converted directly to ONNX with `tf2onnx` (a dense-only head converts cleanly — the full model can't, because the backbone uses ops like `RFFT2D` that ONNX lacks), then its I/O is renamed to `embedding` → `scores`. Output precisions (`data_types`) and `exclude_labels` apply. It reproduces the original logits to within float32 rounding (~`1e-6`) and can be re-`--merge`d with the matching ONNX backbone to rebuild a full single-file model.
- Class names are read from a sibling `<model>_Labels.txt` (BirdNET's `scientific_common` format → common name) or from an explicit `labels_file`. The final `_labels.txt` and a small `_extract_head.json` (source, backbone, classes, outputs) are written alongside the head(s).

**Supported heads:** plain dense chains (with `relu`/`relu6`/`tanh` activations and an optional trailing `sigmoid`/`softmax`). The bit-exact TFLite slice also handles quantized heads; the ONNX rebuild requires a float32 head. Heads with non-dense ops between the embedding and the output raise a clear error. `exclude_labels` and `data_types` precision conversion apply to the ONNX head only — the TFLite slice keeps all source classes at the source precision to stay bit-exact.

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

## Output directory structure

All outputs are written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/`.

The `foundation_model_id` is a 16-bit hex identifier resolved from the registry (see [Foundation model registry](#foundation-model-registry)).  Exported model files also carry a precision suffix (`_fp32`, `_fp16`, or `_int8`; see [Output precision](#output-precision-fp32--fp16--int8)).  For example, a classifier trained on BirdNET 2.4 FP32 ONNX (`0xbb00`) would produce:

```
custom_models/
  my_classifier_0xbb00_v1.0/
    my_classifier_0xbb00_v1.0_labels.txt              # output class names, one per line
    my_classifier_0xbb00_v1.0_metadata.json           # full metadata JSON
    my_classifier_0xbb00_v1.0_dataset_list.csv        # per-sample split/label summary
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
    dataset/                                        # exported chunked WAV files (optional)
      train/
        crow/
        robin/
      test/
        crow/
        robin/
```

### Per-class evaluation

The Keras report ends with a **Per-class Evaluation** table, also written separately as
`*_evaluation.csv` with the same columns and layout as BirdNET-Analyzer's `*_evaluation.csv`,
so the two can be compared or loaded by the same tooling:

| Column | Meaning |
|---|---|
| `Class` | Label name; the first row is `OVERALL (Macro-avg)` |
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
filtering via `exclude_labels` still applies. The setting is a no-op when
`output_activation` is `null`, since there is no activation to remove.

Note that the per-class evaluation is computed on the *trained* model, so its thresholds are
in probability space while the exported head emits logits — apply the activation to the head
output before comparing against them. The report prints a reminder to that effect when
`export_logits` is on, and `_metadata.json` records both `output_activation` and
`export_logits` under `keras_classifier` for inference code to read.

---

## Computing the embedding database + UMAP (no training)

Use `--embeddings` to run the foundation model over the dataset and build the **embedding database** without training or exporting a classifier. If the dataset has not been prepared yet, it is loaded and split first (the same loading, augmentation, and random-shift steps used by the full pipeline).

Embeddings are **always exported** in this mode (unlike the full pipeline, where export is opt-in via `output.export_embeddings`). The storage format follows `output.embeddings_format` — `sqlite` (default, a single `.db` file) or `npy` (one file per sample).

**Reusing an existing store:** if an embedding store already exists at the resolved path, it is **reused as-is and never recomputed** — the run reads the cached embeddings back and only (re)builds the UMAP outputs over them (or exits early if `umap.enabled` is off). To recompute from scratch instead, set `output.embeddings_overwrite: true`, which deletes the old store and re-extracts every embedding.

Optionally, this mode can also fit a [UMAP](https://umap-learn.readthedocs.io/) projection over the embeddings and write a data file and a plot. This is **off by default** and requires the optional `[umap]` extra:

```bash
pip install bioaccx[umap]      # or: uv sync --extra umap
```

Then enable it with `umap.enabled: true` in the config. If `umap.enabled` is set without the extra installed, the run errors with an install hint.

**Config (the `umap` section is optional; UMAP is off unless `enabled: true`):**

```yaml
foundation_model:
  registry_id: "0xbb00"

dataset:
  data_dir: ./my_audio
  label_mode: subfolders

output:
  output_path: ./outputs
  model_name: my_classifier
  model_version: "1.0"
  embeddings_format: sqlite   # or npy
  embeddings_overwrite: false # reuse an existing store; set true to recompute

umap:                # optional — requires the [umap] extra
  enabled: true      # off by default
  n_neighbors: 15
  min_dist: 0.1
  n_components: 2
  metric: euclidean
  random_seed: null  # falls back to dataset.random_seed
```

**Run:**

```bash
bioaccx my_config.yaml --embeddings
```

**Outputs** (written to `output_path/[stem]/`):

| File | Description |
|------|-------------|
| `[fm_id]_embeddings.db` (or `embeddings/`) | The embedding database — SQLite file, or a directory of `.npy` files |
| `[stem]_dataset_list.csv` | The list of samples used, with labels and train/test split |
| `[stem]_umap.csv` | UMAP coordinates per sample (`umap_1 … umap_N`, `label`, `split`) — only when `umap.enabled` |
| `[stem]_umap.png` | 2-D scatter-plot of the UMAP projection, coloured by label — only when `umap.enabled` |

**Python API:**

```python
from bioaccx.config import load_config
from bioaccx.train import run_embeddings

cfg = load_config("my_config.yaml")
outputs = run_embeddings(cfg)
print(outputs["umap_plot"])
```

---

## Merging a pre-existing head into a full model

Use `--merge` to combine a backbone and a separately-produced classifier head into a single full ONNX model — without running training or loading a dataset. This is useful when you already have a trained head (e.g. produced by a previous `bioaccx` run or exported by another tool) and just want to bundle it with the backbone for deployment.

The backbone must be in ONNX format. The head can be either ONNX or TFLite; a TFLite head is automatically converted to ONNX before merging.

The backbone can be loaded from a **local file** or downloaded from **HuggingFace Hub** — the same `foundation_model.source` field used for training.

**Config (registry shorthand — downloads backbone from HuggingFace):**

```yaml
foundation_model:
  registry_id: "0xbb00"   # BirdNET 2.4 ONNX; all defaults filled automatically

output:
  output_path: ./merged_models
  model_name: my_classifier
  model_version: "1.0"
  head_path: /models/my_classifier_v1.0_keras_head.tflite   # or .onnx
```

**Config (registry shorthand — local backbone):**

```yaml
foundation_model:
  registry_id: "0xbb00"
  source: local
  path: /models/birdnet_backbone.onnx

output:
  output_path: ./merged_models
  model_name: my_classifier
  model_version: "1.0"
  head_path: /models/my_classifier_v1.0_keras_head.tflite   # or .onnx
```

**Run:**

```bash
bioaccx my_config.yaml --merge
```

The merged model is written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/[stem]_full.onnx`.

**Python API:**

```python
from bioaccx.config import load_config
from bioaccx.train import run_merge

cfg = load_config("my_config.yaml")
out_path = run_merge(cfg)
print(out_path)
```

---

## Foundation model registry

Each foundation model has a compact 16-bit hex ID.  This ID is embedded in every output filename and embeddings cache filename so that classifiers are unambiguously traceable back to the exact backbone they were trained on.

The registry stores the full set of default parameters for each model — including source URL, audio config, and tensor names — so users can reference a model with a single `registry_id` field in the config instead of spelling out every parameter.

### Browsing the registry

```bash
bioaccx --registry
```

This prints all registered models with their IDs, descriptions, HuggingFace source URLs, and audio parameters.  No config file needed.

### Using a registry ID in config

```yaml
foundation_model:
  registry_id: "0xbb00"   # loads all defaults for BirdNET 2.4 ONNX (not optimized)
```

Any field set alongside `registry_id` overrides the registry default.  Fields not set by the user are filled from the registry.  When `registry_id` is absent, bioaccx attempts an auto-lookup using the `name` / `version` / `data_type` / `format` fields; if those match a registry entry, defaults are applied the same way.

### Supported foundation models

| Registry ID | Model | Format | `sample_rate` | `window_seconds` | `embedding_size` | `input_name` | Notes |
|---|---|---|---|---|---|---|---|
| `0xbb00` | BirdNET 2.4 | ONNX | 48000 | 3.0 | 1024 | `INPUT` | Backbone, no classifier head (`model_backbone.onnx`) |
| `0xbb01` | BirdNET 2.4 | ONNX | 48000 | 3.0 | 1024 | `INPUT` | Backbone, optimized export (`birdnet_backbone.onnx`) |
| `0xbb02` | BirdNET 2.4 | TFLite | 48000 | 3.0 | 1024 | `INPUT` | Full model; `tflite_output_tensor_offset: -1` set automatically |
| `0xbb10` | Perch 2.0 | ONNX | 32000 | 5.0 | 1536 | `inputs` | Backbone with DFT front-end |
| `0xbb11` | Perch 2.0 | ONNX | 32000 | 5.0 | 1536 | `inputs` | Backbone without DFT front-end |
| `0xbb12` | Perch 2.0 | TFLite | 32000 | 5.0 | 1536 | `inputs` | Full model; embedding is the first output (`tflite_output_tensor_offset: 0`). Use when the deliverable must be a TFLite full model |

Any model that accepts a `[batch, samples]` float32 tensor and outputs an embedding vector is compatible.  Use `registry_id` to load a registered model with a single config line; run `bioaccx --registry` to see all registered models with their source URLs.

If no registry entry is found, the fallback ID `0xffff` is used and all model parameters must be set explicitly.

### Adding a new model

The registry lives in `bioaccx/registry.py`.  Add an entry to `_REGISTRY` keyed by its hex ID:

```python
_REGISTRY: dict[int, dict] = {
    0xBB00: {
        "name": "birdnet",
        "version": "2.4",
        "data_type": "FP32",
        "format": "onnx",
        "description": "BirdNET 2.4 ONNX backbone without classifier head.",
        "source": "huggingface",
        "hf_repo": "biodiversica/BirdNET-onnx-backbone",
        "hf_filename": "model_backbone.onnx",
        "sample_rate": 48000,
        "window_seconds": 3.0,
        "input_name": "INPUT",
        "output_name": "embedding",
        "embedding_size": 1024,
    },
    # add new entries here ...
}
```

All keys except `description` must correspond to `FoundationModelConfig` field names.  When multiple entries share the same `(name, version, data_type, format)` tuple, the lowest hex ID is used for auto-lookup; the others must be referenced by explicit `registry_id`.

---

## Python API

bioaccx can also be used as a library:

```python
from bioaccx.config import load_config
from bioaccx.train import run

cfg = load_config("my_config.yaml")
outputs = run(cfg)
print(outputs)
# {'dataset_info': '...', 'keras_onnx_head': '...', 'model_info': '...', ...}
```

Constructing a config programmatically:

```python
from bioaccx.config import (
    BioaccxConfig, FoundationModelConfig, DatasetConfig,
    TrainingConfig, KerasConfig, OutputConfig,
)

cfg = BioaccxConfig(
    foundation_model=FoundationModelConfig(
        name="birdnet",
        version="2.4",
        format="onnx",
        source="local",
        path="/models/birdnet_headless.onnx",
        sample_rate=48000,
        window_seconds=3.0,
        input_name="INPUT",
        embedding_size=1024,
    ),
    dataset=DatasetConfig(
        data_dir="/data/birds",
        label_mode="file_per_label",
        overlap=0.5,
        embedding_workers=8,
    ),
    training=TrainingConfig(
        classifier="both",
        keras=KerasConfig(
            epochs=100,
            hidden_units=512,
            normalize_embeddings=True,
            focal_loss=True,
        ),
    ),
    output=OutputConfig(
        output_path="./models",
        model_name="bird_classifier",
        model_version="1.0",
        output_type="both",
        exclude_labels=["background"],
    ),
)

from bioaccx.train import run
outputs = run(cfg)
```

---

## CLI reference

```
bioaccx [CONFIG] [--validate] [--dataset] [--embeddings] [--merge] [--registry]

Arguments:
  config      Path to YAML or JSON configuration file (required unless
              --registry is used)
  --registry  Print all registered foundation models with their IDs,
              descriptions, HuggingFace source URLs, and audio parameters,
              then exit. No config file required.
  --validate  Parse and validate the config without running training or
              loading the foundation model, then exit
  --dataset   Load, split, and export the dataset as chunked WAV files
              without loading the foundation model or training
  --embeddings  Compute the embedding database without training a classifier
              (preparing the dataset first if needed). Embeddings are always
              exported (SQLite by default). When umap.enabled is set (requires
              the [umap] extra), also fits a UMAP projection and writes a UMAP
              data CSV and a scatter-plot PNG.
  --merge     Merge an existing ONNX backbone and ONNX or TFLite classifier
              head into a single full ONNX model. Requires
              foundation_model.path (backbone) and output.head_path (head)
              in the config. A TFLite head is converted to ONNX automatically
              before merging. No dataset or training is performed.
```
