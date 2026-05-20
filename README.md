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
- [Augmentation and windowing](#augmentation-and-windowing)
  - [Noise augmentation](#noise-augmentation)
  - [Random sample shift](#random-sample-shift)
  - [Windowing and overlap](#windowing-and-overlap)
- [Embedding cache](#embedding-cache)
- [Excluding labels from the exported model](#excluding-labels-from-the-exported-model)
- [Output directory structure](#output-directory-structure)
- [Merging a pre-existing head into a full model](#merging-a-pre-existing-head-into-a-full-model)
- [Foundation model registry](#foundation-model-registry)
- [Supported foundation models](#supported-foundation-models)
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
# 1. Copy and edit the example config
cp example_config.yaml my_config.yaml

# 2. Validate config without running training
bioaccx my_config.yaml --validate

# 3. Export the dataset as chunked WAV files (no model needed)
bioaccx my_config.yaml --dataset

# 4. Train and export
bioaccx my_config.yaml

# 5. Merge an existing classifier head with a backbone (without training)
bioaccx my_config.yaml --merge
```

---

## Configuration reference

All parameters live in a single YAML (or JSON) file. Below is the full reference with defaults and descriptions.

### `foundation_model`

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
| `name` | required | Model name used in reports and output filenames |
| `version` | `"unknown"` | Model version string |
| `data_type` | `"FP32"` | Weight precision (e.g. `FP32`, `INT8`) |
| `format` | `onnx` | File format: `onnx`, `tflite`, or `protobuf` |
| `source` | `local` | Where to load from: `local`, `huggingface`, or `kaggle` |
| `path` | `null` | Path to local model file or directory |
| `hf_repo` | `null` | HuggingFace repo ID, e.g. `biodiversica/birdnet-headless` |
| `hf_filename` | `null` | Filename within HF repo (default: `model.onnx`) |
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

#### Using the full BirdNET TFLite model

The BirdNET `Model_FP32.tflite` file contains a classifier head that outputs 6522 bird species. The embedding lives one tensor slot before the classifier output. To use this model as a backbone:

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
  output_name: embedding
  embedding_size: 1024
  tflite_output_tensor_offset: -1
```

When `tflite_output_tensor_offset` is non-zero, the TFLite interpreter is automatically initialized with `experimental_preserve_all_tensors=True` so that intermediate tensors remain accessible after inference.

When exporting a full TFLite model with this backbone, the original classifier head ops and weight tensors are removed from the merged output — only the backbone computation up to the embedding is retained, followed by your new classifier head.

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
| `augmentation.augmentation_dir` | `null` | Directory containing noise WAV files for augmentation |
| `augmentation.snr_levels` | `null` | List of SNR values in dB; one augmented copy is produced per noise file per level |
| `augmentation.keep_original` | `true` | Also include the clean (unaugmented) sample alongside augmented copies |
| `augmentation.augment_test` | `false` | Apply the same augmentation to the test set |

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
  export_embeddings: false    # save embeddings alongside the model
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
| `exclude_labels` | `[]` | Labels to omit from the exported model output (still used during training) |
| `export_dataset` | `false` | Export chunked audio as WAV files in label subfolders |
| `export_embeddings` | `false` | Save extracted embeddings |
| `embeddings_format` | `sqlite` | Embedding storage format: `sqlite` (single `.db` file named by registry ID) or `npy` (one file per sample) |
| `embeddings_path` | `null` | Custom path for exported embeddings |
| `head_path` | `null` | Path to an existing classifier head (ONNX or TFLite) for use with `--merge` |

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
,,,,abc123,2023-07-14,06:00,-3,Ara macao,10.0,30.0
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
```

**How it works:**

- For every combination of *(training sample, noise file, SNR level)* one augmented copy is produced.
- A window-length chunk is extracted from the noise file at a **random start position** derived from a stable hash of `(random_seed, sample path, times, noise filename, SNR)` — results are fully reproducible across runs regardless of iteration order.
- If the noise file is shorter than the model's input window, the extracted chunk is zero-padded.
- Augmented samples are embedded by mixing signal and noise at inference time; no intermediate audio files are written unless `export_dataset: true` is also set.
- When `export_dataset: true`, augmented WAV files are written alongside clean ones with a `_noise_<stem>_snr<value>` suffix in the filename.
- The dataset list CSV gains two extra columns — `noise_file` and `snr_db` — for every run that uses augmentation (empty for clean samples).
- Augmentation parameters are recorded in the `_metadata.json` output file under an `"augmentation"` key.

**Dataset expansion factor:**

```
total train samples = clean_train × (N_noise_files × N_snr_levels + keep_original)
```

For example, 100 clean train samples with 3 noise files, SNR levels `[0, 10, 20]`, and `keep_original: true` → 100 × (3×3 + 1) = **1000 train samples**.

By default, `augment_test: false` keeps the test set clean for unbiased evaluation. Set it to `true` when you specifically want to measure model robustness under noise conditions.

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

## Output directory structure

All outputs are written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/`.

The `foundation_model_id` is a 16-bit hex identifier resolved from the registry (see [Foundation model registry](#foundation-model-registry)).  For example, a classifier trained on BirdNET 2.4 FP32 ONNX (`0xbb00`) would produce:

```
custom_models/
  my_classifier_0xbb00_v1.0/
    my_classifier_0xbb00_v1.0_labels.txt            # output class names, one per line
    my_classifier_0xbb00_v1.0_metadata.json         # full metadata JSON
    my_classifier_0xbb00_v1.0_dataset_list.csv      # per-sample split/label summary
    my_classifier_0xbb00_v1.0_keras_head.onnx       # Keras head only (embedding input)
    my_classifier_0xbb00_v1.0_keras_full.onnx       # Keras + backbone merged (audio input)
    my_classifier_0xbb00_v1.0_keras_head.tflite     # Keras head only (TFLite)
    my_classifier_0xbb00_v1.0_keras_full.tflite     # Keras + backbone merged (TFLite)
    my_classifier_0xbb00_v1.0_sklearn_head.onnx     # sklearn head only
    my_classifier_0xbb00_v1.0_keras_report.txt      # training history + metrics
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

---

## Merging a pre-existing head into a full model

Use `--merge` to combine a backbone and a separately-produced classifier head into a single full ONNX model — without running training or loading a dataset. This is useful when you already have a trained head (e.g. produced by a previous `bioaccx` run or exported by another tool) and just want to bundle it with the backbone for deployment.

The backbone must be in ONNX format. The head can be either ONNX or TFLite; a TFLite head is automatically converted to ONNX before merging.

The backbone can be loaded from a **local file** or downloaded from **HuggingFace Hub** — the same `foundation_model.source` field used for training.

**Config (local backbone):**

```yaml
foundation_model:
  name: birdnet
  version: "2.4"
  format: onnx
  source: local
  path: /models/birdnet_backbone.onnx
  sample_rate: 48000
  window_seconds: 3.0
  input_name: INPUT
  embedding_size: 1024

output:
  output_path: ./merged_models
  model_name: my_classifier
  model_version: "1.0"
  head_path: /models/my_classifier_v1.0_keras_head.tflite   # or .onnx
```

**Config (HuggingFace backbone):**

```yaml
foundation_model:
  name: birdnet
  version: "2.4"
  format: onnx
  source: huggingface
  hf_repo: biodiversica/BirdNET-onnx-backbone
  sample_rate: 48000
  window_seconds: 3.0
  input_name: INPUT
  embedding_size: 1024

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

Each foundation model is identified by a compact 16-bit hex ID derived from its `name`, `version`, `data_type`, and `format`. This ID is embedded in every output filename and embeddings cache filename so that classifiers are unambiguously traceable back to the exact backbone they were trained on.

| ID | Model | Version | Data type | Format |
|---|---|---|---|---|
| `0xbb00` | BirdNET | 2.4 | FP32 | onnx |
| `0xbb01` | BirdNET | 2.4 | FP32 | tflite |
| `0xbb10` | Perch | 2.0 | FP32 | onnx |

If `(name, version, data_type, format)` does not match any registry entry, the fallback ID `0xffff` is used.

The registry lives in `bioaccx/registry.py`. To add a new model, append an entry to the `_REGISTRY` dict:

```python
_REGISTRY: dict[tuple[str, str, str, str], int] = {
    ("birdnet", "2.4", "FP32", "onnx"):   0xBB00,
    ("birdnet", "2.4", "FP32", "tflite"): 0xBB01,
    ("perch",   "2.0", "FP32", "onnx"):   0xBB10,
}
```

---

## Supported foundation models

| Model | Format | `sample_rate` | `window_seconds` | `embedding_size` | `input_name` | Notes |
|---|---|---|---|---|---|---|
| BirdNET 2.4 | onnx | 48000 | 3.0 | 1024 | `INPUT` | Headless backbone |
| BirdNET 2.4 | tflite | 48000 | 3.0 | 1024 | `INPUT` | Full model; set `tflite_output_tensor_offset: -1` |
| Perch 2.0 | onnx | 32000 | 5.0 | 1536 | `inputs` | Headless backbone |

Any model that accepts a `[batch, samples]` float32 tensor and outputs an embedding vector is compatible.

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
bioaccx CONFIG [--validate] [--dataset] [--merge]

Arguments:
  config      Path to YAML or JSON configuration file (required)
  --validate  Parse and validate the config without running training or
              loading the foundation model, then exit
  --dataset   Load, split, and export the dataset as chunked WAV files
              without loading the foundation model or training
  --merge     Merge an existing ONNX backbone and ONNX or TFLite classifier
              head into a single full ONNX model. Requires
              foundation_model.path (backbone) and output.head_path (head)
              in the config. A TFLite head is converted to ONNX automatically
              before merging. No dataset or training is performed.
```
