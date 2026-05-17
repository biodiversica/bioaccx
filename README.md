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
| `tensorflow-cpu` / `tensorflow` | Keras classifier, protobuf foundation models |
| `tf2onnx` | Exporting Keras head to ONNX |
| `scikit-learn` + `skl2onnx` | sklearn classifier |
| `huggingface-hub` | Downloading foundation models from HuggingFace Hub |

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
  data_type: FP32             # weight precision — used to resolve the foundation model ID

  # Model format on disk
  format: onnx                # onnx | tflite | protobuf (TF SavedModel)

  # --- Local file ---
  source: local
  path: /path/to/model_headless.onnx

  # --- OR from HuggingFace Hub ---
  # source: huggingface
  # hf_repo: biodiversica/birdnet-headless
  # hf_filename: birdnet_headless.onnx   # optional; defaults to model.onnx
  # hf_revision: main                    # branch / tag / commit (optional)

  # Audio preprocessing
  sample_rate: 48000
  window_seconds: 3.0         # duration of each input window
  # window_samples: 144000    # alternative: exact sample count (takes priority)

  # Tensor names (check your model's input/output node names)
  input_name: INPUT
  output_name: embedding
  embedding_size: 1024        # dimensionality of the embedding vector
```

| Parameter | Default | Description |
|---|---|---|
| `name` | required | Model name used in reports and output filenames |
| `version` | `"unknown"` | Model version string |
| `data_type` | `"FP32"` | Weight precision (e.g. `FP32`); combined with `name` and `version` to resolve the foundation model registry ID |
| `format` | `onnx` | File format: `onnx`, `tflite`, or `protobuf` |
| `source` | `local` | Where to load from: `local` or `huggingface` |
| `path` | `null` | Path to local model file or directory |
| `hf_repo` | `null` | HuggingFace repo ID, e.g. `biodiversica/birdnet-headless` |
| `hf_filename` | `null` | Filename within HF repo (default: `model.onnx`) |
| `hf_revision` | `null` | Branch, tag, or commit hash |
| `sample_rate` | `48000` | Expected audio sample rate in Hz |
| `window_seconds` | `null` | Input window duration in seconds |
| `window_samples` | `null` | Input window in samples (takes priority over `window_seconds`) |
| `input_name` | `"input"` | Name of the model's input tensor |
| `output_name` | `"embedding"` | Name of the model's output tensor |
| `embedding_size` | `1024` | Embedding vector dimensionality |

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
  embeddings_cache_path: null # pre-computed .npy cache directory (optional)
  test_ratio: 0.2             # fraction of data for test set
  random_seed: 42
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
| `embeddings_cache_path` | `null` | Directory with pre-computed `.npy` embeddings to load instead of recomputing |
| `append_dataset_path` | `null` | Path to an existing exported dataset to append new samples to (see below) |
| `ext_table_file` | `null` | CSV/TSV of iNaturalist observation IDs (or mixed with local filename rows) |
| `obs_id_col` | `observation_id` | Column name for iNaturalist observation IDs |
| `sound_index_col` | `sound_index` | Column name for iNaturalist sound index (0-based) |
| `xc_id_col` | `xc_id` | Column name for Xeno-canto recording IDs |
| `ext_cache_dir` | `~/.cache/bioaccx/ext` | Local cache for all downloaded remote audio and metadata |
| `xc_api_key` | `null` | Xeno-canto API v3 key — enables scientific name lookup for XC rows; audio downloads without it |
| `test_ratio` | `0.2` | Proportion of data held out for the test set |
| `random_seed` | `42` | Random seed for reproducible splits |

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

  sklearn:
    C: 1.0
    max_iter: 2000
    solver: lbfgs
```

| Parameter | Default | Description |
|---|---|---|
| `classifier` | `keras` | Which classifier(s) to train |
| `keras.hidden_units` | `256` | Units in the hidden Dense layer; `0` = no hidden layer |
| `keras.dropout` | `0.25` | Dropout rate applied before each Dense layer |
| `keras.epochs` | `50` | Training epochs |
| `keras.batch_size` | `32` | Mini-batch size |
| `keras.learning_rate` | `0.0001` | Adam optimizer learning rate |
| `keras.output_activation` | `null` | Output activation; `null` means raw logits |
| `sklearn.C` | `1.0` | Regularisation strength (LogisticRegression) |
| `sklearn.max_iter` | `2000` | Maximum iterations for the solver |
| `sklearn.solver` | `lbfgs` | Solver algorithm |

**Output activation notes:**
- `null` (default): raw logits; numerically most stable; use `softmax` at inference time if needed.
- `sigmoid`: per-class binary probability; use for multi-label problems.
- `softmax`: normalised class probabilities; use when you want the model to output probabilities directly.

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
  export_embeddings: false    # save .npy embeddings alongside the model
  # embeddings_path: /path/to/save/embeddings   # default: output_dir/embeddings
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
| `export_embeddings` | `false` | Save extracted embeddings as `.npy` files |
| `embeddings_path` | `null` | Custom directory for exported embeddings |
| `head_path` | `null` | Path to an existing classifier head (ONNX or TFLite) for use with `--merge` |

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

## Remote sound sources (iNaturalist and Xeno-canto)

Use `ext_table_file` to include audio from [iNaturalist](https://www.inaturalist.org/) and/or [Xeno-canto](https://xeno-canto.org/) alongside (or instead of) local files.

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
| 3 | `filename` | Local audio file |

**All columns:**

| Column | Notes |
|---|---|
| `observation_id` | iNaturalist observation ID |
| `sound_index` | 0-based sound index within the observation; defaults to `0` (iNaturalist only) |
| `xc_id` | Xeno-canto recording ID — numeric or with `XC` prefix |
| `filename` | Path to a local audio file (absolute or relative to `data_dir`) |
| `label` | Falls back to the taxon scientific name for remote rows if empty |
| `start_time` / `end_time` | Seconds; same partial-time rules as local table mode |
| `split` | `train` or `test`; auto-split if empty |

**Mixed table** — all three source types can coexist in one file:

```csv
filename,observation_id,sound_index,xc_id,label,start_time,end_time
/data/rec.wav,,,,cicada,0.0,3.0,
,12345678,0,,,1.0,6.0,
,,,98765,Turdus merula,,,train
```

**Xeno-canto API key** — Xeno-canto uses API v3, which requires a personal key for metadata queries (scientific name lookup).  Without a key, audio is still downloaded directly but the label falls back to `"xc_<id>"` unless you set it explicitly in the table.  Register at [xeno-canto.org/explore/api](https://xeno-canto.org/explore/api).

```yaml
dataset:
  xc_api_key: YOUR_KEY_HERE   # optional; enables scientific name lookup for XC rows
```

**Caching** — audio files and metadata are cached locally on first download; subsequent runs skip the network entirely.

```yaml
dataset:
  ext_cache_dir: /path/to/cache   # default: ~/.cache/bioaccx/ext
```

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

## Windowing and overlap

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

Computing embeddings is the slowest step. Use `embeddings_cache_path` to store and reuse them:

```yaml
dataset:
  embeddings_cache_path: /path/to/cache
```

- On first run, embeddings are computed and saved as `.npy` files.
- On subsequent runs, existing files are loaded directly — skipping inference.
- Cache filenames encode the audio file stem, start time, and end time, so different chunking or overlap settings produce separate cache entries.

To also export embeddings as part of the pipeline output:

```yaml
output:
  export_embeddings: true
  # embeddings_path: /custom/export/path   # optional
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

The `foundation_model_id` is a 16-bit hex identifier resolved from the registry (see [Foundation model registry](#foundation-model-registry)).  For example, a classifier trained on BirdNET 2.4 FP32 (`0xbb00`) would produce:

```
custom_models/
  my_classifier_0xbb00_v1.0/
    my_classifier_0xbb00_v1.0_labels.txt            # output class names, one per line
    my_classifier_0xbb00_v1.0_metadata.json         # full metadata JSON
    my_classifier_0xbb00_v1.0_dataset_list.csv      # per-sample split/label summary
    my_classifier_0xbb00_v1.0_keras_head.onnx       # Keras head only
    my_classifier_0xbb00_v1.0_keras_full.onnx       # Keras + foundation merged
    my_classifier_0xbb00_v1.0_sklearn_head.onnx     # sklearn head only
    my_classifier_0xbb00_v1.0_keras_report.txt      # training history + metrics
    my_classifier_0xbb00_v1.0_sklearn_report.txt    # sklearn metrics
    my_classifier_0xbb00_v1.0_comparison_report.txt # side-by-side comparison
    embeddings/                                     # exported .npy embeddings (optional)
    dataset/                                        # exported chunked WAV files (optional)
      train/
        crow/
        robin/
      test/
        crow/
        robin/
```

---

## Foundation model registry

Each foundation model is identified by a compact 16-bit hex ID derived from its `name`, `version`, and `data_type`.  This ID is embedded in every output filename so that classifiers are unambiguously traceable back to the backbone they were trained on.

| ID | Model | Version | Data type |
|---|---|---|---|
| `0xbb00` | BirdNET | 2.4 | FP32 |
| `0xbb10` | Perch | 2.0 | FP32 |

If `(name, version, data_type)` does not match any registry entry, the fallback ID `0xffff` is used.

The registry lives in `bioaccx/registry.py`.  To add a new model, append an entry to the `_REGISTRY` dict.

---

## Supported foundation models

| Model | Format | `sample_rate` | `window_seconds` | `embedding_size` | `input_name` |
|---|---|---|---|---|---|
| BirdNET 2.4 | onnx | 48000 | 3.0 | 1024 | `INPUT` |
| Perch 2.0 | onnx | 32000 | 5.0 | 1536 | `inputs` |

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
        keras=KerasConfig(epochs=100, hidden_units=512),
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

The merged model is written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/[model_name]_[foundation_model_id]_v[model_version]_full.onnx`.

**Python API:**

```python
from bioaccx.config import load_config
from bioaccx.train import run_merge

cfg = load_config("my_config.yaml")
out_path = run_merge(cfg)
print(out_path)
# ./merged_models/my_classifier_v1.0/my_classifier_v1.0_full.onnx
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
