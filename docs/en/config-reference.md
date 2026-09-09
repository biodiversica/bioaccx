# Config reference

Every key the config file accepts, with its default and what it does.

This page is **generated** from `bioaccx/config.py` — the same dataclasses the
browser form is built from — by `tools/gen_docs.py`. To change a description,
edit the comment above (or beside) the field in `config.py`: the form and this
page both read it, so they cannot disagree.

For the prose, the worked examples and the YAML around these keys, see
[Configuration](configuration.md).

<!-- generated:config-reference — edit bioaccx/config.py, not this block -->

## `foundation_model`

The backbone and the audio window it expects.

| Key | Default | Description |
|---|---|---|
| `registry_id` | `null` | Load every default for a known backbone from one line. Any field set alongside it overrides the registry default. |
| `name` | *required* | used in reports and output filenames |
| `version` | `unknown` | model version string |
| `data_type` | `FP32` | weight precision, e.g. FP32 / INT8 |
| `format` | `onnx` | Model format on disk |
| `source` | `local` | Source: local file path, huggingface hub, or kaggle |
| `path` | `null` | path to local model file/dir |
| `hf_repo` | `null` | e.g. "biodiversica/birdnet" |
| `hf_filename` | `null` | specific file within HF repo |
| `hf_revision` | `null` | branch / tag / commit |
| `kaggle_handle` | `null` | e.g. "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier" |
| `kaggle_filename` | `null` | specific file within the downloaded dir (onnx/tflite) |
| `sample_rate` | `48000` | Sample rate the backbone expects, in Hz |
| `window_samples` | `null` | takes priority over window_seconds |
| `window_seconds` | `null` | e.g. 3.0 |
| `input_name` | `input` | name of the audio input tensor |
| `output_name` | `embedding` | name of the output (embedding) tensor |
| `embedding_size` | `1024` | embedding vector dimensionality |
| `onnx_providers` | `null` | ONNX Runtime execution providers (e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]). Defaults to ORT's own provider priority when None. |
| `onnx_batch_size` | `1` | Number of audio windows to process in a single ONNX inference call. Values > 1 enable GPU batch mode (used only for format="onnx"). |
| `tflite_output_tensor_offset` | `0` | Offset applied to the resolved TFLite output tensor index to reach the embedding tensor.  Use -1 for models like BirdNET tflite where the classifier head is the first output and the embedding sits one slot before it in the graph's tensor list. |
| `tflite_trim_to_embedding` | `true` | Trim a TFLite graph down to the embedding tensor when the model computes more than that (a bundled classifier head, auxiliary outputs). Done once at load time; every later window then skips the discarded branches. Set false to run the model exactly as shipped. |

## `dataset`

Where the audio comes from, how it is labelled, preprocessed, augmented and split.

| Key | Default | Description |
|---|---|---|
| `data_dir` | `[]` | Local audio source(s); may be empty when using ext_table_file exclusively |
| `label_mode` | `subfolders` | How labels are organized in data_dir |
| `table_file` | `null` | CSV/TSV; used when label_mode="table" |
| `audio_extensions` | `["wav", "flac", "mp3", "ogg"]` | Accepted audio file extensions (case-insensitive) — **run level** (cannot be overridden per source) |
| `overlap` | `0.0` | Overlap between consecutive windows when chunking label segments (0.0–1.0) |
| `embedding_workers` | `4` | Parallel workers for embedding extraction (capped to available CPU cores) — **run level** (cannot be overridden per source) |
| `embeddings_cache_path` | `null` | Path to look for pre-computed .npy embeddings before running the model — **run level** (cannot be overridden per source) |
| `append_dataset_path` | `null` | Path to an existing exported dataset (subfolders layout with train/test). New samples from data_dir that are not already present are appended to it. — **run level** (cannot be overridden per source) |
| `test_ratio` | `0.2` | Train / test split; ignored when the dataset already encodes the split — **run level** (cannot be overridden per source) |
| `random_seed` | `42` | seed for splits, shuffling and noise offsets — **run level** (cannot be overridden per source) |
| `filename_col` | `filename` | Column names used in table / ext_table_file modes |
| `label_col` | `label` | class label of the row |
| `start_col` | `start_time` | segment start, in seconds |
| `end_col` | `end_time` | segment end, in seconds |
| `split_col` | `split` | optional; values "train" / "test" |
| `ext_table_file` | `null` | iNaturalist table — CSV/TSV with observation_id rows (or mixed with filename rows) |
| `obs_id_col` | `observation_id` | iNaturalist observation id |
| `sound_index_col` | `sound_index` | iNaturalist sound index (0-based) |
| `xc_id_col` | `xc_id` | Xeno-canto recording id |
| `ext_cache_dir` | `null` | Local cache for downloaded remote audio; defaults to ~/.cache/bioaccx/ext — **run level** (cannot be overridden per source) |
| `xc_api_key` | `null` | Xeno-canto API v3 key — required for metadata (scientific name lookup). Audio can be downloaded without a key via the direct download URL. Register at https://xeno-canto.org/explore/api — **secret** (never echoed back to the browser) — **run level** (cannot be overridden per source) |
| `arbimon_credentials_path` | `null` | Arbimon / rfcx — path to the persisted credentials file produced by rfcx.Client().authenticate(persisted_credentials_path=...). Column names used to identify Arbimon rows in ext_table_file. — **run level** (cannot be overridden per source) |
| `arbimon_stream_id_col` | `stream_id` | Arbimon stream / site id |
| `arbimon_date_col` | `date` | local recording date |
| `arbimon_time_col` | `time` | local recording start time |
| `arbimon_utc_offset_col` | `utc_offset` | UTC offset of those times |
| `filter` | `null` | Audio preprocessing applied before chunking (filter → speed → chunks). Filter type: 'hpf' \| 'lpf' \| 'bpf' \| null |
| `filter_freq` | `null` | Cut-off in Hz for hpf/lpf; [low_hz, high_hz] for bpf |
| `filter_order` | `5` | Butterworth filter order |
| `speed` | `1.0` | Playback speed multiplier (>1 faster / shorter, <1 slower / longer). Label times are scaled accordingly: new_time = old_time / speed. |
| `ssh_host` | `null` | SSH / SFTP access — when ssh_host is set, data_dir paths are treated as remote paths on the SSH server and mirrored locally via paramiko before the pipeline runs.  Requires: pip install paramiko |
| `ssh_user` | `null` | SSH user; defaults to the local one |
| `ssh_port` | `22` | SSH port |
| `ssh_key_path` | `null` | path to private key file |
| `random_sample_shift` | `false` | When True, samples shorter than the foundation model window are placed at a random offset within the window rather than always starting at position 0. Each augmented copy of the same sample gets a distinct offset. |
| `min_anchor_fraction` | `0.1` | Minimum fraction of the window that must be new (uncovered) audio for the anchor chunk at the end of a file to be emitted. Prevents near-duplicate chunks when files are only slightly longer than the window (e.g. 3.013 s with a 3 s window). Set to 0.0 to always emit the anchor chunk. |
| `sources` | `null` | Combine several heterogeneous sources in one run. Each source inherits the fields above and overrides them with its own. Run-level fields are never overridden per source. |

### `dataset.augmentation` — Augmentation

| Key | Default | Description |
|---|---|---|
| `snr_levels` | *required* | SNR values in dB; one augmented copy per noise file per level |
| `augmentation_dir` | `null` | Directory of WAV files used as noise sources. Optional when augmentation_labels is set; if both are given the noise pool is the union. |
| `augmentation_labels` | `null` | Labels of the dataset being created whose audio is used as additional noise sources (in addition to any augmentation_dir). The samples of these labels are mixed into the other samples' augmentation but are themselves never augmented (treated like skip_labels). They remain trainable classes. |
| `keep_original` | `true` | Keep the clean sample alongside its augmented copies |
| `augment_test` | `false` | Augment the test set as well, not just the training set |
| `skip_labels` | `null` | Labels left unaugmented; they stay trainable classes |
| `concatenate_augmentation_dir` | `false` | When True, all files in augmentation_dir are concatenated into a single in-memory array used as the sole noise source. If a sibling Audacity .txt label file exists for an audio file, only the labeled segments are used; otherwise the entire file is included. |
| `random_augmentation_dir` | `false` | When True, files in augmentation_dir are shuffled once (using random_seed) and assigned round-robin to augmented samples — one file per (sample, SNR) pair, cycling without repetition within each pass. This produces one noise condition (like concatenate_augmentation_dir) but draws from individual files rather than a merged track. |

## `training`

Which classifier head to fit, and how.

| Key | Default | Description |
|---|---|---|
| `classifier` | `keras` | Which head(s) to train: keras, sklearn, or both |

### `training.keras` — Keras head

*Only when `training.classifier` = `keras` / `both`.*

| Key | Default | Description |
|---|---|---|
| `hidden_units` | `256` | Units in the hidden Dense layer; 0 = no hidden layer (linear classifier) |
| `dropout` | `0.25` | dropout rate before each Dense layer |
| `epochs` | `50` | maximum epochs; early stopping may halt sooner |
| `batch_size` | `32` | mini-batch size |
| `learning_rate` | `0.0001` | Adam peak learning rate (cosine decay with linear warmup) |
| `output_activation` | `null` | Output activation: None (logits, default) \| "sigmoid" \| "softmax" \| "grouped_softmax" (requires label_groups) |
| `label_groups` | `{}` | Grouped softmax: softmax within each group, groups independent of each other.  Maps group name -> member labels; each group gains a synthetic "<group>_none" output column.  Members of one group are mutually exclusive, members of different groups can fire together.  Training labels not listed in any group are background: they get no output column and supply the "none" target for every group. |
| `export_logits` | `false` | Strip the output activation layer before exporting, so the exported head emits raw logits while training still used output_activation (this is what BirdNET-Analyzer does with classifier.pop()). No-op when output_activation is None. Callers must apply the activation themselves. |
| `normalize_embeddings` | `true` | Z-score normalization of input embeddings (adapted on X_train) |
| `focal_loss` | `false` | Focal loss (replaces cross-entropy when enabled) |
| `focal_loss_gamma` | `2.0` | focal loss focusing parameter γ |
| `focal_loss_alpha` | `0.25` | focal loss class balance parameter α |
| `label_smoothing` | `false` | Label smoothing applied to one-hot targets before training |
| `label_smoothing_alpha` | `0.1` | Subtracted from positive labels and redistributed to the negatives |
| `mixup` | `false` | Mixup data augmentation on training embeddings |
| `mixup_ratio` | `0.25` | fraction of positive samples to mix |
| `mixup_alpha` | `0.2` | Beta parameter of the mixing coefficient |
| `upsampling_ratio` | `0.0` | Upsampling of minority classes before training |
| `upsampling_mode` | `repeat` | repeat = random duplication, mean = pairwise mean, linear = random interpolation, smote = k-NN interpolation |
| `seed` | `true` | Set TF + numpy random seeds before training for reproducibility. Uses dataset.random_seed. Set to false to disable (training will vary run-to-run). |

### `training.sklearn` — sklearn head

*Only when `training.classifier` = `sklearn` / `both`.*

| Key | Default | Description |
|---|---|---|
| `C` | `1.0` | inverse regularisation strength |
| `max_iter` | `2000` | maximum solver iterations |
| `solver` | `lbfgs` | LogisticRegression solver algorithm |

## `output`

Where results are written, in which formats and precisions.

| Key | Default | Description |
|---|---|---|
| `output_path` | `./outputs` | parent directory for all output |
| `model_name` | `custom_classifier` | used in filenames and the subdirectory |
| `model_version` | `1.0` | version string used in filenames |
| `output_type` | `head` | head = classifier only; full = foundation + classifier; both = save both |
| `output_format` | `onnx` | File format(s) of the exported model |
| `exclude_labels` | `[]` | Labels to exclude from the exported classifier output (still used during training) |
| `export_dataset` | `false` | Export chunked audio samples as WAV files in label subfolders |
| `export_embeddings` | `false` | Save computed embeddings for reuse |
| `embeddings_format` | `npy` | Storage format for exported/cached embeddings: npy (one file per sample) or sqlite (single .db file per run, more portable) |
| `embeddings_path` | `null` | Directory (npy) or file path (sqlite) for exported embeddings; defaults to <output_dir>/embeddings or <output_dir>/embeddings.db |
| `embeddings_overwrite` | `false` | --embeddings mode only: when an embedding store already exists at the resolved path, recompute and overwrite it. When False (default), the existing store is reused as-is — no embeddings are recomputed and the run only (re)builds the UMAP outputs (when umap.enabled). |
| `head_path` | `null` | Path to an existing ONNX classifier head for --merge (no training required) |
| `extract_from` | `null` | Path to a full BirdNET-Analyzer model (.tflite) for --extract_head: the trailing classifier head is read from the flatbuffer and re-exported as a head-only model (no backbone conversion, no training). |
| `labels_file` | `null` | Optional class-label file for --extract_head (one label per line). When omitted, a sibling ``<model>_Labels.txt`` is used if present. |
| `data_types` | `null` | Output precisions to export: subset of {"FP32", "FP16", "INT8"}. None → defaults to [foundation_model.data_type]. Each precision yields a separate exported file tagged with the precision in its filename. INT8 uses dynamic/weight-only quantization (no calibration dataset). |

## `umap`

Projection settings for the embeddings command. Requires the [umap] extra.

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Fit the projection and write the CSV and plots (needs the [umap] extra) |
| `n_neighbors` | `15` | Neighbourhood size: low = local structure, high = global structure |
| `min_dist` | `0.1` | how tightly points may be packed |
| `n_components` | `2` | dimensions of the projection |
| `metric` | `euclidean` | distance metric between embeddings |
| `random_seed` | `null` | Random seed for UMAP; falls back to dataset.random_seed when None. |
| `cache_csv` | `null` | Path to a previously written UMAP data CSV. When set and the file exists, the embeddings run skips all computation (dataset load, embedding extraction, KMeans, UMAP fit) and only redraws the plots from the cached coordinates — useful for tweaking plot styling without recomputing. |

<!-- /generated:config-reference -->
