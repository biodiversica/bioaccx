# Config reference

This page lists every key accepted by the configuration file, with its default
value and description.

It is **generated** by `tools/gen_docs.py` from `bioaccx/config.py`, the same
dataclasses from which the browser form is built. To change a description, edit
the comment above (or beside) the field in `config.py`; the form and this page
both read it, so they always agree.

Explanations and worked examples of these keys are given in
[Configuration](configuration.md).

<!-- generated:config-reference — edit bioaccx/config.py, not this block -->

## `foundation_model`

The backbone and the audio window it expects.

| Key | Default | Description |
|---|---|---|
| `registry_id` | `null` | Selects a known backbone and loads all of its defaults. Any field set alongside it overrides the registry default. |
| `name` | *required* | Model name, used in reports and output filenames. |
| `version` | `unknown` | Model version string. |
| `data_type` | `FP32` | Weight precision, for example FP32 or INT8. |
| `format` | `onnx` | Model file format on disk. |
| `source` | `local` | Where the model is loaded from: a local path, the Hugging Face Hub, or Kaggle. |
| `path` | `null` | Path to the local model file or directory. |
| `hf_repo` | `null` | Hugging Face repository, for example "biodiversica/birdnet". |
| `hf_filename` | `null` | File to download from the Hugging Face repository. |
| `hf_revision` | `null` | Branch, tag or commit of the Hugging Face repository. |
| `kaggle_handle` | `null` | Kaggle model handle, for example "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier". |
| `kaggle_filename` | `null` | File to use within the downloaded Kaggle directory (ONNX or TFLite). |
| `sample_rate` | `48000` | Sample rate the backbone expects, in Hz. |
| `window_samples` | `null` | Window length in samples. Takes priority over window_seconds. |
| `window_seconds` | `null` | Window length in seconds, for example 3.0. |
| `input_name` | `input` | Name of the audio input tensor. |
| `output_name` | `embedding` | Name of the embedding output tensor. |
| `embedding_size` | `1024` | Dimensionality of the embedding vector. |
| `onnx_providers` | `null` | ONNX Runtime execution providers, for example ["CUDAExecutionProvider", "CPUExecutionProvider"]. When unset, ONNX Runtime uses its own provider priority. |
| `onnx_batch_size` | `1` | Number of audio windows processed in a single ONNX inference call. Values above 1 enable GPU batch mode. Applies only when format is "onnx". |
| `tflite_output_tensor_offset` | `0` | Offset applied to the resolved TFLite output tensor index to reach the embedding tensor. Use -1 for models such as the BirdNET TFLite model, where the classifier head is the first output and the embedding sits one position before it in the graph's tensor list. |
| `tflite_trim_to_embedding` | `true` | Trim a TFLite graph down to the embedding tensor when the model computes more than that, such as a bundled classifier head or auxiliary outputs. This is done once at load time, and every later window then skips the discarded branches. Set to false to run the model exactly as shipped. |

## `dataset`

Where the audio comes from, how it is labelled, preprocessed, augmented and split.

| Key | Default | Description |
|---|---|---|
| `data_dir` | `[]` | Local audio source or sources. May be empty when only ext_table_file is used. |
| `label_mode` | `subfolders` | How the labels are organised in data_dir. |
| `table_file` | `null` | CSV or TSV label table, used when label_mode is "table". |
| `audio_extensions` | `["wav", "flac", "mp3", "ogg"]` | Accepted audio file extensions (not case-sensitive). — **run level** (cannot be overridden per source) |
| `overlap` | `0.0` | Overlap between consecutive windows when label segments are chunked, from 0.0 to 1.0. |
| `embedding_workers` | `4` | Parallel workers for embedding extraction, capped at the available CPU cores. — **run level** (cannot be overridden per source) |
| `embeddings_cache_path` | `null` | Path searched for pre-computed .npy embeddings before the model is run. — **run level** (cannot be overridden per source) |
| `append_dataset_path` | `null` | Path to an existing exported dataset (subfolders layout with train and test). New samples from data_dir that are not already present are appended to it. — **run level** (cannot be overridden per source) |
| `test_ratio` | `0.2` | Fraction of the data held out for testing. Ignored when the dataset already defines the split. — **run level** (cannot be overridden per source) |
| `random_seed` | `42` | Seed for the split, shuffling and noise offsets. — **run level** (cannot be overridden per source) |
| `filename_col` | `filename` | Column holding the audio filename. |
| `label_col` | `label` | Column holding the class label. |
| `start_col` | `start_time` | Column holding the segment start, in seconds. |
| `end_col` | `end_time` | Column holding the segment end, in seconds. |
| `split_col` | `split` | Optional column holding the split, with the values "train" or "test". |
| `ext_table_file` | `null` | iNaturalist table: a CSV or TSV file of observation_id rows, optionally mixed with filename rows. |
| `obs_id_col` | `observation_id` | Column holding the iNaturalist observation ID. |
| `sound_index_col` | `sound_index` | Column holding the iNaturalist sound index (zero-based). |
| `xc_id_col` | `xc_id` | Column holding the Xeno-canto recording ID. |
| `ext_cache_dir` | `null` | Local cache for downloaded remote audio. Defaults to ~/.cache/bioaccx/ext. — **run level** (cannot be overridden per source) |
| `xc_api_key` | `null` | Xeno-canto API v3 key, required for metadata (scientific name lookup). Audio can be downloaded without a key through the direct download URL. Keys are issued at https://xeno-canto.org/explore/api. — **secret** (never echoed back to the browser) — **run level** (cannot be overridden per source) |
| `arbimon_credentials_path` | `null` | Arbimon (RFCx) credentials file, as written by rfcx.Client().authenticate(persisted_credentials_path=...). — **run level** (cannot be overridden per source) |
| `arbimon_stream_id_col` | `stream_id` | Column holding the Arbimon stream (site) ID. |
| `arbimon_date_col` | `date` | Column holding the local recording date. |
| `arbimon_time_col` | `time` | Column holding the local recording start time. |
| `arbimon_utc_offset_col` | `utc_offset` | Column holding the UTC offset of the recording date and time. |
| `filter` | `null` | Filter applied to the audio before chunking: "hpf" (high-pass), "lpf" (low-pass), "bpf" (band-pass) or null. |
| `filter_freq` | `null` | Cut-off frequency in Hz for hpf and lpf; [low_hz, high_hz] for bpf. |
| `filter_order` | `5` | Butterworth filter order. |
| `speed` | `1.0` | Playback speed multiplier: above 1 is faster and shorter, below 1 slower and longer. Label times are scaled to match: new_time = old_time / speed. |
| `ssh_host` | `null` | SSH server holding the audio. When set, data_dir paths are treated as remote paths on this server and mirrored locally with paramiko before the pipeline runs. Requires paramiko (pip install paramiko). |
| `ssh_user` | `null` | SSH user name. Defaults to the local user. |
| `ssh_port` | `22` | SSH port. |
| `ssh_key_path` | `null` | Path to the private key file. |
| `random_sample_shift` | `false` | When true, samples shorter than the foundation model window are placed at a random offset within the window instead of always starting at position 0. Each augmented copy of the same sample gets a different offset. |
| `min_anchor_fraction` | `0.1` | Minimum fraction of the window that must be new (uncovered) audio for the anchor chunk at the end of a file to be emitted. This prevents near-duplicate chunks when files are only slightly longer than the window (for example, 3.013 s with a 3 s window). Set to 0.0 to always emit the anchor chunk. |
| `sources` | `null` | Combine several heterogeneous sources in one run. Each source inherits the fields above and overrides them with its own. Run-level fields are never overridden per source. |

### `dataset.augmentation` — Augmentation

| Key | Default | Description |
|---|---|---|
| `snr_levels` | *required* | SNR values in dB. One augmented copy is made per noise file per level. |
| `augmentation_dir` | `null` | Directory of WAV files used as noise sources. Optional when augmentation_labels is set; if both are given, the noise pool is their union. |
| `augmentation_labels` | `null` | Labels of the dataset being created whose audio is used as additional noise sources, alongside any augmentation_dir. Samples of these labels are mixed into the augmentation of other samples but are never augmented themselves, as with skip_labels. They remain trainable classes. |
| `keep_original` | `true` | Keep the clean sample alongside its augmented copies. |
| `augment_test` | `false` | Augment the test set as well as the training set. |
| `skip_labels` | `null` | Labels that are not augmented. They remain trainable classes. |
| `concatenate_augmentation_dir` | `false` | When true, all files in augmentation_dir are concatenated into a single in-memory array used as the only noise source. If an Audacity .txt label file sits next to an audio file, only its labelled segments are used; otherwise the entire file is included. |
| `random_augmentation_dir` | `false` | When true, the files in augmentation_dir are shuffled once (using random_seed) and assigned in rotation to augmented samples: one file per (sample, SNR) pair, without repetition within each pass. Like concatenate_augmentation_dir, this produces a single noise condition, but it draws from individual files rather than a merged track. |

## `training`

Which classifier head to fit, and how.

| Key | Default | Description |
|---|---|---|
| `classifier` | `keras` | Which classifier head to train: keras, sklearn, or both. |

### `training.keras` — Keras head

*Only when `training.classifier` = `keras` / `both`.*

| Key | Default | Description |
|---|---|---|
| `hidden_units` | `256` | Units in the hidden Dense layer. 0 means no hidden layer (a linear classifier). |
| `dropout` | `0.25` | Dropout rate before each Dense layer. |
| `epochs` | `50` | Maximum number of epochs. Early stopping may end training sooner. |
| `batch_size` | `32` | Mini-batch size. |
| `learning_rate` | `0.0001` | Peak Adam learning rate (cosine decay with a linear warm-up). |
| `output_activation` | `null` | Output activation: null (trained with softmax, exports logits; the default), "sigmoid", "softmax" or "grouped_softmax" (requires label_groups). |
| `label_groups` | `{}` | Grouped softmax: a softmax within each group, with the groups independent of each other. Maps each group name to its member labels, and each group gains a synthetic "<group>_none" output column. Members of one group are mutually exclusive; members of different groups can be predicted together. Training labels not listed in any group are background: they get no output column and supply the "none" target for every group. |
| `export_logits` | `false` | Remove the output activation layer before export, so that the exported head emits raw logits while training still used output_activation (as BirdNET-Analyzer does with classifier.pop()). Has no effect when output_activation is null. Callers must apply the activation themselves. Rejected together with exclude_labels on a softmax or grouped head, where the excluded logits are needed to rebuild the softmax (output_activation null with exclude_labels is rejected for the same reason). |
| `normalize_embeddings` | `true` | Z-score normalisation of the input embeddings, fitted on the training set. |
| `focal_loss` | `false` | Use focal loss instead of cross-entropy. |
| `focal_loss_gamma` | `2.0` | Focal loss focusing parameter γ. |
| `focal_loss_alpha` | `0.25` | Focal loss class-balance parameter α. |
| `label_smoothing` | `false` | Apply label smoothing to the one-hot targets before training. |
| `label_smoothing_alpha` | `0.1` | Amount subtracted from positive labels and redistributed to the negatives. |
| `mixup` | `false` | Apply mixup data augmentation to the training embeddings. |
| `mixup_ratio` | `0.25` | Fraction of positive samples to mix. |
| `mixup_alpha` | `0.2` | Beta distribution parameter of the mixing coefficient. |
| `upsampling_ratio` | `0.0` | Upsampling ratio for minority classes before training. |
| `upsampling_mode` | `repeat` | Upsampling method: repeat (random duplication), mean (pairwise mean), linear (random interpolation) or smote (k-NN interpolation). |
| `seed` | `true` | Set the TensorFlow and NumPy random seeds from dataset.random_seed before training, for reproducibility. Set to false to disable; training results will then vary from run to run. |

### `training.sklearn` — scikit-learn head

*Only when `training.classifier` = `sklearn` / `both`.*

| Key | Default | Description |
|---|---|---|
| `C` | `1.0` | Inverse regularisation strength. |
| `max_iter` | `2000` | Maximum number of solver iterations. |
| `solver` | `lbfgs` | LogisticRegression solver algorithm. |

### `training.audio_mixup` — Audio mixup

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Mix the audio of training windows from different classes, embed the mixture and train on the union of their labels. Requires a Keras sigmoid head. |
| `labels` | `[]` | Labels whose windows may be mixed. Windows of any other label are never used as a source, although they still train as real samples. Empty means every label. |
| `n_mixes` | `null` | Number of mixes. Overrides ratio when set. |
| `ratio` | `0.5` | Number of mixes as a fraction of the real training windows. |
| `max_sources` | `2` | Maximum number of sources per mix: 2, or 3 (see p_three_sources). |
| `p_three_sources` | `0.0` | Proportion of three-source mixes when max_sources is 3. |
| `snr_db` | `[-6.0, 6.0]` | Level range [min, max] in dB of each added source, relative to the first. |
| `pairing` | `balanced` | balanced: select a class uniformly, then a clip from it; uniform: select clips uniformly. |
| `exclusive_groups` | `{}` | Groups of mutually exclusive labels (for example, call types or intensity levels of one species) that are never mixed with each other. |
| `background_labels` | `[]` | Labels that represent background rather than species. A mix contains at most one of them, and always at least one non-background source. |
| `background_target` | `drop` | drop: a background source adds no label to the mix target; include: it adds its own label. |
| `seed` | `null` | Seed for drawing the mixes. Defaults to dataset.random_seed. |
| `test_mix_ratio` | `0.0` | Also build mixes from test windows only, as a fraction of the test windows that may be mixed (0 means none). They are reported in a separate section and never pooled with the real test set. |

## `output`

Where results are written, in which formats and precisions.

| Key | Default | Description |
|---|---|---|
| `output_path` | `./outputs` | Parent directory for all output. |
| `model_name` | `custom_classifier` | Model name, used in filenames and for the output subdirectory. |
| `model_version` | `1.0` | Version string used in filenames. |
| `output_type` | `head` | What to export: head (the classifier only), full (backbone and classifier combined) or both. |
| `output_format` | `onnx` | File format of the exported model. |
| `exclude_labels` | `[]` | Labels to exclude from the exported classifier output. They are still used during training. Requires a Keras head that exports probabilities or per-class logits, so it is rejected with output_activation null, or with softmax and export_logits. |
| `export_dataset` | `false` | Export the chunked audio samples as WAV files in label subfolders. |
| `export_mixes` | `false` | Export only the audio mixes (training.audio_mixup) as WAV files in dataset/mixes/, for checking them when the real windows already exist. |
| `export_embeddings` | `false` | Save the computed embeddings for reuse. |
| `embeddings_format` | `npy` | Storage format for exported and cached embeddings: npy (one file per sample) or sqlite (a single, more portable .db file per run). |
| `embeddings_path` | `null` | Directory (npy) or file path (sqlite) for exported embeddings. Defaults to <output_dir>/embeddings or <output_dir>/embeddings.db. |
| `embeddings_overwrite` | `false` | Applies to `bioaccx embeddings` only. When an embedding store already exists at the resolved path, recompute and overwrite it. When false (the default), the existing store is reused: stored embeddings are never recomputed, and only the samples missing from it are computed and added. |
| `head_path` | `null` | Path to an existing ONNX classifier head for `bioaccx merge` (no training required). |
| `extract_from` | `null` | Path to a full BirdNET-Analyzer model (.tflite) for `bioaccx extract-head`. The trailing classifier head is read from the flatbuffer and re-exported as a head-only model, without backbone conversion or training. |
| `labels_file` | `null` | Optional class-label file for `bioaccx extract-head`, with one label per line. When omitted, a ``<model>_Labels.txt`` file next to the model is used if present. |
| `data_types` | `null` | Output precisions to export: a subset of {"FP32", "FP16", "INT8"}. When unset, defaults to [foundation_model.data_type]. Each precision produces a separate exported file, tagged with the precision in its filename. INT8 uses dynamic (weight-only) quantisation, with no calibration dataset. |

## `umap`

Projection settings for the embeddings command. Requires the [umap] extra.

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Fit the projection and write the CSV and plots. Requires the [umap] extra. |
| `n_neighbors` | `15` | Neighbourhood size: low values preserve local structure, high values global structure. |
| `min_dist` | `0.1` | How tightly points may be packed. |
| `n_components` | `2` | Number of dimensions of the projection. |
| `metric` | `euclidean` | Distance metric between embeddings. |
| `random_seed` | `null` | Random seed for UMAP. Falls back to dataset.random_seed when unset. |
| `cache_csv` | `null` | Path to a previously written UMAP data CSV. When set and the file exists, the embeddings run skips all computation (dataset loading, embedding extraction, KMeans and the UMAP fit) and only redraws the plots from the cached coordinates. This is useful for adjusting plot styling without recomputing. |

<!-- /generated:config-reference -->
