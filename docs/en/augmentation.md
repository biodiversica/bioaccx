# Augmentation and windowing

## Noise augmentation

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

### Concatenated noise source

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

### Random noise selection

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

### Labels as noise sources

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

## Random sample shift

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
