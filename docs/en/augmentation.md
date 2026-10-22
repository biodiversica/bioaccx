# Augmentation and windowing

## Noise augmentation

`dataset.augmentation` expands the training set by mixing each sample with background noise at one or more signal-to-noise ratios.

```yaml
dataset:
  augmentation:
    augmentation_dir: /path/to/noise_files   # WAV files used as noise sources
    snr_levels: [0, 10, 20]                  # dB (power-based)
    keep_original: true                      # also keep the unaugmented sample
    augment_test: false                      # default: the test set is not augmented
    concatenate_augmentation_dir: false      # see below
```

**How it works:**

- One augmented copy is produced for every combination of *(training sample, noise file, SNR level)*.
- A window-length chunk is extracted from the noise file at a **random start position** derived from a stable hash of `(random_seed, sample path, times, noise filename, SNR)`, so the results are fully reproducible across runs, regardless of iteration order.
- If the noise file is shorter than the input window of the model, the extracted chunk is zero-padded.
- Augmented samples are embedded by mixing the signal and the noise at inference time; no intermediate audio files are written unless `export_dataset: true` is also set.
- When `export_dataset: true` is set, augmented WAV files are written alongside the clean ones with a `_noise_<stem>_t<offset>_snr<value>` suffix in the filename. `<offset>` is the noise start position in seconds, so that several windows of the same noise file receive distinct names.
- For every run that uses augmentation, the dataset list CSV has three additional columns: `noise_file` (the path of the noise file), `snr_db` and `noise_start_time`. They are empty for clean samples.
- The augmentation parameters are recorded in the `_metadata.json` output file under an `"augmentation"` key.

**Dataset expansion factor:**

```
total train samples = clean_train × (N_noise_files × N_snr_levels + keep_original)
```

For example, 100 clean training samples with 3 noise files, the SNR levels `[0, 10, 20]` and `keep_original: true` give 100 × (3×3 + 1) = **1000 training samples**.

By default, `augment_test: false` keeps the test set clean for an unbiased evaluation. Setting it to `true` is appropriate only when the robustness of the model under noise is to be measured.

### Concatenated noise source

With `concatenate_augmentation_dir: true`, the entire contents of `augmentation_dir` are treated as a single noise source, instead of one noise file per augmented copy.

```yaml
dataset:
  augmentation:
    augmentation_dir: /path/to/noise_files
    snr_levels: [0, 10, 20]
    concatenate_augmentation_dir: true
```

- All audio files in `augmentation_dir` are scanned recursively.
- For each file, if an Audacity label file (`.txt`) with the same stem exists next to it, only the labelled segments are used, regardless of the label text. This makes it possible to mark clean background passages in long field recordings and exclude everything else.
- Files without a label file, or with an empty one, are included in full.
- All selected segments are concatenated into a single in-memory mono array at the sample rate of the model. No intermediate audio file is exported.
- This concatenated array is used as the only noise source, so the expansion factor becomes `clean_train × 1 × N_snr_levels` instead of `clean_train × N_files × N_snr_levels`.

### Random noise selection

With `random_augmentation_dir: true`, each augmented sample has a single noise condition, as with `concatenate_augmentation_dir`, but the noise is drawn from individual files rather than from a merged track. The noise files are shuffled once using `random_seed` and then assigned in rotation to each *(sample, SNR)* pair.

```yaml
dataset:
  augmentation:
    augmentation_dir: /path/to/noise_files
    snr_levels: [0, 10, 20]
    random_augmentation_dir: true
```

- The noise files in `augmentation_dir` are sorted and then shuffled with a single pass using `random_seed`.
- Each *(training sample, SNR level)* pair is assigned the next file in the shuffled list. When the list is exhausted, assignment restarts from the beginning, so no file is repeated before all the others have been used once.
- The noise start offset within the chosen file is still derived from a stable hash of `(random_seed, sample path, times, noise filename, SNR)`, so the results are fully reproducible across runs.
- The expansion factor is `clean_train × 1 × N_snr_levels`, as with the concatenated option, rather than `clean_train × N_files × N_snr_levels`.

**Choosing an option:**

| Option | Expansion factor | Noise source |
|---|---|---|
| default | `× N_files × N_SNR` | one copy per noise file |
| `concatenate_augmentation_dir: true` | `× N_SNR` | all files merged into one track |
| `random_augmentation_dir: true` | `× N_SNR` | one randomly assigned file per sample |

`random_augmentation_dir` is suited to large collections of noise files: it avoids the large dataset expansion of the default mode while still exposing the model to varied individual noise recordings rather than a single blended track.

### Labels as noise sources

Instead of, or in addition to, an external `augmentation_dir`, one or more **labels of the dataset being created** can be designated as noise sources with `augmentation_labels`:

```yaml
dataset:
  augmentation:
    augmentation_labels: [rain, wind]   # labels whose audio becomes noise
    augmentation_dir: /path/to/noise    # optional; combined with the labels
    snr_levels: [0, 10, 20]
```

- The audio of every sample with a listed label is added to the noise pool, **together with** any files in `augmentation_dir`. Either option, or both, may be set.
- These labels are **never augmented themselves** (they are treated like `skip_labels`), but they **remain trainable classes** in the dataset, and their clean samples are kept as usual.
- Labels are resolved **across the whole dataset**: when several `sources` are used, a noise label may belong to a different source from the one whose `augmentation` block references it (for example, a `noise` source folder supplying the augmentation of a target-species source).
- The noise pool is drawn per split: training augmentation uses the training samples of these labels, so test audio never leaks into the training noise.
- For `table` and `file_per_label` sources, only the annotated segment of each noise-label sample is used as a noise source; for `subfolders` sources, the whole file is used.
- All three modes apply. By default, each noise-label sample is one noise source (per-file expansion); `random_augmentation_dir` and `concatenate_augmentation_dir` reduce the pool, including the label audio, to a single condition per sample, exactly as with `augmentation_dir`.

This is useful when the dataset already contains background or non-target classes (rain, wind, traffic, silence) that can serve as realistic noise for the target classes, without the need to maintain a separate noise directory.

## Random sample shift

Short samples, whose audio is shorter than the foundation model window, are normally placed at the start of the padded window. `random_sample_shift: true` randomises this placement:

```yaml
dataset:
  random_sample_shift: true   # default: false
```

**How it works:**

- Each short sample is assigned a random integer offset in `[0, window_samples − signal_samples]` that determines where the signal starts within the zero-padded window.
- The offset is derived from a stable hash of `(random_seed, sample path, start/end times, noise_path, SNR)`, so the results are **fully reproducible** across runs.
- When augmentation is also active, each augmented copy of a sample receives a **different offset**, because the noise file and the SNR are part of the hash key.
- The dataset list CSV has a `signal_offset_samples` column when `random_sample_shift` is enabled, and a `signal_duration_seconds` column whenever a sample is shorter than the window. Both are empty for full-window samples.
- The shift is applied both at embedding time and when `export_dataset: true` is set.

## Windowing and overlap

In the `file_per_label` and `table` modes, each annotated segment is divided into fixed-length windows that match the input size of the foundation model.

- **Long segments** are divided into several windows, spaced `window × (1 − overlap)` apart.
- **Short segments** become a single window, zero-padded to the required length.

With `overlap: 0.5` and a 3 s window on a 9 s segment:

```
Window 1: 0.0 – 3.0 s
Window 2: 1.5 – 4.5 s
Window 3: 3.0 – 6.0 s
Window 4: 4.5 – 7.5 s
Window 5: 6.0 – 9.0 s
```
