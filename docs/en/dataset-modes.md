# Datasets

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

If the top-level contains `train/` and `test/` subdirectories, the predefined split is used (see [Manual train/test split](#manual-traintest-split)):

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
  split_col: split             # optional; values: train / test — see Manual train/test split
```

Example CSV:

```csv
filename,label,start_time,end_time,split
soundscapes/rec01.wav,crow,0.0,3.0,train
soundscapes/rec01.wav,robin,5.5,8.5,train
soundscapes/rec02.wav,crow,1.0,4.0,test
```

### Manual train/test split

When the dataset is already split by hand, that split can be declared in the config instead of
letting bioaccx generate one. Two of the three modes support it:

**`subfolders`** — put `train/` and `test/` at the top level, class folders inside each (see the
layout [above](#subfolders)):

```yaml
dataset:
  data_dir: /data/my_dataset   # contains train/<class>/ and test/<class>/
  label_mode: subfolders
```

**`table`** (and `ext_table_file`) — add a `split` column with `train` / `test` values:

```yaml
dataset:
  label_mode: table
  table_file: /data/annotations.csv
  split_col: split             # default column name
```

`file_per_label` has no split mechanism — its `.txt` rows carry only `start_time`, `end_time`
and `label`. Use `table` mode when you need a manual split with timed segments.

**How the split is resolved:**

| Samples carrying a `train`/`test` assignment | Result |
|---|---|
| All of them | Used exactly as given — `test_ratio` and `random_seed` have no effect on the split |
| None | Stratified auto-split using `test_ratio` and `random_seed` |
| Some | Assigned samples keep their split; only the unassigned ones are auto-split, then both are merged |

The mixed case is what makes it possible to grow a manually curated dataset: an
[appended dataset](#appending-to-an-existing-dataset) or a pre-split source can be combined with a
flat one in the same run, and only the new material is shuffled.

The console prints `Using predefined split: train=…, test=…` when a split was taken from the
dataset, and the per-label counts in `[stem]_dataset_metadata.json` reflect it.

> **Note:** the `subfolders` detection triggers when *either* `train/` or `test/` exists. A
> directory holding only `train/` therefore yields an empty test set rather than falling back to
> an auto-split — create both.

### Exporting without a split

The opposite case: you want the chunked samples grouped **only by label**, with no train/test
split at all — to review them, feed them to another tool, or do the split by hand. Add
`--no-split` to `--dataset`:

```bash
bioaccx dataset my_config.yaml --no-split
```

The flag is only accepted together with `--dataset` (there is no such thing as training without a
test set), and it overrides everything else about splitting: `test_ratio` is ignored, and so is
any split the source already defines (`train/`+`test/` folders, or a `split` column).

```
output_path/[stem]/
  dataset/
    crow/
      rec1_0.000_3.000.wav
    robin/
      song_a_0.000_3.000.wav
  [stem]_dataset_list.csv         # split column present but empty
  [stem]_dataset_metadata.json    # "split": "none", one count per label
```

Everything else behaves as in a normal `--dataset` run: windowing, overlap, filter/speed
preprocessing and augmentation all still apply (augmentation covers every sample, since there is
no test set to hold back).

**Round-tripping a manual split.** The emitted `[stem]_dataset_list.csv` already has the columns
`table` mode expects, with the `split` column left blank. Fill in `train` / `test` where you want
them and feed the same file back:

```yaml
dataset:
  label_mode: table
  table_file: /path/to/[stem]_dataset_list.csv
  filename_col: filepath        # the CSV's path column
  split_col: split
```

Rows you left blank are stratified auto-split with `test_ratio`, so a partly filled column also
works — pin down the samples you care about and let bioaccx place the rest.

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

**Multi-label windows in an export:** a window annotated with several classes is written to each of its label folders under one name ending in `_ml<hash>` (a hash of its source file and time bounds). Loading the export back — as `append_dataset_path` or as a `subfolders` `data_dir` — merges the copies into one multi-label window again. Files that merely share a name across label folders carry no tag and stay separate. Audio mixes go to `dataset/mixes/`, which is never read back.

**Split assignment:** existing samples keep their original `train`/`test` assignments. New samples are stratified auto-split using `test_ratio` and `random_seed`.

**Typical workflow:**

```bash
# First run — train on initial data and export the dataset
bioaccx train config_v1.yaml   # with export_dataset: true

# Later — add new recordings and retrain on the full merged set
bioaccx train config_v2.yaml   # with append_dataset_path pointing to the first export
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
