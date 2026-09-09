# Embeddings, cache and UMAP

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

## Computing the embedding database + UMAP (no training)

Use `--embeddings` to run the foundation model over the dataset and build the **embedding database** without training or exporting a classifier. If the dataset has not been prepared yet, it is loaded and split first (the same loading, augmentation, and random-shift steps used by the full pipeline).

Embeddings are **always exported** in this mode (unlike the full pipeline, where export is opt-in via `output.export_embeddings`). The storage format follows `output.embeddings_format` — `sqlite` (default, a single `.db` file) or `npy` (one file per sample).

**Reusing an existing store:** if an embedding store already exists at the resolved path, it is **reused as-is and never recomputed** — the run reads the cached embeddings back and only (re)builds the UMAP outputs over them (or exits early if `umap.enabled` is off). To recompute from scratch instead, set `output.embeddings_overwrite: true`, which deletes the old store and re-extracts every embedding.

Optionally, this mode can also fit a [UMAP](https://umap-learn.readthedocs.io/) projection over the embeddings and write a data file and a plot. This is **off by default** and requires the optional `[umap]` extra:

```bash
pip install bioaccx[umap]      # or: uv sync --extra umap
```

Then enable it with `umap.enabled: true` in the config. If `umap.enabled` is set without the extra installed, the run errors with an install hint.

**Class separability (NMI):** every `bioaccx embeddings` run clusters the full embedding set with KMeans (`k` = the number of classes) and scores the agreement between those unsupervised clusters and the true labels with normalized mutual information — 0 = no agreement, 1 = perfect. This uses only scikit-learn (a core dependency), so it runs whether or not UMAP is enabled, and the score is printed in the log. It is skipped, with a note saying why, when there are fewer than two classes or fewer samples than classes. When UMAP is enabled the same cluster assignments become the `cluster` column of the UMAP CSV and a second figure, `[stem]_clusters.png`, with the NMI in a text box on the plot.

**Replotting without recomputing:** point `umap.cache_csv` at a UMAP CSV a previous run wrote and everything is skipped — dataset load, embedding extraction, KMeans, the UMAP fit — and only `[stem]_umap.png` and `[stem]_clusters.png` are redrawn from the cached coordinates (NMI is recomputed from the cached labels and cluster ids, which costs nothing). It is the way to tweak plot styling on a projection that took an hour to fit. If the path does not exist, the run warns and computes normally.

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
  cache_csv: null    # a previous [stem]_umap.csv — redraw the plots, compute nothing
```

**Run:**

```bash
bioaccx embeddings my_config.yaml
```

**Outputs** (written to `output_path/[stem]/`):

| File | Description |
|------|-------------|
| `[fm_id]_embeddings.db` (or `embeddings/`) | The embedding database — SQLite file, or a directory of `.npy` files |
| `[stem]_dataset_list.csv` | The list of samples used, with labels and train/test split |
| `[stem]_dataset_metadata.json` | How the dataset was built (sources, paths, parameters) and per-label sample counts |
| `[stem]_umap.csv` | UMAP coordinates per sample (`umap_1 … umap_N`, `label`, `split`, `key`, and `cluster` when NMI was computed) — only when `umap.enabled` |
| `[stem]_umap.png` | 2-D scatter-plot of the UMAP projection, coloured by label — only when `umap.enabled` |
| `[stem]_clusters.png` | The same projection coloured by KMeans cluster, annotated with the NMI score — only when `umap.enabled` and NMI was computed |

**Python API:**

```python
from bioaccx.config import load_config
from bioaccx.train import run_embeddings

cfg = load_config("my_config.yaml")
outputs = run_embeddings(cfg)
print(outputs["umap_plot"])
```
