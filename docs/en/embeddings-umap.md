# Embeddings, cache and UMAP

## Embedding cache

Computing embeddings is the slowest step of the pipeline. `embeddings_cache_path` stores them so that they can be reused across runs.

### SQLite cache (default)

```yaml
dataset:
  embeddings_cache_path: /path/to/cache/0xbb00_embeddings.db
  embeddings_format: sqlite
```

All embeddings for a run are stored in a single `.db` file. By convention, the filename encodes the registry ID of the foundation model (for example `0xbb00_embeddings.db`); if the ID in the filename does not match the configured backbone, bioaccx issues a warning and discards the cache.

### NPY cache

```yaml
dataset:
  embeddings_cache_path: /path/to/cache_dir
  embeddings_format: npy
```

Each embedding is stored as an individual `.npy` file. The filenames encode the audio file stem, the start time and the end time, so different chunking or overlap settings produce separate cache entries.

### Exporting embeddings

The computed embeddings can also be saved as a pipeline output, in addition to or instead of caching them:

```yaml
output:
  export_embeddings: true
  embeddings_format: sqlite   # sqlite | npy
  # embeddings_path: /custom/export/path   # default: output_dir
```

---

## Computing the embedding database and UMAP (no training)

`bioaccx embeddings` runs the foundation model over the dataset and builds the **embedding database** without training or exporting a classifier. If the dataset has not yet been prepared, it is loaded and split first, using the same loading, augmentation and random-shift steps as the full pipeline.

In this mode, embeddings are **always exported**, whereas in the full pipeline export must be enabled with `output.export_embeddings`. The storage format is set by `output.embeddings_format`: `sqlite` (the default, a single `.db` file) or `npy` (one file per sample).

**Reusing an existing store:** if an embedding store already exists at the resolved path, it is **reused and never recomputed**. The run reads the stored embeddings and computes only the samples missing from the store, adding them to it, so that after the run the store covers the whole dataset. This makes it possible to complete a store built from part of a dataset. To recompute everything instead, set `output.embeddings_overwrite: true`, which deletes the existing store and extracts every embedding again.

This mode can also fit a [UMAP](https://umap-learn.readthedocs.io/) projection of the embeddings and write a data file and a plot. This is **disabled by default** and requires the optional `[umap]` extra:

```bash
pip install bioaccx[umap]      # or: uv sync --extra umap
```

It is then enabled with `umap.enabled: true` in the configuration. If `umap.enabled` is set without the extra installed, the run stops with an error explaining how to install it.

**Class separability (NMI):** every `bioaccx embeddings` run clusters the full embedding set with KMeans (with `k` equal to the number of classes) and measures the agreement between these unsupervised clusters and the true labels with normalised mutual information, where 0 indicates no agreement and 1 perfect agreement. This step uses only scikit-learn, a core dependency, so it runs whether or not UMAP is enabled, and the score is printed in the log. It is skipped, with a note giving the reason, when there are fewer than two classes or fewer samples than classes. When UMAP is enabled, the same cluster assignments form the `cluster` column of the UMAP CSV and a second figure, `[stem]_clusters.png`, which shows the NMI in a text box.

**Replotting without recomputing:** when `umap.cache_csv` points to a UMAP CSV written by a previous run, all computation is skipped (dataset loading, embedding extraction, KMeans and the UMAP fit), and only `[stem]_umap.png` and `[stem]_clusters.png` are redrawn from the cached coordinates. The NMI is recomputed from the cached labels and cluster IDs at negligible cost. This allows the plot styling of a projection that took a long time to fit to be adjusted. If the path does not exist, the run issues a warning and computes normally.

**Configuration** (the `umap` section is optional, and UMAP is disabled unless `enabled: true` is set):

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

umap:                # optional; requires the [umap] extra
  enabled: true      # disabled by default
  n_neighbors: 15
  min_dist: 0.1
  n_components: 2
  metric: euclidean
  random_seed: null  # falls back to dataset.random_seed
  cache_csv: null    # a previous [stem]_umap.csv: redraw the plots without computing
```

**Run:**

```bash
bioaccx embeddings my_config.yaml
```

**Outputs** (written to `output_path/[stem]/`):

| File | Description |
|------|-------------|
| `[fm_id]_embeddings.db` (or `embeddings/`) | The embedding database: an SQLite file, or a directory of `.npy` files |
| `[stem]_dataset_list.csv` | The list of samples used, with labels and train/test split |
| `[stem]_dataset_metadata.json` | How the dataset was built (sources, paths, parameters) and per-label sample counts |
| `[stem]_umap.csv` | UMAP coordinates per sample (`umap_1 … umap_N`, `label`, `split`, `key`, and `cluster` when the NMI was computed); only when `umap.enabled` is set |
| `[stem]_umap.png` | 2-D scatter plot of the UMAP projection, coloured by label; only when `umap.enabled` is set |
| `[stem]_clusters.png` | The same projection coloured by KMeans cluster and annotated with the NMI score; only when `umap.enabled` is set and the NMI was computed |

**Python API:**

```python
from bioaccx.config import load_config
from bioaccx.train import run_embeddings

cfg = load_config("my_config.yaml")
outputs = run_embeddings(cfg)
print(outputs["umap_plot"])
```
