## Install

No Python needed: the installer sets everything up and adds a **bioaccx** shortcut that opens the GUI in your browser. Run it again to upgrade.

**Windows** — open PowerShell and paste:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | Out-String | iex"
```

**macOS (Apple Silicon) and Linux** — open a terminal and paste:

```sh
curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | sh
```

Already using Python? `uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"` or `pip install "bioaccx[cpu]"`. See the [installation guide](https://github.com/biodiversica/bioaccx/blob/master/docs/en/installation.md) ([português](https://github.com/biodiversica/bioaccx/blob/master/docs/pt-BR/installation.md)).

The download is about 2 GB, mostly TensorFlow. To uninstall, run the installer with `BIOACCX_UNINSTALL=1` (see [Uninstalling](https://github.com/biodiversica/bioaccx/blob/master/docs/en/installation.md#uninstalling)); your configs and trained models are kept.

## What's in this release

### Changes to check before upgrading

- **Keras heads are linear by default.** `training.keras.hidden_units` now defaults to 0: a single Dense layer on the embeddings (a linear probe) instead of a hidden layer of 256 units. A configuration that leaves `hidden_units` out now trains a smaller head than before; set `hidden_units: 256` to keep the old one.
- **SSH access is an extra.** `paramiko` is no longer installed by default, so a configuration that reads `data_dir` over SSH (`ssh_host`) needs `bioaccx[ssh]`. The installer does not add it.
- **The GUI listens on port 8770 by default**, instead of 8765. `bioaccx gui --port 8765` keeps the old address.
- **Dependencies:** the `umap` extra now requires umap-learn 0.6, asked to keep the 0.5 layout, so a projection with the same seed matches the earlier one. `typer` 0.29 is required. The `dev` extra is gone; working on bioaccx uses the `dev` dependency group, which `uv sync` installs.
- **Arbimon downloads are unavailable** while the `rfcx` SDK they rely on is broken. Local recordings, iNaturalist and Xeno-canto are not affected.

### New

- **A choice of clustering for the NMI.** `umap.cluster_method` picks `kmeans` (the default), `agglomerative`, `gmm` or `hdbscan`, with `umap.n_clusters` and `umap.min_cluster_size`. The NMI is shown on the GUI map and recorded in the dataset metadata.
- **Every run records its version.** Each command's output starts with `bioaccx <version>`, and reports and metadata record it as `bioaccx_version`.
- **Export the run log.** The GUI's Run tab has an **Export log** button, which saves the log as a text file named after the version, the command and the time.
- **Zoom and pan on the GUI's UMAP map**, which now fills its box with equal axes.
- **A clearer GUI form.** The file controls are in a bar above the form, with **Save as**. The dataset section is laid out as one source or several, each section is drawn as a card, every run setting is shown, and opt-in blocks come after the advanced fold. `output_activation` is a dropdown, and `label_groups` is under the advanced settings.
- **A logo** for the shortcuts and the browser tab, whose title is now bioaccx and its version.
- **Faster start-up.** `--help` and `--version` no longer load the pipeline, and the GUI server starts without the training stack.
- **The installers can uninstall** (`BIOACCX_UNINSTALL=1`), and say what they will do and ask before changing anything (`BIOACCX_YES=1` skips the question).
- **Documentation:** pages that exist in both languages link to each other under their title.

### Fixes

- The Windows install command works in Windows PowerShell 5.1, which ran the downloaded script line by line and failed on its first multi-line block.
- Syncing a checkout with `uv sync --extra gpu` no longer also installs the CPU ONNX runtime, which could replace the GPU one.
