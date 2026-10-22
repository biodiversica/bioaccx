# bioaccx

**BIOAcoustic Custom Classifier eXchange** is a Python command-line tool for training and sharing custom bioacoustic classifiers built on pre-trained foundation models such as [BirdNET](https://birdnet.cornell.edu/) and [Perch](https://www.kaggle.com/models/google/bird-vocalization-classifier).

bioaccx is built on a clear separation between the **backbone** and the **classifier head**:

- The backbone (foundation model) is a large, general-purpose audio encoder shared by the community.
- The head is a small, task-specific classifier trained on the labelled data, typically only a few kilobytes in size.

This separation supports the following workflows:

**Sharing lightweight heads.** Because the head is very small compared with the backbone, a custom classifier can be published and distributed without the full model. Anyone who already has the same backbone can load the head and run inference immediately.

**Deploying as a single file.** When a self-contained model is required, for example on edge devices, in cloud APIs or in third-party tools, bioaccx can merge the backbone and the head into one ONNX or TFLite file with a single audio input and a class-score output.

**Reusing existing embeddings.** If the backbone has already been run and the resulting embedding vectors saved, bioaccx can read that cache and train or evaluate the head directly, without audio processing or GPU time.

**Building datasets from multiple sources in one configuration.** bioaccx can assemble training data from local recordings, [iNaturalist](https://www.inaturalist.org/) observations, [Xeno-canto](https://xeno-canto.org/) recordings and [Arbimon](https://arbimon.org/) projects, either listed in a single table or combined with a local directory. Remote audio is downloaded and cached automatically, so subsequent runs do not access the network.

bioaccx covers the complete pipeline, from raw audio to the exported model, and is driven by a single configuration file.

---

> 🇧🇷 [Leia em português](README.pt-BR.md)

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

1. **Foundation model:** the backbone of a bioacoustic model extracts feature vectors from raw audio windows. Supported formats are ONNX, TFLite and TF SavedModel (protobuf).
2. **Classifier head:** a small Keras MLP or a scikit-learn LogisticRegression is trained on those embeddings using the labelled data.
3. **Export:** one of the following is produced:
   - **Full model** (`output_type: full`): the backbone and the head merged into a single ONNX or TFLite graph, which runs without external dependencies.
   - **Head only** (`output_type: head`): the classifier alone. The file is small and easy to share, but the backbone is required at inference time.
   - **Both** (`output_type: both`): both variants.

---

## Installation

**Without Python.** The installer sets up everything required and adds a
**bioaccx** shortcut that opens the [browser GUI](#the-browser-gui). Running it
again upgrades the installation.

On Windows, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | iex"
```

On macOS (Apple Silicon) and Linux, open a terminal and run:

```sh
curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | sh
```

**With uv or pip.** bioaccx runs models through ONNX Runtime, and **no runtime
is installed by default**: choose `cpu` or `gpu` at installation. Installing
`bioaccx` without an extra succeeds, but the first attempt to run a model fails,
so in practice the extra is required.

```bash
uv tool install --python 3.12 "bioaccx[cpu]"            # CPU inference
uv tool install --python 3.12 "bioaccx[gpu]"            # CUDA 12
uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"   # with the browser GUI and UMAP plots

uv add "bioaccx[cpu]"                                   # or as a project dependency
```

Extras are given **inside the brackets**, since `uv tool install` has no
`--extra` option, and the whole argument must be quoted so that the shell does
not expand them. `--python` prevents uv from selecting a Python version newer
than TensorFlow supports.

[Installation](docs/en/installation.md) describes what the installer does, the
available extras, the supported systems, development setup, GPU acceleration
with CUDA and the disk space required.

---

## Quick start

```bash
# 0. Browse available foundation models and their registry IDs
bioaccx registry

# 1. Copy and edit the example config
cp example_config.yaml my_config.yaml

# 2. Validate the configuration without training
bioaccx validate my_config.yaml

# 3. Export the dataset as chunked WAV files (no model needed)
bioaccx dataset my_config.yaml

# 4. Compute the embedding database and the UMAP projection (no training)
bioaccx embeddings my_config.yaml

# 5. Train and export
bioaccx train my_config.yaml
```

`bioaccx merge` and `bioaccx extract-head` produce models without training; see
[Merging, extracting and converting heads](docs/en/model-surgery.md).

---

## The browser GUI

`bioaccx gui` opens a configuration editor in the browser. It writes the same
YAML files that every other command reads, and runs can still be started from
the terminal.

```bash
uv tool install "bioaccx[cpu,gui]"      # the GUI needs the [gui] extra

bioaccx gui                             # start from a template
bioaccx gui my_config.yaml              # open an existing configuration
bioaccx gui --lang pt-BR                # open it in Portuguese
```

The form is generated from the configuration dataclasses, so it offers exactly
the keys this version supports, with their actual defaults and help text. The
file being written is shown beside the form during editing, and comments in the
file are preserved. The **Run** tab starts a command on the configuration and
streams its log to the page. The **Analysis** tab presents the output of past
runs: metrics, the UMAP map with the audio behind each point, and side-by-side
comparisons of two runs.

[The browser GUI](docs/en/gui.md) covers the editor in full, including
[interface languages](docs/en/gui.md#interface-languages) and access from
another machine.

---

## Configuration

Each run is described entirely by one YAML file.
[`example_config.yaml`](example_config.yaml) is a working starting point,
[Configuration](docs/en/configuration.md) explains it section by section, and
the [Config reference](docs/en/config-reference.md) lists every key with its
default value.

---

## Documentation

The complete documentation is in [`docs/en/`](docs/en/). This page describes
what bioaccx is, how to install it and how to perform a first run.

| Page | Contents |
|---|---|
| [Installation](docs/en/installation.md) | Install paths, the extras table, GPU acceleration, disk space |
| [Quick start](docs/en/quickstart.md) | The commands of a first run, in order |
| [The browser GUI](docs/en/gui.md) | `bioaccx gui`: configuration editor, results explorer, interface languages |
| [Configuration](docs/en/configuration.md) | The configuration file section by section, with worked examples |
| [Config reference](docs/en/config-reference.md) | Every configuration key, with its default and description |
| [Datasets](docs/en/dataset-modes.md) | Label layouts, train/test split, appending, combining sources |
| [Remote sound sources](docs/en/remote-sources.md) | iNaturalist, Xeno-canto and Arbimon rows in one table |
| [Augmentation and windowing](docs/en/augmentation.md) | Noise augmentation, random shift, windows and overlap |
| [Embeddings, cache and UMAP](docs/en/embeddings-umap.md) | The embedding store, `bioaccx embeddings`, UMAP and KMeans/NMI |
| [Outputs](docs/en/outputs.md) | Directory structure, metadata, precision, excluding labels |
| [Merging, extracting and converting heads](docs/en/model-surgery.md) | `merge`, `extract-head`, `convert-head` |
| [Foundation model registry](docs/en/registry.md) | Registry IDs, supported models, adding a new one |
| [Python API](docs/en/python-api.md) | Using bioaccx as a library |
| [CLI reference](docs/en/cli.md) | Every command and option |

The configuration and CLI references are generated from the code by
`tools/gen_docs.py`, so they always match it; see
[`docs/README.md`](docs/README.md).

---

## License

[MIT](LICENSE)
