# bioaccx

**BIOAcoustic Custom Classifier eXchange** — a Python CLI tool for training and sharing custom bioacoustic classifiers on top of pre-trained foundation models such as [BirdNET](https://birdnet.cornell.edu/) and [Perch](https://www.kaggle.com/models/google/bird-vocalization-classifier).

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

1. **Foundation model** — a backbone version of a bioacoustic model extracts rich feature vectors from raw audio windows. Supported formats: ONNX, TFLite, TF SavedModel (protobuf).
2. **Classifier head** — a small Keras MLP or sklearn LogisticRegression is trained on top of those embeddings using your labeled data.
3. **Export** — choose between:
   - **Full model** (`output_type: full`): merges backbone + head into a single ONNX or TFLite graph. Drop-in inference with no external dependencies.
   - **Head only** (`output_type: head`): exports just the classifier. Tiny file, easy to share; requires the backbone at inference time.
   - **Both** (`output_type: both`): produces both variants.

---

---

## Installation

bioaccx runs models through ONNX Runtime, and **there is no default one** — pick
`cpu` or `gpu` when you install. A plain `bioaccx` resolves and installs happily,
then fails the moment it tries to run a model, so the extra is not optional in
practice.

```bash
uv tool install "bioaccx[cpu]"              # CPU inference
uv tool install "bioaccx[gpu]"              # CUDA 12
uv tool install "bioaccx[cpu,gui,umap]"     # with the browser GUI and UMAP plots

uv add "bioaccx[cpu]"                       # or as a project dependency
```

Extras go **inside the brackets** — `uv tool install` has no `--extra` flag — and
the whole argument needs quoting so the shell does not glob them.

See [Installation](docs/en/installation.md) for the extras table, working on
bioaccx itself, GPU acceleration with CUDA, and how much disk this needs.

---

## Quick start

```bash
# 0. Browse available foundation models and their registry IDs
bioaccx registry

# 1. Copy and edit the example config
cp example_config.yaml my_config.yaml

# 2. Validate config without running training
bioaccx validate my_config.yaml

# 3. Export the dataset as chunked WAV files (no model needed)
bioaccx dataset my_config.yaml

# 4. Compute the embedding database + UMAP (no training)
bioaccx embeddings my_config.yaml

# 5. Train and export
bioaccx train my_config.yaml
```

`bioaccx merge` and `bioaccx extract-head` do the same jobs without training —
see [Merging, extracting and converting heads](docs/en/model-surgery.md).

---

## The browser GUI

`bioaccx gui` opens a config editor in a browser. It writes the same YAML files
every other command reads — the terminal stays the way runs are started.

```bash
uv tool install "bioaccx[cpu,gui]"      # the GUI needs the [gui] extra

bioaccx gui                             # start from a template
bioaccx gui my_config.yaml              # open an existing config
bioaccx gui --lang pt-BR                # open it in Portuguese
```

The form is generated from the config dataclasses, so it offers exactly the keys
this version understands, with their real defaults and help text; the file it
writes is shown beside it as you type, and your comments survive the round-trip.
The **run** tab starts a command on that config and streams its log into the
page; the **analysis** tab reads what past runs wrote — metrics, the UMAP map
with the audio behind each point, and two runs side by side.

See [The browser GUI](docs/en/gui.md) for all of it, including
[interface languages](docs/en/gui.md#interface-languages) and serving the editor
to another machine.

---

## Configuration

Everything a run does is described by one YAML file:
[`example_config.yaml`](example_config.yaml) is a working starting point,
[Configuration](docs/en/configuration.md) explains it section by section, and
[Config reference](docs/en/config-reference.md) lists every key with its default.

---

## Documentation

The full documentation lives in [`docs/en/`](docs/en/). This page covers what
bioaccx is, how to install it, and how to get a first run out of it.

| Page | What is in it |
|---|---|
| [Installation](docs/en/installation.md) | Install paths, the extras table, GPU acceleration, disk space |
| [Quick start](docs/en/quickstart.md) | The commands of a first run, in order |
| [The browser GUI](docs/en/gui.md) | `bioaccx gui` — config editor, results explorer, interface languages |
| [Configuration](docs/en/configuration.md) | The config file section by section, with worked examples |
| [Config reference](docs/en/config-reference.md) | Every config key, its default and its help text |
| [Datasets](docs/en/dataset-modes.md) | Label layouts, train/test split, appending, combining sources |
| [Remote sound sources](docs/en/remote-sources.md) | iNaturalist, Xeno-canto and Arbimon rows in one table |
| [Augmentation and windowing](docs/en/augmentation.md) | Noise augmentation, random shift, windows and overlap |
| [Embeddings, cache and UMAP](docs/en/embeddings-umap.md) | The embedding store, `bioaccx embeddings`, UMAP and KMeans/NMI |
| [Outputs](docs/en/outputs.md) | Directory structure, metadata, precision, excluding labels |
| [Merging, extracting and converting heads](docs/en/model-surgery.md) | `merge`, `extract-head`, `convert-head` |
| [Foundation model registry](docs/en/registry.md) | Registry IDs, supported models, adding a new one |
| [Python API](docs/en/python-api.md) | Using bioaccx as a library |
| [CLI reference](docs/en/cli.md) | Every command and option |

The config and CLI references are generated from the code by
`tools/gen_docs.py`, so they cannot drift behind it — see
[`docs/README.md`](docs/README.md).
