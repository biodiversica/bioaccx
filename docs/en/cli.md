# CLI reference

Every command, argument and option, rendered from the Typer app itself — so it
is what `bioaccx --help` prints, not a copy of it.

```
bioaccx train CONFIG
bioaccx validate CONFIG
bioaccx dataset CONFIG [--no-split]
bioaccx embeddings CONFIG
bioaccx merge (CONFIG | HEAD --backbone ID|PATH) [-o PATH]
bioaccx extract-head (CONFIG | MODEL (--backbone ID|PATH | --embed-dim N))
                     [--labels PATH] [--format {onnx,tflite,both}] [-o DIR]
bioaccx convert-head HEAD [-o PATH]
bioaccx registry
bioaccx gui [CONFIG] [--models-dir DIR] [--host H] [--port N] [--lang CODE]
            [--token T | --no-auth] [--no-open]
```

`merge` and `extract-head` accept either a config file or the model file plus
its few inputs directly; which one you passed is read from the file suffix
(`.yaml` / `.yml` / `.json` = config, `.onnx` / `.tflite` = model).

## Commands

<!-- generated:cli — rendered from bioaccx/cli.py by tools/gen_docs.py -->

### `bioaccx train`

```
bioaccx train [OPTIONS] {CONFIG}
```

Run the full pipeline: dataset, embeddings, training, export, reports.

| Option | Default | Description |
|---|---|---|
| `CONFIG` |  | JSON or YAML config file. |

### `bioaccx validate`

```
bioaccx validate [OPTIONS] {CONFIG}
```

Parse and validate a config without running anything.

| Option | Default | Description |
|---|---|---|
| `CONFIG` |  | JSON or YAML config file. |

### `bioaccx dataset`

```
bioaccx dataset [OPTIONS] {CONFIG}
```

Load, split and export the dataset as chunked WAV files, without training.

| Option | Default | Description |
|---|---|---|
| `CONFIG` |  | JSON or YAML config file. |
| `--no-split` |  | Export every sample into dataset/<label>/ without a train/test split, ignoring test_ratio and any predefined split. The sample list CSV is written with an empty split column, ready to be filled in by hand and fed back as a label_mode: table source. |

### `bioaccx embeddings`

```
bioaccx embeddings [OPTIONS] {CONFIG}
```

Compute the embedding database (and UMAP, when enabled) without training. If the dataset has not been prepared it is loaded and split first. Embeddings are always exported — SQLite by default, or .npy per output.embeddings_format. When umap.enabled is set in the config (requires the [umap] extra), a UMAP projection is fitted and written as a data CSV plus a scatter-plot PNG.

| Option | Default | Description |
|---|---|---|
| `CONFIG` |  | JSON or YAML config file. |

### `bioaccx merge`

```
bioaccx merge [OPTIONS] {CONFIG|HEAD}
```

Merge a backbone and a classifier head into one full ONNX model. An ONNX backbone is combined with an ONNX or TFLite classifier head; a TFLite head is converted to ONNX automatically first. The backbone may be a local file or downloaded from HuggingFace. No dataset or training is involved.

| Option | Default | Description |
|---|---|---|
| `CONFIG|HEAD` |  | A config file carrying foundation_model + output.head_path, or a classifier head file to merge with --backbone. |
| `--backbone` |  | Foundation model for a head given directly: a registry ID such as 0xbb00 (downloaded if needed; see `bioaccx registry`) or a local backbone file. |
| `--embed-dim` |  | Embedding size, when it is not implied by --backbone. |
| `-o`, `--out` |  | Where the merged model is written. Defaults to a sibling of the head file. Ignored when the destination comes from a config. |

### `bioaccx extract-head`

```
bioaccx extract-head [OPTIONS] {CONFIG|MODEL}
```

Extract the classifier head from a full model and re-export it head-only. The head weights are read directly from the source graph — the backbone is never converted or run. No dataset or training is involved.

| Option | Default | Description |
|---|---|---|
| `CONFIG|MODEL` |  | A config file carrying output.extract_from, or a full model to extract from — a BirdNET-Analyzer .tflite or an .onnx backbone+head model. |
| `--backbone` |  | Foundation model for a model given directly: a registry ID such as 0xbb00 (see `bioaccx registry`) or a local backbone file. |
| `--embed-dim` |  | Embedding size, when it is not implied by --backbone. Marks the boundary between backbone and head (e.g. 1024 for BirdNET). |
| `--labels` |  | Class label file (one per line); defaults to a sibling *_Labels.txt. |
| `--format` | `HeadFormat.both` | Head formats to write. |
| `-o`, `--out` |  | Directory to write the extracted head into. Defaults to a sibling of the source file. Ignored when the destination comes from a config. |

### `bioaccx convert-head`

```
bioaccx convert-head [OPTIONS] {HEAD}
```

Convert a classifier head between ONNX and TFLite. The direction is taken from the file suffix and the result is verified against the source. No config file, backbone or training involved.

| Option | Default | Description |
|---|---|---|
| `HEAD` |  | The .onnx or .tflite classifier head to convert. |
| `-o`, `--out` |  | Where the converted head is written. Defaults to the source path with the other suffix. |

### `bioaccx registry`

```
bioaccx registry [OPTIONS]
```

List the registered foundation models and their default parameters.

### `bioaccx gui`

```
bioaccx gui [OPTIONS] [CONFIG]
```

Edit a config file in the browser. The editor writes the same config files the other commands read — the form is generated from the config schema, and the file it produces is shown as you edit. Training is still run from the terminal. Requires the [gui] extra: uv tool install "bioaccx[cpu,gui]"

| Option | Default | Description |
|---|---|---|
| `[CONFIG]` |  | Config file to open on start. Omitted, the editor starts from a template. |
| `--host` | `127.0.0.1` | Address to bind. Use 0.0.0.0 to reach the editor from another machine. |
| `--models-dir` | `custom_models` | Directory of trained models the results explorer reads. |
| `--port`, `-p` | `8765` | Port to listen on. |
| `--lang`, `-l` | `en` | Interface language the editor opens in. The picker in its toolbar lists the languages installed, and remembers a change per browser. |
| `--token` |  | Access token required by the editor. One is generated when binding a non-loopback address; use --no-auth to serve without it. |
| `--no-auth` |  | Serve without an access token. |
| `--open`, `--no-open` | `True` | Open the editor in a browser on start. |

<!-- /generated:cli -->

## The pre-subcommand form still works

Before subcommands every mode was a flag on one command. Those invocations are
translated automatically, so existing scripts keep running unchanged:

```bash
bioaccx my_config.yaml --dataset --no-split   # same as: bioaccx dataset my_config.yaml --no-split
bioaccx my_config.yaml                        # same as: bioaccx train my_config.yaml
bioaccx --registry                            # same as: bioaccx registry
```

New work should use the subcommand form — it is what `--help` documents, and
each command lists only the options that apply to it.
