# Merging, extracting and converting heads

## Merging a pre-existing head into a full model

Use `--merge` to combine a backbone and a separately-produced classifier head into a single full ONNX model — without running training or loading a dataset. This is useful when you already have a trained head (e.g. produced by a previous `bioaccx` run or exported by another tool) and just want to bundle it with the backbone for deployment.

The backbone must be in ONNX format. The head can be either ONNX or TFLite; a TFLite head is automatically converted to ONNX before merging.

The backbone can be loaded from a **local file** or downloaded from **HuggingFace Hub** — the same `foundation_model.source` field used for training.

**Straight from the command line.** Merging needs only two inputs, so it takes them as arguments — no config file:

```bash
bioaccx merge my_head.tflite --backbone 0xbb00              # → my_head_full.onnx
bioaccx merge my_head.onnx --backbone ./birdnet.onnx -o full.onnx
```

`--backbone` accepts a [registry ID](registry.md#foundation-model-registry) (downloaded and cached if needed) or a path to a local backbone file. Without `-o` the merged model is written next to the head as `<head>_full.onnx`. A local backbone keeps its own input tensor name; a registry backbone uses the name the registry declares.

**Or from a config**, when you want the merged model to land in a versioned output directory alongside your other artifacts:

**Config (registry shorthand — downloads backbone from HuggingFace):**

```yaml
foundation_model:
  registry_id: "0xbb00"   # BirdNET 2.4 ONNX; all defaults filled automatically

output:
  output_path: ./merged_models
  model_name: my_classifier
  model_version: "1.0"
  head_path: /models/my_classifier_v1.0_keras_head.tflite   # or .onnx
```

**Config (registry shorthand — local backbone):**

```yaml
foundation_model:
  registry_id: "0xbb00"
  source: local
  path: /models/birdnet_backbone.onnx

output:
  output_path: ./merged_models
  model_name: my_classifier
  model_version: "1.0"
  head_path: /models/my_classifier_v1.0_keras_head.tflite   # or .onnx
```

**Run:**

```bash
bioaccx merge my_config.yaml
```

The merged model is written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/[stem]_full.onnx`. (`-o` applies to the command-line form only — with a config the destination comes from the config.)

**Python API:**

```python
from bioaccx.config import load_config
from bioaccx.train import run_merge

cfg = load_config("my_config.yaml")
out_path = run_merge(cfg)                      # or run_merge(cfg, out_path=Path("full.onnx"))
print(out_path)
```

---

## Extracting a head from a full model

Custom models trained with [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) are distributed as a single TFLite file bundling the BirdNET backbone with a small classifier head. `--extract-head` recovers just that head and re-exports it as a lightweight head-only model — **without converting or running the backbone**. Full **ONNX** models work too (see [ONNX sources](#onnx-sources)).

**Straight from the command line** — the model plus the embedding size are all it needs:

```bash
bioaccx extract-head MyCustomModel.tflite --backbone 0xbb02      # → MyCustomModel_head/
bioaccx extract-head MyCustomModel.tflite --embed-dim 1024 \
        --format onnx --labels labels.txt -o ./heads
```

`--backbone` takes a [registry ID](registry.md#foundation-model-registry) whose embedding size marks where the head starts; `--embed-dim` states it directly when the backbone is not in the registry. The source may be `.tflite` or `.onnx`. Without `-o` the files land in a `<model>_head/` directory next to the source, named after it:

```
MyCustomModel_head/
  MyCustomModel_head.tflite            # bit-exact slice
  MyCustomModel_head_fp32.onnx
  MyCustomModel_labels.txt
  MyCustomModel_extract_head.json
```

`--format` selects which head files to write (`onnx`, `tflite`, or `both` — the default here). `--labels` overrides the sibling `*_Labels.txt` lookup.

**Or from a config**, which additionally gives you `exclude_labels`, `data_types`, and a versioned output directory:

```yaml
foundation_model:
  registry_id: "0xbb02"          # the backbone the head was trained on (defines embed_dim)
output:
  output_path: ./custom_models
  model_name: MyCustomModel
  model_version: "0.0"
  output_format: both            # onnx, tflite, or both
  extract_from: /path/to/MyCustomModel.tflite   # the full BirdNET-Analyzer model
  # labels_file: /path/to/labels.txt   # optional; defaults to the sibling *_Labels.txt
```

```bash
bioaccx extract-head my_config.yaml
```

The head is then written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/`.

**How it works:**

- The classifier head is the trailing chain of `FULLY_CONNECTED` ops ending at the model output. The flatbuffer is read directly to recover each dense layer's weight (`[out, in]`), bias, and fused activation; the chain is walked back from the output until it reaches the layer whose input is the embedding (`embed_dim`), so backbone/frontend ops are never included. The backbone is never decoded.
- Both **single linear heads** (BirdNET-Analyzer *Hidden units = 0*) and **multi-layer (MLP) heads** are supported.
- **TFLite head — bit-exact.** When `output_format` includes `tflite`, the head is produced by **slicing the original flatbuffer**: the head operators and their weight buffers are copied verbatim into a new single-input TFLite model rooted at the embedding tensor, with its I/O tensors renamed to `embedding` → `scores` (labels only — values unchanged). It reproduces the source model's head output *exactly* (`0.0` difference, same precision — quantized heads included). It is written as `<stem>_head.tflite` (no precision suffix — it inherits the source precision).
- **ONNX head.** The sliced head is converted directly to ONNX with `tf2onnx` (a dense-only head converts cleanly — the full model can't, because the backbone uses ops like `RFFT2D` that ONNX lacks), then its I/O is renamed to `embedding` → `scores`. Output precisions (`data_types`) and `exclude_labels` apply. It reproduces the original logits to within float32 rounding (~`1e-6`) and can be re-`--merge`d with the matching ONNX backbone to rebuild a full single-file model.
- Class names are read from a sibling `<model>_Labels.txt` (BirdNET's `scientific_common` format → common name) or from an explicit `labels_file`. The final `_labels.txt` and a small `_extract_head.json` (source, backbone, classes, outputs) are written alongside the head(s).

### ONNX sources

A full ONNX model (backbone + head merged) can be given to `--extract-head` as well:

```bash
bioaccx extract-head my_classifier_full.onnx --backbone 0xbb10
```

There is no flatbuffer to slice here, so the graph's tail is read back into weights and rebuilt — the same reader `--convert-head` uses. Two things differ from the TFLite path:

- **The head is rebuilt, not sliced.** Both the ONNX and TFLite heads are written through the normal exporters, so `data_types` precisions and `exclude_labels` apply to both, and each file carries a precision suffix (`_head_fp32.onnx`, `_head_fp32.tflite`). Outputs match the source to within float32 rounding rather than bit-exactly.
- **The extraction is verified.** The source model is run once with its embedding tensor exposed, the rebuilt head is fed that embedding, and the two outputs are compared:

  ```
  Recovered head: 1536 → 9  (1 dense layer(s), activations=['linear'])
  Source model emits 8 of 9 classes (its own exclude_labels filter) — kept in the extracted head.
  Verification: max output difference = 1.43e-06 (outputs up to 4.38e+00)  [OK]
  ```

Finding where the backbone ends works like the TFLite version: the graph is walked back from the output to the first dense layer whose *input* size is the embedding size. The head's own embedding normalization is pulled in with it (folded into that layer), while elementwise ops belonging to the backbone — a scalar mean-pool divisor, 4-D BatchNorm channels — are left behind, so the extracted head still takes a genuine embedding as input. If the source model restricts its own output (it was exported with `exclude_labels`), that filter is part of the head and is kept.

Labels are looked up as for TFLite sources, plus the bioaccx convention: for `X_keras_full.onnx` the sibling `X_labels.txt` written by the training run is picked up automatically.

> For a model **bioaccx trained**, extraction is usually unnecessary — the run already wrote `<stem>_keras_head*.onnx` next to the full model. Extraction is for full models from elsewhere, or when the head file has been lost.

**Supported heads:** plain dense chains (with `relu`/`relu6`/`tanh` activations and an optional trailing `sigmoid`/`softmax`). The bit-exact TFLite slice also handles quantized heads; the ONNX rebuild requires a float32 head. Heads with non-dense ops between the embedding and the output raise a clear error. `exclude_labels` and `data_types` precision conversion apply to the ONNX head only — the TFLite slice keeps all source classes at the source precision to stay bit-exact.

---

## Converting a head between ONNX and TFLite

`--convert-head` converts an existing classifier head from one format to the other. It takes the
head file directly — **no config file, no backbone, no dataset, no retraining**:

```bash
bioaccx convert-head my_classifier_head.tflite        # → my_classifier_head.onnx
bioaccx convert-head my_classifier_head.onnx          # → my_classifier_head.tflite
bioaccx convert-head head.onnx -o /elsewhere/head.tflite
```

The direction comes from the suffix, and the result is written next to the source with the other
suffix unless `-o` is given.

**Both directions preserve the model exactly.** The weights are copied verbatim, so the converted
head reproduces the source's outputs to within float32 rounding — including the output activation
(`sigmoid` / `softmax`) and any `exclude_labels` filter, which stays *after* the activation so the
scores are unchanged. Every conversion ends with a verification run on random embeddings:

```
bioaccx — convert classifier head (onnx → tflite)
Source : my_classifier_head.onnx
Output : my_classifier_head.tflite
==============================================================

  Recovered head: 1024 → 7  (1 dense layer(s), activations=['linear'])
  Output filter: 6 of 7 classes kept (applied after the activation)
  Keras TFLite head → my_classifier_head.tflite (FP32)
  Verification: max output difference = 1.62e-05 (outputs up to 1.33e+01)  [OK]
```

**How each direction works.** `tflite → onnx` hands the flatbuffer to tf2onnx — a dense-only graph
converts cleanly. `onnx → tflite` has no general converter available, so bioaccx reads the ONNX
graph the same way `--extract_head` reads a TFLite flatbuffer: it recovers the dense chain
(`MatMul`/`Gemm` + `Add`), its activations, the embedding normalization (folded into the first
layer — exactly equivalent) and any output `Gather`, rebuilds an equivalent Keras head and converts
that. Anything outside this op vocabulary is refused with the offending op named, rather than
silently producing a different model.

**Heads only.** A full model (backbone + head) is rejected in both directions — the backbone uses
ops such as `RFFT2D` and `CONV_2D` that this path does not reproduce:

```
$ bioaccx --convert-head my_classifier_full_fp32.tflite
bioaccx: error: 'my_classifier_full_fp32.tflite' is not a classifier head: it contains 379 ops
including AVERAGE_POOL_2D, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, EXPAND_DIMS….
Recover the head from a full model with --extract_head first.
```

**Quantized heads.** Prefer converting the FP32 head — that is the only case that is exact.

- An INT8 **TFLite** head does convert to ONNX, but tf2onnx re-quantizes the activations, so the
  outputs shift slightly. The tool says so and reports the size of the shift
  (`[QUANTIZATION DRIFT]` rather than `[OK]`); on a 1024-dim head expect on the order of `1e-1` on
  logits of ~10.
- An INT8 **ONNX** head cannot be converted to TFLite at all — its `DynamicQuantizeLinear` ops
  carry no rebuildable dense chain, and the command refuses with that explanation.

To bundle a head back into a single-file model, see
[`--merge`](#merging-a-pre-existing-head-into-a-full-model) below.
