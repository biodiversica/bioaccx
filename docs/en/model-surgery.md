# Merging, extracting and converting heads

## Merging a pre-existing head into a full model

`bioaccx merge` combines a backbone and a separately produced classifier head into a single full ONNX model, without training or loading a dataset. This is useful when a trained head already exists, for example from a previous `bioaccx` run or exported by another tool, and only needs to be combined with the backbone for deployment.

The backbone must be in ONNX format. The head can be in ONNX or TFLite format; a TFLite head is converted to ONNX automatically before merging.

The backbone can be loaded from a **local file** or downloaded from the **Hugging Face Hub**, using the same `foundation_model.source` field as for training.

**From the command line.** Merging requires only two inputs, so they are given as arguments and no configuration file is needed:

```bash
bioaccx merge my_head.tflite --backbone 0xbb00              # → my_head_full.onnx
bioaccx merge my_head.onnx --backbone ./birdnet.onnx -o full.onnx
```

`--backbone` accepts a [registry ID](registry.md#foundation-model-registry), which is downloaded and cached if necessary, or the path to a local backbone file. Without `-o`, the merged model is written next to the head as `<head>_full.onnx`. A local backbone keeps its own input tensor name; a registry backbone uses the name declared in the registry.

**From a configuration file**, when the merged model should be written to a versioned output directory together with other artefacts:

**Configuration (registry shorthand, backbone downloaded from Hugging Face):**

```yaml
foundation_model:
  registry_id: "0xbb00"   # BirdNET 2.4 ONNX; all defaults are filled automatically

output:
  output_path: ./merged_models
  model_name: my_classifier
  model_version: "1.0"
  head_path: /models/my_classifier_v1.0_keras_head.tflite   # or .onnx
```

**Configuration (registry shorthand, local backbone):**

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

The merged model is written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/[stem]_full.onnx`. `-o` applies only to the command-line form; with a configuration file, the destination is taken from the configuration.

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

Custom models trained with [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) are distributed as a single TFLite file that combines the BirdNET backbone with a small classifier head. `bioaccx extract-head` recovers that head and re-exports it as a lightweight head-only model, **without converting or running the backbone**. Full **ONNX** models are also supported (see [ONNX sources](#onnx-sources)).

**From the command line**, only the model and the embedding size are required:

```bash
bioaccx extract-head MyCustomModel.tflite --backbone 0xbb02      # → MyCustomModel_head/
bioaccx extract-head MyCustomModel.tflite --embed-dim 1024 \
        --format onnx --labels labels.txt -o ./heads
```

`--backbone` takes a [registry ID](registry.md#foundation-model-registry) whose embedding size marks where the head begins; `--embed-dim` gives the embedding size directly when the backbone is not in the registry. The source may be a `.tflite` or an `.onnx` file. Without `-o`, the files are written to a `<model>_head/` directory next to the source and named after it:

```
MyCustomModel_head/
  MyCustomModel_head.tflite            # bit-exact slice
  MyCustomModel_head_fp32.onnx
  MyCustomModel_labels.txt
  MyCustomModel_extract_head.json
```

`--format` selects which head files are written: `onnx`, `tflite` or `both` (the default in this form). `--labels` overrides the lookup of a `*_Labels.txt` file next to the model.

**From a configuration file**, which additionally provides `exclude_labels`, `data_types` and a versioned output directory:

```yaml
foundation_model:
  registry_id: "0xbb02"          # the backbone the head was trained on (defines embed_dim)
output:
  output_path: ./custom_models
  model_name: MyCustomModel
  model_version: "0.0"
  output_format: both            # onnx, tflite, or both
  extract_from: /path/to/MyCustomModel.tflite   # the full BirdNET-Analyzer model
  # labels_file: /path/to/labels.txt   # optional; defaults to the *_Labels.txt next to the model
```

```bash
bioaccx extract-head my_config.yaml
```

The head is then written to `output_path/[model_name]_[foundation_model_id]_v[model_version]/`.

**How it works:**

- The classifier head is the trailing chain of `FULLY_CONNECTED` operations that ends at the model output. The flatbuffer is read directly to recover the weights (`[out, in]`), bias and fused activation of each dense layer. The chain is followed backwards from the output until it reaches the layer whose input is the embedding (`embed_dim`), so backbone and front-end operations are never included. The backbone is never decoded.
- Both **single linear heads** (BirdNET-Analyzer with *Hidden units = 0*) and **multi-layer (MLP) heads** are supported.
- **TFLite head (bit-exact).** When `output_format` includes `tflite`, the head is produced by **slicing the original flatbuffer**: the head operators and their weight buffers are copied unchanged into a new single-input TFLite model rooted at the embedding tensor, with its input and output tensors renamed to `embedding` and `scores` (only the names change, not the values). It reproduces the head output of the source model *exactly* (a difference of `0.0` at the same precision, including quantised heads). It is written as `<stem>_head.tflite`, without a precision suffix, since it keeps the precision of the source.
- **ONNX head.** The sliced head is converted directly to ONNX with `tf2onnx`. A head consisting only of dense layers converts without difficulty, whereas the full model cannot be converted, because the backbone uses operations such as `RFFT2D` that ONNX does not support. Its input and output are then renamed to `embedding` and `scores`. The output precisions (`data_types`) and `exclude_labels` apply. It reproduces the original logits to within float32 rounding (about `1e-6`) and can be merged again with `bioaccx merge` and the matching ONNX backbone to rebuild a full single-file model.
- Class names are read from a `<model>_Labels.txt` file next to the model (BirdNET's `scientific_common` format, from which the common name is used) or from an explicit `labels_file`. The final `_labels.txt` and a small `_extract_head.json` file (source, backbone, classes and outputs) are written next to the head files.

### ONNX sources

A full ONNX model (backbone and head merged) can also be given to `bioaccx extract-head`:

```bash
bioaccx extract-head my_classifier_full.onnx --backbone 0xbb10
```

In this case there is no flatbuffer to slice, so the final part of the graph is read back into weights and rebuilt, using the same reader as `bioaccx convert-head`. This differs from the TFLite path in two respects:

- **The head is rebuilt rather than sliced.** Both the ONNX and the TFLite heads are written through the standard exporters, so `data_types` precisions and `exclude_labels` apply to both, and each file has a precision suffix (`_head_fp32.onnx`, `_head_fp32.tflite`). The outputs match the source to within float32 rounding rather than bit for bit.
- **The extraction is verified.** The source model is run once with its embedding tensor exposed, the rebuilt head receives that embedding, and the two outputs are compared:

  ```
  Recovered head: 1536 → 9  (1 dense layer(s), activations=['linear'])
  Source model emits 8 of 9 classes (its own exclude_labels filter) — kept in the extracted head.
  Verification: max output difference = 1.43e-06 (outputs up to 4.38e+00)  [OK]
  ```

The end of the backbone is located as in the TFLite path: the graph is followed backwards from the output to the first dense layer whose *input* size equals the embedding size. The embedding normalisation of the head is included and folded into that layer, while element-wise operations that belong to the backbone, such as a scalar mean-pooling divisor or 4-D BatchNorm channels, are excluded, so the extracted head still takes a genuine embedding as input. If the source model restricts its own output (because it was exported with `exclude_labels`), that filter is part of the head and is kept.

Labels are looked up as for TFLite sources, with one addition from the bioaccx convention: for `X_keras_full.onnx`, the `X_labels.txt` file written next to it by the training run is used automatically.

> For a model **trained with bioaccx**, extraction is usually unnecessary, because the run has already written `<stem>_keras_head*.onnx` next to the full model. Extraction is intended for full models from other sources, or for cases in which the head file is no longer available.

**Supported heads:** plain chains of dense layers, with `relu`, `relu6` or `tanh` activations and an optional final `sigmoid` or `softmax`. The bit-exact TFLite slice also supports quantised heads; the ONNX rebuild requires a float32 head. Heads with non-dense operations between the embedding and the output are rejected with an explicit error. `exclude_labels` and the `data_types` precision conversion apply to the ONNX head only; the TFLite slice keeps all source classes at the source precision in order to remain bit-exact.

---

## Converting a head between ONNX and TFLite

`bioaccx convert-head` converts an existing classifier head from one format to the other. It takes the
head file directly and requires **no configuration file, backbone, dataset or retraining**:

```bash
bioaccx convert-head my_classifier_head.tflite        # → my_classifier_head.onnx
bioaccx convert-head my_classifier_head.onnx          # → my_classifier_head.tflite
bioaccx convert-head head.onnx -o /elsewhere/head.tflite
```

The direction is determined by the file suffix, and the result is written next to the source with the
other suffix unless `-o` is given.

**Both directions preserve the model exactly.** The weights are copied unchanged, so the converted
head reproduces the outputs of the source to within float32 rounding. This includes the output
activation (`sigmoid` or `softmax`) and any `exclude_labels` filter, which remains *after* the
activation so that the scores are unchanged. Every conversion ends with a verification run on random
embeddings:

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

**How each direction works.** `tflite → onnx` passes the flatbuffer to tf2onnx, which converts a
graph of dense layers without difficulty. For `onnx → tflite`, no general converter is available, so
bioaccx reads the ONNX graph in the same way that `bioaccx extract-head` reads a TFLite flatbuffer:
it recovers the chain of dense layers (`MatMul`/`Gemm` + `Add`), their activations, the embedding
normalisation (folded into the first layer, which is exactly equivalent) and any output `Gather`,
rebuilds an equivalent Keras head and converts it. Any operation outside this set is rejected, and
the error names it, rather than a different model being produced without notice.

**Heads only.** A full model (backbone and head) is rejected in both directions, because the backbone
uses operations such as `RFFT2D` and `CONV_2D` that this path does not reproduce:

```
$ bioaccx convert-head my_classifier_full_fp32.tflite
bioaccx: error: 'my_classifier_full_fp32.tflite' is not a classifier head: it contains 379 ops
including AVERAGE_POOL_2D, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, EXPAND_DIMS….
Recover the head from the full model with `bioaccx extract-head` first.
```

**Quantised heads.** Converting the FP32 head is recommended, as it is the only exact case.

- An INT8 **TFLite** head can be converted to ONNX, but tf2onnx re-quantises the activations, so the
  outputs change slightly. The tool reports this, together with the size of the change
  (`[QUANTIZATION DRIFT]` rather than `[OK]`); for a 1024-dimensional head, differences of the order
  of `1e-1` on logits of about 10 are to be expected.
- An INT8 **ONNX** head cannot be converted to TFLite: its `DynamicQuantizeLinear` operations do not
  contain a dense chain that can be rebuilt, and the command stops with an error explaining this.

To combine a head with a backbone into a single-file model again, see
[`bioaccx merge`](#merging-a-pre-existing-head-into-a-full-model) above.
