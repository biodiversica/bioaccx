# Foundation model registry

Each foundation model has a compact 16-bit hexadecimal ID. This ID is embedded in every output filename and embeddings cache filename, so that each classifier can be traced unambiguously to the exact backbone it was trained on.

The registry stores the full set of default parameters for each model, including the source URL, the audio settings and the tensor names, so that a model can be referenced with a single `registry_id` field in the configuration instead of specifying every parameter.

## Browsing the registry

```bash
bioaccx registry
```

This command lists all registered models with their IDs, descriptions, Hugging Face source URLs and audio parameters. No configuration file is required.

## Using a registry ID in the configuration

```yaml
foundation_model:
  registry_id: "0xbb00"   # loads all defaults for BirdNET 2.4 ONNX (not optimised)
```

Any field set alongside `registry_id` overrides the registry default, and fields that are not set are filled from the registry. When `registry_id` is absent, bioaccx attempts an automatic lookup using the `name`, `version`, `data_type` and `format` fields; if these match a registry entry, its defaults are applied in the same way.

## Supported foundation models

| Registry ID | Model | Format | `sample_rate` | `window_seconds` | `embedding_size` | `input_name` | Notes |
|---|---|---|---|---|---|---|---|
| `0xbb00` | BirdNET 2.4 | ONNX | 48000 | 3.0 | 1024 | `INPUT` | Backbone, no classifier head (`model_backbone.onnx`) |
| `0xbb01` | BirdNET 2.4 | ONNX | 48000 | 3.0 | 1024 | `INPUT` | Backbone, optimised export (`birdnet_backbone.onnx`) |
| `0xbb02` | BirdNET 2.4 | TFLite | 48000 | 3.0 | 1024 | `INPUT` | Backbone, no classifier head (`BirdNET_V2.4_Model_FP32_backbone.tflite`) |
| `0xbb10` | Perch 2.0 | ONNX | 32000 | 5.0 | 1536 | `inputs` | Backbone with DFT front-end |
| `0xbb11` | Perch 2.0 | ONNX | 32000 | 5.0 | 1536 | `inputs` | Backbone without DFT front-end |
| `0xbb12` | Perch 2.0 | TFLite | 32000 | 5.0 | 1536 | `inputs` | Backbone, no classifier head or auxiliary outputs (`perch_v2_backbone.tflite`) |

Any model that accepts a `[batch, samples]` float32 tensor and outputs an embedding vector is compatible. `registry_id` loads a registered model with a single configuration line, and `bioaccx registry` lists all registered models with their source URLs.

If no registry entry matches, the fallback ID `0xffff` is used and all model parameters must be set explicitly.

## Adding a new model

The registry is defined in `bioaccx/registry.py`. A new model is added as an entry in `_REGISTRY`, keyed by its hexadecimal ID:

```python
_REGISTRY: dict[int, dict] = {
    0xBB00: {
        "name": "birdnet",
        "version": "2.4",
        "data_type": "FP32",
        "format": "onnx",
        "description": "BirdNET 2.4 ONNX backbone without classifier head.",
        "source": "huggingface",
        "hf_repo": "biodiversica/BirdNET-onnx-backbone",
        "hf_filename": "model_backbone.onnx",
        "sample_rate": 48000,
        "window_seconds": 3.0,
        "input_name": "INPUT",
        "output_name": "embedding",
        "embedding_size": 1024,
    },
    # add new entries here ...
}
```

All keys except `description` must correspond to `FoundationModelConfig` field names. When several entries share the same `(name, version, data_type, format)` tuple, the lowest ID is used for the automatic lookup, and the others must be referenced by an explicit `registry_id`.
