# Foundation model registry

Each foundation model has a compact 16-bit hex ID.  This ID is embedded in every output filename and embeddings cache filename so that classifiers are unambiguously traceable back to the exact backbone they were trained on.

The registry stores the full set of default parameters for each model — including source URL, audio config, and tensor names — so users can reference a model with a single `registry_id` field in the config instead of spelling out every parameter.

## Browsing the registry

```bash
bioaccx registry
```

This prints all registered models with their IDs, descriptions, HuggingFace source URLs, and audio parameters.  No config file needed.

## Using a registry ID in config

```yaml
foundation_model:
  registry_id: "0xbb00"   # loads all defaults for BirdNET 2.4 ONNX (not optimized)
```

Any field set alongside `registry_id` overrides the registry default.  Fields not set by the user are filled from the registry.  When `registry_id` is absent, bioaccx attempts an auto-lookup using the `name` / `version` / `data_type` / `format` fields; if those match a registry entry, defaults are applied the same way.

## Supported foundation models

| Registry ID | Model | Format | `sample_rate` | `window_seconds` | `embedding_size` | `input_name` | Notes |
|---|---|---|---|---|---|---|---|
| `0xbb00` | BirdNET 2.4 | ONNX | 48000 | 3.0 | 1024 | `INPUT` | Backbone, no classifier head (`model_backbone.onnx`) |
| `0xbb01` | BirdNET 2.4 | ONNX | 48000 | 3.0 | 1024 | `INPUT` | Backbone, optimized export (`birdnet_backbone.onnx`) |
| `0xbb02` | BirdNET 2.4 | TFLite | 48000 | 3.0 | 1024 | `INPUT` | Full model; `tflite_output_tensor_offset: -1` set automatically |
| `0xbb10` | Perch 2.0 | ONNX | 32000 | 5.0 | 1536 | `inputs` | Backbone with DFT front-end |
| `0xbb11` | Perch 2.0 | ONNX | 32000 | 5.0 | 1536 | `inputs` | Backbone without DFT front-end |
| `0xbb12` | Perch 2.0 | TFLite | 32000 | 5.0 | 1536 | `inputs` | Full model; embedding is the first output (`tflite_output_tensor_offset: 0`). Use when the deliverable must be a TFLite full model |

Any model that accepts a `[batch, samples]` float32 tensor and outputs an embedding vector is compatible.  Use `registry_id` to load a registered model with a single config line; run `bioaccx --registry` to see all registered models with their source URLs.

If no registry entry is found, the fallback ID `0xffff` is used and all model parameters must be set explicitly.

## Adding a new model

The registry lives in `bioaccx/registry.py`.  Add an entry to `_REGISTRY` keyed by its hex ID:

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

All keys except `description` must correspond to `FoundationModelConfig` field names.  When multiple entries share the same `(name, version, data_type, format)` tuple, the lowest hex ID is used for auto-lookup; the others must be referenced by explicit `registry_id`.
