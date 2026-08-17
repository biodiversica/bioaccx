"""Foundation model registry.

``_REGISTRY`` is keyed by the hex model ID (int).  Each entry holds all
default parameters needed to load and run the model without additional config,
including a default external source (HuggingFace where available).

Users reference a model by ``registry_id`` in the YAML config (e.g.
``registry_id: "0xbb00"``).  The config loader merges registry defaults with
any user-supplied fields; user-supplied values always win.

To register a new model add an entry to ``_REGISTRY``.  All value keys must
correspond to ``FoundationModelConfig`` field names.
"""
from __future__ import annotations

_REGISTRY: dict[int, dict] = {
    0xBB00: {
        "name": "birdnet",
        "version": "2.4",
        "data_type": "FP32",
        "format": "onnx",
        "description": "BirdNET 2.4 ONNX backbone (Justin Chu version without classifier head).",
        "source": "huggingface",
        "hf_repo": "biodiversica/BirdNET-onnx-backbone",
        "hf_filename": "model_backbone.onnx",
        "sample_rate": 48000,
        "window_seconds": 3.0,
        "input_name": "INPUT",
        "output_name": "embedding",
        "embedding_size": 1024,
    },
    0xBB01: {
        "name": "birdnet",
        "version": "2.4",
        "data_type": "FP32",
        "format": "onnx",
        "description": "BirdNET 2.4 ONNX backbone (Justin Chu OPTIMIZED version without classifier head).",
        "source": "huggingface",
        "hf_repo": "biodiversica/BirdNET-onnx-backbone",
        "hf_filename": "birdnet_backbone.onnx",
        "sample_rate": 48000,
        "window_seconds": 3.0,
        "input_name": "INPUT",
        "output_name": "embedding",
        "embedding_size": 1024,
    },
    0xBB02: {
        "name": "birdnet",
        "version": "2.4",
        "data_type": "FP32",
        "format": "tflite",
        "description": "BirdNET 2.4 TFLite full model (backbone + classifier head). Embedding extracted via tflite_output_tensor_offset=-1.",
        "source": "huggingface",
        "hf_repo": "justinchuby/BirdNET-onnx",
        "hf_filename": "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite",
        "sample_rate": 48000,
        "window_seconds": 3.0,
        "input_name": "INPUT",
        "embedding_size": 1024,
        "tflite_output_tensor_offset": -1,
    },
    0xBB10: {
        "name": "perch",
        "version": "2.0",
        "data_type": "FP32",
        "format": "onnx",
        "description": "Google Perch 2.0 ONNX backbone (Justin Chu version with DFT and without classifier head).",
        "source": "huggingface",
        "hf_repo": "biodiversica/Perch-onnx-backbone",
        "hf_filename": "perch_v2_backbone.onnx",
        "sample_rate": 32000,
        "window_seconds": 5.0,
        "input_name": "inputs",
        "output_name": "embedding",
        "embedding_size": 1536,
    },
    0xBB11: {
        "name": "perch",
        "version": "2.0",
        "data_type": "FP32",
        "format": "onnx",
        "description": "Google Perch 2.0 ONNX backbone (Justin Chu version without DFT and without classifier head)",
        "source": "huggingface",
        "hf_repo": "biodiversica/Perch-onnx-backbone",
        "hf_filename": "perch_v2_no_dft_backbone.onnx",
        "sample_rate": 32000,
        "window_seconds": 5.0,
        "input_name": "inputs",
        "output_name": "embedding",
        "embedding_size": 1536,
    },
    0xBB12: {
        "name": "perch",
        "version": "2.0",
        "data_type": "FP32",
        "format": "tflite",
        "description": "Google Perch 2.0 TFLite full model (backbone + 14795-class head, spatial embedding and spectrogram outputs). The embedding is the first output, so tflite_output_tensor_offset=0; the unused branches are trimmed at load and at full-model export.",
        "source": "huggingface",
        "hf_repo": "justinchuby/Perch-onnx",
        "hf_filename": "perch_v2.tflite",
        "sample_rate": 32000,
        "window_seconds": 5.0,
        "input_name": "inputs",
        "embedding_size": 1536,
        "tflite_output_tensor_offset": 0,
    },
}

_UNKNOWN_ID = 0xFFFF

# Reverse index: (name_lower, version, data_type_upper, format_lower) → hex id.
# When multiple entries share the same identity tuple the first registered one
# (lowest hex_id) is used for auto-lookup; others must be referenced explicitly
# by registry_id.
_KEY_INDEX: dict[tuple[str, str, str, str], int] = {}
for _hex_id, _e in _REGISTRY.items():
    _k = (_e["name"].lower(), _e["version"], _e["data_type"].upper(), _e["format"].lower())
    if _k not in _KEY_INDEX:
        _KEY_INDEX[_k] = _hex_id


def get_registry_defaults(registry_id: str | int) -> dict | None:
    """Return a copy of the model parameter dict for *registry_id*, or ``None``.

    *registry_id* may be an ``int`` or a hex string such as ``"0xbb00"``.

    >>> get_registry_defaults("0xbb00")["hf_repo"]
    'biodiversica/BirdNET-onnx-backbone'
    >>> get_registry_defaults(0xBB10)["embedding_size"]
    1536
    >>> get_registry_defaults("0xffff") is None
    True
    """
    key = int(registry_id, 16) if isinstance(registry_id, str) else registry_id
    entry = _REGISTRY.get(key)
    return dict(entry) if entry is not None else None


def print_registry() -> None:
    """Print all registered foundation models to stdout in a human-readable format."""
    header = "Available foundation models"
    print(header)
    print("=" * len(header))
    for hex_id, entry in _REGISTRY.items():
        rid = f"0x{hex_id:04x}"
        name = entry["name"].capitalize()
        version = entry["version"]
        fmt = entry["format"].upper()
        dtype = entry["data_type"]
        desc = entry.get("description", "")
        source = entry.get("source", "")
        hf_repo = entry.get("hf_repo", "")
        hf_filename = entry.get("hf_filename", "")
        sr = entry.get("sample_rate", "")
        win = entry.get("window_seconds", "")
        emb = entry.get("embedding_size", "")

        print(f"\n{rid}  {name} v{version} · {fmt} · {dtype}")
        if desc:
            print(f"  {desc}")
        if source == "huggingface" and hf_repo:
            url = f"https://huggingface.co/{hf_repo}"
            file_part = f" / {hf_filename}" if hf_filename else ""
            print(f"  Source : HuggingFace — {hf_repo}{file_part}")
            print(f"           {url}")
        elif source == "local":
            print(f"  Source : local file (path must be set in config)")
        if sr or win or emb:
            parts = []
            if sr:
                parts.append(f"{sr} Hz")
            if win:
                parts.append(f"{win} s window")
            if emb:
                parts.append(f"{emb}-dim embeddings")
            print(f"  Audio  : {' · '.join(parts)}")


def list_registry_ids() -> list[str]:
    """Return all registered model IDs as lowercase hex strings.

    >>> "0xbb00" in list_registry_ids()
    True
    """
    return [f"0x{k:04x}" for k in _REGISTRY]


def lookup_foundation_model_id(name: str, version: str, data_type: str, fmt: str = "onnx") -> str:
    """Return the hex ID string for a foundation model, or ``0xffff`` if unknown.

    >>> lookup_foundation_model_id("birdnet", "2.4", "FP32", "onnx")
    '0xbb00'
    >>> lookup_foundation_model_id("perch", "2.0", "FP32", "onnx")
    '0xbb10'
    >>> lookup_foundation_model_id("unknown", "1.0", "FP32", "onnx")
    '0xffff'
    """
    key = (name.lower(), version, data_type.upper(), fmt.lower())
    val = _KEY_INDEX.get(key, _UNKNOWN_ID)
    return f"0x{val:04x}"
