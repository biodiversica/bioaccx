"""Foundation model ID registry.

Maps (name, version, data_type, format) tuples to compact hex IDs used in
output filenames and model interchange.  Keys are normalised: name is
lower-cased, data_type is upper-cased, format is lower-cased, version is
compared as-is.

To register a new model, add an entry to ``_REGISTRY``.
"""
from __future__ import annotations

_REGISTRY: dict[tuple[str, str, str, str], int] = {
    ("birdnet", "2.4", "FP32", "onnx"): 0xBB00,
    ("perch",   "2.0", "FP32", "onnx"): 0xBB10,
}

_UNKNOWN_ID = 0xFFFF


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
    val = _REGISTRY.get(key, _UNKNOWN_ID)
    return f"0x{val:04x}"
