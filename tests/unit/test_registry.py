"""Unit tests for bioaccx.registry."""
from __future__ import annotations

import pytest

from bioaccx.registry import (
    get_registry_defaults,
    list_registry_ids,
    lookup_foundation_model_id,
)


class TestLookupFoundationModelId:
    def test_birdnet_v24_fp32(self):
        assert lookup_foundation_model_id("birdnet", "2.4", "FP32") == "0xbb00"

    def test_birdnet_v24_fp32_tflite(self):
        assert lookup_foundation_model_id("birdnet", "2.4", "FP32", "tflite") == "0xbb02"

    def test_perch_v20_fp32(self):
        assert lookup_foundation_model_id("perch", "2.0", "FP32") == "0xbb10"

    def test_unknown_model_returns_fallback(self):
        assert lookup_foundation_model_id("unknown", "1.0", "FP32") == "0xffff"

    def test_unknown_version_returns_fallback(self):
        assert lookup_foundation_model_id("birdnet", "9.9", "FP32") == "0xffff"

    def test_unknown_data_type_returns_fallback(self):
        assert lookup_foundation_model_id("birdnet", "2.4", "INT8") == "0xffff"

    def test_name_is_case_insensitive(self):
        assert lookup_foundation_model_id("BirdNET", "2.4", "FP32") == "0xbb00"
        assert lookup_foundation_model_id("BIRDNET", "2.4", "FP32") == "0xbb00"

    def test_data_type_is_case_insensitive(self):
        assert lookup_foundation_model_id("birdnet", "2.4", "fp32") == "0xbb00"
        assert lookup_foundation_model_id("birdnet", "2.4", "Fp32") == "0xbb00"

    def test_return_value_is_lowercase_hex_string(self):
        result = lookup_foundation_model_id("birdnet", "2.4", "FP32")
        assert result.startswith("0x")
        assert result == result.lower()


class TestGetRegistryDefaults:
    def test_lookup_by_hex_string(self):
        d = get_registry_defaults("0xbb00")
        assert d is not None
        assert d["name"] == "birdnet"
        assert d["hf_repo"] == "biodiversica/BirdNET-onnx-backbone"

    def test_lookup_by_int(self):
        d = get_registry_defaults(0xBB00)
        assert d is not None
        assert d["name"] == "birdnet"

    def test_returns_copy(self):
        d1 = get_registry_defaults("0xbb00")
        d2 = get_registry_defaults("0xbb00")
        d1["name"] = "mutated"
        assert d2["name"] == "birdnet"

    def test_unknown_id_returns_none(self):
        assert get_registry_defaults("0xffff") is None
        assert get_registry_defaults(0xDEAD) is None

    def test_birdnet_onnx_defaults(self):
        d = get_registry_defaults("0xbb00")
        assert d["source"] == "huggingface"
        assert d["sample_rate"] == 48000
        assert d["window_seconds"] == 3.0
        assert d["embedding_size"] == 1024
        assert d["input_name"] == "INPUT"

    def test_birdnet_tflite_defaults(self):
        d = get_registry_defaults("0xbb02")
        assert d["format"] == "tflite"
        assert d["tflite_output_tensor_offset"] == -1

    def test_perch_onnx_defaults(self):
        d = get_registry_defaults("0xbb10")
        assert d["sample_rate"] == 32000
        assert d["window_seconds"] == 5.0
        assert d["embedding_size"] == 1536
        assert d["hf_repo"] == "biodiversica/Perch-onnx-backbone"


class TestListRegistryIds:
    def test_contains_known_ids(self):
        ids = list_registry_ids()
        assert "0xbb00" in ids
        assert "0xbb01" in ids
        assert "0xbb10" in ids

    def test_returns_lowercase_hex_strings(self):
        for rid in list_registry_ids():
            assert rid.startswith("0x")
            assert rid == rid.lower()
