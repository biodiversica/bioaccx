"""Unit tests for bioaccx.registry."""
from __future__ import annotations

import pytest

from bioaccx.registry import lookup_foundation_model_id


class TestLookupFoundationModelId:
    def test_birdnet_v24_fp32(self):
        assert lookup_foundation_model_id("birdnet", "2.4", "FP32") == "0xbb00"

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
