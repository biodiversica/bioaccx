"""Unit tests for bioaccx.audio."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bioaccx.audio import AUDIO_EXTENSIONS, load_mono, to_fixed_length

SR = 16_000


def _write_wav(path: Path, n_samples: int, sr: int = SR, channels: int = 1) -> None:
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((n_samples, channels)).astype(np.float32)
    sf.write(str(path), audio.squeeze() if channels == 1 else audio, sr)


class TestToFixedLength:
    def test_exact_length_unchanged(self):
        a = np.ones(100, dtype=np.float32)
        result = to_fixed_length(a, 100)
        np.testing.assert_array_equal(result, a)

    def test_trims_longer_array(self):
        a = np.arange(200, dtype=np.float32)
        result = to_fixed_length(a, 100)
        assert len(result) == 100
        np.testing.assert_array_equal(result, a[:100])

    def test_pads_shorter_array(self):
        a = np.ones(50, dtype=np.float32)
        result = to_fixed_length(a, 100)
        assert len(result) == 100
        np.testing.assert_array_equal(result[:50], a)
        np.testing.assert_array_equal(result[50:], 0)

    def test_output_is_float32(self):
        a = np.ones(50, dtype=np.float64)
        result = to_fixed_length(a, 80)
        assert result.dtype == np.float32


class TestLoadMono:
    def test_loads_mono_file(self, tmp_path):
        wav = tmp_path / "mono.wav"
        _write_wav(wav, 16_000)
        audio = load_mono(wav, SR)
        assert audio.ndim == 1
        assert len(audio) == 16_000
        assert audio.dtype == np.float32

    def test_downmixes_stereo(self, tmp_path):
        wav = tmp_path / "stereo.wav"
        _write_wav(wav, 8_000, channels=2)
        audio = load_mono(wav, SR)
        assert audio.ndim == 1
        assert audio.dtype == np.float32

    def test_offset_and_duration(self, tmp_path):
        wav = tmp_path / "long.wav"
        _write_wav(wav, SR * 2)
        # Read the second half
        audio = load_mono(wav, SR, offset=1.0, duration=0.5)
        assert len(audio) == pytest.approx(SR * 0.5, abs=10)

    def test_resamples_to_target_rate(self, tmp_path):
        source_sr = 8_000
        target_sr = 16_000
        n = 8_000  # 1 second at 8 kHz
        wav = tmp_path / "low_sr.wav"
        rng = np.random.default_rng(1)
        sf.write(str(wav), rng.standard_normal(n).astype(np.float32), source_sr)

        audio = load_mono(wav, target_sr)
        # Output should be approximately 1 second at 16 kHz
        assert abs(len(audio) - target_sr) < target_sr * 0.05

    def test_output_always_float32(self, tmp_path):
        wav = tmp_path / "int16.wav"
        # soundfile writes int16 when subtype is PCM_16
        data = (np.random.default_rng(2).standard_normal(SR) * 32767).astype(np.int16)
        sf.write(str(wav), data, SR, subtype="PCM_16")
        audio = load_mono(wav, SR)
        assert audio.dtype == np.float32


class TestAudioExtensions:
    def test_common_extensions_present(self):
        for ext in {".wav", ".flac", ".mp3", ".ogg"}:
            assert ext in AUDIO_EXTENSIONS

    def test_frozenset_immutable(self):
        assert isinstance(AUDIO_EXTENSIONS, frozenset)
