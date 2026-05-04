"""Unit tests for bioaccx.audio."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bioaccx.audio import AUDIO_EXTENSIONS, apply_filter, apply_speed, load_mono, to_fixed_length

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


# ---------------------------------------------------------------------------
# Helpers for spectral tests
# ---------------------------------------------------------------------------

def _tone(freq_hz: float, duration: float = 1.0, sr: int = SR) -> np.ndarray:
    """Generate a pure sine wave at freq_hz."""
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


# ---------------------------------------------------------------------------
# apply_filter
# ---------------------------------------------------------------------------

class TestApplyFilter:
    def test_output_shape_unchanged(self):
        audio = _tone(1000.0)
        out = apply_filter(audio, SR, "hpf", 4000.0)
        assert out.shape == audio.shape

    def test_output_dtype_float32(self):
        audio = _tone(1000.0).astype(np.float64)
        out = apply_filter(audio, SR, "hpf", 4000.0)
        assert out.dtype == np.float32

    def test_hpf_attenuates_low_frequency(self):
        # Cutoff at 4000 Hz; 500 Hz tone should be heavily attenuated
        audio = _tone(500.0)
        out = apply_filter(audio, SR, "hpf", 4000.0)
        assert _rms(out) < _rms(audio) * 0.1

    def test_hpf_passes_high_frequency(self):
        # Cutoff at 1000 Hz; 6000 Hz tone should pass nearly unchanged
        audio = _tone(6000.0)
        out = apply_filter(audio, SR, "hpf", 1000.0)
        assert _rms(out) > _rms(audio) * 0.9

    def test_lpf_attenuates_high_frequency(self):
        # Cutoff at 2000 Hz; 7000 Hz tone should be heavily attenuated
        audio = _tone(7000.0)
        out = apply_filter(audio, SR, "lpf", 2000.0)
        assert _rms(out) < _rms(audio) * 0.1

    def test_lpf_passes_low_frequency(self):
        # Cutoff at 4000 Hz; 500 Hz tone should pass nearly unchanged
        audio = _tone(500.0)
        out = apply_filter(audio, SR, "lpf", 4000.0)
        assert _rms(out) > _rms(audio) * 0.9

    def test_bpf_passes_in_band(self):
        # Band [2000, 6000] Hz; 4000 Hz tone should pass
        audio = _tone(4000.0)
        out = apply_filter(audio, SR, "bpf", [2000.0, 6000.0])
        assert _rms(out) > _rms(audio) * 0.9

    def test_bpf_attenuates_below_band(self):
        audio = _tone(200.0)
        out = apply_filter(audio, SR, "bpf", [2000.0, 6000.0])
        assert _rms(out) < _rms(audio) * 0.1

    def test_bpf_attenuates_above_band(self):
        audio = _tone(7500.0)
        out = apply_filter(audio, SR, "bpf", [1000.0, 4000.0])
        assert _rms(out) < _rms(audio) * 0.1

    def test_bpf_requires_two_element_freq(self):
        audio = _tone(1000.0)
        with pytest.raises(ValueError, match="bpf requires"):
            apply_filter(audio, SR, "bpf", 2000.0)

    def test_unknown_filter_type_raises(self):
        audio = _tone(1000.0)
        with pytest.raises(ValueError, match="Unknown filter_type"):
            apply_filter(audio, SR, "notch", 1000.0)

    def test_custom_order_accepted(self):
        audio = _tone(500.0)
        out = apply_filter(audio, SR, "hpf", 4000.0, order=2)
        assert out.shape == audio.shape
        assert out.dtype == np.float32

    def test_higher_order_steeper_rolloff(self):
        # At the same cutoff, order-8 should attenuate a stop-band tone more
        audio = _tone(200.0)
        out2 = apply_filter(audio, SR, "hpf", 2000.0, order=2)
        out8 = apply_filter(audio, SR, "hpf", 2000.0, order=8)
        assert _rms(out8) < _rms(out2)


# ---------------------------------------------------------------------------
# apply_speed
# ---------------------------------------------------------------------------

class TestApplySpeed:
    def test_speed_one_returns_same_array(self):
        audio = _tone(1000.0)
        out = apply_speed(audio, 1.0)
        np.testing.assert_array_equal(out, audio)

    def test_speed_two_halves_length(self):
        audio = _tone(1000.0, duration=2.0)
        out = apply_speed(audio, 2.0)
        expected = len(audio) // 2
        assert abs(len(out) - expected) <= 2

    def test_speed_half_doubles_length(self):
        audio = _tone(1000.0, duration=1.0)
        out = apply_speed(audio, 0.5)
        expected = len(audio) * 2
        assert abs(len(out) - expected) <= 2

    def test_output_dtype_float32(self):
        audio = _tone(1000.0).astype(np.float64)
        out = apply_speed(audio, 2.0)
        assert out.dtype == np.float32

    def test_fractional_speed_length(self):
        audio = _tone(1000.0, duration=1.0)  # SR samples
        out = apply_speed(audio, 1.5)
        expected = int(len(audio) / 1.5)
        assert abs(len(out) - expected) <= 2

    def test_slow_speed_length(self):
        audio = _tone(440.0, duration=0.5)
        out = apply_speed(audio, 0.75)
        expected = int(len(audio) / 0.75)
        assert abs(len(out) - expected) <= 2
