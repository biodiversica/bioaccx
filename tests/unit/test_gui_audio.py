"""Tests for clip extraction and spectrogram rendering.

The PNG is encoded here rather than through matplotlib, so these check it is a
real image a browser will accept — signature, header, and a body that inflates
back to the right number of pixels — not merely that some bytes came out.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bioaccx.gui import audio

RATE = 32000


@pytest.fixture
def chirp(tmp_path) -> Path:
    """Four seconds of rising tone, so a spectrogram has visible structure."""
    t = np.linspace(0, 4, 4 * RATE, endpoint=False)
    signal = (0.4 * np.sin(2 * np.pi * (500 + 900 * t) * t)).astype("float32")
    path = tmp_path / "chirp.wav"
    sf.write(path, signal, RATE)
    return path


@pytest.fixture
def stereo(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    path = tmp_path / "stereo.wav"
    sf.write(path, rng.standard_normal((RATE, 2)).astype("float32") * 0.1, RATE)
    return path


def _png_header(data: bytes) -> tuple[int, int, int, int]:
    width, height = struct.unpack(">II", data[16:24])
    return width, height, data[24], data[25]


class TestClip:
    def test_cuts_the_requested_window(self, chirp):
        import io
        data, rate = sf.read(io.BytesIO(audio.clip_wav(chirp, 1.0, 3.0)))
        assert rate == RATE
        assert len(data) == pytest.approx(2 * RATE, abs=2)

    def test_produces_a_real_wav(self, chirp):
        assert audio.clip_wav(chirp, 0.0, 1.0)[:4] == b"RIFF"

    def test_no_window_means_the_whole_file(self, chirp):
        import io
        data, _ = sf.read(io.BytesIO(audio.clip_wav(chirp, None, None)))
        assert len(data) == pytest.approx(4 * RATE, abs=2)

    def test_a_window_past_the_end_is_clamped(self, chirp):
        import io
        data, _ = sf.read(io.BytesIO(audio.clip_wav(chirp, 3.5, 99.0)))
        assert 0 < len(data) <= RATE

    def test_stereo_is_mixed_to_mono(self, stereo):
        import io
        data, _ = sf.read(io.BytesIO(audio.clip_wav(stereo, 0.0, 0.5)))
        assert data.ndim == 1

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(audio.AudioError, match="not found"):
            audio.clip_wav(tmp_path / "gone.wav", 0.0, 1.0)

    def test_an_empty_window_is_reported(self, chirp):
        with pytest.raises(audio.AudioError, match="empty window"):
            audio.clip_wav(chirp, 2.0, 2.0)


class TestSpectrogram:
    def test_renders_a_valid_png(self, chirp):
        png = audio.spectrogram_png(chirp, 1.0, 3.0)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        width, height, depth, colour_type = _png_header(png)
        assert (depth, colour_type) == (8, 2)          # 8-bit truecolour RGB
        assert (width, height) == (800, 320)

    def test_the_pixel_data_inflates_to_the_declared_size(self, chirp):
        """A decoder will reject a PNG whose IDAT does not match its header."""
        png = audio.spectrogram_png(chirp, 1.0, 3.0, width=120, height=60)
        width, height, _, _ = _png_header(png)

        start = png.index(b"IDAT") + 4
        length = struct.unpack(">I", png[start - 8:start - 4])[0]
        raw = zlib.decompress(png[start:start + length])
        # Each row is one filter byte plus three bytes per pixel.
        assert len(raw) == height * (1 + width * 3)

    def test_honours_the_requested_size(self, chirp):
        png = audio.spectrogram_png(chirp, 0.0, 2.0, width=200, height=100)
        assert _png_header(png)[:2] == (200, 100)

    def test_a_frequency_band_can_be_selected(self, chirp):
        assert audio.spectrogram_png(chirp, 0.0, 2.0, fmin=500, fmax=4000)[:4] == b"\x89PNG"[:4]

    def test_an_empty_band_is_reported(self, chirp):
        with pytest.raises(audio.AudioError, match="no frequency content"):
            audio.spectrogram_png(chirp, 0.0, 2.0, fmin=20000, fmax=20001)

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(audio.AudioError, match="not found"):
            audio.spectrogram_png(tmp_path / "gone.wav", 0.0, 1.0)

    def test_the_image_is_not_uniform(self, chirp):
        """A blank render would still be a valid PNG — check it carries signal."""
        png = audio.spectrogram_png(chirp, 0.0, 4.0, width=64, height=64)
        start = png.index(b"IDAT") + 4
        length = struct.unpack(">I", png[start - 8:start - 4])[0]
        raw = zlib.decompress(png[start:start + length])
        assert len(set(raw)) > 8


class TestColourRamp:
    def test_maps_the_range_onto_the_ramp_ends(self):
        image = audio._colourise(np.array([[0.0, 1.0]]))
        assert tuple(image[0, 0]) == tuple(audio._RAMP[0].astype(int))
        assert tuple(image[0, 1]) == tuple(audio._RAMP[-1].astype(int))

    def test_values_outside_the_range_are_clipped(self):
        image = audio._colourise(np.array([[-5.0, 5.0]]))
        assert tuple(image[0, 0]) == tuple(audio._RAMP[0].astype(int))
        assert tuple(image[0, 1]) == tuple(audio._RAMP[-1].astype(int))

    def test_the_ramp_is_monotonic_in_brightness(self):
        """Ordered colours are what make the image readable as intensity."""
        brightness = audio._RAMP.sum(axis=1)
        assert list(brightness) == sorted(brightness)
