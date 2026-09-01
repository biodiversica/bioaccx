"""Serving the audio behind a point on the embedding map.

Clicking a UMAP point should play the clip it was computed from, so this cuts
the window out of the source recording and renders a spectrogram of it.

Both use dependencies bioaccx already has — ``soundfile`` and ``scipy`` — and
the PNG is encoded here rather than through matplotlib, which lives in the
optional ``[umap]`` extra. That keeps the ``[gui]`` extra to three packages and
means the explorer works on a plain ``bioaccx[cpu,gui]`` install.
"""
from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

#: Frequencies below this are dominated by rumble and DC offset in field
#: recordings, and squash the useful range when the colour scale is fitted.
DEFAULT_DB_FLOOR = -80.0

#: Anchors of the colour ramp, dark to bright. A perceptually ordered ramp in
#: the spirit of viridis, written out rather than pulled from matplotlib.
_RAMP = np.array([
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
    (180, 222, 44), (253, 231, 37),
], dtype=np.float64)


class AudioError(Exception):
    """A clip could not be read or rendered."""


def _colourise(norm: np.ndarray) -> np.ndarray:
    """Map values in 0–1 onto the ramp, returning an (H, W, 3) uint8 image."""
    scaled = np.clip(norm, 0.0, 1.0) * (len(_RAMP) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.minimum(lower + 1, len(_RAMP) - 1)
    frac = (scaled - lower)[..., None]
    return (_RAMP[lower] * (1 - frac) + _RAMP[upper] * frac).astype(np.uint8)


def _png(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as a PNG."""
    height, width, _ = rgb.shape
    # Each scanline is prefixed with filter type 0 (none).
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _read(path: Path, start: Optional[float], end: Optional[float]
          ) -> tuple[np.ndarray, int]:
    """Read the window ``[start, end)`` of *path* as mono float32."""
    import soundfile as sf

    if not path.exists():
        raise AudioError(f"source audio not found: {path}")
    try:
        info = sf.info(str(path))
        first = int((start or 0.0) * info.samplerate)
        last = int(end * info.samplerate) if end else info.frames
        first = max(0, min(first, info.frames))
        last = max(first, min(last, info.frames))
        audio, rate = sf.read(str(path), start=first, stop=last, dtype="float32",
                              always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise AudioError(f"could not read {path.name}: {exc}") from exc

    if audio.size == 0:
        raise AudioError(f"empty window in {path.name}")
    return audio.mean(axis=1), rate


def clip_wav(path: Path, start: Optional[float], end: Optional[float]) -> bytes:
    """The requested window as a WAV file, ready to hand to an audio element."""
    import soundfile as sf

    audio, rate = _read(path, start, end)
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def spectrogram_png(
    path: Path,
    start: Optional[float],
    end: Optional[float],
    *,
    fmin: float = 0.0,
    fmax: float = 0.0,
    db_floor: float = DEFAULT_DB_FLOOR,
    width: int = 800,
    height: int = 320,
) -> bytes:
    """Render the window as a spectrogram PNG.

    *fmax* of 0 means Nyquist. The colour scale is fitted to the clip's own peak
    with *db_floor* below it, so a quiet recording is still legible.
    """
    from scipy import signal

    audio, rate = _read(path, start, end)

    nperseg = min(1024, max(64, len(audio) // 128 or 64))
    freqs, _, power = signal.spectrogram(
        audio, fs=rate, nperseg=nperseg, noverlap=nperseg // 2, mode="magnitude")
    if power.size == 0:
        raise AudioError("clip too short to render a spectrogram")

    top = fmax if fmax and fmax > 0 else freqs[-1]
    band = (freqs >= fmin) & (freqs <= top)
    if not band.any():
        raise AudioError(f"no frequency content between {fmin} and {top} Hz")
    power = power[band]

    decibels = 20.0 * np.log10(np.maximum(power, 1e-10))
    peak = float(decibels.max())
    norm = (decibels - (peak + db_floor)) / max(-db_floor, 1e-6)

    # Rows are frequency bins, low first; flip so low frequencies sit at the
    # bottom of the image the way a spectrogram is conventionally read.
    image = _colourise(np.flipud(norm))

    # Nearest-neighbour resample to the requested size — no scipy.ndimage, and
    # exact enough for a thumbnail beside the map.
    rows = np.linspace(0, image.shape[0] - 1, height).round().astype(int)
    cols = np.linspace(0, image.shape[1] - 1, width).round().astype(int)
    return _png(image[rows][:, cols])
