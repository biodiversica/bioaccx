"""Audio I/O and preprocessing utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt


def load_mono(
    path: Path,
    target_sr: int,
    offset: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    """Load audio as float32 mono at *target_sr*, optionally reading a segment."""
    start_frame = int(offset * sf.info(str(path)).samplerate) if offset > 0 else 0
    frames = (
        int(duration * sf.info(str(path)).samplerate) if duration is not None else -1
    )
    audio, sr = sf.read(
        str(path),
        start=start_frame,
        frames=frames if frames > 0 else -1,
        always_2d=False,
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        gcd = np.gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd)
    return audio.astype(np.float32)


def to_fixed_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Trim or zero-pad *audio* to exactly *n_samples*."""
    if len(audio) >= n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - len(audio))).astype(np.float32)


AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff"}
)


def apply_filter(
    audio: np.ndarray,
    sr: int,
    filter_type: str,
    freq: float | list[float],
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth filter.

    filter_type: 'hpf' | 'lpf' | 'bpf'
    freq: cutoff Hz (scalar) for hpf/lpf; [low_hz, high_hz] for bpf
    """
    nyq = sr / 2.0
    if filter_type in ("hpf", "lpf"):
        Wn = float(freq) / nyq  # type: ignore[arg-type]
        btype = "high" if filter_type == "hpf" else "low"
        sos = butter(order, Wn, btype=btype, output="sos")
    elif filter_type == "bpf":
        if not hasattr(freq, "__len__") or len(freq) != 2:  # type: ignore[arg-type]
            raise ValueError("bpf requires filter_freq as [low_hz, high_hz]")
        Wn = [float(freq[0]) / nyq, float(freq[1]) / nyq]  # type: ignore[index]
        sos = butter(order, Wn, btype="band", output="sos")
    else:
        raise ValueError(f"Unknown filter_type {filter_type!r}; use 'hpf', 'lpf', or 'bpf'")
    return sosfilt(sos, audio).astype(np.float32)


def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Change playback speed via resampling (no pitch correction).

    speed > 1.0 → shorter duration (faster); speed < 1.0 → longer (slower).
    """
    if speed == 1.0:
        return audio
    from fractions import Fraction
    ratio = Fraction(1.0 / speed).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
