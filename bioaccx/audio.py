"""Audio I/O and preprocessing utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


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
