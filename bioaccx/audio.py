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
    """Load audio as float32 mono at *target_sr*, optionally reading a segment.

    Converts multi-channel files to mono by averaging channels.  Resamples
    using a polyphase filter (resample_poly) when the file's native sample rate
    differs from *target_sr*; the GCD-reduced up/down ratio minimises the
    filter size.

    Parameters
    ----------
    path:
        Audio file readable by soundfile (WAV, FLAC, OGG, etc.).
    target_sr:
        Desired output sample rate in Hz.
    offset:
        Start of the segment in seconds (0 = beginning of file).
    duration:
        Length of the segment in seconds; None reads to end of file.
    """
    # sf.info() is called before sf.read() to convert time-based offsets to
    # frame counts, which is the only unit soundfile's read() accepts.
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
    # Collapse stereo/multi-channel by averaging; no-op for mono arrays.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample only when necessary; reducing the ratio by GCD avoids unnecessarily
    # large polyphase filter orders.
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


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix *signal* with *noise* at the requested power-based SNR (dB).

    Scale factor derivation:
        SNR_dB = 10 * log10(P_signal / P_noise_scaled)
        scale  = sqrt(P_signal / (P_noise * 10^(snr_db/10)))

    If either signal or noise has zero power the signal is returned unchanged.
    """
    sig_power = float(np.mean(signal ** 2))
    noise_power = float(np.mean(noise ** 2))
    if sig_power == 0.0 or noise_power == 0.0:
        return signal
    scale = float(np.sqrt(sig_power / (noise_power * 10.0 ** (snr_db / 10.0))))
    return (signal + scale * noise).astype(np.float32)


def place_in_window(audio: np.ndarray, n_samples: int, offset_samples: int = 0) -> np.ndarray:
    """Place *audio* into a zero-padded window of *n_samples* at *offset_samples*.

    Signal samples that extend past the end of the window are clipped.
    """
    out = np.zeros(n_samples, dtype=np.float32)
    end = min(offset_samples + len(audio), n_samples)
    sig_end = end - offset_samples
    if sig_end > 0:
        out[offset_samples:end] = audio[:sig_end]
    return out


def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Change playback speed via resampling (no pitch correction).

    speed > 1.0 → shorter duration (faster); speed < 1.0 → longer (slower).

    The speed ratio is converted to a rational up/down pair so that
    resample_poly receives integers.  limit_denominator(1000) keeps the
    polyphase filter order manageable while keeping the approximation error
    below 0.1% for typical speed values.
    """
    if speed == 1.0:
        return audio
    from fractions import Fraction
    # Inverting speed gives the resampling ratio: output_len / input_len.
    ratio = Fraction(1.0 / speed).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
