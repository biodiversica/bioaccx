"""Arbimon audio fetching and caching via the rfcx SDK.

Each row in the ext_table identifies a specific 1-minute recording by:
  stream_id  — Arbimon recording site ID
  date       — local date (YYYY-MM-DD)
  time       — local time (HH:MM or HH:MM:SS)
  utc_offset — UTC offset for the local time in hours (numeric: -3, +5.5, 0)

The label falls back to ``stream_id`` when the table row has no label.

Authentication
--------------
A persisted rfcx credentials file is required.  On first use, run::

    import rfcx
    client = rfcx.Client()
    client.authenticate(persisted_credentials_path="/path/to/.rfcx_credentials")

This opens a browser URL for device authorisation and saves the token.
Subsequent calls load the token from the file without user interaction.

Install the rfcx SDK (not on PyPI; install from the GitHub release wheel)::

    pip install https://github.com/rfcx/rfcx-sdk-python/releases/download/0.3.1/rfcx-0.3.1-py3-none-any.whl

Cache layout::

    <cache_dir>/
        arbimon/
            <stream_id>/
                <files downloaded by rfcx SDK>
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Optional


_AUDIO_EXTS: frozenset[str] = frozenset({".wav", ".flac", ".mp3", ".ogg"})
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

# Module-level rfcx client singleton — authenticated once per credentials path.
_client = None
_authenticated_path: Optional[str] = None


def _get_client(credentials_path: Optional[str]):
    global _client, _authenticated_path
    if _client is not None and credentials_path == _authenticated_path:
        return _client
    try:
        import rfcx
    except ImportError as exc:
        raise ImportError(
            "The rfcx SDK is required for Arbimon downloads. Install it with:\n"
            "  pip install https://github.com/rfcx/rfcx-sdk-python/releases/"
            "download/0.3.1/rfcx-0.3.1-py3-none-any.whl"
        ) from exc
    if not credentials_path:
        raise ValueError(
            "arbimon_credentials_path must be set to authenticate with Arbimon."
        )
    cred = Path(credentials_path)
    if not cred.exists():
        raise FileNotFoundError(
            f"Arbimon credentials file not found: {credentials_path}\n"
            "Run rfcx.Client().authenticate(persisted_credentials_path=...) once "
            "to create it."
        )
    c = rfcx.Client()
    c.authenticate(persisted_credentials_path=str(cred))
    _client = c
    _authenticated_path = credentials_path
    return _client


def _parse_utc_offset(val: float | str) -> float:
    """Parse a UTC offset to a float number of hours.

    Accepts integers, floats, or strings such as ``"-3"``, ``"+5.5"``,
    ``"UTC-3"``, ``"UTC+5:30"`` (treated as 5.5 hours).
    """
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().upper().replace("UTC", "").strip()
    if ":" in s:
        sign = -1 if s.startswith("-") else 1
        parts = s.lstrip("+-").split(":")
        hours = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0.0
        return sign * (hours + minutes / 60.0)
    return float(s)


def _to_utc(date_str: str, time_str: str, utc_offset: float) -> datetime.datetime:
    d = datetime.date.fromisoformat(date_str.strip())
    parts = time_str.strip().split(":")
    h, m = int(parts[0]), int(parts[1])
    s = int(parts[2]) if len(parts) > 2 else 0
    local_dt = datetime.datetime(d.year, d.month, d.day, h, m, s)
    return local_dt - datetime.timedelta(hours=utc_offset)


def _list_audio(directory: Path) -> set[Path]:
    return {p for p in directory.rglob("*") if p.suffix.lower() in _AUDIO_EXTS}


def _sentinel_path(cache_dir: Path, utc_dt: datetime.datetime) -> Path:
    return cache_dir / f"{utc_dt.strftime('%Y-%m-%d_%H-%M-%S')}.cached"


def _read_sentinel(cache_dir: Path, utc_dt: datetime.datetime) -> Optional[Path]:
    """Return the cached audio path from a sentinel file, or None if missing/stale."""
    sentinel = _sentinel_path(cache_dir, utc_dt)
    if not sentinel.exists():
        return None
    audio_path = Path(sentinel.read_text().strip())
    if audio_path.exists():
        return audio_path
    sentinel.unlink()  # stale — audio was deleted
    return None


def _write_sentinel(cache_dir: Path, utc_dt: datetime.datetime, audio_path: Path) -> None:
    _sentinel_path(cache_dir, utc_dt).write_text(str(audio_path))


def get_audio(
    stream_id: str,
    date_str: str,
    time_str: str,
    utc_offset: float | str,
    cache_dir: Path,
    credentials_path: Optional[str] = None,
) -> tuple[Path, str]:
    """Return ``(local_audio_path, stream_id)`` for an Arbimon recording.

    Downloads and caches the 1-minute segment that starts at the given local
    date/time.  Subsequent calls with the same stream/time read from the cache
    without any network access.

    Parameters
    ----------
    stream_id:
        Arbimon stream (recording site) ID.
    date_str:
        Recording date in local time, ISO format: ``"YYYY-MM-DD"``.
    time_str:
        Recording start time in local time: ``"HH:MM"`` or ``"HH:MM:SS"``.
    utc_offset:
        UTC offset for the local time in hours (e.g. ``-3``, ``+5.5``).
    cache_dir:
        Root cache directory.  Arbimon files are stored under
        ``<cache_dir>/arbimon/<stream_id>/``.
    credentials_path:
        Path to the rfcx persisted credentials file created by
        ``rfcx.Client().authenticate(persisted_credentials_path=...)``.
    """
    utc_offset_h = _parse_utc_offset(utc_offset)
    utc_dt = _to_utc(date_str, time_str, utc_offset_h)

    stream_cache = cache_dir / "arbimon" / stream_id
    stream_cache.mkdir(parents=True, exist_ok=True)

    cached = _read_sentinel(stream_cache, utc_dt)
    if cached is not None:
        return cached, stream_id

    before = _list_audio(stream_cache)

    client = _get_client(credentials_path)
    min_date = utc_dt
    max_date = utc_dt + datetime.timedelta(minutes=1)
    print(
        f"  [arbimon] downloading stream={stream_id} "
        f"{utc_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC …"
    )
    client.download_segments(
        dest_path=str(stream_cache),
        stream=stream_id,
        min_date=min_date,
        max_date=max_date,
        parallel=False,
    )

    after = _list_audio(stream_cache)
    new_files = sorted(after - before)
    if not new_files:
        raise RuntimeError(
            f"No audio downloaded for stream={stream_id!r} at "
            f"{utc_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC. "
            "Verify the stream ID, time window, and that your account has access."
        )

    result = new_files[0]
    _write_sentinel(stream_cache, utc_dt, result)
    return result, stream_id
