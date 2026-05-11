"""Xeno-canto audio fetching and caching.

Audio files and recording metadata are cached locally so that subsequent
runs with the same recording IDs skip the network entirely.

Cache layout:
    <cache_dir>/
        xc_<id>.json        — recording metadata (only written when API key provided)
        xc_<id>.<ext>       — downloaded audio file

API key
-------
Xeno-canto API v3 requires a personal API key for metadata queries.
If no key is supplied the metadata step is skipped — audio is downloaded
directly via ``https://xeno-canto.org/<id>/download`` and the scientific name
falls back to ``"xc_<id>"``.  Register at https://xeno-canto.org/explore/api.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.request
from pathlib import Path


_API_BASE = "https://xeno-canto.org/api/3/recordings"
_HEADERS = {"User-Agent": "bioaccx (https://github.com/biodiversica/bioaccx)"}
_API_PAUSE = 0.5      # seconds between API calls to respect rate limits
_CHUNK_SIZE = 65536   # bytes per read chunk during audio download
_READ_TIMEOUT = 30    # seconds to wait for each chunk before giving up


def _strip_prefix(xc_id: str | int) -> str:
    """Return the bare numeric string, stripping any leading 'XC' prefix.

    Accepts ``12345``, ``"12345"``, ``"XC12345"``, or ``"xc12345"``.
    """
    return str(xc_id).upper().lstrip("XC").strip()


def _get_json(url: str) -> dict:
    """Fetch a JSON endpoint and return the parsed dict.

    The User-Agent header identifies the application to the Xeno-canto API
    as required by their terms of service.
    """
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _fetch_recording(xc_num: str, cache_dir: Path, api_key: str) -> dict:
    """Return recording dict, using a cached .json file when available.

    The Xeno-canto API v3 requires a key in the query string.  Only the first
    result is kept; subsequent metadata (total count, page) is discarded.  A
    short sleep after each live call avoids the API rate limit.
    """
    meta = cache_dir / f"xc_{xc_num}.json"
    if meta.exists():
        return json.loads(meta.read_text())
    url = f"{_API_BASE}?query=nr:{xc_num}&key={api_key}"
    data = _get_json(url)
    recordings = data.get("recordings", [])
    if not recordings:
        raise ValueError(f"Xeno-canto recording XC{xc_num} not found")
    rec = recordings[0]
    # Cache the metadata so future runs skip the API call.
    meta.write_text(json.dumps(rec))
    time.sleep(_API_PAUSE)
    return rec


def _download_audio(url: str, dest: Path) -> None:
    """Download *url* to *dest* using chunked reads with per-chunk timeouts.

    ``urllib.request.urlopen(timeout=N)`` only guards the initial connection.
    Setting the underlying socket timeout after opening the response ensures
    that each individual read also times out, preventing infinite hangs on
    stalled transfers.
    """
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        r.fp.raw._sock.settimeout(_READ_TIMEOUT)  # per-chunk read timeout
        tmp = dest.with_suffix(".part")
        try:
            with tmp.open("wb") as fh:
                while True:
                    chunk = r.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
        except socket.timeout:
            tmp.unlink(missing_ok=True)
            raise TimeoutError(
                f"Download stalled (no data for {_READ_TIMEOUT}s): {url}"
            )
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.rename(dest)


def get_audio(
    xc_id: str | int,
    cache_dir: Path,
    api_key: str | None = None,
) -> tuple[Path, str]:
    """Return ``(local_audio_path, scientific_name)`` for a Xeno-canto recording.

    Downloads and caches the audio file on the first call.  Subsequent calls
    with the same ``xc_id`` read from the cache without any network access.

    Parameters
    ----------
    xc_id:
        Xeno-canto recording ID — numeric (``12345``) or with prefix
        (``"XC12345"``).
    cache_dir:
        Directory where audio files and metadata are stored.
    api_key:
        Xeno-canto API v3 key.  When provided, recording metadata is fetched to
        obtain the scientific name.  When omitted, audio is downloaded via the
        direct URL and the name falls back to ``"xc_<id>"``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    xc_num = _strip_prefix(xc_id)

    # Try to read cached metadata for the scientific name (works regardless of key).
    # This lets us return the name even on subsequent calls without hitting the API.
    scientific_name: str | None = None
    meta_path = cache_dir / f"xc_{xc_num}.json"
    if meta_path.exists():
        try:
            rec = json.loads(meta_path.read_text())
            gen = rec.get("gen", "").strip()
            sp  = rec.get("sp",  "").strip()
            scientific_name = f"{gen} {sp}".strip() if (gen or sp) else None
        except Exception:
            pass

    # Return cached audio if already downloaded (any non-.json file matching the pattern).
    cached = [p for p in cache_dir.glob(f"xc_{xc_num}.*") if p.suffix != ".json"]
    if cached:
        return cached[0], scientific_name or f"xc_{xc_num}"

    # Fetch metadata via API to get the scientific name and canonical audio URL.
    file_url: str = ""
    ext = ".mp3"
    if api_key:
        rec = _fetch_recording(xc_num, cache_dir, api_key)
        gen = rec.get("gen", "").strip()
        sp  = rec.get("sp",  "").strip()
        scientific_name = f"{gen} {sp}".strip() if (gen or sp) else None
        file_url = rec.get("file", "")
        # Determine file extension from the URL path, stripping query params.
        if file_url:
            ext = Path(file_url.split("?")[0]).suffix or ".mp3"

    dest = cache_dir / f"xc_{xc_num}{ext}"

    # Prefer the metadata file URL; fall back to the direct download endpoint.
    # The direct endpoint works without an API key but may have rate-limiting.
    download_url = file_url if file_url else f"https://xeno-canto.org/{xc_num}/download"

    print(f"  [xc] downloading XC{xc_num} → {dest.name}")
    _download_audio(download_url, dest)
    time.sleep(_API_PAUSE)

    return dest, scientific_name or f"xc_{xc_num}"
