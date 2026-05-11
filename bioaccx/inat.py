"""iNaturalist audio fetching and caching.

Audio files and observation metadata are cached locally so that subsequent
runs with the same observation IDs skip the network entirely.

Cache layout:
    <cache_dir>/
        inat_<obs_id>.json              — observation metadata
        inat_<obs_id>_<index>.<ext>     — downloaded audio file
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path


_API_BASE = "https://api.inaturalist.org/v1"
_HEADERS = {"User-Agent": "bioaccx (https://github.com/biodiversica/bioaccx)"}
_API_PAUSE = 0.5  # seconds between API calls to respect rate limits


def _get_json(url: str) -> dict:
    """Fetch a JSON endpoint and return the parsed dict.

    The User-Agent header is required by iNaturalist's API terms of service;
    requests without a valid User-Agent may be rate-limited or rejected.
    """
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _fetch_obs(obs_id: str | int, cache_dir: Path) -> dict:
    """Return observation dict, using a cached .json file when available.

    Only the first element of the ``results`` list is kept; subsequent fields
    (pagination metadata) are discarded.  A short sleep after each live request
    avoids hitting the iNaturalist rate limit (60 req/min for anonymous access).
    """
    meta = cache_dir / f"inat_{obs_id}.json"
    if meta.exists():
        return json.loads(meta.read_text())
    data = _get_json(f"{_API_BASE}/observations/{obs_id}")
    results = data.get("results", [])
    if not results:
        raise ValueError(f"iNaturalist observation {obs_id} not found")
    obs = results[0]
    # Cache the observation so future runs don't hit the API.
    meta.write_text(json.dumps(obs))
    time.sleep(_API_PAUSE)
    return obs


def get_audio(
    obs_id: str | int,
    sound_index: int,
    cache_dir: Path,
) -> tuple[Path, str]:
    """Return ``(local_audio_path, scientific_name)`` for an iNaturalist observation.

    Downloads and caches both the audio file and the observation metadata on the
    first call.  Subsequent calls with the same ``obs_id`` / ``sound_index`` read
    from the cache without any network access.

    Parameters
    ----------
    obs_id:
        iNaturalist observation ID (integer or string).
    sound_index:
        0-based index into the observation's ``sounds`` array.  Defaults to 0
        (first/only sound) when omitted by the caller.
    cache_dir:
        Directory where audio files and metadata are stored.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    obs = _fetch_obs(obs_id, cache_dir)

    # Scientific name falls back to a synthetic label so the caller always
    # receives a non-empty string even for taxon-less observations.
    taxon = obs.get("taxon") or {}
    scientific_name: str = taxon.get("name") or f"inat_{obs_id}"

    # Return cached audio if already downloaded (any non-.json file matching the pattern).
    cached = [
        p for p in cache_dir.glob(f"inat_{obs_id}_{sound_index}.*")
        if p.suffix != ".json"
    ]
    if cached:
        return cached[0], scientific_name

    sounds = obs.get("sounds", [])
    if not sounds:
        raise ValueError(f"Observation {obs_id} has no sounds")
    if sound_index >= len(sounds):
        raise ValueError(
            f"Observation {obs_id} has {len(sounds)} sound(s); "
            f"sound_index {sound_index} is out of range (0-based)"
        )

    # Prefer file_url (direct CDN link) over url (may redirect).
    url: str = sounds[sound_index].get("file_url") or sounds[sound_index].get("url", "")
    if not url:
        raise ValueError(
            f"No downloadable URL for observation {obs_id} sound {sound_index}"
        )

    # Strip query params to determine the file extension from the path component.
    ext = Path(url.split("?")[0]).suffix or ".mp3"
    dest = cache_dir / f"inat_{obs_id}_{sound_index}{ext}"

    print(f"  [inat] downloading obs {obs_id} sound {sound_index} → {dest.name}")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())
    time.sleep(_API_PAUSE)

    return dest, scientific_name
