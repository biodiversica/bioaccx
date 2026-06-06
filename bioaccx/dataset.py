"""Dataset loading for three label organization modes.

Modes
-----
subfolders
    data_dir/
        class_a/file1.wav ...
        class_b/file2.wav ...
    OR with pre-defined split:
        data_dir/train/class_a/ ... data_dir/test/class_a/ ...

file_per_label
    data_dir/
        file1.wav
        file1.txt   (rows: start_time, end_time, label — one row per segment)
        file2.wav
        file2.txt
    Delimiter is auto-detected (tab or comma). Multiple rows per file are
    supported; each row yields one AudioSample with its own time bounds.

table
    A CSV/TSV with columns: filename, label, start_time, end_time, [split]
    start_time / end_time are in seconds and define the audio segment.
    If a 'split' column exists its values ("train"/"test") take precedence over
    the automatic split.
"""
from __future__ import annotations

import atexit
import csv
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from bioaccx.audio import AUDIO_EXTENSIONS, apply_filter, apply_speed, load_mono, mix_at_snr, place_in_window, to_fixed_length
from bioaccx.config import DatasetConfig


# Lazily-created temp dir for preprocessed audio; cleaned up on process exit.
_preproc_tempdir: Optional[Path] = None


def _get_preproc_tempdir() -> Path:
    """Return (and lazily create) a shared temp dir for preprocessed audio files.

    The directory is registered with atexit so it is cleaned up automatically
    when the interpreter exits.  All runs within the same process share one
    directory to avoid redundant preprocessing.
    """
    global _preproc_tempdir
    if _preproc_tempdir is None:
        tmp = Path(tempfile.mkdtemp(prefix="bioaccx_preproc_"))
        _preproc_tempdir = tmp
        atexit.register(lambda: shutil.rmtree(tmp, ignore_errors=True))
    return _preproc_tempdir


@dataclass
class AudioSample:
    path: Path
    label: str
    start_time: Optional[float] = None  # seconds; None = use full window
    end_time: Optional[float] = None
    split: Optional[str] = None         # "train" | "test" | None
    is_appended: bool = False           # True for samples from append_dataset_path
    original_path: Optional[Path] = None  # source file before filter/speed preprocessing
    ssh_path: Optional[str] = None      # remote path; set when data_dir is accessed via SSH
    # Augmentation fields (None = clean sample)
    noise_path: Optional[Path] = None       # noise file to mix in
    snr: Optional[float] = None            # signal-to-noise ratio in dB
    noise_start_time: Optional[float] = None  # start offset into noise file (seconds)
    # Random sample shift — set when random_sample_shift is True and the signal
    # is shorter than the foundation model window
    signal_duration_seconds: Optional[float] = None  # actual audio duration before window padding
    signal_offset_samples: Optional[int] = None       # random placement offset within the window


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _as_dirs(data_dir: str | list[str]) -> list[Path]:
    """Normalise data_dir (str or list[str]) to a list of non-empty Paths."""
    if isinstance(data_dir, str):
        return [Path(data_dir)] if data_dir else []
    return [Path(d) for d in data_dir if d]


def _source_stem(s: AudioSample) -> str:
    """Stem of the original source file, before any preprocessing renames it."""
    return (s.original_path or s.path).stem


def _sample_export_key(s: AudioSample) -> str:
    """Filename stem that export_dataset_audio produces for this sample.

    Always derived from the original source file stem so that deduplication
    works regardless of whether the sample has been preprocessed.
    """
    start = s.start_time if s.start_time is not None else 0.0
    end_str = f"{s.end_time:.3f}" if s.end_time is not None else "full"
    return f"{_source_stem(s)}_{start:.3f}_{end_str}"


def _preprocess_audio_files(
    paths: set[Path],
    sample_rate: int,
    filter_type: str | None,
    filter_freq: float | list[float] | None,
    filter_order: int,
    speed: float,
    preproc_dir: Path,
    n_workers: int = 1,
) -> dict[Path, Path]:
    """Apply filter and/or speed change to a set of audio files in parallel.

    Writes processed WAV files to *preproc_dir* (skips if already present).
    Returns a mapping original_path → processed_path.
    Failures fall back to the original path so the pipeline can continue.
    """
    import hashlib
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import soundfile as sf

    preproc_dir.mkdir(parents=True, exist_ok=True)
    path_map: dict[Path, Path] = {}
    to_process: list[tuple[Path, Path]] = []

    for orig_path in sorted(paths):
        # Build a filename tag that encodes all preprocessing parameters so that
        # different settings produce different cached files in the same directory.
        tag_parts: list[str] = []
        if filter_type and filter_freq is not None:
            if isinstance(filter_freq, (list, tuple)):
                freq_tag = "_".join(f"{f:.0f}" for f in filter_freq)
            else:
                freq_tag = f"{filter_freq:.0f}"
            tag_parts.append(f"{filter_type}{freq_tag}hz_o{filter_order}")
        if speed != 1.0:
            tag_parts.append(f"spd{speed:.4g}")
        # Include a short hash of the full path so files with the same stem but
        # from different directories don't collide in the shared cache dir.
        path_hash = hashlib.md5(str(orig_path.resolve()).encode()).hexdigest()[:8]
        tag_parts.append(path_hash)
        tag = "_".join(tag_parts)
        new_path = preproc_dir / f"{orig_path.stem}_{tag}.wav"
        path_map[orig_path] = new_path
        # Skip files that were already preprocessed in a previous call.
        if not new_path.exists():
            to_process.append((orig_path, new_path))

    if not to_process:
        return path_map

    available = os.cpu_count() or 1
    workers = max(1, min(n_workers, available, len(to_process)))

    def _process_one(orig_path: Path, new_path: Path):
        audio = load_mono(orig_path, sample_rate)
        if filter_type is not None:
            if filter_freq is None:
                raise ValueError(f"filter_freq must be set when filter={filter_type!r}")
            audio = apply_filter(audio, sample_rate, filter_type, filter_freq, filter_order)
        if speed != 1.0:
            audio = apply_speed(audio, speed)
        sf.write(str(new_path), audio, sample_rate)

    if workers == 1:
        for orig_path, new_path in to_process:
            try:
                _process_one(orig_path, new_path)
                print(f"  [preprocess] {orig_path.name} → {new_path.name}")
            except Exception as exc:
                print(f"  [preprocess error] {orig_path.name}: {exc}")
                path_map[orig_path] = orig_path
    else:
        print(f"  [preprocess] {len(to_process)} file(s) — workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, o, n): (o, n) for o, n in to_process}
            for fut in as_completed(futures):
                orig_path, new_path = futures[fut]
                try:
                    fut.result()
                    print(f"  [preprocess] {orig_path.name} → {new_path.name}")
                except Exception as exc:
                    print(f"  [preprocess error] {orig_path.name}: {exc}")
                    path_map[orig_path] = orig_path

    return path_map


def _load_ssh_samples(
    cfg: DatasetConfig,
    data_dirs: list[Path],
    exts: frozenset[str],
    window_seconds: float,
) -> list[AudioSample]:
    """List remote audio files via SFTP and return chunked AudioSamples with ssh_path set.

    Audio files are NOT downloaded to disk here — only directory listings and
    audio durations (via in-memory SFTP reads) are fetched.  Actual audio bytes
    are downloaded on demand by extract_embeddings.

    filter/speed preprocessing is not supported for SSH data dirs (skipped with
    a warning if configured).
    """
    import csv as _csv
    from bioaccx.ssh import (
        open_sftp_client, list_remote_audio, get_audio_duration,
        list_remote_file_per_label, read_remote_text,
    )

    print(f"  [ssh] connecting to {cfg.ssh_user or ''}@{cfg.ssh_host}:{cfg.ssh_port}")
    ssh_client, sftp = open_sftp_client(
        cfg.ssh_host, cfg.ssh_user or "", cfg.ssh_port, cfg.ssh_key_path
    )
    rows: list[_LabelRow] = []

    try:
        if cfg.label_mode == "subfolders":
            for remote_dir in [str(d) for d in data_dirs]:
                print(f"  [ssh] listing {remote_dir}")
                for remote_path, label, split in list_remote_audio(sftp, remote_dir, exts):
                    try:
                        duration = get_audio_duration(sftp, remote_path)
                    except Exception as exc:
                        print(f"  [ssh] cannot read duration of {remote_path}: {exc} — skipping")
                        continue
                    rows.append(_LabelRow(
                        path=Path(remote_path),
                        label=label,
                        start_time=None,
                        end_time=duration,
                        split=split,
                    ))

        elif cfg.label_mode == "table":
            if cfg.table_file is None:
                raise ValueError("dataset.table_file must be set when label_mode='table'")
            table_text = Path(cfg.table_file).read_text()
            delimiter = "\t" if "\t" in table_text.splitlines()[0] else ","
            reader = _csv.DictReader(table_text.splitlines(), delimiter=delimiter)
            remote_base = str(data_dirs[0]) if data_dirs else ""
            for row in reader:
                fname = row.get(cfg.filename_col, "").strip()
                if not fname:
                    continue
                remote_path = fname if Path(fname).is_absolute() else f"{remote_base}/{fname}"
                try:
                    start = float(row[cfg.start_col]) if cfg.start_col in row and row[cfg.start_col].strip() else None
                    end   = float(row[cfg.end_col])   if cfg.end_col   in row and row[cfg.end_col].strip()   else None
                except ValueError:
                    continue
                if start is not None and end is not None and end <= start:
                    continue
                label = row.get(cfg.label_col, "").strip()
                if not label:
                    print(f"  [ssh] skip table row {fname}: no label")
                    continue
                split = row.get(cfg.split_col, "").strip() or None
                rows.append(_LabelRow(
                    path=Path(remote_path), label=label,
                    start_time=start, end_time=end, split=split,
                ))

        elif cfg.label_mode == "file_per_label":
            for remote_dir in [str(d) for d in data_dirs]:
                print(f"  [ssh] listing {remote_dir}")
                for audio_rpath, txt_rpath in list_remote_file_per_label(sftp, remote_dir, exts):
                    try:
                        text = read_remote_text(sftp, txt_rpath)
                    except Exception as exc:
                        print(f"  [ssh] cannot read {txt_rpath}: {exc} — skipping")
                        continue
                    if not text.strip():
                        continue
                    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
                    for line in text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(delimiter)
                        if len(parts) < 3:
                            continue
                        try:
                            start, end = float(parts[0]), float(parts[1])
                        except ValueError:
                            continue
                        label = parts[-1].strip()
                        if label and end > start:
                            rows.append(_LabelRow(
                                path=Path(audio_rpath), label=label,
                                start_time=start, end_time=end,
                            ))
        else:
            raise ValueError(f"Unknown label_mode: {cfg.label_mode!r}")

    finally:
        sftp.close()
        ssh_client.close()
        print("  [ssh] listing complete, connection closed")

    if cfg.filter is not None or cfg.speed != 1.0:
        print(
            "  [warning] filter/speed settings are not applied to SSH data dirs — "
            "preprocessing skipped"
        )

    chunks = _chunk_rows(rows, window_seconds, cfg.overlap, cfg.min_anchor_fraction)
    return [
        AudioSample(
            path=c.path,
            label=c.label,
            start_time=c.start_time,
            end_time=c.end_time,
            split=c.split,
            original_path=c.original_path,
            ssh_path=str(c.path),
        )
        for c in chunks
    ]


def load_samples(
    cfg: DatasetConfig,
    window_seconds: float,
    sample_rate: Optional[int] = None,
) -> list[AudioSample]:
    """Load all audio samples described by *cfg* and return them as AudioSamples.

    Handles the three label modes (subfolders, file_per_label, table) and the
    optional remote sources (iNaturalist, Xeno-canto, Arbimon via ext_table_file).
    When append_dataset_path is set, existing exported samples are loaded first
    and only new samples (not already present) are added.

    Preprocessing (filter / speed change) is applied to source files before
    chunking when both filter/speed settings are non-trivial and sample_rate is
    provided.  Preprocessing writes WAV files to a shared temp directory and is
    skipped for files that were already processed in an earlier call.

    Parameters
    ----------
    cfg:
        Full dataset configuration.
    window_seconds:
        Duration of each audio chunk, derived from the foundation model window.
    sample_rate:
        Target sample rate.  Required for preprocessing; if omitted, filter and
        speed settings are silently ignored.

    Returns
    -------
    list[AudioSample]
        Combined list of existing (appended) and new samples, each with path,
        label, optional time bounds, and optional split assignment.
    """
    exts = frozenset(f".{e.lstrip('.')}" for e in cfg.audio_extensions)
    data_dirs = _as_dirs(cfg.data_dir)

    if not data_dirs and not cfg.ext_table_file:
        raise ValueError(
            "dataset.data_dir must be set unless ext_table_file is provided"
        )

    needs_preproc = (cfg.filter is not None or cfg.speed != 1.0) and sample_rate is not None
    if (cfg.filter is not None or cfg.speed != 1.0) and sample_rate is None:
        print(
            "  [warning] filter/speed set but sample_rate not provided to load_samples "
            "— preprocessing skipped"
        )

    def _preprocess_samples(
        samples: list[AudioSample],
        paths_to_process: set[Path] | None = None,
    ) -> list[AudioSample]:
        """Preprocess audio files and remap sample paths + adjust times.

        ``paths_to_process`` limits which source files are preprocessed; if
        None, all unique paths in *samples* are processed.
        """
        unique_paths = paths_to_process if paths_to_process is not None else {s.path for s in samples}
        path_map = _preprocess_audio_files(
            unique_paths, sample_rate, cfg.filter, cfg.filter_freq,  # type: ignore[arg-type]
            cfg.filter_order, cfg.speed, _get_preproc_tempdir(),
            n_workers=cfg.embedding_workers,
        )
        speed = cfg.speed
        return [
            AudioSample(
                path=path_map.get(s.path, s.path), label=s.label,
                start_time=s.start_time / speed if s.start_time is not None else None,
                end_time=s.end_time / speed if s.end_time is not None else None,
                split=s.split, is_appended=s.is_appended,
                original_path=s.original_path if s.original_path is not None else s.path,
            )
            for s in samples
        ]

    def _collect_raw_rows() -> list[_LabelRow]:
        """Parse all label rows from data_dir without preprocessing."""
        rows: list[_LabelRow] = []
        if cfg.label_mode == "file_per_label":
            for d in data_dirs:
                rows.extend(_parse_file_per_label(d, exts))
        elif cfg.label_mode == "table":
            if cfg.table_file is None:
                raise ValueError(
                    "dataset.table_file must be set when label_mode='table'"
                )
            rows = _parse_table(
                data_dirs, Path(cfg.table_file),
                cfg.filename_col, cfg.label_col,
                cfg.start_col, cfg.end_col, cfg.split_col,
            )
        else:
            raise ValueError(f"Unknown label_mode: {cfg.label_mode!r}")
        return rows

    def _preprocess_rows(rows: list[_LabelRow]) -> list[_LabelRow]:
        """Preprocess all source files referenced by rows and adjust times."""
        unique_paths = {r.path for r in rows}
        path_map = _preprocess_audio_files(
            unique_paths, sample_rate, cfg.filter, cfg.filter_freq,  # type: ignore[arg-type]
            cfg.filter_order, cfg.speed, _get_preproc_tempdir(),
            n_workers=cfg.embedding_workers,
        )
        speed = cfg.speed
        return [
            _LabelRow(
                path=path_map.get(r.path, r.path), label=r.label,
                start_time=r.start_time / speed if r.start_time is not None else None,
                end_time=r.end_time / speed if r.end_time is not None else None,
                split=r.split,
                original_path=r.path,
            )
            for r in rows
        ]

    def _load_ext_rows() -> list[_LabelRow]:
        remote_cache = (
            Path(cfg.ext_cache_dir)
            if cfg.ext_cache_dir
            else Path.home() / ".cache" / "bioaccx" / "ext"
        )
        return _parse_ext_table(
            data_dirs, Path(cfg.ext_table_file),  # type: ignore[arg-type]
            cfg.filename_col, cfg.label_col,
            cfg.start_col, cfg.end_col, cfg.split_col,
            cfg.obs_id_col, cfg.sound_index_col,
            cfg.xc_id_col,
            remote_cache,
            xc_api_key=cfg.xc_api_key,
            arbimon_stream_id_col=cfg.arbimon_stream_id_col,
            arbimon_date_col=cfg.arbimon_date_col,
            arbimon_time_col=cfg.arbimon_time_col,
            arbimon_utc_offset_col=cfg.arbimon_utc_offset_col,
            arbimon_credentials_path=cfg.arbimon_credentials_path,
        )

    # --- Load existing exported dataset (append) BEFORE any preprocessing ---
    existing: list[AudioSample] = []
    existing_keys: set[tuple[str, str]] = set()
    if cfg.append_dataset_path:
        existing = _load_subfolders(Path(cfg.append_dataset_path), exts)
        for s in existing:
            s.is_appended = True
        existing_keys = {(s.label, s.path.stem) for s in existing}

    def _filter_new(samples: list[AudioSample]) -> list[AudioSample]:
        """Drop samples already present in the existing exported dataset."""
        if not existing_keys:
            return samples
        return [s for s in samples if (s.label, _sample_export_key(s)) not in existing_keys]

    # --- Local data_dir samples ---
    local_samples: list[AudioSample] = []
    if data_dirs and cfg.ssh_host:
        local_samples = _load_ssh_samples(cfg, data_dirs, exts, window_seconds)
        data_dirs = []  # skip local processing below

    if data_dirs:
        if cfg.label_mode == "subfolders":
            for d in data_dirs:
                local_samples.extend(_load_subfolders(d, exts))
            # Filter before preprocessing so we only process files with new samples.
            local_samples = _filter_new(local_samples)
            if needs_preproc and local_samples:
                local_samples = _preprocess_samples(local_samples)
            # Chunk each source file into fixed-size windows, same as other modes.
            rows = [
                _LabelRow(
                    path=s.path, label=s.label,
                    start_time=None, end_time=None,
                    split=s.split, original_path=s.original_path,
                )
                for s in local_samples
            ]
            local_samples = _chunk_rows(rows, window_seconds, cfg.overlap, cfg.min_anchor_fraction)
        else:
            rows = _collect_raw_rows()
            if cfg.append_dataset_path:
                # Chunk first (original times) to know which samples are new,
                # then preprocess only the source files that contribute new chunks.
                candidate = _chunk_rows(rows, window_seconds, cfg.overlap, cfg.min_anchor_fraction)
                candidate = _filter_new(candidate)
                if needs_preproc and candidate:
                    needed_paths = {s.path for s in candidate}
                    local_samples = _preprocess_samples(candidate, needed_paths)
                else:
                    local_samples = candidate
            else:
                if needs_preproc:
                    rows = _preprocess_rows(rows)
                local_samples = _chunk_rows(rows, window_seconds, cfg.overlap, cfg.min_anchor_fraction)

    # --- Remote table samples (iNaturalist, Xeno-canto, Arbimon, or mixed) ---
    ext_samples: list[AudioSample] = []
    if cfg.ext_table_file:
        ext_rows = _load_ext_rows()
        if cfg.append_dataset_path:
            candidate = _chunk_rows(ext_rows, window_seconds, cfg.overlap, cfg.min_anchor_fraction)
            candidate = _filter_new(candidate)
            if needs_preproc and candidate:
                needed_paths = {s.path for s in candidate}
                ext_samples = _preprocess_samples(candidate, needed_paths)
            else:
                ext_samples = candidate
        else:
            if needs_preproc:
                ext_rows = _preprocess_rows(ext_rows)
            ext_samples = _chunk_rows(ext_rows, window_seconds, cfg.overlap, cfg.min_anchor_fraction)

    new_samples = local_samples + ext_samples

    if not cfg.append_dataset_path:
        return new_samples

    print(f"  Existing: {len(existing)}  |  New: {len(new_samples)}")
    return existing + new_samples


def _parse_ext_table(
    data_dirs: list[Path],
    table_file: Path,
    filename_col: str,
    label_col: str,
    start_col: str,
    end_col: str,
    split_col: str,
    obs_id_col: str,
    sound_index_col: str,
    xc_id_col: str,
    ext_cache_dir: Path,
    xc_api_key: str | None = None,
    arbimon_stream_id_col: str = "stream_id",
    arbimon_date_col: str = "date",
    arbimon_time_col: str = "time",
    arbimon_utc_offset_col: str = "utc_offset",
    arbimon_credentials_path: str | None = None,
) -> list[_LabelRow]:
    """Parse a table that may contain local-file rows, iNaturalist rows,
    Xeno-canto rows, Arbimon rows, or any combination.

    Row dispatch priority (first non-empty column wins):
      1. ``obs_id_col``          → iNaturalist observation
      2. ``xc_id_col``           → Xeno-canto recording
      3. ``arbimon_stream_id_col`` → Arbimon recording (requires date + time)
      4. ``filename_col``        → local audio file
      Rows with none of the above filled are skipped.
    """
    from bioaccx.inat    import get_audio as inat_get_audio
    from bioaccx.xc      import get_audio as xc_get_audio
    from bioaccx.arbimon import get_audio as arbimon_get_audio

    text = table_file.read_text()
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    rows: list[_LabelRow] = []

    for row in reader:
        obs_id    = row.get(obs_id_col,             "").strip()
        xc_id     = row.get(xc_id_col,              "").strip()
        stream_id = row.get(arbimon_stream_id_col,  "").strip()
        fname     = row.get(filename_col,           "").strip()

        try:
            start = float(row[start_col]) if start_col in row and row[start_col].strip() else None
            end   = float(row[end_col])   if end_col   in row and row[end_col].strip()   else None
        except ValueError:
            continue
        if start is not None and end is not None and end <= start:
            continue

        split     = row.get(split_col, "").strip() or None
        label_raw = row.get(label_col, "").strip()

        if obs_id:
            # iNaturalist observation
            idx_raw = row.get(sound_index_col, "").strip()
            sound_index = int(idx_raw) if idx_raw.lstrip("-").isdigit() else 0
            try:
                audio_path, scientific_name = inat_get_audio(obs_id, sound_index, ext_cache_dir)
            except Exception as exc:
                print(f"  [skip] iNat obs {obs_id} sound {sound_index}: {exc}")
                continue
            rows.append(_LabelRow(
                path=audio_path, label=label_raw or scientific_name,
                start_time=start, end_time=end, split=split,
            ))

        elif xc_id:
            # Xeno-canto recording
            try:
                audio_path, scientific_name = xc_get_audio(xc_id, ext_cache_dir, xc_api_key)
            except Exception as exc:
                print(f"  [skip] XC{xc_id}: {exc}")
                continue
            rows.append(_LabelRow(
                path=audio_path, label=label_raw or scientific_name,
                start_time=start, end_time=end, split=split,
            ))

        elif stream_id:
            # Arbimon recording
            date_val       = row.get(arbimon_date_col,       "").strip()
            time_val       = row.get(arbimon_time_col,       "").strip()
            utc_offset_val = row.get(arbimon_utc_offset_col, "0").strip() or "0"
            if not date_val or not time_val:
                print(
                    f"  [skip] Arbimon row stream={stream_id!r}: "
                    f"missing {arbimon_date_col!r} or {arbimon_time_col!r}"
                )
                continue
            try:
                audio_path, default_label = arbimon_get_audio(
                    stream_id, date_val, time_val, utc_offset_val,
                    ext_cache_dir, arbimon_credentials_path,
                )
            except Exception as exc:
                print(f"  [skip] Arbimon stream={stream_id!r} {date_val} {time_val}: {exc}")
                continue
            rows.append(_LabelRow(
                path=audio_path, label=label_raw or default_label,
                start_time=start, end_time=end, split=split,
            ))

        elif fname:
            # Local file
            if Path(fname).is_absolute():
                path = Path(fname)
            else:
                path = next(
                    (d / fname for d in data_dirs if (d / fname).exists()),
                    data_dirs[0] / fname if data_dirs else Path(fname),
                )
            if not label_raw:
                print(f"  [skip] local row {fname}: no label")
                continue
            rows.append(_LabelRow(
                path=path, label=label_raw,
                start_time=start, end_time=end, split=split,
            ))

        else:
            print(
                f"  [skip] row has none of {obs_id_col!r}, {xc_id_col!r}, "
                f"{arbimon_stream_id_col!r}, {filename_col!r}"
            )

    return rows


def _is_audio(f: Path, exts: frozenset[str]) -> bool:
    """Return True if *f* is a regular file whose extension is in *exts*."""
    return f.is_file() and f.suffix.lower() in exts


def _load_subfolders(data_dir: Path, exts: frozenset[str]) -> list[AudioSample]:
    """Collect AudioSamples from a subfolder-organised dataset directory.

    Supports two layouts:
      1. Pre-split: ``data_dir/train/<class>/`` and ``data_dir/test/<class>/``.
         When both ``train`` and ``test`` subdirectories exist, sample split
         assignments are taken from the directory names.
      2. Flat: ``data_dir/<class>/``.  No split is assigned; split_samples()
         will assign one later via stratified shuffling.

    Files are collected in sorted order for reproducibility across platforms.
    """
    # Check for train/test top-level split
    split_dirs = {
        s: data_dir / s
        for s in ("train", "test")
        if (data_dir / s).is_dir()
    }
    if split_dirs:
        samples: list[AudioSample] = []
        for split_name, split_dir in split_dirs.items():
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for f in sorted(class_dir.iterdir()):
                    if _is_audio(f, exts):
                        samples.append(
                            AudioSample(path=f, label=class_dir.name, split=split_name)
                        )
        return samples

    # Single-level: class subfolders directly under data_dir
    samples = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for f in sorted(class_dir.iterdir()):
            if _is_audio(f, exts):
                samples.append(AudioSample(path=f, label=class_dir.name))
    return samples


@dataclass
class _LabelRow:
    """Raw label row before chunking."""
    path: Path
    label: str
    start_time: Optional[float]  # None → start of file (0.0)
    end_time: Optional[float]    # None → end of file
    split: Optional[str] = None
    original_path: Optional[Path] = None  # set when path was remapped by preprocessing


def _parse_file_per_label(data_dir: Path, exts: frozenset[str]) -> list[_LabelRow]:
    """Parse paired .txt label files. Each txt row: start_time, end_time, label."""
    rows: list[_LabelRow] = []
    for audio_file in sorted(data_dir.iterdir()):
        if not _is_audio(audio_file, exts):
            continue
        label_file = audio_file.with_suffix(".txt")
        if not label_file.exists():
            print(f"  [skip] no label file for {audio_file.name} — ignoring")
            continue
        text = label_file.read_text().strip()
        if not text:
            print(f"  [skip] empty label file for {audio_file.name} — ignoring")
            continue
        delimiter = "\t" if "\t" in text.splitlines()[0] else ","
        n_before = len(rows)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            if len(parts) < 3:
                continue
            start_str, end_str, label = parts[0].strip(), parts[1].strip(), parts[-1].strip()
            try:
                start, end = float(start_str), float(end_str)
            except ValueError:
                continue  # header row or malformed line
            if label and end > start:
                rows.append(_LabelRow(path=audio_file, label=label, start_time=start, end_time=end))
        if len(rows) == n_before:
            print(f"  [skip] no valid rows in {label_file.name} — ignoring {audio_file.name}")
    return rows


def _parse_table(
    data_dirs: list[Path],
    table_file: Path,
    filename_col: str,
    label_col: str,
    start_col: str,
    end_col: str,
    split_col: str,
) -> list[_LabelRow]:
    """Parse a CSV/TSV annotation table into _LabelRow objects.

    Relative filenames are resolved against each directory in *data_dirs* in
    order; the first match wins.  If no match is found, the filename is
    combined with the first data_dir (which will likely fail at embed time,
    making the error visible).

    Rows with non-numeric start/end or where end <= start are silently skipped
    (these are typically header-row misparses or malformed entries).
    """
    text = table_file.read_text()
    # Auto-detect delimiter from the first line so both TSV and CSV are supported.
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    rows: list[_LabelRow] = []
    for row in reader:
        fname = row[filename_col].strip()
        if Path(fname).is_absolute():
            path = Path(fname)
        else:
            # Walk data_dirs in order; fall back to the first dir if not found.
            path = next(
                (d / fname for d in data_dirs if (d / fname).exists()),
                data_dirs[0] / fname,
            )
        try:
            start = float(row[start_col]) if start_col in row and row[start_col].strip() else None
            end   = float(row[end_col])   if end_col   in row and row[end_col].strip()   else None
        except ValueError:
            continue
        if start is not None and end is not None and end <= start:
            continue
        split = row.get(split_col, "").strip() or None
        rows.append(_LabelRow(
            path=path,
            label=row[label_col].strip(),
            start_time=start,
            end_time=end,
            split=split,
        ))
    return rows


def _chunk_rows(
    rows: list[_LabelRow],
    window: float,
    overlap: float,
    min_anchor_fraction: float = 0.1,
) -> list[AudioSample]:
    """Expand each label row into fixed-window AudioSamples.

    For each row [start, end]:
      - If duration < window: one padded chunk [start, start + window]
      - Otherwise: sliding windows with step = window * (1 - overlap)
        The last window is anchored so it ends at row.end_time; if it would
        be a duplicate of the previous one it is skipped.

    start_time=None means beginning of file (0.0).
    end_time=None means end of file (resolved via soundfile header).
    """
    import soundfile as sf

    step = window * (1.0 - max(0.0, min(overlap, 0.999)))
    samples: list[AudioSample] = []

    for r in rows:
        start_time = r.start_time if r.start_time is not None else 0.0
        if r.end_time is not None:
            end_time = r.end_time
        else:
            try:
                end_time = sf.info(str(r.path)).duration
            except Exception as exc:
                print(f"  [skip] {r.path.name}: cannot read duration — {exc}")
                continue

        duration = end_time - start_time
        if duration <= 0:
            continue

        if duration < window:
            # Single chunk; embedder will zero-pad. Record actual duration so
            # that random_sample_shift can place the signal at a random offset.
            samples.append(AudioSample(
                path=r.path, label=r.label,
                start_time=start_time, end_time=start_time + window,
                split=r.split, original_path=r.original_path,
                signal_duration_seconds=duration,
            ))
            continue

        pos = start_time
        while pos < end_time:
            chunk_end = pos + window
            samples.append(AudioSample(
                path=r.path, label=r.label,
                start_time=round(pos, 6), end_time=round(chunk_end, 6),
                split=r.split, original_path=r.original_path,
            ))
            pos += step
            # Avoid a tiny sliver at the end: if what remains is less than
            # half a step, anchor a final window ending at end_time.
            # Only add the anchor when it brings at least 10% new content
            # (i.e. content past the end of the previous window) to avoid
            # near-duplicate chunks for files that are just barely over the
            # window length due to audio encoding (e.g. 3.013 s with a 3 s window).
            if pos < end_time and (end_time - pos) < step / 2:
                final_start = end_time - window
                new_fraction = (end_time - pos) / window
                if round(final_start, 6) > round(pos - step, 6) and new_fraction >= min_anchor_fraction:
                    samples.append(AudioSample(
                        path=r.path, label=r.label,
                        start_time=round(final_start, 6),
                        end_time=round(end_time, 6),
                        split=r.split, original_path=r.original_path,
                    ))
                break

    return samples


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_samples(
    samples: list[AudioSample],
    test_ratio: float,
    random_seed: int,
) -> tuple[list[AudioSample], list[AudioSample]]:
    """Return (train, test) lists.

    - All samples predefined → use as-is.
    - No samples predefined  → stratified auto-split.
    - Mixed (append workflow) → keep predefined assignments, stratified
      auto-split only the unassigned samples, then merge.
    """
    with_split    = [s for s in samples if s.split in ("train", "test")]
    without_split = [s for s in samples if s.split not in ("train", "test")]

    if not without_split:
        train = [s for s in with_split if s.split == "train"]
        test  = [s for s in with_split if s.split == "test"]
        print(f"  Using predefined split: train={len(train)}, test={len(test)}")
        return train, test

    if not with_split:
        return _auto_split(samples, test_ratio, random_seed)

    # Mixed: predefined splits for existing samples + auto-split new ones
    print(
        f"  {len(with_split)} samples with predefined split, "
        f"{len(without_split)} new sample(s) to be auto-split"
    )
    new_train, new_test = _auto_split(without_split, test_ratio, random_seed)
    train = [s for s in with_split if s.split == "train"] + new_train
    test  = [s for s in with_split if s.split == "test"]  + new_test
    print(f"  Total — train={len(train)}, test={len(test)}")
    return train, test


def _auto_split(
    samples: list[AudioSample],
    test_ratio: float,
    random_seed: int,
) -> tuple[list[AudioSample], list[AudioSample]]:
    """Stratified random split that preserves class proportions in both subsets.

    Uses sklearn's StratifiedShuffleSplit with a single split so that the
    random seed fully determines the result without fitting a cross-validator.
    The dummy X (zeros) is required by the sklearn API but is otherwise ignored.
    """
    labels = [s.label for s in samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    return [samples[i] for i in train_idx], [samples[i] for i in test_idx]


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _parse_augmentation_labels(label_path: Path) -> list[tuple[float, float]]:
    """Parse an Audacity label file; return (start, end) for every valid region."""
    regions: list[tuple[float, float]] = []
    try:
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    start, end = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                if end > start:
                    regions.append((start, end))
    except Exception:
        pass
    return regions


def _build_concatenated_noise(aug_dir: Path, sample_rate: int) -> Optional[Path]:
    """Concatenate all augmentation audio into a single temp WAV file.

    For each audio file found recursively under *aug_dir*:
    - If a sibling Audacity .txt label file exists, only the labeled segments
      are used (regardless of label text).
    - Otherwise the entire file is included.

    Returns the path to the written temp WAV, or None if nothing could be loaded.
    """
    import hashlib
    import soundfile as sf

    audio_files = sorted(
        f for f in aug_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        return None

    segments: list[np.ndarray] = []
    for audio_file in audio_files:
        label_file = audio_file.with_suffix(".txt")
        if label_file.exists():
            regions = _parse_augmentation_labels(label_file)
            if regions:
                for start, end in regions:
                    try:
                        seg = load_mono(audio_file, sample_rate, offset=start,
                                        duration=end - start)
                        if len(seg) > 0:
                            segments.append(seg)
                    except Exception as exc:
                        print(f"  [augmentation] error loading segment from {audio_file.name}: {exc}")
                continue
        # No label file (or empty): include the entire file.
        try:
            seg = load_mono(audio_file, sample_rate)
            if len(seg) > 0:
                segments.append(seg)
        except Exception as exc:
            print(f"  [augmentation] error loading {audio_file.name}: {exc}")

    if not segments:
        return None

    concatenated = np.concatenate(segments)
    total_dur = len(concatenated) / sample_rate

    hash_tag = hashlib.md5(str(aug_dir.resolve()).encode()).hexdigest()[:8]
    temp_path = _get_preproc_tempdir() / f"aug_concat_{hash_tag}_{sample_rate}hz.wav"
    sf.write(str(temp_path), concatenated, sample_rate)

    print(
        f"  [augmentation] concatenated {len(segments)} segment(s) from "
        f"{aug_dir.name} → {total_dur:.1f}s"
    )
    return temp_path


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def apply_augmentation(
    samples: list[AudioSample],
    aug_cfg,               # AugmentationConfig
    window_seconds: float,
    sample_rate: int,
    random_seed: int,
) -> list[AudioSample]:
    """Expand *samples* by mixing each one with every noise file at every SNR level.

    For each (sample, noise_file, snr) triplet one augmented AudioSample is created.
    The noise start offset is derived deterministically from a stable hash of
    (random_seed, sample path, start/end, noise filename, snr) so that results are
    reproducible across runs regardless of iteration order.

    If ``aug_cfg.keep_original`` is True the original clean samples are prepended
    to the returned list.

    When ``aug_cfg.concatenate_augmentation_dir`` is True, all files in
    augmentation_dir are concatenated into a single in-memory noise source
    (labeled segments only when a sibling .txt file exists; full file otherwise).

    Only train samples should be passed; the test set is never augmented.
    """
    import hashlib
    import soundfile as sf

    aug_dir = Path(aug_cfg.augmentation_dir)

    if getattr(aug_cfg, "concatenate_augmentation_dir", False):
        concat_path = _build_concatenated_noise(aug_dir, sample_rate)
        if concat_path is None:
            print(f"  [augmentation] could not build concatenated noise from {aug_dir} — skipping")
            return samples
        noise_files = [concat_path]
        noise_durations: dict[Path, float] = {concat_path: sf.info(str(concat_path)).duration}
    else:
        noise_files = sorted(
            f for f in aug_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        )

        if not noise_files:
            print(f"  [augmentation] no audio files found in {aug_dir} — skipping")
            return samples

        noise_durations = {}
        for nf in noise_files:
            try:
                noise_durations[nf] = sf.info(str(nf)).duration
            except Exception as exc:
                print(f"  [augmentation] cannot read {nf.name}: {exc} — skipping")

    skip_labels: set[str] = set(aug_cfg.skip_labels) if aug_cfg.skip_labels else set()
    skipped = [s for s in samples if s.label in skip_labels]
    to_augment = [s for s in samples if s.label not in skip_labels]

    result: list[AudioSample] = list(samples) if aug_cfg.keep_original else list(skipped)

    for s in to_augment:
        for nf in noise_files:
            if nf not in noise_durations:
                continue
            max_offset = max(0.0, noise_durations[nf] - window_seconds)
            for snr in aug_cfg.snr_levels:
                key = f"{random_seed}:{s.path}:{s.start_time}:{s.end_time}:{nf.name}:{snr}"
                seed_int = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed_int)
                noise_offset = float(rng.uniform(0.0, max_offset)) if max_offset > 0.0 else 0.0
                result.append(AudioSample(
                    path=s.path,
                    label=s.label,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    split=s.split,
                    is_appended=s.is_appended,
                    original_path=s.original_path,
                    noise_path=nf,
                    snr=float(snr),
                    noise_start_time=noise_offset,
                    signal_duration_seconds=s.signal_duration_seconds,
                ))

    n_aug = len(result) - (len(samples) if aug_cfg.keep_original else len(skipped))
    if skip_labels:
        print(f"  [augmentation] skipping augmentation for labels: {sorted(skip_labels)}")
    print(
        f"  [augmentation] {len(to_augment)} clean × {len(noise_durations)} noise file(s) "
        f"× {len(aug_cfg.snr_levels)} SNR level(s) → {n_aug} augmented"
        + (f" + {len(samples)} originals" if aug_cfg.keep_original else "")
        + f" = {len(result)} total"
    )
    return result


# ---------------------------------------------------------------------------
# Random sample shift
# ---------------------------------------------------------------------------

def apply_random_shifts(
    samples: list[AudioSample],
    window_seconds: float,
    sample_rate: int,
    random_seed: int,
) -> list[AudioSample]:
    """Assign a random placement offset to samples shorter than the window.

    For each sample whose ``signal_duration_seconds`` is set (i.e. the audio is
    shorter than the foundation model window), a random integer offset in
    [0, window_n - n_signal] is computed deterministically from a stable hash of
    (random_seed, path, start/end, noise_path, snr).  Augmented copies of the
    same clean sample therefore receive distinct offsets because the noise_path
    and snr fields differ.

    Samples whose signal fills the entire window are left unchanged.
    """
    import hashlib

    window_n = round(window_seconds * sample_rate)
    for s in samples:
        if s.signal_duration_seconds is None:
            continue
        n_signal = round(s.signal_duration_seconds * sample_rate)
        max_offset = window_n - n_signal
        if max_offset <= 0:
            continue
        key = (
            f"{random_seed}:{s.path}:{s.start_time}:{s.end_time}"
            f":{s.noise_path}:{s.snr}"
        )
        seed_int = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        s.signal_offset_samples = int(rng.integers(0, max_offset + 1))
    return samples


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _npy_filename(s: AudioSample) -> str:
    """Canonical .npy filename for a sample, unique across start/end times and augmentation."""
    stem = s.path.stem
    if s.start_time is not None and s.end_time is not None:
        base = f"{stem}_{s.start_time:.3f}_{s.end_time:.3f}"
    else:
        base = stem
    if s.signal_offset_samples is not None:
        base = f"{base}_off{s.signal_offset_samples}"
    if s.noise_path is not None and s.snr is not None:
        return f"{base}_noise_{s.noise_path.stem}_snr{s.snr:g}.npy"
    return f"{base}.npy"


def _extract_embeddings_onnx_batch(
    samples: list[AudioSample],
    embedder,
    embedding_size: int,
    batch_size: int,
    cache_dir: Optional[Path] = None,
    export_dir: Optional[Path] = None,
    cache_sqlite: Optional[Path] = None,
    export_sqlite: Optional[Path] = None,
    ssh_config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """GPU batch inference path for ONNXEmbedder.

    Phase 1 loads and preprocesses audio sequentially (cache hits are skipped).
    Phase 2 feeds uncached audio to the GPU in chunks of *batch_size*.
    """
    import time
    from bioaccx.embeddings_sqlite import init_db, load_embedding, save_embedding

    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
    if export_sqlite is not None:
        init_db(export_sqlite)

    label_names = sorted(set(s.label for s in samples))
    label_to_idx = {n: i for i, n in enumerate(label_names)}
    total = len(samples)
    window_n = embedder._window
    sr = embedder.cfg.sample_rate

    # SSH: open a single client for the batch path (sequential downloads)
    ssh_temp: dict[str, Path] = {}
    _sftp = None
    _ssh_client = None
    if ssh_config:
        from bioaccx.ssh import open_sftp_client
        _ssh_client, _sftp = open_sftp_client(**ssh_config)

    def _resolve_local(s: AudioSample) -> Path:
        if not s.ssh_path:
            return s.path
        if s.ssh_path not in ssh_temp:
            from bioaccx.ssh import download_to_temp
            print(f"  [ssh] ↓ {Path(s.ssh_path).name}")
            ssh_temp[s.ssh_path] = download_to_temp(_sftp, s.ssh_path)
        return ssh_temp[s.ssh_path]

    result_map: dict[int, tuple[Optional[np.ndarray], int]] = {}
    # (rank, audio_array, label_idx, npy_name, db_key, log_prefix)
    pending: list[tuple] = []

    # ------------------------------------------------------------------ #
    # Phase 1: cache lookup + audio loading                               #
    # ------------------------------------------------------------------ #
    for rank, s in enumerate(samples, 1):
        npy_name = _npy_filename(s)
        db_key = npy_name[:-4] if npy_name.endswith(".npy") else npy_name
        start_tag = f"{s.start_time:.2f}s" if s.start_time is not None else "0.00s"
        end_tag   = f"{s.end_time:.2f}s"   if s.end_time   is not None else "full"
        label_idx = label_to_idx[s.label]
        log_pfx   = f"  [{rank}/{total}] {s.path.name} [{start_tag}–{end_tag}]"

        if cache_sqlite is not None:
            try:
                emb = load_embedding(cache_sqlite, db_key)
                if emb is not None and emb.shape == (embedding_size,):
                    print(f"{log_pfx}  cached (sqlite)")
                    result_map[rank] = (emb, label_idx)
                    continue
                if emb is not None:
                    print(f"  [cache mismatch] {db_key}: shape {emb.shape}, recomputing")
            except Exception as exc:
                print(f"  [cache error] {db_key}: {exc}, recomputing")

        if cache_dir is not None:
            cached = cache_dir / npy_name
            if cached.exists():
                try:
                    emb = np.load(cached).astype(np.float32)
                    if emb.shape == (embedding_size,):
                        print(f"{log_pfx}  cached")
                        result_map[rank] = (emb, label_idx)
                        continue
                    print(f"  [cache mismatch] {npy_name}: shape {emb.shape}, recomputing")
                except Exception as exc:
                    print(f"  [cache error] {npy_name}: {exc}, recomputing")

        local_path = _resolve_local(s)
        try:
            if local_path.suffix.lower() == ".npy":
                emb = np.load(local_path).astype(np.float32)
                if emb.shape != (embedding_size,):
                    print(f"  [skip] {s.path.name}: shape {emb.shape} != ({embedding_size},)")
                    result_map[rank] = (None, label_idx)
                else:
                    result_map[rank] = (emb, label_idx)
                continue

            offset = s.start_time or 0.0
            if s.signal_duration_seconds is not None or s.signal_offset_samples is not None:
                if s.signal_duration_seconds is not None:
                    signal = load_mono(local_path, sr, offset=offset,
                                       duration=s.signal_duration_seconds)
                    signal = place_in_window(signal, window_n,
                                            s.signal_offset_samples or 0)
                else:
                    sig_dur = (s.end_time - offset) if s.end_time is not None else None
                    signal = load_mono(local_path, sr, offset=offset, duration=sig_dur)
                    signal = to_fixed_length(signal, window_n)
                if s.noise_path is not None and s.snr is not None:
                    noise = load_mono(s.noise_path, sr,
                                      offset=s.noise_start_time or 0.0,
                                      duration=window_n / sr)
                    noise = to_fixed_length(noise, window_n)
                    audio = mix_at_snr(signal, noise, s.snr)
                else:
                    audio = signal
            else:
                duration = (s.end_time - offset) if s.end_time is not None else None
                audio = load_mono(local_path, sr, offset=offset, duration=duration)
                audio = to_fixed_length(audio, window_n)
        except Exception as exc:
            print(f"  [skip] {s.path.name}: {exc}")
            result_map[rank] = (None, label_idx)
            continue

        pending.append((rank, audio, label_idx, npy_name, db_key, log_pfx))

    for tmp in ssh_temp.values():
        tmp.unlink(missing_ok=True)
    if _sftp:
        _sftp.close()
    if _ssh_client:
        _ssh_client.close()

    # ------------------------------------------------------------------ #
    # Phase 2: batched GPU inference                                      #
    # ------------------------------------------------------------------ #
    print(f"  [batch] {len(pending)} sample(s) to infer — batch_size={batch_size}")
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        audio_batch = np.stack([a for _, a, *_ in chunk])
        t0 = time.perf_counter()
        try:
            embs = embedder.embed_batch(audio_batch)
        except Exception as exc:
            print(f"  [batch error] batch starting at {i}: {exc}")
            for rank, _, label_idx, *_ in chunk:
                result_map[rank] = (None, label_idx)
            continue
        elapsed = time.perf_counter() - t0
        per_sample = elapsed / len(chunk)
        for (rank, _, label_idx, npy_name, db_key, log_pfx), emb in zip(chunk, embs):
            print(f"{log_pfx}  {per_sample:.3f}s")
            result_map[rank] = (emb, label_idx)
            if export_sqlite is not None:
                try:
                    save_embedding(export_sqlite, db_key, emb)
                except Exception as exc:
                    print(f"  [export error] {db_key}: {exc}")
            if export_dir is not None:
                try:
                    np.save(export_dir / npy_name, emb)
                except Exception as exc:
                    print(f"  [export error] {npy_name}: {exc}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for rank in range(1, total + 1):
        emb, label_idx = result_map.get(rank, (None, 0))
        if emb is not None:
            X_list.append(emb)
            y_list.append(label_idx)

    if not X_list:
        raise RuntimeError(
            "No embeddings were computed — all samples were skipped. "
            "Check that the audio files are accessible and valid."
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, label_names


def extract_embeddings(
    samples: list[AudioSample],
    embedder_cfg,                           # FoundationModelConfig
    embedding_size: int,
    n_workers: int = 4,
    cache_dir: Optional[Path] = None,       # load existing .npy from here
    export_dir: Optional[Path] = None,      # save newly computed .npy here
    cache_sqlite: Optional[Path] = None,    # load existing embeddings from a .db file
    export_sqlite: Optional[Path] = None,   # save newly computed embeddings to a .db file
    ssh_config: Optional[dict] = None,      # SSH credentials for on-demand download
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the embedder over all samples in parallel; return X, y (int), label_names.

    For each sample the lookup order is:
      1. cache_sqlite[key]          — SQLite DB (if cache_sqlite is set)
      2. cache_dir/<npy_filename>   — .npy file  (if cache_dir is set)
      3. Compute with the foundation model
      4. Optionally save to export_sqlite or export_dir

    Each worker thread creates its own embedder instance so that non-thread-safe
    backends (TFLite) are safe. ONNX sessions release the GIL during inference,
    so threads provide genuine parallelism.

    When embedder_cfg.format=="onnx" and embedder_cfg.onnx_batch_size > 1, a
    single-threaded GPU batch path is used instead: audio is loaded sequentially
    then fed to the GPU in chunks of onnx_batch_size.

    SSH samples (ssh_path set): each source file is downloaded to a temp file
    once, used for all its chunks, then deleted.  One temp file per source file
    exists on disk at a time; files from different source files may overlap.
    """
    import os
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from bioaccx.embedder import load_embedder
    from bioaccx.embeddings_sqlite import init_db, load_embedding, save_embedding

    if getattr(embedder_cfg, "format", None) == "onnx" and getattr(embedder_cfg, "onnx_batch_size", 1) > 1:
        embedder = load_embedder(embedder_cfg)
        print(f"  GPU batch mode — batch_size={embedder_cfg.onnx_batch_size}")
        return _extract_embeddings_onnx_batch(
            samples, embedder, embedding_size,
            batch_size=embedder_cfg.onnx_batch_size,
            cache_dir=cache_dir, export_dir=export_dir,
            cache_sqlite=cache_sqlite, export_sqlite=export_sqlite,
            ssh_config=ssh_config,
        )

    available = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, available))
    print(f"  workers={n_workers} (machine has {available} logical cores)")

    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    if export_sqlite is not None:
        init_db(export_sqlite)

    label_names = sorted(set(s.label for s in samples))
    label_to_idx = {n: i for i, n in enumerate(label_names)}
    total = len(samples)

    _local = threading.local()

    def _get_embedder():
        if not hasattr(_local, "emb"):
            _local.emb = load_embedder(embedder_cfg)
        return _local.emb

    # ------------------------------------------------------------------
    # SSH: per-source-file ref-counted temp download
    # ------------------------------------------------------------------
    # _ssh_temp  : remote_path → local temp Path (once downloaded)
    # _ssh_refs  : remote_path → number of chunks still to be processed
    # _ssh_plocks: per-path lock so only one thread downloads each file
    # _ssh_clients: all (SSHClient, SFTPClient) pairs opened by worker threads
    # ------------------------------------------------------------------
    _ssh_temp: dict[str, Path] = {}
    _ssh_refs: dict[str, int] = {}
    _ssh_plocks: dict[str, threading.Lock] = {}
    _ssh_plocks_lock = threading.Lock()
    _ssh_refs_lock = threading.Lock()
    _ssh_clients: list[tuple] = []
    _ssh_clients_lock = threading.Lock()

    if ssh_config:
        for s in samples:
            if s.ssh_path:
                _ssh_refs[s.ssh_path] = _ssh_refs.get(s.ssh_path, 0) + 1

    def _get_sftp():
        if not hasattr(_local, "sftp"):
            from bioaccx.ssh import open_sftp_client
            ssh_c, sftp = open_sftp_client(**ssh_config)
            _local.sftp = sftp
            _local.ssh_client = ssh_c
            with _ssh_clients_lock:
                _ssh_clients.append((ssh_c, sftp))
        return _local.sftp

    def _get_path_lock(path: str) -> threading.Lock:
        with _ssh_plocks_lock:
            if path not in _ssh_plocks:
                _ssh_plocks[path] = threading.Lock()
            return _ssh_plocks[path]

    def _resolve_local_path(s: AudioSample) -> Path:
        """Return a local path for *s*, downloading the remote file if needed."""
        if not s.ssh_path:
            return s.path
        sftp = _get_sftp()
        with _get_path_lock(s.ssh_path):
            if s.ssh_path not in _ssh_temp:
                from bioaccx.ssh import download_to_temp
                print(f"  [ssh] ↓ {Path(s.ssh_path).name}")
                _ssh_temp[s.ssh_path] = download_to_temp(sftp, s.ssh_path)
        return _ssh_temp[s.ssh_path]

    def _release_ssh(s: AudioSample) -> None:
        """Decrement ref count; delete the temp file once all its chunks are done."""
        if not s.ssh_path:
            return
        to_delete = None
        with _ssh_refs_lock:
            _ssh_refs[s.ssh_path] -= 1
            if _ssh_refs[s.ssh_path] == 0:
                to_delete = _ssh_temp.pop(s.ssh_path, None)
                _ssh_plocks.pop(s.ssh_path, None)
        if to_delete is not None:
            to_delete.unlink(missing_ok=True)

    def _process(args):
        rank, s = args
        npy_name = _npy_filename(s)
        db_key = npy_name[:-4] if npy_name.endswith(".npy") else npy_name
        start_tag = f"{s.start_time:.2f}s" if s.start_time is not None else "0.00s"
        end_tag   = f"{s.end_time:.2f}s"   if s.end_time   is not None else "full"

        # 1a. Check SQLite cache
        if cache_sqlite is not None:
            try:
                emb = load_embedding(cache_sqlite, db_key)
                if emb is not None:
                    if emb.shape == (embedding_size,):
                        print(f"  [{rank}/{total}] {s.path.name} [{start_tag}–{end_tag}]  cached (sqlite)")
                        _release_ssh(s)
                        return rank, emb, label_to_idx[s.label]
                    print(f"  [cache mismatch] {db_key}: shape {emb.shape}, recomputing")
            except Exception as exc:
                print(f"  [cache error] {db_key}: {exc}, recomputing")

        # 1b. Check .npy cache
        if cache_dir is not None:
            cached = cache_dir / npy_name
            if cached.exists():
                try:
                    emb = np.load(cached).astype(np.float32)
                    if emb.shape == (embedding_size,):
                        print(f"  [{rank}/{total}] {s.path.name} [{start_tag}–{end_tag}]  cached")
                        _release_ssh(s)
                        return rank, emb, label_to_idx[s.label]
                    print(f"  [cache mismatch] {npy_name}: shape {emb.shape}, recomputing")
                except Exception as exc:
                    print(f"  [cache error] {npy_name}: {exc}, recomputing")

        # 2. Compute — download SSH file if needed
        local_path = _resolve_local_path(s)
        t0 = time.perf_counter()
        try:
            if local_path.suffix.lower() == ".npy":
                emb = np.load(local_path).astype(np.float32)
                if emb.shape != (embedding_size,):
                    print(f"  [skip] {s.path.name}: shape {emb.shape} != ({embedding_size},)")
                    _release_ssh(s)
                    return rank, None, label_to_idx[s.label]
            elif s.noise_path is not None or s.signal_offset_samples is not None:
                embedder = _get_embedder()
                window_n = embedder._window
                sr = embedder.cfg.sample_rate
                offset = s.start_time or 0.0
                if s.signal_duration_seconds is not None:
                    # Short signal: load only the real audio, then place at offset
                    signal = load_mono(local_path, sr, offset=offset,
                                       duration=s.signal_duration_seconds)
                    signal = place_in_window(signal, window_n,
                                            s.signal_offset_samples or 0)
                else:
                    sig_dur = (s.end_time - offset) if s.end_time is not None else None
                    signal = load_mono(local_path, sr, offset=offset, duration=sig_dur)
                    signal = to_fixed_length(signal, window_n)
                if s.noise_path is not None and s.snr is not None:
                    window_dur = window_n / sr
                    noise = load_mono(
                        s.noise_path, sr,
                        offset=s.noise_start_time or 0.0,
                        duration=window_dur,
                    )
                    noise = to_fixed_length(noise, window_n)
                    emb = embedder.embed(mix_at_snr(signal, noise, s.snr))
                else:
                    emb = embedder.embed(signal)
            else:
                emb = _get_embedder().embed_file(local_path, s.start_time, s.end_time)
        except Exception as exc:
            print(f"  [skip] {s.path.name}: {exc}")
            _release_ssh(s)
            return rank, None, label_to_idx[s.label]

        _release_ssh(s)
        elapsed = time.perf_counter() - t0
        print(f"  [{rank}/{total}] {s.path.name} [{start_tag}–{end_tag}]  {elapsed:.3f}s")

        # 3. Export
        if export_sqlite is not None:
            try:
                save_embedding(export_sqlite, db_key, emb)
            except Exception as exc:
                print(f"  [export error] {db_key}: {exc}")

        if export_dir is not None:
            try:
                np.save(export_dir / npy_name, emb)
            except Exception as exc:
                print(f"  [export error] {npy_name}: {exc}")

        return rank, emb, label_to_idx[s.label]

    result_map: dict[int, tuple[Optional[np.ndarray], int]] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process, (i + 1, s)): i for i, s in enumerate(samples)}
        for fut in as_completed(futures):
            rank, emb, label_idx = fut.result()
            result_map[rank] = (emb, label_idx)

    for ssh_c, sftp_c in _ssh_clients:
        try:
            sftp_c.close()
            ssh_c.close()
        except Exception:
            pass

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for rank in range(1, total + 1):
        emb, label_idx = result_map[rank]
        if emb is not None:
            X_list.append(emb)
            y_list.append(label_idx)

    if not X_list:
        raise RuntimeError(
            "No embeddings were computed — all samples were skipped. "
            "Check that the audio files are accessible and valid."
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, label_names


# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------

def export_dataset_audio(
    train_samples: list[AudioSample],
    test_samples: list[AudioSample],
    out_dir: Path,
    sample_rate: int,
    window_samples: int,
) -> None:
    """Write chunked audio samples as WAV files into label subfolders.

    Layout:
        out_dir/
            train/
                <label>/
                    <stem>_<start>_<end>.wav
            test/
                <label>/
                    <stem>_<start>_<end>.wav
    """
    import shutil
    import soundfile as sf

    splits = [("train", train_samples), ("test", test_samples)]
    total = len(train_samples) + len(test_samples)
    done = 0

    for split_name, samples in splits:
        for s in samples:
            label_dir = out_dir / split_name / s.label
            label_dir.mkdir(parents=True, exist_ok=True)

            if s.is_appended:
                # Already an exported chunk — copy the file verbatim to preserve
                # its original filename (re-exporting would add _0.000_full).
                out_file = label_dir / s.path.name
                try:
                    if out_file.resolve() != s.path.resolve():
                        shutil.copy2(s.path, out_file)
                except Exception as exc:
                    print(f"  [export skip] {s.path.name}: {exc}")
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"  exported {done}/{total} chunks…")
                continue

            start = s.start_time if s.start_time is not None else 0.0
            end   = s.end_time   if s.end_time   is not None else None
            duration = (end - start) if end is not None else None

            stem = _source_stem(s)  # use original source stem, not preprocessed temp-file name
            start_tag = f"{start:.3f}"
            end_tag   = f"{end:.3f}" if end is not None else "full"

            if s.noise_path is not None and s.snr is not None:
                noise_tag = f"_noise_{s.noise_path.stem}_snr{s.snr:g}"
                out_file = label_dir / f"{stem}_{start_tag}_{end_tag}{noise_tag}.wav"
            else:
                out_file = label_dir / f"{stem}_{start_tag}_{end_tag}.wav"

            try:
                if s.signal_duration_seconds is not None:
                    audio = load_mono(s.path, sample_rate, offset=start,
                                      duration=s.signal_duration_seconds)
                    audio = place_in_window(audio, window_samples,
                                            s.signal_offset_samples or 0)
                else:
                    audio = load_mono(s.path, sample_rate, offset=start,
                                      duration=duration)
                    audio = to_fixed_length(audio, window_samples)
                if s.noise_path is not None and s.snr is not None:
                    noise_dur = window_samples / sample_rate
                    noise = load_mono(
                        s.noise_path, sample_rate,
                        offset=s.noise_start_time or 0.0,
                        duration=noise_dur,
                    )
                    noise = to_fixed_length(noise, window_samples)
                    audio = mix_at_snr(audio, noise, s.snr)
                sf.write(str(out_file), audio, sample_rate)
            except Exception as exc:
                print(f"  [export skip] {s.path.name}: {exc}")
                done += 1
                continue

            done += 1
            if done % 100 == 0 or done == total:
                print(f"  exported {done}/{total} chunks…")

    print(f"  Dataset exported → {out_dir}")
