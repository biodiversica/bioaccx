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

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from bioaccx.audio import AUDIO_EXTENSIONS
from bioaccx.config import DatasetConfig


@dataclass
class AudioSample:
    path: Path
    label: str
    start_time: Optional[float] = None  # seconds; None = use full window
    end_time: Optional[float] = None
    split: Optional[str] = None         # "train" | "test" | None
    is_appended: bool = False           # True for samples from append_dataset_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _as_dirs(data_dir: str | list[str]) -> list[Path]:
    if isinstance(data_dir, str):
        return [Path(data_dir)] if data_dir else []
    return [Path(d) for d in data_dir if d]


def _sample_export_key(s: AudioSample) -> str:
    """Filename stem that export_dataset_audio would produce for this sample.

    Matches the naming scheme  {stem}_{start:.3f}_{end:.3f}  used by
    export_dataset_audio so that samples already present in an existing exported
    dataset can be identified by comparing path stems.
    """
    start = s.start_time if s.start_time is not None else 0.0
    end_str = f"{s.end_time:.3f}" if s.end_time is not None else "full"
    return f"{s.path.stem}_{start:.3f}_{end_str}"


def load_samples(cfg: DatasetConfig, window_seconds: float) -> list[AudioSample]:
    exts = frozenset(f".{e.lstrip('.')}" for e in cfg.audio_extensions)
    data_dirs = _as_dirs(cfg.data_dir)

    if not data_dirs and not cfg.ext_table_file:
        raise ValueError(
            "dataset.data_dir must be set unless ext_table_file is provided"
        )

    # --- Local data_dir samples ---
    local_samples: list[AudioSample] = []
    if data_dirs:
        if cfg.label_mode == "subfolders":
            for d in data_dirs:
                local_samples.extend(_load_subfolders(d, exts))
        else:
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
            local_samples = _chunk_rows(rows, window_seconds, cfg.overlap)

    # --- Remote table samples (iNaturalist, Xeno-canto, or mixed with local) ---
    ext_samples: list[AudioSample] = []
    if cfg.ext_table_file:
        remote_cache = (
            Path(cfg.ext_cache_dir)
            if cfg.ext_cache_dir
            else Path.home() / ".cache" / "bioaccx" / "ext"
        )
        ext_rows = _parse_ext_table(
            data_dirs, Path(cfg.ext_table_file),
            cfg.filename_col, cfg.label_col,
            cfg.start_col, cfg.end_col, cfg.split_col,
            cfg.obs_id_col, cfg.sound_index_col,
            cfg.xc_id_col,
            remote_cache,
            xc_api_key=cfg.xc_api_key,
        )
        ext_samples = _chunk_rows(ext_rows, window_seconds, cfg.overlap)

    new_samples = local_samples + ext_samples

    if not cfg.append_dataset_path:
        return new_samples

    # --- Append to existing exported dataset ---
    existing = _load_subfolders(Path(cfg.append_dataset_path), exts)
    for s in existing:
        s.is_appended = True
    existing_keys: set[tuple[str, str]] = {(s.label, s.path.stem) for s in existing}
    filtered = [s for s in new_samples if (s.label, _sample_export_key(s)) not in existing_keys]
    n_dupes = len(new_samples) - len(filtered)
    if n_dupes:
        print(f"  {n_dupes} sample(s) already in existing dataset — skipped")
    print(f"  Existing: {len(existing)}  |  New: {len(filtered)}")
    return existing + filtered


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
) -> list[_LabelRow]:
    """Parse a table that may contain local-file rows, iNaturalist rows,
    Xeno-canto rows, or any combination.

    Row dispatch priority (first non-empty column wins):
      1. ``obs_id_col``   → iNaturalist observation
      2. ``xc_id_col``    → Xeno-canto recording
      3. ``filename_col`` → local audio file
      Rows with none of the above filled are skipped.
    """
    from bioaccx.inat import get_audio as inat_get_audio
    from bioaccx.xc   import get_audio as xc_get_audio

    text = table_file.read_text()
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    rows: list[_LabelRow] = []

    for row in reader:
        obs_id = row.get(obs_id_col, "").strip()
        xc_id  = row.get(xc_id_col,  "").strip()
        fname  = row.get(filename_col, "").strip()

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
                f"  [skip] row has none of {obs_id_col!r}, "
                f"{xc_id_col!r}, {filename_col!r}"
            )

    return rows


def _is_audio(f: Path, exts: frozenset[str]) -> bool:
    return f.is_file() and f.suffix.lower() in exts


def _load_subfolders(data_dir: Path, exts: frozenset[str]) -> list[AudioSample]:
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
    text = table_file.read_text()
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    rows: list[_LabelRow] = []
    for row in reader:
        fname = row[filename_col].strip()
        if Path(fname).is_absolute():
            path = Path(fname)
        else:
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
            # Single chunk; embedder will zero-pad
            samples.append(AudioSample(
                path=r.path, label=r.label,
                start_time=start_time, end_time=start_time + window,
                split=r.split,
            ))
            continue

        pos = start_time
        while pos < end_time:
            chunk_end = pos + window
            samples.append(AudioSample(
                path=r.path, label=r.label,
                start_time=round(pos, 6), end_time=round(chunk_end, 6),
                split=r.split,
            ))
            pos += step
            # Avoid a tiny sliver at the end: if what remains is less than
            # half a step, anchor a final window ending at end_time
            if pos < end_time and (end_time - pos) < step / 2:
                final_start = end_time - window
                if round(final_start, 6) > round(pos - step, 6):
                    samples.append(AudioSample(
                        path=r.path, label=r.label,
                        start_time=round(final_start, 6),
                        end_time=round(end_time, 6),
                        split=r.split,
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
    labels = [s.label for s in samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    return [samples[i] for i in train_idx], [samples[i] for i in test_idx]


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _npy_filename(s: AudioSample) -> str:
    """Canonical .npy filename for a sample, unique across start/end times."""
    stem = s.path.stem
    if s.start_time is not None and s.end_time is not None:
        return f"{stem}_{s.start_time:.3f}_{s.end_time:.3f}.npy"
    return f"{stem}.npy"


def extract_embeddings(
    samples: list[AudioSample],
    embedder_cfg,                           # FoundationModelConfig
    embedding_size: int,
    n_workers: int = 4,
    cache_dir: Optional[Path] = None,       # load existing .npy from here
    export_dir: Optional[Path] = None,      # save newly computed .npy here
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the embedder over all samples in parallel; return X, y (int), label_names.

    For each sample the lookup order is:
      1. cache_dir/<npy_filename>  — load and skip inference if present
      2. Compute with the foundation model
      3. Optionally save to export_dir/<npy_filename>

    Each worker thread creates its own embedder instance so that non-thread-safe
    backends (TFLite) are safe. ONNX sessions release the GIL during inference,
    so threads provide genuine parallelism.
    """
    import os
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from bioaccx.embedder import load_embedder

    available = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, available))
    print(f"  workers={n_workers} (machine has {available} logical cores)")

    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    label_names = sorted(set(s.label for s in samples))
    label_to_idx = {n: i for i, n in enumerate(label_names)}
    total = len(samples)

    _local = threading.local()

    def _get_embedder():
        if not hasattr(_local, "emb"):
            _local.emb = load_embedder(embedder_cfg)
        return _local.emb

    def _process(args):
        rank, s = args
        npy_name = _npy_filename(s)
        start_tag = f"{s.start_time:.2f}s" if s.start_time is not None else "0.00s"
        end_tag   = f"{s.end_time:.2f}s"   if s.end_time   is not None else "full"

        # 1. Check cache
        if cache_dir is not None:
            cached = cache_dir / npy_name
            if cached.exists():
                try:
                    emb = np.load(cached).astype(np.float32)
                    if emb.shape == (embedding_size,):
                        print(f"  [{rank}/{total}] {s.path.name} [{start_tag}–{end_tag}]  cached")
                        return rank, emb, label_to_idx[s.label]
                    print(f"  [cache mismatch] {npy_name}: shape {emb.shape}, recomputing")
                except Exception as exc:
                    print(f"  [cache error] {npy_name}: {exc}, recomputing")

        # 2. Compute
        t0 = time.perf_counter()
        try:
            if s.path.suffix.lower() == ".npy":
                emb = np.load(s.path).astype(np.float32)
                if emb.shape != (embedding_size,):
                    print(f"  [skip] {s.path.name}: shape {emb.shape} != ({embedding_size},)")
                    return rank, None, label_to_idx[s.label]
            else:
                emb = _get_embedder().embed_file(s.path, s.start_time, s.end_time)
        except Exception as exc:
            print(f"  [skip] {s.path.name}: {exc}")
            return rank, None, label_to_idx[s.label]

        elapsed = time.perf_counter() - t0
        print(f"  [{rank}/{total}] {s.path.name} [{start_tag}–{end_tag}]  {elapsed:.3f}s")

        # 3. Export
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

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for rank in range(1, total + 1):
        emb, label_idx = result_map[rank]
        if emb is not None:
            X_list.append(emb)
            y_list.append(label_idx)

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
    from bioaccx.audio import load_mono, to_fixed_length

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

            stem = s.path.stem
            start_tag = f"{start:.3f}"
            end_tag   = f"{end:.3f}" if end is not None else "full"
            out_file  = label_dir / f"{stem}_{start_tag}_{end_tag}.wav"

            try:
                audio = load_mono(s.path, sample_rate, offset=start, duration=duration)
                audio = to_fixed_length(audio, window_samples)
                sf.write(str(out_file), audio, sample_rate)
            except Exception as exc:
                print(f"  [export skip] {s.path.name}: {exc}")
                done += 1
                continue

            done += 1
            if done % 100 == 0 or done == total:
                print(f"  exported {done}/{total} chunks…")

    print(f"  Dataset exported → {out_dir}")
