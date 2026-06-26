"""Unit tests for multi-source dataset orchestration (train.load_and_prepare_blocks)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from bioaccx.config import _parse_config
from bioaccx.train import load_and_prepare_blocks

SR = 16_000
WINDOW_SEC = 1.0


def _wav(path: Path, duration: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(duration * SR), dtype=np.float32), SR)


def _subfolders(root: Path, classes: dict[str, int]) -> Path:
    """Create a flat subfolders dataset: ``root/<class>/<i>.wav``."""
    for label, n in classes.items():
        for i in range(n):
            _wav(root / label / f"{i}.wav")
    return root


def _cfg(tmp_path: Path, sources: list[dict], ds_extra: dict | None = None):
    ds = {"random_seed": 0, "test_ratio": 0.5}
    if ds_extra:
        ds.update(ds_extra)
    ds["sources"] = sources
    return _parse_config({
        "foundation_model": {
            "name": "birdnet", "version": "2.4", "format": "onnx",
            "source": "local", "path": str(tmp_path / "m.onnx"),
            "sample_rate": SR, "window_seconds": WINDOW_SEC,
        },
        "dataset": ds,
    })


def test_single_source_unchanged(tmp_path):
    root = _subfolders(tmp_path / "a", {"bird": 4, "frog": 4})
    cfg = _parse_config({
        "foundation_model": {
            "name": "birdnet", "version": "2.4", "format": "onnx",
            "source": "local", "path": str(tmp_path / "m.onnx"),
            "sample_rate": SR, "window_seconds": WINDOW_SEC,
        },
        "dataset": {"data_dir": str(root), "random_seed": 0, "test_ratio": 0.5},
    })
    train, test = load_and_prepare_blocks(cfg, WINDOW_SEC, SR)
    assert len(train) + len(test) == 8


def test_two_sources_are_merged(tmp_path):
    a = _subfolders(tmp_path / "a", {"bird": 4})
    b = _subfolders(tmp_path / "b", {"frog": 4})
    cfg = _cfg(tmp_path, [
        {"data_dir": str(a), "label_mode": "subfolders"},
        {"data_dir": str(b), "label_mode": "subfolders"},
    ])
    train, test = load_and_prepare_blocks(cfg, WINDOW_SEC, SR)
    all_labels = {s.label for s in train + test}
    assert all_labels == {"bird", "frog"}
    assert len(train) + len(test) == 8


def test_per_source_split_is_independent(tmp_path):
    # Each source split independently with test_ratio=0.5 → both contribute
    # to train and test, so neither subset is empty.
    a = _subfolders(tmp_path / "a", {"bird": 6})
    b = _subfolders(tmp_path / "b", {"frog": 6})
    cfg = _cfg(tmp_path, [
        {"data_dir": str(a), "label_mode": "subfolders"},
        {"data_dir": str(b), "label_mode": "subfolders"},
    ])
    train, test = load_and_prepare_blocks(cfg, WINDOW_SEC, SR)
    assert {s.label for s in train} == {"bird", "frog"}
    assert {s.label for s in test} == {"bird", "frog"}


def test_run_level_append_prepended_once(tmp_path):
    # Existing exported dataset with a predefined train/test split.
    existing = tmp_path / "existing"
    _wav(existing / "train" / "bird" / "e0.wav")
    _wav(existing / "train" / "bird" / "e1.wav")
    _wav(existing / "test" / "bird" / "e2.wav")

    a = _subfolders(tmp_path / "a", {"frog": 4})
    b = _subfolders(tmp_path / "b", {"toad": 4})
    cfg = _cfg(
        tmp_path,
        [{"data_dir": str(a), "label_mode": "subfolders"},
         {"data_dir": str(b), "label_mode": "subfolders"}],
        ds_extra={"append_dataset_path": str(existing)},
    )
    train, test = load_and_prepare_blocks(cfg, WINDOW_SEC, SR)

    appended = [s for s in train + test if s.is_appended]
    # Exactly the three existing files, counted once (not once per source).
    assert len(appended) == 3
    assert sum(1 for s in train if s.is_appended) == 2  # predefined train
    assert sum(1 for s in test if s.is_appended) == 1   # predefined test
    assert {"frog", "toad", "bird"} <= {s.label for s in train + test}
