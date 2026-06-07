"""Unit tests for bioaccx.dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from unittest.mock import patch

from bioaccx.dataset import (
    AudioSample,
    _LabelRow,
    _as_dirs,
    _chunk_rows,
    _is_audio,
    _npy_filename,
    _parse_file_per_label,
    _parse_ext_table,
    _parse_table,
    _load_subfolders,
    _sample_export_key,
    split_samples,
)

SR = 16_000
AUDIO_EXTS = frozenset({".wav", ".flac", ".mp3", ".ogg"})


def _wav(path: Path, duration: float = 0.2) -> Path:
    n = int(duration * SR)
    sf.write(str(path), np.zeros(n, dtype=np.float32), SR)
    return path


def _sample(label: str = "bird", path: str = "x.wav",
             start: float | None = None, end: float | None = None,
             split: str | None = None) -> AudioSample:
    return AudioSample(Path(path), label, start, end, split)


def _row(label: str = "bird", start: float = 0.0, end: float = 1.0,
         path: str = "x.wav", split: str | None = None) -> _LabelRow:
    return _LabelRow(Path(path), label, start, end, split)


# ---------------------------------------------------------------------------
# _is_audio
# ---------------------------------------------------------------------------

class TestIsAudio:
    def test_recognises_wav(self, tmp_path):
        f = _wav(tmp_path / "a.wav")
        assert _is_audio(f, AUDIO_EXTS)

    def test_case_insensitive_extension(self, tmp_path):
        f = _wav(tmp_path / "a.WAV")
        assert _is_audio(f, AUDIO_EXTS)

    def test_rejects_txt(self, tmp_path):
        f = tmp_path / "a.txt"
        f.touch()
        assert not _is_audio(f, AUDIO_EXTS)

    def test_rejects_directory(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        assert not _is_audio(d, AUDIO_EXTS)

    def test_rejects_nonexistent(self, tmp_path):
        assert not _is_audio(tmp_path / "ghost.wav", AUDIO_EXTS)


# ---------------------------------------------------------------------------
# _chunk_rows
# ---------------------------------------------------------------------------

class TestChunkRows:
    WINDOW = 1.0

    def test_short_segment_yields_one_chunk(self):
        rows = [_row(start=0.0, end=0.5)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        assert len(chunks) == 1
        assert chunks[0].end_time == pytest.approx(0.0 + self.WINDOW)

    def test_exact_window_yields_one_chunk(self):
        rows = [_row(start=0.0, end=1.0)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        assert len(chunks) == 1
        assert chunks[0].start_time == pytest.approx(0.0)

    def test_no_overlap_integer_multiple(self):
        rows = [_row(start=0.0, end=3.0)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        starts = [c.start_time for c in chunks]
        assert starts[:3] == pytest.approx([0.0, 1.0, 2.0])

    def test_half_overlap_increases_chunk_count(self):
        rows = [_row(start=0.0, end=2.0)]
        no_overlap = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        with_overlap = _chunk_rows(rows, self.WINDOW, overlap=0.5)
        assert len(with_overlap) > len(no_overlap)

    def test_chunk_label_preserved(self):
        rows = [_row(label="frog", start=0.0, end=2.0)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        assert all(c.label == "frog" for c in chunks)

    def test_chunk_path_preserved(self):
        rows = [_row(path="/tmp/audio.wav", start=0.0, end=2.0)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        assert all(c.path == Path("/tmp/audio.wav") for c in chunks)

    def test_split_field_propagated(self):
        rows = [_row(start=0.0, end=2.0, split="train")]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        assert all(c.split == "train" for c in chunks)

    def test_zero_or_negative_duration_skipped(self):
        rows = [_row(start=1.0, end=1.0), _row(start=2.0, end=1.0)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        assert len(chunks) == 0

    def test_chunk_end_times_monotone(self):
        rows = [_row(start=0.0, end=5.0)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.5)
        for i in range(1, len(chunks)):
            assert chunks[i].start_time >= chunks[i - 1].start_time


# ---------------------------------------------------------------------------
# split_samples
# ---------------------------------------------------------------------------

class TestSplitSamples:
    def _make_samples(self, n: int, labels: list[str]) -> list[AudioSample]:
        return [_sample(label=labels[i % len(labels)]) for i in range(n)]

    def test_predefined_split_respected(self):
        samples = (
            [_sample(label="bird", split="train")] * 8 +
            [_sample(label="bird", split="test")] * 2
        )
        train, test = split_samples(samples, test_ratio=0.2, random_seed=0)
        assert len(train) == 8
        assert len(test) == 2

    def test_auto_stratified_split(self):
        samples = self._make_samples(30, ["bird", "frog", "background"])
        train, test = split_samples(samples, test_ratio=0.2, random_seed=42)
        assert len(train) + len(test) == 30
        total_test = len(test)
        assert 5 <= total_test <= 8  # roughly 20% of 30

    def test_all_labels_appear_in_both_splits(self):
        samples = self._make_samples(30, ["bird", "frog", "background"])
        train, test = split_samples(samples, test_ratio=0.2, random_seed=42)
        train_labels = {s.label for s in train}
        test_labels = {s.label for s in test}
        assert train_labels == {"bird", "frog", "background"}
        assert test_labels == {"bird", "frog", "background"}

    def test_reproducible_with_same_seed(self):
        samples = self._make_samples(50, ["a", "b"])
        t1, e1 = split_samples(samples, 0.2, 7)
        t2, e2 = split_samples(samples, 0.2, 7)
        assert [s.label for s in t1] == [s.label for s in t2]


# ---------------------------------------------------------------------------
# _npy_filename
# ---------------------------------------------------------------------------

class TestNpyFilename:
    def test_with_times(self):
        s = AudioSample(Path("/tmp/rec.wav"), "bird", 1.5, 4.5)
        name = _npy_filename(s)
        assert name == "rec_1.500_4.500.npy"

    def test_without_times(self):
        s = AudioSample(Path("/tmp/rec.wav"), "bird")
        name = _npy_filename(s)
        assert name == "rec.npy"


# ---------------------------------------------------------------------------
# _load_subfolders
# ---------------------------------------------------------------------------

class TestLoadSubfolders:
    def test_flat_class_dirs(self, tmp_path):
        for cls in ("bird", "frog"):
            d = tmp_path / cls
            d.mkdir()
            for i in range(3):
                _wav(d / f"{cls}_{i}.wav")
        samples = _load_subfolders(tmp_path, AUDIO_EXTS)
        assert len(samples) == 6
        labels = {s.label for s in samples}
        assert labels == {"bird", "frog"}

    def test_non_audio_files_ignored(self, tmp_path):
        d = tmp_path / "bird"
        d.mkdir()
        _wav(d / "a.wav")
        (d / "README.txt").write_text("ignore me")
        samples = _load_subfolders(tmp_path, AUDIO_EXTS)
        assert len(samples) == 1

    def test_predefined_train_test_split(self, tmp_path):
        for split, n in [("train", 4), ("test", 2)]:
            d = tmp_path / split / "bird"
            d.mkdir(parents=True)
            for i in range(n):
                _wav(d / f"bird_{i}.wav")
        samples = _load_subfolders(tmp_path, AUDIO_EXTS)
        train = [s for s in samples if s.split == "train"]
        test  = [s for s in samples if s.split == "test"]
        assert len(train) == 4
        assert len(test) == 2

    def test_split_samples_have_split_field(self, tmp_path):
        d = tmp_path / "train" / "bird"
        d.mkdir(parents=True)
        _wav(d / "bird.wav")
        samples = _load_subfolders(tmp_path, AUDIO_EXTS)
        assert samples[0].split == "train"


# ---------------------------------------------------------------------------
# _parse_file_per_label
# ---------------------------------------------------------------------------

class TestParseFilePerLabel:
    def test_tab_delimited(self, tmp_path):
        _wav(tmp_path / "a.wav", 0.5)
        (tmp_path / "a.txt").write_text("0.0\t0.3\tbird\n")
        rows = _parse_file_per_label(tmp_path, AUDIO_EXTS)
        assert len(rows) == 1
        assert rows[0].label == "bird"
        assert rows[0].start_time == pytest.approx(0.0)
        assert rows[0].end_time == pytest.approx(0.3)

    def test_comma_delimited(self, tmp_path):
        _wav(tmp_path / "a.wav", 0.5)
        (tmp_path / "a.txt").write_text("0.0,0.3,bird\n")
        rows = _parse_file_per_label(tmp_path, AUDIO_EXTS)
        assert len(rows) == 1
        assert rows[0].label == "bird"

    def test_multiple_rows_per_file(self, tmp_path):
        _wav(tmp_path / "a.wav", 1.0)
        (tmp_path / "a.txt").write_text("0.0\t0.4\tbird\n0.5\t0.9\tfrog\n")
        rows = _parse_file_per_label(tmp_path, AUDIO_EXTS)
        assert len(rows) == 2
        assert rows[0].label == "bird"
        assert rows[1].label == "frog"

    def test_missing_label_file_skipped(self, tmp_path, capsys):
        _wav(tmp_path / "a.wav", 0.5)
        # No .txt companion
        rows = _parse_file_per_label(tmp_path, AUDIO_EXTS)
        assert len(rows) == 0
        captured = capsys.readouterr()
        assert "skip" in captured.out.lower()

    def test_audio_without_label_does_not_raise(self, tmp_path):
        _wav(tmp_path / "unlabeled.wav", 0.5)
        _wav(tmp_path / "labeled.wav", 0.5)
        (tmp_path / "labeled.txt").write_text("0.0\t0.3\tbird\n")
        rows = _parse_file_per_label(tmp_path, AUDIO_EXTS)
        assert len(rows) == 1

    def test_malformed_rows_skipped(self, tmp_path):
        # First line sets delimiter to tab; first value is non-numeric → skipped
        # Second line is a valid tab-delimited row
        _wav(tmp_path / "a.wav", 0.5)
        (tmp_path / "a.txt").write_text("not_a_number\t0.3\tbird\n0.0\t0.3\tbird\n")
        rows = _parse_file_per_label(tmp_path, AUDIO_EXTS)
        assert len(rows) == 1  # only the valid row


# ---------------------------------------------------------------------------
# _parse_table
# ---------------------------------------------------------------------------

class TestParseTable:
    def _csv(self, tmp_path: Path, has_split: bool = False) -> Path:
        rows = ["filename,label,start_time,end_time" + (",split" if has_split else "")]
        _wav(tmp_path / "a.wav", 0.5)
        _wav(tmp_path / "b.wav", 0.5)
        if has_split:
            rows += ["a.wav,bird,0.0,0.3,train", "b.wav,frog,0.0,0.3,test"]
        else:
            rows += ["a.wav,bird,0.0,0.3", "b.wav,frog,0.0,0.3"]
        p = tmp_path / "table.csv"
        p.write_text("\n".join(rows) + "\n")
        return p

    def test_basic_parse(self, tmp_path):
        p = self._csv(tmp_path)
        rows = _parse_table([tmp_path], p, "filename", "label", "start_time", "end_time", "split")
        assert len(rows) == 2
        labels = {r.label for r in rows}
        assert labels == {"bird", "frog"}

    def test_with_predefined_split(self, tmp_path):
        p = self._csv(tmp_path, has_split=True)
        rows = _parse_table([tmp_path], p, "filename", "label", "start_time", "end_time", "split")
        splits = {r.label: r.split for r in rows}
        assert splits["bird"] == "train"
        assert splits["frog"] == "test"

    def test_start_end_times_parsed(self, tmp_path):
        p = self._csv(tmp_path)
        rows = _parse_table([tmp_path], p, "filename", "label", "start_time", "end_time", "split")
        assert rows[0].start_time == pytest.approx(0.0)
        assert rows[0].end_time == pytest.approx(0.3)

    def test_skips_row_with_end_le_start(self, tmp_path):
        _wav(tmp_path / "a.wav", 0.5)
        p = tmp_path / "bad.csv"
        p.write_text("filename,label,start_time,end_time\na.wav,bird,0.5,0.0\n")
        rows = _parse_table([tmp_path], p, "filename", "label", "start_time", "end_time", "split")
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Multiple data_dir paths
# ---------------------------------------------------------------------------

class TestAsDirs:
    def test_string_returns_single_path(self):
        result = _as_dirs("/tmp/data")
        assert result == [Path("/tmp/data")]

    def test_list_returns_multiple_paths(self):
        result = _as_dirs(["/tmp/a", "/tmp/b"])
        assert result == [Path("/tmp/a"), Path("/tmp/b")]


class TestMultipleDirsSubfolders:
    def test_samples_combined_from_two_dirs(self, tmp_path):
        for i, cls in enumerate(["bird", "frog"]):
            d = tmp_path / f"dir{i}" / cls
            d.mkdir(parents=True)
            _wav(d / f"{cls}.wav")
        samples = _load_subfolders(tmp_path / "dir0", AUDIO_EXTS)
        samples += _load_subfolders(tmp_path / "dir1", AUDIO_EXTS)
        labels = {s.label for s in samples}
        assert labels == {"bird", "frog"}
        assert len(samples) == 2

    def test_duplicate_labels_across_dirs_merged(self, tmp_path):
        for i in range(2):
            d = tmp_path / f"dir{i}" / "bird"
            d.mkdir(parents=True)
            _wav(d / f"bird_{i}.wav")
        s0 = _load_subfolders(tmp_path / "dir0", AUDIO_EXTS)
        s1 = _load_subfolders(tmp_path / "dir1", AUDIO_EXTS)
        assert len(s0 + s1) == 2
        assert all(s.label == "bird" for s in s0 + s1)


class TestMultipleDirsFilePerLabel:
    def test_rows_combined_from_two_dirs(self, tmp_path):
        for i, label in enumerate(["bird", "frog"]):
            d = tmp_path / f"dir{i}"
            d.mkdir()
            _wav(d / "a.wav", 0.5)
            (d / "a.txt").write_text(f"0.0\t0.3\t{label}\n")
        rows0 = _parse_file_per_label(tmp_path / "dir0", AUDIO_EXTS)
        rows1 = _parse_file_per_label(tmp_path / "dir1", AUDIO_EXTS)
        labels = {r.label for r in rows0 + rows1}
        assert labels == {"bird", "frog"}


class TestMultipleDirsTable:
    def test_files_resolved_across_dirs(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _wav(dir_a / "x.wav", 0.5)
        _wav(dir_b / "y.wav", 0.5)
        table = tmp_path / "table.csv"
        table.write_text("filename,label,start_time,end_time\nx.wav,bird,0.0,0.3\ny.wav,frog,0.0,0.3\n")
        rows = _parse_table([dir_a, dir_b], table, "filename", "label", "start_time", "end_time", "split")
        assert len(rows) == 2
        assert rows[0].path == dir_a / "x.wav"
        assert rows[1].path == dir_b / "y.wav"


# ---------------------------------------------------------------------------
# Partial / missing time bounds in table mode
# ---------------------------------------------------------------------------

class TestPartialTimeBoundsTable:
    def _table(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "table.csv"
        p.write_text(content)
        return p

    def test_both_times_empty_yields_none_none(self, tmp_path):
        _wav(tmp_path / "a.wav", 1.0)
        p = self._table(tmp_path, "filename,label,start_time,end_time\na.wav,bird,,\n")
        rows = _parse_table([tmp_path], p, "filename", "label", "start_time", "end_time", "split")
        assert len(rows) == 1
        assert rows[0].start_time is None
        assert rows[0].end_time is None

    def test_only_start_time_yields_none_end(self, tmp_path):
        _wav(tmp_path / "a.wav", 1.0)
        p = self._table(tmp_path, "filename,label,start_time,end_time\na.wav,bird,1.5,\n")
        rows = _parse_table([tmp_path], p, "filename", "label", "start_time", "end_time", "split")
        assert len(rows) == 1
        assert rows[0].start_time == pytest.approx(1.5)
        assert rows[0].end_time is None

    def test_only_end_time_yields_none_start(self, tmp_path):
        _wav(tmp_path / "a.wav", 1.0)
        p = self._table(tmp_path, "filename,label,start_time,end_time\na.wav,bird,,0.8\n")
        rows = _parse_table([tmp_path], p, "filename", "label", "start_time", "end_time", "split")
        assert len(rows) == 1
        assert rows[0].start_time is None
        assert rows[0].end_time == pytest.approx(0.8)


class TestPartialTimeBoundsChunking:
    WINDOW = 0.5

    def test_none_none_uses_full_file_duration(self, tmp_path):
        _wav(tmp_path / "a.wav", 1.5)
        rows = [_LabelRow(tmp_path / "a.wav", "bird", None, None)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        # 1.5s / 0.5s window = 3 chunks
        assert len(chunks) >= 1
        assert chunks[0].start_time == pytest.approx(0.0)

    def test_none_end_uses_file_duration_from_start(self, tmp_path):
        _wav(tmp_path / "a.wav", 2.0)
        rows = [_LabelRow(tmp_path / "a.wav", "bird", 1.0, None)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        # 1.0s remaining from 1.0 → end
        assert len(chunks) >= 1
        assert chunks[0].start_time == pytest.approx(1.0)

    def test_none_start_uses_zero_to_end_time(self, tmp_path):
        _wav(tmp_path / "a.wav", 1.0)
        rows = [_LabelRow(tmp_path / "a.wav", "bird", None, 0.8)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        assert len(chunks) >= 1
        assert chunks[0].start_time == pytest.approx(0.0)
        assert all(c.end_time <= 0.8 + self.WINDOW for c in chunks)

    def test_unreadable_file_skipped(self, tmp_path):
        rows = [_LabelRow(tmp_path / "ghost.wav", "bird", None, None)]
        chunks = _chunk_rows(rows, self.WINDOW, overlap=0.0)
        assert chunks == []


# ---------------------------------------------------------------------------
# _sample_export_key
# ---------------------------------------------------------------------------

class TestSampleExportKey:
    def test_timed_sample(self):
        s = AudioSample(Path("/data/rec.wav"), "bird", 0.0, 3.0)
        assert _sample_export_key(s) == "rec_0.000_3.000"

    def test_none_start_defaults_to_zero(self):
        s = AudioSample(Path("/data/rec.wav"), "bird", None, 3.0)
        assert _sample_export_key(s) == "rec_0.000_3.000"

    def test_none_end_uses_full(self):
        s = AudioSample(Path("/data/rec.wav"), "bird", 0.0, None)
        assert _sample_export_key(s) == "rec_0.000_full"

    def test_both_none_whole_file(self):
        s = AudioSample(Path("/data/rec.wav"), "bird", None, None)
        assert _sample_export_key(s) == "rec_0.000_full"


# ---------------------------------------------------------------------------
# Append dataset deduplication (via load_samples with append_dataset_path)
# ---------------------------------------------------------------------------

def _make_exported_dataset(base: Path, samples: list[tuple[str, str, str]]) -> None:
    """Create a minimal exported dataset layout.

    samples: list of (split, label, wav_stem) — creates <split>/<label>/<wav_stem>.wav
    """
    for split, label, stem in samples:
        d = base / split / label
        d.mkdir(parents=True, exist_ok=True)
        _wav(d / f"{stem}.wav")


class TestSplitSamplesMixed:
    """split_samples with mixed predefined / unassigned samples."""

    def test_all_predefined_uses_existing_splits(self):
        samples = (
            [_sample(label="bird", split="train")] * 6 +
            [_sample(label="bird", split="test")] * 2
        )
        train, test = split_samples(samples, test_ratio=0.2, random_seed=0)
        assert len(train) == 6 and len(test) == 2

    def test_mixed_keeps_predefined_auto_splits_new(self):
        predefined = (
            [_sample(label="bird", split="train")] * 8 +
            [_sample(label="frog", split="train")] * 8 +
            [_sample(label="bird", split="test")] * 2 +
            [_sample(label="frog", split="test")] * 2
        )
        new_unsplit = [_sample(label="bird")] * 10 + [_sample(label="frog")] * 10
        train, test = split_samples(predefined + new_unsplit, test_ratio=0.2, random_seed=42)
        # predefined: 16 train + 4 test; new 20 auto-split ≈ 16 train + 4 test
        assert len(train) + len(test) == len(predefined) + len(new_unsplit)
        assert len(train) >= 16  # at least the predefined train samples

    def test_no_predefined_auto_splits_all(self):
        samples = [_sample(label="bird")] * 10 + [_sample(label="frog")] * 10
        train, test = split_samples(samples, test_ratio=0.2, random_seed=0)
        assert len(train) + len(test) == 20


class TestAppendDatasetDedup:
    """Integration: load_samples with append_dataset_path filters duplicates."""

    def _make_config(self, data_dir, append_path=None):
        from bioaccx.config import DatasetConfig
        return DatasetConfig(
            data_dir=str(data_dir),
            label_mode="subfolders",
            append_dataset_path=str(append_path) if append_path else None,
        )

    def test_no_duplicates_all_new_samples_added(self, tmp_path):
        # Existing dataset: bird only
        existing = tmp_path / "existing"
        _make_exported_dataset(existing, [
            ("train", "bird", "rec_0.000_full"),
        ])
        # New data: frog only (different label → never a duplicate)
        new_dir = tmp_path / "new" / "frog"
        new_dir.mkdir(parents=True)
        _wav(new_dir / "rec2.wav")

        cfg = self._make_config(tmp_path / "new", append_path=existing)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0)
        labels = {s.label for s in samples}
        assert "bird" in labels and "frog" in labels

    def test_duplicate_sample_is_skipped(self, tmp_path):
        # Existing dataset already has rec_0.000_full.wav for bird
        existing = tmp_path / "existing"
        _make_exported_dataset(existing, [
            ("train", "bird", "rec_0.000_full"),
        ])
        # New data: same file name → would produce the same export key
        new_dir = tmp_path / "new" / "bird"
        new_dir.mkdir(parents=True)
        _wav(new_dir / "rec.wav")  # stem="rec" → key "rec_0.000_full" → duplicate

        cfg = self._make_config(tmp_path / "new", append_path=existing)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0)
        # Only the one from the existing dataset; the new one is deduplicated
        bird_samples = [s for s in samples if s.label == "bird"]
        assert len(bird_samples) == 1
        assert bird_samples[0].split in ("train", "test")  # came from existing

    def test_existing_samples_retain_split_assignment(self, tmp_path):
        existing = tmp_path / "existing"
        _make_exported_dataset(existing, [
            ("train", "bird", "a_0.000_full"),
            ("test",  "bird", "b_0.000_full"),
        ])
        # New data with a different stem → not a duplicate
        new_dir = tmp_path / "new" / "bird"
        new_dir.mkdir(parents=True)
        _wav(new_dir / "c.wav")

        cfg = self._make_config(tmp_path / "new", append_path=existing)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0)
        split_map = {s.path.stem: s.split for s in samples}
        assert split_map.get("a_0.000_full") == "train"
        assert split_map.get("b_0.000_full") == "test"
        # new sample has no split yet
        assert split_map.get("c_0.000_full") is None

    def test_existing_samples_are_marked_is_appended(self, tmp_path):
        existing = tmp_path / "existing"
        _make_exported_dataset(existing, [("train", "bird", "rec_0.000_3.000")])
        new_dir = tmp_path / "new" / "bird"
        new_dir.mkdir(parents=True)
        _wav(new_dir / "new_rec.wav")

        cfg = self._make_config(tmp_path / "new", append_path=existing)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0)
        appended = [s for s in samples if s.is_appended]
        fresh    = [s for s in samples if not s.is_appended]
        assert len(appended) == 1
        assert appended[0].path.stem == "rec_0.000_3.000"
        assert len(fresh) == 1
        assert fresh[0].path.stem == "new_rec"


class TestAppendWithPreprocessing:
    """Preprocessing is skipped for source files whose chunks already exist."""

    def _make_fpl_dir(self, base: Path, stem: str, duration: float = 1.0,
                      label: str = "bird") -> Path:
        """Create a file_per_label data_dir with one audio+label file."""
        data = base / "data"
        data.mkdir(parents=True, exist_ok=True)
        _wav(data / f"{stem}.wav", duration=duration)
        (data / f"{stem}.txt").write_text(f"0.0,{duration},{label}\n")
        return data

    def test_preprocess_skipped_for_existing_sample(self, tmp_path):
        """When all chunks from a source file are already exported, do not preprocess it."""
        from bioaccx.config import DatasetConfig
        from bioaccx.dataset import load_samples, _get_preproc_tempdir

        data = self._make_fpl_dir(tmp_path, "rec", duration=1.0)

        # Simulate first run: existing exported dataset already has the chunk
        existing = tmp_path / "exported"
        _make_exported_dataset(existing, [("train", "bird", "rec_0.000_1.000")])

        cfg = DatasetConfig(
            data_dir=str(data),
            label_mode="file_per_label",
            append_dataset_path=str(existing),
            filter="hpf",
            filter_freq=500.0,
        )
        preproc_before = set(_get_preproc_tempdir().iterdir()) if _get_preproc_tempdir().exists() else set()
        samples = load_samples(cfg, window_seconds=1.0, sample_rate=SR)

        # The existing chunk is returned; no new sample was preprocessed.
        assert len(samples) == 1
        assert samples[0].is_appended
        preproc_after = set(_get_preproc_tempdir().iterdir())
        new_preproc_files = preproc_after - preproc_before
        assert len(new_preproc_files) == 0, (
            "source file should NOT be preprocessed when all its chunks already exist"
        )

    def test_preprocess_applied_only_to_new_source_files(self, tmp_path):
        """Only the source file with new chunks gets preprocessed."""
        from bioaccx.config import DatasetConfig
        from bioaccx.dataset import load_samples, _get_preproc_tempdir

        data = tmp_path / "data"
        data.mkdir()
        # existing.wav → already exported
        _wav(data / "existing.wav", duration=1.0)
        (data / "existing.txt").write_text("0.0,1.0,bird\n")
        # new_rec.wav → genuinely new
        _wav(data / "new_rec.wav", duration=1.0)
        (data / "new_rec.txt").write_text("0.0,1.0,bird\n")

        exported = tmp_path / "exported"
        _make_exported_dataset(exported, [("train", "bird", "existing_0.000_1.000")])

        cfg = DatasetConfig(
            data_dir=str(data),
            label_mode="file_per_label",
            append_dataset_path=str(exported),
            filter="hpf",
            filter_freq=500.0,
        )
        preproc_before = set(_get_preproc_tempdir().iterdir()) if _get_preproc_tempdir().exists() else set()
        samples = load_samples(cfg, window_seconds=1.0, sample_rate=SR)

        new_preproc = {p.name for p in _get_preproc_tempdir().iterdir()} - {p.name for p in preproc_before}
        # Only new_rec should have been preprocessed; existing.wav should not.
        assert any("new_rec" in n for n in new_preproc), "new_rec.wav should be preprocessed"
        assert not any("existing" in n for n in new_preproc), "existing.wav must NOT be preprocessed"

        # Result: 1 appended + 1 new
        assert sum(1 for s in samples if s.is_appended) == 1
        assert sum(1 for s in samples if not s.is_appended) == 1


class TestExportDatasetAudioAppended:
    """export_dataset_audio copies appended samples verbatim."""

    def test_appended_sample_copied_with_original_name(self, tmp_path):
        from bioaccx.dataset import export_dataset_audio
        # Simulate an already-exported chunk in the existing dataset
        src = tmp_path / "existing" / "train" / "bird"
        src.mkdir(parents=True)
        _wav(src / "rec_1.500_4.500.wav")

        s = AudioSample(
            path=src / "rec_1.500_4.500.wav",
            label="bird",
            split="train",
            is_appended=True,
        )
        out_dir = tmp_path / "merged"
        export_dataset_audio([s], [], out_dir=out_dir, sample_rate=SR, window_samples=int(3.0 * SR))

        out_file = out_dir / "train" / "bird" / "rec_1.500_4.500.wav"
        assert out_file.exists(), "appended sample should be copied with its original filename"
        # Must NOT produce rec_1.500_4.500_0.000_full.wav
        wrong = out_dir / "train" / "bird" / "rec_1.500_4.500_0.000_full.wav"
        assert not wrong.exists(), "re-export with _0.000_full suffix must not happen"

    def test_new_sample_exported_normally(self, tmp_path):
        from bioaccx.dataset import export_dataset_audio
        src = tmp_path / "data" / "bird"
        src.mkdir(parents=True)
        _wav(src / "rec.wav", duration=1.0)

        s = AudioSample(path=src / "rec.wav", label="bird", start_time=0.0, end_time=0.5)
        out_dir = tmp_path / "out"
        export_dataset_audio([s], [], out_dir=out_dir, sample_rate=SR, window_samples=int(0.5 * SR))

        out_file = out_dir / "train" / "bird" / "rec_0.000_0.500.wav"
        assert out_file.exists()


# ---------------------------------------------------------------------------
# _parse_ext_table
# ---------------------------------------------------------------------------

def _make_inat_get_audio(tmp_path: Path):
    """Return a fake get_audio that creates a real WAV file and returns a name."""
    def _fake(obs_id, sound_index, cache_dir):
        audio_path = tmp_path / f"inat_{obs_id}_{sound_index}.wav"
        _wav(audio_path, duration=0.5)
        return audio_path, f"Species {obs_id}"
    return _fake


class TestParseExtTableInat:
    def _table(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "inat.csv"
        p.write_text(content)
        return p

    def test_inat_row_uses_scientific_name_when_label_empty(self, tmp_path):
        p = self._table(tmp_path, "observation_id,sound_index,label,start_time,end_time\n"
                                   "99999,0,,0.0,0.3\n")
        with patch("bioaccx.inat.get_audio", side_effect=_make_inat_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert len(rows) == 1
        assert rows[0].label == "Species 99999"

    def test_inat_row_label_overrides_scientific_name(self, tmp_path):
        p = self._table(tmp_path, "observation_id,sound_index,label,start_time,end_time\n"
                                   "99999,0,cicada,0.0,0.3\n")
        with patch("bioaccx.inat.get_audio", side_effect=_make_inat_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert rows[0].label == "cicada"

    def test_sound_index_defaults_to_zero_when_missing(self, tmp_path):
        p = self._table(tmp_path, "observation_id,label\n99999,bird\n")
        calls = []
        def _fake(obs_id, sound_index, cache_dir):
            calls.append(sound_index)
            return _make_inat_get_audio(tmp_path)(obs_id, sound_index, cache_dir)
        with patch("bioaccx.inat.get_audio", side_effect=_fake):
            _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert calls == [0]

    def test_local_row_resolved_from_data_dirs(self, tmp_path):
        data_dir = tmp_path / "audio"
        data_dir.mkdir()
        _wav(data_dir / "rec.wav", duration=0.5)
        p = self._table(tmp_path, "filename,observation_id,label,start_time,end_time\n"
                                   "rec.wav,,bird,0.0,0.3\n")
        rows = _parse_ext_table(
            [data_dir], p, "filename", "label", "start_time", "end_time", "split",
            "observation_id", "sound_index", "xc_id", tmp_path / "cache",
        )
        assert len(rows) == 1
        assert rows[0].path == data_dir / "rec.wav"
        assert rows[0].label == "bird"

    def test_mixed_table_yields_both_row_types(self, tmp_path):
        data_dir = tmp_path / "audio"
        data_dir.mkdir()
        _wav(data_dir / "local.wav", duration=0.5)
        p = self._table(tmp_path,
            "filename,observation_id,sound_index,label,start_time,end_time\n"
            "local.wav,,,bird,0.0,0.3\n"
            ",88888,0,,0.0,0.3\n"
        )
        with patch("bioaccx.inat.get_audio", side_effect=_make_inat_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [data_dir], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert len(rows) == 2
        labels = {r.label for r in rows}
        assert "bird" in labels
        assert "Species 88888" in labels

    def test_row_with_neither_column_skipped(self, tmp_path):
        p = self._table(tmp_path, "filename,observation_id,label\n,,bird\n")
        rows = _parse_ext_table(
            [], p, "filename", "label", "start_time", "end_time", "split",
            "observation_id", "sound_index", "xc_id", tmp_path / "cache",
        )
        assert rows == []

    def test_failed_download_skipped_with_message(self, tmp_path, capsys):
        p = self._table(tmp_path, "observation_id,label\n99999,bird\n")
        with patch("bioaccx.inat.get_audio", side_effect=ValueError("network error")):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert rows == []
        assert "skip" in capsys.readouterr().out.lower()

    def test_partial_times_preserved(self, tmp_path):
        p = self._table(tmp_path, "observation_id,label,start_time,end_time\n"
                                   "99999,bird,1.5,\n")
        with patch("bioaccx.inat.get_audio", side_effect=_make_inat_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert rows[0].start_time == pytest.approx(1.5)
        assert rows[0].end_time is None


def _make_xc_get_audio(tmp_path: Path):
    def _fake(xc_id, cache_dir, api_key=None):
        audio_path = tmp_path / f"xc_{xc_id}.wav"
        _wav(audio_path, duration=0.5)
        return audio_path, f"XC species {xc_id}"
    return _fake


class TestParseExtTableXenoCanto:
    def _table(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "remote.csv"
        p.write_text(content)
        return p

    def test_xc_row_uses_scientific_name_when_label_empty(self, tmp_path):
        p = self._table(tmp_path, "xc_id,label,start_time,end_time\n12345,,0.0,0.3\n")
        with patch("bioaccx.xc.get_audio", side_effect=_make_xc_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert len(rows) == 1
        assert rows[0].label == "XC species 12345"

    def test_xc_row_label_overrides_scientific_name(self, tmp_path):
        p = self._table(tmp_path, "xc_id,label\n12345,frog\n")
        with patch("bioaccx.xc.get_audio", side_effect=_make_xc_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert rows[0].label == "frog"

    def test_xc_row_failed_download_skipped(self, tmp_path, capsys):
        p = self._table(tmp_path, "xc_id,label\n12345,bird\n")
        with patch("bioaccx.xc.get_audio", side_effect=ValueError("not found")):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert rows == []
        assert "skip" in capsys.readouterr().out.lower()

    def test_inat_takes_priority_over_xc_when_both_filled(self, tmp_path):
        p = self._table(tmp_path, "observation_id,xc_id,label\n99999,12345,bird\n")
        inat_called, xc_called = [], []
        def _inat(obs_id, sound_index, cache_dir):
            inat_called.append(obs_id)
            return _make_inat_get_audio(tmp_path)(obs_id, sound_index, cache_dir)
        def _xc(xc_id, cache_dir):
            xc_called.append(xc_id)
            return _make_xc_get_audio(tmp_path)(xc_id, cache_dir)
        with patch("bioaccx.inat.get_audio", side_effect=_inat), \
             patch("bioaccx.xc.get_audio", side_effect=_xc):
            _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert inat_called == ["99999"]
        assert xc_called == []

    def test_all_three_sources_in_one_table(self, tmp_path):
        data_dir = tmp_path / "audio"
        data_dir.mkdir()
        _wav(data_dir / "local.wav", duration=0.5)
        p = self._table(tmp_path,
            "filename,observation_id,xc_id,label\n"
            "local.wav,,,sparrow\n"
            ",111,,\n"
            ",,222,\n"
        )
        with patch("bioaccx.inat.get_audio", side_effect=_make_inat_get_audio(tmp_path)), \
             patch("bioaccx.xc.get_audio", side_effect=_make_xc_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [data_dir], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert len(rows) == 3
        labels = {r.label for r in rows}
        assert "sparrow" in labels
        assert "Species 111" in labels
        assert "XC species 222" in labels


class TestXcStripPrefix:
    def test_numeric_id_unchanged(self):
        from bioaccx.xc import _strip_prefix
        assert _strip_prefix("12345") == "12345"

    def test_xc_prefix_stripped(self):
        from bioaccx.xc import _strip_prefix
        assert _strip_prefix("XC12345") == "12345"

    def test_lowercase_xc_stripped(self):
        from bioaccx.xc import _strip_prefix
        assert _strip_prefix("xc12345") == "12345"

    def test_integer_input(self):
        from bioaccx.xc import _strip_prefix
        assert _strip_prefix(12345) == "12345"


# ---------------------------------------------------------------------------
# Filter and speed preprocessing
# ---------------------------------------------------------------------------

def _wav_long(path: Path, duration: float = 2.0, sr: int = SR) -> Path:
    n = int(duration * sr)
    sf.write(str(path), np.zeros(n, dtype=np.float32), sr)
    return path


def _make_file_per_label_dir(tmp_path: Path, label: str, duration: float = 2.0) -> Path:
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    _wav_long(d / "rec.wav", duration)
    (d / "rec.txt").write_text(f"0.0\t{duration}\t{label}\n")
    return d


def _make_cfg(**kwargs) -> "DatasetConfig":
    from bioaccx.config import DatasetConfig
    kwargs.setdefault("label_mode", "file_per_label")
    return DatasetConfig(**kwargs)


class TestPreprocessAudioFiles:
    """Unit tests for _preprocess_audio_files."""

    def test_creates_output_file(self, tmp_path):
        from bioaccx.dataset import _preprocess_audio_files

        src = _wav_long(tmp_path / "audio.wav")
        path_map = _preprocess_audio_files(
            {src}, SR, "hpf", 1000.0, 5, 1.0, tmp_path / "out"
        )
        assert path_map[src].exists()

    def test_original_path_in_map(self, tmp_path):
        from bioaccx.dataset import _preprocess_audio_files

        src = _wav_long(tmp_path / "audio.wav")
        path_map = _preprocess_audio_files(
            {src}, SR, "hpf", 1000.0, 5, 1.0, tmp_path / "out"
        )
        assert src in path_map

    def test_output_is_wav(self, tmp_path):
        from bioaccx.dataset import _preprocess_audio_files

        src = _wav_long(tmp_path / "audio.wav")
        path_map = _preprocess_audio_files(
            {src}, SR, "lpf", 4000.0, 5, 1.0, tmp_path / "out"
        )
        out = path_map[src]
        assert out.suffix == ".wav"

    def test_speed_changes_output_duration(self, tmp_path):
        from bioaccx.dataset import _preprocess_audio_files

        src = _wav_long(tmp_path / "audio.wav", duration=2.0)
        path_map = _preprocess_audio_files(
            {src}, SR, None, None, 5, 2.0, tmp_path / "out"
        )
        info = sf.info(str(path_map[src]))
        assert abs(info.duration - 1.0) < 0.05

    def test_speed_half_doubles_duration(self, tmp_path):
        from bioaccx.dataset import _preprocess_audio_files

        src = _wav_long(tmp_path / "audio.wav", duration=1.0)
        path_map = _preprocess_audio_files(
            {src}, SR, None, None, 5, 0.5, tmp_path / "out"
        )
        info = sf.info(str(path_map[src]))
        assert abs(info.duration - 2.0) < 0.05

    def test_existing_file_not_rewritten(self, tmp_path):
        from bioaccx.dataset import _preprocess_audio_files

        src = _wav_long(tmp_path / "audio.wav")
        out_dir = tmp_path / "out"
        path_map = _preprocess_audio_files({src}, SR, "hpf", 1000.0, 5, 1.0, out_dir)
        mtime1 = path_map[src].stat().st_mtime
        _preprocess_audio_files({src}, SR, "hpf", 1000.0, 5, 1.0, out_dir)
        mtime2 = path_map[src].stat().st_mtime
        assert mtime1 == mtime2

    def test_missing_filter_freq_falls_back_to_original(self, tmp_path, capsys):
        from bioaccx.dataset import _preprocess_audio_files

        src = _wav_long(tmp_path / "audio.wav")
        path_map = _preprocess_audio_files(
            {src}, SR, "hpf", None, 5, 1.0, tmp_path / "out"
        )
        # Should fall back to original path on error
        assert path_map[src] == src
        assert "error" in capsys.readouterr().out.lower()

    def test_multiple_files_processed(self, tmp_path):
        from bioaccx.dataset import _preprocess_audio_files

        srcs = {_wav_long(tmp_path / f"audio_{i}.wav") for i in range(3)}
        path_map = _preprocess_audio_files(srcs, SR, "hpf", 500.0, 5, 1.0, tmp_path / "out")
        assert all(path_map[p].exists() for p in srcs)


class TestLoadSamplesWithFilter:
    """load_samples applies filter preprocessing before chunking."""

    def test_filter_creates_preprocessed_file(self, tmp_path):
        d = _make_file_per_label_dir(tmp_path, "bird")
        cfg = _make_cfg(data_dir=str(d), filter="hpf", filter_freq=1000.0)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0, sample_rate=SR)
        # All sample paths should point to preprocessed files (not the original)
        orig = d / "rec.wav"
        assert all(s.path != orig for s in samples)
        assert all(s.path.exists() for s in samples)

    def test_lpf_preprocessed_path_differs_from_original(self, tmp_path):
        d = _make_file_per_label_dir(tmp_path, "frog")
        cfg = _make_cfg(data_dir=str(d), filter="lpf", filter_freq=4000.0)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0, sample_rate=SR)
        assert all(s.path != d / "rec.wav" for s in samples)

    def test_bpf_requires_list_freq(self, tmp_path):
        d = _make_file_per_label_dir(tmp_path, "bird")
        cfg = _make_cfg(data_dir=str(d), filter="bpf", filter_freq=[500.0, 4000.0])
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0, sample_rate=SR)
        assert len(samples) >= 1
        assert all(s.path.exists() for s in samples)

    def test_no_preproc_without_sample_rate(self, tmp_path, capsys):
        d = _make_file_per_label_dir(tmp_path, "bird")
        orig = d / "rec.wav"
        cfg = _make_cfg(data_dir=str(d), filter="hpf", filter_freq=1000.0)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0)  # no sample_rate
        assert "warning" in capsys.readouterr().out.lower()
        # Paths stay in their original location (chunking still happens)
        assert all(s.path == orig for s in samples)

    def test_subfolders_mode_filter(self, tmp_path):
        cls_dir = tmp_path / "data" / "bird"
        cls_dir.mkdir(parents=True)
        _wav_long(cls_dir / "rec.wav", duration=1.0)
        cfg = _make_cfg(
            data_dir=str(tmp_path / "data"),
            label_mode="subfolders",
            filter="hpf", filter_freq=1000.0,
        )
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0, sample_rate=SR)
        assert len(samples) == 1
        assert samples[0].path != cls_dir / "rec.wav"
        assert samples[0].path.exists()


class TestLoadSamplesWithSpeed:
    """load_samples adjusts label times and audio duration for speed changes."""

    def test_speed_2x_halves_label_times(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        _wav_long(d / "rec.wav", duration=4.0)
        (d / "rec.txt").write_text("1.0\t3.0\tbird\n")
        cfg = _make_cfg(data_dir=str(d), speed=2.0)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=0.5, sample_rate=SR)
        # All chunk times should be in the speed-adjusted domain (halved)
        assert all(s.start_time < 2.0 for s in samples)  # original end was 3.0 → 1.5
        assert all(s.end_time <= 1.5 + 0.5 + 0.01 for s in samples)

    def test_speed_half_doubles_label_times(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        _wav_long(d / "rec.wav", duration=2.0)
        (d / "rec.txt").write_text("0.5\t1.5\tbird\n")
        cfg = _make_cfg(data_dir=str(d), speed=0.5)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=0.5, sample_rate=SR)
        # start=0.5/0.5=1.0, end=1.5/0.5=3.0
        assert samples[0].start_time == pytest.approx(1.0, abs=0.01)

    def test_speed_changes_preprocessed_audio_duration(self, tmp_path):
        d = _make_file_per_label_dir(tmp_path, "bird", duration=2.0)
        cfg = _make_cfg(data_dir=str(d), speed=2.0)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=0.5, sample_rate=SR)
        info = sf.info(str(samples[0].path))
        assert abs(info.duration - 1.0) < 0.05

    def test_speed_one_leaves_times_unchanged(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        _wav_long(d / "rec.wav", duration=2.0)
        (d / "rec.txt").write_text("0.0\t2.0\tbird\n")
        cfg = _make_cfg(data_dir=str(d), speed=1.0)
        from bioaccx.dataset import load_samples
        samples_with = load_samples(cfg, window_seconds=1.0, sample_rate=SR)
        cfg_no_speed = _make_cfg(data_dir=str(d))
        samples_without = load_samples(cfg_no_speed, window_seconds=1.0, sample_rate=SR)
        assert len(samples_with) == len(samples_without)
        for a, b in zip(samples_with, samples_without):
            assert a.start_time == pytest.approx(b.start_time, abs=1e-6)
            assert a.end_time == pytest.approx(b.end_time, abs=1e-6)

    def test_none_start_time_unchanged_by_speed(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        _wav_long(d / "rec.wav", duration=2.0)
        (d / "rec.txt").write_text(",2.0,bird\n")  # start = empty → None
        cfg = _make_cfg(data_dir=str(d), speed=2.0)
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=0.5, sample_rate=SR)
        # None start stays None (treated as 0.0); chunking resolves from file
        assert all(s.start_time is not None for s in samples)  # chunk_rows fills it in

    def test_subfolders_mode_speed(self, tmp_path):
        cls_dir = tmp_path / "data" / "bird"
        cls_dir.mkdir(parents=True)
        _wav_long(cls_dir / "rec.wav", duration=2.0)
        cfg = _make_cfg(
            data_dir=str(tmp_path / "data"),
            label_mode="subfolders",
            speed=2.0,
        )
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=1.0, sample_rate=SR)
        assert len(samples) == 1
        info = sf.info(str(samples[0].path))
        assert abs(info.duration - 1.0) < 0.05


class TestLoadSamplesFilterAndSpeed:
    """Filter and speed applied together, in the correct order."""

    def test_filter_then_speed_both_applied(self, tmp_path):
        d = _make_file_per_label_dir(tmp_path, "bird", duration=2.0)
        cfg = _make_cfg(
            data_dir=str(d),
            filter="hpf", filter_freq=500.0,
            speed=2.0,
        )
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=0.5, sample_rate=SR)
        # Preprocessed file should be half the original duration
        info = sf.info(str(samples[0].path))
        assert abs(info.duration - 1.0) < 0.05
        # Chunk times should be in speed-adjusted domain
        assert all(s.end_time <= 1.0 + 0.5 + 0.01 for s in samples)

    def test_tag_in_preprocessed_filename_reflects_both(self, tmp_path):
        d = _make_file_per_label_dir(tmp_path, "bird")
        cfg = _make_cfg(
            data_dir=str(d),
            filter="lpf", filter_freq=3000.0,
            speed=1.5,
        )
        from bioaccx.dataset import load_samples
        samples = load_samples(cfg, window_seconds=0.5, sample_rate=SR)
        name = samples[0].path.name
        assert "lpf" in name
        assert "spd" in name


# ---------------------------------------------------------------------------
# arbimon module — unit tests (no network; rfcx client is mocked)
# ---------------------------------------------------------------------------

class TestArbimonParseUtcOffset:
    def test_integer(self):
        from bioaccx.arbimon import _parse_utc_offset
        assert _parse_utc_offset(-3) == pytest.approx(-3.0)

    def test_float(self):
        from bioaccx.arbimon import _parse_utc_offset
        assert _parse_utc_offset(5.5) == pytest.approx(5.5)

    def test_string_negative(self):
        from bioaccx.arbimon import _parse_utc_offset
        assert _parse_utc_offset("-3") == pytest.approx(-3.0)

    def test_string_positive(self):
        from bioaccx.arbimon import _parse_utc_offset
        assert _parse_utc_offset("+5") == pytest.approx(5.0)

    def test_utc_prefix_stripped(self):
        from bioaccx.arbimon import _parse_utc_offset
        assert _parse_utc_offset("UTC-3") == pytest.approx(-3.0)

    def test_colon_minutes(self):
        from bioaccx.arbimon import _parse_utc_offset
        assert _parse_utc_offset("UTC+5:30") == pytest.approx(5.5)

    def test_zero(self):
        from bioaccx.arbimon import _parse_utc_offset
        assert _parse_utc_offset(0) == pytest.approx(0.0)

    def test_string_zero(self):
        from bioaccx.arbimon import _parse_utc_offset
        assert _parse_utc_offset("0") == pytest.approx(0.0)


class TestArbimonToUtc:
    def test_negative_offset(self):
        from bioaccx.arbimon import _to_utc
        import datetime
        utc = _to_utc("2024-01-15", "10:30:00", -3.0)
        assert utc == datetime.datetime(2024, 1, 15, 13, 30, 0)

    def test_positive_offset(self):
        from bioaccx.arbimon import _to_utc
        import datetime
        utc = _to_utc("2024-01-15", "08:00:00", 2.0)
        assert utc == datetime.datetime(2024, 1, 15, 6, 0, 0)

    def test_zero_offset(self):
        from bioaccx.arbimon import _to_utc
        import datetime
        utc = _to_utc("2024-06-01", "12:00", 0.0)
        assert utc == datetime.datetime(2024, 6, 1, 12, 0, 0)

    def test_hhmm_format(self):
        from bioaccx.arbimon import _to_utc
        import datetime
        utc = _to_utc("2024-01-15", "10:30", -3.0)
        assert utc == datetime.datetime(2024, 1, 15, 13, 30, 0)

    def test_crosses_midnight(self):
        from bioaccx.arbimon import _to_utc
        import datetime
        utc = _to_utc("2024-01-15", "01:00:00", -3.0)
        assert utc == datetime.datetime(2024, 1, 15, 4, 0, 0)


class TestArbimonSentinel:
    def test_read_sentinel_returns_none_when_absent(self, tmp_path):
        from bioaccx.arbimon import _read_sentinel
        import datetime
        result = _read_sentinel(tmp_path, datetime.datetime(2024, 1, 15, 13, 30, 0))
        assert result is None

    def test_write_then_read_sentinel(self, tmp_path):
        from bioaccx.arbimon import _write_sentinel, _read_sentinel
        import datetime
        audio = tmp_path / "recording.wav"
        audio.touch()
        utc_dt = datetime.datetime(2024, 1, 15, 13, 30, 0)
        _write_sentinel(tmp_path, utc_dt, audio)
        assert _read_sentinel(tmp_path, utc_dt) == audio

    def test_read_sentinel_removes_stale_file(self, tmp_path):
        from bioaccx.arbimon import _write_sentinel, _read_sentinel, _sentinel_path
        import datetime
        utc_dt = datetime.datetime(2024, 3, 10, 8, 0, 0)
        _write_sentinel(tmp_path, utc_dt, tmp_path / "gone.wav")  # audio never created
        assert _read_sentinel(tmp_path, utc_dt) is None
        assert not _sentinel_path(tmp_path, utc_dt).exists()

    def test_sentinel_filename_format(self, tmp_path):
        from bioaccx.arbimon import _sentinel_path
        import datetime
        p = _sentinel_path(tmp_path, datetime.datetime(2024, 6, 1, 0, 0, 0))
        assert p.name == "2024-06-01_00-00-00.cached"


def _make_arbimon_get_audio(tmp_path: Path):
    """Return a fake arbimon.get_audio that creates a real WAV file."""
    def _fake(stream_id, date_str, time_str, utc_offset, cache_dir, credentials_path=None):
        audio_path = tmp_path / f"arbimon_{stream_id}_{date_str}_{time_str.replace(':', '-')}.wav"
        _wav(audio_path, duration=0.5)
        return audio_path, stream_id
    return _fake


class TestParseExtTableArbimon:
    def _table(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "arbimon.csv"
        p.write_text(content)
        return p

    def test_arbimon_row_dispatched(self, tmp_path):
        p = self._table(
            tmp_path,
            "stream_id,date,time,utc_offset,label,start_time,end_time\n"
            "abc123,2024-01-15,10:30:00,-3,bird,0.0,1.0\n",
        )
        with patch("bioaccx.arbimon.get_audio", side_effect=_make_arbimon_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert len(rows) == 1
        assert rows[0].label == "bird"
        assert rows[0].start_time == pytest.approx(0.0)
        assert rows[0].end_time == pytest.approx(1.0)

    def test_arbimon_label_falls_back_to_stream_id(self, tmp_path):
        p = self._table(
            tmp_path,
            "stream_id,date,time,utc_offset,label\n"
            "abc123,2024-01-15,10:30:00,-3,\n",
        )
        with patch("bioaccx.arbimon.get_audio", side_effect=_make_arbimon_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert rows[0].label == "abc123"

    def test_arbimon_missing_date_skipped(self, tmp_path, capsys):
        p = self._table(
            tmp_path,
            "stream_id,date,time,label\n"
            "abc123,,10:30:00,bird\n",
        )
        with patch("bioaccx.arbimon.get_audio", side_effect=_make_arbimon_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert rows == []
        assert "skip" in capsys.readouterr().out.lower()

    def test_arbimon_missing_time_skipped(self, tmp_path, capsys):
        p = self._table(
            tmp_path,
            "stream_id,date,time,label\n"
            "abc123,2024-01-15,,bird\n",
        )
        with patch("bioaccx.arbimon.get_audio", side_effect=_make_arbimon_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert rows == []
        assert "skip" in capsys.readouterr().out.lower()

    def test_arbimon_download_error_skipped(self, tmp_path, capsys):
        p = self._table(
            tmp_path,
            "stream_id,date,time,utc_offset,label\n"
            "bad_id,2024-01-15,10:30:00,-3,bird\n",
        )
        with patch("bioaccx.arbimon.get_audio", side_effect=RuntimeError("not found")):
            rows = _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert rows == []
        assert "skip" in capsys.readouterr().out.lower()

    def test_arbimon_utc_offset_defaults_to_zero(self, tmp_path):
        """Rows without a utc_offset column should default to UTC+0."""
        p = self._table(
            tmp_path,
            "stream_id,date,time,label\n"
            "abc123,2024-01-15,10:30:00,bird\n",
        )
        calls = []
        def _fake(stream_id, date_str, time_str, utc_offset, cache_dir, credentials_path=None):
            calls.append(utc_offset)
            return _make_arbimon_get_audio(tmp_path)(stream_id, date_str, time_str, utc_offset, cache_dir)
        with patch("bioaccx.arbimon.get_audio", side_effect=_fake):
            _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert calls == ["0"]

    def test_obs_id_takes_priority_over_stream_id(self, tmp_path):
        """iNat obs_id wins even when stream_id is also filled."""
        p = self._table(
            tmp_path,
            "observation_id,stream_id,date,time,label\n"
            "99999,abc123,2024-01-15,10:30:00,bird\n",
        )
        inat_called, arbimon_called = [], []
        def _fake_inat(obs_id, sound_index, cache_dir):
            inat_called.append(obs_id)
            return _make_inat_get_audio(tmp_path)(obs_id, sound_index, cache_dir)
        def _fake_arbimon(stream_id, *args, **kwargs):
            arbimon_called.append(stream_id)
            return _make_arbimon_get_audio(tmp_path)(stream_id, *args, **kwargs)
        with patch("bioaccx.inat.get_audio", side_effect=_fake_inat), \
             patch("bioaccx.arbimon.get_audio", side_effect=_fake_arbimon):
            _parse_ext_table(
                [], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert inat_called == ["99999"]
        assert arbimon_called == []

    def test_all_four_sources_in_one_table(self, tmp_path):
        data_dir = tmp_path / "audio"
        data_dir.mkdir()
        _wav(data_dir / "local.wav", duration=0.5)
        p = self._table(
            tmp_path,
            "filename,observation_id,xc_id,stream_id,date,time,label\n"
            "local.wav,,,,,,sparrow\n"
            ",111,,,,,\n"
            ",,222,,,,\n"
            ",,,abc123,2024-01-15,10:30:00,\n"
        )
        with patch("bioaccx.inat.get_audio", side_effect=_make_inat_get_audio(tmp_path)), \
             patch("bioaccx.xc.get_audio", side_effect=_make_xc_get_audio(tmp_path)), \
             patch("bioaccx.arbimon.get_audio", side_effect=_make_arbimon_get_audio(tmp_path)):
            rows = _parse_ext_table(
                [data_dir], p, "filename", "label", "start_time", "end_time", "split",
                "observation_id", "sound_index", "xc_id", tmp_path / "cache",
            )
        assert len(rows) == 4
        labels = {r.label for r in rows}
        assert "sparrow" in labels
        assert "Species 111" in labels
        assert "XC species 222" in labels
        assert "abc123" in labels  # stream_id as fallback label


# ---------------------------------------------------------------------------
# _parse_augmentation_labels
# ---------------------------------------------------------------------------

class TestParseAugmentationLabels:
    from bioaccx.dataset import _parse_augmentation_labels

    def test_basic_tab_delimited(self, tmp_path):
        from bioaccx.dataset import _parse_augmentation_labels
        f = tmp_path / "labels.txt"
        f.write_text("1.0\t3.5\tnoise\n5.0\t7.0\tbackground\n")
        regions = _parse_augmentation_labels(f)
        assert regions == [(1.0, 3.5), (5.0, 7.0)]

    def test_any_label_text_included(self, tmp_path):
        from bioaccx.dataset import _parse_augmentation_labels
        f = tmp_path / "labels.txt"
        f.write_text("0.0\t1.0\talpha\n2.0\t3.0\tbeta\n")
        regions = _parse_augmentation_labels(f)
        assert len(regions) == 2

    def test_blank_lines_skipped(self, tmp_path):
        from bioaccx.dataset import _parse_augmentation_labels
        f = tmp_path / "labels.txt"
        f.write_text("\n1.0\t2.0\tnoise\n\n")
        regions = _parse_augmentation_labels(f)
        assert len(regions) == 1

    def test_end_less_than_start_skipped(self, tmp_path):
        from bioaccx.dataset import _parse_augmentation_labels
        f = tmp_path / "labels.txt"
        f.write_text("5.0\t2.0\tnoise\n1.0\t3.0\tvalid\n")
        regions = _parse_augmentation_labels(f)
        assert regions == [(1.0, 3.0)]

    def test_non_numeric_timestamps_skipped(self, tmp_path):
        from bioaccx.dataset import _parse_augmentation_labels
        f = tmp_path / "labels.txt"
        f.write_text("abc\tdef\tnoise\n1.0\t2.0\tvalid\n")
        regions = _parse_augmentation_labels(f)
        assert regions == [(1.0, 2.0)]

    def test_fewer_than_two_fields_skipped(self, tmp_path):
        from bioaccx.dataset import _parse_augmentation_labels
        f = tmp_path / "labels.txt"
        f.write_text("1.0\n1.0\t2.0\tnoise\n")
        regions = _parse_augmentation_labels(f)
        assert regions == [(1.0, 2.0)]

    def test_empty_file_returns_empty_list(self, tmp_path):
        from bioaccx.dataset import _parse_augmentation_labels
        f = tmp_path / "labels.txt"
        f.write_text("")
        assert _parse_augmentation_labels(f) == []

    def test_nonexistent_file_returns_empty_list(self, tmp_path):
        from bioaccx.dataset import _parse_augmentation_labels
        assert _parse_augmentation_labels(tmp_path / "ghost.txt") == []


# ---------------------------------------------------------------------------
# _build_concatenated_noise
# ---------------------------------------------------------------------------

def _noise_wav(path: Path, duration: float = 0.5, sr: int = SR) -> Path:
    n = int(duration * sr)
    rng = np.random.default_rng(0)
    sf.write(str(path), rng.uniform(-0.1, 0.1, n).astype(np.float32), sr)
    return path


class TestBuildConcatenatedNoise:
    def test_no_audio_files_returns_none(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is None

    def test_single_file_no_label_uses_full_file(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        import soundfile as sf
        _noise_wav(tmp_path / "noise.wav", duration=0.5)
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is not None and result.exists()
        info = sf.info(str(result))
        assert abs(info.duration - 0.5) < 0.05

    def test_multiple_files_no_labels_concatenated(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        import soundfile as sf
        _noise_wav(tmp_path / "a.wav", duration=0.4)
        _noise_wav(tmp_path / "b.wav", duration=0.6)
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is not None
        info = sf.info(str(result))
        assert abs(info.duration - 1.0) < 0.1

    def test_label_file_selects_segments_only(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        import soundfile as sf
        # 1s file; label covers 0.2s–0.5s (0.3s)
        _noise_wav(tmp_path / "rec.wav", duration=1.0)
        (tmp_path / "rec.txt").write_text("0.2\t0.5\tnoise\n")
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is not None
        info = sf.info(str(result))
        assert abs(info.duration - 0.3) < 0.05

    def test_multiple_labeled_segments_all_included(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        import soundfile as sf
        _noise_wav(tmp_path / "rec.wav", duration=2.0)
        # Two non-overlapping segments: 0.2s + 0.3s = 0.5s total
        (tmp_path / "rec.txt").write_text("0.0\t0.2\tnoise\n1.0\t1.3\tbackground\n")
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is not None
        info = sf.info(str(result))
        assert abs(info.duration - 0.5) < 0.05

    def test_mixed_files_with_and_without_labels(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        import soundfile as sf
        # labeled: use 0.3s segment only
        _noise_wav(tmp_path / "labeled.wav", duration=1.0)
        (tmp_path / "labeled.txt").write_text("0.0\t0.3\tnoise\n")
        # unlabeled: use entire 0.5s
        _noise_wav(tmp_path / "unlabeled.wav", duration=0.5)
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is not None
        info = sf.info(str(result))
        assert abs(info.duration - 0.8) < 0.1

    def test_empty_label_file_falls_back_to_full_file(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        import soundfile as sf
        _noise_wav(tmp_path / "rec.wav", duration=0.5)
        (tmp_path / "rec.txt").write_text("")  # empty → no valid regions
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is not None
        info = sf.info(str(result))
        assert abs(info.duration - 0.5) < 0.05

    def test_output_is_mono(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        import soundfile as sf
        _noise_wav(tmp_path / "rec.wav", duration=0.5)
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is not None
        info = sf.info(str(result))
        assert info.channels == 1

    def test_output_resampled_to_target_sr(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        import soundfile as sf
        _noise_wav(tmp_path / "rec.wav", duration=0.5, sr=SR)
        target_sr = SR // 2
        result = _build_concatenated_noise(tmp_path, target_sr)
        assert result is not None
        info = sf.info(str(result))
        assert info.samplerate == target_sr

    def test_result_cached_to_temp_dir(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise, _get_preproc_tempdir
        _noise_wav(tmp_path / "rec.wav", duration=0.3)
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is not None
        assert result.parent == _get_preproc_tempdir()

    def test_recursive_scan_finds_nested_files(self, tmp_path):
        from bioaccx.dataset import _build_concatenated_noise
        import soundfile as sf
        sub = tmp_path / "sub"
        sub.mkdir()
        _noise_wav(sub / "deep.wav", duration=0.4)
        result = _build_concatenated_noise(tmp_path, SR)
        assert result is not None
        info = sf.info(str(result))
        assert abs(info.duration - 0.4) < 0.05


# ---------------------------------------------------------------------------
# apply_augmentation with concatenate_augmentation_dir=True
# ---------------------------------------------------------------------------

def _make_aug_config(aug_dir: Path, snr_levels=None, **kwargs):
    from bioaccx.config import AugmentationConfig
    return AugmentationConfig(
        augmentation_dir=str(aug_dir),
        snr_levels=snr_levels or [0.0],
        **kwargs,
    )


def _make_train_samples(n: int = 2, label: str = "bird") -> list[AudioSample]:
    return [
        AudioSample(Path(f"/fake/rec_{i}.wav"), label, 0.0, 0.5)
        for i in range(n)
    ]


class TestApplyAugmentationConcat:
    def test_concat_flag_uses_single_noise_source(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "a.wav", duration=1.0)
        _noise_wav(tmp_path / "b.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                               concatenate_augmentation_dir=True)
        samples = _make_train_samples(n=2)
        result = apply_augmentation(samples, cfg, window_seconds=0.5,
                                    sample_rate=SR, random_seed=42)
        # 2 samples × 1 concatenated noise × 1 SNR = 2 augmented
        assert len(result) == 2

    def test_concat_vs_per_file_sample_count(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "a.wav", duration=1.0)
        _noise_wav(tmp_path / "b.wav", duration=1.0)
        samples = _make_train_samples(n=2)

        cfg_concat = _make_aug_config(tmp_path, snr_levels=[0.0, 10.0],
                                      keep_original=False,
                                      concatenate_augmentation_dir=True)
        result_concat = apply_augmentation(samples, cfg_concat, window_seconds=0.5,
                                           sample_rate=SR, random_seed=42)
        # 2 samples × 1 source × 2 SNR = 4
        assert len(result_concat) == 4

        cfg_perfile = _make_aug_config(tmp_path, snr_levels=[0.0, 10.0],
                                       keep_original=False,
                                       concatenate_augmentation_dir=False)
        result_perfile = apply_augmentation(samples, cfg_perfile, window_seconds=0.5,
                                            sample_rate=SR, random_seed=42)
        # 2 samples × 2 files × 2 SNR = 8
        assert len(result_perfile) == 8

    def test_concat_augmented_samples_have_noise_path_set(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "noise.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, keep_original=False,
                               concatenate_augmentation_dir=True)
        result = apply_augmentation(_make_train_samples(), cfg,
                                    window_seconds=0.5, sample_rate=SR, random_seed=0)
        assert all(s.noise_path is not None for s in result)
        assert all(s.noise_path.exists() for s in result)

    def test_concat_keep_original_prepends_clean_samples(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "noise.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=True,
                               concatenate_augmentation_dir=True)
        samples = _make_train_samples(n=3)
        result = apply_augmentation(samples, cfg, window_seconds=0.5,
                                    sample_rate=SR, random_seed=0)
        # 3 originals + 3×1×1 augmented = 6
        assert len(result) == 6
        clean = [s for s in result if s.noise_path is None]
        assert len(clean) == 3

    def test_concat_labeled_segments_used_when_label_file_present(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        import soundfile as sf
        # 2s file, label covers only 0.3s; unlabeled file adds 0.5s
        _noise_wav(tmp_path / "long.wav", duration=2.0)
        (tmp_path / "long.txt").write_text("0.0\t0.3\tnoise\n")
        _noise_wav(tmp_path / "short.wav", duration=0.5)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                               concatenate_augmentation_dir=True)
        result = apply_augmentation(_make_train_samples(n=1), cfg,
                                    window_seconds=0.5, sample_rate=SR, random_seed=0)
        assert len(result) == 1
        noise_path = result[0].noise_path
        assert noise_path is not None
        info = sf.info(str(noise_path))
        # 0.3s (labeled segment) + 0.5s (full unlabeled) = 0.8s
        assert abs(info.duration - 0.8) < 0.1

    def test_concat_empty_dir_returns_original_samples(self, tmp_path, capsys):
        from bioaccx.dataset import apply_augmentation
        cfg = _make_aug_config(tmp_path, keep_original=False,
                               concatenate_augmentation_dir=True)
        samples = _make_train_samples()
        result = apply_augmentation(samples, cfg, window_seconds=0.5,
                                    sample_rate=SR, random_seed=0)
        assert result == samples
        assert "skipping" in capsys.readouterr().out.lower()

    def test_concat_snr_field_set_on_augmented_samples(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "noise.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[-6.0, 6.0], keep_original=False,
                               concatenate_augmentation_dir=True)
        result = apply_augmentation(_make_train_samples(n=1), cfg,
                                    window_seconds=0.5, sample_rate=SR, random_seed=0)
        snrs = {s.snr for s in result}
        assert snrs == {-6.0, 6.0}

    def test_concat_noise_offset_within_bounds(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "noise.wav", duration=2.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                               concatenate_augmentation_dir=True)
        result = apply_augmentation(_make_train_samples(n=4), cfg,
                                    window_seconds=0.5, sample_rate=SR, random_seed=0)
        noise_dur = 2.0
        window = 0.5
        for s in result:
            assert 0.0 <= s.noise_start_time <= max(0.0, noise_dur - window)

    def test_concat_default_flag_false_uses_per_file_behaviour(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "a.wav", duration=1.0)
        _noise_wav(tmp_path / "b.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False)
        # concatenate_augmentation_dir defaults to False → 2 noise files
        result = apply_augmentation(_make_train_samples(n=1), cfg,
                                    window_seconds=0.5, sample_rate=SR, random_seed=0)
        assert len(result) == 2  # 1 sample × 2 files × 1 SNR


# apply_augmentation with random_augmentation_dir=True
# ---------------------------------------------------------------------------

class TestApplyAugmentationRandom:
    def test_random_produces_one_condition_per_sample_snr(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "a.wav", duration=1.0)
        _noise_wav(tmp_path / "b.wav", duration=1.0)
        _noise_wav(tmp_path / "c.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                               random_augmentation_dir=True)
        samples = _make_train_samples(n=4)
        result = apply_augmentation(samples, cfg, window_seconds=0.5,
                                    sample_rate=SR, random_seed=42)
        # 4 samples × 1 condition × 1 SNR = 4 (not 4×3=12)
        assert len(result) == 4

    def test_random_vs_per_file_sample_count(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "a.wav", duration=1.0)
        _noise_wav(tmp_path / "b.wav", duration=1.0)
        samples = _make_train_samples(n=3)

        cfg_random = _make_aug_config(tmp_path, snr_levels=[0.0, 10.0],
                                      keep_original=False,
                                      random_augmentation_dir=True)
        result_random = apply_augmentation(samples, cfg_random, window_seconds=0.5,
                                           sample_rate=SR, random_seed=42)
        # 3 samples × 1 condition × 2 SNR = 6
        assert len(result_random) == 6

        cfg_perfile = _make_aug_config(tmp_path, snr_levels=[0.0, 10.0],
                                       keep_original=False)
        result_perfile = apply_augmentation(samples, cfg_perfile, window_seconds=0.5,
                                            sample_rate=SR, random_seed=42)
        # 3 samples × 2 files × 2 SNR = 12
        assert len(result_perfile) == 12

    def test_random_noise_files_drawn_from_dir(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "a.wav", duration=1.0)
        _noise_wav(tmp_path / "b.wav", duration=1.0)
        _noise_wav(tmp_path / "c.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                               random_augmentation_dir=True)
        samples = _make_train_samples(n=6)
        result = apply_augmentation(samples, cfg, window_seconds=0.5,
                                    sample_rate=SR, random_seed=42)
        noise_stems = {s.noise_path.stem for s in result}
        # all assigned noise files must come from augmentation_dir
        assert noise_stems <= {"a", "b", "c"}
        # with 6 samples and 3 files, each file should be used at least once
        assert len(noise_stems) == 3

    def test_random_no_repetition_within_first_pass(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        for name in ("a", "b", "c"):
            _noise_wav(tmp_path / f"{name}.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                               random_augmentation_dir=True)
        # exactly as many samples as noise files → each file used exactly once
        samples = _make_train_samples(n=3)
        result = apply_augmentation(samples, cfg, window_seconds=0.5,
                                    sample_rate=SR, random_seed=42)
        noise_stems = [s.noise_path.stem for s in result]
        assert sorted(noise_stems) == ["a", "b", "c"]

    def test_random_reproducible_with_same_seed(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        for name in ("x", "y", "z"):
            _noise_wav(tmp_path / f"{name}.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                               random_augmentation_dir=True)
        samples = _make_train_samples(n=5)
        r1 = apply_augmentation(samples, cfg, window_seconds=0.5,
                                 sample_rate=SR, random_seed=7)
        r2 = apply_augmentation(samples, cfg, window_seconds=0.5,
                                 sample_rate=SR, random_seed=7)
        assert [s.noise_path for s in r1] == [s.noise_path for s in r2]
        assert [s.noise_start_time for s in r1] == [s.noise_start_time for s in r2]

    def test_random_different_seeds_give_different_order(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        for name in ("a", "b", "c", "d"):
            _noise_wav(tmp_path / f"{name}.wav", duration=1.0)
        cfg1 = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                                random_augmentation_dir=True)
        cfg2 = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                                random_augmentation_dir=True)
        samples = _make_train_samples(n=4)
        r1 = apply_augmentation(samples, cfg1, window_seconds=0.5,
                                 sample_rate=SR, random_seed=1)
        r2 = apply_augmentation(samples, cfg2, window_seconds=0.5,
                                 sample_rate=SR, random_seed=99)
        paths1 = [s.noise_path for s in r1]
        paths2 = [s.noise_path for s in r2]
        assert paths1 != paths2

    def test_random_all_augmented_have_noise_and_snr(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "n.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[-6.0, 6.0], keep_original=False,
                               random_augmentation_dir=True)
        result = apply_augmentation(_make_train_samples(n=2), cfg,
                                    window_seconds=0.5, sample_rate=SR, random_seed=0)
        assert all(s.noise_path is not None for s in result)
        assert all(s.snr is not None for s in result)
        assert {s.snr for s in result} == {-6.0, 6.0}

    def test_random_noise_offset_within_bounds(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "noise.wav", duration=2.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=False,
                               random_augmentation_dir=True)
        result = apply_augmentation(_make_train_samples(n=4), cfg,
                                    window_seconds=0.5, sample_rate=SR, random_seed=0)
        for s in result:
            assert 0.0 <= s.noise_start_time <= max(0.0, 2.0 - 0.5)

    def test_random_keep_original_prepends_clean_samples(self, tmp_path):
        from bioaccx.dataset import apply_augmentation
        _noise_wav(tmp_path / "noise.wav", duration=1.0)
        cfg = _make_aug_config(tmp_path, snr_levels=[0.0], keep_original=True,
                               random_augmentation_dir=True)
        samples = _make_train_samples(n=3)
        result = apply_augmentation(samples, cfg, window_seconds=0.5,
                                    sample_rate=SR, random_seed=0)
        # 3 originals + 3×1×1 augmented = 6
        assert len(result) == 6
        assert len([s for s in result if s.noise_path is None]) == 3

    def test_random_empty_dir_returns_original_samples(self, tmp_path, capsys):
        from bioaccx.dataset import apply_augmentation
        cfg = _make_aug_config(tmp_path, keep_original=False,
                               random_augmentation_dir=True)
        samples = _make_train_samples()
        result = apply_augmentation(samples, cfg, window_seconds=0.5,
                                    sample_rate=SR, random_seed=0)
        assert result == samples
        assert "skipping" in capsys.readouterr().out.lower()
