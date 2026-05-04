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
