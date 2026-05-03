"""Unit tests for bioaccx.dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bioaccx.dataset import (
    AudioSample,
    _LabelRow,
    _chunk_rows,
    _is_audio,
    _npy_filename,
    _parse_file_per_label,
    _parse_table,
    _load_subfolders,
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
        rows = _parse_table(tmp_path, p, "filename", "label", "start_time", "end_time", "split")
        assert len(rows) == 2
        labels = {r.label for r in rows}
        assert labels == {"bird", "frog"}

    def test_with_predefined_split(self, tmp_path):
        p = self._csv(tmp_path, has_split=True)
        rows = _parse_table(tmp_path, p, "filename", "label", "start_time", "end_time", "split")
        splits = {r.label: r.split for r in rows}
        assert splits["bird"] == "train"
        assert splits["frog"] == "test"

    def test_start_end_times_parsed(self, tmp_path):
        p = self._csv(tmp_path)
        rows = _parse_table(tmp_path, p, "filename", "label", "start_time", "end_time", "split")
        assert rows[0].start_time == pytest.approx(0.0)
        assert rows[0].end_time == pytest.approx(0.3)

    def test_skips_row_with_end_le_start(self, tmp_path):
        _wav(tmp_path / "a.wav", 0.5)
        p = tmp_path / "bad.csv"
        p.write_text("filename,label,start_time,end_time\na.wav,bird,0.5,0.0\n")
        rows = _parse_table(tmp_path, p, "filename", "label", "start_time", "end_time", "split")
        assert len(rows) == 0
