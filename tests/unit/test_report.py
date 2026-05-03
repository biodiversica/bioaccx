"""Unit tests for bioaccx.report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bioaccx.dataset import AudioSample
from bioaccx.report import write_dataset_info, write_model_info, _metrics


def _sample(label: str, split: str, start: float | None = None, end: float | None = None) -> AudioSample:
    return AudioSample(Path(f"/tmp/{label}.wav"), label, start, end, split)


class TestMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        report, m = _metrics(y, y, ["a", "b", "c"])
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        _, m = _metrics(y_true, y_pred, ["a", "b"])
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["macro_f1"] == pytest.approx(0.0)

    def test_returns_classification_report_string(self):
        y = np.array([0, 1])
        report, _ = _metrics(y, y, ["bird", "frog"])
        assert "bird" in report
        assert "frog" in report


class TestWriteDatasetInfo:
    def _make_samples(self):
        train = [
            _sample("bird", "train", 0.0, 3.0),
            _sample("frog", "train", 1.5, 4.5),
        ]
        test = [
            _sample("bird", "test", 0.0, 3.0),
        ]
        return train, test

    def test_creates_csv(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_info(path, train, test, window_seconds=3.0)
        assert path.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_info(path, train, test, window_seconds=3.0)
        with path.open() as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) >= {"filename", "start_time", "end_time", "label", "split"}

    def test_all_samples_included(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_info(path, train, test)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_start_end_times_populated_with_window(self, tmp_path):
        """Subfolders-mode samples (None times) should get 0.0 / window_seconds."""
        path = tmp_path / "info.csv"
        train = [_sample("bird", "train")]   # no start/end
        write_dataset_info(path, train, [], window_seconds=3.0)
        with path.open() as f:
            row = next(csv.DictReader(f))
        assert float(row["start_time"]) == pytest.approx(0.0)
        assert float(row["end_time"]) == pytest.approx(3.0)

    def test_split_column_correct(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_info(path, train, test)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        splits = {r["split"] for r in rows}
        assert splits == {"train", "test"}


class TestWriteModelInfo:
    def test_creates_valid_json(self, tmp_path):
        path = tmp_path / "info.json"
        write_model_info(path, ["bird", "frog"], {"keras_onnx": "/tmp/model.onnx"},
                         foundation_name="birdnet", foundation_version="2.4",
                         foundation_format="onnx", embed_dim=1024)
        info = json.loads(path.read_text())
        assert info["labels"] == ["bird", "frog"]
        assert info["num_classes"] == 2

    def test_outputs_dict_present(self, tmp_path):
        path = tmp_path / "info.json"
        outputs = {"keras_head": "/tmp/head.onnx"}
        write_model_info(path, ["bird"], outputs)
        info = json.loads(path.read_text())
        assert info["outputs"]["keras_head"] == "/tmp/head.onnx"

    def test_created_at_field_present(self, tmp_path):
        path = tmp_path / "info.json"
        write_model_info(path, [], {})
        info = json.loads(path.read_text())
        assert "created_at" in info
