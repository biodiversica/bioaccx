"""Unit tests for bioaccx.report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bioaccx.config import (
    AugmentationConfig,
    BioaccxConfig,
    DatasetConfig,
    FoundationModelConfig,
    OutputConfig,
    _parse_config,
)
from bioaccx.dataset import AudioSample
from bioaccx.report import (
    EVAL_COLUMNS,
    _metrics,
    _scores_from_outputs,
    per_class_evaluation,
    write_dataset_list,
    write_dataset_metadata,
    write_evaluation_csv,
    write_model_metadata,
)


def _sample(
    label: str,
    split: str,
    start: float | None = None,
    end: float | None = None,
    original_path: "Path | None" = None,
) -> AudioSample:
    return AudioSample(Path(f"/tmp/{label}.wav"), label, start, end, split,
                       original_path=original_path)


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


class TestScoresFromOutputs:
    def test_logits_are_softmaxed(self):
        raw = np.array([[2.0, -1.0, 0.0]])
        scores, desc = _scores_from_outputs(raw, "linear (logits)")
        assert scores.sum(axis=1)[0] == pytest.approx(1.0)
        assert "softmax" in desc

    def test_sigmoid_outputs_passed_through(self):
        raw = np.array([[0.9, 0.8, 0.7]])
        scores, desc = _scores_from_outputs(raw, "sigmoid")
        assert scores == pytest.approx(raw)
        assert "sigmoid" in desc

    def test_softmax_outputs_passed_through(self):
        raw = np.array([[0.6, 0.3, 0.1]])
        scores, _ = _scores_from_outputs(raw, "softmax")
        assert scores == pytest.approx(raw)

    def test_unknown_activation_with_out_of_range_values_softmaxed(self):
        raw = np.array([[5.0, -3.0]])
        scores, _ = _scores_from_outputs(raw, None)
        assert scores.sum(axis=1)[0] == pytest.approx(1.0)


class TestPerClassEvaluation:
    def test_perfect_scores(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        rows, macro = per_class_evaluation(y, scores, ["a", "b"])
        assert len(rows) == 2
        assert macro["F1 Score (0.5)"] == pytest.approx(1.0)
        assert macro["AUROC"] == pytest.approx(1.0)
        assert all(r["False Positives"] == 0 and r["False Negatives"] == 0 for r in rows)

    def test_confusion_counts_sum_to_sample_count(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, size=40)
        scores = rng.random((40, 3))
        rows, _ = per_class_evaluation(y, scores, ["a", "b", "c"])
        for r in rows:
            total = (r["True Positives"] + r["False Positives"]
                     + r["True Negatives"] + r["False Negatives"])
            assert total == 40

    def test_samples_and_percentage(self):
        y = np.array([0, 0, 0, 1])
        scores = np.tile([0.5, 0.5], (4, 1))
        rows, _ = per_class_evaluation(y, scores, ["a", "b"])
        assert rows[0]["Samples"] == 3
        assert rows[0]["Percentage (%)"] == pytest.approx(75.0)
        assert rows[1]["Samples"] == 1

    def test_optimal_threshold_beats_default(self):
        # Class 1 positives score just above 0.3 — below the 0.5 default.
        y = np.array([0, 0, 1, 1])
        scores = np.array([[0.9, 0.05], [0.9, 0.05], [0.6, 0.35], [0.6, 0.35]])
        rows, _ = per_class_evaluation(y, scores, ["a", "b"])
        b = rows[1]
        assert b["F1 Score (0.5)"] == pytest.approx(0.0)
        assert b["F1 Score (opt)"] == pytest.approx(1.0)
        assert b["Optimal Threshold"] <= 0.35

    def test_class_without_positives_has_nan_curve_metrics(self):
        y = np.array([0, 0, 1])
        scores = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])
        rows, macro = per_class_evaluation(y, scores, ["a", "b", "empty"])
        assert np.isnan(rows[2]["AUPRC"])
        assert np.isnan(rows[2]["AUROC"])
        # Macro-averages skip the undefined entries rather than becoming nan.
        assert not np.isnan(macro["AUPRC"])


class TestWriteEvaluationCsv:
    def _write(self, tmp_path):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, size=30)
        scores = rng.random((30, 3))
        rows, macro = per_class_evaluation(y, scores, ["a", "b", "c"])
        path = tmp_path / "eval.csv"
        write_evaluation_csv(rows, macro, path)
        return path

    def test_header_matches_birdnet_columns(self, tmp_path):
        path = self._write(tmp_path)
        with path.open() as f:
            assert next(csv.reader(f)) == EVAL_COLUMNS

    def test_macro_row_first_then_one_row_per_class(self, tmp_path):
        path = self._write(tmp_path)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["Class"] == "OVERALL (Macro-avg)"
        assert [r["Class"] for r in rows[1:]] == ["a", "b", "c"]

    def test_macro_row_leaves_per_class_only_columns_empty(self, tmp_path):
        path = self._write(tmp_path)
        with path.open() as f:
            macro_row = next(csv.DictReader(f))
        for col in ("Optimal Threshold", "True Positives", "Samples", "Percentage (%)"):
            assert macro_row[col] == ""
        assert macro_row["AUPRC"] != ""


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
        write_dataset_list(path, train, test, window_seconds=3.0)
        assert path.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_list(path, train, test, window_seconds=3.0)
        with path.open() as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) >= {"filepath", "start_time", "end_time", "label", "split"}
            assert "filename" not in reader.fieldnames

    def test_all_samples_included(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_list(path, train, test)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_start_end_times_populated_with_window(self, tmp_path):
        """Subfolders-mode samples (None times) should get 0.0 / window_seconds."""
        path = tmp_path / "info.csv"
        train = [_sample("bird", "train")]   # no start/end
        write_dataset_list(path, train, [], window_seconds=3.0)
        with path.open() as f:
            row = next(csv.DictReader(f))
        assert float(row["start_time"]) == pytest.approx(0.0)
        assert float(row["end_time"]) == pytest.approx(3.0)

    def test_split_column_correct(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_list(path, train, test)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        splits = {r["split"] for r in rows}
        assert splits == {"train", "test"}

    def test_no_preproc_columns_by_default(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_list(path, train, test)
        with path.open() as f:
            reader = csv.DictReader(f)
            assert "filter" not in reader.fieldnames
            assert "speed" not in reader.fieldnames

    def test_preproc_columns_added_when_filter_set(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_list(path, train, test, filter="hpf", filter_freq=1000.0, filter_order=5, speed=1.0)
        with path.open() as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) >= {"filter", "filter_freq", "filter_order", "speed"}

    def test_preproc_columns_added_when_speed_set(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_list(path, train, test, speed=2.0)
        with path.open() as f:
            reader = csv.DictReader(f)
            assert "speed" in reader.fieldnames

    def test_preproc_column_values_correct(self, tmp_path):
        path = tmp_path / "info.csv"
        train = [_sample("bird", "train", 0.0, 3.0)]
        write_dataset_list(path, train, [], filter="bpf", filter_freq=[500.0, 4000.0],
                           filter_order=3, speed=1.5)
        with path.open() as f:
            row = next(csv.DictReader(f))
        assert row["filter"] == "bpf"
        assert "500" in row["filter_freq"] and "4000" in row["filter_freq"]
        assert row["filter_order"] == "3"
        assert float(row["speed"]) == pytest.approx(1.5)

    def test_original_path_used_when_set(self, tmp_path):
        orig = Path("/original/data/rec.wav")
        tmp_preproc = Path("/tmp/bioaccx_preproc_abc/rec_hpf_xyz.wav")
        s = AudioSample(tmp_preproc, "bird", 0.0, 3.0, "train", original_path=orig)
        path = tmp_path / "info.csv"
        write_dataset_list(path, [s], [], filter="hpf", filter_freq=1000.0)
        with path.open() as f:
            row = next(csv.DictReader(f))
        assert row["filepath"] == str(orig)

    def test_no_original_path_uses_sample_path(self, tmp_path):
        path = tmp_path / "info.csv"
        train, test = self._make_samples()
        write_dataset_list(path, train, test)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert all("/tmp/" in r["filepath"] for r in rows)


class TestWriteDatasetMetadata:
    def _cfg(self, **ds_kwargs):
        fm = FoundationModelConfig(name="birdnet", version="2.4", sample_rate=48000,
                                   window_seconds=3.0)
        ds = DatasetConfig(data_dir="/data/birds", label_mode="subfolders",
                           test_ratio=0.25, random_seed=7, **ds_kwargs)
        return BioaccxConfig(foundation_model=fm, dataset=ds,
                             output=OutputConfig(model_name="clf", model_version="1.0"))

    def _samples(self):
        train = [_sample("bird", "train"), _sample("bird", "train"), _sample("frog", "train")]
        test = [_sample("bird", "test")]
        return train, test

    def _write(self, tmp_path, cfg=None, samples=None):
        path = tmp_path / "dataset_metadata.json"
        train, test = samples if samples is not None else self._samples()
        write_dataset_metadata(path, cfg or self._cfg(), train, test, window_seconds=3.0)
        return json.loads(path.read_text())

    def test_creates_valid_json(self, tmp_path):
        info = self._write(tmp_path)
        assert info["created_at"]
        assert info["model_stem"].startswith("clf_")

    def test_foundation_model_window_recorded(self, tmp_path):
        fmi = self._write(tmp_path)["foundation_model"]
        assert fmi["name"] == "birdnet"
        assert fmi["sample_rate"] == 48000
        assert fmi["window_samples"] == 144000
        assert fmi["window_seconds"] == pytest.approx(3.0)

    def test_source_paths_and_params(self, tmp_path):
        sources = self._write(tmp_path, self._cfg(overlap=0.5, filter="hpf",
                                                  filter_freq=1000.0))["sources"]
        assert len(sources) == 1
        assert sources[0]["data_dir"] == "/data/birds"
        assert sources[0]["label_mode"] == "subfolders"
        assert sources[0]["overlap"] == pytest.approx(0.5)
        assert sources[0]["filter"] == "hpf"

    def test_run_level_fields_separated_from_sources(self, tmp_path):
        info = self._write(tmp_path)
        assert info["run"]["test_ratio"] == pytest.approx(0.25)
        assert info["run"]["random_seed"] == 7
        assert "test_ratio" not in info["sources"][0]
        assert "random_seed" not in info["sources"][0]

    def test_unset_fields_omitted(self, tmp_path):
        src = self._write(tmp_path)["sources"][0]
        assert "filter" not in src        # None
        assert "table_file" not in src    # None

    def test_default_valued_fields_omitted(self, tmp_path):
        src = self._write(tmp_path)["sources"][0]
        assert "filename_col" not in src   # untouched default
        assert "ssh_port" not in src
        assert "min_anchor_fraction" not in src

    def test_shape_defining_fields_kept_at_default(self, tmp_path):
        info = self._write(tmp_path)
        assert info["sources"][0]["label_mode"] == "subfolders"
        assert info["sources"][0]["overlap"] == pytest.approx(0.0)
        assert info["sources"][0]["speed"] == pytest.approx(1.0)
        assert "audio_extensions" in info["run"]

    def test_non_default_field_recorded(self, tmp_path):
        src = self._write(tmp_path, self._cfg(min_anchor_fraction=0.5))["sources"][0]
        assert src["min_anchor_fraction"] == pytest.approx(0.5)

    def test_api_key_redacted(self, tmp_path):
        info = self._write(tmp_path, self._cfg(xc_api_key="super-secret"))
        assert "super-secret" not in json.dumps(info)
        assert info["run"]["xc_api_key"] == "***redacted***"

    def test_per_label_counts(self, tmp_path):
        labels = self._write(tmp_path)["labels"]
        assert labels["bird"] == {"train": 2, "test": 1, "total": 3}
        assert labels["frog"] == {"train": 1, "test": 0, "total": 1}

    def test_totals(self, tmp_path):
        totals = self._write(tmp_path)["totals"]
        assert totals["num_labels"] == 2
        assert totals["n_train"] == 3
        assert totals["n_test"] == 1
        assert totals["n_total"] == 4

    def test_augmented_counts_only_when_augmented(self, tmp_path):
        info = self._write(tmp_path)
        assert "n_augmented" not in info["totals"]
        assert "train_augmented" not in info["labels"]["bird"]

        aug = _sample("bird", "train")
        aug.noise_path = Path("/noise/rain.wav")
        aug.snr = 10.0
        train, test = self._samples()
        info = self._write(tmp_path, samples=(train + [aug], test))
        assert info["totals"]["n_augmented"] == 1
        assert info["labels"]["bird"]["train_augmented"] == 1
        assert info["labels"]["bird"]["test_augmented"] == 0

    def test_multiple_sources_listed(self, tmp_path):
        cfg = _parse_config({
            "foundation_model": {"name": "birdnet", "version": "2.4", "window_seconds": 3.0},
            "dataset": {
                "test_ratio": 0.3,
                "sources": [
                    {"data_dir": "/data/a"},
                    {"data_dir": "/data/b", "label_mode": "file_per_label"},
                ],
            },
        })
        info = self._write(tmp_path, cfg)
        assert [s["data_dir"] for s in info["sources"]] == ["/data/a", "/data/b"]
        assert info["sources"][1]["label_mode"] == "file_per_label"
        assert info["run"]["test_ratio"] == pytest.approx(0.3)

    def test_augmentation_block_recorded(self, tmp_path):
        cfg = self._cfg(augmentation=AugmentationConfig(snr_levels=[0.0, 10.0],
                                                        augmentation_dir="/noise"))
        src = self._write(tmp_path, cfg)["sources"][0]
        assert src["augmentation"]["augmentation_dir"] == "/noise"
        assert src["augmentation"]["snr_levels"] == [0.0, 10.0]

    def test_source_file_count_uses_original_path(self, tmp_path):
        orig = Path("/original/rec.wav")
        s1 = _sample("bird", "train", original_path=orig)
        s2 = _sample("bird", "train", original_path=orig)
        info = self._write(tmp_path, samples=([s1, s2], []))
        assert info["totals"]["n_source_files"] == 1


class TestWriteModelInfo:
    def test_creates_valid_json(self, tmp_path):
        path = tmp_path / "info.json"
        write_model_metadata(path, ["bird", "frog"], {"keras_onnx": "/tmp/model.onnx"},
                         foundation_name="birdnet", foundation_version="2.4",
                         foundation_format="onnx", embed_dim=1024)
        info = json.loads(path.read_text())
        assert info["labels"] == ["bird", "frog"]
        assert info["num_classes"] == 2

    def test_outputs_dict_present(self, tmp_path):
        path = tmp_path / "info.json"
        outputs = {"keras_head": "/tmp/head.onnx"}
        write_model_metadata(path, ["bird"], outputs)
        info = json.loads(path.read_text())
        assert info["outputs"]["keras_head"] == "/tmp/head.onnx"

    def test_created_at_field_present(self, tmp_path):
        path = tmp_path / "info.json"
        write_model_metadata(path, [], {})
        info = json.loads(path.read_text())
        assert "created_at" in info
