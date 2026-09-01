"""Tests for reading a model directory back.

The explorer is a reader over artifacts a run already wrote, so these build
model directories the way a run leaves them and check what comes back —
including the older shapes: a run with no evaluation table, and a projection
written before the key column existed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bioaccx.gui import results


def _model(root: Path, stem: str, *, evaluation=True, umap=True, keys=True,
           created="2026-01-01T00:00:00", macro_f1=0.80, classes=("A", "B")):
    """A model directory shaped like one a real run writes."""
    directory = root / stem
    directory.mkdir(parents=True, exist_ok=True)

    (directory / f"{stem}_metadata.json").write_text(json.dumps({
        "created_at": created,
        "foundation_model": {"name": "birdnet", "version": "2.4",
                             "format": "onnx", "embedding_size": 1024},
        "labels": list(classes),
        "num_classes": len(classes),
        "excluded_labels": ["noise"],
        "classifier": "keras",
        "dataset": {"n_train": 100, "n_test": 25, "test_ratio": 0.2},
        "keras_classifier": {"epochs": 50, "hidden_units": 256},
    }))

    rows = [{"filepath": str(root / f"{name}.wav"), "start_time": "0.0",
             "end_time": "3.0", "label": name, "split": "train",
             "noise_file": "", "snr_db": "", "signal_offset_samples": ""}
            for name in classes]
    with (directory / f"{stem}_dataset_list.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if evaluation:
        with (directory / f"{stem}_evaluation.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Class", "Precision (0.5)", "Recall (0.5)",
                             "F1 Score (0.5)", "Precision (opt)", "Recall (opt)",
                             "F1 Score (opt)", "AUPRC", "AUROC",
                             "Optimal Threshold", "True Positives",
                             "False Positives", "True Negatives",
                             "False Negatives", "Samples", "Percentage (%)"])
            writer.writerow([results.OVERALL_ROW, 0.9, 0.8, macro_f1, 0.9, 0.85,
                             0.87, 0.88, 0.97, "", "", "", "", "", "", ""])
            for i, name in enumerate(classes):
                score = 0.4 if i == 0 else 0.95      # one deliberately weak class
                writer.writerow([name, score, score, score, score, score, score,
                                 score, score, 0.35, 1, 1, 1, 1, 10, 5.0])

    if umap:
        header = ["umap_1", "umap_2", "label", "split"]
        if keys:
            header.append("key")
        header.append("cluster")
        with (directory / f"{stem}_umap.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for i, name in enumerate(classes):
                row = [float(i), float(i) * 2, name, "train"]
                if keys:
                    row.append(f"{name}_0.000_3.000")
                row.append(i)
                writer.writerow(row)

    return directory


class TestScan:
    def test_lists_model_directories(self, tmp_path):
        _model(tmp_path, "A_0xbb00_v1")
        _model(tmp_path, "B_0xbb00_v1")
        assert [m["stem"] for m in results.scan(tmp_path)] == ["A_0xbb00_v1", "B_0xbb00_v1"]

    def test_newest_first(self, tmp_path):
        _model(tmp_path, "old", created="2025-01-01T00:00:00")
        _model(tmp_path, "new", created="2026-06-01T00:00:00")
        assert [m["stem"] for m in results.scan(tmp_path)] == ["new", "old"]

    def test_ignores_directories_without_metadata(self, tmp_path):
        _model(tmp_path, "real")
        (tmp_path / "not_a_model").mkdir()
        (tmp_path / "notes.txt").write_text("hello")
        assert [m["stem"] for m in results.scan(tmp_path)] == ["real"]

    def test_ignores_loose_config_files_beside_the_directories(self, tmp_path):
        _model(tmp_path, "real")
        (tmp_path / "some_config.yaml").write_text("foundation_model: {}\n")
        assert len(results.scan(tmp_path)) == 1

    def test_a_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(results.ResultsError, match="not a directory"):
            results.scan(tmp_path / "nope")

    def test_summary_carries_the_macro_score(self, tmp_path):
        _model(tmp_path, "m", macro_f1=0.77)
        assert results.scan(tmp_path)[0]["macro"]["f1"] == pytest.approx(0.77)

    def test_a_run_without_an_evaluation_table_still_lists(self, tmp_path):
        _model(tmp_path, "m", evaluation=False)
        record = results.scan(tmp_path)[0]
        assert record["macro"]["f1"] is None
        assert record["has_evaluation"] is False


class TestDetail:
    def test_returns_per_class_rows(self, tmp_path):
        _model(tmp_path, "m", classes=("A", "B", "C"))
        detail = results.detail(tmp_path, "m")
        assert len(detail["evaluation"]) == 4          # overall + three classes
        assert sum(row["overall"] for row in detail["evaluation"]) == 1

    def test_records_the_training_settings(self, tmp_path):
        _model(tmp_path, "m")
        assert results.detail(tmp_path, "m")["config"]["keras_classifier"]["epochs"] == 50

    def test_rejects_a_name_that_escapes_the_models_directory(self, tmp_path):
        with pytest.raises(results.ResultsError, match="not a model name"):
            results.detail(tmp_path, "../etc")

    def test_rejects_a_path_separator(self, tmp_path):
        with pytest.raises(results.ResultsError, match="not a model name"):
            results.detail(tmp_path, "a/b")

    def test_unknown_model(self, tmp_path):
        with pytest.raises(results.ResultsError, match="no such model"):
            results.detail(tmp_path, "missing")


class TestUmapPoints:
    def test_points_carry_their_key(self, tmp_path):
        _model(tmp_path, "m")
        points = results.umap_points(tmp_path, "m")
        assert points["key_source"] == "column"
        assert all(point["key"] for point in points["points"])

    def test_labels_and_clusters_are_listed(self, tmp_path):
        _model(tmp_path, "m", classes=("A", "B", "C"))
        points = results.umap_points(tmp_path, "m")
        assert points["labels"] == ["A", "B", "C"]
        assert points["clusters"] == [0, 1, 2]

    def test_a_projection_without_keys_falls_back_to_row_order(self, tmp_path):
        """CSVs written before the key column still resolve, when it is safe."""
        _model(tmp_path, "m", keys=False)
        points = results.umap_points(tmp_path, "m")
        assert points["key_source"] == "row-order"
        assert points["points"][0]["key"] == "A_0.000_3.000"

    def test_the_fallback_is_refused_when_the_files_disagree(self, tmp_path):
        """Mismatched lengths mean row order cannot be trusted — say so instead."""
        directory = _model(tmp_path, "m", keys=False)
        with (directory / "m_dataset_list.csv").open("a", newline="") as fh:
            fh.write(f"{tmp_path}/extra.wav,0.0,3.0,A,train,,,\n")
        points = results.umap_points(tmp_path, "m")
        assert points["key_source"] == "none"
        assert points["has_keys"] is False

    def test_the_fallback_is_refused_when_labels_do_not_line_up(self, tmp_path):
        directory = _model(tmp_path, "m", keys=False, classes=("A", "B"))
        rows = list(csv.DictReader((directory / "m_dataset_list.csv").open(newline="")))
        rows[0]["label"] = "MISMATCH"
        with (directory / "m_dataset_list.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        assert results.umap_points(tmp_path, "m")["key_source"] == "none"

    def test_a_model_without_a_projection(self, tmp_path):
        _model(tmp_path, "m", umap=False)
        with pytest.raises(results.ResultsError, match="no UMAP data"):
            results.umap_points(tmp_path, "m")


class TestKeyResolution:
    def test_an_exact_key_resolves(self, tmp_path):
        _model(tmp_path, "m")
        index = results.dataset_index(tmp_path, "m")
        assert results.resolve_key(index, "A_0.000_3.000")["label"] == "A"

    def test_an_augmented_key_resolves_to_its_source_window(self, tmp_path):
        """The noise suffix is not in the dataset list; the prefix still finds it."""
        _model(tmp_path, "m")
        index = results.dataset_index(tmp_path, "m")
        augmented = "A_0.000_3.000_noise_rain_t1.500_snr10"
        assert results.resolve_key(index, augmented)["label"] == "A"

    def test_an_unknown_key_resolves_to_nothing(self, tmp_path):
        _model(tmp_path, "m")
        index = results.dataset_index(tmp_path, "m")
        assert results.resolve_key(index, "nothing_like_this") is None

    def test_an_empty_key_resolves_to_nothing(self, tmp_path):
        _model(tmp_path, "m")
        assert results.resolve_key(results.dataset_index(tmp_path, "m"), "") is None

    def test_the_longest_matching_prefix_wins(self):
        index = {"rec_0.000_3.000": {"n": "short"},
                 "rec_0.000_3.000_off99": {"n": "long"}}
        assert results.resolve_key(index, "rec_0.000_3.000_off99_noise_x")["n"] == "long"


class TestCompare:
    def test_reports_per_class_deltas(self, tmp_path):
        _model(tmp_path, "left", macro_f1=0.70)
        _model(tmp_path, "right", macro_f1=0.85)
        body = results.compare(tmp_path, "left", "right")
        overall = next(row for row in body["metrics"] if row["overall"])
        assert overall["delta"] == pytest.approx(0.15)

    def test_flags_a_class_only_one_model_has(self, tmp_path):
        _model(tmp_path, "left", classes=("A", "B"))
        _model(tmp_path, "right", classes=("A", "B", "C"))
        body = results.compare(tmp_path, "left", "right")
        only = next(row for row in body["metrics"] if row["label"] == "C")
        assert only["only_in"] == "right" and only["delta"] is None

    def test_lists_settings_that_differ(self, tmp_path):
        _model(tmp_path, "left")
        directory = _model(tmp_path, "right")
        meta_path = directory / "right_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["keras_classifier"]["epochs"] = 200
        meta_path.write_text(json.dumps(meta))

        diff = results.compare(tmp_path, "left", "right")["config"]
        epochs = next(row for row in diff if row["path"].endswith("epochs"))
        assert (epochs["left"], epochs["right"]) == (50, 200)

    def test_identical_runs_differ_in_nothing(self, tmp_path):
        _model(tmp_path, "left")
        _model(tmp_path, "right")
        assert results.compare(tmp_path, "left", "right")["config"] == []
