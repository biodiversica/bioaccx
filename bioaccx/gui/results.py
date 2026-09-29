"""Reading what a run left behind in ``custom_models/``.

Every training run already writes a metadata file, an evaluation table, a
dataset list, a report and — when UMAP is enabled — a projection CSV. On disk
those are inert files opened one at a time. This module parses a model
directory into records the browser can navigate, compare and plot.

Nothing here computes anything: it is a reader over artifacts that already
exist, which is why the explorer works on models trained months ago.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bioaccx.umap import hex_colors

#: Written by a training run. Embeddings and dataset runs do not write it.
METADATA_SUFFIX = "_metadata.json"

#: The row `write_evaluation_csv` puts the macro average on.
OVERALL_ROW = "OVERALL (Macro-avg)"


class ResultsError(Exception):
    """A model directory could not be read."""


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: Files that mark a directory as the output of a run, in the order the stem is
#: taken from. A training run writes the metadata; `bioaccx embeddings` and
#: `bioaccx dataset` do not, so their directories are recognised by the
#: artifacts they do write — otherwise a projection built to inspect a dataset
#: before training it would be invisible.
STEM_MARKERS = (
    METADATA_SUFFIX,
    "_umap.csv",
    "_evaluation.csv",
    "_dataset_list.csv",
)


def _stem_of(directory: Path) -> Optional[str]:
    """The stem a directory's files share, from whichever marker it carries."""
    for suffix in STEM_MARKERS:
        for candidate in sorted(directory.glob(f"*{suffix}")):
            stem = candidate.name[: -len(suffix)]
            # `*_metadata.json` also matches `<stem>_dataset_metadata.json`,
            # which sorts first.
            if suffix == METADATA_SUFFIX and stem.endswith("_dataset"):
                continue
            return stem
    return None


def _kind(files: dict[str, str]) -> str:
    """What the directory is: a trained model, a projection, or a dataset."""
    if files.get("models") or "evaluation" in files:
        return "model"
    if "umap_data" in files:
        return "embeddings"
    return "dataset"


def _created(directory: Path, meta: dict) -> str:
    """When the run happened — recorded if there is metadata, else from disk.

    Without this an embeddings run would sort as if it had no date at all,
    which puts the thing you just made at the bottom of the list.
    """
    recorded = meta.get("created_at", "")
    if recorded:
        return recorded
    stamps = [child.stat().st_mtime for child in directory.iterdir() if child.is_file()]
    if not stamps:
        return ""
    return datetime.fromtimestamp(max(stamps)).isoformat(timespec="seconds")


def _backbone_from_embeddings(directory: Path) -> Optional[str]:
    """The backbone named by an embedding store, e.g. ``0xbb10_embeddings.db``.

    An embeddings-only run leaves no metadata, but it does name the store after
    the registry ID it used, which is enough to say which model produced it.
    """
    from bioaccx.registry import get_registry_defaults

    for store in directory.glob("0x*_embeddings.db"):
        entry = get_registry_defaults(store.name.split("_", 1)[0])
        if entry:
            return f"{entry['name']} v{entry['version']}"
    return None


def _files(directory: Path, stem: str) -> dict[str, str]:
    """The artifacts present, by role, so the UI can enable only what exists."""
    roles = {
        "metadata": f"{stem}{METADATA_SUFFIX}",
        "evaluation": f"{stem}_evaluation.csv",
        "dataset_list": f"{stem}_dataset_list.csv",
        "umap_data": f"{stem}_umap.csv",
        "umap_plot": f"{stem}_umap.png",
        "clusters_plot": f"{stem}_clusters.png",
        "labels": f"{stem}_labels.txt",
    }
    found = {role: str(directory / name)
             for role, name in roles.items() if (directory / name).exists()}
    # Reports are written under a couple of names depending on the classifier.
    for report in (f"{stem}_report.txt", f"{stem}_keras_report.txt",
                   f"{stem}_sklearn_report.txt"):
        if (directory / report).exists():
            found.setdefault("report", str(directory / report))
    found["models"] = ",".join(
        sorted(p.name for p in directory.glob(f"{stem}*")
               if p.suffix in (".onnx", ".tflite")))
    return found


def macro_scores(rows: list[dict]) -> dict[str, Optional[float]]:
    """The macro-average line of an evaluation table, as numbers."""
    for row in rows:
        if row.get("Class", "").startswith("OVERALL"):
            return {
                "f1": _number(row.get("F1 Score (0.5)")),
                "precision": _number(row.get("Precision (0.5)")),
                "recall": _number(row.get("Recall (0.5)")),
                "auprc": _number(row.get("AUPRC")),
                "auroc": _number(row.get("AUROC")),
            }
    return {"f1": None, "precision": None, "recall": None,
            "auprc": None, "auroc": None}


def summarise(directory: Path) -> Optional[dict]:
    """One run's directory as a card: what it is and, if it trained, how it did.

    Covers all three shapes a run leaves behind — a trained model, an
    embeddings run with a projection but no classifier, or a bare dataset
    export — since inspecting a projection before committing to training is a
    normal way to work.
    """
    stem = _stem_of(directory)
    if stem is None:
        return None
    meta = _read_json(directory / f"{stem}{METADATA_SUFFIX}")
    evaluation = _read_csv(directory / f"{stem}_evaluation.csv")
    foundation = meta.get("foundation_model", {})
    dataset = meta.get("dataset", {})
    files = _files(directory, stem)
    backbone = (f"{foundation['name']} v{foundation.get('version', '?')}"
                if foundation.get("name")
                else _backbone_from_embeddings(directory) or "unknown backbone")
    return {
        "stem": stem,
        "dir": str(directory),
        "kind": _kind(files),
        "created_at": _created(directory, meta),
        "backbone": backbone,
        "backbone_format": foundation.get("format", ""),
        "embedding_size": foundation.get("embedding_size"),
        "classifier": meta.get("classifier", ""),
        "n_classes": meta.get("num_classes"),
        "labels": meta.get("labels", []),
        "excluded_labels": meta.get("excluded_labels", []),
        "n_train": dataset.get("n_train"),
        "n_test": dataset.get("n_test"),
        "macro": macro_scores(evaluation),
        "has_umap": "umap_data" in files,
        "has_evaluation": "evaluation" in files,
        "files": files,
    }


def scan(models_dir: Path) -> list[dict]:
    """Every run's directory under *models_dir*, newest first.

    Loose files beside the directories (stray configs, notes) are ignored, as
    are directories holding neither a projection, an evaluation, a dataset list
    nor metadata — an extracted head, for instance, is not a run.
    """
    if not models_dir.is_dir():
        raise ResultsError(f"not a directory: {models_dir}")
    found = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        record = summarise(child)
        if record is not None:
            found.append(record)
    return sorted(found, key=lambda r: r["created_at"], reverse=True)


def _resolve_dir(models_dir: Path, stem: str) -> Path:
    """The directory holding *stem*, refusing anything outside *models_dir*."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", stem):
        raise ResultsError(f"not a model name: {stem!r}")
    directory = (models_dir / stem).resolve()
    if not str(directory).startswith(str(models_dir.resolve())):
        raise ResultsError(f"outside the models directory: {stem!r}")
    if not directory.is_dir():
        raise ResultsError(f"no such model: {stem}")
    return directory


def detail(models_dir: Path, stem: str) -> dict:
    """Everything about one model: summary, metrics table, config, report."""
    directory = _resolve_dir(models_dir, stem)
    summary = summarise(directory)
    if summary is None:
        raise ResultsError(f"{directory.name} holds no run output")
    meta = _read_json(directory / f"{stem}{METADATA_SUFFIX}")
    evaluation = _read_csv(directory / f"{stem}_evaluation.csv")

    report = ""
    if "report" in summary["files"]:
        try:
            report = Path(summary["files"]["report"]).read_text()
        except OSError:
            report = ""

    return {
        **summary,
        "evaluation": [
            {
                "label": row.get("Class", ""),
                "precision": _number(row.get("Precision (0.5)")),
                "recall": _number(row.get("Recall (0.5)")),
                "f1": _number(row.get("F1 Score (0.5)")),
                "precision_opt": _number(row.get("Precision (opt)")),
                "recall_opt": _number(row.get("Recall (opt)")),
                "f1_opt": _number(row.get("F1 Score (opt)")),
                "auprc": _number(row.get("AUPRC")),
                "auroc": _number(row.get("AUROC")),
                "threshold": _number(row.get("Optimal Threshold")),
                "samples": _number(row.get("Samples")),
                "overall": row.get("Class", "").startswith("OVERALL"),
            }
            for row in evaluation
        ],
        "config": {k: v for k, v in meta.items()
                   if k in ("keras_classifier", "sklearn_classifier", "dataset",
                            "output_type", "output_format", "output_data_types")},
        "report": report,
    }


# ── the embedding map ────────────────────────────────────────────────────────

def umap_points(models_dir: Path, stem: str) -> dict:
    """The UMAP projection as plottable points, each carrying its sample key."""
    directory = _resolve_dir(models_dir, stem)
    rows = _read_csv(directory / f"{stem}_umap.csv")
    if not rows:
        raise ResultsError(f"{stem} has no UMAP data (run `bioaccx embeddings`)")

    dims = [c for c in rows[0] if c.startswith("umap_")]
    points = [
        {
            "x": _number(row.get(dims[0])),
            "y": _number(row.get(dims[1])) if len(dims) > 1 else 0.0,
            "label": row.get("label", ""),
            "split": row.get("split", ""),
            "cluster": _number(row.get("cluster")),
            "key": row.get("key", ""),
        }
        for row in rows
    ]

    key_source = "column" if any(p["key"] for p in points) else "none"
    if key_source == "none":
        # CSVs written before the key column exist in quantity. Row order is
        # not guaranteed to line up — samples that fail to embed are dropped
        # from the projection but stay in the dataset list — so it is used only
        # when the two files are the same length and agree label for label.
        # The caller is told which source was used so it can say so.
        dataset_rows = _read_csv(directory / f"{stem}_dataset_list.csv")
        if len(dataset_rows) == len(points) and all(
            row.get("label", "") == point["label"]
            for row, point in zip(dataset_rows, points)
        ):
            for row, point in zip(dataset_rows, points):
                point["key"] = _base_key(row)
            key_source = "row-order"

    labels = sorted({p["label"] for p in points})
    clusters = sorted({int(p["cluster"]) for p in points if p["cluster"] is not None})
    # The colours come from the same generator the PNG plots use, so a class is
    # the same colour in the figure on disk and on the map in the browser — and
    # the browser needs no palette of its own to fall out of step with.
    return {
        "points": points,
        "labels": labels,
        "clusters": clusters,
        "label_colors": hex_colors(len(labels)),
        "cluster_colors": hex_colors(len(clusters)),
        "key_source": key_source,
        "has_keys": key_source != "none",
        "dims": len(dims),
    }


def _base_key(row: dict) -> str:
    """Rebuild the leading part of a sample's embedding key from a dataset row.

    Mirrors ``dataset._npy_filename`` as far as the dataset list can: the file
    stem, the window, and the signal offset. The augmentation suffix cannot be
    rebuilt — the noise offset is not recorded in the CSV — so lookups match on
    this prefix, which is enough to reach the source audio and its window.
    """
    stem = Path(row.get("filepath", "")).stem
    start, end = _number(row.get("start_time")), _number(row.get("end_time"))
    base = f"{stem}_{start:.3f}_{end:.3f}" if start is not None and end is not None else stem
    offset = row.get("signal_offset_samples", "")
    if offset not in ("", None):
        base = f"{base}_off{offset}"
    return base


def dataset_index(models_dir: Path, stem: str) -> dict[str, dict]:
    """Dataset rows keyed by the prefix their embedding key starts with."""
    directory = _resolve_dir(models_dir, stem)
    rows = _read_csv(directory / f"{stem}_dataset_list.csv")
    return {_base_key(row): row for row in rows}


def resolve_key(index: dict[str, dict], key: str) -> Optional[dict]:
    """The dataset row a sample key belongs to.

    An augmented sample's key carries a noise suffix the dataset list cannot
    reproduce, so the longest matching prefix wins — that is the un-augmented
    window the clip was cut from, which is what playback needs.

    >>> index = {"rec_0.000_3.000": {"filepath": "/a/rec.wav"}}
    >>> resolve_key(index, "rec_0.000_3.000_noise_x_t0.000_snr10")["filepath"]
    '/a/rec.wav'
    >>> resolve_key(index, "nothing_like_it") is None
    True
    """
    if not key:
        return None
    if key in index:
        return index[key]
    matches = [k for k in index if key.startswith(k)]
    if not matches:
        return None
    return index[max(matches, key=len)]


def compare(models_dir: Path, left: str, right: str) -> dict:
    """Two models side by side: per-class metric deltas and config differences."""
    a, b = detail(models_dir, left), detail(models_dir, right)

    def by_label(record):
        return {row["label"]: row for row in record["evaluation"]}

    rows_a, rows_b = by_label(a), by_label(b)
    labels = sorted(set(rows_a) | set(rows_b),
                    key=lambda name: (not name.startswith("OVERALL"), name))

    metrics = []
    for label in labels:
        left_row, right_row = rows_a.get(label), rows_b.get(label)
        left_f1 = left_row["f1"] if left_row else None
        right_f1 = right_row["f1"] if right_row else None
        metrics.append({
            "label": label,
            "left_f1": left_f1,
            "right_f1": right_f1,
            "delta": (right_f1 - left_f1)
                     if left_f1 is not None and right_f1 is not None else None,
            "only_in": None if left_row and right_row else (left if left_row else right),
            "overall": label.startswith("OVERALL"),
        })

    return {
        "left": {k: a[k] for k in ("stem", "backbone", "classifier", "n_classes",
                                   "n_train", "n_test", "macro", "created_at")},
        "right": {k: b[k] for k in ("stem", "backbone", "classifier", "n_classes",
                                    "n_train", "n_test", "macro", "created_at")},
        "metrics": metrics,
        "config": _config_diff(a["config"], b["config"]),
    }


def _config_diff(left: dict, right: dict, prefix: str = "") -> list[dict]:
    """Flattened settings that differ between two runs' recorded config."""
    out: list[dict] = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        path = f"{prefix}{key}"
        if isinstance(a, dict) or isinstance(b, dict):
            out.extend(_config_diff(a if isinstance(a, dict) else {},
                                    b if isinstance(b, dict) else {},
                                    prefix=f"{path}."))
        elif a != b:
            out.append({"path": path, "left": a, "right": b})
    return out
