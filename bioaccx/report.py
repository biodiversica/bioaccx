"""Training report and model info JSON generation."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import csv
import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def _metrics(y_true, y_pred, label_names) -> tuple[str, dict]:
    """Compute a per-class text report and a dict of macro-averaged summary stats.

    ``labels`` is passed explicitly so that classes with zero test samples still
    appear in the report rather than being silently omitted by sklearn.
    ``zero_division=0`` prevents warnings when a class has no predicted samples.
    """
    report = classification_report(y_true, y_pred, target_names=label_names,
                                   labels=list(range(len(label_names))), zero_division=0)
    return report, {
        "accuracy":   accuracy_score(y_true, y_pred),
        "macro_f1":   f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "macro_pre":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_rec":  recall_score(y_true, y_pred, average="macro",    zero_division=0),
    }


def _header(title: str) -> list[str]:
    """Return a three-line banner: bar, title, bar — for plain-text reports."""
    bar = "=" * 62
    return [bar, title, bar]


def _predict_onnx(session: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    """Run ONNX inference and return integer class predictions for all rows of X.

    sklearn ONNX models output ``[label, probabilities]``; we pick
    ``probabilities`` and argmax.  Keras ONNX models output a single scores
    tensor; we take the last output and argmax.  If the output is already 1-D
    integers (e.g. a label-only export) it is cast directly.
    """
    input_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    # sklearn → [label, probabilities]; keras → [scores]
    out_name = "probabilities" if "probabilities" in out_names else out_names[-1]
    probs = session.run([out_name], {input_name: X})[0]
    return np.argmax(probs, axis=1) if probs.ndim > 1 else probs.astype(np.int64)


def write_sklearn_report(
    pipe,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    path: Path,
    **meta,
) -> None:
    """Write a plain-text training report for the sklearn LogisticRegression pipeline.

    Extracts the 'clf' step from the pipeline to report hyperparameters.
    Additional metadata (data_dir, foundation_name, etc.) is passed via **meta.
    """
    from sklearn.pipeline import Pipeline
    clf = pipe.named_steps["clf"]
    y_pred = pipe.predict(X_test)
    report, m = _metrics(y_test, y_pred, label_names)

    lines = _header("Training Report — Sklearn LogisticRegression")
    lines += [
        f"Generated:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Data dir:    {meta.get('data_dir', '—')}",
        f"Model:       {meta.get('foundation_name', '—')} v{meta.get('foundation_version', '—')}  "
        f"(embed_dim={meta.get('embed_dim', '—')})",
        f"Test ratio:  {meta.get('test_ratio', '—')}",
        f"Samples:     train={meta.get('n_train', '—')}  test={meta.get('n_test', '—')}",
        f"Classes ({len(label_names)}): {', '.join(label_names)}",
        "",
        "--- Parameters ---",
        f"  solver:    {clf.solver}",
        f"  C:         {clf.C}",
        f"  max_iter:  {clf.max_iter}",
        f"  scaler:    StandardScaler",
        "",
        "--- Test Set Performance ---",
        report,
        "--- Summary ---",
        f"  Accuracy:        {m['accuracy']:.4f}",
        f"  Macro F1:        {m['macro_f1']:.4f}",
        f"  Macro Precision: {m['macro_pre']:.4f}",
        f"  Macro Recall:    {m['macro_rec']:.4f}",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"  Sklearn report   → {path}")


def write_keras_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    path: Path,
    **meta,
) -> None:
    """Write a plain-text training report for the Keras dense classifier.

    Reads training history and hyperparameter metadata stored on the model
    object by train_keras (``_report_history``, ``_report_params``).
    Falls back gracefully when those attributes are missing.
    """
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report, m = _metrics(y_test, y_pred, label_names)

    hist   = getattr(model, "_report_history", {})
    params = getattr(model, "_report_params", {})
    best   = params.get("best_epoch", "—")
    total  = params.get("epochs", len(hist.get("loss", [])))

    lines = _header("Training Report — Keras Classifier")
    lines += [
        f"Generated:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Data dir:    {meta.get('data_dir', '—')}",
        f"Model:       {meta.get('foundation_name', '—')} v{meta.get('foundation_version', '—')}  "
        f"(embed_dim={meta.get('embed_dim', '—')})",
        f"Test ratio:  {meta.get('test_ratio', '—')}",
        f"Samples:     train={meta.get('n_train', '—')}  test={meta.get('n_test', '—')}",
        f"Classes ({len(label_names)}): {', '.join(label_names)}",
        "",
        "--- Parameters ---",
        f"  hidden_units:     {params.get('hidden_units', '—')}",
        f"  dropout:          {params.get('dropout', '—')}",
        f"  epochs:           {total}",
        f"  batch_size:       {params.get('batch_size', '—')}",
        f"  learning_rate:    {params.get('learning_rate', '—')}",
        f"  output_activation:{params.get('output_activation', '—')}",
        f"  loss:             {params.get('loss', '—')}",
        "",
        "--- Training History ---",
    ]

    if hist:
        n = len(hist["loss"])
        hdr = f"{'Epoch':>6}  {'Loss':>8}  {'Val Loss':>8}  {'Acc':>8}  {'Val Acc':>8}"
        lines += [hdr, "-" * len(hdr)]
        for ep in range(n):
            marker = " ←best" if (ep + 1) == best else ""
            lines.append(
                f"{ep+1:>6}  {hist['loss'][ep]:>8.4f}  {hist['val_loss'][ep]:>8.4f}  "
                f"{hist['accuracy'][ep]:>8.4f}  {hist['val_accuracy'][ep]:>8.4f}{marker}"
            )

    val_loss_best = (
        f"{hist['val_loss'][best - 1]:.4f}" if hist and isinstance(best, int) else "—"
    )
    lines += [
        "",
        "--- Test Set Performance ---",
        report,
        "--- Summary ---",
        f"  Best epoch:      {best}/{total}",
        f"  Final val_loss:  {val_loss_best}",
        f"  Accuracy:        {m['accuracy']:.4f}",
        f"  Macro F1:        {m['macro_f1']:.4f}",
        f"  Macro Precision: {m['macro_pre']:.4f}",
        f"  Macro Recall:    {m['macro_rec']:.4f}",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"  Keras report     → {path}")


def write_comparison_report(
    onnx_paths: dict[str, Path],
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    path: Path,
    **meta,
) -> None:
    """Write a side-by-side comparison for all exported head ONNX classifiers."""
    sessions = {name: ort.InferenceSession(str(p)) for name, p in onnx_paths.items()}
    preds    = {name: _predict_onnx(sess, X_test) for name, sess in sessions.items()}
    metrics  = {name: _metrics(y_test, p, label_names)[1] for name, p in preds.items()}

    lines = _header("Comparison Report — ONNX Classifier(s)")
    lines += [
        f"Generated:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Data dir:    {meta.get('data_dir', '—')}",
        f"Test samples: {len(y_test)}",
        "",
        "--- Summary ---",
        f"  {'Metric':<22}" + "".join(f"  {n:>12}" for n in sessions),
        f"  {'-'*22}" + "".join(f"  {'-'*12}" for _ in sessions),
    ]
    for metric in ("accuracy", "macro_f1", "macro_pre", "macro_rec"):
        row = f"  {metric:<22}"
        for name in sessions:
            row += f"  {metrics[name][metric]:>12.4f}"
        lines.append(row)

    if len(sessions) == 2:
        names = list(preds)
        agree = float(np.mean(preds[names[0]] == preds[names[1]]))
        lines.append(f"\n  Agreement: {agree:.4f}")

    for name, p in preds.items():
        lines += ["", f"--- {name} — Per-class Report ---",
                  classification_report(y_test, p, target_names=label_names)]

    path.write_text("\n".join(lines))
    print(f"  Comparison report → {path}")


def write_dataset_list(
    path: Path,
    train_samples: list,
    test_samples: list,
    window_seconds: float | None = None,
    filter: str | None = None,
    filter_freq: float | list[float] | None = None,
    filter_order: int = 5,
    speed: float = 1.0,
) -> None:
    """Write a CSV listing every sample used, its time bounds, label, and split.

    For samples without explicit start/end times (subfolders mode), the window
    is filled in as 0.0 / window_seconds when that value is provided.

    When filter or speed preprocessing was applied, the *filename* and *filepath*
    columns contain the original source file (before preprocessing), and extra
    columns document the preprocessing parameters.
    """
    has_preproc = filter is not None or speed != 1.0
    all_samples = list(train_samples) + list(test_samples)
    has_augmentation = any(getattr(s, "noise_path", None) is not None for s in all_samples)
    has_shift = any(getattr(s, "signal_offset_samples", None) is not None for s in all_samples)

    def _row(s, split: str) -> dict:
        # Use original path when available (preprocessing was applied)
        src = s.original_path if s.original_path is not None else s.path
        start = s.start_time if s.start_time is not None else 0.0
        if s.end_time is not None:
            end = s.end_time
        elif window_seconds is not None:
            end = start + window_seconds
        else:
            end = ""
        row: dict = {
            "filepath":   str(src),
            "start_time": start,
            "end_time":   end,
            "label":      s.label,
            "split":      split,
        }
        if has_preproc:
            freq_str = (
                str(filter_freq) if not isinstance(filter_freq, (list, tuple))
                else "[" + ", ".join(str(f) for f in filter_freq) + "]"
            )
            row["filter"]       = filter or ""
            row["filter_freq"]  = freq_str if filter is not None else ""
            row["filter_order"] = filter_order if filter is not None else ""
            row["speed"]        = speed
        if has_augmentation:
            noise_path = getattr(s, "noise_path", None)
            snr_val = getattr(s, "snr", None)
            row["noise_file"] = noise_path.name if noise_path is not None else ""
            row["snr_db"]     = snr_val if snr_val is not None else ""
        if has_shift:
            offset = getattr(s, "signal_offset_samples", None)
            row["signal_offset_samples"] = offset if offset is not None else ""
        return row

    rows = [_row(s, "train") for s in train_samples] + [_row(s, "test") for s in test_samples]

    fieldnames = ["filepath", "start_time", "end_time", "label", "split"]
    if has_preproc:
        fieldnames += ["filter", "filter_freq", "filter_order", "speed"]
    if has_augmentation:
        fieldnames += ["noise_file", "snr_db"]
    if has_shift:
        fieldnames += ["signal_offset_samples"]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Dataset info CSV → {path}")


def write_model_metadata(
    path: Path,
    label_names: list[str],
    outputs: dict[str, str],
    **meta,
) -> None:
    """Write a machine-readable JSON summary of the trained model and its outputs.

    ``label_names`` contains only the exported labels (excluded labels are
    omitted).  ``outputs`` maps logical names (e.g. ``"keras_onnx_head"``) to
    the actual file paths written during the export phase.
    """
    info = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "foundation_model": {
            "name":          meta.get("foundation_name"),
            "version":       meta.get("foundation_version"),
            "data_type":     meta.get("foundation_data_type"),
            "format":        meta.get("foundation_format"),
            "embedding_size": meta.get("embed_dim"),
        },
        "labels":          label_names,
        "num_classes":     len(label_names),
        "excluded_labels": meta.get("excluded_labels", []),
        "classifier":    meta.get("classifier"),
        "output_type":   meta.get("output_type"),
        "output_format": meta.get("output_format"),
        "dataset": {
            "data_dir":  meta.get("data_dir"),
            "n_train":   meta.get("n_train"),
            "n_test":    meta.get("n_test"),
            "test_ratio": meta.get("test_ratio"),
        },
        "outputs": outputs,
    }
    aug = meta.get("augmentation")
    if aug is not None:
        info["augmentation"] = {
            "augmentation_dir": aug.augmentation_dir,
            "snr_levels":       list(aug.snr_levels),
            "keep_original":    aug.keep_original,
            "augment_test":     aug.augment_test,
        }
    keras_params = meta.get("keras_params")
    if keras_params:
        info["keras_classifier"] = dict(keras_params)
    path.write_text(json.dumps(info, indent=2))
    print(f"  Model info JSON  → {path}")
