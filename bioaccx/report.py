"""Training report and model info JSON generation."""
from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path
import csv
import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


# Threshold grid used to search for the F1-optimal per-class threshold.
# Matches BirdNET-Analyzer (np.arange(0.1, 0.9, 0.05)) so that the reported
# "Optimal Threshold" is directly comparable with its *_evaluation.csv.
_THRESHOLD_GRID = np.arange(0.1, 0.9, 0.05)

# Column names of BirdNET-Analyzer's *_evaluation.csv, in its own order.
EVAL_COLUMNS = [
    "Class",
    "Precision (0.5)", "Recall (0.5)", "F1 Score (0.5)",
    "Precision (opt)", "Recall (opt)", "F1 Score (opt)",
    "AUPRC", "AUROC", "Optimal Threshold",
    "True Positives", "False Positives", "True Negatives", "False Negatives",
    "Samples", "Percentage (%)",
]


def _scores_from_outputs(raw: np.ndarray, output_activation: str | None) -> tuple[np.ndarray, str]:
    """Turn raw model outputs into per-class scores in [0, 1] plus a description.

    ``output_activation`` is the label stored by the trainer
    (``"sigmoid"``, ``"softmax"`` or ``"linear (logits)"``).  Logits are mapped
    with a softmax — the same transform ``train_keras`` uses for its own
    predictions — so the reported thresholds refer to the values a caller would
    actually see.  When the label is missing or unrecognised, the outputs are
    inspected: anything outside [0, 1] is treated as logits.
    """
    raw = np.asarray(raw, dtype=np.float64)
    act = (output_activation or "").lower()

    if act.startswith("grouped"):
        return raw, "grouped softmax outputs (each group sums to 1)"
    if act.startswith("sigmoid"):
        return raw, "sigmoid outputs"
    if act.startswith("softmax"):
        return raw, "softmax outputs"
    if not act.startswith("linear") and raw.min() >= 0.0 and raw.max() <= 1.0:
        return raw, "model outputs (already in [0, 1])"

    shifted = raw - raw.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True), "softmax of logits"


def _optimal_threshold(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    """Return the threshold in ``_THRESHOLD_GRID`` that maximises the F1 score."""
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in _THRESHOLD_GRID:
        f1 = f1_score(y_true_bin, (scores >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold


def per_class_evaluation(
    y_true: np.ndarray,
    scores: np.ndarray,
    label_names: list[str],
    y_positives: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """One-vs-rest evaluation mirroring BirdNET-Analyzer's *_evaluation.csv.

    ``y_true`` holds integer class indices (bioaccx heads are single-label), and
    ``scores`` is the (n_samples, n_classes) score matrix.  Each class is scored
    as its own binary problem: positives are the samples of that class, and the
    score is that class' column.

    ``y_positives`` overrides how positives are derived, as an explicit
    (n_samples, n_classes) 0/1 matrix.  A grouped head needs this because its
    columns are not training classes: a ``<group>_none`` column counts every
    sample whose label falls outside that group as a positive, which no
    ``y_true == i`` test can express.  When given, ``y_true`` is ignored.

    Returns ``(rows, macro)`` where *rows* has one dict per class keyed by
    :data:`EVAL_COLUMNS` names and *macro* holds the macro-averages.  AUPRC and
    AUROC are undefined for a class with no positive (or no negative) test
    sample and are reported as ``nan``; those are skipped when macro-averaging.
    """
    y_true = np.asarray(y_true).astype(int)
    n_samples = len(y_true) if y_positives is None else len(y_positives)
    positives = None if y_positives is None else np.asarray(y_positives).astype(int)
    rows: list[dict] = []

    for i, name in enumerate(label_names):
        y_bin = (y_true == i).astype(int) if positives is None else positives[:, i]
        s = np.asarray(scores)[:, i]
        n_pos = int(y_bin.sum())

        y_default = (s >= 0.5).astype(int)
        threshold = _optimal_threshold(y_bin, s)
        y_opt = (s >= threshold).astype(int)

        # roc_auc/average_precision need both classes present.
        both_present = 0 < n_pos < n_samples
        tn, fp, fn, tp = confusion_matrix(y_bin, y_opt, labels=[0, 1]).ravel()

        rows.append({
            "Class":            name,
            "Precision (0.5)":  precision_score(y_bin, y_default, zero_division=0),
            "Recall (0.5)":     recall_score(y_bin, y_default, zero_division=0),
            "F1 Score (0.5)":   f1_score(y_bin, y_default, zero_division=0),
            "Precision (opt)":  precision_score(y_bin, y_opt, zero_division=0),
            "Recall (opt)":     recall_score(y_bin, y_opt, zero_division=0),
            "F1 Score (opt)":   f1_score(y_bin, y_opt, zero_division=0),
            "AUPRC":            average_precision_score(y_bin, s) if both_present else float("nan"),
            "AUROC":            roc_auc_score(y_bin, s) if both_present else float("nan"),
            "Optimal Threshold": threshold,
            "True Positives":   int(tp),
            "False Positives":  int(fp),
            "True Negatives":   int(tn),
            "False Negatives":  int(fn),
            "Samples":          n_pos,
            "Percentage (%)":   (n_pos / n_samples * 100) if n_samples else 0.0,
        })

    def _macro(col: str) -> float:
        values = np.array([r[col] for r in rows], dtype=float)
        values = values[~np.isnan(values)]
        return float(np.mean(values)) if len(values) else float("nan")

    macro = {col: _macro(col) for col in EVAL_COLUMNS[1:9]}
    return rows, macro


def _evaluation_lines(rows: list[dict], macro: dict, score_desc: str) -> list[str]:
    """Render :func:`per_class_evaluation` output as a fixed-width text table."""
    name_w = max([len("OVERALL (Macro-avg)")] + [len(r["Class"]) for r in rows])
    # (header, source key, width, formatter)
    cols: list[tuple[str, str, int, str]] = [
        ("Prec(0.5)", "Precision (0.5)",   9, "f4"),
        ("Rec(0.5)",  "Recall (0.5)",      9, "f4"),
        ("F1(0.5)",   "F1 Score (0.5)",    9, "f4"),
        ("Prec(opt)", "Precision (opt)",   9, "f4"),
        ("Rec(opt)",  "Recall (opt)",      9, "f4"),
        ("F1(opt)",   "F1 Score (opt)",    9, "f4"),
        ("AUPRC",     "AUPRC",             9, "f4"),
        ("AUROC",     "AUROC",             9, "f4"),
        ("OptThr",    "Optimal Threshold", 7, "f2"),
        ("TP",        "True Positives",    6, "d"),
        ("FP",        "False Positives",   6, "d"),
        ("TN",        "True Negatives",    6, "d"),
        ("FN",        "False Negatives",   6, "d"),
        ("Samples",   "Samples",           8, "d"),
        ("%",         "Percentage (%)",    7, "f2"),
    ]

    def _cell(value, width: int, fmt: str) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return f"{'—':>{width}}"
        if fmt == "d":
            return f"{int(value):>{width}d}"
        return f"{value:>{width}.{2 if fmt == 'f2' else 4}f}"

    header = f"{'Class':<{name_w}}" + "".join(f"  {h:>{w}}" for h, _, w, _ in cols)
    lines = [
        f"Scores: {score_desc}; each class evaluated one-vs-rest.",
        "Optimal threshold maximises per-class F1 over a 0.10–0.85 grid (step 0.05);",
        "TP/FP/TN/FN are counted at that optimal threshold.",
        "",
        header,
        "-" * len(header),
        (
            f"{'OVERALL (Macro-avg)':<{name_w}}"
            + "".join(
                f"  {_cell(macro[key], w, fmt)}" if key in macro else " " * (w + 2)
                for _, key, w, fmt in cols
            )
        ).rstrip(),
    ]
    for r in rows:
        lines.append(
            f"{r['Class']:<{name_w}}"
            + "".join(f"  {_cell(r[key], w, fmt)}" for _, key, w, fmt in cols)
        )
    return lines


def write_evaluation_csv(rows: list[dict], macro: dict, path: Path) -> None:
    """Write the per-class evaluation as a BirdNET-Analyzer-compatible CSV."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EVAL_COLUMNS)

        def _fmt(value, col: str) -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return ""
            if col in ("True Positives", "False Positives", "True Negatives",
                       "False Negatives", "Samples"):
                return str(int(value))
            if col in ("Optimal Threshold", "Percentage (%)"):
                return f"{value:.2f}"
            return f"{value:.4f}"

        writer.writerow(
            ["OVERALL (Macro-avg)"]
            + [_fmt(macro.get(c), c) if c in macro else "" for c in EVAL_COLUMNS[1:]]
        )
        for r in rows:
            writer.writerow([r["Class"]] + [_fmt(r[c], c) for c in EVAL_COLUMNS[1:]])
    print(f"  Evaluation CSV   → {path}")


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
    eval_csv_path: Path | None = None,
    **meta,
) -> None:
    """Write a plain-text training report for the Keras dense classifier.

    Reads training history and hyperparameter metadata stored on the model
    object by train_keras (``_report_history``, ``_report_params``).
    Falls back gracefully when those attributes are missing.

    When ``eval_csv_path`` is given, the per-class one-vs-rest evaluation is
    also written there in BirdNET-Analyzer's *_evaluation.csv layout.
    """
    hist   = getattr(model, "_report_history", {})
    params = getattr(model, "_report_params", {})
    best   = params.get("best_epoch", "—")
    total  = params.get("epochs", len(hist.get("loss", [])))

    raw = model.predict(X_test, verbose=0)
    scores, score_desc = _scores_from_outputs(raw, params.get("output_activation"))

    label_groups = params.get("label_groups") or {}
    if label_groups:
        # A grouped head's columns are not the training labels: each group holds its
        # members plus a synthetic "<group>_none".  Every group is its own
        # single-label problem, so accuracy is reported per group and the
        # one-vs-rest table is driven by an explicit positives matrix.
        from bioaccx.trainers.grouped import (
            build_group_space, grouped_predictions, grouped_targets,
        )
        output_labels, group_slices = build_group_space(label_groups, label_names)
        positives = grouped_targets(y_test, label_names, label_groups, group_slices)
        picks = grouped_predictions(scores, group_slices)
        truth_cols = np.stack(
            [start + np.argmax(positives[:, start:end], axis=1) for start, end in group_slices],
            axis=1,
        )

        group_reports, group_accuracies = [], []
        for gi, ((start, end), group) in enumerate(zip(group_slices, label_groups)):
            names = output_labels[start:end]
            text, gm = _metrics(truth_cols[:, gi] - start, picks[:, gi] - start, names)
            group_reports += [f"--- group: {group} ---", text]
            group_accuracies.append(gm["accuracy"])
        report = "\n".join(group_reports)
        m = {"accuracy": float(np.mean(group_accuracies)),
             "macro_f1": float("nan"), "macro_pre": float("nan"), "macro_rec": float("nan")}
        eval_rows, eval_macro = per_class_evaluation(
            y_test, scores, output_labels, y_positives=positives)
        grouped_summary = [
            f"  Groups:          {len(group_slices)} "
            f"({', '.join(f'{g}[{len(mm)}]' for g, mm in label_groups.items())})",
            f"  Mean group acc:  {m['accuracy']:.4f}",
            # the strictest reading of a correct window: every group simultaneously right
            f"  All-groups acc:  {float(np.mean(np.all(picks == truth_cols, axis=1))):.4f}",
        ]
    else:
        y_pred = np.argmax(scores, axis=1)
        report, m = _metrics(y_test, y_pred, label_names)
        eval_rows, eval_macro = per_class_evaluation(y_test, scores, label_names)
        grouped_summary = []

    lines = _header("Training Report — Keras Classifier")
    lines += [
        f"Generated:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Data dir:    {meta.get('data_dir', '—')}",
        f"Model:       {meta.get('foundation_name', '—')} v{meta.get('foundation_version', '—')}  "
        f"(embed_dim={meta.get('embed_dim', '—')})",
        f"Test ratio:  {meta.get('test_ratio', '—')}",
        f"Samples:     train={meta.get('n_train', '—')}  test={meta.get('n_test', '—')}",
        f"Classes ({len(label_names)}): {', '.join(label_names)}",
        *([f"Outputs ({len(eval_rows)}): {', '.join(r['Class'] for r in eval_rows)}"]
          if label_groups else []),
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
        "--- Per-class Evaluation (thresholded, one-vs-rest) ---",
        *_evaluation_lines(eval_rows, eval_macro, score_desc),
        *([
            "",
            "NOTE: export_logits is enabled — the exported head emits raw logits, not "
            f"{params.get('output_activation')}.",
            "      Apply that activation to the head output before using the thresholds above.",
        ] if params.get("export_logits") else []),
        "",
        "--- Summary ---",
        f"  Best epoch:      {best}/{total}",
        f"  Final val_loss:  {val_loss_best}",
        f"  Accuracy:        {m['accuracy']:.4f}",
        *(grouped_summary if grouped_summary else [
            f"  Macro F1:        {m['macro_f1']:.4f}",
            f"  Macro Precision: {m['macro_pre']:.4f}",
            f"  Macro Recall:    {m['macro_rec']:.4f}",
        ]),
        f"  Macro AUPRC:     {eval_macro['AUPRC']:.4f}",
        f"  Macro AUROC:     {eval_macro['AUROC']:.4f}",
        f"  Macro F1 (opt):  {eval_macro['F1 Score (opt)']:.4f}",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"  Keras report     → {path}")

    if eval_csv_path is not None:
        write_evaluation_csv(eval_rows, eval_macro, eval_csv_path)


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
        **({"label_groups": meta["label_groups"],
            "group_slices": meta["group_slices"],
            "grouped_output_labels": meta["grouped_output_labels"]}
           if meta.get("label_groups") else {}),
        "classifier":    meta.get("classifier"),
        "output_type":   meta.get("output_type"),
        "output_format": meta.get("output_format"),
        "output_data_types": meta.get("output_data_types", []),
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
            "augmentation_labels": list(aug.augmentation_labels) if aug.augmentation_labels else None,
            "snr_levels":       list(aug.snr_levels),
            "keep_original":    aug.keep_original,
            "augment_test":     aug.augment_test,
        }
    keras_params = meta.get("keras_params")
    if keras_params:
        info["keras_classifier"] = dict(keras_params)
    path.write_text(json.dumps(info, indent=2))
    print(f"  Model info JSON  → {path}")


# Dataset fields whose value is a credential and must never be written to disk.
_SECRET_DATASET_FIELDS = frozenset({"xc_api_key"})

# Config fields recorded even when left at their default value: they describe
# the shape of the dataset rather than an opt-in feature, so a reader needs
# them to understand (or reproduce) the run.  Every other field is recorded
# only when it differs from its default, which keeps the file down to what
# actually shaped the dataset.
_ALWAYS_RECORDED_FIELDS = frozenset({
    "data_dir", "label_mode", "overlap", "speed", "audio_extensions",
    "test_ratio", "random_seed", "embedding_workers",
    "snr_levels", "keep_original", "augment_test",
})


def _field_defaults(cls) -> dict:
    """Default value of every field of a dataclass (factories are evaluated)."""
    defaults = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            defaults[f.name] = f.default_factory()  # type: ignore[misc]
    return defaults


def _config_dict(obj) -> dict:
    """JSON-ready record of a config dataclass, dropping unset/default fields.

    ``None`` values, empty containers and values equal to the field default are
    omitted (see :data:`_ALWAYS_RECORDED_FIELDS` for the exceptions), nested
    config dataclasses are expanded, and credentials are replaced by a
    placeholder so the metadata file can travel with the dataset.
    """
    defaults = _field_defaults(type(obj))
    out: dict = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if value is None:
            continue
        if f.name in _SECRET_DATASET_FIELDS:
            out[f.name] = "***redacted***"
            continue
        if dataclasses.is_dataclass(value):
            nested = _config_dict(value)
            if nested:
                out[f.name] = nested
            continue
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, (list, dict, str)) and len(value) == 0:
            continue
        if f.name not in _ALWAYS_RECORDED_FIELDS and value == defaults.get(f.name):
            continue
        out[f.name] = value
    return out


def _label_counts(train_samples: list, test_samples: list) -> dict:
    """Per-label train/test/total counts, including augmented-copy counts."""
    labels = sorted({s.label for s in train_samples} | {s.label for s in test_samples})
    has_augmentation = any(
        getattr(s, "noise_path", None) is not None
        for s in list(train_samples) + list(test_samples)
    )

    counts: dict = {}
    for label in labels:
        tr = [s for s in train_samples if s.label == label]
        te = [s for s in test_samples if s.label == label]
        entry = {"train": len(tr), "test": len(te), "total": len(tr) + len(te)}
        if has_augmentation:
            entry["train_augmented"] = sum(
                1 for s in tr if getattr(s, "noise_path", None) is not None)
            entry["test_augmented"] = sum(
                1 for s in te if getattr(s, "noise_path", None) is not None)
        counts[label] = entry
    return counts


def write_dataset_metadata(
    path: Path,
    cfg,
    train_samples: list,
    test_samples: list,
    window_seconds: float | None = None,
) -> None:
    """Write a JSON record of how the dataset was built and what it contains.

    Complements the per-sample ``_dataset_list.csv`` with the run-level view:
    the foundation-model window the samples were chunked for, every dataset
    source block with its paths and loading parameters, the run-level split /
    seed / cache settings, and per-label train and test sample counts.

    Credentials (``xc_api_key``) are redacted; unset (``None`` / empty) config
    fields are omitted so the file lists only what actually shaped the dataset.
    """
    from bioaccx.config import RUN_LEVEL_DATASET_FIELDS  # noqa: PLC0415  (avoids import cycle)

    fm = cfg.foundation_model
    base = cfg.dataset
    all_samples = list(train_samples) + list(test_samples)

    run_level = {k: v for k, v in _config_dict(base).items() if k in RUN_LEVEL_DATASET_FIELDS}
    sources = [
        {k: v for k, v in _config_dict(b).items() if k not in RUN_LEVEL_DATASET_FIELDS}
        for b in cfg.dataset_blocks
    ]

    label_counts = _label_counts(train_samples, test_samples)
    n_appended = sum(1 for s in all_samples if getattr(s, "is_appended", False))
    n_augmented = sum(1 for s in all_samples if getattr(s, "noise_path", None) is not None)

    info = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_stem": cfg.model_stem,
        "foundation_model": {
            "name":            fm.name,
            "version":         fm.version,
            "data_type":       fm.data_type,
            "format":          fm.format,
            "sample_rate":     fm.sample_rate,
            "window_samples":  fm.get_window_samples(),
            "window_seconds":  window_seconds if window_seconds is not None
                               else fm.get_window_samples() / fm.sample_rate,
        },
        "sources": sources,
        "run": run_level,
        "totals": {
            "num_labels":  len(label_counts),
            "n_train":     len(train_samples),
            "n_test":      len(test_samples),
            "n_total":     len(all_samples),
            "n_source_files": len({
                str(s.original_path if s.original_path is not None else s.path)
                for s in all_samples
            }),
            **({"n_augmented": n_augmented} if n_augmented else {}),
            **({"n_appended": n_appended} if n_appended else {}),
        },
        "labels": label_counts,
    }
    path.write_text(json.dumps(info, indent=2))
    print(f"  Dataset metadata → {path}")
