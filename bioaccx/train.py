"""Main training pipeline orchestration."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bioaccx.config import BioaccxConfig
from bioaccx.dataset import export_dataset_audio, extract_embeddings, load_samples, split_samples
from bioaccx.embedder import load_embedder
from bioaccx.exporters.onnx_exporter import export_onnx
from bioaccx.exporters.tflite_exporter import export_tflite
from bioaccx.report import (
    write_comparison_report,
    write_dataset_info,
    write_keras_report,
    write_model_info,
    write_sklearn_report,
)
from bioaccx.trainers.keras_trainer import train_keras
from bioaccx.trainers.sklearn_trainer import train_sklearn


def run_dataset_export(cfg: BioaccxConfig) -> dict[str, str]:
    """Load, split, and export the dataset as chunked WAV files.

    Does not load or run the foundation model.
    Returns a dict of output file paths.
    """
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = cfg.foundation_model
    ds = cfg.dataset
    out = cfg.output

    stem = f"{out.model_name}_v{out.model_version}"
    window_samples = fm.get_window_samples()
    window_sec = window_samples / fm.sample_rate

    data_dir_display = ds.data_dir if isinstance(ds.data_dir, str) else ", ".join(ds.data_dir)
    print(f"\n{'='*62}")
    print(f"bioaccx — dataset export")
    print(f"Data dir: {data_dir_display}")
    print(f"Output:   {out_dir}")
    print(f"{'='*62}")

    print("\n[1/2] Loading dataset…")
    samples = load_samples(ds, window_seconds=window_sec, sample_rate=fm.sample_rate)
    print(f"  {len(samples)} samples found across {len(set(s.label for s in samples))} classes")
    if ds.label_mode in ("file_per_label", "table"):
        print(f"  window={window_sec}s  overlap={ds.overlap}")

    train_samples, test_samples = split_samples(samples, ds.test_ratio, ds.random_seed)
    print(f"  Train: {len(train_samples)}  |  Test: {len(test_samples)}")

    dataset_info_path = out_dir / f"{stem}_dataset_info.csv"
    write_dataset_info(
        dataset_info_path, train_samples, test_samples, window_seconds=window_sec,
        filter=ds.filter, filter_freq=ds.filter_freq,
        filter_order=ds.filter_order, speed=ds.speed,
    )
    outputs: dict[str, str] = {"dataset_info": str(dataset_info_path)}

    print("\n[2/2] Exporting chunked audio dataset…")
    if ds.label_mode == "subfolders":
        print("  Skipping: label_mode=subfolders already has the expected structure.")
    else:
        export_dataset_audio(
            train_samples, test_samples,
            out_dir=out_dir / "dataset",
            sample_rate=fm.sample_rate,
            window_samples=window_samples,
        )
        outputs["dataset"] = str(out_dir / "dataset")

    print(f"\nDone. Outputs saved to: {out_dir}")
    return outputs


def run(cfg: BioaccxConfig) -> dict[str, str]:
    """Execute the full pipeline; return a dict of output file paths."""
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = cfg.foundation_model
    ds = cfg.dataset
    tr = cfg.training
    out = cfg.output

    stem = f"{out.model_name}_v{out.model_version}"

    print(f"\n{'='*62}")
    print(f"bioaccx — {fm.name} v{fm.version}  |  embed_dim={fm.embedding_size}")
    data_dir_display = ds.data_dir if isinstance(ds.data_dir, str) else ", ".join(ds.data_dir)
    print(f"Data dir: {data_dir_display}")
    print(f"Output:   {out_dir}")
    print(f"{'='*62}")

    # ------------------------------------------------------------------
    # 1. Load foundation model
    # ------------------------------------------------------------------
    print("\n[1/5] Validating foundation model…")
    load_embedder(fm)  # fail fast if model path / format is wrong

    # ------------------------------------------------------------------
    # 2. Load and split dataset
    # ------------------------------------------------------------------
    print("\n[2/5] Loading dataset…")
    window_sec = fm.get_window_samples() / fm.sample_rate
    samples = load_samples(ds, window_seconds=window_sec, sample_rate=fm.sample_rate)
    print(f"  {len(samples)} samples found across {len(set(s.label for s in samples))} classes")
    if ds.label_mode in ("file_per_label", "table"):
        print(f"  window={window_sec}s  overlap={ds.overlap}")

    train_samples, test_samples = split_samples(samples, ds.test_ratio, ds.random_seed)
    print(f"  Train: {len(train_samples)}  |  Test: {len(test_samples)}")

    # ------------------------------------------------------------------
    # 2b. Export chunked audio (optional)
    # ------------------------------------------------------------------
    if out.export_dataset and ds.label_mode == "subfolders":
        print("\n[2b] Skipping dataset export: label_mode=subfolders already has the expected structure.")
    elif out.export_dataset:
        print("\n[2b] Exporting chunked audio dataset…")
        export_dataset_audio(
            train_samples, test_samples,
            out_dir=out_dir / "dataset",
            sample_rate=fm.sample_rate,
            window_samples=fm.get_window_samples(),
        )

    # ------------------------------------------------------------------
    # 2c. Save dataset info CSV
    # ------------------------------------------------------------------
    dataset_info_path = out_dir / f"{stem}_dataset_info.csv"
    write_dataset_info(
        dataset_info_path, train_samples, test_samples, window_seconds=window_sec,
        filter=ds.filter, filter_freq=ds.filter_freq,
        filter_order=ds.filter_order, speed=ds.speed,
    )

    # ------------------------------------------------------------------
    # 3. Extract embeddings
    # ------------------------------------------------------------------
    cache_dir  = Path(ds.embeddings_cache_path) if ds.embeddings_cache_path else None
    export_dir = (
        Path(out.embeddings_path) if out.embeddings_path
        else out_dir / "embeddings"
    ) if out.export_embeddings else None

    if cache_dir:
        print(f"  Embeddings cache : {cache_dir}")
    if export_dir:
        print(f"  Embeddings export: {export_dir}")

    print("\n[3/5] Extracting embeddings…")
    print("  Train set:")
    X_train, y_train, label_names = extract_embeddings(
        train_samples, fm, fm.embedding_size,
        n_workers=ds.embedding_workers,
        cache_dir=cache_dir,
        export_dir=export_dir,
    )
    print("  Test set:")
    X_test,  y_test,  _           = extract_embeddings(
        test_samples, fm, fm.embedding_size,
        n_workers=ds.embedding_workers,
        cache_dir=cache_dir,
        export_dir=export_dir,
    )

    embed_dim = fm.embedding_size
    n_train, n_test = len(X_train), len(X_test)
    report_meta = dict(
        data_dir=data_dir_display,
        foundation_name=fm.name,
        foundation_version=fm.version,
        foundation_format=fm.format,
        embed_dim=embed_dim,
        test_ratio=ds.test_ratio,
        n_train=n_train,
        n_test=n_test,
        classifier=tr.classifier,
        output_type=out.output_type,
        output_format=out.output_format,
    )

    # Foundation model path as ONNX (for full-model merge)
    foundation_local_path = (
        Path(fm.path)
        if fm.source == "local" and fm.path is not None
        else _cached_hf_path(fm)
    )

    # ------------------------------------------------------------------
    # 4. Train classifiers
    # ------------------------------------------------------------------
    print("\n[4/5] Training classifier(s)…")
    keras_model = None
    sklearn_pipe = None

    if tr.classifier in ("keras", "both"):
        keras_model = train_keras(X_train, y_train, X_test, y_test, label_names, tr.keras)

    if tr.classifier in ("sklearn", "both"):
        sklearn_pipe = train_sklearn(X_train, y_train, X_test, y_test, label_names, tr.sklearn)

    # ------------------------------------------------------------------
    # 5. Export models and write reports
    # ------------------------------------------------------------------
    print("\n[5/5] Exporting models and writing reports…")
    outputs: dict[str, str] = {"dataset_info": str(dataset_info_path)}
    onnx_head_paths: dict[str, Path] = {}

    # Compute output label set (may exclude background/noise labels)
    excluded = set(out.exclude_labels)
    unknown_excluded = excluded - set(label_names)
    if unknown_excluded:
        print(f"  Warning: exclude_labels not found in training data: {sorted(unknown_excluded)}")
    keep_indices = [i for i, n in enumerate(label_names) if n not in excluded]
    output_label_names = [label_names[i] for i in keep_indices]
    if excluded & set(label_names):
        print(f"  Excluding from output: {sorted(excluded & set(label_names))}")
        print(f"  Output classes ({len(output_label_names)}): {output_label_names}")
    keep_indices_arg = keep_indices if len(keep_indices) < len(label_names) else None

    do_onnx   = out.output_format in ("onnx", "both")
    do_tflite = out.output_format in ("tflite", "both")
    do_head   = out.output_type in ("head", "both")
    do_full   = out.output_type in ("full", "both")

    # Full-model export requires the backbone format to match the output format:
    #   onnx backbone   → onnx_full, tflite_head, onnx_head
    #   tflite/protobuf → tflite_full, tflite_head, onnx_head
    do_onnx_full   = do_full and fm.format == "onnx"
    do_tflite_full = do_full and fm.format in ("tflite", "protobuf")
    if do_full and not do_onnx_full and not do_tflite_full:
        print(f"  Warning: full-model export not supported for backbone format '{fm.format}'")

    # ---- Keras exports ----
    if keras_model is not None:
        if do_onnx:
            for otype in _export_types(do_head, do_onnx_full):
                fpath = out_dir / f"{stem}_keras_{otype}.onnx"
                exported = export_onnx(
                    keras_model, "keras", embed_dim, fpath,
                    foundation_onnx_path=foundation_local_path,
                    foundation_input_name=fm.input_name,
                    output_type=otype,
                    keep_indices=keep_indices_arg,
                )
                outputs[f"keras_onnx_{otype}"] = str(exported)
                if otype == "head":
                    onnx_head_paths["keras"] = exported

        if do_tflite:
            for otype in _export_types(do_head, do_tflite_full):
                fpath = out_dir / f"{stem}_keras_{otype}.tflite"
                exported = export_tflite(
                    keras_model, "keras", embed_dim, fpath,
                    foundation_savedmodel_path=foundation_local_path,
                    foundation_input_name=fm.input_name,
                    output_type=otype,
                    keep_indices=keep_indices_arg,
                )
                if exported:
                    outputs[f"keras_tflite_{otype}"] = str(exported)

        keras_report_path = out_dir / f"{stem}_keras_report.txt"
        write_keras_report(keras_model, X_test, y_test, label_names, keras_report_path, **report_meta)
        outputs["keras_report"] = str(keras_report_path)

    # ---- Sklearn exports ----
    if sklearn_pipe is not None:
        if do_onnx:
            for otype in _export_types(do_head, do_onnx_full):
                fpath = out_dir / f"{stem}_sklearn_{otype}.onnx"
                exported = export_onnx(
                    sklearn_pipe, "sklearn", embed_dim, fpath,
                    foundation_onnx_path=foundation_local_path,
                    foundation_input_name=fm.input_name,
                    output_type=otype,
                    keep_indices=keep_indices_arg,
                )
                outputs[f"sklearn_onnx_{otype}"] = str(exported)
                if otype == "head":
                    onnx_head_paths["sklearn"] = exported

        sklearn_report_path = out_dir / f"{stem}_sklearn_report.txt"
        write_sklearn_report(sklearn_pipe, X_test, y_test, label_names, sklearn_report_path, **report_meta)
        outputs["sklearn_report"] = str(sklearn_report_path)

    # ---- Labels file (output labels only — excluded labels omitted) ----
    labels_path = out_dir / f"{stem}_labels.txt"
    labels_path.write_text("\n".join(output_label_names) + "\n")
    outputs["labels"] = str(labels_path)
    print(f"  Labels           → {labels_path}")

    # ---- Comparison report (when both classifiers trained) ----
    # The exported ONNX heads are already filtered to output_label_names.
    # y_test must be remapped to the reduced label space; samples whose true
    # label was excluded are dropped from evaluation entirely.
    if len(onnx_head_paths) > 1:
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        mask = np.array([int(y) in old_to_new for y in y_test])
        X_test_cmp = X_test[mask]
        y_test_cmp = np.array([old_to_new[int(y)] for y in y_test[mask]])
        cmp_path = out_dir / f"{stem}_comparison_report.txt"
        write_comparison_report(
            onnx_head_paths, X_test_cmp, y_test_cmp, output_label_names, cmp_path, **report_meta
        )
        outputs["comparison_report"] = str(cmp_path)

    # ---- Model info JSON ----
    report_meta["excluded_labels"] = sorted(excluded & set(label_names))
    report_meta["output_labels"] = output_label_names
    info_path = out_dir / f"{stem}_model_info.json"
    write_model_info(info_path, output_label_names, outputs, **report_meta)
    outputs["model_info"] = str(info_path)

    print(f"\nDone. All outputs saved to: {out_dir}")
    return outputs


def _export_types(do_head: bool, do_full: bool) -> list[str]:
    types = []
    if do_head:
        types.append("head")
    if do_full:
        types.append("full")
    return types


def _cached_hf_path(fm) -> Path | None:
    """Return the locally cached HF model path if available, else None."""
    if fm.source != "huggingface" or fm.hf_repo is None:
        return None
    try:
        from huggingface_hub import hf_hub_download
        from bioaccx.embedder import _default_hf_filename
        filename = fm.hf_filename or _default_hf_filename(fm)
        return Path(hf_hub_download(repo_id=fm.hf_repo, filename=filename, revision=fm.hf_revision))
    except Exception:
        return None
