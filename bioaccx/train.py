"""Main training pipeline orchestration."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from bioaccx.config import BioaccxConfig
from bioaccx.registry import lookup_foundation_model_id
from bioaccx.dataset import apply_augmentation, apply_random_shifts, export_dataset_audio, extract_embeddings, load_samples, sample_labels, split_samples
from bioaccx.embedder import _resolve_model_path, load_embedder
from bioaccx.exporters.onnx_exporter import export_onnx, export_tflite_head_to_onnx
from bioaccx.exporters.tflite_exporter import export_tflite
from bioaccx.report import (
    write_comparison_report,
    write_dataset_list,
    write_dataset_metadata,
    write_keras_report,
    write_model_metadata,
    write_sklearn_report,
)
from bioaccx.trainers.grouped import build_group_space
from bioaccx.audio_mixup import build_audio_mixes, check_audio_mixup_head, training_composition
from bioaccx.trainers.keras_trainer import (
    append_softmax,
    strip_output_activation,
    train_keras,
)
from bioaccx.trainers.sklearn_trainer import train_sklearn


def _augment_and_shift(ds, train_samples, test_samples, window_sec, sample_rate,
                       train_noise_pool=None, test_noise_pool=None):
    """Apply a block's augmentation and random-shift settings to its samples.

    Both operate on already-split samples: augmentation always covers the train
    set (and the test set only when augment_test is set); random shifts cover
    both. Returns the (possibly expanded) (train, test) lists.

    ``train_noise_pool`` / ``test_noise_pool`` are the sample pools from which
    ``augmentation_labels`` noise is drawn — passed when those labels may live in
    a different source block. They default to the block's own samples.
    """
    if ds.augmentation is not None:
        train_samples = apply_augmentation(
            train_samples, ds.augmentation,
            window_seconds=window_sec, sample_rate=sample_rate, random_seed=ds.random_seed,
            label_noise_samples=train_noise_pool,
        )
        if ds.augmentation.augment_test:
            test_samples = apply_augmentation(
                test_samples, ds.augmentation,
                window_seconds=window_sec, sample_rate=sample_rate, random_seed=ds.random_seed,
                label_noise_samples=test_noise_pool,
            )
    if ds.random_sample_shift:
        train_samples = apply_random_shifts(train_samples, window_sec, sample_rate, ds.random_seed)
        test_samples = apply_random_shifts(test_samples, window_sec, sample_rate, ds.random_seed)
    return train_samples, test_samples


def _data_dir_display(ds) -> str:
    return ds.data_dir if isinstance(ds.data_dir, str) else ", ".join(ds.data_dir)


def _split_or_all(samples: list, test_ratio: float, random_seed: int, no_split: bool):
    """Split samples into (train, test), or return them all as one unsplit set.

    With ``no_split`` no split is computed at all — any predefined ``train`` /
    ``test`` assignment the samples carry is ignored too. The samples are
    returned in the train slot so that the rest of the pipeline (augmentation,
    export) treats them as a single group; callers must keep them unsplit.
    """
    if no_split:
        print(f"  No train/test split requested: {len(samples)} samples kept as one set")
        return list(samples), []
    return split_samples(samples, test_ratio, random_seed)


def load_and_prepare_blocks(cfg: BioaccxConfig, window_sec: float, sample_rate: int,
                            no_split: bool = False):
    """Load, split and preprocess every dataset source, returning merged samples.

    For the common single-source config this is exactly the legacy path: load →
    split → augment/shift. When ``dataset.sources`` defines multiple blocks each
    one is loaded, split (with the run-level test_ratio / random_seed) and
    augmented/shifted independently, then the per-source train/test sets are
    concatenated. Splitting per source keeps each source's class proportions in
    both subsets and lets augmentation apply to only the sources that request it.

    Run-level ``append_dataset_path`` is honoured once: each block de-duplicates
    its new samples against the existing exported dataset (via
    ``include_existing=False``), and the existing samples are loaded and merged
    in a single time here, keeping their predefined train/test split.

    With ``no_split`` (``--dataset --no-split`` only) the split step is skipped
    entirely: every sample is returned in the train slot and the test slot is
    empty, so augmentation still covers the whole set.

    Returns (train_samples, test_samples).
    """
    blocks = cfg.dataset_blocks

    # Single-source: preserve the original behaviour and console output exactly.
    if len(blocks) == 1:
        ds = blocks[0]
        samples = load_samples(ds, window_seconds=window_sec, sample_rate=sample_rate)
        print(f"  {len(samples)} samples found across {len(set(s.label for s in samples))} classes")
        if ds.label_mode in ("file_per_label", "table"):
            print(f"  window={window_sec}s  overlap={ds.overlap}")
        train_samples, test_samples = _split_or_all(
            samples, ds.test_ratio, ds.random_seed, no_split)
        if not no_split:
            print(f"  Train: {len(train_samples)}  |  Test: {len(test_samples)}")
        return _augment_and_shift(ds, train_samples, test_samples, window_sec, sample_rate)

    # Multi-source: load + split each block, then augment/merge. Splitting all
    # blocks first lets augmentation_labels noise be resolved across the whole
    # dataset (a noise label may live in a different source than the one that
    # requests it), while still drawing train noise only from the train split.
    base = cfg.dataset
    all_train: list = []
    all_test: list = []
    print(f"  {len(blocks)} dataset sources:")
    loaded: list = []  # (index, block, tr, te)
    for i, block in enumerate(blocks, 1):
        print(f"\n  Source [{i}/{len(blocks)}]: {_data_dir_display(block)}  "
              f"(label_mode={block.label_mode}"
              f"{', augmented' if block.augmentation is not None else ''})")
        samples = load_samples(
            block, window_seconds=window_sec, sample_rate=sample_rate,
            include_existing=False,
        )
        print(f"    {len(samples)} samples across {len(set(s.label for s in samples))} classes")
        tr, te = _split_or_all(samples, base.test_ratio, base.random_seed, no_split)
        loaded.append((i, block, tr, te))

    # Dataset-wide noise pools used to resolve augmentation_labels across blocks.
    global_train = [s for _, _, tr, _ in loaded for s in tr]
    global_test = [s for _, _, _, te in loaded for s in te]

    for i, block, tr, te in loaded:
        tr, te = _augment_and_shift(
            block, tr, te, window_sec, sample_rate,
            train_noise_pool=global_train, test_noise_pool=global_test,
        )
        if no_split:
            print(f"    Source [{i}/{len(blocks)}] — {len(tr)} samples")
        else:
            print(f"    Source [{i}/{len(blocks)}] — Train: {len(tr)}  |  Test: {len(te)}")
        all_train += tr
        all_test += te

    # Run-level append: load the existing exported dataset once and merge it in,
    # keeping its predefined split (split_samples passes predefined splits through).
    if base.append_dataset_path:
        from bioaccx.dataset import _load_subfolders
        exts = frozenset(f".{e.lstrip('.')}" for e in base.audio_extensions)
        existing = _load_subfolders(Path(base.append_dataset_path), exts)
        for s in existing:
            s.is_appended = True
        ex_train, ex_test = _split_or_all(existing, base.test_ratio, base.random_seed, no_split)
        if no_split:
            print(f"\n  Appended existing dataset: {len(ex_train)} samples")
        else:
            print(f"\n  Appended existing dataset: train={len(ex_train)}  test={len(ex_test)}")
        all_train = ex_train + all_train
        all_test = ex_test + all_test

    if no_split:
        print(f"\n  Combined — {len(all_train)} samples")
    else:
        print(f"\n  Combined — Train: {len(all_train)}  |  Test: {len(all_test)}")
    return all_train, all_test


def _add_audio_mixes(cfg: BioaccxConfig, train_samples: list, test_samples: list):
    """Draw the audio mixes a config asks for; returns (train, test_mixes, mix_cfg, seed).

    Train mixes come from the train windows only and are appended to them;
    optional test mixes come from the test windows only and are returned apart,
    since they are never pooled with the real test set. Without an enabled
    ``training.audio_mixup`` the samples come back unchanged.
    """
    tr = cfg.training
    mix_cfg = tr.audio_mixup if tr.audio_mixup is not None and tr.audio_mixup.enabled else None
    if mix_cfg is None:
        return train_samples, [], None, None
    check_audio_mixup_head(tr.classifier, tr.keras.output_activation, tr.keras.label_groups)
    seed = mix_cfg.seed if mix_cfg.seed is not None else cfg.dataset.random_seed
    train = train_samples + build_audio_mixes(train_samples, mix_cfg, seed, split="train")
    test_mixes = (build_audio_mixes(test_samples, mix_cfg, seed + 1,
                                    ratio=mix_cfg.test_mix_ratio, split="test")
                  if mix_cfg.test_mix_ratio > 0 and test_samples else [])
    return train, test_mixes, mix_cfg, seed


def _export_mixes_only(train_samples: list, test_mix_samples: list, mix_cfg, out_dir: Path,
                       sample_rate: int, window_samples: int, no_split: bool = False,
                       step: str = "") -> bool:
    """Write just the audio mixes to ``dataset/mixes/`` (``output.export_mixes``).

    For a dataset whose real windows are exported already: the mixes are the
    only new audio, and listening to them is how to check the pairing and
    levels. Returns whether anything was written.
    """
    if mix_cfg is None:
        print(f"\n{step}export_mixes is set but training.audio_mixup is not enabled — "
              "no mixes to export.")
        return False
    print(f"\n{step}Exporting audio mixes only…")
    export_dataset_audio(
        train_samples, test_mix_samples, out_dir=out_dir / "dataset",
        sample_rate=sample_rate, window_samples=window_samples,
        no_split=no_split, mixes_only=True,
    )
    return True


def _any_block_uses_ssh(cfg: BioaccxConfig) -> bool:
    """True if any dataset source is accessed over SSH (export is unsupported then)."""
    return any(b.ssh_host for b in cfg.dataset_blocks)


def run_dataset_export(cfg: BioaccxConfig, no_split: bool = False) -> dict[str, str]:
    """Load, split, and export the dataset as chunked WAV files.

    With ``no_split`` (the CLI's ``--dataset --no-split``) no train/test split is
    computed and any predefined one is ignored: the chunks are written straight
    into ``dataset/<label>/``, the sample list's split column is left empty, and
    the metadata records ``"split": "none"``. Useful to review the chunks, feed
    them to another tool, or split them by hand — the emitted CSV can be filled
    in and fed back as a ``label_mode: table`` source.

    Does not load or run the foundation model.
    Returns a dict of output file paths.
    """
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = cfg.foundation_model
    ds = cfg.dataset

    stem = cfg.model_stem
    window_samples = fm.get_window_samples()
    window_sec = window_samples / fm.sample_rate

    data_dir_display = ds.data_dir if isinstance(ds.data_dir, str) else ", ".join(ds.data_dir)
    print(f"\n{'='*62}")
    print(f"bioaccx — dataset export{' (no train/test split)' if no_split else ''}")
    print(f"Data dir: {data_dir_display}")
    print(f"Output:   {out_dir}")
    print(f"{'='*62}")

    print("\n[1/2] Loading dataset…")
    train_samples, test_samples = load_and_prepare_blocks(
        cfg, window_sec, fm.sample_rate, no_split=no_split)
    # The same mixes a training run with this config and seed would draw, so
    # they can be listened to before training.
    train_samples, test_mix_samples, mix_cfg, _ = _add_audio_mixes(
        cfg, train_samples, test_samples)
    if mix_cfg is not None and no_split:
        print("  Note: without a split the mixes are drawn from every window, so they "
              "differ from the ones a training run draws from its train split.")

    dataset_list_path = out_dir / f"{stem}_dataset_list.csv"
    write_dataset_list(
        dataset_list_path, train_samples, test_samples, window_seconds=window_sec,
        filter=ds.filter, filter_freq=ds.filter_freq,
        filter_order=ds.filter_order, speed=ds.speed, no_split=no_split,
        test_mix_samples=test_mix_samples,
    )
    dataset_meta_path = out_dir / f"{stem}_dataset_metadata.json"
    write_dataset_metadata(
        dataset_meta_path, cfg, train_samples, test_samples, window_seconds=window_sec,
        no_split=no_split,
        mixup_composition=training_composition(
            train_samples, test_samples,
            sorted({l for s in train_samples for l in sample_labels(s)}),
        ) if mix_cfg is not None else None,
    )
    outputs: dict[str, str] = {
        "dataset_list": str(dataset_list_path),
        "dataset_metadata": str(dataset_meta_path),
    }

    if cfg.output.export_mixes:
        if _export_mixes_only(train_samples, test_mix_samples, mix_cfg, out_dir,
                              fm.sample_rate, window_samples, no_split=no_split,
                              step="[2/2] "):
            outputs["mixes"] = str(out_dir / "dataset" / "mixes")
        print(f"\nDone. Outputs saved to: {out_dir}")
        return outputs

    print("\n[2/2] Exporting chunked audio dataset…")
    single = len(cfg.dataset_blocks) == 1
    # A pre-split source (train/<label>/ … test/<label>/) is *not* already in the
    # expected structure when the export must drop the split, so it is written out.
    pre_split = any(s.split in ("train", "test") for s in train_samples + test_samples)
    if _any_block_uses_ssh(cfg):
        print("  Skipping: dataset export is not supported for SSH data dirs.")
    elif (
        single
        and ds.label_mode == "subfolders"
        and ds.augmentation is None
        and not ds.append_dataset_path
        and not ds.random_sample_shift
        and not (no_split and pre_split)
        and mix_cfg is None
    ):
        print("  Skipping: label_mode=subfolders with no augmentation already has the expected structure.")
    else:
        export_dataset_audio(
            train_samples, test_samples + test_mix_samples,
            out_dir=out_dir / "dataset",
            sample_rate=fm.sample_rate,
            window_samples=window_samples,
            no_split=no_split,
        )
        outputs["dataset"] = str(out_dir / "dataset")

    print(f"\nDone. Outputs saved to: {out_dir}")
    return outputs


def _resolve_embedding_paths(cfg: BioaccxConfig, out_dir: Path) -> tuple[Path | None, Path | None]:
    """Return (export_dir, export_sqlite) for newly computed embeddings.

    Exactly one of the two is non-None depending on output.embeddings_format.
    Used by both the full pipeline (when export_embeddings is set) and the
    --embeddings mode (which always exports).
    """
    fm = cfg.foundation_model
    out = cfg.output
    fm_id = lookup_foundation_model_id(fm.name, fm.version, fm.data_type, fm.format)
    if out.embeddings_format == "sqlite":
        base = Path(out.embeddings_path) if out.embeddings_path else out_dir
        return None, base / f"{fm_id}_embeddings.db"
    export_dir = (
        Path(f"{out.embeddings_path}/embeddings/{cfg.model_stem}") if out.embeddings_path
        else out_dir / "embeddings"
    )
    return export_dir, None


def _embeddings_store_exists(export_dir: Path | None, export_sqlite: Path | None) -> bool:
    """Return True if a previously computed embedding store is already on disk.

    sqlite → the .db file exists; npy → the directory exists and holds at least
    one .npy file (an empty leftover directory is not treated as a store).
    """
    if export_sqlite is not None:
        return export_sqlite.exists()
    if export_dir is not None:
        return export_dir.is_dir() and any(export_dir.glob("*.npy"))
    return False


def _replot_umap_from_cache(
    cfg: BioaccxConfig, cache_path: Path, out_dir: Path, stem: str,
) -> dict[str, str]:
    """Redraw the UMAP/cluster plots from a cached UMAP CSV — no computation.

    Used by :func:`run_embeddings` when ``umap.cache_csv`` points at an existing
    CSV: dataset loading, embedding extraction, KMeans and the UMAP fit are all
    skipped. Coordinates, labels and (when present) cluster ids are reloaded
    from the CSV; NMI is recomputed cheaply from the cached labels/clusters.
    """
    from bioaccx.umap import (
        nmi_from_assignments, read_umap_csv, write_cluster_plot, write_umap_plot,
    )

    fm = cfg.foundation_model
    print(f"\nUMAP CSV cache found → skipping computation, only updating plots:\n  {cache_path}")
    coords, point_labels, splits, cluster_ids = read_umap_csv(cache_path)
    # Stable, first-appearance ordering for deterministic plot colours.
    label_names = list(dict.fromkeys(point_labels))
    n_samples = len(coords)
    outputs: dict[str, str] = {"umap_data": str(cache_path)}

    umap_plot_path = out_dir / f"{stem}_umap.png"
    write_umap_plot(
        umap_plot_path, coords, point_labels, label_names,
        title=f"UMAP — {fm.name} v{fm.version} ({n_samples} samples, {len(label_names)} classes)",
    )
    if umap_plot_path.exists():
        outputs["umap_plot"] = str(umap_plot_path)

    if cluster_ids is not None:
        nmi = nmi_from_assignments(point_labels, cluster_ids)
        n_clusters = len(np.unique(cluster_ids))
        print(f"  NMI (cached clusters vs labels): {nmi:.4f}")
        cluster_plot_path = out_dir / f"{stem}_clusters.png"
        metrics = {
            "NMI": f"{nmi:.4f}",
            "clusters (k)": n_clusters,
            "classes": len(label_names),
            "samples": n_samples,
        }
        write_cluster_plot(
            cluster_plot_path, coords, cluster_ids,
            title=f"KMeans clusters (k={n_clusters}) — {fm.name} v{fm.version}",
            metrics=metrics,
        )
        if cluster_plot_path.exists():
            outputs["cluster_plot"] = str(cluster_plot_path)
    else:
        print("  Cluster plot skipped (cached CSV has no 'cluster' column).")

    print(f"\nDone. Plots updated from cache: {out_dir}")
    return outputs


def run_embeddings(cfg: BioaccxConfig) -> dict[str, str]:
    """Compute the embedding database (and optionally a UMAP projection) — no training.

    Pipeline steps:
      1. Validate the foundation model path / download.
      2. Load and split the dataset (with augmentation / random shifts), then
         write the dataset_list CSV.
      3. Extract embeddings for train and test sets, always exporting them to
         a SQLite database (or .npy directory, per output.embeddings_format).
      4. (Only when umap.enabled) Fit UMAP over the combined embedding set and
         write the UMAP data CSV and a scatter-plot PNG coloured by label. This
         step requires the optional ``[umap]`` extra.

    Overwrite behaviour: if an embedding store already exists at the resolved
    path it is reused as-is and no embeddings are recomputed — the run only
    (re)builds the UMAP outputs over the existing store. Set
    ``output.embeddings_overwrite: true`` to delete the old store and recompute
    from scratch.

    No classifier is trained and no model is exported. Returns a dict of
    output file paths.
    """
    import shutil

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = cfg.foundation_model
    ds = cfg.dataset
    stem = cfg.model_stem
    do_umap = cfg.umap.enabled

    # Fast path: a cached UMAP CSV was provided — skip all computation and only
    # redraw the plots from the cached coordinates.
    if do_umap and cfg.umap.cache_csv:
        cache_path = Path(cfg.umap.cache_csv)
        if cache_path.exists():
            return _replot_umap_from_cache(cfg, cache_path, out_dir, stem)
        print(f"\n[warning] umap.cache_csv set but not found → computing normally:\n  {cache_path}")

    # Resolve the embedding store path up front so we can decide whether to
    # reuse an existing one (no recompute) or overwrite it.
    export_dir, export_sqlite = _resolve_embedding_paths(cfg, out_dir)
    store_exists = _embeddings_store_exists(export_dir, export_sqlite)
    reuse = store_exists and not cfg.output.embeddings_overwrite
    store_display = str(export_sqlite) if export_sqlite else str(export_dir)

    data_dir_display = ds.data_dir if isinstance(ds.data_dir, str) else ", ".join(ds.data_dir)
    print(f"\n{'='*62}")
    title = "embeddings + UMAP" if do_umap else "embeddings"
    print(f"bioaccx — {title}  |  {fm.name} v{fm.version}  embed_dim={fm.embedding_size}")
    print(f"Data dir: {data_dir_display}")
    print(f"Output:   {out_dir}")
    print(f"{'='*62}")

    # Reuse path: an embedding store already exists and overwrite is off.
    if reuse:
        print(f"\nExisting embedding store found → reusing (no recompute):\n  {store_display}")
        print("  Set output.embeddings_overwrite: true to recompute and overwrite it.")
        if not do_umap:
            # Nothing left to do — embeddings already exist, UMAP not requested.
            print("\n  UMAP disabled (set umap.enabled: true and install the [umap] "
                  "extra to build UMAP outputs over the existing store).")
            print(f"\nDone. Embeddings: {store_display}")
            return {"embeddings": store_display}
        # Fall through: load the dataset (to recover labels/splits), read the
        # cached embeddings back from the store, then build the UMAP outputs.

    # Overwrite path: drop the stale store so recomputation starts clean.
    if store_exists and not reuse:
        print(f"\nOverwriting existing embedding store:\n  {store_display}")
        if export_sqlite is not None:
            export_sqlite.unlink(missing_ok=True)
        elif export_dir is not None and export_dir.is_dir():
            shutil.rmtree(export_dir)

    # Plan the step labels (foundation-model validation + extraction are skipped
    # when reusing an existing store).
    step_labels = []
    if not reuse:
        step_labels.append("Validating foundation model")
    step_labels.append("Loading dataset")
    step_labels.append("Reading cached embeddings" if reuse else "Extracting embeddings")
    if do_umap:
        step_labels.append("Computing UMAP projection")
    n_steps = len(step_labels)
    step_no = 0

    def _next_step(name: str) -> None:
        nonlocal step_no
        step_no += 1
        print(f"\n[{step_no}/{n_steps}] {name}…")

    # 1. Load foundation model (fail fast) — skipped when reusing the store.
    if not reuse:
        _next_step("Validating foundation model")
        load_embedder(fm)

    # 2. Load and split dataset
    _next_step("Loading dataset")
    window_sec = fm.get_window_samples() / fm.sample_rate
    train_samples, test_samples = load_and_prepare_blocks(cfg, window_sec, fm.sample_rate)

    dataset_list_path = out_dir / f"{stem}_dataset_list.csv"
    write_dataset_list(
        dataset_list_path, train_samples, test_samples, window_seconds=window_sec,
        filter=ds.filter, filter_freq=ds.filter_freq,
        filter_order=ds.filter_order, speed=ds.speed,
    )
    dataset_meta_path = out_dir / f"{stem}_dataset_metadata.json"
    write_dataset_metadata(
        dataset_meta_path, cfg, train_samples, test_samples, window_seconds=window_sec,
    )
    outputs: dict[str, str] = {
        "dataset_list": str(dataset_list_path),
        "dataset_metadata": str(dataset_meta_path),
    }

    # 3. Embeddings: either read from the existing store (reuse) or compute and
    #    export a fresh one. When reusing, the existing store is passed as the
    #    *cache* (read-only) and nothing is exported, so the store on disk is
    #    left untouched.
    if reuse:
        cache_dir, cache_sqlite = export_dir, export_sqlite
        export_dir, export_sqlite = None, None
    else:
        cache_dir = cache_sqlite = None
        export_dir, export_sqlite = _resolve_embedding_paths(cfg, out_dir)
        if export_sqlite:
            print(f"  Embeddings export: {export_sqlite} (sqlite)")
        else:
            print(f"  Embeddings export: {export_dir}")

    ssh_config = None
    if ds.ssh_host:
        ssh_config = {
            "host": ds.ssh_host,
            "user": ds.ssh_user or "",
            "port": ds.ssh_port,
            "key_path": ds.ssh_key_path,
        }

    _next_step("Reading cached embeddings" if reuse else "Extracting embeddings")
    print("  Train set:")
    # Keys of the samples that actually produced an embedding, in row order —
    # the join back to the dataset list, which failed samples would otherwise
    # knock out of alignment.
    sample_keys: list[str] = []
    # One label list for both splits, so a label index means the same class in
    # train and test (a class missing from one split must not shift the rest).
    label_names = sorted({l for s in train_samples + test_samples for l in sample_labels(s)})
    X_train, y_train, _ = extract_embeddings(
        train_samples, fm, fm.embedding_size,
        n_workers=ds.embedding_workers,
        cache_dir=cache_dir, cache_sqlite=cache_sqlite,
        export_dir=export_dir, export_sqlite=export_sqlite,
        ssh_config=ssh_config, kept_keys=sample_keys, label_names=label_names,
    )
    splits = ["train"] * len(X_train)
    X_all, y_all = X_train, y_train
    if test_samples:
        print("  Test set:")
        X_test, y_test, _ = extract_embeddings(
            test_samples, fm, fm.embedding_size,
            n_workers=ds.embedding_workers,
            cache_dir=cache_dir, cache_sqlite=cache_sqlite,
            export_dir=export_dir, export_sqlite=export_sqlite,
            ssh_config=ssh_config, kept_keys=sample_keys, label_names=label_names,
        )
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        splits += ["test"] * len(X_test)

    outputs["embeddings"] = store_display

    # 3b. Normalized mutual information: cluster the full embedding set with
    #     KMeans (k = n_classes) and score agreement with the ground-truth
    #     labels — an unsupervised measure of class separability. Uses only
    #     scikit-learn (core dep), so it runs regardless of umap.enabled.
    from bioaccx.umap import compute_nmi

    nmi, cluster_ids = compute_nmi(
        X_all, y_all, n_clusters=len(label_names), random_seed=ds.random_seed,
    )
    if nmi is not None:
        print(f"  NMI (KMeans k={len(label_names)} vs labels): {nmi:.4f}")
    else:
        print(f"  NMI skipped (need >=2 classes and >=n_classes samples; "
              f"got {len(label_names)} classes, {len(X_all)} samples)")

    # 4. UMAP projection (opt-in; requires the [umap] extra)
    if do_umap:
        _next_step("Computing UMAP projection")
        from bioaccx.umap import (
            compute_umap, write_cluster_plot, write_umap_csv, write_umap_plot,
        )

        um = cfg.umap
        seed = um.random_seed if um.random_seed is not None else ds.random_seed
        try:
            coords = compute_umap(
                X_all,
                n_neighbors=um.n_neighbors,
                min_dist=um.min_dist,
                n_components=um.n_components,
                metric=um.metric,
                random_seed=seed,
            )
        except ImportError as exc:
            # umap-learn / matplotlib are imported lazily inside bioaccx.umap;
            # surface a clear install hint instead of a bare ModuleNotFoundError.
            raise RuntimeError(
                "umap.enabled is set but the UMAP extra is not installed. "
                "Install it with 'pip install bioaccx[umap]' (or 'uv sync --extra umap')."
            ) from exc
        point_labels = [label_names[int(i)] for i in y_all]

        umap_csv_path = out_dir / f"{stem}_umap.csv"
        write_umap_csv(umap_csv_path, coords, point_labels, splits, cluster_ids,
                       keys=sample_keys)
        outputs["umap_data"] = str(umap_csv_path)

        umap_plot_path = out_dir / f"{stem}_umap.png"
        write_umap_plot(
            umap_plot_path, coords, point_labels, label_names,
            title=f"UMAP — {fm.name} v{fm.version} ({len(X_all)} samples, {len(label_names)} classes)",
        )
        if umap_plot_path.exists():
            outputs["umap_plot"] = str(umap_plot_path)

        # Cluster plot: same UMAP coords coloured by the KMeans clusters that
        # the NMI score reflects (only when NMI was actually computed). The NMI
        # metrics are annotated as a text box on the figure.
        if cluster_ids is not None:
            cluster_plot_path = out_dir / f"{stem}_clusters.png"
            metrics = {
                "NMI": f"{nmi:.4f}",
                "clusters (k)": len(label_names),
                "classes": len(label_names),
                "samples": len(X_all),
            }
            write_cluster_plot(
                cluster_plot_path, coords, cluster_ids,
                title=f"KMeans clusters (k={len(label_names)}) — {fm.name} v{fm.version}",
                metrics=metrics,
            )
            if cluster_plot_path.exists():
                outputs["cluster_plot"] = str(cluster_plot_path)
    else:
        print("\n  UMAP disabled (set umap.enabled: true and install the [umap] extra to compute it).")

    print(f"\nDone. All outputs saved to: {out_dir}")
    return outputs


def run(cfg: BioaccxConfig) -> dict[str, str]:
    """Execute the full training pipeline and return a dict of output file paths.

    Pipeline steps:
      1. Validate the foundation model path / download.
      2. Load and (optionally) split the dataset.
      2b. Export chunked audio WAV files (when output.export_dataset is True).
      2c. Write the dataset_list CSV.
      3. Extract embeddings for train and test sets (with caching support).
      4. Train classifier(s) according to training.classifier.
      5. Export ONNX / TFLite heads and optionally full models; write reports.

    The dict of output paths uses logical keys like ``"keras_onnx_head"`` so
    callers can locate specific files without parsing filenames.
    """
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = cfg.foundation_model
    ds = cfg.dataset
    tr = cfg.training
    out = cfg.output

    stem = cfg.model_stem

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
    train_samples, test_samples = load_and_prepare_blocks(cfg, window_sec, fm.sample_rate)

    train_samples, test_mix_samples, mix_cfg, mix_seed = _add_audio_mixes(
        cfg, train_samples, test_samples)
    # What each label's training signal is made of, for the report, both
    # metadata files and the GUI.
    composition = training_composition(
        train_samples, test_samples,
        sorted({l for s in train_samples for l in sample_labels(s)}),
    ) if mix_cfg is not None else None

    # ------------------------------------------------------------------
    # 2b. Export chunked audio (optional)
    # ------------------------------------------------------------------
    single = len(cfg.dataset_blocks) == 1
    if out.export_dataset and _any_block_uses_ssh(cfg):
        print("\n[2b] Skipping dataset export: not supported for SSH data dirs.")
    elif out.export_dataset and (
        single
        and ds.label_mode == "subfolders"
        and ds.augmentation is None
        and not ds.append_dataset_path
        and not ds.random_sample_shift
        and mix_cfg is None
    ):
        print("\n[2b] Skipping dataset export: label_mode=subfolders with no "
              "augmentation already has the expected structure.")
    elif out.export_dataset:
        print("\n[2b] Exporting chunked audio dataset…")
        export_dataset_audio(
            train_samples, test_samples + test_mix_samples,
            out_dir=out_dir / "dataset",
            sample_rate=fm.sample_rate,
            window_samples=fm.get_window_samples(),
        )
    elif out.export_mixes:
        _export_mixes_only(train_samples, test_mix_samples, mix_cfg, out_dir,
                           fm.sample_rate, fm.get_window_samples(), step="[2b] ")

    # ------------------------------------------------------------------
    # 2c. Save dataset info CSV
    # ------------------------------------------------------------------
    dataset_list_path = out_dir / f"{stem}_dataset_list.csv"
    write_dataset_list(
        dataset_list_path, train_samples, test_samples, window_seconds=window_sec,
        filter=ds.filter, filter_freq=ds.filter_freq,
        filter_order=ds.filter_order, speed=ds.speed, test_mix_samples=test_mix_samples,
    )
    dataset_meta_path = out_dir / f"{stem}_dataset_metadata.json"
    write_dataset_metadata(
        dataset_meta_path, cfg, train_samples, test_samples, window_seconds=window_sec,
        mixup_composition=composition,
    )

    # ------------------------------------------------------------------
    # 3. Extract embeddings
    # ------------------------------------------------------------------
    _sqlite_exts = {".db", ".sqlite", ".sqlite3"}

    # Determine the embedding cache source (npy directory or sqlite file).
    cache_dir = None
    cache_sqlite = None
    if ds.embeddings_cache_path:
        cp = Path(ds.embeddings_cache_path)
        if cp.suffix.lower() in _sqlite_exts:
            cache_sqlite = cp
        else:
            cache_dir = cp

    # Determine where to export newly computed embeddings.
    export_dir = None
    export_sqlite = None
    fm_id = lookup_foundation_model_id(fm.name, fm.version, fm.data_type, fm.format)
    if out.export_embeddings:
        if out.embeddings_format == "sqlite":
            base = Path(out.embeddings_path) if out.embeddings_path else out_dir
            export_sqlite = base / f"{fm_id}_embeddings.db"
        else:
            export_dir = (
                Path(f"{out.embeddings_path}/embeddings/{stem}") if out.embeddings_path
                else out_dir / "embeddings"
            )

    # Guard against accidentally using embeddings from a different backbone —
    # the SQLite filename encodes the registry ID (e.g. 0xbb01).
    if cache_sqlite:
        if fm_id not in cache_sqlite.stem:
            print(
                f"  [warning] SQLite cache '{cache_sqlite.name}' does not match backbone "
                f"{fm_id} ({fm.name} v{fm.version}) — embeddings will be recomputed"
            )
            cache_sqlite = None

    if cache_sqlite:
        print(f"  Embeddings cache : {cache_sqlite} (sqlite)")
    elif cache_dir:
        print(f"  Embeddings cache : {cache_dir}")
    if export_sqlite:
        print(f"  Embeddings export: {export_sqlite} (sqlite)")
    elif export_dir:
        print(f"  Embeddings export: {export_dir}")

    ssh_config = None
    if ds.ssh_host:
        ssh_config = {
            "host": ds.ssh_host,
            "user": ds.ssh_user or "",
            "port": ds.ssh_port,
            "key_path": ds.ssh_key_path,
        }

    print("\n[3/5] Extracting embeddings…")
    # One label list, taken from the train set, indexes train and test alike.
    label_names = sorted({l for s in train_samples for l in sample_labels(s)})
    # Multi-hot targets only when some window really holds several labels
    # (mixes, merged overlapping annotations); otherwise the classic path.
    multi_label = any(s.extra_labels for s in train_samples + test_samples + test_mix_samples)
    embed_kwargs = dict(
        n_workers=ds.embedding_workers,
        cache_dir=cache_dir,
        export_dir=export_dir,
        cache_sqlite=cache_sqlite,
        export_sqlite=export_sqlite,
        ssh_config=ssh_config,
        label_names=label_names,
        multi_hot=multi_label,
    )
    train_keys: list[str] = []
    print("  Train set:")
    X_train, y_train, _ = extract_embeddings(
        train_samples, fm, fm.embedding_size, kept_keys=train_keys, **embed_kwargs)
    print("  Test set:")
    X_test, y_test, _ = extract_embeddings(test_samples, fm, fm.embedding_size, **embed_kwargs)
    mixed_test = None
    if test_mix_samples:
        print("  Mixed test set:")
        mixed_test = extract_embeddings(
            test_mix_samples, fm, fm.embedding_size, **embed_kwargs)[:2]
    is_mix = np.array([k.startswith("mix_") for k in train_keys], dtype=bool)

    # A sigmoid head learns from the multi-hot targets directly; any other head
    # (and sklearn) sees a multi-label window once per label, as before merging,
    # and never sees a mix.
    sigmoid_head = tr.keras.output_activation == "sigmoid" and not tr.keras.label_groups
    if multi_label:
        X_train_1, y_train_1 = _expand_multi_hot(X_train[~is_mix], y_train[~is_mix])
        X_test_1, y_test_1 = _expand_multi_hot(X_test, y_test)
    else:
        X_train_1, y_train_1, X_test_1, y_test_1 = X_train, y_train, X_test, y_test
    keras_multi = multi_label and sigmoid_head
    if keras_multi:
        X_k, y_k, X_kt, y_kt = X_train, y_train, X_test, y_test
    else:
        X_k, y_k, X_kt, y_kt = X_train_1, y_train_1, X_test_1, y_test_1

    embed_dim = fm.embedding_size
    n_train, n_test = len(X_train), len(X_test)
    n_train_mixes = int(is_mix.sum())
    report_meta = dict(
        data_dir=data_dir_display,
        foundation_name=fm.name,
        foundation_version=fm.version,
        foundation_data_type=fm.data_type,
        foundation_format=fm.format,
        embed_dim=embed_dim,
        test_ratio=ds.test_ratio,
        n_train=n_train,
        n_test=n_test,
        classifier=tr.classifier,
        output_type=out.output_type,
        output_format=out.output_format,
        output_data_types=cfg.output_data_types,
        augmentation=ds.augmentation,
        n_train_mixes=n_train_mixes,
    )
    if mix_cfg is not None:
        # The settings that shaped the training set, so a model's files say how
        # it was trained without the config at hand.
        report_meta["audio_mixup"] = {
            **dataclasses.asdict(mix_cfg), "seed": mix_seed,
            "n_train_mixes": n_train_mixes,
            "n_train_real": n_train - n_train_mixes,
            "n_test_mixes": len(mixed_test[0]) if mixed_test else 0,
            "composition": composition,
        }

    # Foundation model path (for full-model merge)
    foundation_local_path = _resolve_model_path(fm)

    # ------------------------------------------------------------------
    # 4. Train classifiers
    # ------------------------------------------------------------------
    print("\n[4/5] Training classifier(s)…")
    keras_model = None
    sklearn_pipe = None

    if tr.classifier in ("keras", "both"):
        keras_model = train_keras(X_k, y_k, X_kt, y_kt, label_names, tr.keras,
                                  seed=ds.random_seed if tr.keras.seed else None)
        if keras_model is not None:
            report_meta["keras_params"] = getattr(keras_model, "_report_params", {})

    if tr.classifier in ("sklearn", "both"):
        if n_train_mixes:
            print(f"  Note: sklearn trains on the {n_train - n_train_mixes} real train "
                  f"windows only; the {n_train_mixes} audio mixes need a multi-label head.")
        sklearn_pipe = train_sklearn(X_train_1, y_train_1, X_test_1, y_test_1,
                                     label_names, tr.sklearn)

    # ------------------------------------------------------------------
    # 5. Export models and write reports
    # ------------------------------------------------------------------
    print("\n[5/5] Exporting models and writing reports…")
    outputs: dict[str, str] = {
        "dataset_list": str(dataset_list_path),
        "dataset_metadata": str(dataset_meta_path),
    }
    onnx_head_paths: dict[str, Path] = {}

    # Compute output label set (may exclude background/noise labels).
    # The training set always uses all labels; exclusion only affects the exported
    # model outputs so that e.g. a 'background' class is not surfaced at inference.
    #
    # A grouped head emits its own label space (group members + "<group>_none"),
    # not the raw training labels, so exclusion is resolved against that space —
    # which is how the "_none" columns are dropped from the exported head.
    grouped_head = bool(tr.keras.label_groups) and keras_model is not None
    if grouped_head:
        group_output_labels, group_slices = build_group_space(tr.keras.label_groups, label_names)
        exportable_labels = group_output_labels
    else:
        group_output_labels, group_slices = None, None
        exportable_labels = list(label_names)

    excluded = set(out.exclude_labels)
    unknown_excluded = excluded - set(exportable_labels)
    if unknown_excluded:
        print(f"  Warning: exclude_labels not found in the model output labels: "
              f"{sorted(unknown_excluded)}")
    keep_indices = [i for i, n in enumerate(exportable_labels) if n not in excluded]
    output_label_names = [exportable_labels[i] for i in keep_indices]
    if not output_label_names:
        raise ValueError("exclude_labels removes every output column — nothing left to export")
    # Set before the reports are written: they add a macro-average over the
    # labels the exported model keeps.
    report_meta["excluded_labels"] = sorted(excluded & set(exportable_labels))
    if excluded & set(exportable_labels):
        print(f"  Excluding from output: {sorted(excluded & set(exportable_labels))}")
        print(f"  Output classes ({len(output_label_names)}): {output_label_names}")
    # Pass None when no filtering is needed so exporters skip the Gather node.
    keep_indices_arg = keep_indices if len(keep_indices) < len(exportable_labels) else None

    # A grouped head normalises *inside* each group before the Gather, so dropping
    # the "_none" columns afterwards is lossless — P(none) is recoverable as
    # 1 - sum(remaining columns of that group).  Stripping the activation would put
    # the Gather ahead of any softmax, which silently rebuilds the very defect the
    # grouped layout exists to avoid.
    if grouped_head and tr.keras.export_logits and keep_indices_arg is not None:
        raise ValueError(
            "export_logits cannot be combined with exclude_labels on a grouped head: the "
            "exported graph would gather raw logits, so the consumer could not reconstruct "
            "the per-group softmax. Drop one of the two."
        )

    # Output precisions to export (FP32/FP16/INT8). Defaults to the foundation
    # model's data_type; each precision produces a separately tagged file.
    data_types = cfg.output_data_types

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
        keras_export_model, report_meta["keras_exported_output"] = _prepare_keras_export(
            keras_model, tr.keras.output_activation, tr.keras.export_logits,
            grouped_head, keep_indices_arg, len(exportable_labels),
        )

        if do_onnx:
            for otype in _export_types(do_head, do_onnx_full):
                for dt in data_types:
                    fpath = out_dir / f"{stem}_{otype}_{dt.lower()}.onnx"
                    exported = export_onnx(
                        keras_export_model, "keras", embed_dim, fpath,
                        foundation_onnx_path=foundation_local_path,
                        foundation_input_name=fm.input_name,
                        output_type=otype,
                        keep_indices=keep_indices_arg,
                        data_type=dt,
                    )
                    outputs[f"keras_onnx_{otype}_{dt.lower()}"] = str(exported)
                    # Comparison report compares classifiers, not precisions —
                    # use the default (first) precision head for a 1:1 comparison.
                    if otype == "head" and dt == data_types[0]:
                        onnx_head_paths["keras"] = exported

        if do_tflite:
            for otype in _export_types(do_head, do_tflite_full):
                for dt in data_types:
                    fpath = out_dir / f"{stem}_{otype}_{dt.lower()}.tflite"
                    exported = export_tflite(
                        keras_export_model, "keras", embed_dim, fpath,
                        foundation_path=foundation_local_path,
                        foundation_input_name=fm.input_name,
                        output_type=otype,
                        keep_indices=keep_indices_arg,
                        tflite_output_tensor_offset=fm.tflite_output_tensor_offset,
                        data_type=dt,
                    )
                    if exported:
                        outputs[f"keras_tflite_{otype}_{dt.lower()}"] = str(exported)

        keras_report_path = out_dir / f"{stem}_report.txt"
        keras_eval_path = out_dir / f"{stem}_evaluation.csv"
        mixed_eval_path = out_dir / f"{stem}_evaluation_mixed.csv" if mixed_test else None
        write_keras_report(
            keras_model, X_kt, y_kt, label_names, keras_report_path,
            eval_csv_path=keras_eval_path, mixed_test=mixed_test,
            mixed_eval_csv_path=mixed_eval_path, **report_meta,
        )
        outputs["keras_report"] = str(keras_report_path)
        outputs["keras_evaluation"] = str(keras_eval_path)
        if mixed_eval_path is not None:
            outputs["keras_evaluation_mixed"] = str(mixed_eval_path)

    # ---- Sklearn exports ----
    # sklearn ONNX uses ai.onnx.ml ops that don't quantize/convert cleanly, so
    # sklearn heads are exported at FP32 only; other requested precisions are skipped.
    if sklearn_pipe is not None:
        skipped_sklearn_dts = [dt for dt in data_types if dt != "FP32"]
        if skipped_sklearn_dts:
            print(f"  Note: sklearn ONNX exported at FP32 only; skipping "
                  f"{', '.join(skipped_sklearn_dts)} (ai.onnx.ml ops are not quantizable).")
        if do_onnx and "FP32" in data_types:
            for otype in _export_types(do_head, do_onnx_full):
                fpath = out_dir / f"{stem}_sklearn_{otype}_fp32.onnx"
                exported = export_onnx(
                    sklearn_pipe, "sklearn", embed_dim, fpath,
                    foundation_onnx_path=foundation_local_path,
                    foundation_input_name=fm.input_name,
                    output_type=otype,
                    keep_indices=keep_indices_arg,
                    data_type="FP32",
                )
                outputs[f"sklearn_onnx_{otype}_fp32"] = str(exported)
                if otype == "head":
                    onnx_head_paths["sklearn"] = exported

        sklearn_report_path = out_dir / f"{stem}_sklearn_report.txt"
        write_sklearn_report(sklearn_pipe, X_test_1, y_test_1, label_names, sklearn_report_path,
                             **report_meta)
        outputs["sklearn_report"] = str(sklearn_report_path)

    # ---- Labels file (output labels only — excluded labels omitted) ----
    labels_path = out_dir / f"{stem}_labels.txt"
    labels_path.write_text("\n".join(output_label_names) + "\n")
    outputs["labels"] = str(labels_path)
    print(f"  Labels           → {labels_path}")

    # ---- Comparison report (when both classifiers trained) ----
    # The exported ONNX heads are already filtered to output_label_names.
    # y_test must be remapped to the reduced label space; samples whose true
    # label was excluded are dropped from evaluation entirely so that the
    # comparison metrics are computed over the same class space as the exports.
    if len(onnx_head_paths) > 1 and not grouped_head:
        # Build a mapping from old (full-label) integer → new (output-label) integer.
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        # Drop test samples belonging to excluded classes.
        mask = np.array([int(y) in old_to_new for y in y_test_1])
        X_test_cmp = X_test_1[mask]
        y_test_cmp = np.array([old_to_new[int(y)] for y in y_test_1[mask]])
        cmp_path = out_dir / f"{stem}_comparison_report.txt"
        write_comparison_report(
            onnx_head_paths, X_test_cmp, y_test_cmp, output_label_names, cmp_path, **report_meta
        )
        outputs["comparison_report"] = str(cmp_path)

    # ---- Model info JSON ----
    report_meta["output_labels"] = output_label_names
    if grouped_head:
        # Slice bounds index into the *unexcluded* grouped space, so a consumer can
        # map each exported column back to its group even when "_none" was dropped.
        report_meta["label_groups"] = {g: list(m) for g, m in tr.keras.label_groups.items()}
        report_meta["group_slices"] = [list(sl) for sl in group_slices]
        report_meta["grouped_output_labels"] = group_output_labels
    info_path = out_dir / f"{stem}_metadata.json"
    write_model_metadata(info_path, output_label_names, outputs, **report_meta)
    outputs["model_info"] = str(info_path)

    print(f"\nDone. All outputs saved to: {out_dir}")
    return outputs


def run_merge(cfg: BioaccxConfig, out_path: Path | None = None) -> Path:
    """Merge an existing ONNX backbone and ONNX or TFLite classifier head into a full ONNX model.

    Requires:
      - foundation_model.source == 'local'  → foundation_model.path (backbone ONNX file)
      - foundation_model.source == 'huggingface' → hf_repo (backbone downloaded via HF Hub)
      - output.head_path — path to the classifier head (ONNX or TFLite)
    The backbone must be ONNX. A TFLite head is converted to ONNX before merging.
    No training or dataset loading is done.
    """
    import tempfile

    fm = cfg.foundation_model
    out = cfg.output

    if fm.format != "onnx":
        raise ValueError(
            f"--merge requires backbone in ONNX format, got format='{fm.format}'"
        )
    if out.head_path is None:
        raise ValueError("output.head_path must be set for --merge")

    if fm.source == "local" and fm.path is not None:
        backbone_path = Path(fm.path)
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone not found: {backbone_path}")
    elif fm.source == "huggingface":
        backbone_path = _cached_hf_path(fm)
        if backbone_path is None:
            raise RuntimeError(
                f"Failed to download backbone from HuggingFace repo '{fm.hf_repo}'"
            )
    else:
        raise ValueError(
            "foundation_model.path must be set (local) or foundation_model.source must be 'huggingface' for --merge"
        )

    head_path = Path(out.head_path)
    if not head_path.exists():
        raise FileNotFoundError(f"Classifier head not found: {head_path}")

    head_is_tflite = head_path.suffix.lower() == ".tflite"

    # Without an explicit destination the merged model joins the config's
    # versioned output directory, like every other config-driven run.
    if out_path is None:
        out_path = cfg.output_dir / f"{cfg.model_stem}_full.onnx"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"bioaccx — merge ONNX backbone + classifier head")
    print(f"Backbone : {backbone_path}")
    print(f"Head     : {head_path}{' (tflite → onnx)' if head_is_tflite else ''}")
    print(f"Output   : {out_path}")
    print(f"{'='*62}\n")

    from bioaccx.exporters.onnx_exporter import _merge_onnx, _tflite_head_to_onnx

    if head_is_tflite:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_onnx = Path(tmp.name)
        try:
            print("  Converting TFLite head to ONNX…")
            _tflite_head_to_onnx(head_path, tmp_onnx)
            _merge_onnx(backbone_path, tmp_onnx, out_path, fm.input_name)
        finally:
            tmp_onnx.unlink(missing_ok=True)
    else:
        _merge_onnx(backbone_path, head_path, out_path, fm.input_name)

    print(f"\nDone. Merged model saved to: {out_path}")
    return out_path


def _write_extract_head_metadata(out_dir: Path, stem: str, src: Path, backbone_label: str,
                                 embed_dim: int, n_classes: int, label_names: list[str],
                                 output_label_names: list[str], outputs: dict) -> None:
    """Write the labels file and the extraction record next to the head(s)."""
    import json

    labels_out = out_dir / f"{stem}_labels.txt"
    labels_out.write_text("\n".join(output_label_names) + "\n", encoding="utf-8")
    outputs["labels"] = str(labels_out)

    info_path = out_dir / f"{stem}_extract_head.json"
    info_path.write_text(json.dumps({
        "source_model": str(src),
        "backbone": backbone_label,
        "embedding_size": embed_dim,
        "n_classes": n_classes,
        "labels": label_names,
        "output_labels": output_label_names,
        "outputs": outputs,
    }, indent=2), encoding="utf-8")
    outputs["info"] = str(info_path)


def run_extract_head(cfg: BioaccxConfig, out_dir: Path | None = None,
                     stem: str | None = None) -> dict[str, str]:
    """Extract the classifier head from a full model — TFLite or ONNX.

    From a **TFLite** source (a BirdNET-Analyzer model) the head is sliced
    straight out of the flatbuffer — without converting or running the backbone.
    The slice is the bit-exact TFLite head; the ONNX head is converted from that
    slice (tf2onnx handles a dense-only head, unlike the full model whose
    backbone uses ops ONNX lacks).

    From an **ONNX** source the graph's tail is read back into dense weights and
    rebuilt as an equivalent head, which is then exported through the normal
    exporters. The rebuilt head is checked against the source model before the
    run finishes (the source's own backbone supplies the embedding).

    Requires:
      - output.extract_from — path to the full ``.tflite`` or ``.onnx`` model
      - foundation_model embedding size (e.g. via registry_id) to validate the
        recovered head and define the head's input dimension.
    """
    import tempfile

    from bioaccx.extract_head import (
        build_keras_head,
        extract_onnx_head,
        extract_tflite_head,
        find_labels_file,
        read_labels_file,
        slice_tflite_head,
        verify_onnx_head,
    )

    fm = cfg.foundation_model
    out = cfg.output

    if out.extract_from is None:
        raise ValueError("output.extract_from must be set for --extract_head")
    src = Path(out.extract_from)
    if not src.exists():
        raise FileNotFoundError(f"Model to extract from not found: {src}")
    source_format = src.suffix.lower()
    if source_format not in (".tflite", ".onnx"):
        raise ValueError(
            f"--extract-head expects a .tflite or .onnx model, got '{src.suffix}'."
        )

    embed_dim = fm.embedding_size
    # Without an explicit destination the head lands in the config's versioned
    # output directory, named after the config's model stem.
    stem = stem or cfg.model_stem
    out_dir = Path(out_dir) if out_dir is not None else cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"bioaccx — extract classifier head from {source_format.lstrip('.').upper()}")
    print(f"Source   : {src}")
    # A CLI-supplied backbone may have no version to report ("--embed-dim 1024").
    backbone_label = fm.name if fm.version == "unknown" else f"{fm.name} v{fm.version}"
    print(f"Backbone : {backbone_label}  embed_dim={embed_dim}")
    print(f"Output   : {out_dir}")
    print(f"{'='*62}\n")

    # The source's own output filter (an exclude_labels export) is part of the
    # head: it is reapplied to the rebuilt head, and it defines how many classes
    # the head actually emits.
    source_filter: list[int] | None = None
    embedding_tensor = ""
    if source_format == ".onnx":
        layers, source_filter, embedding_tensor = extract_onnx_head(src, embed_dim)
    else:
        layers = extract_tflite_head(src, embed_dim)
    n_classes = len(source_filter) if source_filter else int(layers[-1].W.shape[0])
    arch = " → ".join([str(embed_dim)] + [str(int(l.W.shape[0])) for l in layers])
    acts = [l.activation or "linear" for l in layers]
    print(f"  Recovered head: {arch}  ({len(layers)} dense layer(s), activations={acts})")
    if source_filter:
        print(f"  Source model emits {n_classes} of {int(layers[-1].W.shape[0])} classes "
              f"(its own exclude_labels filter) — kept in the extracted head.")

    # Resolve class labels: explicit labels_file, else sibling *_Labels.txt.
    labels_path = find_labels_file(src, out.labels_file)
    label_names: list[str]
    if labels_path is not None and labels_path.exists():
        label_names = read_labels_file(labels_path)
        if len(label_names) != n_classes:
            print(f"  Warning: {labels_path.name} has {len(label_names)} labels "
                  f"but the head has {n_classes} classes — using generic names.")
            label_names = [f"class_{i}" for i in range(n_classes)]
        else:
            print(f"  Labels   : {label_names}")
    else:
        label_names = [f"class_{i}" for i in range(n_classes)]
        print(f"  Labels   : none found — using generic names {label_names}")

    # Output label filtering (exclude_labels), mirroring the training pipeline.
    excluded = set(out.exclude_labels)
    keep_indices = [i for i, n in enumerate(label_names) if n not in excluded]
    output_label_names = [label_names[i] for i in keep_indices]
    if excluded & set(label_names):
        print(f"  Excluding from output: {sorted(excluded & set(label_names))}")
    keep_indices_arg = keep_indices if len(keep_indices) < len(label_names) else None
    # The training loss of a source head is unknown here, so a linear head is
    # filtered as-is.  That is right for a sigmoid-trained head (BirdNET), but a
    # softmax-trained one loses the excluded classes' share of the softmax.
    if keep_indices_arg is not None and layers[-1].activation in (None, "linear"):
        print("  Warning: the head emits logits and exclude_labels drops columns from them. "
              "If the source was softmax-trained, the filtered logits cannot reproduce its "
              "probabilities — export without exclude_labels and filter after the softmax.")

    data_types = cfg.output_data_types
    do_onnx = out.output_format in ("onnx", "both")
    do_tflite = out.output_format in ("tflite", "both")

    outputs: dict[str, str] = {}

    if source_format == ".onnx":
        # No flatbuffer to slice: the recovered weights are rebuilt as a Keras
        # head (plus the source's own output filter) and written through the
        # normal exporters, so precisions and exclude_labels apply to both formats.
        from bioaccx.exporters.tflite_exporter import _slice_keras_output

        head_model = build_keras_head(layers, embed_dim)
        if source_filter:
            head_model = _slice_keras_output(head_model, source_filter)
        for dt in data_types:
            if do_onnx:
                fpath = out_dir / f"{stem}_head_{dt.lower()}.onnx"
                export_onnx(head_model, "keras", embed_dim, fpath,
                            output_type="head", keep_indices=keep_indices_arg, data_type=dt)
                outputs[f"onnx_head_{dt.lower()}"] = str(fpath)
            if do_tflite:
                fpath = out_dir / f"{stem}_head_{dt.lower()}.tflite"
                export_tflite(head_model, "keras", embed_dim, fpath,
                              output_type="head", keep_indices=keep_indices_arg, data_type=dt)
                outputs[f"tflite_head_{dt.lower()}"] = str(fpath)

        # Check the rebuilt head against the model it came from: the source runs
        # once with its embedding exposed, and the head is fed that embedding.
        # Only meaningful while both emit the same columns.
        onnx_head = outputs.get(f"onnx_head_{data_types[0].lower()}")
        if onnx_head and keep_indices_arg is None:
            diff, magnitude = verify_onnx_head(src, Path(onnx_head), embedding_tensor)
            tolerance = 1e-4 * max(1.0, magnitude)
            print(f"  Verification: max output difference = {diff:.2e} "
                  f"(outputs up to {magnitude:.2e})  "
                  f"[{'OK' if diff <= tolerance else 'MISMATCH'}]")
            if diff > tolerance:
                print("  Warning: the extracted head does not reproduce the source outputs.")
        elif onnx_head:
            print("  Verification skipped: exclude_labels changes the head's output columns.")

        _write_extract_head_metadata(
            out_dir, stem, src, backbone_label, embed_dim, n_classes,
            label_names, output_label_names, outputs,
        )
        print(f"\nDone. Head exported to: {out_dir}")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        return outputs

    # Slice the head straight out of the source flatbuffer — a bit-exact copy of
    # the original head ops/weights (preserving the source precision). This slice
    # is the TFLite head, and also the source for the ONNX head: because it
    # contains only dense/activation ops, tf2onnx converts it cleanly (the full
    # model can't be converted — its backbone uses ops like RFFT2D that ONNX
    # lacks). No Keras rebuild is needed.
    tflite_head = out_dir / f"{stem}_head.tflite" if do_tflite else Path(
        tempfile.NamedTemporaryFile(suffix=".tflite", delete=False).name
    )
    slice_tflite_head(src, embed_dim, tflite_head)
    try:
        if do_tflite:
            outputs["tflite_head"] = str(tflite_head)
            print(f"  TFLite head      → {tflite_head} (bit-exact slice)")
            if keep_indices_arg is not None:
                print("  Note: the bit-exact TFLite head keeps all source classes; "
                      "exclude_labels is applied to the ONNX head only.")
            if any(dt != "FP32" for dt in data_types):
                print("  Note: the TFLite head preserves the source precision; "
                      "data_types precision conversion applies to the ONNX head only.")
        if do_onnx:
            for dt in data_types:
                fpath = out_dir / f"{stem}_head_{dt.lower()}.onnx"
                export_tflite_head_to_onnx(
                    tflite_head, fpath, keep_indices=keep_indices_arg, data_type=dt,
                )
                outputs[f"onnx_head_{dt.lower()}"] = str(fpath)
                print(f"  ONNX head        → {fpath}")
    finally:
        if not do_tflite:
            tflite_head.unlink(missing_ok=True)

    _write_extract_head_metadata(
        out_dir, stem, src, backbone_label, embed_dim, n_classes,
        label_names, output_label_names, outputs,
    )

    print(f"\nDone. Head exported to: {out_dir}")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
    return outputs


def _prepare_keras_export(keras_model, output_activation: str | None, export_logits: bool,
                          grouped: bool, keep_indices: list[int] | None,
                          n_classes: int) -> tuple:
    """Return the Keras model the exporters convert, and what its output means.

    The exporters append the ``keep_indices`` Gather to this model's output.
    ``export_logits`` strips the output activation (BirdNET-style: training used
    sigmoid/softmax, the exported graph emits raw logits); the report keeps
    using *keras_model*, so its metrics stay in probability space.

    A flat softmax head's classes compete, so a Gather on its logits loses the
    excluded classes' share of the softmax and the consumer can no longer
    reproduce the trained probabilities.  With ``output_activation: null`` the
    softmax lives only in the loss, so it is inserted ahead of the Gather; with
    softmax + ``export_logits`` the two requests contradict and are rejected.
    """
    if (not grouped and output_activation == "softmax" and export_logits
            and keep_indices is not None):
        raise ValueError(
            "export_logits cannot be combined with exclude_labels on a softmax head: the "
            "exported graph would gather raw logits, so the consumer could not reconstruct "
            "the softmax over all trained classes. Drop one of the two."
        )

    export_model = keras_model
    if export_logits:
        export_model = strip_output_activation(keras_model)
        if export_model is keras_model:
            print("  Note: export_logits has no effect — the head already outputs logits "
                  "(output_activation: null).")
        else:
            print(f"  Stripping '{output_activation}' activation from exported "
                  f"head — exports emit logits.")

    insert_softmax = not grouped and output_activation is None and keep_indices is not None
    if insert_softmax:
        export_model = append_softmax(export_model)
        print(f"  Note: inserting a softmax over all {n_classes} trained classes before "
              f"dropping the excluded ones — exports emit the trained probabilities "
              f"for the {len(keep_indices)} kept classes (they do not sum to 1; the "
              f"remainder is P(excluded)). Consumers must not apply another activation.")

    return export_model, _keras_exported_output(
        output_activation, grouped, export_logits, insert_softmax,
    )


def _expand_multi_hot(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Repeat each row once per label it carries, with that label's index as target.

    This is how a single-label head sees a multi-label window: the same audio
    once as each of its classes, exactly as the separate annotations looked
    before they were merged.
    """
    rows, labels = np.nonzero(Y)
    return X[rows], labels.astype(np.int64)


def _keras_exported_output(output_activation: str | None, grouped: bool,
                           export_logits: bool, softmax_inserted: bool) -> dict:
    """Describe what the exported Keras head emits and what a consumer must apply.

    ``scores`` is ``"probabilities"`` or ``"logits"``; ``activation`` is the
    function a consumer applies to the exported output (``"identity"`` when it
    already holds probabilities).
    """
    act = "grouped_softmax" if grouped else output_activation
    if softmax_inserted:
        return {"scores": "probabilities", "activation": "identity",
                "softmax_before_filter": True,
                "note": "softmax over all trained classes applied before excluded columns "
                        "were dropped; kept columns do not sum to 1"}
    if act is None:
        return {"scores": "logits", "activation": "softmax"}
    if export_logits:
        return {"scores": "logits", "activation": act}
    return {"scores": "probabilities", "activation": "identity"}


def _export_types(do_head: bool, do_full: bool) -> list[str]:
    """Return the list of output_type strings to iterate over during export."""
    types = []
    if do_head:
        types.append("head")
    if do_full:
        types.append("full")
    return types


def _cached_hf_path(fm) -> Path | None:
    """Return the locally cached HuggingFace model path, downloading if needed.

    Returns None on any exception (missing repo, no internet, bad credentials)
    so that callers can surface a cleaner error message rather than a traceback
    from deep inside huggingface_hub.
    """
    if fm.source != "huggingface" or fm.hf_repo is None:
        return None
    try:
        from huggingface_hub import hf_hub_download
        from bioaccx.embedder import _default_hf_filename
        filename = fm.hf_filename or _default_hf_filename(fm)
        return Path(hf_hub_download(repo_id=fm.hf_repo, filename=filename, revision=fm.hf_revision))
    except Exception:
        return None
