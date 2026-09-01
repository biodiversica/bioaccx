"""UMAP dimensionality reduction and visualisation for embedding databases.

Used by the ``--embeddings`` CLI mode to project the computed embedding
database into 2-D (or N-D) and write two artefacts:

  * a CSV with one row per sample (UMAP coordinates + label + split), and
  * a scatter-plot PNG coloured by label.

``umap-learn`` and ``matplotlib`` are imported lazily inside the functions so
that the rest of the package (training, export, dataset) does not pay their
import cost and so that environments without them still work for everything
except this mode.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def compute_nmi(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_clusters: int,
    random_seed: int | None = None,
) -> tuple[float, np.ndarray] | tuple[None, None]:
    """Normalized mutual information between KMeans clusters of *X* and labels *y*.

    Clusters the embedding matrix *X* with KMeans (``k = n_clusters``, normally
    the number of classes) and scores the agreement between those unsupervised
    clusters and the ground-truth labels *y* with
    ``sklearn.metrics.normalized_mutual_info_score``. This is an unsupervised
    measure of how well the classes separate in embedding space (0 = no
    agreement, 1 = perfect).

    Returns ``(score, cluster_ids)`` where ``cluster_ids`` is the per-sample
    KMeans assignment (so callers can plot the same clustering the score
    reflects), or ``(None, None)`` when the score is undefined — fewer than two
    classes, or fewer samples than requested clusters.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    n_samples = len(X)
    if n_clusters < 2 or n_samples < n_clusters:
        return None, None
    cluster_ids = KMeans(
        n_clusters=n_clusters, random_state=random_seed, n_init=10,
    ).fit_predict(X)
    return float(normalized_mutual_info_score(y, cluster_ids)), cluster_ids


def compute_umap(
    X: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "euclidean",
    random_seed: int | None = None,
) -> np.ndarray:
    """Fit UMAP on the embedding matrix *X* and return the projected coordinates.

    ``n_neighbors`` is clamped to ``len(X) - 1`` so that small datasets (common
    in quick test runs) do not raise inside umap-learn.
    """
    import umap  # deferred — heavy import, optional dependency

    n_samples = len(X)
    n_neighbors = max(2, min(n_neighbors, n_samples - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_seed,
    )
    return reducer.fit_transform(X)


def write_umap_csv(
    path: Path,
    coords: np.ndarray,
    labels: list[str],
    splits: list[str],
    cluster_ids: np.ndarray | None = None,
    keys: list[str] | None = None,
) -> None:
    """Write the UMAP coordinates to a CSV: one row per sample.

    Columns are ``umap_1 … umap_N``, ``label``, ``split``, and — when given —
    ``key`` (the sample's embedding key) and ``cluster`` (the KMeans
    assignment). The cluster column lets the CSV be reloaded later to redraw the
    cluster plot without recomputing embeddings or KMeans (see
    :func:`read_umap_csv`).

    The key column is what lets a point be traced back to the audio it came
    from. Row order alone cannot do that: samples whose audio fails to load are
    dropped from the embedding matrix, so this file can be shorter than the
    dataset list it would otherwise be zipped against.
    """
    n_components = coords.shape[1]
    dim_cols = [f"umap_{i + 1}" for i in range(n_components)]
    fieldnames = dim_cols + ["label", "split"]
    if keys is not None:
        fieldnames.append("key")
    if cluster_ids is not None:
        fieldnames.append("cluster")
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i, (row, label, split) in enumerate(zip(coords, labels, splits)):
            out_row = [f"{v:.6f}" for v in row] + [label, split]
            if keys is not None:
                out_row.append(keys[i] if i < len(keys) else "")
            if cluster_ids is not None:
                out_row.append(int(cluster_ids[i]))
            writer.writerow(out_row)
    print(f"  UMAP data CSV    → {path}")


def read_umap_csv(
    path: Path,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray | None]:
    """Read a UMAP data CSV back into ``(coords, labels, splits, cluster_ids)``.

    Inverse of :func:`write_umap_csv`. ``cluster_ids`` is ``None`` when the CSV
    has no ``cluster`` column (older caches) or any cluster cell is blank. A
    ``key`` column, when present, is ignored here — replotting needs only the
    coordinates; the GUI reads the keys itself.
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    dim_cols = [c for c in fields if c.startswith("umap_")]
    coords = np.array([[float(r[c]) for c in dim_cols] for r in rows], dtype=float)
    labels = [r["label"] for r in rows]
    splits = [r.get("split", "") for r in rows]

    cluster_ids = None
    if "cluster" in fields and all(r.get("cluster", "") != "" for r in rows):
        cluster_ids = np.array([int(r["cluster"]) for r in rows], dtype=int)
    return coords, labels, splits, cluster_ids


def nmi_from_assignments(labels, cluster_ids) -> float:
    """Normalized mutual information between two label assignments.

    Cheap recompute (no clustering) used when the cluster assignments are
    already known — e.g. reloaded from a cached UMAP CSV.
    """
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(labels, cluster_ids))


def write_umap_plot(
    path: Path,
    coords: np.ndarray,
    labels: list[str],
    label_names: list[str],
    title: str | None = None,
) -> None:
    """Write a 2-D scatter-plot PNG of the UMAP projection, coloured by label.

    Only the first two UMAP components are plotted. ``label_names`` provides a
    stable colour ordering so the legend is deterministic.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    if coords.shape[1] < 2:
        print("  [warning] UMAP plot needs n_components >= 2; skipping figure")
        return

    cmap = plt.get_cmap("tab20" if len(label_names) > 10 else "tab10")
    color_for = {name: cmap(i % cmap.N) for i, name in enumerate(label_names)}

    fig, ax = plt.subplots(figsize=(10, 8))
    labels_arr = np.asarray(labels)
    for name in label_names:
        mask = labels_arr == name
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=12, alpha=0.7, color=color_for[name], label=name, edgecolors="none",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or "UMAP projection of embeddings")
    ax.legend(loc="best", fontsize="small", markerscale=1.5, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  UMAP plot        → {path}")


def write_cluster_plot(
    path: Path,
    coords: np.ndarray,
    cluster_ids: np.ndarray,
    title: str | None = None,
    metrics: dict | None = None,
) -> None:
    """Write a 2-D scatter-plot PNG of the UMAP projection, coloured by KMeans cluster.

    Plots the same points as :func:`write_umap_plot` but coloured by the cluster
    assignment that the NMI score reflects, so the two figures can be compared
    side by side. Only the first two UMAP components are plotted. When *metrics*
    is given, its key/value pairs are rendered as a text box on the figure.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    if coords.shape[1] < 2:
        print("  [warning] cluster plot needs n_components >= 2; skipping figure")
        return

    cluster_ids = np.asarray(cluster_ids)
    unique_clusters = np.unique(cluster_ids)
    cmap = plt.get_cmap("tab20" if len(unique_clusters) > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cid in enumerate(unique_clusters):
        mask = cluster_ids == cid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=12, alpha=0.7, color=cmap(i % cmap.N),
            label=f"cluster {cid}", edgecolors="none",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or "UMAP projection coloured by KMeans cluster")
    ax.legend(loc="best", fontsize="small", markerscale=1.5, framealpha=0.8)

    if metrics:
        text = "\n".join(f"{k}: {v}" for k, v in metrics.items())
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes,
            fontsize="small", va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"),
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Cluster plot     → {path}")
