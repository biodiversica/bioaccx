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
) -> None:
    """Write the UMAP coordinates to a CSV: one row per sample.

    Columns are ``umap_1 … umap_N``, ``label``, ``split``.
    """
    n_components = coords.shape[1]
    dim_cols = [f"umap_{i + 1}" for i in range(n_components)]
    fieldnames = dim_cols + ["label", "split"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row, label, split in zip(coords, labels, splits):
            writer.writerow([f"{v:.6f}" for v in row] + [label, split])
    print(f"  UMAP data CSV    → {path}")


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
