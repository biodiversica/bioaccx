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
import itertools
import math
from functools import lru_cache
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


#: Lightness and chroma of each band, in Oklab terms. Two colours can differ by
#: hue, by lightness, or by both; bands give the palette the second axis, which
#: is what keeps neighbouring hues apart. Kept mid-range so every colour stays
#: legible on the white figure and on the dark background of the browser's map.
_BANDS = {
    2: ((0.55, 0.20), (0.76, 0.15)),
    3: ((0.52, 0.21), (0.66, 0.18), (0.80, 0.14)),
    4: ((0.50, 0.21), (0.60, 0.19), (0.70, 0.17), (0.80, 0.14)),
}

#: Where on the hue circle the first colour sits. Chroma is limited by the sRGB
#: gamut differently at every hue, so the offset changes how much of the circle
#: survives — it is searched rather than chosen.
_HUE_OFFSETS = (10.0, 25.0, 40.0, 55.0, 70.0, 85.0)


def _linear_srgb(lightness: float, a: float, b: float) -> tuple[float, float, float]:
    """Oklab → linear sRGB (Björn Ottosson's matrices)."""
    long = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    med = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    short = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (+4.0767416621 * long - 3.3077115913 * med + 0.2309699292 * short,
            -1.2684380046 * long + 2.6097574011 * med - 0.3413193965 * short,
            -0.0041960863 * long - 0.7034186147 * med + 1.7076147010 * short)


def _encode(channel: float) -> float:
    """Linear light → sRGB, the transfer function a display expects."""
    return (12.92 * channel if channel <= 0.0031308
            else 1.055 * channel ** (1 / 2.4) - 0.055)


def _oklch(lightness: float, chroma: float, hue: float) -> tuple[tuple, tuple]:
    """One colour as (oklab coordinates, RGB), with chroma pulled into gamut.

    Oklab describes more colours than a screen can show, and how many depends
    on the hue — yellow reaches far, blue does not. Rather than clipping the
    RGB (which silently moves the colour somewhere else), the chroma is reduced
    until the colour fits, keeping its hue and lightness exactly.
    """
    radians = math.radians(hue)
    while chroma > 0.01:
        a, b = chroma * math.cos(radians), chroma * math.sin(radians)
        rgb = _linear_srgb(lightness, a, b)
        if all(-1e-4 <= channel <= 1 + 1e-4 for channel in rgb):
            break
        chroma -= 0.002
    a, b = chroma * math.cos(radians), chroma * math.sin(radians)
    rgb = _linear_srgb(lightness, a, b)
    return (lightness, a, b), tuple(min(1.0, max(0.0, _encode(c))) for c in rgb)


def _stride(count: int) -> int:
    """A step that visits every slot of *count* once, in a scattered order.

    Coprime with *count*, so stepping by it is a permutation rather than a short
    cycle; near the golden ratio, so consecutive classes land on opposite sides
    of the hue circle instead of side by side in the legend.
    """
    step = max(1, round(count * 0.381966))
    while step < count and math.gcd(step, count) != 1:
        step += 1
    return step if step < count else 1


def _ring(count: int, bands: tuple, offset: float) -> list[tuple[tuple, tuple]]:
    """One candidate palette: the hue circle shared evenly, bands alternating."""
    stride, out = _stride(count), []
    for index in range(count):
        slot = (index * stride) % count
        lightness, chroma = bands[slot % len(bands)]
        out.append(_oklch(lightness, chroma, offset + slot * 360.0 / count))
    return out


@lru_cache(maxsize=None)
def distinct_colors(count: int) -> list[tuple[float, float, float]]:
    """*count* RGB colours, as far apart as this scheme can put them.

    Every fixed palette runs out: matplotlib's ``tab10``/``tab20`` hold 10 and
    20 entries and were indexed with ``i % N``, which handed two classes the
    same colour outright. Even below that limit they are not as distinct as
    they look — ``tab20`` is ten hues in a light and a dark version, and its
    closest pair sits at ΔE 0.05, which reads as one colour on a scatter plot.

    So the palette is generated for the number of classes actually present:
    the hue circle is divided evenly in Oklab, where equal steps look equally
    different, and split across lightness bands so that neighbouring hues also
    differ in brightness. Band count and starting hue are searched — a few
    dozen candidates — and the one whose closest pair is furthest apart wins.

    >>> len(set(distinct_colors(64))) == 64
    True
    >>> distinct_colors(3) == distinct_colors(9)[:3]
    False
    >>> min_gap(distinct_colors(20)) > 0.09       # tab20's closest pair: 0.054
    True
    """
    if count <= 0:
        return []
    if count == 1:
        return [_oklch(*_BANDS[2][0], _HUE_OFFSETS[1])[1]]

    best, best_gap = None, -1.0
    for bands in _BANDS.values():
        for offset in _HUE_OFFSETS:
            ring = _ring(count, bands, offset)
            # The Oklab coordinates are the ones asked for, and Euclidean
            # distance in Oklab is the point of the space: it approximates how
            # different two colours look.
            gap = min(math.dist(a[0], b[0]) for a, b in itertools.combinations(ring, 2))
            if gap > best_gap:
                best, best_gap = ring, gap
    return [rgb for _, rgb in best]


def hex_colors(count: int) -> list[str]:
    """:func:`distinct_colors` as ``#rrggbb``, for the browser and for CSS.

    >>> hex_colors(2)
    ['#ca2356', '#0dcbc3']
    """
    return ["#%02x%02x%02x" % tuple(round(channel * 255) for channel in colour)
            for colour in distinct_colors(count)]


def min_gap(colors) -> float:
    """The closest pair in *colors*, in Oklab ΔE. Used by the tests.

    >>> round(min_gap([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]), 3)   # black vs white
    1.0
    """
    def oklab(rgb):
        def linear(u):
            return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
        red, green, blue = (linear(c) for c in rgb)
        long = (0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue) ** (1 / 3)
        med = (0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue) ** (1 / 3)
        short = (0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue) ** (1 / 3)
        return (0.2104542553 * long + 0.7936177850 * med - 0.0040720468 * short,
                1.9779984951 * long - 2.4285922050 * med + 0.4505937099 * short,
                0.0259040371 * long + 0.7827717662 * med - 0.8086757660 * short)

    labs = [oklab(c) for c in colors]
    return min(math.dist(a, b) for a, b in itertools.combinations(labs, 2))


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

    color_for = dict(zip(label_names, distinct_colors(len(label_names))))

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
    colors = distinct_colors(len(unique_clusters))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cid in enumerate(unique_clusters):
        mask = cluster_ids == cid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=12, alpha=0.7, color=colors[i],
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
