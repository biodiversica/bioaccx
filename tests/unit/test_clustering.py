"""The clustering behind the NMI score, the cluster column and the cluster plot."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from bioaccx.config import _parse_umap
from bioaccx.umap import (
    CLUSTER_METHODS,
    NOISE,
    cluster_embeddings,
    cluster_summary,
    compute_nmi,
    read_cluster_method,
    read_umap_csv,
    write_umap_csv,
)


def _blobs(n_per=30, dims=16, seed=0):
    """Three well-separated groups, so every method should recover them."""
    rng = np.random.default_rng(seed)
    centres = np.eye(3, dims) * 20.0
    X = np.concatenate([c + rng.normal(0, 0.5, (n_per, dims)) for c in centres])
    y = np.repeat(np.arange(3), n_per)
    return X.astype(np.float32), y


class TestMethods:
    @pytest.mark.parametrize("method", sorted(CLUSTER_METHODS))
    def test_every_method_recovers_separated_groups(self, method):
        X, y = _blobs()
        nmi, ids = compute_nmi(X, y, n_clusters=3, random_seed=0, method=method)
        assert ids.shape == (len(X),)
        assert nmi > 0.95, f"{method} scored {nmi}"

    def test_hdbscan_chooses_the_number_of_clusters(self):
        X, y = _blobs()
        ids = cluster_embeddings(X, method="hdbscan", n_clusters=10, min_cluster_size=5)
        assert len(np.unique(ids[ids != NOISE])) == 3

    def test_hdbscan_leaves_an_outlier_as_noise(self):
        X, _ = _blobs()
        X = np.vstack([X, np.full((1, X.shape[1]), 500.0, dtype=np.float32)])
        ids = cluster_embeddings(X, method="hdbscan", n_clusters=3, min_cluster_size=5)
        assert ids[-1] == NOISE

    @pytest.mark.parametrize("method", ["kmeans", "agglomerative", "gmm"])
    def test_k_based_methods_use_the_requested_k(self, method):
        X, _ = _blobs()
        ids = cluster_embeddings(X, method=method, n_clusters=4, random_seed=0)
        assert len(np.unique(ids)) == 4

    def test_undefined_clusterings_return_nothing(self):
        X, y = _blobs(n_per=2)
        assert compute_nmi(X, y, n_clusters=1) == (None, None)
        assert compute_nmi(X, y, n_clusters=len(X) + 1) == (None, None)
        assert compute_nmi(X, y, n_clusters=3, method="hdbscan",
                           min_cluster_size=len(X) + 1) == (None, None)

    def test_an_unknown_method_is_refused(self):
        X, _ = _blobs()
        with pytest.raises(ValueError, match="unknown cluster method"):
            cluster_embeddings(X, method="dbscan", n_clusters=3)


class TestSummary:
    def test_names_the_method_and_size(self):
        assert cluster_summary("gmm", [0, 1, 2, 2]) == "Gaussian mixture, k=3"
        assert cluster_summary("hdbscan", [0, -1, -1, 1]) == \
            "HDBSCAN, 2 clusters, 2 noise points"


class TestCsv:
    def _write(self, path, method="hdbscan"):
        coords = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        write_umap_csv(path, coords, ["a", "b", "a"], ["train"] * 3,
                       np.array([0, NOISE, 1]), cluster_method=method)

    def test_the_method_is_recorded_and_read_back(self, tmp_path):
        path = tmp_path / "u.csv"
        self._write(path)
        assert read_cluster_method(path) == "hdbscan"
        _, _, _, ids = read_umap_csv(path)
        assert list(ids) == [0, NOISE, 1]

    def test_an_older_csv_reads_as_kmeans(self, tmp_path):
        path = tmp_path / "old.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["umap_1", "umap_2", "label", "split", "cluster"],
                                     ["0", "0", "a", "train", "0"]])
        assert read_cluster_method(path) == "kmeans"


class TestConfig:
    def test_defaults_keep_kmeans(self):
        um = _parse_umap({})
        assert (um.cluster_method, um.n_clusters, um.min_cluster_size) == ("kmeans", None, 5)

    @pytest.mark.parametrize("raw,message", [
        ({"cluster_method": "dbscan"}, "cluster_method"),
        ({"n_clusters": 1}, "n_clusters"),
        ({"min_cluster_size": 1}, "min_cluster_size"),
    ])
    def test_bad_values_are_refused(self, raw, message):
        with pytest.raises(ValueError, match=message):
            _parse_umap(raw)


class TestMap:
    def test_noise_is_grey_and_the_method_is_named(self, tmp_path):
        from bioaccx.gui import results
        from bioaccx.umap import NOISE_HEX

        directory = tmp_path / "m"
        directory.mkdir()
        (directory / "m_metadata.json").write_text("{}", encoding="utf-8")
        write_umap_csv(directory / "m_umap.csv", np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
                       ["a", "b", "a"], ["train"] * 3, np.array([0, NOISE, 1]),
                       keys=["k0", "k1", "k2"], cluster_method="hdbscan")
        umap = results.umap_points(tmp_path, "m")
        assert umap["clusters"] == [NOISE, 0, 1]
        assert umap["cluster_colors"][0] == NOISE_HEX
        assert len(umap["cluster_colors"]) == 3
        assert umap["cluster_method_name"] == "HDBSCAN"

    def test_the_nmi_is_recomputed_from_the_csv(self, tmp_path):
        from bioaccx.gui import results

        directory = tmp_path / "m"
        directory.mkdir()
        (directory / "m_metadata.json").write_text("{}", encoding="utf-8")
        coords = np.zeros((4, 2))
        write_umap_csv(directory / "m_umap.csv", coords, ["a", "a", "b", "b"], ["train"] * 4,
                       np.array([0, 0, 1, 1]), keys=["k0", "k1", "k2", "k3"])
        assert results.umap_points(tmp_path, "m")["nmi"] == pytest.approx(1.0)

    def test_no_nmi_without_a_cluster_column(self, tmp_path):
        from bioaccx.gui import results

        directory = tmp_path / "m"
        directory.mkdir()
        (directory / "m_metadata.json").write_text("{}", encoding="utf-8")
        write_umap_csv(directory / "m_umap.csv", np.zeros((2, 2)), ["a", "b"], ["train"] * 2,
                       keys=["k0", "k1"])
        assert results.umap_points(tmp_path, "m")["nmi"] is None
