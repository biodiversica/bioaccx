"""Integration tests for the --embeddings mode (run_embeddings).

Verifies that the embedding database, dataset list, UMAP CSV, and UMAP plot
are all produced, that no classifier/model files are written, and that the
UMAP CSV has one row per sample.

Marked @pytest.mark.slow because it loads the embedder over many samples and
fits UMAP.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.conftest import CLASS_FREQS, make_sine_wav

# 3 classes × 12 files × 3 windows (0.3 s file / 0.1 s window) = 108 chunks.
N_SAMPLES = len(CLASS_FREQS) * 12 * 3


@pytest.fixture(scope="module")
def umap_dataset(tmp_path_factory):
    """3 classes × 12 short WAV files in subfolders."""
    root = tmp_path_factory.mktemp("umap_ds")
    for ci, (cls, freq) in enumerate(CLASS_FREQS.items()):
        d = root / cls
        d.mkdir()
        for j in range(12):
            make_sine_wav(d / f"{cls}_{j:03d}.wav", freq, 0.3,
                          noise_std=0.05, seed=ci * 100 + j)
    return root


def _run_embeddings(dft_foundation_cfg, data_dir: Path, output_path: Path,
                    embeddings_format: str = "sqlite",
                    embeddings_overwrite: bool = False,
                    umap_enabled: bool = True,
                    umap_cache_csv: str | None = None) -> dict[str, str]:
    from bioaccx.config import _parse_config
    from bioaccx.train import run_embeddings

    fm = dft_foundation_cfg
    cfg_dict = {
        "foundation_model": {
            "name": fm.name, "version": fm.version,
            "format": fm.format, "source": "local", "path": fm.path,
            "sample_rate": fm.sample_rate, "window_samples": fm.window_samples,
            "input_name": fm.input_name, "output_name": fm.output_name,
            "embedding_size": fm.embedding_size,
        },
        "dataset": {
            "data_dir": str(data_dir),
            "label_mode": "subfolders",
            "embedding_workers": 2,
            "test_ratio": 0.25,
            "random_seed": 42,
        },
        "output": {
            "output_path": str(output_path),
            "model_name": "umap_test",
            "model_version": "0.1",
            "embeddings_format": embeddings_format,
            "embeddings_overwrite": embeddings_overwrite,
        },
        "umap": {"enabled": umap_enabled, "n_neighbors": 5, "random_seed": 42,
                 "cache_csv": umap_cache_csv},
    }
    cfg = _parse_config(cfg_dict)
    return run_embeddings(cfg)


@pytest.mark.slow
class TestEmbeddingsUmap:
    def test_produces_all_outputs(self, dft_foundation_cfg, umap_dataset, tmp_path):
        outputs = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        for key in ("embeddings", "dataset_list", "umap_data", "umap_plot"):
            assert key in outputs, f"missing output key: {key}"
            assert Path(outputs[key]).exists(), f"missing file for {key}"

    def test_embedding_database_is_sqlite(self, dft_foundation_cfg, umap_dataset, tmp_path):
        outputs = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        db_path = Path(outputs["embeddings"])
        assert db_path.suffix == ".db"
        import sqlite3
        with sqlite3.connect(db_path) as con:
            n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        assert n == N_SAMPLES, f"expected {N_SAMPLES} embeddings, got {n}"

    def test_umap_rows_carry_the_key_that_identifies_their_sample(
            self, dft_foundation_cfg, umap_dataset, tmp_path):
        """Without a key, a point cannot be traced back to the audio it came from.

        Row order alone is not enough: samples whose audio fails to load are
        dropped from the embedding matrix, so this file can be shorter than the
        dataset list it would otherwise be zipped against.
        """
        import csv

        outputs = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        with open(outputs["umap_data"], newline="") as fh:
            rows = list(csv.DictReader(fh))

        assert "key" in rows[0]
        keys = [row["key"] for row in rows]
        assert all(keys), "every point must carry a key"
        assert len(set(keys)) == len(keys), "keys must be unique"

    def test_umap_keys_match_the_embedding_database(
            self, dft_foundation_cfg, umap_dataset, tmp_path):
        """The key is the same identity the embedding store is keyed on."""
        import csv
        import sqlite3

        outputs = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        with open(outputs["umap_data"], newline="") as fh:
            keys = {row["key"] for row in csv.DictReader(fh)}
        with sqlite3.connect(outputs["embeddings"]) as con:
            stored = {row[0] for row in con.execute("SELECT key FROM embeddings")}
        assert keys == stored

    def test_umap_keys_resolve_to_rows_of_the_dataset_list(
            self, dft_foundation_cfg, umap_dataset, tmp_path):
        """Each key must locate the file, offset and label the point came from."""
        import csv

        outputs = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        with open(outputs["umap_data"], newline="") as fh:
            umap_rows = list(csv.DictReader(fh))
        with open(outputs["dataset_list"], newline="") as fh:
            dataset_rows = list(csv.DictReader(fh))

        by_stem = {Path(row["filepath"]).stem: row for row in dataset_rows}
        for row in umap_rows:
            # The key starts with the source file's stem; the rest encodes the
            # window and any augmentation applied to it.
            stem = next((s for s in by_stem if row["key"].startswith(s)), None)
            assert stem is not None, f"no dataset row for key {row['key']}"
            assert by_stem[stem]["label"] == row["label"]

    def test_no_classifier_or_model_written(self, dft_foundation_cfg, umap_dataset, tmp_path):
        outputs = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        out_dir = Path(outputs["dataset_list"]).parent
        assert not list(out_dir.glob("*.onnx"))
        assert not list(out_dir.glob("*.tflite"))
        assert not list(out_dir.glob("*_report.txt"))
        assert "keras_onnx_head_fp32" not in outputs
        assert "sklearn_onnx_head_fp32" not in outputs

    def test_umap_csv_has_cluster_column(self, dft_foundation_cfg, umap_dataset, tmp_path):
        outputs = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        with Path(outputs["umap_data"]).open() as f:
            reader = csv.DictReader(f)
            assert "cluster" in reader.fieldnames
            assert all(r["cluster"] != "" for r in reader)
        assert "cluster_plot" in outputs
        assert Path(outputs["cluster_plot"]).exists()

    def test_cache_csv_skips_computation_and_replots(
        self, dft_foundation_cfg, umap_dataset, tmp_path,
    ):
        # First run produces the UMAP CSV (with cluster column).
        out1 = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        cache_csv = out1["umap_data"]

        # Second run into a fresh dir, pointing at the cached CSV: it should
        # only redraw plots — no embedding store or dataset list written.
        out_dir2 = tmp_path / "replot"
        out2 = _run_embeddings(
            dft_foundation_cfg, umap_dataset, out_dir2, umap_cache_csv=cache_csv,
        )
        assert out2["umap_data"] == cache_csv
        assert Path(out2["umap_plot"]).exists()
        assert Path(out2["cluster_plot"]).exists()
        assert "embeddings" not in out2
        assert "dataset_list" not in out2
        assert not list(out_dir2.glob("*.db"))

    def test_umap_csv_one_row_per_sample(self, dft_foundation_cfg, umap_dataset, tmp_path):
        outputs = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        with Path(outputs["umap_data"]).open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert reader.fieldnames[:2] == ["umap_1", "umap_2"]
            assert "label" in reader.fieldnames
            assert "split" in reader.fieldnames
        assert len(rows) == N_SAMPLES
        assert {r["split"] for r in rows} <= {"train", "test"}
        assert {r["label"] for r in rows} == set(CLASS_FREQS)

    def test_npy_format(self, dft_foundation_cfg, umap_dataset, tmp_path):
        outputs = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path,
                                  embeddings_format="npy")
        emb_dir = Path(outputs["embeddings"])
        assert emb_dir.is_dir()
        assert len(list(emb_dir.glob("*.npy"))) == N_SAMPLES

    def test_existing_store_reused_not_overwritten(
        self, dft_foundation_cfg, umap_dataset, tmp_path,
    ):
        # First run creates the store; record its mtime.
        out1 = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        db_path = Path(out1["embeddings"])
        assert db_path.exists()
        mtime = db_path.stat().st_mtime_ns

        # Second run with overwrite off must reuse the store untouched while
        # still (re)building the UMAP outputs over it.
        out2 = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path,
                               embeddings_overwrite=False)
        assert Path(out2["embeddings"]) == db_path
        assert db_path.stat().st_mtime_ns == mtime, "store should not be rewritten"
        assert Path(out2["umap_data"]).exists()
        assert Path(out2["umap_plot"]).exists()

    def test_reuse_without_umap_skips_recompute(
        self, dft_foundation_cfg, umap_dataset, tmp_path,
    ):
        out1 = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        db_path = Path(out1["embeddings"])
        mtime = db_path.stat().st_mtime_ns

        # Reuse with UMAP disabled: nothing left to do, store is left as-is.
        out2 = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path,
                               umap_enabled=False)
        assert Path(out2["embeddings"]) == db_path
        assert db_path.stat().st_mtime_ns == mtime
        assert "umap_data" not in out2

    def test_overwrite_recomputes_store(
        self, dft_foundation_cfg, umap_dataset, tmp_path,
    ):
        out1 = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path)
        db_path = Path(out1["embeddings"])
        mtime = db_path.stat().st_mtime_ns

        # Force a recompute: the store is deleted and rebuilt from scratch.
        out2 = _run_embeddings(dft_foundation_cfg, umap_dataset, tmp_path,
                               embeddings_overwrite=True)
        assert Path(out2["embeddings"]) == db_path
        assert db_path.stat().st_mtime_ns != mtime, "store should be rewritten"
        import sqlite3
        with sqlite3.connect(db_path) as con:
            n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        assert n == N_SAMPLES
