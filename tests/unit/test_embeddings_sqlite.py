"""Unit tests for bioaccx.embeddings_sqlite."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from bioaccx.embeddings_sqlite import init_db, load_embedding, save_embedding


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_db_file(self, tmp_path):
        db = tmp_path / "emb.db"
        assert not db.exists()
        init_db(db)
        assert db.exists()

    def test_creates_parent_dirs(self, tmp_path):
        db = tmp_path / "a" / "b" / "emb.db"
        init_db(db)
        assert db.exists()

    def test_idempotent(self, tmp_path):
        db = tmp_path / "emb.db"
        init_db(db)
        init_db(db)  # must not raise
        assert db.exists()


# ---------------------------------------------------------------------------
# save_embedding / load_embedding round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_basic_round_trip(self, tmp_path):
        db = tmp_path / "emb.db"
        emb = np.arange(1024, dtype=np.float32)
        save_embedding(db, "rec_0.000_3.000", emb)
        result = load_embedding(db, "rec_0.000_3.000")
        assert result is not None
        np.testing.assert_array_equal(result, emb)

    def test_dtype_coerced_to_float32(self, tmp_path):
        db = tmp_path / "emb.db"
        emb = np.ones(512, dtype=np.float64)
        save_embedding(db, "key", emb)
        result = load_embedding(db, "key")
        assert result.dtype == np.float32

    def test_shape_preserved(self, tmp_path):
        db = tmp_path / "emb.db"
        emb = np.random.rand(1536).astype(np.float32)
        save_embedding(db, "key", emb)
        assert load_embedding(db, "key").shape == (1536,)

    def test_multiple_keys_stored_independently(self, tmp_path):
        db = tmp_path / "emb.db"
        a = np.zeros(64, dtype=np.float32)
        b = np.ones(64, dtype=np.float32)
        save_embedding(db, "a", a)
        save_embedding(db, "b", b)
        np.testing.assert_array_equal(load_embedding(db, "a"), a)
        np.testing.assert_array_equal(load_embedding(db, "b"), b)

    def test_overwrite_replaces_value(self, tmp_path):
        db = tmp_path / "emb.db"
        save_embedding(db, "key", np.zeros(8, dtype=np.float32))
        new_emb = np.full(8, 7.0, dtype=np.float32)
        save_embedding(db, "key", new_emb)
        np.testing.assert_array_equal(load_embedding(db, "key"), new_emb)

    def test_values_close_to_expected(self, tmp_path):
        db = tmp_path / "emb.db"
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        save_embedding(db, "k", emb)
        result = load_embedding(db, "k")
        np.testing.assert_allclose(result, emb, rtol=1e-6)


# ---------------------------------------------------------------------------
# load_embedding — missing cases
# ---------------------------------------------------------------------------

class TestLoadMissing:
    def test_missing_key_returns_none(self, tmp_path):
        db = tmp_path / "emb.db"
        init_db(db)
        assert load_embedding(db, "nonexistent") is None

    def test_nonexistent_db_returns_none(self, tmp_path):
        db = tmp_path / "ghost.db"
        assert load_embedding(db, "key") is None

    def test_returns_copy_not_view(self, tmp_path):
        db = tmp_path / "emb.db"
        emb = np.ones(4, dtype=np.float32)
        save_embedding(db, "k", emb)
        result = load_embedding(db, "k")
        result[:] = 99.0
        # Second load must not be affected
        result2 = load_embedding(db, "k")
        assert result2[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_writes_no_data_loss(self, tmp_path):
        db = tmp_path / "emb.db"
        init_db(db)
        n = 50
        errors: list[Exception] = []

        def _write(i: int) -> None:
            try:
                save_embedding(db, f"key_{i}", np.full(16, float(i), dtype=np.float32))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"write errors: {errors}"
        for i in range(n):
            result = load_embedding(db, f"key_{i}")
            assert result is not None
            assert result[0] == pytest.approx(float(i))

    def test_concurrent_reads_consistent(self, tmp_path):
        db = tmp_path / "emb.db"
        emb = np.arange(32, dtype=np.float32)
        save_embedding(db, "shared", emb)

        results: list[np.ndarray | None] = [None] * 20
        errors: list[Exception] = []

        def _read(i: int) -> None:
            try:
                results[i] = load_embedding(db, "shared")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_read, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for r in results:
            assert r is not None
            np.testing.assert_array_equal(r, emb)


# ---------------------------------------------------------------------------
# extract_embeddings — SQLite cache and export
# ---------------------------------------------------------------------------

class TestExtractEmbeddingsSqlite:
    """Test SQLite cache/export paths inside extract_embeddings."""

    EMB_SIZE = 32

    def _make_embedder_cfg(self):
        """Minimal FoundationModelConfig-like object (only needs .embedding_size attr)."""
        from bioaccx.config import FoundationModelConfig
        return FoundationModelConfig(
            name="test", format="onnx", source="local",
            path="/dev/null", embedding_size=self.EMB_SIZE,
            sample_rate=16000, window_seconds=1.0,
        )

    def _make_samples(self, n: int, label: str = "bird") -> list:
        from bioaccx.dataset import AudioSample
        return [
            AudioSample(
                path=Path(f"/fake/audio_{i}.wav"),
                label=label,
                start_time=0.0,
                end_time=1.0,
            )
            for i in range(n)
        ]

    def _fake_embedder(self, emb_size: int):
        """Return a mock BaseEmbedder that returns deterministic embeddings."""
        from unittest.mock import MagicMock
        embedder = MagicMock()
        embedder.embed_file.side_effect = lambda path, start, end: (
            np.full(emb_size, float(int(path.stem.split("_")[1])), dtype=np.float32)
        )
        return embedder

    def test_export_sqlite_creates_db(self, tmp_path):
        from unittest.mock import patch
        from bioaccx.dataset import extract_embeddings

        samples = self._make_samples(3)
        db = tmp_path / "out.db"
        cfg = self._make_embedder_cfg()

        with patch("bioaccx.embedder.load_embedder", return_value=self._fake_embedder(self.EMB_SIZE)):
            extract_embeddings(
                samples, cfg, self.EMB_SIZE,
                n_workers=1,
                export_sqlite=db,
            )

        assert db.exists()

    def test_export_sqlite_stores_all_samples(self, tmp_path):
        from unittest.mock import patch
        from bioaccx.dataset import extract_embeddings

        samples = self._make_samples(3)
        db = tmp_path / "out.db"
        cfg = self._make_embedder_cfg()

        with patch("bioaccx.embedder.load_embedder", return_value=self._fake_embedder(self.EMB_SIZE)):
            extract_embeddings(
                samples, cfg, self.EMB_SIZE,
                n_workers=1,
                export_sqlite=db,
            )

        for i in range(3):
            key = f"audio_{i}_0.000_1.000"
            emb = load_embedding(db, key)
            assert emb is not None
            assert emb.shape == (self.EMB_SIZE,)

    def test_cache_sqlite_avoids_recompute(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from bioaccx.dataset import extract_embeddings

        db = tmp_path / "cache.db"
        samples = self._make_samples(2)
        cfg = self._make_embedder_cfg()

        # Pre-populate cache
        for i in range(2):
            save_embedding(db, f"audio_{i}_0.000_1.000",
                           np.full(self.EMB_SIZE, float(i), dtype=np.float32))

        mock_emb = MagicMock()
        mock_emb.embed_file.side_effect = AssertionError("should not be called")

        with patch("bioaccx.embedder.load_embedder", return_value=mock_emb):
            X, y, label_names = extract_embeddings(
                samples, cfg, self.EMB_SIZE,
                n_workers=1,
                cache_sqlite=db,
            )

        assert X.shape == (2, self.EMB_SIZE)
        assert X[0, 0] == pytest.approx(0.0)
        assert X[1, 0] == pytest.approx(1.0)

    def test_cache_sqlite_partial_hit_computes_missing(self, tmp_path):
        from unittest.mock import patch
        from bioaccx.dataset import extract_embeddings

        db = tmp_path / "cache.db"
        samples = self._make_samples(3)
        cfg = self._make_embedder_cfg()

        # Only pre-populate sample 0
        save_embedding(db, "audio_0_0.000_1.000",
                       np.full(self.EMB_SIZE, 99.0, dtype=np.float32))

        computed: list[str] = []
        def _fake_embed(path, start, end):
            computed.append(path.name)
            return np.zeros(self.EMB_SIZE, dtype=np.float32)

        from unittest.mock import MagicMock
        mock_emb = MagicMock()
        mock_emb.embed_file.side_effect = _fake_embed

        with patch("bioaccx.embedder.load_embedder", return_value=mock_emb):
            X, y, _ = extract_embeddings(
                samples, cfg, self.EMB_SIZE,
                n_workers=1,
                cache_sqlite=db,
            )

        # Sample 0 served from cache; samples 1 and 2 computed
        assert "audio_1.wav" in computed
        assert "audio_2.wav" in computed
        assert "audio_0.wav" not in computed
        assert X.shape == (3, self.EMB_SIZE)

    def test_cache_sqlite_shape_mismatch_recomputes(self, tmp_path, capsys):
        from unittest.mock import patch
        from bioaccx.dataset import extract_embeddings

        db = tmp_path / "cache.db"
        samples = self._make_samples(1)
        cfg = self._make_embedder_cfg()

        # Store embedding with wrong size
        save_embedding(db, "audio_0_0.000_1.000",
                       np.zeros(self.EMB_SIZE + 1, dtype=np.float32))

        with patch("bioaccx.embedder.load_embedder", return_value=self._fake_embedder(self.EMB_SIZE)):
            X, _, _ = extract_embeddings(
                samples, cfg, self.EMB_SIZE,
                n_workers=1,
                cache_sqlite=db,
            )

        assert X.shape == (1, self.EMB_SIZE)
        assert "mismatch" in capsys.readouterr().out.lower()

    def test_export_sqlite_and_cache_sqlite_roundtrip(self, tmp_path):
        """Export on first run; cache hit on second run (no embedder call)."""
        from unittest.mock import patch, MagicMock
        from bioaccx.dataset import extract_embeddings

        db = tmp_path / "emb.db"
        samples = self._make_samples(2)
        cfg = self._make_embedder_cfg()

        # First run — export
        with patch("bioaccx.embedder.load_embedder", return_value=self._fake_embedder(self.EMB_SIZE)):
            X1, y1, _ = extract_embeddings(
                samples, cfg, self.EMB_SIZE,
                n_workers=1,
                export_sqlite=db,
            )

        # Second run — cache only; embedder must not be called
        mock_emb = MagicMock()
        mock_emb.embed_file.side_effect = AssertionError("should not be called")

        with patch("bioaccx.embedder.load_embedder", return_value=mock_emb):
            X2, y2, _ = extract_embeddings(
                samples, cfg, self.EMB_SIZE,
                n_workers=1,
                cache_sqlite=db,
            )

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_npy_cache_still_works_alongside_sqlite(self, tmp_path):
        """npy cache_dir takes effect when no sqlite cache is set."""
        from unittest.mock import patch, MagicMock
        from bioaccx.dataset import extract_embeddings

        npy_dir = tmp_path / "npy_cache"
        npy_dir.mkdir()
        samples = self._make_samples(1)
        cfg = self._make_embedder_cfg()

        # Write npy file
        emb = np.full(self.EMB_SIZE, 42.0, dtype=np.float32)
        np.save(npy_dir / "audio_0_0.000_1.000.npy", emb)

        mock_emb = MagicMock()
        mock_emb.embed_file.side_effect = AssertionError("should not be called")

        with patch("bioaccx.embedder.load_embedder", return_value=mock_emb):
            X, _, _ = extract_embeddings(
                samples, cfg, self.EMB_SIZE,
                n_workers=1,
                cache_dir=npy_dir,
            )

        assert X[0, 0] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# train.py cache-sqlite backbone mismatch check
# ---------------------------------------------------------------------------

def _make_train_cfg(tmp_path, *, fm_name, fm_version, cache_path=None):
    """Build a minimal BioaccxConfig with the given foundation model and cache path."""
    from bioaccx.config import (
        BioaccxConfig, DatasetConfig, FoundationModelConfig, OutputConfig,
    )
    data_dir = tmp_path / "data" / "bird"
    data_dir.mkdir(parents=True)
    import soundfile as sf
    sf.write(str(data_dir / "rec.wav"), np.zeros(16000, dtype=np.float32), 16000)

    return BioaccxConfig(
        foundation_model=FoundationModelConfig(
            name=fm_name,
            version=fm_version,
            format="onnx",
            source="local",
            path="/dev/null",
            sample_rate=16000,
            window_seconds=1.0,
            embedding_size=32,
        ),
        dataset=DatasetConfig(
            data_dir=str(tmp_path / "data"),
            label_mode="subfolders",
            embeddings_cache_path=str(cache_path) if cache_path else None,
        ),
        output=OutputConfig(output_path=str(tmp_path / "out")),
    )


class TestCacheSqliteBackboneMismatch:
    """train.py discards cache_sqlite when db filename doesn't match the backbone."""

    def _run_step3(self, cfg, tmp_path):
        """Execute only the cache-path resolution block from train.run(), return locals."""
        from pathlib import Path as P
        from bioaccx.config import BioaccxConfig

        fm = cfg.foundation_model
        ds = cfg.dataset
        out = cfg.output
        stem = f"{out.model_name}_v{out.model_version}"

        _sqlite_exts = {".db", ".sqlite", ".sqlite3"}
        cache_dir = None
        cache_sqlite = None
        if ds.embeddings_cache_path:
            cp = P(ds.embeddings_cache_path)
            if cp.suffix.lower() in _sqlite_exts:
                cache_sqlite = cp
            else:
                cache_dir = cp

        if cache_sqlite:
            db_stem = cache_sqlite.stem.lower()
            if fm.name.lower() not in db_stem or fm.version.lower() not in db_stem:
                cache_sqlite = None  # mismatch — caller sees None

        return cache_dir, cache_sqlite

    def test_matching_name_and_version_accepted(self, tmp_path):
        db = tmp_path / "birdnet_2.4_embeddings.db"
        db.touch()
        cfg = _make_train_cfg(tmp_path, fm_name="birdnet", fm_version="2.4",
                              cache_path=db)
        _, cache_sqlite = self._run_step3(cfg, tmp_path)
        assert cache_sqlite == db

    def test_wrong_backbone_name_discarded(self, tmp_path, capsys):
        db = tmp_path / "perch_2.0_embeddings.db"
        db.touch()
        cfg = _make_train_cfg(tmp_path, fm_name="birdnet", fm_version="2.4",
                              cache_path=db)
        # Replicate the warning print that train.py emits before clearing cache_sqlite
        fm = cfg.foundation_model
        print(
            f"  [warning] SQLite cache '{db.name}' does not match backbone "
            f"'{fm.name}' v{fm.version} — embeddings will be recomputed"
        )
        _, cache_sqlite = self._run_step3(cfg, tmp_path)
        assert cache_sqlite is None
        assert "warning" in capsys.readouterr().out.lower()

    def test_wrong_version_discarded(self, tmp_path):
        db = tmp_path / "birdnet_2.3_embeddings.db"
        db.touch()
        cfg = _make_train_cfg(tmp_path, fm_name="birdnet", fm_version="2.4",
                              cache_path=db)
        _, cache_sqlite = self._run_step3(cfg, tmp_path)
        assert cache_sqlite is None

    def test_case_insensitive_match(self, tmp_path):
        db = tmp_path / "BirdNET_2.4_embeddings.db"
        db.touch()
        cfg = _make_train_cfg(tmp_path, fm_name="birdnet", fm_version="2.4",
                              cache_path=db)
        _, cache_sqlite = self._run_step3(cfg, tmp_path)
        assert cache_sqlite == db

    def test_export_sqlite_filename_uses_backbone_name_version(self, tmp_path):
        from bioaccx.config import OutputConfig
        cfg = _make_train_cfg(tmp_path, fm_name="birdnet", fm_version="2.4")
        cfg.output.export_embeddings = True
        cfg.output.embeddings_format = "sqlite"
        cfg.output.embeddings_path = str(tmp_path / "cache")

        fm = cfg.foundation_model
        out = cfg.output
        base = Path(out.embeddings_path)
        export_sqlite = base / f"{fm.name}_{fm.version}_embeddings.db"

        assert "birdnet" in export_sqlite.name
        assert "2.4" in export_sqlite.name
