"""Unit tests for bioaccx.cli argument handling."""
from __future__ import annotations

import pytest

from bioaccx.cli import main


class TestNoSplitFlag:
    def test_rejected_without_dataset_mode(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["config.yaml", "--no-split"])
        assert exc.value.code == 2
        assert "--no-split is only valid together with --dataset" in capsys.readouterr().err

    def test_rejected_with_other_modes(self, capsys):
        with pytest.raises(SystemExit):
            main(["config.yaml", "--embeddings", "--no-split"])
        assert "--no-split" in capsys.readouterr().err

    def test_forwarded_to_dataset_export(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.config.load_config", lambda p: "CFG")
        monkeypatch.setattr(
            "bioaccx.train.run_dataset_export",
            lambda cfg, no_split=False: calls.update(cfg=cfg, no_split=no_split),
        )
        main([str(tmp_path / "config.yaml"), "--dataset", "--no-split"])
        assert calls == {"cfg": "CFG", "no_split": True}

    def test_dataset_export_defaults_to_split(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.config.load_config", lambda p: "CFG")
        monkeypatch.setattr(
            "bioaccx.train.run_dataset_export",
            lambda cfg, no_split=False: calls.update(no_split=no_split),
        )
        main([str(tmp_path / "config.yaml"), "--dataset"])
        assert calls == {"no_split": False}
