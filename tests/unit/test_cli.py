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


class TestConvertHeadFlag:
    def test_forwards_source_and_output(self, monkeypatch, tmp_path):
        from pathlib import Path
        calls = {}
        monkeypatch.setattr(
            "bioaccx.convert_head.convert_head",
            lambda src, dst=None: calls.update(src=src, dst=dst),
        )
        main(["--convert-head", str(tmp_path / "head.tflite"), "-o", str(tmp_path / "x.onnx")])
        assert calls == {"src": Path(tmp_path / "head.tflite"), "dst": Path(tmp_path / "x.onnx")}

    def test_output_optional(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr(
            "bioaccx.convert_head.convert_head",
            lambda src, dst=None: calls.update(dst=dst),
        )
        main(["--convert-head", str(tmp_path / "head.onnx")])
        assert calls == {"dst": None}

    def test_needs_no_config(self, monkeypatch, tmp_path):
        """The head path is the only argument the mode requires."""
        monkeypatch.setattr("bioaccx.convert_head.convert_head", lambda src, dst=None: None)
        monkeypatch.setattr("bioaccx.config.load_config",
                            lambda p: pytest.fail("config must not be loaded"))
        main(["--convert-head", str(tmp_path / "head.onnx")])

    def test_conversion_error_becomes_cli_error(self, monkeypatch, capsys, tmp_path):
        def _raise(src, dst=None):
            raise ValueError("not a classifier head")
        monkeypatch.setattr("bioaccx.convert_head.convert_head", _raise)
        with pytest.raises(SystemExit) as exc:
            main(["--convert-head", str(tmp_path / "head.tflite")])
        assert exc.value.code == 2
        assert "not a classifier head" in capsys.readouterr().err

    def test_out_rejected_for_config_driven_modes(self, capsys):
        with pytest.raises(SystemExit):
            main(["config.yaml", "--dataset", "-o", "x.onnx"])
        assert "-o/--out is only valid with --convert-head" in capsys.readouterr().err


class TestMergeCli:
    """--merge with its inputs on the command line (no config file)."""

    def _head(self, tmp_path):
        head = tmp_path / "head.onnx"
        head.write_bytes(b"")
        return head

    def test_builds_config_from_backbone_id(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_merge",
                            lambda cfg, out_path=None: calls.update(cfg=cfg, out=out_path))
        head = self._head(tmp_path)
        main(["--merge", str(head), "--backbone", "0xbb00"])
        cfg = calls["cfg"]
        assert cfg.foundation_model.name == "birdnet"          # registry defaults applied
        assert cfg.foundation_model.source == "huggingface"
        assert cfg.output.head_path == str(head)

    def test_default_output_is_sibling_of_head(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_merge",
                            lambda cfg, out_path=None: calls.update(out=out_path))
        main(["--merge", str(self._head(tmp_path)), "--backbone", "0xbb00"])
        assert calls["out"] == tmp_path / "head_full.onnx"

    def test_explicit_output(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_merge",
                            lambda cfg, out_path=None: calls.update(out=out_path))
        main(["--merge", str(self._head(tmp_path)), "--backbone", "0xbb00",
              "-o", str(tmp_path / "full.onnx")])
        assert calls["out"] == tmp_path / "full.onnx"

    def test_requires_backbone(self, capsys, tmp_path):
        with pytest.raises(SystemExit):
            main(["--merge", str(self._head(tmp_path))])
        assert "requires --backbone" in capsys.readouterr().err

    def test_unknown_backbone_id(self, capsys, tmp_path):
        with pytest.raises(SystemExit):
            main(["--merge", str(self._head(tmp_path)), "--backbone", "0xdead"])
        err = capsys.readouterr().err
        assert "neither an existing file nor a known registry ID" in err

    def test_config_and_cli_inputs_are_exclusive(self, capsys, tmp_path):
        with pytest.raises(SystemExit):
            main(["cfg.yaml", "--merge", str(self._head(tmp_path)), "--backbone", "0xbb00"])
        assert "not both" in capsys.readouterr().err

    def test_flag_without_value_still_uses_config(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.config.load_config", lambda p: "CFG")
        monkeypatch.setattr("bioaccx.train.run_merge",
                            lambda cfg, out_path=None: calls.update(cfg=cfg, out=out_path))
        main([str(tmp_path / "cfg.yaml"), "--merge"])
        assert calls == {"cfg": "CFG", "out": None}


class TestExtractHeadCli:
    """--extract-head with its inputs on the command line (no config file)."""

    def _model(self, tmp_path):
        src = tmp_path / "full.tflite"
        src.write_bytes(b"")
        return src

    def test_embed_dim_without_backbone(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_extract_head",
                            lambda cfg, out_dir=None, stem=None: calls.update(
                                cfg=cfg, out_dir=out_dir, stem=stem))
        src = self._model(tmp_path)
        main(["--extract-head", str(src), "--embed-dim", "1024"])
        assert calls["cfg"].foundation_model.embedding_size == 1024
        assert calls["cfg"].output.extract_from == str(src)
        assert calls["out_dir"] == tmp_path / "full_head"
        assert calls["stem"] == "full"

    def test_backbone_id_supplies_embed_dim(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_extract_head",
                            lambda cfg, out_dir=None, stem=None: calls.update(cfg=cfg))
        main(["--extract-head", str(self._model(tmp_path)), "--backbone", "0xbb02"])
        assert calls["cfg"].foundation_model.embedding_size == 1024

    def test_format_and_labels_forwarded(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_extract_head",
                            lambda cfg, out_dir=None, stem=None: calls.update(
                                cfg=cfg, out_dir=out_dir))
        labels = tmp_path / "L.txt"
        labels.write_text("a\n")
        main(["--extract-head", str(self._model(tmp_path)), "--embed-dim", "1024",
              "--format", "onnx", "--labels", str(labels), "-o", str(tmp_path / "out")])
        assert calls["cfg"].output.output_format == "onnx"
        assert calls["cfg"].output.labels_file == str(labels)
        assert calls["out_dir"] == tmp_path / "out"

    def test_defaults_to_both_formats(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_extract_head",
                            lambda cfg, out_dir=None, stem=None: calls.update(cfg=cfg))
        main(["--extract-head", str(self._model(tmp_path)), "--embed-dim", "1024"])
        assert calls["cfg"].output.output_format == "both"

    def test_requires_backbone_or_embed_dim(self, capsys, tmp_path):
        with pytest.raises(SystemExit):
            main(["--extract-head", str(self._model(tmp_path))])
        assert "--backbone" in capsys.readouterr().err

    def test_underscore_alias_still_accepted(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.config.load_config", lambda p: "CFG")
        monkeypatch.setattr("bioaccx.train.run_extract_head",
                            lambda cfg, out_dir=None, stem=None: calls.update(cfg=cfg))
        main([str(tmp_path / "cfg.yaml"), "--extract_head"])
        assert calls == {"cfg": "CFG"}

    def test_backbone_rejected_without_a_cli_mode(self, capsys):
        with pytest.raises(SystemExit):
            main(["cfg.yaml", "--backbone", "0xbb00"])
        assert "only valid with --merge" in capsys.readouterr().err
