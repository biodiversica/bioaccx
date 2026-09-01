"""Unit tests for the subcommand interface.

The pre-subcommand flag interface is covered by ``test_cli.py``, which drives
the same entry point through :mod:`bioaccx._legacy_cli`; these tests cover the
subcommand forms directly, plus the translation between the two.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bioaccx._legacy_cli import is_legacy, translate
from bioaccx.cli import main


@pytest.fixture
def cfg(monkeypatch):
    """A config file path whose contents never need to be read."""
    monkeypatch.setattr("bioaccx.config.load_config", lambda p: "CFG")
    return "my_config.yaml"


class TestPipelineCommands:
    def test_train_runs_the_pipeline(self, monkeypatch, cfg):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run", lambda c: calls.update(cfg=c))
        main(["train", cfg])
        assert calls == {"cfg": "CFG"}

    def test_dataset_defaults_to_splitting(self, monkeypatch, cfg):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_dataset_export",
                            lambda c, no_split=False: calls.update(no_split=no_split))
        main(["dataset", cfg])
        assert calls == {"no_split": False}

    def test_dataset_no_split(self, monkeypatch, cfg):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_dataset_export",
                            lambda c, no_split=False: calls.update(no_split=no_split))
        main(["dataset", cfg, "--no-split"])
        assert calls == {"no_split": True}

    def test_embeddings_runs_embeddings_only(self, monkeypatch, cfg):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_embeddings", lambda c: calls.update(cfg=c))
        main(["embeddings", cfg])
        assert calls == {"cfg": "CFG"}

    def test_validate_reads_the_config_without_training(self, monkeypatch, capsys, tmp_path):
        from bioaccx.config import BioaccxConfig, FoundationModelConfig
        monkeypatch.setattr(
            "bioaccx.config.load_config",
            lambda p: BioaccxConfig(foundation_model=FoundationModelConfig(name="birdnet")),
        )
        monkeypatch.setattr("bioaccx.train.run",
                            lambda c: pytest.fail("training must not run"))
        main(["validate", str(tmp_path / "cfg.yaml")])
        assert "Config parsed successfully." in capsys.readouterr().out

    def test_missing_config_is_a_usage_error(self, monkeypatch, capsys, tmp_path):
        def _missing(path):
            raise FileNotFoundError(f"no such config: {path}")
        monkeypatch.setattr("bioaccx.config.load_config", _missing)
        with pytest.raises(SystemExit) as exc:
            main(["train", str(tmp_path / "nope.yaml")])
        assert exc.value.code == 2
        assert "no such config" in capsys.readouterr().err


class TestFlagsAreScopedToTheirCommand:
    """Combinations the old interface had to reject by hand are now unspeakable."""

    def test_no_split_does_not_exist_on_train(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["train", "cfg.yaml", "--no-split"])
        assert exc.value.code == 2
        assert "--no-split" in capsys.readouterr().err

    def test_no_split_does_not_exist_on_embeddings(self, capsys):
        with pytest.raises(SystemExit):
            main(["embeddings", "cfg.yaml", "--no-split"])
        assert "--no-split" in capsys.readouterr().err

    def test_out_does_not_exist_on_dataset(self, capsys):
        with pytest.raises(SystemExit):
            main(["dataset", "cfg.yaml", "-o", "x.onnx"])
        assert "-o" in capsys.readouterr().err

    def test_backbone_does_not_exist_on_train(self, capsys):
        with pytest.raises(SystemExit):
            main(["train", "cfg.yaml", "--backbone", "0xbb00"])
        assert "--backbone" in capsys.readouterr().err


class TestMerge:
    def _head(self, tmp_path) -> Path:
        head = tmp_path / "head.onnx"
        head.write_bytes(b"")
        return head

    def test_head_plus_backbone_id(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_merge",
                            lambda cfg, out_path=None: calls.update(cfg=cfg, out=out_path))
        head = self._head(tmp_path)
        main(["merge", str(head), "--backbone", "0xbb00"])
        assert calls["cfg"].foundation_model.name == "birdnet"
        assert calls["cfg"].output.head_path == str(head)
        assert calls["out"] == tmp_path / "head_full.onnx"

    def test_explicit_output(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_merge",
                            lambda cfg, out_path=None: calls.update(out=out_path))
        main(["merge", str(self._head(tmp_path)), "--backbone", "0xbb00",
              "-o", str(tmp_path / "full.onnx")])
        assert calls["out"] == tmp_path / "full.onnx"

    def test_config_form_takes_no_backbone(self, monkeypatch, cfg):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_merge",
                            lambda c, out_path=None: calls.update(cfg=c, out=out_path))
        main(["merge", cfg])
        assert calls == {"cfg": "CFG", "out": None}

    def test_head_without_backbone_is_rejected(self, capsys, tmp_path):
        with pytest.raises(SystemExit) as exc:
            main(["merge", str(self._head(tmp_path))])
        assert exc.value.code == 2
        assert "requires --backbone" in capsys.readouterr().err

    def test_config_with_backbone_is_rejected(self, capsys, cfg):
        with pytest.raises(SystemExit):
            main(["merge", cfg, "--backbone", "0xbb00"])
        assert "not both" in capsys.readouterr().err

    def test_unknown_suffix_names_both_forms(self, capsys, tmp_path):
        with pytest.raises(SystemExit):
            main(["merge", str(tmp_path / "head.bin"), "--backbone", "0xbb00"])
        err = capsys.readouterr().err
        assert "config file" in err and "model file" in err

    def test_unknown_backbone_id(self, capsys, tmp_path):
        with pytest.raises(SystemExit):
            main(["merge", str(self._head(tmp_path)), "--backbone", "0xdead"])
        assert "neither an existing file nor a known registry ID" in capsys.readouterr().err


class TestExtractHead:
    def _model(self, tmp_path) -> Path:
        src = tmp_path / "full.tflite"
        src.write_bytes(b"")
        return src

    def test_embed_dim_without_backbone(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_extract_head",
                            lambda cfg, out_dir=None, stem=None: calls.update(
                                cfg=cfg, out_dir=out_dir, stem=stem))
        src = self._model(tmp_path)
        main(["extract-head", str(src), "--embed-dim", "1024"])
        assert calls["cfg"].foundation_model.embedding_size == 1024
        assert calls["cfg"].output.extract_from == str(src)
        assert calls["out_dir"] == tmp_path / "full_head"
        assert calls["stem"] == "full"

    def test_defaults_to_both_formats(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_extract_head",
                            lambda cfg, out_dir=None, stem=None: calls.update(cfg=cfg))
        main(["extract-head", str(self._model(tmp_path)), "--embed-dim", "1024"])
        assert calls["cfg"].output.output_format == "both"

    def test_format_and_labels_forwarded(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_extract_head",
                            lambda cfg, out_dir=None, stem=None: calls.update(
                                cfg=cfg, out_dir=out_dir))
        labels = tmp_path / "L.txt"
        labels.write_text("a\n")
        main(["extract-head", str(self._model(tmp_path)), "--embed-dim", "1024",
              "--format", "onnx", "--labels", str(labels), "-o", str(tmp_path / "out")])
        assert calls["cfg"].output.output_format == "onnx"
        assert calls["cfg"].output.labels_file == str(labels)
        assert calls["out_dir"] == tmp_path / "out"

    def test_rejects_an_unknown_format(self, capsys, tmp_path):
        with pytest.raises(SystemExit):
            main(["extract-head", str(self._model(tmp_path)), "--embed-dim", "1024",
                  "--format", "keras"])
        assert "keras" in capsys.readouterr().err

    def test_model_without_backbone_or_embed_dim(self, capsys, tmp_path):
        with pytest.raises(SystemExit):
            main(["extract-head", str(self._model(tmp_path))])
        assert "--backbone" in capsys.readouterr().err

    def test_config_form(self, monkeypatch, cfg):
        calls = {}
        monkeypatch.setattr("bioaccx.train.run_extract_head",
                            lambda c, out_dir=None, stem=None: calls.update(cfg=c))
        main(["extract-head", cfg])
        assert calls == {"cfg": "CFG"}


class TestConvertHead:
    def test_forwards_source_and_output(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.convert_head.convert_head",
                            lambda src, dst=None: calls.update(src=src, dst=dst))
        main(["convert-head", str(tmp_path / "head.tflite"), "-o", str(tmp_path / "x.onnx")])
        assert calls == {"src": tmp_path / "head.tflite", "dst": tmp_path / "x.onnx"}

    def test_output_optional(self, monkeypatch, tmp_path):
        calls = {}
        monkeypatch.setattr("bioaccx.convert_head.convert_head",
                            lambda src, dst=None: calls.update(dst=dst))
        main(["convert-head", str(tmp_path / "head.onnx")])
        assert calls == {"dst": None}

    def test_conversion_error_becomes_a_usage_error(self, monkeypatch, capsys, tmp_path):
        def _raise(src, dst=None):
            raise ValueError("not a classifier head")
        monkeypatch.setattr("bioaccx.convert_head.convert_head", _raise)
        with pytest.raises(SystemExit) as exc:
            main(["convert-head", str(tmp_path / "head.tflite")])
        assert exc.value.code == 2
        assert "not a classifier head" in capsys.readouterr().err


class TestRegistry:
    def test_lists_models(self, capsys):
        main(["registry"])
        out = capsys.readouterr().out
        assert "0xbb00" in out and "Available foundation models" in out


class TestServeAddresses:
    """The hint lines `bioaccx gui` prints when it binds a reachable address."""

    def test_loopback_is_recognised(self):
        from bioaccx.cli import _is_loopback
        assert _is_loopback("127.0.0.1") and _is_loopback("localhost")
        assert _is_loopback("::1")

    def test_a_routable_address_is_not_loopback(self):
        from bioaccx.cli import _is_loopback
        assert not _is_loopback("0.0.0.0")
        assert not _is_loopback("100.97.127.55")   # a tailnet address

    def test_no_tailscale_binary_means_no_tailnet_line(self, monkeypatch):
        import shutil
        from bioaccx.cli import _tailscale_address
        monkeypatch.setattr(shutil, "which", lambda _: None)
        assert _tailscale_address() is None

    def test_reports_the_first_tailnet_address(self, monkeypatch):
        import shutil
        import subprocess
        from bioaccx.cli import _tailscale_address

        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/tailscale")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, stdout="100.97.127.55\nfd7a:115c::1\n", stderr=""))
        assert _tailscale_address() == "100.97.127.55"

    def test_a_failing_tailscale_call_is_quiet(self, monkeypatch):
        import shutil
        import subprocess
        from bioaccx.cli import _tailscale_address

        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/tailscale")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a[0], 1, stdout="", stderr="not running"))
        assert _tailscale_address() is None

    def test_tailscale_blowing_up_is_not_fatal(self, monkeypatch):
        import shutil
        import subprocess
        from bioaccx.cli import _tailscale_address

        def _boom(*a, **k):
            raise OSError("no such binary")
        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/tailscale")
        monkeypatch.setattr(subprocess, "run", _boom)
        assert _tailscale_address() is None


class TestLegacyDetection:
    @pytest.mark.parametrize("argv", [
        ["cfg.yaml"],
        ["cfg.yaml", "--dataset"],
        ["--registry"],
        ["--convert-head", "head.onnx"],
        ["--merge", "head.onnx", "--backbone", "0xbb00"],
        ["--extract_head", "model.tflite", "--embed-dim", "1024"],
    ])
    def test_old_forms_are_recognised(self, argv):
        assert is_legacy(argv)

    @pytest.mark.parametrize("argv", [
        [],
        ["--help"],
        ["-h"],
        ["train", "cfg.yaml"],
        ["dataset", "cfg.yaml", "--no-split"],
        ["merge", "head.onnx", "--backbone", "0xbb00"],
        ["extract-head", "model.tflite", "--embed-dim", "1024"],
    ])
    def test_new_forms_are_left_alone(self, argv):
        assert not is_legacy(argv)

    @pytest.mark.parametrize("old,new", [
        (["cfg.yaml"], ["train", "cfg.yaml"]),
        (["cfg.yaml", "--validate"], ["validate", "cfg.yaml"]),
        (["cfg.yaml", "--dataset"], ["dataset", "cfg.yaml"]),
        (["cfg.yaml", "--dataset", "--no-split"], ["dataset", "cfg.yaml", "--no-split"]),
        (["cfg.yaml", "--embeddings"], ["embeddings", "cfg.yaml"]),
        (["cfg.yaml", "--merge"], ["merge", "cfg.yaml"]),
        (["cfg.yaml", "--extract-head"], ["extract-head", "cfg.yaml"]),
        (["cfg.yaml", "--extract_head"], ["extract-head", "cfg.yaml"]),
        (["--registry"], ["registry"]),
        (["--convert-head", "h.tflite"], ["convert-head", "h.tflite"]),
        (["--convert-head", "h.tflite", "-o", "h.onnx"],
         ["convert-head", "h.tflite", "-o", "h.onnx"]),
        (["--merge", "h.onnx", "--backbone", "0xbb00"],
         ["merge", "h.onnx", "--backbone", "0xbb00"]),
    ])
    def test_translation(self, old, new):
        assert translate(old) == new
