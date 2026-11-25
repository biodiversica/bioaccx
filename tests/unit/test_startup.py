"""The CLI starts without the training stack.

Every command pays for what the package imports at start-up, `--help` included,
so the heavy libraries must load only inside the commands that use them. Each
check runs in a fresh interpreter: this test process has long since imported
them all.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY = ("numpy", "scipy", "sklearn", "onnxruntime", "onnx", "tensorflow",
         "requests", "fastapi", "uvicorn", "matplotlib", "umap")

_PROBE = """
import sys
from bioaccx.cli import main
try:
    main({argv!r})
except SystemExit:
    pass
print("LOADED:" + ",".join(m for m in {heavy!r} if m in sys.modules))
"""


@pytest.mark.parametrize("argv", [
    ["--help"], ["--version"], ["train", "--help"], ["embeddings", "--help"], ["gui", "--help"],
])
def test_help_and_version_load_no_heavy_library(argv):
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(argv=argv, heavy=HEAVY)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    line = next(l for l in result.stdout.splitlines() if l.startswith("LOADED:"))
    loaded = [m for m in line.removeprefix("LOADED:").split(",") if m]
    assert not loaded, f"`bioaccx {' '.join(argv)}` imported {loaded}"


_GUI_PROBE = """
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from bioaccx.gui.server import create_app
client = TestClient(create_app(config_path=None, token=None, models_dir=Path({models_dir!r})))
for path in ("/", "/api/bootstrap", "/api/template", "/api/models"):
    assert client.get(path).status_code == 200, path
print("LOADED:" + ",".join(m for m in {heavy!r} if m in sys.modules))
"""


def test_the_gui_starts_and_edits_without_heavy_libraries(tmp_path):
    """Opening the editor and listing runs needs none of them; clips,
    spectrograms and the map load numpy when first asked for."""
    heavy = tuple(m for m in HEAVY if m not in ("fastapi", "uvicorn"))
    result = subprocess.run(
        [sys.executable, "-c", _GUI_PROBE.format(models_dir=str(tmp_path), heavy=heavy)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    line = next(l for l in result.stdout.splitlines() if l.startswith("LOADED:"))
    loaded = [m for m in line.removeprefix("LOADED:").split(",") if m]
    assert not loaded, f"the GUI server imported {loaded} at start-up"


def test_the_public_api_still_imports_from_the_package():
    from bioaccx import BioaccxConfig, load_config, run
    from bioaccx.config import BioaccxConfig as Config, load_config as load
    from bioaccx.train import run as train_run

    assert (BioaccxConfig, load_config, run) == (Config, load, train_run)
