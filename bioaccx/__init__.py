"""bioaccx — bioacoustic custom classifier generator."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.2.1"

# The public names load on first use rather than here. The CLI's entry point
# lives inside this package, so whatever this file imports is paid by every
# command, `--help` and `--version` included — and the training pipeline pulls
# in numpy, scipy, scikit-learn and onnxruntime. `from bioaccx import run` still
# works: it goes through __getattr__ below.
_LAZY = {
    "BioaccxConfig": "bioaccx.config",
    "load_config": "bioaccx.config",
    "run": "bioaccx.train",
}

__all__ = ["BioaccxConfig", "load_config", "run"]

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from bioaccx.config import BioaccxConfig, load_config
    from bioaccx.train import run


def __getattr__(name: str):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value   # later lookups skip this function
        return value
    raise AttributeError(f"module 'bioaccx' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
