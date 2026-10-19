"""Every text file bioaccx opens names its encoding.

Without one, Python uses the locale's: UTF-8 on Linux and macOS, cp1252 on
Windows, where writing a report with an arrow in it raises and a label like
`Pequeno-gavião` is read back garbled. Nothing on a Linux machine notices, so
this scans the source for text-mode file I/O that leaves the encoding out.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = sorted(p for d in ("bioaccx", "tests", "tools") for p in (ROOT / d).rglob("*.py"))


#: tempfile's constructors: mode is their first argument, and binary by default.
TEMPFILES = ("NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile")


def _mode(call: ast.Call, name: str) -> object:
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value.value if isinstance(kw.value, ast.Constant) else None
    # open(path, mode) for the builtin, path.open(mode) for pathlib
    index = 1 if isinstance(call.func, ast.Name) and name == "open" else 0
    if len(call.args) > index:
        arg = call.args[index]
        return arg.value if isinstance(arg, ast.Constant) else None
    return "w+b" if name in TEMPFILES else "r"


def _unspecified(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "webbrowser":
                continue
            name = func.attr
        else:
            continue
        if name not in ("open", "read_text", "write_text", *TEMPFILES):
            continue
        if any(kw.arg == "encoding" for kw in node.keywords):
            continue
        mode = _mode(node, name) if name in ("open", *TEMPFILES) else "r"
        if isinstance(mode, str) and "b" not in mode:
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_text_io_names_its_encoding(path):
    lines = _unspecified(ast.parse(path.read_text(encoding="utf-8")))
    assert not lines, f"pass encoding= at line(s) {lines}"


def test_the_scan_catches_a_bare_open():
    tree = ast.parse("open(p)\np.read_text()\np.open('w')\nopen(p, 'rb')\n"
                     "p.write_text(s, encoding='utf-8')\nwebbrowser.open(url)\n"
                     "tempfile.NamedTemporaryFile('w', suffix='.yaml')\n"
                     "tempfile.NamedTemporaryFile(suffix='.onnx')\n")
    assert _unspecified(tree) == [1, 2, 3, 7]
