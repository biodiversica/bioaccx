"""Comment-preserving read and write of config files.

The editor's real workflow is not writing a config from scratch — it is opening
``v0.7``, changing two parameters and saving ``v0.8``. Round-tripping through
PyYAML would silently drop every comment, and ``example_config.yaml`` is mostly
comments that document the format. So the GUI reads and writes through
``ruamel.yaml`` in round-trip mode, which preserves comments, key order and
quoting style.

:func:`bioaccx.config.load_config` is untouched and stays on PyYAML; nothing
here changes how a config is interpreted, only how it is edited in place.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

CONFIG_SUFFIXES = frozenset({".yaml", ".yml", ".json"})


class ConfigIOError(Exception):
    """A config file could not be read or written as asked."""


def _represent_none(representer, _data):
    """Emit ``null`` rather than an empty value.

    ruamel writes a bare ``key:`` for None, which parses identically but rewrites
    every explicit ``null`` in the file the first time it is saved — and in this
    config ``null`` is documented (``output_activation: null`` means logits). A
    save should change only what was edited.
    """
    return representer.represent_scalar("tag:yaml.org,2002:null", "null")


def _yaml():
    from ruamel.yaml import YAML
    y = YAML()               # round-trip mode: comments and key order survive
    y.preserve_quotes = True
    y.width = 4096           # don't reflow long scalars into folded lines
    y.indent(mapping=2, sequence=4, offset=2)
    y.representer.add_representer(type(None), _represent_none)
    return y


def load(path: Path) -> Any:
    """Read a config file into a round-trippable document.

    JSON is read as plain data — it carries no comments to preserve — and is
    written back as JSON.
    """
    if not path.exists():
        raise ConfigIOError(f"no such config file: {path}")
    text = path.read_text()
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigIOError(f"{path.name} is not valid JSON: {exc}") from exc
    try:
        return _yaml().load(text) or {}
    except Exception as exc:  # ruamel raises several parse error types
        raise ConfigIOError(f"{path.name} is not valid YAML: {exc}") from exc


def loads(text: str) -> Any:
    """Parse YAML text from the editor's raw pane."""
    try:
        return _yaml().load(text) or {}
    except Exception as exc:
        raise ConfigIOError(f"not valid YAML: {exc}") from exc


def dumps(document: Any, *, as_json: bool = False) -> str:
    """Serialise a document back to text, keeping whatever comments it carries."""
    if as_json:
        return json.dumps(document, indent=2) + "\n"
    stream = io.StringIO()
    _yaml().dump(document, stream)
    return stream.getvalue()


def save(path: Path, document: Any) -> None:
    """Write *document* to *path*, creating parent directories as needed."""
    if path.suffix.lower() not in CONFIG_SUFFIXES:
        raise ConfigIOError(
            f"{path.name}: config files must end in "
            f"{', '.join(sorted(CONFIG_SUFFIXES))}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(document, as_json=path.suffix.lower() == ".json"))


def apply_edits(document: Any, edits: dict[str, Any]) -> Any:
    """Apply ``{"dotted.path": value}`` edits onto *document* in place.

    Setting a value to ``None`` removes the key, which is how the form clears an
    optional field without writing an explicit ``null`` the loader would then
    have to interpret. Intermediate mappings are created on demand, and existing
    ones are edited rather than replaced so their comments survive.

    >>> doc = {"output": {"model_name": "old", "model_version": "0.1"}}
    >>> apply_edits(doc, {"output.model_name": "new", "output.model_version": None})
    {'output': {'model_name': 'new'}}
    >>> apply_edits({}, {"training.keras.epochs": 10})
    {'training': {'keras': {'epochs': 10}}}
    """
    for dotted, value in edits.items():
        parts = dotted.split(".")
        node = document
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                node[part] = {}
            node = node[part]
        leaf = parts[-1]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return document


def redact(document: Any, secret_paths: list[str]) -> Any:
    """A copy of *document* with secret values replaced by a presence marker.

    The browser is told whether a credential is set, never what it is, so a
    config open in the editor cannot leak its API key into a page, a screenshot
    or a browser cache. Writing the field back is handled by
    :func:`apply_edits`, which leaves a key untouched when the form submits it
    empty.
    """
    out = json.loads(json.dumps(document, default=str))
    for dotted in secret_paths:
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and node.get(parts[-1]) not in (None, ""):
            node[parts[-1]] = "__SET__"
    return out
