"""Form schema generated from the config dataclasses.

The browser form is never hand-written: it is rendered from the descriptors
produced here, so a field added to :mod:`bioaccx.config` appears in the GUI
with the right widget, default and help text without anyone editing the UI.

Help text comes from the ``#`` comments already sitting above (or beside) each
field in ``config.py``, read out of the source with :mod:`ast`. That keeps one
copy of the documentation rather than a second one drifting in the front end.

Two config keys are not dataclass fields and are added synthetically, because
the loader consumes them before the dataclasses are built:
``foundation_model.registry_id`` (:func:`bioaccx.config._resolve_foundation_model`)
and ``dataset.sources`` (:func:`bioaccx.config._parse_dataset_section`).
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import types
import typing
from typing import Any, Literal, Union

from bioaccx import config as cfg_mod
from bioaccx.config import (
    RUN_LEVEL_DATASET_FIELDS,
    AugmentationConfig,
    BioaccxConfig,
    DatasetConfig,
    FoundationModelConfig,
    KerasConfig,
    OutputConfig,
    SklearnConfig,
    TrainingConfig,
    UmapConfig,
)
from bioaccx.registry import registry_entries

#: Fields shown before the reader expands "advanced". Everything else is real
#: but rarely touched; tiering keeps the form readable without hiding anything.
COMMON_FIELDS: dict[str, frozenset[str]] = {
    "foundation_model": frozenset({
        "registry_id", "name", "version", "format", "source", "path",
        "sample_rate", "window_seconds", "embedding_size",
    }),
    "dataset": frozenset({
        "data_dir", "label_mode", "table_file", "audio_extensions", "overlap",
        "test_ratio", "random_seed", "sources", "augmentation",
    }),
    "training": frozenset({"classifier"}),
    "training.keras": frozenset({
        "hidden_units", "dropout", "epochs", "batch_size", "learning_rate",
        "output_activation", "label_groups",
    }),
    "training.sklearn": frozenset({"C", "max_iter"}),
    "output": frozenset({
        "output_path", "model_name", "model_version", "output_type",
        "output_format", "exclude_labels", "export_dataset", "export_embeddings",
    }),
    "umap": frozenset({"enabled", "n_neighbors", "min_dist", "n_components"}),
}

#: Fields whose value must never travel back to the browser. Sourced from the
#: redaction set the metadata writer already uses, so there is one list.
def _secret_fields() -> frozenset[str]:
    from bioaccx.report import _SECRET_DATASET_FIELDS
    return _SECRET_DATASET_FIELDS


def _field_docs(cls: type) -> dict[str, str]:
    """Field name → the ``#`` comments written above or beside it in the source.

    Comments are the only per-field documentation the config dataclasses carry,
    so the form reads them rather than duplicating the text. A field with no
    comment simply has no help.
    """
    try:
        src = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):  # pragma: no cover - source not available
        return {}

    lines = src.splitlines()
    docs: dict[str, str] = {}
    for node in ast.parse(src).body[0].body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        # Comment lines directly above the field, nearest last.
        above: list[str] = []
        i = node.lineno - 2
        while i >= 0 and lines[i].strip().startswith("#"):
            above.insert(0, lines[i].strip().lstrip("#").strip())
            i -= 1
        # A trailing comment on the field's own line.
        own = lines[node.lineno - 1]
        inline = ""
        if "#" in own:
            inline = own.split("#", 1)[1].strip()
        parts = [p for p in (" ".join(above).strip(), inline) if p]
        if parts:
            docs[node.target.id] = " — ".join(parts)
    return docs


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """Strip ``None`` from a union, returning ``(inner_type, is_optional)``."""
    origin = typing.get_origin(tp)
    if origin is not Union and not isinstance(tp, types.UnionType):
        return tp, False
    args = [a for a in typing.get_args(tp) if a is not type(None)]
    if len(args) == len(typing.get_args(tp)):
        return tp, False
    return (args[0] if len(args) == 1 else Union[tuple(args)]), True


def _widget_for(tp: Any) -> dict[str, Any]:
    """Map a resolved type annotation onto a widget descriptor.

    Anything unrecognised becomes a ``raw`` widget — a YAML text field — so an
    exotic future type degrades to something editable instead of breaking the
    form.
    """
    inner, optional = _unwrap_optional(tp)
    origin = typing.get_origin(inner)

    if origin is Literal:
        return {"widget": "select", "choices": list(typing.get_args(inner)),
                "optional": optional}

    if dataclasses.is_dataclass(inner):
        return {"widget": "nested", "dataclass": inner.__name__, "optional": optional}

    if inner is bool:
        return {"widget": "checkbox", "optional": optional}
    if inner is int:
        return {"widget": "number", "step": "1", "optional": optional}
    if inner is float:
        return {"widget": "number", "step": "any", "optional": optional}
    if inner is str:
        return {"widget": "text", "optional": optional}

    if origin in (list, typing.List):
        (item,) = typing.get_args(inner) or (str,)
        return {"widget": "tags", "item": getattr(item, "__name__", "str"),
                "optional": optional}

    if origin in (dict, typing.Dict):
        key_t, val_t = typing.get_args(inner) or (str, str)
        if typing.get_origin(val_t) in (list, typing.List):
            return {"widget": "groups", "optional": optional}
        return {"widget": "raw", "optional": optional}

    # Unions that survived the None strip: the two hand-shaped cases in the
    # config, then a raw fallback.
    args = set(typing.get_args(inner))
    if args == {str, list[str]}:
        return {"widget": "paths", "optional": optional}
    if args == {float, list[float]}:
        return {"widget": "freq", "optional": optional}

    return {"widget": "raw", "optional": optional}


def _default_of(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return None


def _fields_of(cls: type, section: str) -> list[dict]:
    """Descriptors for every public field of *cls*, in declaration order."""
    hints = typing.get_type_hints(cls, vars(cfg_mod))
    docs = _field_docs(cls)
    secrets = _secret_fields()
    common = COMMON_FIELDS.get(section, frozenset())

    out: list[dict] = []
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue  # resolved internals such as _dataset_blocks
        descriptor = {
            "name": f.name,
            "path": f"{section}.{f.name}",
            "label": f.name.replace("_", " "),
            "default": _default_of(f),
            "required": f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING,  # type: ignore[misc]
            "help": docs.get(f.name, ""),
            "secret": f.name in secrets,
            "advanced": f.name not in common,
            "run_level": section == "dataset" and f.name in RUN_LEVEL_DATASET_FIELDS,
            **_widget_for(hints[f.name]),
        }
        out.append(descriptor)
    return out


def _registry_choices() -> list[dict]:
    """Registry IDs with enough detail to choose between them in a drop-down."""
    return [
        {
            "id": e["id"],
            "label": f"{e['id']} — {e['name'].capitalize()} v{e['version']} "
                     f"({e['format'].upper()})",
            "description": e.get("description", ""),
            "defaults": {k: v for k, v in e.items() if k not in ("id", "description")},
        }
        for e in registry_entries()
    ]


def build_schema() -> dict:
    """The complete form schema: sections, fields, and the synthetic keys.

    >>> schema = build_schema()
    >>> [s["name"] for s in schema["sections"]]
    ['foundation_model', 'dataset', 'training', 'output', 'umap']
    >>> any(f["name"] == "registry_id" for f in schema["sections"][0]["fields"])
    True
    """
    foundation = _fields_of(FoundationModelConfig, "foundation_model")
    # registry_id is consumed by the loader before the dataclass is built, but
    # it is the shorthand that fills every other field here — so it leads.
    foundation.insert(0, {
        "name": "registry_id",
        "path": "foundation_model.registry_id",
        "label": "registry id",
        "widget": "registry",
        "default": None,
        "optional": True,
        "required": False,
        "secret": False,
        "advanced": False,
        "run_level": False,
        "help": "Load every default for a known backbone from one line. Any field "
                "set alongside it overrides the registry default.",
        "choices": _registry_choices(),
    })

    dataset = _fields_of(DatasetConfig, "dataset")
    dataset.append({
        "name": "sources",
        "path": "dataset.sources",
        "label": "sources",
        "widget": "sources",
        "default": None,
        "optional": True,
        "required": False,
        "secret": False,
        "advanced": False,
        "run_level": False,
        "help": "Combine several heterogeneous sources in one run. Each source "
                "inherits the fields above and overrides them with its own. "
                "Run-level fields are never overridden per source.",
    })

    return {
        "sections": [
            {"name": "foundation_model", "title": "Foundation model",
             "help": "The backbone and the audio window it expects.",
             "fields": foundation},
            {"name": "dataset", "title": "Dataset",
             "help": "Where the audio comes from, how it is labelled, "
                     "preprocessed, augmented and split.",
             "fields": dataset},
            {"name": "training", "title": "Training",
             "help": "Which classifier head to fit, and how.",
             "fields": _fields_of(TrainingConfig, "training"),
             "groups": [
                 {"name": "keras", "title": "Keras head",
                  "fields": _fields_of(KerasConfig, "training.keras")},
                 {"name": "sklearn", "title": "sklearn head",
                  "fields": _fields_of(SklearnConfig, "training.sklearn")},
             ]},
            {"name": "output", "title": "Output",
             "help": "Where results are written, in which formats and precisions.",
             "fields": _fields_of(OutputConfig, "output")},
            {"name": "umap", "title": "UMAP",
             "help": "Projection settings for the embeddings command. Requires "
                     "the [umap] extra.",
             "fields": _fields_of(UmapConfig, "umap")},
        ],
        "nested": {
            "AugmentationConfig": {
                "title": "Augmentation",
                "fields": _fields_of(AugmentationConfig, "dataset.augmentation"),
            },
        },
        "run_level_fields": sorted(RUN_LEVEL_DATASET_FIELDS),
    }


def known_widgets() -> set[str]:
    """Widget names the front end implements. Used by the schema test."""
    return {"text", "number", "checkbox", "select", "tags", "nested",
            "groups", "paths", "freq", "registry", "sources", "raw"}


#: Every config dataclass the schema walks, for tests that assert coverage.
CONFIG_DATACLASSES = (
    FoundationModelConfig, DatasetConfig, AugmentationConfig, TrainingConfig,
    KerasConfig, SklearnConfig, OutputConfig, UmapConfig, BioaccxConfig,
)
