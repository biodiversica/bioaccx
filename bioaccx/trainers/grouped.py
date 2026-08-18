"""Grouped softmax head: exclusive within a group, independent across groups.

A grouped head partitions its outputs into disjoint groups.  Each group holds
its member labels plus a synthetic ``<group>_none`` column, and a softmax is
applied over each group separately.  Two members of the same group can
therefore never both score high — their probabilities sum to at most 1 — while
members of *different* groups are free to fire together in the same window.

The motivating case is one species with several distinct call types: the calls
of a species are mutually exclusive within a 5 s window, but two species can
easily overlap.  A flat softmax cannot express that (all outputs sum to 1, so
two species compete), and a sigmoid head cannot express the exclusivity.

Training labels not named in any group are *background*: they get no output
column and instead supply the ``none`` target for every group, so a background
clip teaches every group "absent" at once.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# Suffix appended to a group name to form its "nothing from this group" column.
NONE_SUFFIX = "_none"

__all__ = [
    "NONE_SUFFIX",
    "build_group_space",
    "grouped_targets",
    "grouped_predictions",
    "background_labels",
    "grouped_softmax_layer",
    "make_grouped_loss",
    "make_grouped_accuracy",
]


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------

def build_group_space(
    label_groups: Mapping[str, Sequence[str]],
    label_names: Sequence[str],
) -> tuple[list[str], list[tuple[int, int]]]:
    """Return ``(output_labels, slices)`` for a grouped head.

    *output_labels* lists every output column in model order: each group's
    members followed by its ``<group>_none`` column.  *slices* holds one
    ``(start, end)`` half-open range per group, indexing into *output_labels*.

    Group order follows ``label_groups`` insertion order (i.e. the order the
    groups appear in the config), so the layout is reproducible across runs.

    Raises ValueError on an empty group, a member missing from the training
    data, a member claimed by two groups, or a ``_none`` column that collides
    with a real training label.
    """
    if not label_groups:
        raise ValueError("label_groups is empty — nothing to group")

    known = set(label_names)
    owner: dict[str, str] = {}
    output_labels: list[str] = []
    slices: list[tuple[int, int]] = []
    offset = 0

    for group, members in label_groups.items():
        members = list(members)
        if not members:
            raise ValueError(f"label group {group!r} is empty")

        for member in members:
            if member not in known:
                raise ValueError(
                    f"label group {group!r} references {member!r}, which is not present in the "
                    f"training data (found: {', '.join(sorted(known))})"
                )
            if member in owner:
                raise ValueError(
                    f"label {member!r} is claimed by both group {owner[member]!r} and {group!r}; "
                    f"groups must be disjoint"
                )
            owner[member] = group

        none_column = f"{group}{NONE_SUFFIX}"
        if none_column in known:
            raise ValueError(
                f"group {group!r} would emit a column named {none_column!r}, which collides with a "
                f"training label of the same name — rename the group or the label"
            )

        output_labels += members + [none_column]
        slices.append((offset, offset + len(members) + 1))
        offset += len(members) + 1

    return output_labels, slices


def background_labels(
    label_groups: Mapping[str, Sequence[str]],
    label_names: Sequence[str],
) -> list[str]:
    """Training labels that belong to no group — the ones driving every ``none``."""
    grouped = {m for members in label_groups.values() for m in members}
    return [n for n in label_names if n not in grouped]


def grouped_targets(
    y: np.ndarray,
    label_names: Sequence[str],
    label_groups: Mapping[str, Sequence[str]],
    slices: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Class indices → G-hot targets, exactly one active column per group.

    A sample whose label is not in a given group activates that group's ``none``
    column, so every sample supplies a complete target for every group.

    The result doubles as the one-vs-rest positive matrix used for reporting:
    column *j* is 1 exactly when sample *i* counts as a positive for output *j*.
    """
    y = np.asarray(y).astype(int)
    members_per_group = [list(m) for m in label_groups.values()]
    targets = np.zeros((len(y), slices[-1][1]), dtype=np.float32)

    for i, class_index in enumerate(y):
        name = label_names[class_index]
        for (start, _), members in zip(slices, members_per_group):
            offset = members.index(name) if name in members else len(members)
            targets[i, start + offset] = 1.0

    return targets


def grouped_predictions(scores: np.ndarray, slices: Sequence[tuple[int, int]]) -> np.ndarray:
    """Per-group argmax as ``(n_samples, n_groups)`` of *global* column indices.

    Each group votes independently, so a row may name a call from several
    groups at once — that is the whole point of the layout.
    """
    scores = np.asarray(scores)
    return np.stack(
        [start + np.argmax(scores[:, start:end], axis=1) for start, end in slices],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Keras pieces (TensorFlow imported lazily — importing this module stays cheap)
# ---------------------------------------------------------------------------

_LAYER_CLASS = None


def _layer_class():
    """Define (once) and return the serializable GroupedSoftmax layer class.

    Built lazily so that importing this module does not pull in TensorFlow, and
    cached so the Keras serialization registry is not written to twice.
    """
    global _LAYER_CLASS
    if _LAYER_CLASS is not None:
        return _LAYER_CLASS

    import tensorflow as tf

    try:                                            # Keras 3
        from keras.saving import register_keras_serializable
    except ImportError:                             # Keras 2 / tf.keras
        from tensorflow.keras.utils import register_keras_serializable

    @register_keras_serializable(package="bioaccx")
    class GroupedSoftmax(tf.keras.layers.Layer):
        """Softmax over each group slice independently, then concatenate.

        Exports to a plain ``Slice``/``Softmax``/``Concat`` subgraph in both
        TFLite and ONNX — no custom operator is needed at inference time.
        """

        def __init__(self, slices, **kwargs):
            super().__init__(**kwargs)
            self.slices = [tuple(s) for s in slices]

        def call(self, inputs):
            return tf.concat(
                [tf.nn.softmax(inputs[:, start:end]) for start, end in self.slices],
                axis=-1,
            )

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            return {**super().get_config(), "slices": [list(s) for s in self.slices]}

    _LAYER_CLASS = GroupedSoftmax
    return _LAYER_CLASS


def grouped_softmax_layer(slices, name: str = "grouped_softmax"):
    """Instantiate the grouped-softmax activation layer for *slices*."""
    return _layer_class()([tuple(s) for s in slices], name=name)


def make_grouped_loss(slices: Sequence[tuple[int, int]]):
    """Sum of per-group categorical cross-entropies.

    Deliberately not ``CategoricalCrossentropy``: that loss renormalises by
    ``sum(y_pred)``, which for a G-group output is always exactly G.  The
    gradients are unaffected (it is a constant), but every reported loss would
    be inflated by ``G * log(G)`` — 3.296 for three groups — which makes
    val_loss incomparable with any other head.
    """
    import tensorflow as tf

    slices = [tuple(s) for s in slices]

    def grouped_crossentropy(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        return tf.add_n([
            -tf.reduce_sum(
                y_true[:, start:end] * tf.math.log(tf.clip_by_value(y_pred[:, start:end], epsilon, 1.0)),
                axis=-1,
            )
            for start, end in slices
        ])

    return grouped_crossentropy


def make_grouped_accuracy(slices: Sequence[tuple[int, int]]):
    """Mean per-group top-1 accuracy, named ``accuracy``.

    Keeps the ``accuracy``/``val_accuracy`` history keys that the training
    report table already expects.  A plain argmax over the whole output vector
    would be meaningless here, since each group carries its own maximum.
    """
    import tensorflow as tf

    slices = [tuple(s) for s in slices]

    def accuracy(y_true, y_pred):
        per_group = [
            tf.cast(
                tf.equal(
                    tf.argmax(y_true[:, start:end], axis=-1),
                    tf.argmax(y_pred[:, start:end], axis=-1),
                ),
                tf.float32,
            )
            for start, end in slices
        ]
        return tf.reduce_mean(tf.stack(per_group, axis=-1), axis=-1)

    return accuracy
