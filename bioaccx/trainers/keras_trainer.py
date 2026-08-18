"""Keras dense classifier trainer."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from bioaccx.config import KerasConfig
from bioaccx.trainers.grouped import (
    background_labels,
    build_group_space,
    grouped_predictions,
    grouped_softmax_layer,
    grouped_targets,
    make_grouped_accuracy,
    make_grouped_loss,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Maps output_activation → (activation_fn, loss_key, from_logits)
# output_activation value selecting the grouped head (see bioaccx.trainers.grouped)
GROUPED_ACTIVATION = "grouped_softmax"

_ACTIVATION_LOSS = {
    None:        (None,      "categorical_crossentropy", True),
    "sigmoid":   ("sigmoid", "binary_crossentropy",      False),
    "softmax":   ("softmax", "categorical_crossentropy", False),
}


def train_keras(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    cfg: KerasConfig,
    seed: Optional[int] = None,
):
    """Train a dense Keras classifier head on top of pre-computed embeddings.

    Architecture (input → optional normalization → optional hidden → output):
      - Optional Normalization layer adapted on X_train (Z-score per feature).
      - Optional hidden Dense(relu) layer with L2 regularisation.
      - Linear Dense output followed by an optional activation layer.

    Training uses:
      - AUPRC and AUROC tracked as Keras metrics every epoch.
      - EarlyStopping on val_loss (restores best weights).
      - Cosine LR schedule with a linear warmup phase.
      - Optional: focal loss, label smoothing, mixup, upsampling.
    """
    import tensorflow as tf
    from sklearn.metrics import classification_report
    from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
    from tensorflow.keras.layers import Activation, Dense, Dropout, Input, Normalization
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers

    if seed is not None:
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
            print(f"  Keras seed: {seed} (deterministic ops enabled)")
        except RuntimeError as e:
            print(
                f"  Keras seed: {seed} (WARNING: could not enable deterministic ops "
                f"— a TF op already ran earlier in this process, so results may not "
                f"be fully reproducible run-to-run: {e})"
            )

    activation_key = cfg.output_activation
    grouped = bool(cfg.label_groups) or activation_key == GROUPED_ACTIVATION
    if grouped:
        if activation_key not in (None, GROUPED_ACTIVATION):
            raise ValueError(
                f"label_groups is set, so output_activation must be {GROUPED_ACTIVATION!r} "
                f"(or omitted); got {activation_key!r}"
            )
        if not cfg.label_groups:
            raise ValueError(
                f"output_activation={GROUPED_ACTIVATION!r} requires a non-empty label_groups"
            )
        for option in ("label_smoothing", "upsampling_ratio"):
            if getattr(cfg, option):
                raise NotImplementedError(
                    f"{option} is not supported with a grouped softmax head: it assumes one-hot "
                    f"targets, while grouped targets are one-hot per group. Disable it or use a "
                    f"flat head."
                )
        output_labels, group_slices = build_group_space(cfg.label_groups, label_names)
        background = background_labels(cfg.label_groups, label_names)
        activation_fn, loss_key, from_logits = None, "grouped_crossentropy", False
        activation_label = GROUPED_ACTIVATION
        num_classes = len(output_labels)
    elif activation_key not in _ACTIVATION_LOSS:
        raise ValueError(
            f"output_activation must be None, 'sigmoid', 'softmax', or {GROUPED_ACTIVATION!r}; "
            f"got {activation_key!r}"
        )
    else:
        activation_fn, loss_key, from_logits = _ACTIVATION_LOSS[activation_key]
        activation_label = activation_key or "linear (logits)"
        output_labels, group_slices, background = list(label_names), None, []
        num_classes = len(label_names)
    embed_dim = X_train.shape[1]
    warmup_epochs = max(3, cfg.epochs // 10)
    rng = np.random.default_rng(seed)

    print(f"\n=== Keras classifier  [activation={activation_label}  loss={loss_key}] ===")

    # Encode targets: one-hot for a flat head, one-hot *per group* for a grouped one
    if grouped:
        y_train_oh = grouped_targets(y_train, label_names, cfg.label_groups, group_slices)
        y_test_oh  = grouped_targets(y_test,  label_names, cfg.label_groups, group_slices)
        print(f"  Grouped softmax: {len(group_slices)} groups over "
              f"{sum(len(m) for m in cfg.label_groups.values())} labels -> {num_classes} outputs")
        for group, members in cfg.label_groups.items():
            print(f"    {group}: {', '.join(members)}")
        n_bg = int(sum(label_names[int(i)] in set(background) for i in y_train))
        print(f"  Background (-> none in every group): "
              f"{', '.join(background) if background else '(none)'}"
              f"  [{n_bg} train samples, {n_bg / max(len(y_train), 1) * 100:.1f}%]")
    else:
        y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes).astype(np.float32)
        y_test_oh  = tf.keras.utils.to_categorical(y_test,  num_classes).astype(np.float32)

    # --- Optional pre-processing on training data ---
    X_tr, y_tr = X_train.copy(), y_train_oh.copy()

    if cfg.upsampling_ratio > 0:
        X_tr, y_tr = _upsampling(X_tr, y_tr, cfg.upsampling_ratio, cfg.upsampling_mode, rng)
        print(f"  Upsampled training set: {len(X_tr)} samples")

    if cfg.mixup:
        X_tr, y_tr = _mixup(X_tr, y_tr, cfg.mixup_ratio, cfg.mixup_alpha, rng)
        print(f"  Mixup applied (ratio={cfg.mixup_ratio}, alpha={cfg.mixup_alpha})")

    if cfg.label_smoothing:
        y_tr = _label_smoothing(y_tr, cfg.label_smoothing_alpha)
        print(f"  Label smoothing applied (alpha={cfg.label_smoothing_alpha})")

    # --- Build model ---
    inp = Input(shape=(embed_dim,), name="embedding")
    if cfg.normalize_embeddings:
        norm = Normalization(axis=-1)
        norm.adapt(X_train)
        x = norm(inp)
    else:
        x = inp

    if cfg.hidden_units > 0:
        x = Dropout(cfg.dropout)(x)
        x = Dense(
            cfg.hidden_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-5),
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
            name="hidden",
        )(x)
    x = Dropout(cfg.dropout)(x)
    x = Dense(
        num_classes,
        activation=None,
        kernel_regularizer=regularizers.l2(1e-5),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        name="scores",
    )(x)
    if grouped:
        out = grouped_softmax_layer(group_slices, name="output_activation")(x)
    elif activation_fn:
        out = Activation(activation_fn, name="output_activation")(x)
    else:
        out = x
    model = Model(inp, out)

    # --- Loss ---
    if grouped:
        loss_fn = make_grouped_loss(group_slices)
        if cfg.focal_loss:
            raise NotImplementedError("focal_loss is not supported with a grouped softmax head")
    elif cfg.focal_loss:
        loss_fn = _make_focal_loss(cfg.focal_loss_gamma, cfg.focal_loss_alpha)
        print(f"  Focal loss (gamma={cfg.focal_loss_gamma}, alpha={cfg.focal_loss_alpha})")
    elif "categorical" in loss_key:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    # --- Metrics: always track AUPRC and AUROC ---
    auc_kwargs = dict(from_logits=from_logits, multi_label=False)
    metrics = [
        make_grouped_accuracy(group_slices) if grouped else "accuracy",
        tf.keras.metrics.AUC(curve="PR",  name="AUPRC", **auc_kwargs),
        tf.keras.metrics.AUC(curve="ROC", name="AUROC", **auc_kwargs),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss=loss_fn,
        metrics=metrics,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=max(5, cfg.epochs // 10),
            min_delta=1e-3,
            restore_best_weights=True,
        ),
        LearningRateScheduler(
            _cosine_schedule(cfg.learning_rate, cfg.epochs, warmup_epochs), verbose=0
        ),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_test, y_test_oh),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"  Best epoch: {best_epoch}/{cfg.epochs}")

    raw = model.predict(X_test, verbose=0)
    if grouped:
        # Each group is its own single-label problem (members + "none").
        picks = grouped_predictions(raw, group_slices)
        for gi, ((start, end), group) in enumerate(zip(group_slices, cfg.label_groups)):
            names = output_labels[start:end]
            truth = np.argmax(y_test_oh[:, start:end], axis=1)
            print(f"\n--- group: {group} ---")
            print(classification_report(truth, picks[:, gi] - start, target_names=names,
                                        labels=list(range(len(names))), zero_division=0))
    else:
        y_pred = np.argmax(tf.nn.softmax(raw).numpy() if activation_fn is None else raw, axis=1)
        print(classification_report(y_test, y_pred, target_names=label_names,
                                    labels=list(range(len(label_names))), zero_division=0))

    model._report_history = history.history
    model._report_params = dict(
        hidden_units=cfg.hidden_units,
        dropout=cfg.dropout,
        epochs=cfg.epochs,
        best_epoch=best_epoch,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        output_activation=activation_label,
        export_logits=bool(cfg.export_logits and (activation_fn is not None or grouped)),
        label_groups={g: list(m) for g, m in cfg.label_groups.items()} if grouped else {},
        group_slices=[list(s) for s in group_slices] if grouped else [],
        loss=loss_key,
        normalize_embeddings=cfg.normalize_embeddings,
        focal_loss=cfg.focal_loss,
        label_smoothing=cfg.label_smoothing,
        mixup=cfg.mixup,
        upsampling_ratio=cfg.upsampling_ratio,
    )
    return model


def strip_output_activation(model):
    """Return a copy of *model* whose graph ends at the linear ``scores`` layer.

    Mirrors BirdNET-Analyzer's ``classifier.pop()``: the head is trained with
    sigmoid (or softmax) but exported emitting raw logits, leaving the
    activation to the inference code.  The returned model shares weights with
    the original — nothing is retrained or copied.

    Returns *model* unchanged when it has no activation layer to strip (i.e.
    ``output_activation: null``, where the output already is the logits).
    """
    import tensorflow as tf

    try:
        scores = model.get_layer("scores")
    except ValueError:
        return model
    if model.output is scores.output:
        return model            # already linear — nothing to strip

    stripped = tf.keras.Model(model.input, scores.output, name="head_logits")
    stripped._report_history = getattr(model, "_report_history", {})
    stripped._report_params = getattr(model, "_report_params", {})
    return stripped


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _cosine_schedule(base_lr: float, total_epochs: int, warmup_epochs: int):
    def schedule(epoch: int, _lr: float) -> float:
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return base_lr * (0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2)
    return schedule


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _make_focal_loss(gamma: float, alpha: float):
    def focal_loss(y_true, y_pred):
        import tensorflow as tf
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_sum(alpha_factor * focal_weight * cross_entropy, axis=-1)
    return focal_loss


# ---------------------------------------------------------------------------
# Data augmentation helpers
# ---------------------------------------------------------------------------

def _label_smoothing(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    y = y.copy()
    y[y > 0] -= alpha
    y[y == 0] = alpha / y.shape[1]
    return y


def _mixup(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float = 0.25,
    alpha: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    x, y = x.copy(), y.copy()
    positive_indices = np.unique(np.where(y == 1)[0])
    num_to_mix = int(len(positive_indices) * ratio)
    already_mixed: list[int] = []

    for _ in range(num_to_mix):
        idx = int(rng.choice(positive_indices))
        while idx in already_mixed:
            idx = int(rng.choice(positive_indices))

        idx2 = int(rng.choice(positive_indices))
        while idx2 == idx or idx2 in already_mixed:
            idx2 = int(rng.choice(positive_indices))

        lam = float(rng.beta(alpha, alpha))
        x[idx] = lam * x[idx] + (1 - lam) * x[idx2]
        y[idx] = lam * y[idx] + (1 - lam) * y[idx2]
        already_mixed.append(idx)

    return x, y


def _upsampling(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float = 0.5,
    mode: str = "repeat",
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    min_samples = int(np.max(y.sum(axis=0)) * ratio)
    x_extra: list[np.ndarray] = []
    y_extra: list[np.ndarray] = []

    for i in range(y.shape[1]):
        class_indices = np.where(y[:, i] == 1)[0]
        current = len(class_indices)
        needed = min_samples - current
        if needed <= 0:
            continue

        if mode == "repeat":
            chosen = rng.choice(class_indices, size=needed, replace=True)
            x_extra.append(x[chosen])
            y_extra.append(y[chosen])

        elif mode == "mean":
            for _ in range(needed):
                pair = rng.choice(class_indices, size=2, replace=False)
                x_extra.append(np.mean(x[pair], axis=0, keepdims=True))
                y_extra.append(y[pair[:1]])

        elif mode == "linear":
            for _ in range(needed):
                pair = rng.choice(class_indices, size=2, replace=False)
                a = float(rng.uniform(0, 1))
                x_extra.append((a * x[pair[0]] + (1 - a) * x[pair[1]])[np.newaxis])
                y_extra.append(y[pair[:1]])

        elif mode == "smote":
            k = min(5, len(class_indices) - 1)
            if k < 1:
                chosen = rng.choice(class_indices, size=needed, replace=True)
                x_extra.append(x[chosen])
                y_extra.append(y[chosen])
                continue
            for _ in range(needed):
                src = int(rng.choice(class_indices))
                dists = np.linalg.norm(x[class_indices] - x[src], axis=1)
                neighbors = class_indices[np.argsort(dists)[1:k + 1]]
                nb = int(rng.choice(neighbors))
                w = float(rng.uniform(0, 1))
                x_extra.append((x[src] + w * (x[nb] - x[src]))[np.newaxis])
                y_extra.append(y[src:src + 1])

    if x_extra:
        x = np.vstack([x] + x_extra)
        y = np.vstack([y] + y_extra)

    idx = np.arange(len(x))
    rng.shuffle(idx)
    return x[idx], y[idx]
