"""Keras dense classifier trainer."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from bioaccx.config import KerasConfig

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Maps output_activation → (activation_fn, loss_key, from_logits)
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
        except RuntimeError:
            pass  # already enabled, or called after TF ops — best-effort
        print(f"  Keras seed: {seed} (deterministic ops enabled)")

    activation_key = cfg.output_activation
    if activation_key not in _ACTIVATION_LOSS:
        raise ValueError(
            f"output_activation must be None, 'sigmoid', or 'softmax'; got {activation_key!r}"
        )
    activation_fn, loss_key, from_logits = _ACTIVATION_LOSS[activation_key]
    activation_label = activation_key or "linear (logits)"

    num_classes = len(label_names)
    embed_dim = X_train.shape[1]
    warmup_epochs = max(3, cfg.epochs // 10)
    rng = np.random.default_rng(seed)

    print(f"\n=== Keras classifier  [activation={activation_label}  loss={loss_key}] ===")

    # One-hot encode labels
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
            kernel_initializer="he_normal",
            name="hidden",
        )(x)
    x = Dropout(cfg.dropout)(x)
    x = Dense(
        num_classes,
        activation=None,
        kernel_regularizer=regularizers.l2(1e-5),
        kernel_initializer="glorot_uniform",
        name="scores",
    )(x)
    out = Activation(activation_fn, name="output_activation")(x) if activation_fn else x
    model = Model(inp, out)

    # --- Loss ---
    if cfg.focal_loss:
        loss_fn = _make_focal_loss(cfg.focal_loss_gamma, cfg.focal_loss_alpha)
        print(f"  Focal loss (gamma={cfg.focal_loss_gamma}, alpha={cfg.focal_loss_alpha})")
    elif "categorical" in loss_key:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    # --- Metrics: always track AUPRC and AUROC ---
    auc_kwargs = dict(from_logits=from_logits, multi_label=False)
    metrics = [
        "accuracy",
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
        loss=loss_key,
        normalize_embeddings=cfg.normalize_embeddings,
        focal_loss=cfg.focal_loss,
        label_smoothing=cfg.label_smoothing,
        mixup=cfg.mixup,
        upsampling_ratio=cfg.upsampling_ratio,
    )
    return model


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
