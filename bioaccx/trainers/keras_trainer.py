"""Keras dense classifier trainer."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from bioaccx.config import KerasConfig

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Maps output_activation → (activation_fn, loss, from_logits)
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
):
    """Train a dense Keras classifier head on top of pre-computed embeddings.

    Architecture (input → normalization → optional hidden → output):
      - A Normalization layer is adapted on X_train (Z-score per feature).
      - An optional hidden Dense(relu) layer with L2 regularisation.
      - A linear Dense output followed by an optional activation layer.

    Training uses:
      - EarlyStopping on val_loss (restores best weights).
      - Cosine LR schedule with a linear warmup phase.

    The returned model has two extra attributes attached:
      - ``_report_history``: the Keras history dict for report writing.
      - ``_report_params``: hyperparameter dict for report writing.

    Parameters
    ----------
    X_train, X_test:
        Float32 embedding arrays of shape (n_samples, embedding_size).
    y_train, y_test:
        Integer class indices in [0, len(label_names)).
    label_names:
        Ordered list of class names; index i corresponds to y == i.
    cfg:
        Keras training hyperparameters.
    """
    import tensorflow as tf
    from sklearn.metrics import classification_report
    from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
    from tensorflow.keras.layers import Activation, Dense, Dropout, Input, Normalization
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers

    activation_key = cfg.output_activation
    if activation_key not in _ACTIVATION_LOSS:
        raise ValueError(
            f"output_activation must be None, 'sigmoid', or 'softmax'; got {activation_key!r}"
        )
    activation_fn, loss_fn, from_logits = _ACTIVATION_LOSS[activation_key]
    activation_label = activation_key or "linear (logits)"

    print(f"\n=== Keras classifier  [activation={activation_label}  loss={loss_fn}] ===")
    num_classes = len(label_names)
    embed_dim = X_train.shape[1]
    warmup_epochs = max(3, cfg.epochs // 10)

    # One-hot encode labels for CategoricalCrossentropy.
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes).astype(np.float32)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  num_classes).astype(np.float32)

    # Compute per-feature mean/variance from training data only.
    norm = Normalization(axis=-1)
    norm.adapt(X_train)

    inp = Input(shape=(embed_dim,), name="embedding")
    x = norm(inp)
    # The hidden layer is optional; setting hidden_units=0 creates a linear model.
    if cfg.hidden_units > 0:
        x = Dropout(cfg.dropout)(x)
        x = Dense(
            cfg.hidden_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-5),
            name="hidden",
        )(x)
    x = Dropout(cfg.dropout)(x)
    # Always output raw logits from Dense so that the loss can apply from_logits=True
    # when no explicit activation is configured (numerically more stable).
    x = Dense(
        num_classes,
        activation=None,
        kernel_regularizer=regularizers.l2(1e-5),
        name="scores",
    )(x)
    # Optional activation layer after the linear Dense (e.g. sigmoid as in BirdNET-Analyzer)
    out = Activation(activation_fn, name="output_activation")(x) if activation_fn else x
    model = Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
              if "categorical" in loss_fn
              else tf.keras.losses.BinaryCrossentropy(from_logits=from_logits),
        metrics=["accuracy"],
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
        X_train, y_train_oh,
        validation_data=(X_test, y_test_oh),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"  Best epoch: {best_epoch}/{cfg.epochs}")

    # For logits, apply softmax before argmax to get class predictions.
    raw = model.predict(X_test, verbose=0)
    y_pred = np.argmax(tf.nn.softmax(raw).numpy() if activation_fn is None else raw, axis=1)
    print(classification_report(y_test, y_pred, target_names=label_names,
                                labels=list(range(len(label_names))), zero_division=0))

    # Attach metadata so the report writer can read them without re-fitting.
    model._report_history = history.history
    model._report_params = dict(
        hidden_units=cfg.hidden_units,
        dropout=cfg.dropout,
        epochs=cfg.epochs,
        best_epoch=best_epoch,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        output_activation=activation_label,
        loss=loss_fn,
    )
    return model


def _cosine_schedule(base_lr: float, total_epochs: int, warmup_epochs: int):
    """Return a LearningRateScheduler-compatible function with warmup + cosine decay.

    During warmup, LR rises linearly from 0 to base_lr over warmup_epochs.
    Afterwards, LR decays from base_lr to 10% of base_lr following a cosine
    curve, ensuring a smooth tail rather than an abrupt stop.
    """
    def schedule(epoch: int, _lr: float) -> float:
        if epoch < warmup_epochs:
            # Linear warmup: avoids large gradient updates at the start of training.
            return base_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        # Cosine decay: floor at 10% of base_lr so the network keeps learning slowly.
        return base_lr * (0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2)
    return schedule
