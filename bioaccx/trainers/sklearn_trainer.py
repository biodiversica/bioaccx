"""Sklearn LogisticRegression classifier trainer."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bioaccx.config import SklearnConfig


def train_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    cfg: SklearnConfig,
) -> Pipeline:
    print("\n=== Sklearn LogisticRegression ===")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, solver=cfg.solver)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_names,
                                labels=list(range(len(label_names))), zero_division=0))
    return pipe
