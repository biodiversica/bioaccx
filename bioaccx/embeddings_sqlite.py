"""Thread-safe SQLite-backed embedding cache."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np

_locks: dict[str, threading.Lock] = {}
_locks_meta = threading.Lock()


def _lock_for(db_path: Path) -> threading.Lock:
    key = str(db_path.resolve())
    with _locks_meta:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _init_db(con: sqlite3.Connection) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS embeddings "
        "(key TEXT PRIMARY KEY, data BLOB NOT NULL)"
    )
    con.commit()


def init_db(db_path: Path) -> None:
    """Create the embeddings table if it does not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        _init_db(con)


def load_embedding(db_path: Path, key: str) -> Optional[np.ndarray]:
    """Return the embedding for *key*, or None if not present."""
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT data FROM embeddings WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).copy()


def save_embedding(db_path: Path, key: str, emb: np.ndarray) -> None:
    """Insert or replace the embedding for *key* (thread-safe)."""
    data = emb.astype(np.float32).tobytes()
    lock = _lock_for(db_path)
    with lock:
        with sqlite3.connect(db_path) as con:
            _init_db(con)
            con.execute(
                "INSERT OR REPLACE INTO embeddings (key, data) VALUES (?, ?)",
                (key, data),
            )
            con.commit()
