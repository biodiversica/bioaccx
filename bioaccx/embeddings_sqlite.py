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
    """Return (and lazily create) the per-database write lock.

    SQLite's WAL mode allows concurrent readers but still requires serialised
    writes when multiple threads share the same connection path.  Using one
    lock per resolved path avoids cross-process deadlocks from symlinks or
    relative-path aliases pointing to the same file.
    """
    key = str(db_path.resolve())
    # _locks_meta guards the _locks dict itself so that two threads don't race
    # on creating the per-db lock for the same key.
    with _locks_meta:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _init_db(con: sqlite3.Connection) -> None:
    """Ensure the embeddings table exists on an already-open connection.

    Called both from init_db (standalone setup) and from save_embedding
    (idempotent guard for the first write in a new file).
    """
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
    """Return the embedding for *key*, or None if not present.

    np.frombuffer returns a read-only view over the bytes object; .copy()
    makes it writable so callers can safely modify or stack the array.
    """
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
    """Insert or replace the embedding for *key* (thread-safe).

    The embedding is stored as raw float32 bytes (native endian); load_embedding
    must use the same dtype when deserialising.  _init_db is called inside the
    lock so that the table is guaranteed to exist before the INSERT even when
    multiple threads race to write to a brand-new database file.
    """
    data = emb.astype(np.float32).tobytes()
    lock = _lock_for(db_path)
    # Hold the lock for the full connect + write + commit cycle to prevent
    # concurrent writers from hitting SQLITE_BUSY on the same file.
    with lock:
        with sqlite3.connect(db_path) as con:
            _init_db(con)
            con.execute(
                "INSERT OR REPLACE INTO embeddings (key, data) VALUES (?, ?)",
                (key, data),
            )
            con.commit()
