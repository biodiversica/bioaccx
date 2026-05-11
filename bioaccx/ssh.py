"""SSH/SFTP helpers for on-demand remote audio access.

When ``ssh_host`` is set in DatasetConfig:
  - Load time  : remote dirs are listed and audio durations are read via
                 in-memory SFTP reads (no files written to disk).
  - Embed time : each source file is downloaded to a temp file just before its
                 chunks are embedded, then deleted once the last chunk is done.
                 Only one copy of each source file exists on disk at a time.
"""
from __future__ import annotations

import io  # used by read_remote_text
import stat
import tempfile
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import paramiko


def open_sftp_client(
    host: str,
    user: str,
    port: int = 22,
    key_path: Optional[str] = None,
) -> tuple["paramiko.SSHClient", "paramiko.SFTPClient"]:
    """Open an SSH connection and return (SSHClient, SFTPClient).

    Caller is responsible for closing both when done.
    Authentication is key-based only; if *key_path* is None, paramiko falls
    back to the SSH agent and default key locations (~/.ssh/id_rsa, etc.).
    """
    try:
        import paramiko
    except ImportError as exc:
        raise ImportError(
            "paramiko is required for SSH data_dir access. "
            "Install it with: pip install paramiko"
        ) from exc

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs: dict = {"hostname": host, "username": user, "port": port}
    if key_path:
        connect_kwargs["key_filename"] = key_path

    client.connect(**connect_kwargs)
    sftp = client.open_sftp()
    return client, sftp


def list_remote_audio(
    sftp,
    remote_dir: str,
    exts: frozenset[str],
) -> list[tuple[str, str, Optional[str]]]:
    """Recursively list audio files under *remote_dir*, inferring labels from subfolders.

    Supports the same two layouts as ``_load_subfolders``:
      - Pre-split : ``remote_dir/train/<label>/file.wav``
      - Flat      : ``remote_dir/<label>/file.wav``

    Returns a list of ``(remote_path, label, split)`` tuples.
    ``split`` is ``"train"`` / ``"test"`` for the pre-split layout, else ``None``.
    """
    results: list[tuple[str, str, Optional[str]]] = []

    try:
        top_entries = sftp.listdir_attr(remote_dir)
    except Exception as exc:
        print(f"  [ssh] cannot list {remote_dir}: {exc}")
        return results

    top_names = {e.filename for e in top_entries}
    has_split = "train" in top_names and "test" in top_names

    if has_split:
        for split_name in ("train", "test"):
            split_dir = f"{remote_dir}/{split_name}"
            try:
                class_entries = sftp.listdir_attr(split_dir)
            except Exception as exc:
                print(f"  [ssh] cannot list {split_dir}: {exc}")
                continue
            for ce in sorted(class_entries, key=lambda e: e.filename):
                if not stat.S_ISDIR(ce.st_mode):
                    continue
                _collect_audio_flat(
                    sftp, f"{split_dir}/{ce.filename}",
                    ce.filename, split_name, exts, results,
                )
    else:
        for ce in sorted(top_entries, key=lambda e: e.filename):
            if not stat.S_ISDIR(ce.st_mode):
                continue
            _collect_audio_flat(
                sftp, f"{remote_dir}/{ce.filename}",
                ce.filename, None, exts, results,
            )

    return results


def _collect_audio_flat(
    sftp,
    remote_dir: str,
    label: str,
    split: Optional[str],
    exts: frozenset[str],
    results: list[tuple[str, str, Optional[str]]],
) -> None:
    """Append (remote_path, label, split) for each audio file in *remote_dir*."""
    try:
        entries = sftp.listdir_attr(remote_dir)
    except Exception as exc:
        print(f"  [ssh] cannot list {remote_dir}: {exc}")
        return
    for e in sorted(entries, key=lambda e: e.filename):
        if stat.S_ISREG(e.st_mode) and Path(e.filename).suffix.lower() in exts:
            results.append((f"{remote_dir}/{e.filename}", label, split))


def get_audio_duration(sftp, remote_path: str) -> float:
    """Return the duration (seconds) of a remote audio file.

    Downloads to a temp file and reads duration via soundfile using a plain
    file path (native C I/O).  Using a BytesIO buffer instead would trigger
    cffi virtual-I/O callbacks which fail when the local libffi version does
    not match the one soundfile was compiled against.  The temp file is deleted
    immediately after the header is read.
    """
    import soundfile as sf

    tmp = download_to_temp(sftp, remote_path)
    try:
        return sf.info(str(tmp)).duration
    finally:
        tmp.unlink(missing_ok=True)


def list_remote_file_per_label(
    sftp,
    remote_dir: str,
    exts: frozenset[str],
) -> list[tuple[str, str]]:
    """List paired (audio_path, txt_path) from a flat remote dir.

    Files without a matching .txt sibling are skipped with a warning.
    """
    try:
        entries = sftp.listdir_attr(remote_dir)
    except Exception as exc:
        print(f"  [ssh] cannot list {remote_dir}: {exc}")
        return []

    filenames = {e.filename for e in entries if stat.S_ISREG(e.st_mode)}
    pairs: list[tuple[str, str]] = []

    for e in sorted(entries, key=lambda e: e.filename):
        if not stat.S_ISREG(e.st_mode):
            continue
        if Path(e.filename).suffix.lower() not in exts:
            continue
        txt_name = Path(e.filename).stem + ".txt"
        if txt_name in filenames:
            pairs.append((f"{remote_dir}/{e.filename}", f"{remote_dir}/{txt_name}"))
        else:
            print(f"  [ssh] no label file for {e.filename} — ignoring")

    return pairs


def read_remote_text(sftp, remote_path: str) -> str:
    """Download a small remote text file and return its contents as a string."""
    buf = io.BytesIO()
    sftp.getfo(remote_path, buf)
    return buf.getvalue().decode()


def download_to_temp(sftp, remote_path: str) -> Path:
    """Download *remote_path* to a temporary local file and return its path.

    The caller is responsible for deleting the file when done.
    """
    suffix = Path(remote_path).suffix
    fd, tmp_str = tempfile.mkstemp(suffix=suffix, prefix="bioaccx_ssh_")
    import os
    os.close(fd)
    sftp.get(remote_path, tmp_str)
    return Path(tmp_str)
