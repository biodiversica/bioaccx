"""Supervision of one bioaccx run, started as a subprocess.

The GUI does not import :mod:`bioaccx.train`; it runs the same command line a
user would have typed and reads its stdout. That buys four things for free:

* ``train.py`` prints rather than logs (107 bare ``print`` calls, no ``logging``
  import), so a subprocess is the only way to stream progress without
  restructuring it;
* cancelling is ``SIGINT``, which the CLI already turns into a clean
  "Interrupted." exit;
* TensorFlow never loads in the server process, which stays small and fast;
* a crashed run cannot take the editor down — its exit code is just data.

The child is started in its own session, so it survives the server being
restarted or killed. What is lost in that case is the buffered log tail, not
the run: it keeps writing to ``custom_models/`` either way.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

#: Commands the editor is allowed to launch. The model-surgery commands take
#: their inputs as paths rather than a config, and `gui` would be recursive.
RUNNABLE = ("train", "dataset", "embeddings", "validate")

#: How many output lines are kept for the browser. Keras prints a progress bar
#: per epoch, so a long run produces far more than anyone will read; the tail is
#: what matters and the full output is the terminal's job.
LOG_LINES = 4000

#: `[3/5] Extracting embeddings…` — the step markers train.py already prints.
#: Sub-steps (`[2b]`) and `[warning]` deliberately do not match, and neither do
#: the indented per-sample lines (`  [876/3154] rec.wav …`).
STEP_RE = re.compile(r"^\[(\d+)([a-z]?)/(\d+)\]\s*(.*?)\s*$")


class JobError(Exception):
    """A run could not be started."""


@dataclass
class Job:
    """A single run, and everything the browser is told about it."""

    id: str
    command: str
    argv: list[str]
    config: str
    started_at: str
    status: str = "running"          # running | done | failed | cancelled
    returncode: Optional[int] = None
    finished_at: Optional[str] = None
    step: Optional[int] = None
    step_total: Optional[int] = None
    step_name: str = ""
    pid: Optional[int] = None
    _started: float = field(default=0.0, repr=False)
    _ended: float = field(default=0.0, repr=False)

    def as_dict(self) -> dict:
        end = self._ended or time.monotonic()
        return {
            "id": self.id,
            "command": self.command,
            "config": self.config,
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": round(end - self._started, 1) if self._started else 0.0,
            "step": self.step,
            "step_total": self.step_total,
            "step_name": self.step_name,
            "pid": self.pid,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobRunner:
    """Runs one bioaccx command at a time and buffers its output.

    Concurrent training on one machine competes for the same CPU, GPU and
    output directory, so a second run is refused rather than queued — the
    queue, if one is wanted, is the user's shell.
    """

    def __init__(self, *, log_lines: int = LOG_LINES) -> None:
        self._lock = threading.Lock()
        self._lines: deque[tuple[int, str]] = deque(maxlen=log_lines)
        self._cursor = 0
        self._job: Optional[Job] = None
        self._process: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None

    # ── state ────────────────────────────────────────────────────────────

    @property
    def job(self) -> Optional[Job]:
        return self._job

    def is_running(self) -> bool:
        return self._job is not None and self._job.status == "running"

    def lines_since(self, cursor: int) -> tuple[int, list[str]]:
        """Buffered lines newer than *cursor*, with the new cursor.

        A client that falls far behind simply misses the dropped lines; the
        cursor it gets back is the newest, so it resynchronises rather than
        replaying a buffer that has already rolled over.
        """
        with self._lock:
            new = [text for index, text in self._lines if index > cursor]
            return self._cursor, new

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, command: str, config: Path) -> Job:
        """Launch ``bioaccx <command> <config>`` and start reading its output."""
        if command not in RUNNABLE:
            raise JobError(
                f"{command!r} cannot be launched from the editor "
                f"(runnable: {', '.join(RUNNABLE)})"
            )
        if self.is_running():
            raise JobError("a run is already in progress; cancel it first")
        if not config.exists():
            raise JobError(f"no such config file: {config}")

        argv = [sys.executable, "-m", "bioaccx", command, str(config)]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                # The working directory is inherited, not set to the config's
                # folder: relative paths in a config — `output_path:
                # ./custom_models` — are written relative to where bioaccx was
                # started, so running one that lives in custom_models/ from its
                # own folder would nest the results a level deeper. Inheriting
                # makes a run land exactly where the shown command would put it
                # if it were typed in the terminal that started the editor.
                # Its own session, so SIGINT reaches the child alone and the run
                # outlives the server that started it.
                start_new_session=True,
            )
        except OSError as exc:
            raise JobError(f"could not start {command}: {exc}") from exc

        job = Job(
            id=f"{int(time.time())}-{process.pid}",
            command=f"bioaccx {command} {config}",
            argv=argv,
            config=str(config),
            started_at=_now(),
            pid=process.pid,
        )
        job._started = time.monotonic()

        with self._lock:
            self._lines.clear()
            self._cursor = 0
            self._job = job
            self._process = process

        self._append(f"$ bioaccx {command} {config}")
        self._reader = threading.Thread(
            target=self._read, args=(process, job), daemon=True,
            name=f"bioaccx-job-{process.pid}")
        self._reader.start()
        return job

    def cancel(self) -> bool:
        """Interrupt the running job the way Ctrl+C would."""
        with self._lock:
            process, job = self._process, self._job
        if process is None or job is None or job.status != "running":
            return False
        try:
            # The child owns its process group, so this is the same signal the
            # CLI's KeyboardInterrupt handler already knows how to exit on.
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (OSError, ProcessLookupError):
            return False
        job.status = "cancelled"
        return True

    # ── internals ────────────────────────────────────────────────────────

    def _append(self, text: str) -> None:
        with self._lock:
            self._cursor += 1
            self._lines.append((self._cursor, text))

    def _read(self, process: subprocess.Popen, job: Job) -> None:
        """Drain the child's output until it exits, then record how it ended."""
        try:
            if process.stdout is not None:
                for raw in process.stdout:
                    line = raw.rstrip("\n")
                    self._append(line)
                    self._note_step(line, job)
        finally:
            returncode = process.wait()
            job.returncode = returncode
            job.finished_at = _now()
            job._ended = time.monotonic()
            if job.status != "cancelled":
                job.status = "done" if returncode == 0 else "failed"
            self._append(
                f"[{job.status}] exit code {returncode} "
                f"after {job.as_dict()['elapsed']}s"
            )

    @staticmethod
    def _note_step(line: str, job: Job) -> None:
        """Track `[n/total] name` markers for a coarse progress indicator.

        Deliberately dumb: if a marker is reworded or renumbered the UI falls
        back to a plain log tail rather than breaking.
        """
        match = STEP_RE.match(line.rstrip())
        if not match:
            return
        job.step = int(match.group(1))
        job.step_total = int(match.group(3))
        job.step_name = match.group(4).rstrip("…").strip()
