"""Tests for run supervision.

The runner drives a real subprocess, so the lifecycle tests launch the cheapest
real command there is — ``bioaccx validate`` — rather than mocking the thing
under test. Signal handling is checked against a fake process, since a run slow
enough to interrupt reliably would mean training a model.
"""
from __future__ import annotations

import signal
import time
from pathlib import Path

import pytest

from bioaccx.gui.jobs import STEP_RE, Job, JobError, JobRunner

EXAMPLE = Path(__file__).resolve().parents[2] / "example_config.yaml"


def _wait(runner: JobRunner, timeout: float = 90.0) -> None:
    """Block until the current job finishes, or fail the test."""
    deadline = time.monotonic() + timeout
    while runner.is_running():
        if time.monotonic() > deadline:
            pytest.fail("the job did not finish in time")
        time.sleep(0.05)


@pytest.fixture
def config(tmp_path) -> Path:
    target = tmp_path / "cfg.yaml"
    target.write_text(EXAMPLE.read_text())
    return target


class TestStepMarkers:
    @pytest.mark.parametrize("line,expected", [
        ("[1/5] Validating foundation model…", (1, 5, "Validating foundation model")),
        ("[3/5] Extracting embeddings…", (3, 5, "Extracting embeddings")),
        ("[1/2] Loading dataset…", (1, 2, "Loading dataset")),
    ])
    def test_a_marker_advances_progress(self, line, expected):
        job = Job(id="x", command="c", argv=[], config="c.yaml", started_at="t")
        JobRunner._note_step(line, job)
        assert (job.step, job.step_total, job.step_name) == expected

    @pytest.mark.parametrize("line", [
        "[2b] Skipping dataset export: not supported for SSH data dirs.",
        "[warning] umap.cache_csv set but not found",
        "Epoch 1/50",
        "  UMAP data CSV    → /tmp/x.csv",
        "",
    ])
    def test_other_output_is_not_mistaken_for_a_step(self, line):
        assert STEP_RE.match(line.strip()) is None

    def test_per_sample_lines_are_not_steps(self):
        """Embedding progress is indented `[n/total]`; it must not move the bar."""
        job = Job(id="x", command="c", argv=[], config="c.yaml", started_at="t")
        JobRunner._note_step("[3/5] Extracting embeddings…", job)
        JobRunner._note_step("  [876/3154] rec.wav [0.00s–3.00s]  0.034s", job)
        assert (job.step, job.step_total) == (3, 5)

    def test_progress_survives_a_reworded_marker(self):
        """An unrecognised marker leaves the last known step in place."""
        job = Job(id="x", command="c", argv=[], config="c.yaml", started_at="t")
        JobRunner._note_step("[2/5] Loading dataset…", job)
        JobRunner._note_step("something else entirely", job)
        assert job.step == 2


class TestGuards:
    def test_refuses_a_command_it_may_not_launch(self, config):
        runner = JobRunner()
        with pytest.raises(JobError, match="cannot be launched"):
            runner.start("gui", config)

    def test_refuses_model_surgery_commands(self, config):
        runner = JobRunner()
        with pytest.raises(JobError, match="cannot be launched"):
            runner.start("merge", config)

    def test_refuses_a_missing_config(self, tmp_path):
        runner = JobRunner()
        with pytest.raises(JobError, match="no such config"):
            runner.start("validate", tmp_path / "nope.yaml")

    def test_cancelling_nothing_is_not_an_error(self):
        assert JobRunner().cancel() is False

    def test_no_job_before_the_first_run(self):
        runner = JobRunner()
        assert runner.job is None
        assert runner.is_running() is False


class TestLifecycle:
    def test_a_run_completes_and_reports_its_output(self, config):
        runner = JobRunner()
        job = runner.start("validate", config)
        assert job.status == "running"
        _wait(runner)

        assert runner.job.status == "done"
        assert runner.job.returncode == 0
        _, lines = runner.lines_since(0)
        assert any("Config parsed successfully." in line for line in lines)

    def test_the_command_line_is_recorded_verbatim(self, config):
        runner = JobRunner()
        job = runner.start("validate", config)
        _wait(runner)
        assert job.command == f"bioaccx validate {config}"
        _, lines = runner.lines_since(0)
        assert lines[0] == f"$ bioaccx validate {config}"

    def test_a_failing_run_is_reported_as_failed(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("foundation_model:\n  registry_id: '0xdead'\n")
        runner = JobRunner()
        runner.start("validate", bad)
        _wait(runner)
        assert runner.job.status == "failed"
        assert runner.job.returncode != 0

    def test_a_second_run_is_refused_while_one_is_active(self, config, monkeypatch):
        runner = JobRunner()
        runner.start("validate", config)
        monkeypatch.setattr(runner, "is_running", lambda: True)
        with pytest.raises(JobError, match="already in progress"):
            runner.start("validate", config)

    def test_output_is_readable_from_a_cursor(self, config):
        runner = JobRunner()
        runner.start("validate", config)
        _wait(runner)
        cursor, first = runner.lines_since(0)
        again_cursor, again = runner.lines_since(cursor)
        assert first and not again
        assert again_cursor == cursor

    def test_the_buffer_keeps_only_the_tail(self):
        runner = JobRunner(log_lines=5)
        for i in range(20):
            runner._append(f"line {i}")
        _, lines = runner.lines_since(0)
        assert lines == [f"line {i}" for i in range(15, 20)]

    def test_a_new_run_clears_the_previous_output(self, config):
        runner = JobRunner()
        runner.start("validate", config)
        _wait(runner)
        runner.start("validate", config)
        _wait(runner)
        _, lines = runner.lines_since(0)
        assert sum(1 for line in lines if line.startswith("$ bioaccx")) == 1


class TestWorkingDirectory:
    """Where a run writes its results.

    Paths in a config are relative to where bioaccx was started, so launching
    a config from its own folder would resolve `output_path: ./custom_models`
    against that folder — nesting the results a level deeper for any config
    kept alongside the models it produced.
    """

    def _popen_kwargs(self, monkeypatch, config):
        import subprocess

        captured = {}

        class FakeProcess:
            pid = 5150
            stdout = iter(())

            def wait(self):
                return 0

        def _record(argv, **kwargs):
            captured.update(argv=argv, **kwargs)
            return FakeProcess()

        monkeypatch.setattr(subprocess, "Popen", _record)
        JobRunner().start("validate", config)
        return captured

    def test_the_child_inherits_the_working_directory(self, monkeypatch, tmp_path):
        nested = tmp_path / "custom_models"
        nested.mkdir()
        config = nested / "cfg.yaml"
        config.write_text("foundation_model:\n  registry_id: '0xbb00'\n")

        kwargs = self._popen_kwargs(monkeypatch, config)
        assert kwargs.get("cwd") is None, (
            "a run must inherit the working directory, not adopt the config's "
            "folder, or relative output paths resolve twice"
        )

    def test_the_config_is_passed_as_given(self, monkeypatch, config):
        """So the shown command and the launched one are the same invocation."""
        kwargs = self._popen_kwargs(monkeypatch, config)
        assert kwargs["argv"][-1] == str(config)
        assert kwargs["argv"][-2] == "validate"


class TestCancel:
    def test_cancel_interrupts_the_process_group(self, monkeypatch, config):
        """Cancelling must send SIGINT, which the CLI already exits cleanly on."""
        import os
        import subprocess

        sent = {}

        class FakeProcess:
            pid = 4242
            stdout = iter(())

            def wait(self):
                return 0

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg",
                            lambda pgid, sig: sent.update(pgid=pgid, sig=sig))

        runner = JobRunner()
        job = runner.start("validate", config)
        job.status = "running"          # the fake exits instantly; hold it open
        assert runner.cancel() is True
        assert sent == {"pgid": 4242, "sig": signal.SIGINT}
        assert job.status == "cancelled"

    def test_a_dead_process_cancels_without_raising(self, monkeypatch, config):
        import os
        import subprocess

        class FakeProcess:
            pid = 4243
            stdout = iter(())

            def wait(self):
                return 0

        def _gone(*_a):
            raise ProcessLookupError("already exited")

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", _gone)

        runner = JobRunner()
        job = runner.start("validate", config)
        job.status = "running"
        assert runner.cancel() is False
