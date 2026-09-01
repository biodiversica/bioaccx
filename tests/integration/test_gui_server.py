"""End-to-end tests for the config editor's HTTP API.

These drive the app the way the browser does: open a file, apply a form edit,
read back the exact text a save would write, then save it and load the result
through the real config loader.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the GUI needs the [gui] extra")

from fastapi.testclient import TestClient  # noqa: E402

from bioaccx.gui.server import create_app  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[2] / "example_config.yaml"


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def config_file(tmp_path):
    target = tmp_path / "cfg.yaml"
    target.write_text(EXAMPLE.read_text())
    return target


class TestBootstrap:
    def test_serves_the_schema(self, client):
        body = client.get("/api/bootstrap").json()
        assert [s["name"] for s in body["schema"]["sections"]] == [
            "foundation_model", "dataset", "training", "output", "umap"]

    def test_reports_which_paths_are_secret(self, client):
        assert client.get("/api/bootstrap").json()["secret_paths"] == ["dataset.xc_api_key"]

    def test_names_the_file_to_open(self, tmp_path):
        client = TestClient(create_app(config_path=tmp_path / "given.yaml"))
        assert client.get("/api/bootstrap").json()["open_path"].endswith("given.yaml")

    def test_serves_the_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "bioaccx" in response.text


class TestOpen:
    def test_returns_text_values_and_validation(self, client, config_file):
        body = client.get("/api/config", params={"path": str(config_file)}).json()
        assert body["text"] == EXAMPLE.read_text()
        assert body["values"]["output.model_name"] == "test_classifier"
        assert body["validation"]["valid"] is True

    def test_missing_file_is_a_404(self, client, tmp_path):
        response = client.get("/api/config", params={"path": str(tmp_path / "nope.yaml")})
        assert response.status_code == 404

    def test_a_secret_is_never_sent_to_the_browser(self, client, tmp_path):
        target = tmp_path / "cfg.yaml"
        target.write_text(
            "foundation_model:\n  registry_id: '0xbb00'\n"
            "dataset:\n  data_dir: []\n  xc_api_key: super-secret\n")
        body = client.get("/api/config", params={"path": str(target)}).json()
        assert body["values"]["dataset.xc_api_key"] == "__SET__"
        assert "super-secret" not in str(body["values"])


class TestDocument:
    def test_an_edit_comes_back_as_the_file_it_would_write(self, client):
        body = client.post("/api/document", json={
            "text": "output:\n  model_name: old\n",
            "edits": {"output.model_name": "new"},
        }).json()
        assert body["text"] == "output:\n  model_name: new\n"
        assert body["values"]["output.model_name"] == "new"

    def test_comments_survive_an_edit_through_the_api(self, client, config_file):
        opened = client.get("/api/config", params={"path": str(config_file)}).json()
        edited = client.post("/api/document", json={
            "text": opened["text"], "edits": {"training.keras.epochs": 7},
        }).json()
        before = [l for l in opened["text"].splitlines() if l.strip().startswith("#")]
        after = [l for l in edited["text"].splitlines() if l.strip().startswith("#")]
        assert before == after
        assert "epochs: 7" in edited["text"]

    def test_clearing_a_field_removes_the_key(self, client):
        body = client.post("/api/document", json={
            "text": "output:\n  model_name: x\n  model_version: '1'\n",
            "edits": {"output.model_name": None},
        }).json()
        assert "model_name" not in body["text"]

    def test_an_unchanged_secret_is_not_written_back(self, client):
        """The browser echoes the marker; the real key must stay in the file."""
        body = client.post("/api/document", json={
            "text": "dataset:\n  xc_api_key: real-key\n",
            "edits": {"dataset.xc_api_key": "__SET__"},
        }).json()
        assert "real-key" in body["text"]

    def test_a_replaced_secret_is_written(self, client):
        body = client.post("/api/document", json={
            "text": "dataset:\n  xc_api_key: old-key\n",
            "edits": {"dataset.xc_api_key": "new-key"},
        }).json()
        assert "new-key" in body["text"]

    def test_malformed_yaml_is_a_400(self, client):
        response = client.post("/api/document",
                               json={"text": "a:\n - [oops\n", "edits": {}})
        assert response.status_code == 400


class TestValidate:
    def test_the_example_config_validates(self, client):
        body = client.post("/api/validate", json={"text": EXAMPLE.read_text()}).json()
        assert body["valid"] is True
        assert body["summary"]["classifier"] == "keras"

    def test_an_unknown_registry_id_is_reported(self, client):
        body = client.post("/api/validate", json={
            "text": "foundation_model:\n  registry_id: '0xdead'\n"}).json()
        assert body["valid"] is False
        assert "0xdead" in body["errors"][0]["message"]

    def test_a_cross_field_rule_is_caught_before_the_run(self, client):
        """Augmentation with neither a directory nor labels is rejected."""
        body = client.post("/api/validate", json={
            "text": "foundation_model:\n  registry_id: '0xbb00'\n"
                    "dataset:\n  data_dir: []\n  augmentation:\n    snr_levels: [10]\n",
        }).json()
        assert body["valid"] is False
        assert "augmentation" in body["errors"][0]["message"]

    def test_the_error_is_pointed_at_a_section_when_it_names_one(self, client):
        body = client.post("/api/validate", json={
            "text": "foundation_model:\n  registry_id: '0xbb00'\n"
                    "dataset:\n  data_dir: []\n  augmentation:\n    snr_levels: [10]\n",
        }).json()
        assert body["errors"][0]["path"] == "dataset"

    def test_the_summary_counts_sources(self, client):
        body = client.post("/api/validate", json={
            "text": "foundation_model:\n  registry_id: '0xbb00'\n"
                    "dataset:\n  sources:\n    - data_dir: /a\n    - data_dir: /b\n",
        }).json()
        assert body["summary"]["sources"] == 2


class TestSave:
    def test_writes_a_file_the_loader_accepts(self, client, tmp_path):
        from bioaccx.config import load_config
        target = tmp_path / "out.yaml"
        response = client.post("/api/save", json={
            "path": str(target), "text": EXAMPLE.read_text()})
        assert response.status_code == 200
        assert load_config(str(target)).training.classifier == "keras"

    def test_round_trip_through_save_preserves_comments(self, client, tmp_path):
        target = tmp_path / "out.yaml"
        client.post("/api/save", json={"path": str(target), "text": EXAMPLE.read_text()})
        assert target.read_text() == EXAMPLE.read_text()

    def test_rejects_a_path_that_is_not_a_config(self, client, tmp_path):
        response = client.post("/api/save", json={
            "path": str(tmp_path / "notes.txt"), "text": "a: 1\n"})
        assert response.status_code == 400

    def test_creates_missing_directories(self, client, tmp_path):
        target = tmp_path / "new" / "dir" / "cfg.yaml"
        client.post("/api/save", json={"path": str(target), "text": "a: 1\n"})
        assert target.exists()


class TestTemplate:
    def test_a_new_config_starts_valid(self, client):
        body = client.get("/api/template").json()
        assert body["validation"]["valid"] is True
        assert "registry_id" in body["text"]


class TestBrowse:
    def test_lists_directories(self, client, tmp_path):
        (tmp_path / "audio").mkdir()
        body = client.get("/api/browse", params={"path": str(tmp_path)}).json()
        assert "audio" in [d["name"] for d in body["dirs"]]

    def test_lists_only_config_files(self, client, tmp_path):
        (tmp_path / "cfg.yaml").touch()
        (tmp_path / "notes.txt").touch()
        body = client.get("/api/browse", params={"path": str(tmp_path)}).json()
        assert [f["name"] for f in body["files"]] == ["cfg.yaml"]

    def test_hides_dotfiles(self, client, tmp_path):
        (tmp_path / ".hidden").mkdir()
        body = client.get("/api/browse", params={"path": str(tmp_path)}).json()
        assert ".hidden" not in [d["name"] for d in body["dirs"]]

    def test_offers_the_parent_for_walking_up(self, client, tmp_path):
        body = client.get("/api/browse", params={"path": str(tmp_path)}).json()
        assert body["parent"] == str(tmp_path.parent)

    def test_a_file_path_lists_its_folder(self, client, config_file):
        body = client.get("/api/browse", params={"path": str(config_file)}).json()
        assert body["path"] == str(config_file.parent)


class TestRuns:
    """Launching a run, watching it, and being told how it ended."""

    def _wait(self, client, timeout=90.0):
        import time
        deadline = time.monotonic() + timeout
        while client.get("/api/run").json()["running"]:
            if time.monotonic() > deadline:
                pytest.fail("the run did not finish in time")
            time.sleep(0.05)
        return client.get("/api/run").json()["job"]

    def test_bootstrap_lists_what_can_be_run(self, client):
        assert client.get("/api/bootstrap").json()["runnable"] == [
            "train", "dataset", "embeddings", "validate"]

    def test_nothing_is_running_to_begin_with(self, client):
        body = client.get("/api/run").json()
        assert body["running"] is False and body["job"] is None

    def test_a_run_completes_and_its_output_is_readable(self, client, config_file):
        started = client.post("/api/run", json={
            "command": "validate", "path": str(config_file)})
        assert started.status_code == 200
        assert started.json()["command"] == f"bioaccx validate {config_file}"

        job = self._wait(client)
        assert job["status"] == "done"
        assert job["returncode"] == 0

        log = client.get("/api/run/log", params={"cursor": 0}).json()
        assert any("Config parsed successfully." in line for line in log["lines"])
        assert log["cursor"] > 0

    def test_a_failing_run_reports_a_non_zero_exit(self, client, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("foundation_model:\n  registry_id: '0xdead'\n")
        client.post("/api/run", json={"command": "validate", "path": str(bad)})
        job = self._wait(client)
        assert job["status"] == "failed"
        assert job["returncode"] != 0

    def test_the_log_cursor_only_returns_new_lines(self, client, config_file):
        client.post("/api/run", json={"command": "validate", "path": str(config_file)})
        self._wait(client)
        first = client.get("/api/run/log", params={"cursor": 0}).json()
        second = client.get("/api/run/log", params={"cursor": first["cursor"]}).json()
        assert first["lines"] and second["lines"] == []

    def test_a_command_outside_the_allowed_set_is_refused(self, client, config_file):
        response = client.post("/api/run", json={
            "command": "merge", "path": str(config_file)})
        assert response.status_code == 409
        assert "cannot be launched" in response.json()["detail"]

    def test_a_missing_config_is_refused(self, client, tmp_path):
        response = client.post("/api/run", json={
            "command": "validate", "path": str(tmp_path / "nope.yaml")})
        assert response.status_code == 409

    def test_cancelling_nothing_is_a_409(self, client):
        assert client.post("/api/run/cancel").status_code == 409

    def test_the_stream_reports_lines_and_a_final_status(self, client, config_file):
        client.post("/api/run", json={"command": "validate", "path": str(config_file)})
        self._wait(client)
        body = client.get("/api/run/stream").text
        assert "event: line" in body
        assert "event: status" in body
        assert "Config parsed successfully." in body
        assert '"status": "done"' in body

    def test_the_stream_ends_when_no_run_is_active(self, client):
        """It must not hang the connection open when there is nothing to watch."""
        body = client.get("/api/run/stream").text
        assert "event: status" in body

class TestExplorer:
    """Reading trained models back through the API."""

    @pytest.fixture
    def models(self, tmp_path):
        """Two model directories plus the audio one of their samples names."""
        import numpy as np
        import soundfile as sf
        from tests.unit.test_gui_results import _model

        root = tmp_path / "custom_models"
        root.mkdir()
        _model(root, "left_0xbb00_v1", macro_f1=0.70)
        _model(root, "right_0xbb00_v1", macro_f1=0.85)
        rate = 32000
        tone = (0.3 * np.sin(2 * np.pi * 800 * np.linspace(0, 4, 4 * rate))).astype("float32")
        for name in ("A", "B"):
            sf.write(root / f"{name}.wav", tone, rate)
        return root

    @pytest.fixture
    def client(self, models):
        return TestClient(create_app(models_dir=models))

    def test_lists_models(self, client):
        body = client.get("/api/models").json()
        assert {m["stem"] for m in body["models"]} == {"left_0xbb00_v1", "right_0xbb00_v1"}

    def test_a_missing_models_directory_is_a_404(self, tmp_path):
        client = TestClient(create_app(models_dir=tmp_path / "nowhere"))
        assert client.get("/api/models").status_code == 404

    def test_model_detail_carries_metrics(self, client):
        body = client.get("/api/models/left_0xbb00_v1").json()
        assert body["n_classes"] == 2
        assert any(row["overall"] for row in body["evaluation"])

    def test_an_unknown_model_is_a_404(self, client):
        assert client.get("/api/models/nope").status_code == 404

    def test_a_traversal_attempt_is_refused(self, client):
        assert client.get("/api/models/..%2F..%2Fetc").status_code == 404

    def test_umap_points_are_served_with_keys(self, client):
        body = client.get("/api/models/left_0xbb00_v1/umap").json()
        assert body["key_source"] == "column"
        assert len(body["points"]) == 2

    def test_a_sample_resolves_to_its_source_recording(self, client, models):
        body = client.get("/api/models/left_0xbb00_v1/sample",
                          params={"key": "A_0.000_3.000"}).json()
        assert body["exists"] is True
        assert body["filename"] == "A.wav"
        assert (body["start_time"], body["end_time"]) == (0.0, 3.0)

    def test_an_unknown_sample_is_a_404(self, client):
        assert client.get("/api/models/left_0xbb00_v1/sample",
                          params={"key": "nope"}).status_code == 404

    def test_the_clip_is_served_as_wav(self, client):
        response = client.get("/api/models/left_0xbb00_v1/clip",
                              params={"key": "A_0.000_3.000"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content[:4] == b"RIFF"

    def test_the_clip_supports_range_requests(self, client):
        """The audio element needs Range to seek within a clip."""
        response = client.get("/api/models/left_0xbb00_v1/clip",
                              params={"key": "A_0.000_3.000"},
                              headers={"Range": "bytes=0-99"})
        assert response.status_code == 206
        assert len(response.content) == 100
        assert response.headers["content-range"].startswith("bytes 0-99/")

    def test_the_spectrogram_is_served_as_png(self, client):
        response = client.get("/api/models/left_0xbb00_v1/spectrogram",
                              params={"key": "A_0.000_3.000", "fmax": 8000})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_an_impossible_frequency_band_is_a_422(self, client):
        response = client.get("/api/models/left_0xbb00_v1/spectrogram",
                              params={"key": "A_0.000_3.000",
                                      "fmin": 30000, "fmax": 30001})
        assert response.status_code == 422

    def test_compare_reports_deltas(self, client):
        body = client.get("/api/models/compare",
                          params={"left": "left_0xbb00_v1",
                                  "right": "right_0xbb00_v1"}).json()
        overall = next(row for row in body["metrics"] if row["overall"])
        assert overall["delta"] == pytest.approx(0.15)

    def test_compare_is_not_mistaken_for_a_model_name(self, client):
        """`/api/models/compare` must not be captured by the {stem} route."""
        response = client.get("/api/models/compare",
                              params={"left": "left_0xbb00_v1",
                                      "right": "right_0xbb00_v1"})
        assert response.status_code == 200
        assert "metrics" in response.json()

    def test_a_missing_plot_is_a_404(self, client):
        assert client.get("/api/models/left_0xbb00_v1/plot/umap").status_code == 404

    def test_an_unknown_plot_name_is_a_404(self, client):
        assert client.get("/api/models/left_0xbb00_v1/plot/evil").status_code == 404

    def test_bootstrap_names_the_models_directory(self, client, models):
        assert client.get("/api/bootstrap").json()["models_dir"] == str(models)


class TestNonAsciiFilenames:
    """Recordings named with characters HTTP headers cannot carry.

    Headers are latin-1, so a name with an en dash used to crash the clip
    endpoint on its way into Content-Disposition.
    """

    @pytest.fixture
    def models(self, tmp_path):
        import numpy as np
        import soundfile as sf
        from tests.unit.test_gui_results import _model

        root = tmp_path / "custom_models"
        root.mkdir()
        _model(root, "crickets_0xbb10_v1", classes=("Miogryllus – Holotype", "Anaxipha sp.1"))
        rate = 32000
        tone = (0.3 * np.sin(2 * np.pi * 900 * np.linspace(0, 4, 4 * rate))).astype("float32")
        for name in ("Miogryllus – Holotype", "Anaxipha sp.1"):
            sf.write(root / f"{name}.wav", tone, rate)
        return root

    @pytest.fixture
    def client(self, models):
        return TestClient(create_app(models_dir=models))

    def _key(self):
        return "Miogryllus – Holotype_0.000_3.000"

    def test_the_clip_is_served(self, client):
        response = client.get("/api/models/crickets_0xbb10_v1/clip",
                              params={"key": self._key()})
        assert response.status_code == 200
        assert response.content[:4] == b"RIFF"

    def test_a_range_request_is_served(self, client):
        response = client.get("/api/models/crickets_0xbb10_v1/clip",
                              params={"key": self._key()},
                              headers={"Range": "bytes=0-99"})
        assert response.status_code == 206
        assert len(response.content) == 100

    def test_the_real_name_is_carried_utf8_encoded(self, client):
        response = client.get("/api/models/crickets_0xbb10_v1/clip",
                              params={"key": self._key()})
        disposition = response.headers["content-disposition"]
        assert "filename*=UTF-8''" in disposition
        assert "%E2%80%93" in disposition            # the en dash

    def test_the_header_is_latin1_safe(self, client):
        response = client.get("/api/models/crickets_0xbb10_v1/clip",
                              params={"key": self._key()})
        response.headers["content-disposition"].encode("latin-1")

    def test_the_spectrogram_is_served(self, client):
        response = client.get("/api/models/crickets_0xbb10_v1/spectrogram",
                              params={"key": self._key()})
        assert response.status_code == 200
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


class TestContentDisposition:
    def test_a_plain_name_needs_no_encoded_form(self):
        from bioaccx.gui.server import _content_disposition
        assert _content_disposition("clip.wav") == 'inline; filename="clip.wav"'

    def test_a_non_ascii_name_gains_an_encoded_form(self):
        from bioaccx.gui.server import _content_disposition
        header = _content_disposition("açaí.wav")
        assert header.startswith('inline; filename="a')
        assert "filename*=UTF-8''a%C3%A7a%C3%AD.wav" in header

    def test_quotes_and_backslashes_cannot_break_the_header(self):
        """They are legal in a file name and would end the quoted string."""
        from bioaccx.gui.server import _content_disposition
        header = _content_disposition('a"b\\c.wav')
        assert header.count('"') == 2
        assert "filename*=UTF-8''" in header

    @pytest.mark.parametrize("name", [
        "Miogryllus itaquiensis – Holotype MW02.wav",
        "açaí_ñandú.wav", "日本語.wav", "emoji_🦗.wav", 'quote".wav', "tab\tname.wav",
    ])
    def test_every_name_produces_a_sendable_header(self, name):
        from bioaccx.gui.server import _content_disposition
        _content_disposition(name).encode("latin-1")


class TestTokenGuard:
    def test_the_api_is_closed_without_the_token(self):
        client = TestClient(create_app(token="s3cret"))
        assert client.get("/api/bootstrap").status_code == 401

    def test_the_token_opens_it(self):
        client = TestClient(create_app(token="s3cret"))
        response = client.get("/api/bootstrap", params={"token": "s3cret"})
        assert response.status_code == 200

    def test_a_wrong_token_is_refused(self):
        client = TestClient(create_app(token="s3cret"))
        assert client.get("/api/bootstrap", params={"token": "nope"}).status_code == 401

    def test_health_stays_reachable(self):
        client = TestClient(create_app(token="s3cret"))
        assert client.get("/health").status_code == 200
