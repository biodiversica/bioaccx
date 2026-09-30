"""Audio mixup and multi-label training for sigmoid heads.

Toy data: one pure tone per species class (two intensity levels of one species
share an exclusive group) plus a noise background class, embedded with the
spectral DFT embedder so a mixture of two tones lights up both tones' bins.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import EMBED_DIM, SAMPLE_RATE, WINDOW_SAMPLES, make_sine_wav

# DFT embedder bins: 440 Hz (k=44), 1000 Hz (k=100), 2000 Hz (k=200)
TONES = {"BOABIS1": 440, "BOABIS2": 1_000, "DENMIN": 2_000, "ambiente": 0}
GROUPS = {"BOABIS": ["BOABIS1", "BOABIS2"]}


@pytest.fixture(scope="module")
def tone_dataset(tmp_path_factory):
    """4 classes × 8 files of 0.3 s (3 windows each) in label subfolders."""
    root = tmp_path_factory.mktemp("mixup_ds")
    for ci, (label, freq) in enumerate(TONES.items()):
        (root / label).mkdir()
        for j in range(8):
            make_sine_wav(root / label / f"{label}_{j:02d}.wav", freq, 0.3,
                          seed=ci * 100 + j)
    return root


def _mix_cfg(**kw):
    from bioaccx.config import _parse_audio_mixup
    return _parse_audio_mixup({"n_mixes": 60, "exclusive_groups": GROUPS,
                               "background_labels": ["ambiente"], **kw})


def _split(tone_dataset):
    from bioaccx.config import DatasetConfig
    from bioaccx.dataset import load_samples, split_samples
    samples = load_samples(DatasetConfig(data_dir=str(tone_dataset)),
                           window_seconds=WINDOW_SAMPLES / SAMPLE_RATE)
    return split_samples(samples, test_ratio=0.25, random_seed=0)


def _window(s):
    return (str(s.path), s.start_time, s.end_time)


# ---------------------------------------------------------------------------
# Drawing mixes
# ---------------------------------------------------------------------------

class TestBuildAudioMixes:
    def test_mixes_use_train_windows_only(self, tone_dataset):
        from bioaccx.audio_mixup import build_audio_mixes
        train, test = _split(tone_dataset)
        mixes = build_audio_mixes(train, _mix_cfg(), seed=0)
        assert mixes
        train_windows = {_window(s) for s in train}
        test_windows = {_window(s) for s in test}
        for m in mixes:
            for src in m.mix_sources:
                w = (str(src.path), src.start_time, src.end_time)
                assert w in train_windows and w not in test_windows

    def test_target_is_the_union_of_the_sources(self, tone_dataset):
        from bioaccx.audio_mixup import build_audio_mixes
        from bioaccx.dataset import sample_labels
        train, _ = _split(tone_dataset)
        for m in build_audio_mixes(train, _mix_cfg(background_target="include"), seed=0):
            assert set(sample_labels(m)) == {src.label for src in m.mix_sources}

    def test_background_is_dropped_from_the_target_by_default(self, tone_dataset):
        from bioaccx.audio_mixup import build_audio_mixes
        from bioaccx.dataset import sample_labels
        train, _ = _split(tone_dataset)
        mixes = build_audio_mixes(train, _mix_cfg(), seed=0)
        with_bg = [m for m in mixes if any(s.label == "ambiente" for s in m.mix_sources)]
        assert with_bg, "expected some species + background mixes"
        for m in mixes:
            species = {src.label for src in m.mix_sources} - {"ambiente"}
            assert set(sample_labels(m)) == species

    def test_forbidden_pairs_are_never_mixed(self, tone_dataset):
        from bioaccx.audio_mixup import build_audio_mixes
        train, _ = _split(tone_dataset)
        mixes = build_audio_mixes(train, _mix_cfg(n_mixes=200, max_sources=3,
                                                  p_three_sources=0.5), seed=1)
        assert any(len(m.mix_sources) == 3 for m in mixes)
        for m in mixes:
            labels = [src.label for src in m.mix_sources]
            assert len(set(labels)) == len(labels)                  # no label twice
            assert not {"BOABIS1", "BOABIS2"} <= set(labels)        # exclusive group
            assert labels.count("ambiente") <= 1                    # one background at most
            assert set(labels) != {"ambiente"}                      # never background alone

    def test_levels_are_drawn_from_the_snr_range(self, tone_dataset):
        from bioaccx.audio_mixup import build_audio_mixes
        train, _ = _split(tone_dataset)
        for m in build_audio_mixes(train, _mix_cfg(snr_db=[-3, 3]), seed=0):
            assert m.mix_sources[0].snr_db == 0.0
            assert all(-3 <= src.snr_db <= 3 for src in m.mix_sources[1:])

    def test_the_draw_is_reproducible(self, tone_dataset):
        from bioaccx.audio_mixup import build_audio_mixes
        from bioaccx.dataset import mix_key
        train, _ = _split(tone_dataset)
        keys = [[mix_key(m.mix_sources) for m in build_audio_mixes(train, _mix_cfg(), seed=5)]
                for _ in range(2)]
        assert keys[0] == keys[1]
        other = [mix_key(m.mix_sources) for m in build_audio_mixes(train, _mix_cfg(), seed=6)]
        assert other != keys[0]

    def test_composition_counts_targets_sources_and_partners(self, tone_dataset):
        from bioaccx.audio_mixup import build_audio_mixes, training_composition
        from bioaccx.dataset import sample_labels
        train, test = _split(tone_dataset)
        mixes = build_audio_mixes(train, _mix_cfg(), seed=0)
        comp = training_composition(train + mixes, test, sorted(TONES))
        assert comp["n_mixes"] == len(mixes) and comp["n_real_train"] == len(train)
        for label, row in comp["labels"].items():
            assert row["real_train"] == sum(1 for s in train if s.label == label)
            assert row["test"] == sum(1 for s in test if s.label == label)
            assert row["in_mixes"] == sum(1 for m in mixes if label in sample_labels(m))
            assert sum(row["partners"].values()) == sum(
                len(m.mix_sources) - 1 for m in mixes
                if label in {src.label for src in m.mix_sources})
        # a dropped background is summed into mixes but never in their target
        bg = comp["labels"]["ambiente"]
        assert bg["in_mixes"] == 0 and bg["as_source"] > 0 and bg["mix_share"] == 0.0

    @pytest.mark.parametrize("activation,groups", [
        ("softmax", {}), (None, {}), ("grouped_softmax", GROUPS),
    ])
    def test_heads_other_than_sigmoid_are_rejected(self, activation, groups):
        from bioaccx.audio_mixup import check_audio_mixup_head
        with pytest.raises(ValueError, match="output_activation: sigmoid"):
            check_audio_mixup_head("keras", activation, groups)

    def test_sklearn_only_is_rejected(self):
        from bioaccx.audio_mixup import check_audio_mixup_head
        with pytest.raises(ValueError, match="keras head"):
            check_audio_mixup_head("sklearn", "sigmoid", {})


# ---------------------------------------------------------------------------
# Rendering and caching mixes
# ---------------------------------------------------------------------------

class TestMixAudio:
    def test_levels_follow_snr_and_the_sum_never_clips(self, tmp_path):
        from bioaccx.dataset import MixSource, mix_audio
        a = make_sine_wav(tmp_path / "a.wav", 440, 0.1, noise_std=0.0)
        b = make_sine_wav(tmp_path / "b.wav", 2_000, 0.1, noise_std=0.0)
        src = lambda p, snr: MixSource(path=p, label=p.stem, start_time=0.0,
                                       end_time=0.1, snr_db=snr)
        mix = mix_audio((src(a, 0.0), src(b, 6.0)), SAMPLE_RATE, WINDOW_SAMPLES)
        assert np.max(np.abs(mix)) <= 0.99 + 1e-6
        spectrum = np.abs(np.fft.rfft(mix))
        bin_hz = SAMPLE_RATE / WINDOW_SAMPLES
        ratio_db = 20 * np.log10(spectrum[int(2_000 / bin_hz)] / spectrum[int(440 / bin_hz)])
        assert ratio_db == pytest.approx(6.0, abs=0.2)

    def test_mix_embeddings_hit_the_cache_on_rerun(self, tone_dataset, tmp_path,
                                                    dft_foundation_cfg, capsys):
        from bioaccx.audio_mixup import build_audio_mixes
        from bioaccx.dataset import extract_embeddings
        train, _ = _split(tone_dataset)
        mixes = build_audio_mixes(train, _mix_cfg(n_mixes=10), seed=0)
        db = tmp_path / "cache.db"
        kwargs = dict(n_workers=1, cache_sqlite=db, export_sqlite=db, multi_hot=True,
                      label_names=sorted(TONES))
        X1, Y1, _ = extract_embeddings(mixes, dft_foundation_cfg, EMBED_DIM, **kwargs)
        capsys.readouterr()
        X2, Y2, _ = extract_embeddings(mixes, dft_foundation_cfg, EMBED_DIM, **kwargs)
        assert capsys.readouterr().out.count("cached (sqlite)") == len(mixes)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)
        assert (Y1.sum(axis=1) >= 1).all()


# ---------------------------------------------------------------------------
# Multi-label windows from overlapping annotations
# ---------------------------------------------------------------------------

class TestMergeOverlappingAnnotations:
    def test_rows_sharing_a_window_become_one_multilabel_sample(self, tmp_path):
        from bioaccx.config import DatasetConfig
        from bioaccx.dataset import load_samples, sample_labels
        make_sine_wav(tmp_path / "rec.wav", 440, 0.3)
        table = tmp_path / "t.csv"
        table.write_text("filename,label,start_time,end_time\n"
                         "rec.wav,BOABIS1,0.0,0.1\nrec.wav,DENMIN,0.0,0.1\n"
                         "rec.wav,DENMIN,0.1,0.2\n")
        samples = load_samples(
            DatasetConfig(data_dir=str(tmp_path), label_mode="table", table_file=str(table)),
            window_seconds=WINDOW_SAMPLES / SAMPLE_RATE)
        assert [sample_labels(s) for s in samples] == [("BOABIS1", "DENMIN"), ("DENMIN",)]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _cfg(foundation_cfg, data_dir, output_path, activation="sigmoid", **mixup) -> dict:
    fm = foundation_cfg
    return {
        "foundation_model": {
            "name": fm.name, "version": fm.version, "format": fm.format,
            "source": "local", "path": fm.path, "sample_rate": fm.sample_rate,
            "window_samples": fm.window_samples, "input_name": fm.input_name,
            "output_name": fm.output_name, "embedding_size": fm.embedding_size,
        },
        "dataset": {"data_dir": str(data_dir), "embedding_workers": 2, "test_ratio": 0.25},
        "training": {
            "classifier": "keras",
            "keras": {"output_activation": activation, "hidden_units": 0, "epochs": 150,
                      "batch_size": 16, "learning_rate": 0.01, "dropout": 0.0},
            "audio_mixup": {"ratio": 1.0, "exclusive_groups": GROUPS,
                            "background_labels": ["ambiente"], "snr_db": [-3, 3],
                            **mixup},
        },
        "output": {"output_path": str(output_path), "model_name": "mixup",
                   "model_version": "0.1", "output_type": "head", "output_format": "onnx"},
    }


def _run(cfg_dict):
    from bioaccx.config import _parse_config
    from bioaccx.train import run
    return run(_parse_config(cfg_dict))


@pytest.mark.slow
class TestAudioMixupPipeline:
    @pytest.fixture(scope="class")
    def outputs(self, dft_foundation_cfg, tone_dataset, tmp_path_factory):
        return _run(_cfg(dft_foundation_cfg, tone_dataset, tmp_path_factory.mktemp("out"),
                         test_mixes=20))

    @staticmethod
    def _mixture_scores(outputs, dft_onnx_model) -> dict:
        """Head scores for a real two-tone mixture (BOABIS1 + DENMIN), never seen in training."""
        import onnxruntime as ort
        t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
        mixture = (0.4 * np.sin(2 * np.pi * 440 * t)
                   + 0.4 * np.sin(2 * np.pi * 2_000 * t)).astype(np.float32)[None]
        backbone = ort.InferenceSession(str(dft_onnx_model))
        emb = backbone.run(None, {backbone.get_inputs()[0].name: mixture})[0]
        head = ort.InferenceSession(outputs["keras_onnx_head_fp32"])
        scores = head.run(None, {head.get_inputs()[0].name: emb})[0][0]
        return dict(zip(Path(outputs["labels"]).read_text().split(), scores))

    def test_real_two_source_mixture_scores_both_classes(self, outputs, dft_onnx_model,
                                                         dft_foundation_cfg, tone_dataset,
                                                         tmp_path):
        scores = self._mixture_scores(outputs, dft_onnx_model)
        assert scores["BOABIS1"] > 0.9 and scores["DENMIN"] > 0.9, scores
        assert scores["BOABIS2"] < 0.5, scores
        # The same head trained without mixes is less sure of both classes.
        control = self._mixture_scores(
            _run(_cfg(dft_foundation_cfg, tone_dataset, tmp_path, enabled=False)),
            dft_onnx_model)
        assert min(scores["BOABIS1"], scores["DENMIN"]) > max(
            control["BOABIS1"], control["DENMIN"]), (scores, control)

    def test_dataset_list_records_train_only_mixes(self, outputs):
        # The split is per window, so the same recording can give train and test
        # windows: sources are checked window by window.
        rows = list(csv.DictReader(open(outputs["dataset_list"])))

        def windows(split):
            return {f"{r['filepath']}@{float(r['start_time']):.3f}-{float(r['end_time']):.3f}"
                    for r in rows if r["split"] == split and not r["mix_sources"]}

        def sources(row):
            return {part.split(" [")[0] for part in row["mix_sources"].split(" | ")}

        train_windows, test_windows = windows("train"), windows("test")
        mixes = [r for r in rows if r["split"] == "train" and r["mix_sources"]]
        assert mixes
        for r in mixes:
            assert sources(r) <= train_windows and not sources(r) & test_windows
            assert r["labels"] and r["mix_snr_db"]
        test_mixes = [r for r in rows if r["split"] == "test_mix"]
        assert len(test_mixes) == 20
        for r in test_mixes:
            assert sources(r) <= test_windows

    def test_report_states_the_synthetic_share_and_the_mixed_test_set(self, outputs):
        text = Path(outputs["keras_report"]).read_text()
        assert "are synthetic audio mixes" in text
        assert "subset accuracy" in text
        assert "Synthetic Mixed Test Set" in text
        assert Path(outputs["keras_evaluation_mixed"]).exists()
        meta = json.loads(Path(outputs["dataset_metadata"]).read_text())
        assert meta["totals"]["n_audio_mixes"] > 0

    def test_mixed_averages_skip_classes_no_mix_carries(self, outputs):
        rows = list(csv.DictReader(open(outputs["keras_evaluation_mixed"])))
        overall, classes = rows[0], rows[1:]
        ambiente = next(r for r in classes if r["Class"] == "ambiente")
        assert int(ambiente["Samples"]) == 0          # background dropped from mix targets
        scored = [float(r["F1 Score (opt)"]) for r in classes if int(r["Samples"]) > 0]
        assert float(overall["F1 Score (opt)"]) == pytest.approx(np.mean(scored), abs=1e-4)
        assert "No mix carries ambiente" in Path(outputs["keras_report"]).read_text()

    def test_model_metadata_records_the_mixup(self, outputs):
        info = json.loads(Path(outputs["model_info"]).read_text())
        mix = info["audio_mixup"]
        assert mix["background_target"] == "drop"
        assert mix["exclusive_groups"] == GROUPS
        assert mix["n_train_mixes"] > 0 and mix["n_train_real"] > 0
        assert mix["n_test_mixes"] == 20
        assert isinstance(mix["seed"], int)

    def test_report_shows_the_training_set_make_up(self, outputs):
        text = Path(outputs["keras_report"]).read_text()
        section = text.split("--- Training Set (real windows + audio mixes) ---")[1]
        section = section.split("--- Training History ---")[0]
        assert "Mix share" in section and "Main mix partners" in section
        for label in TONES:
            assert f"\n{label}" in section
        assert "pairing balanced" in section and "drop from targets" in section

    def test_both_metadata_files_hold_the_composition(self, outputs):
        dataset_meta = json.loads(Path(outputs["dataset_metadata"]).read_text())
        model_meta = json.loads(Path(outputs["model_info"]).read_text())
        comp = model_meta["audio_mixup"]["composition"]
        assert dataset_meta["audio_mixup"] == comp
        assert set(comp["labels"]) == set(TONES)
        assert comp["n_mixes"] == model_meta["audio_mixup"]["n_train_mixes"]

    def test_gui_reads_the_mixup(self, outputs):
        from bioaccx.gui import results
        directory = Path(outputs["model_info"]).parent
        record = results.detail(directory.parent, directory.name)
        assert record["mixup"]["n_mixes"] > 0 and record["mixup"]["n_test_mixes"] == 20
        assert set(record["composition"]) == set(TONES)
        assert record["mixup_settings"]["pairing"] == "balanced"
        assert "composition" not in record["config"]["audio_mixup"]
        assert record["evaluation_mixed"][0]["overall"]
        assert {row["label"] for row in record["evaluation_mixed"] if not row["overall"]} == set(TONES)
        compared = results.compare(directory.parent, directory.name, directory.name)
        shares = {row["label"]: row["left_mix_share"] for row in compared["metrics"]}
        assert shares["DENMIN"] == record["composition"]["DENMIN"]["mix_share"] > 0
        assert compared["left"]["mixup"] == record["mixup"]

    def test_softmax_config_raises(self, dft_foundation_cfg, tone_dataset, tmp_path):
        with pytest.raises(ValueError, match="output_activation: sigmoid"):
            _run(_cfg(dft_foundation_cfg, tone_dataset, tmp_path, activation="softmax"))
