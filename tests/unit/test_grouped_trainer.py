"""Grouped-head behaviour in the Keras trainer and the training report."""
import numpy as np
import pytest

from bioaccx.config import KerasConfig
from bioaccx.trainers.grouped import build_group_space

LABELS = ["a1", "a2", "b1", "bg"]
GROUPS = {"A": ["a1", "a2"], "B": ["b1"]}


def _data(n=48, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(len(LABELS)), n // len(LABELS))
    X = (rng.standard_normal((len(y), dim)) + y[:, None]).astype(np.float32)
    return X, y


def _cfg(**kw):
    base = dict(hidden_units=0, epochs=2, batch_size=8,
                output_activation="grouped_softmax", label_groups=GROUPS)
    return KerasConfig(**{**base, **kw})


@pytest.fixture(scope="module")
def trained():
    from bioaccx.trainers.keras_trainer import train_keras
    X, y = _data()
    return train_keras(X, y, X, y, LABELS, _cfg(), seed=0), X, y


class TestGroupedTraining:
    def test_output_width_is_the_group_space_not_the_label_count(self, trained):
        model, X, _ = trained
        assert model.predict(X, verbose=0).shape[1] == 5   # a1 a2 A_none b1 B_none

    def test_each_group_sums_to_one(self, trained):
        model, X, _ = trained
        p = model.predict(X, verbose=0)
        _, slices = build_group_space(GROUPS, LABELS)
        for s, e in slices:
            assert np.allclose(p[:, s:e].sum(1), 1.0, atol=1e-5)

    def test_two_members_of_one_group_can_never_both_exceed_half(self, trained):
        model, X, _ = trained
        p = model.predict(X, verbose=0)
        _, slices = build_group_space(GROUPS, LABELS)
        for s, e in slices:
            assert int((p[:, s:e - 1] > 0.5).sum(1).max()) <= 1

    def test_report_params_record_the_layout(self, trained):
        model, _, _ = trained
        params = model._report_params
        assert params["output_activation"] == "grouped_softmax"
        assert params["loss"] == "grouped_crossentropy"
        assert params["label_groups"] == GROUPS
        assert params["group_slices"] == [[0, 3], [3, 5]]

    def test_flat_head_records_no_group_layout(self):
        from bioaccx.trainers.keras_trainer import train_keras
        X, y = _data()
        model = train_keras(X, y, X, y, LABELS,
                            KerasConfig(hidden_units=0, epochs=1, output_activation="softmax"),
                            seed=0)
        assert model._report_params["label_groups"] == {}
        assert model.predict(X, verbose=0).shape[1] == len(LABELS)


class TestGroupedGuards:
    @pytest.mark.parametrize("kw, exc", [
        (dict(output_activation="softmax", label_groups=GROUPS), ValueError),
        (dict(output_activation="grouped_softmax", label_groups={}), ValueError),
        (dict(label_smoothing=True), NotImplementedError),
        (dict(upsampling_ratio=0.5), NotImplementedError),
        (dict(focal_loss=True), NotImplementedError),
    ])
    def test_incompatible_options_are_rejected(self, kw, exc):
        from bioaccx.trainers.keras_trainer import train_keras
        X, y = _data()
        with pytest.raises(exc):
            train_keras(X, y, X, y, LABELS, _cfg(**kw), seed=0)


class TestGroupedReport:
    def test_report_uses_the_group_space_and_group_summaries(self, trained, tmp_path):
        from bioaccx.report import write_keras_report
        model, X, y = trained
        path = tmp_path / "r.txt"
        write_keras_report(model, X, y, LABELS, path, eval_csv_path=tmp_path / "e.csv")
        text = path.read_text()
        assert "Outputs (5): a1, a2, A_none, b1, B_none" in text
        assert "--- group: A ---" in text and "--- group: B ---" in text
        assert "Mean group acc:" in text and "All-groups acc:" in text
        assert "Macro F1:" not in text          # undefined for a grouped head
        assert "grouped softmax outputs" in text

    def test_evaluation_csv_has_one_row_per_output_column(self, trained, tmp_path):
        import csv
        from bioaccx.report import write_keras_report
        model, X, y = trained
        write_keras_report(model, X, y, LABELS, tmp_path / "r.txt",
                           eval_csv_path=tmp_path / "e.csv")
        rows = list(csv.DictReader(open(tmp_path / "e.csv")))
        assert [r["Class"] for r in rows[1:]] == ["a1", "a2", "A_none", "b1", "B_none"]

    def test_none_column_positives_are_the_samples_outside_that_group(self, trained, tmp_path):
        import csv
        from bioaccx.report import write_keras_report
        model, X, y = trained
        write_keras_report(model, X, y, LABELS, tmp_path / "r.txt",
                           eval_csv_path=tmp_path / "e.csv")
        rows = {r["Class"]: r for r in csv.DictReader(open(tmp_path / "e.csv"))}
        # "bg" and "b1" both sit outside group A
        assert int(rows["A_none"]["Samples"]) == int(np.sum([LABELS[i] in ("b1", "bg") for i in y]))
