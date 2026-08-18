"""Unit tests for the grouped softmax label space and targets."""
import numpy as np
import pytest

from bioaccx.trainers.grouped import (
    NONE_SUFFIX,
    background_labels,
    build_group_space,
    grouped_predictions,
    grouped_targets,
)

LABELS = ["SPECIES_A1", "SPECIES_A2", "SPECIES_B1", "ambiente", "aves"]
GROUPS = {"SPECIES_A": ["SPECIES_A1", "SPECIES_A2"], "SPECIES_B": ["SPECIES_B1"]}


class TestBuildGroupSpace:
    def test_layout_is_members_then_none_per_group(self):
        out, slices = build_group_space(GROUPS, LABELS)
        assert out == ["SPECIES_A1", "SPECIES_A2", "SPECIES_A_none", "SPECIES_B1", "SPECIES_B_none"]
        assert slices == [(0, 3), (3, 5)]

    def test_slices_are_contiguous_and_cover_every_column(self):
        out, slices = build_group_space(GROUPS, LABELS)
        assert slices[0][0] == 0 and slices[-1][1] == len(out)
        assert all(a[1] == b[0] for a, b in zip(slices, slices[1:]))

    def test_group_order_follows_config_order(self):
        out, _ = build_group_space({"Z": ["SPECIES_B1"], "A": ["SPECIES_A1"]}, LABELS)
        assert out == ["SPECIES_B1", f"Z{NONE_SUFFIX}", "SPECIES_A1", f"A{NONE_SUFFIX}"]

    def test_singleton_group_yields_a_two_column_detector(self):
        out, slices = build_group_space({"ANURA": ["aves"]}, LABELS)
        assert out == ["aves", "ANURA_none"] and slices == [(0, 2)]

    def test_rejects_empty_mapping(self):
        with pytest.raises(ValueError, match="empty"):
            build_group_space({}, LABELS)

    def test_rejects_empty_group(self):
        with pytest.raises(ValueError, match="is empty"):
            build_group_space({"A": []}, LABELS)

    def test_rejects_label_absent_from_training_data(self):
        with pytest.raises(ValueError, match="not present in the training data"):
            build_group_space({"A": ["nope"]}, LABELS)

    def test_rejects_label_claimed_by_two_groups(self):
        with pytest.raises(ValueError, match="disjoint"):
            build_group_space({"A": ["SPECIES_A1"], "B": ["SPECIES_A1"]}, LABELS)

    def test_rejects_none_column_colliding_with_a_real_label(self):
        with pytest.raises(ValueError, match="collides"):
            build_group_space({"aves": ["SPECIES_A1"]}, LABELS + ["aves_none"])


class TestBackgroundLabels:
    def test_ungrouped_labels_are_background(self):
        assert background_labels(GROUPS, LABELS) == ["ambiente", "aves"]

    def test_no_background_when_every_label_is_grouped(self):
        assert background_labels({"A": list(LABELS)}, LABELS) == []


class TestGroupedTargets:
    def setup_method(self):
        self.out, self.slices = build_group_space(GROUPS, LABELS)

    def _target(self, label):
        y = np.array([LABELS.index(label)])
        return grouped_targets(y, LABELS, GROUPS, self.slices)[0]

    def test_exactly_one_active_column_per_group(self):
        for label in LABELS:
            t = self._target(label)
            assert t.sum() == len(GROUPS)
            for s, e in self.slices:
                assert t[s:e].sum() == 1.0

    def test_member_activates_its_own_column(self):
        t = self._target("SPECIES_A2")
        assert t[self.out.index("SPECIES_A2")] == 1.0

    def test_member_activates_none_in_every_other_group(self):
        t = self._target("SPECIES_A2")
        assert t[self.out.index("SPECIES_B_none")] == 1.0

    def test_background_activates_none_everywhere(self):
        for label in ("ambiente", "aves"):
            t = self._target(label)
            assert t[self.out.index("SPECIES_A_none")] == 1.0
            assert t[self.out.index("SPECIES_B_none")] == 1.0

    def test_shape_and_dtype(self):
        t = grouped_targets(np.arange(len(LABELS)), LABELS, GROUPS, self.slices)
        assert t.shape == (len(LABELS), len(self.out))
        assert t.dtype == np.float32

    def test_doubles_as_one_vs_rest_positive_matrix(self):
        # A "_none" column's positives are exactly the samples outside that group.
        t = grouped_targets(np.arange(len(LABELS)), LABELS, GROUPS, self.slices)
        col = self.out.index("SPECIES_A_none")
        expected = [lbl not in GROUPS["SPECIES_A"] for lbl in LABELS]
        assert list(t[:, col].astype(bool)) == expected


class TestGroupedPredictions:
    def test_picks_argmax_within_each_group_as_global_indices(self):
        _, slices = build_group_space(GROUPS, LABELS)
        scores = np.array([[0.1, 0.7, 0.2, 0.3, 0.7]])
        assert grouped_predictions(scores, slices).tolist() == [[1, 4]]

    def test_groups_vote_independently(self):
        _, slices = build_group_space(GROUPS, LABELS)
        scores = np.array([[0.9, 0.05, 0.05, 0.9, 0.1],
                           [0.0, 0.0, 1.0, 0.0, 1.0]])
        assert grouped_predictions(scores, slices).tolist() == [[0, 3], [2, 4]]

    def test_one_column_per_group(self):
        _, slices = build_group_space(GROUPS, LABELS)
        picks = grouped_predictions(np.random.rand(7, 5), slices)
        assert picks.shape == (7, len(GROUPS))
