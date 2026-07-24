import numpy as np
import pytest

from pftf_alpha.synthetic import (
    PanelSplit,
    SyntheticFamily,
    make_minimal_panel,
    make_synthetic_case,
)


def test_minimal_panel_contains_six_deterministic_3d_families() -> None:
    first = make_minimal_panel(point_count=32, reference_count=96, seed=123)
    second = make_minimal_panel(point_count=32, reference_count=96, seed=123)

    assert {case.family for case in first} == set(SyntheticFamily)
    assert len(first) == 6
    expected_betti = {
        SyntheticFamily.U_CONCAVITY: (1, 1, 0),
        SyntheticFamily.OPPOSING_SHEETS: (2, 0, 0),
        SyntheticFamily.TORUS: (1, 2, 1),
        SyntheticFamily.DISCONNECTED_PARTS: (2, 0, 2),
        SyntheticFamily.SHARP_CREASE: (1, 0, 0),
        SyntheticFamily.MISSING_PATCH: (1, 0, 1),
    }
    for left, right in zip(first, second, strict=True):
        assert left.points.shape == (32, 3)
        assert left.reference_points.shape == (96, 3)
        assert np.unique(left.points, axis=0).shape[0] == 32
        assert left.expected_surface_betti == expected_betti[left.family]
        assert left.expected_surface_betti[0] == left.expected_components
        assert right.expected_surface_betti == left.expected_surface_betti
        np.testing.assert_array_equal(left.points, right.points)
        np.testing.assert_array_equal(left.reference_points, right.reference_points)


def test_held_out_split_changes_shape_and_noise_conditions() -> None:
    calibration = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.CALIBRATION,
        point_count=32,
        reference_count=64,
        seed=7,
    )
    held_out = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.HELD_OUT,
        point_count=32,
        reference_count=64,
        seed=7,
    )

    assert calibration.variation["sheet_gap"] > held_out.variation["sheet_gap"]
    assert calibration.variation["noise"] < held_out.variation["noise"]
    assert calibration.expected_components == held_out.expected_components == 2


def test_synthetic_case_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least 16"):
        make_synthetic_case(
            SyntheticFamily.TORUS,
            point_count=8,
            reference_count=32,
        )
