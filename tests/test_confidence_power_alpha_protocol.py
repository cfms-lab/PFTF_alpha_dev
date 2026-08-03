import numpy as np
import pytest

from pftf_alpha.confidence_power_alpha_protocol import (
    CALIBRATION_SEEDS,
    HELD_OUT_SEEDS,
    make_confidence_power_alpha_panel,
    preregister_confidence_power_alpha,
)
from pftf_alpha.synthetic import PanelSplit


def test_phase45_protocol_freezes_disjoint_panel_and_construction() -> None:
    protocol = preregister_confidence_power_alpha()

    assert set(CALIBRATION_SEEDS).isdisjoint(HELD_OUT_SEEDS)
    assert protocol.calibration_case_count == 9
    assert protocol.held_out_case_count == 27
    assert "spacing_i^2" in protocol.confidence_power_weight_formula
    assert "fail closed" in protocol.point_submersion_policy
    assert "zero-score cells always included" in protocol.critical_score_selection
    assert "connectivity must differ" in protocol.validation_gate


def test_phase45_panel_is_deterministic_and_new() -> None:
    calibration = make_confidence_power_alpha_panel(PanelSplit.CALIBRATION)
    repeated = make_confidence_power_alpha_panel(PanelSplit.CALIBRATION)
    held_out = make_confidence_power_alpha_panel(PanelSplit.HELD_OUT)

    assert len(calibration) == 9
    assert len(held_out) == 27
    assert {case.seed for case in calibration} == set(CALIBRATION_SEEDS)
    assert {case.seed for case in held_out} == set(HELD_OUT_SEEDS)
    for left, right in zip(calibration, repeated, strict=True):
        np.testing.assert_array_equal(left.points, right.points)
        np.testing.assert_array_equal(left.reference_points, right.reference_points)


def test_phase45_rejects_training_split() -> None:
    with pytest.raises(ValueError, match="calibration or held_out"):
        make_confidence_power_alpha_panel(PanelSplit.TRAIN)
