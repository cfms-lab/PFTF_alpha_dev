import numpy as np

from pftf_alpha.pftf_shear_identifiability import (
    _slope_and_r2,
    evaluate_pftf_shear_identifiability,
)


def test_phase49_linear_signal_helper_recovers_exact_slope() -> None:
    strengths = np.asarray((0.0, 0.1, 0.2, 0.3, 0.4))
    values = 2.0 - 3.0 * strengths

    slope, r2 = _slope_and_r2(strengths, values)

    assert np.isclose(slope, -3.0)
    assert r2 == 1.0


def test_phase49_constant_feature_is_not_an_identifiable_signal() -> None:
    strengths = np.asarray((0.0, 0.1, 0.2, 0.3, 0.4))

    slope, r2 = _slope_and_r2(strengths, np.ones_like(strengths))

    assert np.isclose(slope, 0.0)
    assert r2 == 0.0


def test_phase49_audit_uses_only_train_and_calibration() -> None:
    result = evaluate_pftf_shear_identifiability()

    assert result.train_case_count == 60
    assert result.calibration_case_count == 30
    assert result.prohibited_held_out_case_count == 0
    assert result.selected_pftf.feature_name in {
        score.feature_name for score in result.pftf_training_scores
    }
    assert result.selected_geometry.feature_name in {
        score.feature_name for score in result.geometry_training_scores
    }
    assert 0.0 <= result.selected_pftf.calibration_median_within_block_r2 <= 1.0
    assert not result.pftf_reconstruction_value_supported
    assert not result.global_alpha_selection_supported
    assert not result.real_scan_transfer_supported
    assert not result.deployment_supported
