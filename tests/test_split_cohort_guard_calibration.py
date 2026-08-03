import pytest

from pftf_alpha.matched_guard_signature import (
    GUARD_PROFILE_SPECS,
    SIGNATURE_FEATURE_NAMES,
    MatchedGuardSignature,
    fit_matched_guard_model,
    score_matched_guard_signature,
)
from pftf_alpha.sensor_stress import SensorStress
from pftf_alpha.split_cohort_guard_calibration import (
    CONSERVATIVE_GAP_FRACTION,
    FORBIDDEN_PRIOR_SEEDS,
    calibrate_split_cohort_cutoff,
    evaluate_split_cohort_guard_calibration,
)


def _signature(value: float) -> MatchedGuardSignature:
    return MatchedGuardSignature(
        values=(value,) * len(SIGNATURE_FEATURE_NAMES)
    )


def test_split_cohort_cutoff_uses_only_calibration_gap() -> None:
    fit_signatures = tuple(
        _signature(value) for value in (-2.0, -1.0, 1.0, 2.0)
    )
    fit_harmful = (False, False, True, True)
    score_model = fit_matched_guard_model(fit_signatures, fit_harmful)
    calibration_signatures = tuple(
        _signature(value) for value in (-1.5, -0.5, 0.5, 1.5)
    )
    calibration_harmful = (False, False, True, True)
    calibration_focus = (True, True, False, False)

    model, summary = calibrate_split_cohort_cutoff(
        score_model,
        score_fit_signatures=fit_signatures,
        score_fit_harmful_labels=fit_harmful,
        calibration_signatures=calibration_signatures,
        calibration_harmful_labels=calibration_harmful,
        calibration_focus_safe_labels=calibration_focus,
        calibration_seed=101,
    )

    expected = summary.maximum_focus_safe_score + CONSERVATIVE_GAP_FRACTION * (
        summary.minimum_harmful_score - summary.maximum_focus_safe_score
    )
    assert summary.calibration_valid is True
    assert summary.rejection_cutoff == pytest.approx(expected)
    assert model.rejection_cutoff == pytest.approx(expected)
    assert summary.retained_harmful_case_count == 0
    assert summary.rejected_focus_safe_case_count == 0
    assert all(
        score_matched_guard_signature(model, signature) >= model.rejection_cutoff
        for signature in calibration_signatures[2:]
    )
    assert all(
        score_matched_guard_signature(model, signature) < model.rejection_cutoff
        for signature in calibration_signatures[:2]
    )


def test_phase24_reduced_panel_cannot_open_validation_seeds() -> None:
    result = evaluate_split_cohort_guard_calibration(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        score_fit_seed=283,
        cutoff_calibration_seed=293,
        validation_a_seed=307,
        validation_b_seed=311,
        final_held_out_seed=313,
        surface_sample_count=64,
    )

    assert result.score_fit.case_count == len(GUARD_PROFILE_SPECS)
    assert result.cutoff_calibration_panel.case_count == len(GUARD_PROFILE_SPECS)
    assert result.score_fit.full_protocol is False
    assert result.cutoff_calibration.calibration_valid is False
    assert result.prevalidation_gate_passed is False
    assert result.validation_a is None
    assert result.validation_b is None
    assert result.final_held_out is None
    assert result.phase24_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase24_rejects_all_prior_opened_or_reserved_seeds() -> None:
    forbidden = max(FORBIDDEN_PRIOR_SEEDS)
    with pytest.raises(ValueError, match="must not be reused"):
        evaluate_split_cohort_guard_calibration(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            score_fit_seed=283,
            cutoff_calibration_seed=293,
            validation_a_seed=307,
            validation_b_seed=311,
            final_held_out_seed=forbidden,
            surface_sample_count=64,
        )


def test_split_cohort_gap_fraction_must_be_internal() -> None:
    signatures = (_signature(-1.0), _signature(1.0))
    harmful = (False, True)
    score_model = fit_matched_guard_model(signatures, harmful)

    with pytest.raises(ValueError, match="strictly between"):
        calibrate_split_cohort_cutoff(
            score_model,
            score_fit_signatures=signatures,
            score_fit_harmful_labels=harmful,
            calibration_signatures=signatures,
            calibration_harmful_labels=harmful,
            calibration_focus_safe_labels=(True, False),
            calibration_seed=101,
            gap_fraction=1.0,
        )
