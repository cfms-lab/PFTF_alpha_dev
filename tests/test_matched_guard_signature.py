from typing import cast

import numpy as np
import pytest

from pftf_alpha.local_surface_consensus import GeometryTopologyHarmEndpoint
from pftf_alpha.matched_guard_signature import (
    FORBIDDEN_PRIOR_SEEDS,
    GUARD_PROFILE_SPECS,
    SIGNATURE_FEATURE_NAMES,
    MatchedGuardSignature,
    evaluate_matched_guard_signature,
    fit_matched_guard_model,
    matched_guard_signature,
    matched_guard_signature_from_evidence,
    score_matched_guard_signature,
)
from pftf_alpha.matched_pair_consistency import MatchedPairEvidence
from pftf_alpha.matched_pair_stress import (
    MatchedPairStressProfile,
    MatchedPairStressRawCase,
)
from pftf_alpha.sampling_gate import SamplingGateDecision
from pftf_alpha.sensor_stress import SensorStress


def _direct_signature(value: float) -> MatchedGuardSignature:
    return MatchedGuardSignature(
        values=(value,) * len(SIGNATURE_FEATURE_NAMES)
    )


def _raw_case() -> MatchedPairStressRawCase:
    evidence = MatchedPairEvidence(
        information_boundary="test",
        primary_point_count=9,
        repeat_point_count=9,
        observed_characteristic_length=2.0,
        displacement_location=(0.01, -0.02, 0.03),
        axis_scales=(0.02, 0.03, 0.04),
        median_standardized_displacement=1.0,
        percentile95_standardized_displacement=2.0,
        peak_standardized_displacement=5.0,
        support_standardized_displacement=2.5,
        leading_standardized_displacements=(5.0, 2.5, 2.0, 1.5),
        maximum_centered_displacement=0.2,
    )
    return MatchedPairStressRawCase(
        profile=MatchedPairStressProfile.MISSING_10PCT,
        stress=SensorStress.CONTROL,
        point_count=10,
        repeat=0,
        seed=11,
        replicate_seed=13,
        perturbation_seed=17,
        repeat_transient_outlier_count=0,
        repeat_transient_outlier_index_sha256="0" * 64,
        retained_pair_count=9,
        missing_pair_count=1,
        mismatched_pair_count=0,
        rotation_degrees=0.0,
        rotation_axis=(0.0, 0.0, 1.0),
        presented_pair_map_sha256="1" * 64,
        evidence=evidence,
        endpoint=cast(GeometryTopologyHarmEndpoint, None),
        matched_subset_endpoint=cast(GeometryTopologyHarmEndpoint, None),
        frozen_partition_endpoint=cast(GeometryTopologyHarmEndpoint, None),
        unguarded_decision=SamplingGateDecision.ACCEPT,
    )


def test_matched_guard_signature_is_finite_and_has_frozen_dimension() -> None:
    raw = _raw_case()
    signature = matched_guard_signature(raw)
    direct = matched_guard_signature_from_evidence(
        raw.evidence,
        retained_pair_count=raw.retained_pair_count,
        point_count=raw.point_count,
    )

    assert len(signature.values) == len(SIGNATURE_FEATURE_NAMES)
    assert np.all(np.isfinite(signature.values))
    assert signature.values[1] == pytest.approx(0.9)
    assert direct == signature


def test_direct_signature_rejects_invalid_pair_counts() -> None:
    raw = _raw_case()

    with pytest.raises(ValueError, match="point_count must be at least"):
        matched_guard_signature_from_evidence(
            raw.evidence,
            retained_pair_count=11,
            point_count=10,
        )


def test_guard_cutoff_rejects_every_harmful_training_case() -> None:
    safe = (_direct_signature(-2.0), _direct_signature(-1.0))
    harmful = (_direct_signature(1.0), _direct_signature(2.0))
    signatures = safe + harmful
    labels = (False, False, True, True)

    model = fit_matched_guard_model(signatures, labels)

    assert model.calibration_valid is True
    assert model.retained_training_harmful_case_count == 0
    assert model.rejected_training_harmful_case_count == len(harmful)
    assert all(
        score_matched_guard_signature(model, row) >= model.rejection_cutoff
        for row in harmful
    )
    assert any(
        score_matched_guard_signature(model, row) < model.rejection_cutoff
        for row in safe
    )


def test_phase23_reduced_panel_cannot_open_later_seeds() -> None:
    result = evaluate_matched_guard_signature(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        training_a_seed=263,
        development_b_seed=269,
        validation_a_seed=271,
        validation_b_seed=277,
        final_held_out_seed=281,
        surface_sample_count=64,
    )

    assert result.training_a.case_count == len(GUARD_PROFILE_SPECS)
    assert result.training_a.full_protocol is False
    assert result.training_a.panel_gate_passed is False
    assert result.development_b is None
    assert result.development_screen_passed is False
    assert result.validation_a is None
    assert result.validation_b is None
    assert result.final_held_out is None
    assert result.phase23_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase23_rejects_all_prior_opened_or_reserved_seeds() -> None:
    forbidden = max(FORBIDDEN_PRIOR_SEEDS)
    with pytest.raises(ValueError, match="must not be reused"):
        evaluate_matched_guard_signature(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            training_a_seed=263,
            development_b_seed=269,
            validation_a_seed=271,
            validation_b_seed=277,
            final_held_out_seed=forbidden,
            surface_sample_count=64,
        )
