from __future__ import annotations

import math

import pytest

from pftf_alpha.focus_envelope_cutoff import (
    FINAL_HELD_OUT_SEED,
    VALIDATION_A_SEED,
    VALIDATION_B_SEED,
    audit_phase27_case_seed_disjointness,
    calibrate_focus_envelope_cutoff,
    evaluate_focus_envelope_cutoff,
)
from pftf_alpha.local_surface_consensus import GeometryTopologyHarmEndpoint
from pftf_alpha.matched_guard_signature import (
    SIGNATURE_FEATURE_NAMES,
    MatchedGuardModel,
    MatchedGuardSignature,
)
from pftf_alpha.matched_pair_consistency import MatchedPairEvidence
from pftf_alpha.matched_pair_stress import (
    MatchedPairStressProfile,
    MatchedPairStressRawCase,
)
from pftf_alpha.sampling_gate import SamplingGateDecision
from pftf_alpha.sensor_stress import SensorStress


def _endpoint(*, harm: bool) -> GeometryTopologyHarmEndpoint:
    return GeometryTopologyHarmEndpoint(
        information_boundary="evaluation_only",
        harmful_distance_threshold=0.1,
        source_outlier_vertex_count=int(harm),
        used_source_outlier_vertex_count=int(harm),
        provenance_violation_face_count=int(harm),
        provenance_violation_edge_count=int(harm),
        harmful_outlier_vertex_count=int(harm),
        harmful_outlier_face_count=int(harm),
        clean_cross_layer_face_count=0,
        component_error=0,
        betti_error=0,
        provenance_violation_present=harm,
        geometry_topology_harm_present=harm,
    )


def _model() -> MatchedGuardModel:
    dimension = len(SIGNATURE_FEATURE_NAMES)
    return MatchedGuardModel(
        feature_names=SIGNATURE_FEATURE_NAMES,
        feature_center=(0.0,) * dimension,
        feature_scale=(1.0,) * dimension,
        intercept=0.0,
        coefficients=(1.0,) + (0.0,) * (dimension - 1),
        rejection_cutoff=1.0,
        ridge_penalty=1.0,
        calibration_valid=True,
        training_case_count=2,
        training_harmful_case_count=1,
        training_safe_case_count=1,
        rejected_training_harmful_case_count=1,
        retained_training_harmful_case_count=0,
        rejected_training_safe_case_count=0,
        retained_training_safe_case_count=1,
    )


def _raw_case(
    *,
    score: float,
    stress: SensorStress,
    harm: bool,
) -> MatchedPairStressRawCase:
    evidence = MatchedPairEvidence(
        information_boundary="observed",
        primary_point_count=10,
        repeat_point_count=10,
        observed_characteristic_length=2.0,
        displacement_location=(0.0, 0.0, 0.0),
        axis_scales=(0.02, 0.03, 0.04),
        median_standardized_displacement=1.0,
        percentile95_standardized_displacement=2.0,
        peak_standardized_displacement=5.0,
        support_standardized_displacement=2.5,
        leading_standardized_displacements=(5.0, 2.5, 2.0, 1.5),
        maximum_centered_displacement=0.2,
    )
    return MatchedPairStressRawCase(
        profile=MatchedPairStressProfile.EXACT,
        stress=stress,
        point_count=10,
        repeat=0,
        seed=11,
        replicate_seed=13,
        perturbation_seed=17,
        repeat_transient_outlier_count=0,
        repeat_transient_outlier_index_sha256="0" * 64,
        retained_pair_count=max(1, round(math.exp(score))),
        missing_pair_count=0,
        mismatched_pair_count=0,
        rotation_degrees=0.0,
        rotation_axis=(0.0, 0.0, 1.0),
        presented_pair_map_sha256="1" * 64,
        evidence=evidence,
        endpoint=_endpoint(harm=harm),
        matched_subset_endpoint=_endpoint(harm=harm),
        frozen_partition_endpoint=_endpoint(harm=harm),
        unguarded_decision=SamplingGateDecision.ACCEPT,
    )


def test_focus_envelope_cutoff_is_next_float_above_maximum_focus() -> None:
    safe = _raw_case(score=1.0, stress=SensorStress.CONTROL, harm=False)
    harmful = _raw_case(score=2.0, stress=SensorStress.OUTLIERS_01, harm=True)
    signatures = (
        MatchedGuardSignature((0.0,) * len(SIGNATURE_FEATURE_NAMES)),
        MatchedGuardSignature((2.0,) + (0.0,) * (len(SIGNATURE_FEATURE_NAMES) - 1)),
    )

    model, audit = calibrate_focus_envelope_cutoff(
        _model(),
        score_fit_signatures=signatures,
        score_fit_harmful_labels=(False, True),
        design_a_rows=(safe, harmful),
        design_b_rows=(),
    )

    assert audit.calibration_valid is True
    assert audit.maximum_focus_safe_score == pytest.approx(math.log(3.0))
    assert model.rejection_cutoff == math.nextafter(
        audit.maximum_focus_safe_score,
        math.inf,
    )


def test_phase27_default_case_seeds_are_disjoint() -> None:
    audit = audit_phase27_case_seed_disjointness(
        VALIDATION_A_SEED,
        VALIDATION_B_SEED,
        FINAL_HELD_OUT_SEED,
    )

    assert audit.panel_case_count == 216
    assert audit.passed is True
    assert audit.validation_a_prior_overlap_count == 0
    assert audit.validation_a_b_overlap_count == 0


def test_phase27_rejects_phase26_seed_reuse() -> None:
    with pytest.raises(ValueError, match="Phase-26 panels"):
        evaluate_focus_envelope_cutoff(
            validation_a_seed=27500804,
            validation_b_seed=27600804,
            final_held_out_seed=27700804,
        )


def test_phase27_reduced_protocol_does_not_open_fresh_seeds() -> None:
    result = evaluate_focus_envelope_cutoff(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        surface_sample_count=64,
    )

    assert result.case_seed_disjointness.passed is True
    assert result.design_gate_passed is False
    assert result.validation_a is None
    assert result.validation_b is None
    assert result.final_held_out is None
    assert result.phase27_supported is False
