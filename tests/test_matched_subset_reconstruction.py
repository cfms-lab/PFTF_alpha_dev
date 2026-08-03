from dataclasses import replace

import pytest

from pftf_alpha.local_surface_consensus import GeometryTopologyHarmEndpoint
from pftf_alpha.matched_guard_signature import (
    SIGNATURE_FEATURE_NAMES,
    MatchedGuardModel,
)
from pftf_alpha.matched_pair_consistency import MatchedPairEvidence
from pftf_alpha.matched_pair_stress import (
    MatchedPairStressProfile,
    MatchedPairStressRawCase,
)
from pftf_alpha.matched_subset_reconstruction import (
    FORBIDDEN_FRESH_SEEDS,
    _materialize_case,
    evaluate_matched_subset_reconstruction,
)
from pftf_alpha.sampling_gate import SamplingGateDecision
from pftf_alpha.sensor_stress import SensorStress


def _endpoint(*, harm: bool) -> GeometryTopologyHarmEndpoint:
    return GeometryTopologyHarmEndpoint(
        information_boundary="evaluation-only test",
        harmful_distance_threshold=0.1,
        source_outlier_vertex_count=1,
        used_source_outlier_vertex_count=1,
        provenance_violation_face_count=1 if harm else 0,
        provenance_violation_edge_count=1 if harm else 0,
        harmful_outlier_vertex_count=1 if harm else 0,
        harmful_outlier_face_count=2 if harm else 0,
        clean_cross_layer_face_count=0,
        component_error=0,
        betti_error=0,
        provenance_violation_present=harm,
        geometry_topology_harm_present=harm,
    )


def _raw_case() -> MatchedPairStressRawCase:
    evidence = MatchedPairEvidence(
        information_boundary="observed test",
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
        stress=SensorStress.OUTLIERS_01,
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
        endpoint=_endpoint(harm=True),
        matched_subset_endpoint=_endpoint(harm=False),
        frozen_partition_endpoint=_endpoint(harm=False),
        unguarded_decision=SamplingGateDecision.ACCEPT,
    )


def _accepting_model() -> MatchedGuardModel:
    dimension = len(SIGNATURE_FEATURE_NAMES)
    return MatchedGuardModel(
        feature_names=SIGNATURE_FEATURE_NAMES,
        feature_center=(0.0,) * dimension,
        feature_scale=(1.0,) * dimension,
        intercept=0.0,
        coefficients=(0.0,) * dimension,
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


def test_missing_pair_route_uses_matched_subset_endpoint() -> None:
    result = _materialize_case(_raw_case(), _accepting_model())

    assert result.matched_subset_applied is True
    assert result.unguarded_harmful_outlier_false_safe is True
    assert result.guarded_harmful_outlier_false_safe is False
    assert result.guarded_safe_accept is True
    assert result.guarded_decision is SamplingGateDecision.ACCEPT
    assert result.original_endpoint.geometry_topology_harm_present is True
    assert result.routed_endpoint.geometry_topology_harm_present is False


def test_exact_pair_route_keeps_original_endpoint() -> None:
    raw = replace(
        _raw_case(),
        profile=MatchedPairStressProfile.EXACT,
        missing_pair_count=0,
    )
    result = _materialize_case(raw, _accepting_model())

    assert result.matched_subset_applied is False
    assert result.guarded_harmful_outlier_false_safe is True
    assert result.routed_endpoint is raw.endpoint


def test_route_marks_newly_introduced_accepted_endpoint_harm() -> None:
    raw = replace(
        _raw_case(),
        stress=SensorStress.UPPER_OCCLUSION,
        endpoint=_endpoint(harm=False),
        matched_subset_endpoint=_endpoint(harm=True),
    )
    result = _materialize_case(raw, _accepting_model())

    assert result.guarded_decision is SamplingGateDecision.ACCEPT
    assert result.introduced_routed_endpoint_harm_accept is True
    assert result.guarded_safe_accept is False


def test_phase25_reduced_protocol_does_not_open_fresh_seeds() -> None:
    result = evaluate_matched_subset_reconstruction(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        validation_a_seed=307,
        validation_b_seed=311,
        final_held_out_seed=313,
        surface_sample_count=64,
    )

    assert result.design_score_fit.full_protocol is False
    assert result.design_cutoff_calibration.full_protocol is False
    assert result.design_gate_passed is False
    assert result.validation_a is None
    assert result.validation_b is None
    assert result.final_held_out is None
    assert result.phase25_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_supported is False
    assert result.real_trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase25_rejects_all_prior_opened_or_reserved_fresh_seeds() -> None:
    forbidden = max(FORBIDDEN_FRESH_SEEDS)
    with pytest.raises(ValueError, match="must not be reused"):
        evaluate_matched_subset_reconstruction(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            validation_a_seed=307,
            validation_b_seed=311,
            final_held_out_seed=forbidden,
            surface_sample_count=64,
        )
