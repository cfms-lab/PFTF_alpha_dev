from __future__ import annotations

from dataclasses import replace

import pytest

from pftf_alpha.frozen_partition_reconstruction import (
    FINAL_HELD_OUT_SEED,
    VALIDATION_A_SEED,
    VALIDATION_B_SEED,
    _materialize_case,
    audit_case_seed_disjointness,
    evaluate_frozen_partition_reconstruction,
)
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


def _raw_case() -> MatchedPairStressRawCase:
    evidence = MatchedPairEvidence(
        information_boundary="observed",
        primary_point_count=10,
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
        matched_subset_endpoint=_endpoint(harm=True),
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


def test_missing_pair_route_uses_frozen_partition_endpoint() -> None:
    result = _materialize_case(_raw_case(), _accepting_model())

    assert result.frozen_partition_applied is True
    assert result.guarded_harmful_outlier_false_safe is False
    assert result.guarded_safe_accept is True
    assert result.routed_endpoint.geometry_topology_harm_present is False


def test_route_marks_new_frozen_partition_endpoint_harm() -> None:
    raw = replace(
        _raw_case(),
        stress=SensorStress.UPPER_OCCLUSION,
        endpoint=_endpoint(harm=False),
        frozen_partition_endpoint=_endpoint(harm=True),
    )
    result = _materialize_case(raw, _accepting_model())

    assert result.introduced_routed_endpoint_harm_accept is True
    assert result.guarded_safe_accept is False


def test_phase26_default_case_seeds_are_disjoint() -> None:
    audit = audit_case_seed_disjointness(
        VALIDATION_A_SEED,
        VALIDATION_B_SEED,
        FINAL_HELD_OUT_SEED,
    )

    assert audit.panel_case_count == 216
    assert audit.passed is True
    assert audit.validation_a_prior_overlap_count == 0
    assert audit.validation_a_b_overlap_count == 0


def test_phase26_rejects_legacy_sequential_base_seeds() -> None:
    with pytest.raises(ValueError, match="case seeds must be mutually disjoint"):
        evaluate_frozen_partition_reconstruction(
            validation_a_seed=25700804,
            validation_b_seed=25800804,
            final_held_out_seed=25900804,
        )


def test_phase26_reduced_protocol_does_not_open_fresh_seeds() -> None:
    result = evaluate_frozen_partition_reconstruction(
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
    assert result.phase26_supported is False
