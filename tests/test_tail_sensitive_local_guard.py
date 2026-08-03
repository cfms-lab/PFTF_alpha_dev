from __future__ import annotations

import math
from dataclasses import replace

import pytest

from pftf_alpha.local_spatial_displacement import (
    LocalSpatialDisplacementEvidence,
)
from pftf_alpha.local_spatial_residual_guard import LocalSpatialGuardCase
from pftf_alpha.local_surface_consensus import GeometryTopologyHarmEndpoint
from pftf_alpha.matched_pair_stress import MatchedPairStressProfile
from pftf_alpha.sampling_gate import SamplingGateDecision
from pftf_alpha.sensor_stress import SensorStress
from pftf_alpha.tail_sensitive_local_guard import (
    FINAL_HELD_OUT_SEED,
    VALIDATION_A_SEED,
    VALIDATION_B_SEED,
    TailFeatureCandidate,
    _materialize_tail_case,
    audit_phase30_case_seed_disjointness,
    select_tail_candidate,
    tail_feature_value,
)


def _evidence() -> LocalSpatialDisplacementEvidence:
    return LocalSpatialDisplacementEvidence(
        information_boundary="test",
        point_count=96,
        neighbor_count=8,
        maximum_local_residual=5.0,
        support_local_residual=3.5,
        percentile95_local_residual=2.0,
        maximum_local_score_excess=4.0,
        peak_neighbor_score_support_fraction=0.25,
        median_neighbor_radius_fraction=0.1,
        peak_neighbor_radius_fraction=0.2,
    )


def _candidate(
    feature_name: str,
    *,
    focus: int,
    safe: int,
    eligible: bool = True,
) -> TailFeatureCandidate:
    return TailFeatureCandidate(
        feature_name=feature_name,
        rejection_direction="reject_if_greater_than_or_equal_to_cutoff",
        residual_harmful_case_count=3,
        limiting_tail_value=2.0,
        rejection_cutoff=math.nextafter(2.0, -math.inf),
        binary64_decrement=2.0 - math.nextafter(2.0, -math.inf),
        original_focus_safe_accept_count=100,
        predecessor_focus_safe_accept_count=99,
        combined_focus_safe_accept_count=focus,
        original_safe_accept_count=200,
        predecessor_safe_accept_count=198,
        combined_safe_accept_count=safe,
        design_panel_count=9,
        passing_design_panel_count=9 if eligible else 8,
        all_design_gates_passed=eligible,
        eligible=eligible,
    )


def _endpoint(*, harmful: bool) -> GeometryTopologyHarmEndpoint:
    return GeometryTopologyHarmEndpoint(
        information_boundary="test",
        harmful_distance_threshold=0.1,
        source_outlier_vertex_count=int(harmful),
        used_source_outlier_vertex_count=int(harmful),
        provenance_violation_face_count=0,
        provenance_violation_edge_count=0,
        harmful_outlier_vertex_count=int(harmful),
        harmful_outlier_face_count=4 if harmful else 0,
        clean_cross_layer_face_count=0,
        component_error=0,
        betti_error=0,
        provenance_violation_present=False,
        geometry_topology_harm_present=harmful,
    )


def test_isolated_tail_features_measure_peak_above_q95() -> None:
    evidence = _evidence()

    assert tail_feature_value(evidence, "isolated_tail_gap") == 3.0
    assert tail_feature_value(evidence, "isolated_tail_ratio") == 2.5
    assert tail_feature_value(evidence, "support_tail_gap") == 1.5


def test_unknown_tail_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tail feature"):
        tail_feature_value(_evidence(), "endpoint_truth")


def test_tail_rejection_marks_harmful_original_accept_unsupported() -> None:
    harmful = _endpoint(harmful=True)
    case = LocalSpatialGuardCase(
        profile=MatchedPairStressProfile.EXACT,
        stress=SensorStress.OUTLIERS_01,
        point_count=96,
        repeat=1,
        seed=1,
        replicate_seed=2,
        perturbation_seed=3,
        model_score=0.1,
        local_spatial_evidence=_evidence(),
        original_endpoint=harmful,
        routed_endpoint=harmful,
        unguarded_decision=SamplingGateDecision.ACCEPT,
        predecessor_guarded_decision=SamplingGateDecision.ACCEPT,
        guarded_decision=SamplingGateDecision.ACCEPT,
        unguarded_safe_accept=False,
        predecessor_guarded_safe_accept=False,
        guarded_safe_accept=False,
        unguarded_harmful_outlier_false_safe=True,
        predecessor_guarded_harmful_outlier_false_safe=True,
        guarded_harmful_outlier_false_safe=True,
        introduced_routed_endpoint_harm_accept=False,
    )

    materialized = _materialize_tail_case(
        case,
        feature_name="isolated_tail_ratio",
        rejection_cutoff=2.0,
    )

    assert materialized.guarded_decision is SamplingGateDecision.UNSUPPORTED
    assert materialized.guarded_harmful_outlier_false_safe is False


def test_default_phase30_case_seeds_are_disjoint() -> None:
    audit = audit_phase30_case_seed_disjointness(
        VALIDATION_A_SEED,
        VALIDATION_B_SEED,
        FINAL_HELD_OUT_SEED,
    )

    assert audit.passed is True
    assert audit.panel_case_count == 192
    assert audit.targeted_prior_base_seeds == (30200804, 30300804, 30400804)


def test_phase29_reserved_seed_is_not_fresh_for_phase30() -> None:
    audit = audit_phase30_case_seed_disjointness(
        30400804,
        VALIDATION_B_SEED,
        FINAL_HELD_OUT_SEED,
    )

    assert audit.passed is False
    assert audit.validation_a_prior_overlap_count == 192


def test_candidate_selection_prefers_focus_then_safe_then_declared_order() -> None:
    maximum = _candidate("maximum_local_residual", focus=95, safe=180)
    ratio = _candidate("isolated_tail_ratio", focus=97, safe=175)
    score_excess = _candidate(
        "maximum_local_score_excess",
        focus=97,
        safe=176,
    )

    assert select_tail_candidate((maximum, ratio, score_excess)) == score_excess
    tied_score = replace(score_excess, combined_safe_accept_count=175)
    assert (
        select_tail_candidate(
            (
                tied_score,
                ratio,
            )
        )
        == tied_score
    )
