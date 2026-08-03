"""Global tangential one-to-one correspondence audit for Phase 20."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from .local_insertion_influence import InfluenceRectangle
from .local_surface_consensus import GeometryTopologyHarmEndpoint
from .matched_pair_consistency import (
    MatchedPairConfig,
    MatchedPairEvidence,
    estimate_matched_pair_evidence,
)
from .matched_pair_stress import (
    DEFAULT_STRESS_SPECS,
    MatchedPairStressConfig,
    MatchedPairStressProfile,
    MatchedPairStressSpec,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .sensor_stress import DEFAULT_POINT_COUNTS, DEFAULT_STRESSES, SensorStress
from .shared_trend_inference import SharedTrendConfig
from .tangential_pair_confidence import (
    FROZEN_PHASE18_RECTANGLE,
    PairConfidenceConfig,
    PairConfidenceScores,
    TangentialPairRawCase,
    _local_normals_and_spacings,
    _raw_panel,
    tangential_pair_confidence_scores,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CALIBRATION_A_SEED = 23300804
CALIBRATION_B_SEED = 23400804
FINAL_HELD_OUT_SEED = 23500804
PHASE19_FINAL_HELD_OUT_SEED = 23200804


@dataclass(frozen=True)
class GlobalTangentialAssignment:
    repeat_row_for_primary: IntArray
    presented_costs: FloatArray
    assigned_costs: FloatArray
    presented_normal_costs: FloatArray
    assigned_normal_costs: FloatArray
    confidence: PairConfidenceScores
    tie_perturbation_unit: float


@dataclass(frozen=True)
class GlobalAssignmentCaseResult:
    profile: MatchedPairStressProfile
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    perturbation_seed: int
    repeat_transient_outlier_count: int
    repeat_transient_outlier_index_sha256: str
    missing_pair_count: int
    rotation_degrees: float
    rotation_axis: tuple[float, float, float]
    presented_pair_map_sha256: str
    pair_count: int
    presented_mismatch_pair_count: int
    changed_assignment_count: int
    correct_assignment_count: int
    assignment_accuracy: float
    repaired_presented_mismatch_count: int
    presented_mismatch_repair_fraction: float
    introduced_mismatch_count: int
    assigned_cost_minimum: float
    assigned_cost_median: float
    assigned_cost_percentile95: float
    assigned_cost_maximum: float
    alignment_rotation_degrees: float
    alignment_translation: tuple[float, float, float]
    alignment_inlier_count: int
    evidence: MatchedPairEvidence
    endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision
    guarded_decision: SamplingGateDecision
    unguarded_safe_accept: bool
    guarded_safe_accept: bool
    unguarded_harmful_outlier_false_safe: bool
    guarded_harmful_outlier_false_safe: bool
    unguarded_provenance_violation_accept: bool
    guarded_provenance_violation_accept: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        payload["stress"] = self.stress.value
        payload["unguarded_decision"] = self.unguarded_decision.value
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class GlobalAssignmentProfileSummary:
    profile: MatchedPairStressProfile
    case_count: int
    pair_count: int
    correct_assignment_count: int
    assignment_accuracy: float
    presented_mismatch_pair_count: int
    repaired_presented_mismatch_count: int
    presented_mismatch_repair_fraction: float
    introduced_mismatch_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    assignment_gate_passed: bool
    profile_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        return payload


@dataclass(frozen=True)
class GlobalAssignmentPanel:
    panel_role: str
    seed: int
    frozen_rectangle: InfluenceRectangle
    cases: tuple[GlobalAssignmentCaseResult, ...]
    profile_summaries: tuple[GlobalAssignmentProfileSummary, ...]
    case_count: int
    pair_count: int
    correct_assignment_count: int
    assignment_accuracy: float
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    full_protocol: bool
    panel_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_role": self.panel_role,
            "seed": self.seed,
            "frozen_rectangle": self.frozen_rectangle.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "profile_summaries": [
                summary.to_dict() for summary in self.profile_summaries
            ],
            "case_count": self.case_count,
            "pair_count": self.pair_count,
            "correct_assignment_count": self.correct_assignment_count,
            "assignment_accuracy": self.assignment_accuracy,
            "unguarded_harmful_outlier_false_safe_count": (
                self.unguarded_harmful_outlier_false_safe_count
            ),
            "guarded_harmful_outlier_false_safe_count": (
                self.guarded_harmful_outlier_false_safe_count
            ),
            "focus_unguarded_safe_accept_count": (
                self.focus_unguarded_safe_accept_count
            ),
            "focus_guarded_safe_accept_count": self.focus_guarded_safe_accept_count,
            "focus_safe_accept_retention": self.focus_safe_accept_retention,
            "full_protocol": self.full_protocol,
            "panel_gate_passed": self.panel_gate_passed,
        }


@dataclass(frozen=True)
class GlobalTangentialAssignmentResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    calibration_a_seed: int
    calibration_b_seed: int
    final_held_out_seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    matched_pair_config: MatchedPairConfig
    stress_config: MatchedPairStressConfig
    confidence_config: PairConfidenceConfig
    profile_specs: tuple[MatchedPairStressSpec, ...]
    frozen_rectangle: InfluenceRectangle
    assignment_accuracy_gate: float
    mismatch_repair_gate: float
    calibration_a: GlobalAssignmentPanel
    calibration_b: GlobalAssignmentPanel
    final_held_out: GlobalAssignmentPanel | None
    phase20_supported: bool
    global_tangential_assignment_synthetic_supported: bool
    real_correspondence_supported: bool
    real_paired_scan_supported: bool
    trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_predecessor": self.frozen_predecessor,
            "calibration_a_seed": self.calibration_a_seed,
            "calibration_b_seed": self.calibration_b_seed,
            "final_held_out_seed": self.final_held_out_seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "matched_pair_config": asdict(self.matched_pair_config),
            "stress_config": asdict(self.stress_config),
            "confidence_config": asdict(self.confidence_config),
            "profile_specs": [
                {**asdict(spec), "profile": spec.profile.value}
                for spec in self.profile_specs
            ],
            "frozen_rectangle": self.frozen_rectangle.to_dict(),
            "assignment_accuracy_gate": self.assignment_accuracy_gate,
            "mismatch_repair_gate": self.mismatch_repair_gate,
            "calibration_a": self.calibration_a.to_dict(),
            "calibration_b": self.calibration_b.to_dict(),
            "final_held_out": (
                None
                if self.final_held_out is None
                else self.final_held_out.to_dict()
            ),
            "phase20_supported": self.phase20_supported,
            "global_tangential_assignment_synthetic_supported": (
                self.global_tangential_assignment_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def _assignment_from_confidence(
    primary: FloatArray,
    confidence: PairConfidenceScores,
    config: PairConfidenceConfig,
) -> GlobalTangentialAssignment:
    minimum_spacing = (
        config.minimum_spacing_fraction * confidence.observed_characteristic_length
    )
    normals, spacings = _local_normals_and_spacings(
        primary,
        neighbor_count=config.local_neighbor_count,
        minimum_spacing=minimum_spacing,
    )
    offsets = primary[:, None, :] - confidence.aligned_repeat_points[None, :, :]
    normal_coordinates = np.einsum("ijk,ik->ij", offsets, normals)
    tangential = offsets - normal_coordinates[:, :, None] * normals[:, None, :]
    costs = np.linalg.norm(tangential, axis=2) / spacings[:, None]
    normal_costs = np.abs(normal_coordinates) / spacings[:, None]
    scale = max(1.0, float(np.max(costs)))
    perturbation_unit = float(np.spacing(scale))
    count = primary.shape[0]
    ranks = np.arange(count * count, dtype=np.float64).reshape(count, count) + 1.0
    tie_break = perturbation_unit * (ranks / (count * count + 1.0)) ** 2
    row_indices, repeat_rows = linear_sum_assignment(costs + tie_break)
    assignment = np.empty(count, dtype=np.int64)
    assignment[row_indices] = repeat_rows
    return GlobalTangentialAssignment(
        repeat_row_for_primary=assignment,
        presented_costs=np.diag(costs).copy(),
        assigned_costs=costs[np.arange(count), assignment],
        presented_normal_costs=np.diag(normal_costs).copy(),
        assigned_normal_costs=normal_costs[np.arange(count), assignment],
        confidence=confidence,
        tie_perturbation_unit=perturbation_unit,
    )


def global_tangential_assignment(
    primary_points: FloatArray,
    repeat_points: FloatArray,
    config: PairConfidenceConfig | None = None,
) -> GlobalTangentialAssignment:
    """Assign repeat rows globally using observed local-tangent costs only."""

    selected = PairConfidenceConfig() if config is None else config
    primary = np.asarray(primary_points, dtype=np.float64)
    repeat = np.asarray(repeat_points, dtype=np.float64)
    confidence = tangential_pair_confidence_scores(primary, repeat, selected)
    return _assignment_from_confidence(primary, confidence, selected)


def _materialize_case(
    raw: TangentialPairRawCase,
    *,
    rectangle: InfluenceRectangle,
    confidence_config: PairConfidenceConfig,
    matched_pair_config: MatchedPairConfig,
) -> GlobalAssignmentCaseResult:
    assignment = _assignment_from_confidence(
        raw.primary_points,
        raw.confidence,
        confidence_config,
    )
    repeat_rows = assignment.repeat_row_for_primary
    assigned_source_ids = raw.repeat_source_ids[repeat_rows]
    correct_assignment = raw.primary_ids == assigned_source_ids
    presented_mismatch = raw.primary_ids != raw.repeat_source_ids
    repaired_presented = presented_mismatch & correct_assignment
    introduced = (~presented_mismatch) & (~correct_assignment)
    evidence = replace(
        estimate_matched_pair_evidence(
            raw.primary_points,
            raw.repeat_points[repeat_rows],
            matched_pair_config,
        ),
        information_boundary=(
            "repeat_rows_reordered_by_global_observed_tangential_assignment; "
            "pair_source_truth_unknown_to_route"
        ),
    )
    consistent = bool(
        evidence.peak_standardized_displacement <= rectangle.peak_threshold
        and evidence.support_standardized_displacement <= rectangle.support_threshold
    )
    unguarded_accept = raw.unguarded_decision is SamplingGateDecision.ACCEPT
    guarded_decision = (
        SamplingGateDecision.UNSUPPORTED
        if unguarded_accept and not consistent
        else raw.unguarded_decision
    )
    guarded_accept = guarded_decision is SamplingGateDecision.ACCEPT
    harmful_outlier = bool(
        raw.stress.is_outlier_stress
        and raw.endpoint.geometry_topology_harm_present
    )
    pair_count = int(correct_assignment.size)
    correct_count = int(np.sum(correct_assignment))
    presented_mismatch_count = int(np.sum(presented_mismatch))
    repaired_count = int(np.sum(repaired_presented))
    costs = assignment.assigned_costs
    return GlobalAssignmentCaseResult(
        profile=raw.profile,
        stress=raw.stress,
        point_count=raw.point_count,
        repeat=raw.repeat,
        seed=raw.seed,
        replicate_seed=raw.replicate_seed,
        perturbation_seed=raw.perturbation_seed,
        repeat_transient_outlier_count=raw.repeat_transient_outlier_count,
        repeat_transient_outlier_index_sha256=(
            raw.repeat_transient_outlier_index_sha256
        ),
        missing_pair_count=raw.missing_pair_count,
        rotation_degrees=raw.rotation_degrees,
        rotation_axis=raw.rotation_axis,
        presented_pair_map_sha256=raw.presented_pair_map_sha256,
        pair_count=pair_count,
        presented_mismatch_pair_count=presented_mismatch_count,
        changed_assignment_count=int(np.sum(repeat_rows != np.arange(pair_count))),
        correct_assignment_count=correct_count,
        assignment_accuracy=correct_count / pair_count,
        repaired_presented_mismatch_count=repaired_count,
        presented_mismatch_repair_fraction=(
            1.0
            if not presented_mismatch_count
            else repaired_count / presented_mismatch_count
        ),
        introduced_mismatch_count=int(np.sum(introduced)),
        assigned_cost_minimum=float(np.min(costs)),
        assigned_cost_median=float(np.median(costs)),
        assigned_cost_percentile95=float(np.percentile(costs, 95.0)),
        assigned_cost_maximum=float(np.max(costs)),
        alignment_rotation_degrees=(
            assignment.confidence.alignment_rotation_degrees
        ),
        alignment_translation=assignment.confidence.alignment_translation,
        alignment_inlier_count=assignment.confidence.alignment_inlier_count,
        evidence=evidence,
        endpoint=raw.endpoint,
        unguarded_decision=raw.unguarded_decision,
        guarded_decision=guarded_decision,
        unguarded_safe_accept=bool(
            unguarded_accept and not raw.endpoint.geometry_topology_harm_present
        ),
        guarded_safe_accept=bool(
            guarded_accept and not raw.endpoint.geometry_topology_harm_present
        ),
        unguarded_harmful_outlier_false_safe=bool(
            unguarded_accept and harmful_outlier
        ),
        guarded_harmful_outlier_false_safe=bool(
            guarded_accept and harmful_outlier
        ),
        unguarded_provenance_violation_accept=bool(
            unguarded_accept and raw.endpoint.provenance_violation_present
        ),
        guarded_provenance_violation_accept=bool(
            guarded_accept and raw.endpoint.provenance_violation_present
        ),
    )


def _profile_summary(
    rows: Sequence[GlobalAssignmentCaseResult],
    spec: MatchedPairStressSpec,
    *,
    full_protocol: bool,
    assignment_accuracy_gate: float,
    mismatch_repair_gate: float,
) -> GlobalAssignmentProfileSummary:
    selected = [row for row in rows if row.profile is spec.profile]
    pair_count = sum(row.pair_count for row in selected)
    correct_count = sum(row.correct_assignment_count for row in selected)
    accuracy = 0.0 if not pair_count else correct_count / pair_count
    presented_mismatch_count = sum(
        row.presented_mismatch_pair_count for row in selected
    )
    repaired_count = sum(
        row.repaired_presented_mismatch_count for row in selected
    )
    repair_fraction = (
        1.0
        if not presented_mismatch_count
        else repaired_count / presented_mismatch_count
    )
    focus = [
        row
        for row in selected
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    focus_retention = (
        0.0 if not unguarded_focus else guarded_focus / unguarded_focus
    )
    unguarded_harm = sum(
        row.unguarded_harmful_outlier_false_safe for row in selected
    )
    guarded_harm = sum(
        row.guarded_harmful_outlier_false_safe for row in selected
    )
    assignment_gate = bool(
        accuracy >= assignment_accuracy_gate
        and (
            spec.mismatch_fraction == 0.0
            or (
                presented_mismatch_count > 0
                and repair_fraction >= mismatch_repair_gate
            )
        )
    )
    return GlobalAssignmentProfileSummary(
        profile=spec.profile,
        case_count=len(selected),
        pair_count=pair_count,
        correct_assignment_count=correct_count,
        assignment_accuracy=accuracy,
        presented_mismatch_pair_count=presented_mismatch_count,
        repaired_presented_mismatch_count=repaired_count,
        presented_mismatch_repair_fraction=repair_fraction,
        introduced_mismatch_count=sum(
            row.introduced_mismatch_count for row in selected
        ),
        unguarded_harmful_outlier_false_safe_count=unguarded_harm,
        guarded_harmful_outlier_false_safe_count=guarded_harm,
        unguarded_provenance_violation_accept_count=sum(
            row.unguarded_provenance_violation_accept for row in selected
        ),
        guarded_provenance_violation_accept_count=sum(
            row.guarded_provenance_violation_accept for row in selected
        ),
        focus_unguarded_safe_accept_count=unguarded_focus,
        focus_guarded_safe_accept_count=guarded_focus,
        focus_safe_accept_retention=focus_retention,
        assignment_gate_passed=assignment_gate,
        profile_gate_passed=bool(
            full_protocol
            and assignment_gate
            and unguarded_harm > 0
            and guarded_harm == 0
            and focus_retention >= 0.90
        ),
    )


def _materialize_panel(
    raw_rows: tuple[TangentialPairRawCase, ...],
    *,
    panel_role: str,
    seed: int,
    rectangle: InfluenceRectangle,
    confidence_config: PairConfidenceConfig,
    matched_pair_config: MatchedPairConfig,
    profile_specs: tuple[MatchedPairStressSpec, ...],
    full_protocol: bool,
    assignment_accuracy_gate: float,
    mismatch_repair_gate: float,
) -> GlobalAssignmentPanel:
    rows = tuple(
        _materialize_case(
            raw,
            rectangle=rectangle,
            confidence_config=confidence_config,
            matched_pair_config=matched_pair_config,
        )
        for raw in raw_rows
    )
    summaries = tuple(
        _profile_summary(
            rows,
            spec,
            full_protocol=full_protocol,
            assignment_accuracy_gate=assignment_accuracy_gate,
            mismatch_repair_gate=mismatch_repair_gate,
        )
        for spec in profile_specs
    )
    pair_count = sum(row.pair_count for row in rows)
    correct_count = sum(row.correct_assignment_count for row in rows)
    focus = [
        row
        for row in rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    return GlobalAssignmentPanel(
        panel_role=panel_role,
        seed=seed,
        frozen_rectangle=rectangle,
        cases=rows,
        profile_summaries=summaries,
        case_count=len(rows),
        pair_count=pair_count,
        correct_assignment_count=correct_count,
        assignment_accuracy=correct_count / pair_count,
        unguarded_harmful_outlier_false_safe_count=sum(
            row.unguarded_harmful_outlier_false_safe for row in rows
        ),
        guarded_harmful_outlier_false_safe_count=sum(
            row.guarded_harmful_outlier_false_safe for row in rows
        ),
        focus_unguarded_safe_accept_count=unguarded_focus,
        focus_guarded_safe_accept_count=guarded_focus,
        focus_safe_accept_retention=(
            0.0 if not unguarded_focus else guarded_focus / unguarded_focus
        ),
        full_protocol=full_protocol,
        panel_gate_passed=bool(
            full_protocol
            and len(summaries) == len(profile_specs)
            and all(summary.profile_gate_passed for summary in summaries)
        ),
    )


def evaluate_global_tangential_assignment(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    calibration_a_seed: int = CALIBRATION_A_SEED,
    calibration_b_seed: int = CALIBRATION_B_SEED,
    final_held_out_seed: int = FINAL_HELD_OUT_SEED,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
    matched_pair_config: MatchedPairConfig | None = None,
    stress_config: MatchedPairStressConfig | None = None,
    confidence_config: PairConfidenceConfig | None = None,
    profile_specs: Sequence[MatchedPairStressSpec] = DEFAULT_STRESS_SPECS,
    assignment_accuracy_gate: float = 0.99,
    mismatch_repair_gate: float = 0.90,
) -> GlobalTangentialAssignmentResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_matched = (
        MatchedPairConfig() if matched_pair_config is None else matched_pair_config
    )
    selected_stress = (
        MatchedPairStressConfig() if stress_config is None else stress_config
    )
    selected_confidence = (
        PairConfidenceConfig() if confidence_config is None else confidence_config
    )
    selected_profiles = tuple(profile_specs)
    seeds = (calibration_a_seed, calibration_b_seed, final_held_out_seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("calibration and final seeds must differ")
    if PHASE19_FINAL_HELD_OUT_SEED in seeds:
        raise ValueError("Phase-19 unopened final seed must not be reused")
    if not 0.0 < assignment_accuracy_gate <= 1.0:
        raise ValueError("assignment_accuracy_gate must lie in (0, 1]")
    if not 0.0 < mismatch_repair_gate <= 1.0:
        raise ValueError("mismatch_repair_gate must lie in (0, 1]")
    if reference_count < 1 or repeats < 1 or surface_sample_count < 1:
        raise ValueError("panel sizes must be positive")
    if not selected_counts or min(selected_counts) < 4:
        raise ValueError("point_counts must contain values of at least four")
    if not selected_stresses or not selected_profiles:
        raise ValueError("stresses and profile_specs must not be empty")
    if len({spec.profile for spec in selected_profiles}) != len(selected_profiles):
        raise ValueError("stress profiles must be unique")
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and selected_profiles == DEFAULT_STRESS_SPECS
        and repeats >= 8
        and reference_count >= 2048
        and surface_sample_count >= 256
        and seeds
        == (CALIBRATION_A_SEED, CALIBRATION_B_SEED, FINAL_HELD_OUT_SEED)
        and assignment_accuracy_gate == 0.99
        and mismatch_repair_gate == 0.90
    )
    common = {
        "point_counts": selected_counts,
        "stresses": selected_stresses,
        "reference_count": reference_count,
        "repeats": repeats,
        "surface_sample_count": surface_sample_count,
        "base_gate_config": base_gate_config,
        "shared_trend_config": shared_trend_config,
        "matched_pair_config": selected_matched,
        "stress_config": selected_stress,
        "confidence_config": selected_confidence,
        "profile_specs": selected_profiles,
    }
    raw_a = _raw_panel(seed=calibration_a_seed, **common)
    raw_b = _raw_panel(seed=calibration_b_seed, **common)
    panel_common = {
        "rectangle": FROZEN_PHASE18_RECTANGLE,
        "confidence_config": selected_confidence,
        "matched_pair_config": selected_matched,
        "profile_specs": selected_profiles,
        "full_protocol": full_protocol,
        "assignment_accuracy_gate": assignment_accuracy_gate,
        "mismatch_repair_gate": mismatch_repair_gate,
    }
    panel_a = _materialize_panel(
        raw_a,
        panel_role="calibration_a",
        seed=calibration_a_seed,
        **panel_common,
    )
    panel_b = _materialize_panel(
        raw_b,
        panel_role="calibration_b",
        seed=calibration_b_seed,
        **panel_common,
    )
    final_panel: GlobalAssignmentPanel | None = None
    if panel_a.panel_gate_passed and panel_b.panel_gate_passed:
        raw_final = _raw_panel(seed=final_held_out_seed, **common)
        final_panel = _materialize_panel(
            raw_final,
            panel_role="final_held_out",
            seed=final_held_out_seed,
            **panel_common,
        )
    supported = bool(final_panel is not None and final_panel.panel_gate_passed)
    return GlobalTangentialAssignmentResult(
        artifact_schema="pftf_alpha_global_tangential_assignment_phase20/v1",
        role="synthetic_global_one_to_one_tangential_assignment_audit",
        information_boundary=(
            "route uses presented coordinate sets and row pairs for robust "
            "alignment only; source IDs and endpoints are evaluation-only"
        ),
        frozen_predecessor=(
            "phase19_scalar_tangential_confidence_negative_final_unopened"
        ),
        calibration_a_seed=calibration_a_seed,
        calibration_b_seed=calibration_b_seed,
        final_held_out_seed=final_held_out_seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        matched_pair_config=selected_matched,
        stress_config=selected_stress,
        confidence_config=selected_confidence,
        profile_specs=selected_profiles,
        frozen_rectangle=FROZEN_PHASE18_RECTANGLE,
        assignment_accuracy_gate=assignment_accuracy_gate,
        mismatch_repair_gate=mismatch_repair_gate,
        calibration_a=panel_a,
        calibration_b=panel_b,
        final_held_out=final_panel,
        phase20_supported=supported,
        global_tangential_assignment_synthetic_supported=supported,
        real_correspondence_supported=False,
        real_paired_scan_supported=False,
        trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--calibration-a-seed", type=int, default=CALIBRATION_A_SEED)
    parser.add_argument("--calibration-b-seed", type=int, default=CALIBRATION_B_SEED)
    parser.add_argument("--final-seed", type=int, default=FINAL_HELD_OUT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_global_tangential_assignment(
        reference_count=args.reference,
        repeats=args.repeats,
        calibration_a_seed=args.calibration_a_seed,
        calibration_b_seed=args.calibration_b_seed,
        final_held_out_seed=args.final_seed,
        surface_sample_count=args.surface_samples,
    )
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
