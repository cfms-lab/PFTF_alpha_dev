"""Cycle-gated preserve-versus-reassign correspondence audit for Phase 21."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .global_tangential_assignment import (
    GlobalTangentialAssignment,
    _assignment_from_confidence,
)
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
    TangentialPairRawCase,
    _raw_panel,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEVELOPMENT_A_SEED = 23300804
DEVELOPMENT_B_SEED = 23400804
VALIDATION_A_SEED = 23600804
VALIDATION_B_SEED = 23700804
FINAL_HELD_OUT_SEED = 23800804
PHASE20_FINAL_HELD_OUT_SEED = 23500804


@dataclass(frozen=True)
class AssignmentCycle:
    rows: tuple[int, ...]
    relative_gain: float
    truth_correct_before: int
    truth_correct_after: int

    @property
    def truth_improving(self) -> bool:
        return self.truth_correct_after > self.truth_correct_before


@dataclass(frozen=True)
class CycleCandidateCase:
    raw: TangentialPairRawCase
    global_assignment: GlobalTangentialAssignment
    cycles: tuple[AssignmentCycle, ...]


@dataclass(frozen=True)
class CycleCutoffCalibration:
    cutoff: float
    maximum_non_improving_gain: float
    cycle_count: int
    truth_improving_cycle_count: int
    non_improving_cycle_count: int
    accepted_truth_improving_cycle_count: int
    rejected_truth_improving_cycle_count: int
    accepted_non_improving_cycle_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CycleGatedCaseResult:
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
    candidate_cycle_count: int
    accepted_cycle_count: int
    rejected_cycle_count: int
    candidate_changed_assignment_count: int
    applied_changed_assignment_count: int
    maximum_candidate_cycle_gain: float
    minimum_accepted_cycle_gain: float | None
    presented_mismatch_pair_count: int
    correct_assignment_count: int
    assignment_accuracy: float
    repaired_presented_mismatch_count: int
    presented_mismatch_repair_fraction: float
    introduced_mismatch_count: int
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
class CycleGatedProfileSummary:
    profile: MatchedPairStressProfile
    case_count: int
    pair_count: int
    correct_assignment_count: int
    assignment_accuracy: float
    presented_mismatch_pair_count: int
    repaired_presented_mismatch_count: int
    presented_mismatch_repair_fraction: float
    introduced_mismatch_count: int
    candidate_cycle_count: int
    accepted_cycle_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
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
class CycleGatedPanel:
    panel_role: str
    seed: int
    cycle_gain_cutoff: float
    frozen_rectangle: InfluenceRectangle
    cases: tuple[CycleGatedCaseResult, ...]
    profile_summaries: tuple[CycleGatedProfileSummary, ...]
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
            "cycle_gain_cutoff": self.cycle_gain_cutoff,
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
class CycleGatedAssignmentResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    development_a_seed: int
    development_b_seed: int
    validation_a_seed: int
    validation_b_seed: int
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
    cycle_cutoff_calibration: CycleCutoffCalibration
    development_a: CycleGatedPanel
    development_b: CycleGatedPanel
    development_screen_passed: bool
    validation_a: CycleGatedPanel | None
    validation_b: CycleGatedPanel | None
    final_held_out: CycleGatedPanel | None
    phase21_supported: bool
    cycle_gated_assignment_synthetic_supported: bool
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
            "development_a_seed": self.development_a_seed,
            "development_b_seed": self.development_b_seed,
            "validation_a_seed": self.validation_a_seed,
            "validation_b_seed": self.validation_b_seed,
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
            "cycle_cutoff_calibration": self.cycle_cutoff_calibration.to_dict(),
            "development_a": self.development_a.to_dict(),
            "development_b": self.development_b.to_dict(),
            "development_screen_passed": self.development_screen_passed,
            "validation_a": (
                None if self.validation_a is None else self.validation_a.to_dict()
            ),
            "validation_b": (
                None if self.validation_b is None else self.validation_b.to_dict()
            ),
            "final_held_out": (
                None
                if self.final_held_out is None
                else self.final_held_out.to_dict()
            ),
            "phase21_supported": self.phase21_supported,
            "cycle_gated_assignment_synthetic_supported": (
                self.cycle_gated_assignment_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def assignment_cycles(
    assignment: GlobalTangentialAssignment,
    primary_ids: IntArray,
    repeat_source_ids: IntArray,
) -> tuple[AssignmentCycle, ...]:
    permutation = np.asarray(assignment.repeat_row_for_primary, dtype=np.int64)
    primary = np.asarray(primary_ids, dtype=np.int64)
    repeat_sources = np.asarray(repeat_source_ids, dtype=np.int64)
    count = permutation.size
    if primary.shape != (count,) or repeat_sources.shape != (count,):
        raise ValueError("pair ID arrays must align with the assignment")
    if not np.array_equal(np.sort(permutation), np.arange(count)):
        raise ValueError("assignment must be a permutation")
    visited = np.zeros(count, dtype=bool)
    cycles: list[AssignmentCycle] = []
    for start in range(count):
        if visited[start]:
            continue
        rows: list[int] = []
        current = start
        while not visited[current]:
            visited[current] = True
            rows.append(current)
            current = int(permutation[current])
        if len(rows) <= 1:
            continue
        indices = np.asarray(rows, dtype=np.int64)
        identity_cost = float(np.sum(assignment.presented_costs[indices]))
        assigned_cost = float(np.sum(assignment.assigned_costs[indices]))
        denominator = max(identity_cost, np.finfo(float).eps)
        gain = max(0.0, (identity_cost - assigned_cost) / denominator)
        before = int(np.sum(primary[indices] == repeat_sources[indices]))
        after = int(
            np.sum(primary[indices] == repeat_sources[permutation[indices]])
        )
        cycles.append(
            AssignmentCycle(
                rows=tuple(rows),
                relative_gain=gain,
                truth_correct_before=before,
                truth_correct_after=after,
            )
        )
    return tuple(cycles)


def _candidate_cases(
    rows: Sequence[TangentialPairRawCase],
    confidence_config: PairConfidenceConfig,
) -> tuple[CycleCandidateCase, ...]:
    candidates: list[CycleCandidateCase] = []
    for raw in rows:
        assignment = _assignment_from_confidence(
            raw.primary_points,
            raw.confidence,
            confidence_config,
        )
        candidates.append(
            CycleCandidateCase(
                raw=raw,
                global_assignment=assignment,
                cycles=assignment_cycles(
                    assignment,
                    raw.primary_ids,
                    raw.repeat_source_ids,
                ),
            )
        )
    return tuple(candidates)


def calibrate_cycle_gain_cutoff(
    cases: Sequence[CycleCandidateCase],
) -> CycleCutoffCalibration:
    cycles = tuple(cycle for case in cases for cycle in case.cycles)
    non_improving = tuple(cycle for cycle in cycles if not cycle.truth_improving)
    maximum_non_improving = (
        0.0
        if not non_improving
        else max(cycle.relative_gain for cycle in non_improving)
    )
    cutoff = (
        0.0
        if not non_improving
        else float(np.nextafter(maximum_non_improving, math.inf))
    )
    accepted_improving = sum(
        cycle.truth_improving and cycle.relative_gain >= cutoff for cycle in cycles
    )
    accepted_non_improving = sum(
        (not cycle.truth_improving) and cycle.relative_gain >= cutoff
        for cycle in cycles
    )
    improving_count = sum(cycle.truth_improving for cycle in cycles)
    return CycleCutoffCalibration(
        cutoff=cutoff,
        maximum_non_improving_gain=maximum_non_improving,
        cycle_count=len(cycles),
        truth_improving_cycle_count=improving_count,
        non_improving_cycle_count=len(non_improving),
        accepted_truth_improving_cycle_count=accepted_improving,
        rejected_truth_improving_cycle_count=improving_count - accepted_improving,
        accepted_non_improving_cycle_count=accepted_non_improving,
    )


def _materialize_case(
    candidate: CycleCandidateCase,
    *,
    cutoff: float,
    rectangle: InfluenceRectangle,
    matched_pair_config: MatchedPairConfig,
) -> CycleGatedCaseResult:
    accepted = tuple(
        cycle for cycle in candidate.cycles if cycle.relative_gain >= cutoff
    )
    return _materialize_case_with_selection(
        candidate,
        accepted=accepted,
        rectangle=rectangle,
        matched_pair_config=matched_pair_config,
    )


def _materialize_case_with_selection(
    candidate: CycleCandidateCase,
    *,
    accepted: Sequence[AssignmentCycle],
    rectangle: InfluenceRectangle,
    matched_pair_config: MatchedPairConfig,
) -> CycleGatedCaseResult:
    raw = candidate.raw
    pair_count = raw.primary_points.shape[0]
    selected_assignment = np.arange(pair_count, dtype=np.int64)
    accepted_cycles = tuple(accepted)
    for cycle in accepted_cycles:
        rows = np.asarray(cycle.rows, dtype=np.int64)
        selected_assignment[rows] = (
            candidate.global_assignment.repeat_row_for_primary[rows]
        )
    assigned_source_ids = raw.repeat_source_ids[selected_assignment]
    correct_assignment = raw.primary_ids == assigned_source_ids
    presented_mismatch = raw.primary_ids != raw.repeat_source_ids
    repaired_presented = presented_mismatch & correct_assignment
    introduced = (~presented_mismatch) & (~correct_assignment)
    evidence = replace(
        estimate_matched_pair_evidence(
            raw.primary_points,
            raw.repeat_points[selected_assignment],
            matched_pair_config,
        ),
        information_boundary=(
            "presented_pairing_preserved_except_cycles_above_frozen_observed_"
            "gain_cutoff; pair_truth_unknown_to_route"
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
    correct_count = int(np.sum(correct_assignment))
    presented_mismatch_count = int(np.sum(presented_mismatch))
    repaired_count = int(np.sum(repaired_presented))
    return CycleGatedCaseResult(
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
        candidate_cycle_count=len(candidate.cycles),
        accepted_cycle_count=len(accepted_cycles),
        rejected_cycle_count=len(candidate.cycles) - len(accepted_cycles),
        candidate_changed_assignment_count=int(
            np.sum(
                candidate.global_assignment.repeat_row_for_primary
                != np.arange(pair_count)
            )
        ),
        applied_changed_assignment_count=int(
            np.sum(selected_assignment != np.arange(pair_count))
        ),
        maximum_candidate_cycle_gain=(
            0.0
            if not candidate.cycles
            else max(cycle.relative_gain for cycle in candidate.cycles)
        ),
        minimum_accepted_cycle_gain=(
            None
            if not accepted_cycles
            else min(cycle.relative_gain for cycle in accepted_cycles)
        ),
        presented_mismatch_pair_count=presented_mismatch_count,
        correct_assignment_count=correct_count,
        assignment_accuracy=correct_count / pair_count,
        repaired_presented_mismatch_count=repaired_count,
        presented_mismatch_repair_fraction=(
            1.0
            if not presented_mismatch_count
            else repaired_count / presented_mismatch_count
        ),
        introduced_mismatch_count=int(np.sum(introduced)),
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
    rows: Sequence[CycleGatedCaseResult],
    spec: MatchedPairStressSpec,
    *,
    full_protocol: bool,
    assignment_accuracy_gate: float,
    mismatch_repair_gate: float,
) -> CycleGatedProfileSummary:
    selected = [row for row in rows if row.profile is spec.profile]
    pair_count = sum(row.pair_count for row in selected)
    correct_count = sum(row.correct_assignment_count for row in selected)
    accuracy = correct_count / pair_count
    mismatch_count = sum(row.presented_mismatch_pair_count for row in selected)
    repaired_count = sum(
        row.repaired_presented_mismatch_count for row in selected
    )
    repair_fraction = 1.0 if not mismatch_count else repaired_count / mismatch_count
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
            or (mismatch_count > 0 and repair_fraction >= mismatch_repair_gate)
        )
    )
    return CycleGatedProfileSummary(
        profile=spec.profile,
        case_count=len(selected),
        pair_count=pair_count,
        correct_assignment_count=correct_count,
        assignment_accuracy=accuracy,
        presented_mismatch_pair_count=mismatch_count,
        repaired_presented_mismatch_count=repaired_count,
        presented_mismatch_repair_fraction=repair_fraction,
        introduced_mismatch_count=sum(
            row.introduced_mismatch_count for row in selected
        ),
        candidate_cycle_count=sum(row.candidate_cycle_count for row in selected),
        accepted_cycle_count=sum(row.accepted_cycle_count for row in selected),
        unguarded_harmful_outlier_false_safe_count=unguarded_harm,
        guarded_harmful_outlier_false_safe_count=guarded_harm,
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
    candidates: tuple[CycleCandidateCase, ...],
    *,
    panel_role: str,
    seed: int,
    cutoff: float,
    rectangle: InfluenceRectangle,
    matched_pair_config: MatchedPairConfig,
    profile_specs: tuple[MatchedPairStressSpec, ...],
    full_protocol: bool,
    assignment_accuracy_gate: float,
    mismatch_repair_gate: float,
) -> CycleGatedPanel:
    rows = tuple(
        _materialize_case(
            candidate,
            cutoff=cutoff,
            rectangle=rectangle,
            matched_pair_config=matched_pair_config,
        )
        for candidate in candidates
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
    return CycleGatedPanel(
        panel_role=panel_role,
        seed=seed,
        cycle_gain_cutoff=cutoff,
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


def evaluate_cycle_gated_assignment(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    development_a_seed: int = DEVELOPMENT_A_SEED,
    development_b_seed: int = DEVELOPMENT_B_SEED,
    validation_a_seed: int = VALIDATION_A_SEED,
    validation_b_seed: int = VALIDATION_B_SEED,
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
) -> CycleGatedAssignmentResult:
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
    seeds = (
        development_a_seed,
        development_b_seed,
        validation_a_seed,
        validation_b_seed,
        final_held_out_seed,
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError("development, validation, and final seeds must differ")
    if PHASE20_FINAL_HELD_OUT_SEED in seeds:
        raise ValueError("Phase-20 unopened final seed must not be reused")
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
        == (
            DEVELOPMENT_A_SEED,
            DEVELOPMENT_B_SEED,
            VALIDATION_A_SEED,
            VALIDATION_B_SEED,
            FINAL_HELD_OUT_SEED,
        )
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
    raw_development_a = _raw_panel(seed=development_a_seed, **common)
    raw_development_b = _raw_panel(seed=development_b_seed, **common)
    development_a_candidates = _candidate_cases(
        raw_development_a,
        selected_confidence,
    )
    development_b_candidates = _candidate_cases(
        raw_development_b,
        selected_confidence,
    )
    cutoff_calibration = calibrate_cycle_gain_cutoff(
        development_a_candidates + development_b_candidates
    )
    panel_common = {
        "cutoff": cutoff_calibration.cutoff,
        "rectangle": FROZEN_PHASE18_RECTANGLE,
        "matched_pair_config": selected_matched,
        "profile_specs": selected_profiles,
        "full_protocol": full_protocol,
        "assignment_accuracy_gate": assignment_accuracy_gate,
        "mismatch_repair_gate": mismatch_repair_gate,
    }
    development_a = _materialize_panel(
        development_a_candidates,
        panel_role="development_a",
        seed=development_a_seed,
        **panel_common,
    )
    development_b = _materialize_panel(
        development_b_candidates,
        panel_role="development_b",
        seed=development_b_seed,
        **panel_common,
    )
    development_passed = bool(
        development_a.panel_gate_passed and development_b.panel_gate_passed
    )
    validation_a: CycleGatedPanel | None = None
    validation_b: CycleGatedPanel | None = None
    final_panel: CycleGatedPanel | None = None
    if development_passed:
        validation_a = _materialize_panel(
            _candidate_cases(
                _raw_panel(seed=validation_a_seed, **common),
                selected_confidence,
            ),
            panel_role="validation_a",
            seed=validation_a_seed,
            **panel_common,
        )
        validation_b = _materialize_panel(
            _candidate_cases(
                _raw_panel(seed=validation_b_seed, **common),
                selected_confidence,
            ),
            panel_role="validation_b",
            seed=validation_b_seed,
            **panel_common,
        )
        if validation_a.panel_gate_passed and validation_b.panel_gate_passed:
            final_panel = _materialize_panel(
                _candidate_cases(
                    _raw_panel(seed=final_held_out_seed, **common),
                    selected_confidence,
                ),
                panel_role="final_held_out",
                seed=final_held_out_seed,
                **panel_common,
            )
    supported = bool(final_panel is not None and final_panel.panel_gate_passed)
    return CycleGatedAssignmentResult(
        artifact_schema="pftf_alpha_cycle_gated_assignment_phase21/v1",
        role="synthetic_supervised_cycle_gated_assignment_audit",
        information_boundary=(
            "route uses presented coordinates, row alignment, and a prior "
            "truth-supervised cycle-gain cutoff; IDs/endpoints are evaluation-only"
        ),
        frozen_predecessor=(
            "phase20_global_assignment_zero_harm_but_retention_negative"
        ),
        development_a_seed=development_a_seed,
        development_b_seed=development_b_seed,
        validation_a_seed=validation_a_seed,
        validation_b_seed=validation_b_seed,
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
        cycle_cutoff_calibration=cutoff_calibration,
        development_a=development_a,
        development_b=development_b,
        development_screen_passed=development_passed,
        validation_a=validation_a,
        validation_b=validation_b,
        final_held_out=final_panel,
        phase21_supported=supported,
        cycle_gated_assignment_synthetic_supported=supported,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_cycle_gated_assignment(
        reference_count=args.reference,
        repeats=args.repeats,
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
