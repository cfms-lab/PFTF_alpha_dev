"""Tail-sensitive observed local guard transfer audit for Phase 30."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .frozen_partition_reconstruction import _materialize_panel, _panel_case_seeds
from .local_spatial_displacement import LocalSpatialDisplacementEvidence
from .local_spatial_residual_guard import (
    LocalSpatialGuardCase,
    LocalSpatialPanel,
    LocalSpatialResidualGuardResult,
    _materialize_local_panel,
    _profile_summary,
    evaluate_local_spatial_residual_guard,
)
from .matched_guard_signature import GUARD_PROFILE_SPECS
from .matched_pair_stress import _raw_panel
from .sampling_gate import SamplingGateDecision
from .sensor_stress import DEFAULT_POINT_COUNTS, DEFAULT_STRESSES, SensorStress
from .targeted_local_residual_challenge import (
    FINAL_HELD_OUT_SEED as PHASE29_FINAL_SEED,
)
from .targeted_local_residual_challenge import (
    TARGET_POINT_COUNTS,
    TARGET_REPEATS,
    TARGET_STRESSES,
    IncrementalEvidenceSummary,
    summarize_incremental_evidence,
)
from .targeted_local_residual_challenge import (
    VALIDATION_A_SEED as TAIL_DESIGN_SEED,
)
from .targeted_local_residual_challenge import (
    VALIDATION_B_SEED as PHASE29_VALIDATION_B_SEED,
)

VALIDATION_A_SEED = 30500804
VALIDATION_B_SEED = 30600804
FINAL_HELD_OUT_SEED = 30700804
FULL_PRIOR_BASE_SEEDS = tuple(
    index * 100000 + 804 for index in range(203, 260)
) + tuple(index * 100000 + 804 for index in range(275, 284))
TARGET_PRIOR_BASE_SEEDS = (
    TAIL_DESIGN_SEED,
    PHASE29_VALIDATION_B_SEED,
    PHASE29_FINAL_SEED,
)
TAIL_FEATURE_NAMES = (
    "maximum_local_residual",
    "support_local_residual",
    "maximum_local_score_excess",
    "isolated_tail_gap",
    "isolated_tail_ratio",
    "support_tail_gap",
)
EXPECTED_SELECTED_FEATURE = "isolated_tail_ratio"
EXPECTED_RESIDUAL_HARMFUL_COUNT = 3
EXPECTED_LIMITING_TAIL_VALUE = 1.6636368999089544
EXPECTED_TAIL_REJECTION_CUTOFF = 1.6636368999089541
EXPECTED_ORIGINAL_FOCUS_COUNT = 1281
EXPECTED_PREDECESSOR_FOCUS_COUNT = 1268
EXPECTED_TAIL_RETAINED_FOCUS_COUNT = 1248
EXPECTED_ORIGINAL_SAFE_COUNT = 3144
EXPECTED_PREDECESSOR_SAFE_COUNT = 3092
EXPECTED_TAIL_RETAINED_SAFE_COUNT = 3027
REPRODUCTION_TOLERANCE = 1.0e-15


@dataclass(frozen=True)
class Phase30CaseSeedAudit:
    case_seed_formula: str
    full_prior_base_seed_ranges: tuple[str, ...]
    targeted_prior_base_seeds: tuple[int, ...]
    targeted_point_counts: tuple[int, ...]
    targeted_stresses: tuple[SensorStress, ...]
    targeted_repeats: int
    panel_case_count: int
    validation_a_prior_overlap_count: int
    validation_b_prior_overlap_count: int
    final_prior_overlap_count: int
    validation_a_b_overlap_count: int
    validation_a_final_overlap_count: int
    validation_b_final_overlap_count: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["targeted_stresses"] = [
            stress.value for stress in self.targeted_stresses
        ]
        return payload


@dataclass(frozen=True)
class TailFeatureCandidate:
    feature_name: str
    rejection_direction: str
    residual_harmful_case_count: int
    limiting_tail_value: float | None
    rejection_cutoff: float
    binary64_decrement: float | None
    original_focus_safe_accept_count: int
    predecessor_focus_safe_accept_count: int
    combined_focus_safe_accept_count: int
    original_safe_accept_count: int
    predecessor_safe_accept_count: int
    combined_safe_accept_count: int
    design_panel_count: int
    passing_design_panel_count: int
    all_design_gates_passed: bool
    eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TailFeatureCalibration:
    selection_rule: str
    design_panel_roles: tuple[str, ...]
    candidates: tuple[TailFeatureCandidate, ...]
    selected_feature_name: str | None
    selected_rejection_cutoff: float | None
    calibration_valid: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_rule": self.selection_rule,
            "design_panel_roles": list(self.design_panel_roles),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_feature_name": self.selected_feature_name,
            "selected_rejection_cutoff": self.selected_rejection_cutoff,
            "calibration_valid": self.calibration_valid,
        }


@dataclass(frozen=True)
class TailSensitiveLocalGuardResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    target_point_counts: tuple[int, ...]
    target_stresses: tuple[SensorStress, ...]
    target_repeats: int
    tail_design_seed: int
    validation_a_seed: int
    validation_b_seed: int
    final_held_out_seed: int
    reference_count: int
    surface_sample_count: int
    case_seed_disjointness: Phase30CaseSeedAudit
    phase28_reproduced_and_supported: bool
    phase29_failure_reproduced: bool
    calibration: TailFeatureCalibration
    design_panels: tuple[LocalSpatialPanel, ...]
    design_reproduced: bool
    design_gate_passed: bool
    fresh_execution_requested: bool
    validation_a: LocalSpatialPanel | None
    validation_b: LocalSpatialPanel | None
    final_held_out: LocalSpatialPanel | None
    incremental_evidence: IncrementalEvidenceSummary
    phase30_supported: bool
    tail_sensitive_local_guard_synthetic_supported: bool
    real_correspondence_supported: bool
    real_paired_scan_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_predecessor": self.frozen_predecessor,
            "target_point_counts": list(self.target_point_counts),
            "target_stresses": [stress.value for stress in self.target_stresses],
            "target_repeats": self.target_repeats,
            "tail_design_seed": self.tail_design_seed,
            "validation_a_seed": self.validation_a_seed,
            "validation_b_seed": self.validation_b_seed,
            "final_held_out_seed": self.final_held_out_seed,
            "reference_count": self.reference_count,
            "surface_sample_count": self.surface_sample_count,
            "case_seed_disjointness": self.case_seed_disjointness.to_dict(),
            "phase28_reproduced_and_supported": (
                self.phase28_reproduced_and_supported
            ),
            "phase29_failure_reproduced": self.phase29_failure_reproduced,
            "calibration": self.calibration.to_dict(),
            "design_panels": [
                _panel_summary_dict(panel) for panel in self.design_panels
            ],
            "design_reproduced": self.design_reproduced,
            "design_gate_passed": self.design_gate_passed,
            "fresh_execution_requested": self.fresh_execution_requested,
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
            "incremental_evidence": self.incremental_evidence.to_dict(),
            "phase30_supported": self.phase30_supported,
            "tail_sensitive_local_guard_synthetic_supported": (
                self.tail_sensitive_local_guard_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "real_trimmed_reconstruction_supported": (
                self.real_trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def _panel_summary_dict(panel: LocalSpatialPanel) -> dict[str, object]:
    payload = panel.to_dict()
    payload.pop("cases")
    return payload


def audit_phase30_case_seed_disjointness(
    validation_a_seed: int,
    validation_b_seed: int,
    final_held_out_seed: int,
) -> Phase30CaseSeedAudit:
    prior = frozenset().union(
        *(
            _panel_case_seeds(
                seed,
                point_counts=DEFAULT_POINT_COUNTS,
                stresses=DEFAULT_STRESSES,
                repeats=8,
            )
            for seed in FULL_PRIOR_BASE_SEEDS
        ),
        *(
            _panel_case_seeds(
                seed,
                point_counts=TARGET_POINT_COUNTS,
                stresses=TARGET_STRESSES,
                repeats=TARGET_REPEATS,
            )
            for seed in TARGET_PRIOR_BASE_SEEDS
        ),
    )
    panels = tuple(
        _panel_case_seeds(
            seed,
            point_counts=TARGET_POINT_COUNTS,
            stresses=TARGET_STRESSES,
            repeats=TARGET_REPEATS,
        )
        for seed in (validation_a_seed, validation_b_seed, final_held_out_seed)
    )
    prior_overlaps = tuple(len(panel & prior) for panel in panels)
    mutual_overlaps = (
        len(panels[0] & panels[1]),
        len(panels[0] & panels[2]),
        len(panels[1] & panels[2]),
    )
    return Phase30CaseSeedAudit(
        case_seed_formula=(
            "base + count_index*1000003 + stress_index*100003 + repeat*10007"
        ),
        full_prior_base_seed_ranges=(
            "20300804--25900804",
            "27500804--28300804",
        ),
        targeted_prior_base_seeds=TARGET_PRIOR_BASE_SEEDS,
        targeted_point_counts=TARGET_POINT_COUNTS,
        targeted_stresses=TARGET_STRESSES,
        targeted_repeats=TARGET_REPEATS,
        panel_case_count=len(panels[0]),
        validation_a_prior_overlap_count=prior_overlaps[0],
        validation_b_prior_overlap_count=prior_overlaps[1],
        final_prior_overlap_count=prior_overlaps[2],
        validation_a_b_overlap_count=mutual_overlaps[0],
        validation_a_final_overlap_count=mutual_overlaps[1],
        validation_b_final_overlap_count=mutual_overlaps[2],
        passed=not any(prior_overlaps + mutual_overlaps),
    )


def tail_feature_value(
    evidence: LocalSpatialDisplacementEvidence,
    feature_name: str,
) -> float:
    functions: dict[
        str, Callable[[LocalSpatialDisplacementEvidence], float]
    ] = {
        "maximum_local_residual": lambda item: item.maximum_local_residual,
        "support_local_residual": lambda item: item.support_local_residual,
        "maximum_local_score_excess": (
            lambda item: item.maximum_local_score_excess
        ),
        "isolated_tail_gap": (
            lambda item: item.maximum_local_residual
            - item.percentile95_local_residual
        ),
        "isolated_tail_ratio": (
            lambda item: item.maximum_local_residual
            / max(item.percentile95_local_residual, math.ulp(0.0))
        ),
        "support_tail_gap": (
            lambda item: item.support_local_residual
            - item.percentile95_local_residual
        ),
    }
    try:
        value = float(functions[feature_name](evidence))
    except KeyError as error:
        raise ValueError(f"unknown tail feature: {feature_name}") from error
    if not math.isfinite(value):
        raise ValueError("tail feature must be finite")
    return value


def _materialize_tail_case(
    case: LocalSpatialGuardCase,
    *,
    feature_name: str,
    rejection_cutoff: float,
) -> LocalSpatialGuardCase:
    predecessor_accept = case.guarded_decision is SamplingGateDecision.ACCEPT
    tail_accept = bool(
        tail_feature_value(case.local_spatial_evidence, feature_name)
        < rejection_cutoff
    )
    guarded_accept = bool(predecessor_accept and tail_accept)
    guarded_decision = (
        SamplingGateDecision.ACCEPT
        if guarded_accept
        else (
            SamplingGateDecision.UNSUPPORTED
            if case.unguarded_decision is SamplingGateDecision.ACCEPT
            else case.unguarded_decision
        )
    )
    return LocalSpatialGuardCase(
        profile=case.profile,
        stress=case.stress,
        point_count=case.point_count,
        repeat=case.repeat,
        seed=case.seed,
        replicate_seed=case.replicate_seed,
        perturbation_seed=case.perturbation_seed,
        model_score=case.model_score,
        local_spatial_evidence=case.local_spatial_evidence,
        original_endpoint=case.original_endpoint,
        routed_endpoint=case.routed_endpoint,
        unguarded_decision=case.unguarded_decision,
        predecessor_guarded_decision=case.guarded_decision,
        guarded_decision=guarded_decision,
        unguarded_safe_accept=case.unguarded_safe_accept,
        predecessor_guarded_safe_accept=case.guarded_safe_accept,
        guarded_safe_accept=bool(
            guarded_accept
            and not case.routed_endpoint.geometry_topology_harm_present
        ),
        unguarded_harmful_outlier_false_safe=(
            case.unguarded_harmful_outlier_false_safe
        ),
        predecessor_guarded_harmful_outlier_false_safe=(
            case.guarded_harmful_outlier_false_safe
        ),
        guarded_harmful_outlier_false_safe=bool(
            guarded_accept
            and case.stress.is_outlier_stress
            and case.routed_endpoint.geometry_topology_harm_present
        ),
        introduced_routed_endpoint_harm_accept=bool(
            guarded_accept
            and not case.original_endpoint.geometry_topology_harm_present
            and case.routed_endpoint.geometry_topology_harm_present
        ),
    )


def _materialize_tail_panel(
    panel: LocalSpatialPanel,
    *,
    feature_name: str,
    rejection_cutoff: float,
    panel_role: str,
    full_protocol: bool,
) -> LocalSpatialPanel:
    rows = tuple(
        _materialize_tail_case(
            case,
            feature_name=feature_name,
            rejection_cutoff=rejection_cutoff,
        )
        for case in panel.cases
    )
    summaries = tuple(
        _profile_summary(
            rows,
            spec.profile,
            full_protocol=full_protocol,
            calibration_valid=True,
        )
        for spec in GUARD_PROFILE_SPECS
    )
    focus = tuple(
        row
        for row in rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    )
    original_focus = sum(row.unguarded_safe_accept for row in focus)
    all_safe = tuple(row for row in rows if row.unguarded_safe_accept)
    return LocalSpatialPanel(
        panel_role=panel_role,
        seed=panel.seed,
        cases=rows,
        profile_summaries=summaries,
        case_count=len(rows),
        unguarded_harmful_outlier_false_safe_count=sum(
            row.unguarded_harmful_outlier_false_safe for row in rows
        ),
        predecessor_guarded_harmful_outlier_false_safe_count=sum(
            row.predecessor_guarded_harmful_outlier_false_safe for row in rows
        ),
        guarded_harmful_outlier_false_safe_count=sum(
            row.guarded_harmful_outlier_false_safe for row in rows
        ),
        focus_unguarded_safe_accept_count=original_focus,
        focus_predecessor_guarded_safe_accept_count=sum(
            row.predecessor_guarded_safe_accept for row in focus
        ),
        focus_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in focus
        ),
        focus_safe_accept_retention=(
            0.0
            if not original_focus
            else sum(row.guarded_safe_accept for row in focus) / original_focus
        ),
        all_stress_unguarded_safe_accept_count=len(all_safe),
        all_stress_predecessor_guarded_safe_accept_count=sum(
            row.predecessor_guarded_safe_accept for row in all_safe
        ),
        all_stress_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in all_safe
        ),
        introduced_routed_endpoint_harm_accept_count=sum(
            row.introduced_routed_endpoint_harm_accept for row in rows
        ),
        full_protocol=full_protocol,
        panel_gate_passed=bool(
            full_protocol
            and len(summaries) == len(GUARD_PROFILE_SPECS)
            and all(summary.profile_gate_passed for summary in summaries)
        ),
    )


def _candidate_calibration(
    panels: Sequence[LocalSpatialPanel],
    feature_name: str,
) -> TailFeatureCandidate:
    selected = tuple(panels)
    rows = tuple(case for panel in selected for case in panel.cases)
    residual = tuple(
        case for case in rows if case.guarded_harmful_outlier_false_safe
    )
    values = tuple(
        tail_feature_value(case.local_spatial_evidence, feature_name)
        for case in residual
    )
    limiting = min(values) if values else None
    cutoff = (
        math.nextafter(limiting, -math.inf)
        if limiting is not None
        else -math.inf
    )
    materialized = tuple(
        _materialize_tail_panel(
            panel,
            feature_name=feature_name,
            rejection_cutoff=cutoff,
            panel_role=panel.panel_role,
            full_protocol=True,
        )
        for panel in selected
    )
    original_focus = sum(
        panel.focus_unguarded_safe_accept_count for panel in materialized
    )
    predecessor_focus = sum(
        panel.focus_predecessor_guarded_safe_accept_count
        for panel in materialized
    )
    combined_focus = sum(
        panel.focus_guarded_safe_accept_count for panel in materialized
    )
    original_safe = sum(
        panel.all_stress_unguarded_safe_accept_count for panel in materialized
    )
    predecessor_safe = sum(
        panel.all_stress_predecessor_guarded_safe_accept_count
        for panel in materialized
    )
    combined_safe = sum(
        panel.all_stress_guarded_safe_accept_count for panel in materialized
    )
    passing = sum(panel.panel_gate_passed for panel in materialized)
    all_passed = bool(materialized and passing == len(materialized))
    combined_harm = sum(
        panel.guarded_harmful_outlier_false_safe_count
        for panel in materialized
    )
    eligible = bool(residual and combined_harm == 0 and all_passed)
    return TailFeatureCandidate(
        feature_name=feature_name,
        rejection_direction="reject_if_greater_than_or_equal_to_cutoff",
        residual_harmful_case_count=len(residual),
        limiting_tail_value=limiting,
        rejection_cutoff=cutoff,
        binary64_decrement=(None if limiting is None else limiting - cutoff),
        original_focus_safe_accept_count=original_focus,
        predecessor_focus_safe_accept_count=predecessor_focus,
        combined_focus_safe_accept_count=combined_focus,
        original_safe_accept_count=original_safe,
        predecessor_safe_accept_count=predecessor_safe,
        combined_safe_accept_count=combined_safe,
        design_panel_count=len(materialized),
        passing_design_panel_count=passing,
        all_design_gates_passed=all_passed,
        eligible=eligible,
    )


def select_tail_candidate(
    candidates: Sequence[TailFeatureCandidate],
) -> TailFeatureCandidate | None:
    order = {name: index for index, name in enumerate(TAIL_FEATURE_NAMES)}
    eligible = tuple(candidate for candidate in candidates if candidate.eligible)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (
            candidate.combined_focus_safe_accept_count,
            candidate.combined_safe_accept_count,
            -order[candidate.feature_name],
        ),
    )


def calibrate_tail_features(
    panels: Sequence[LocalSpatialPanel],
) -> TailFeatureCalibration:
    selected = tuple(panels)
    candidates = tuple(
        _candidate_calibration(selected, feature_name)
        for feature_name in TAIL_FEATURE_NAMES
    )
    winner = select_tail_candidate(candidates)
    return TailFeatureCalibration(
        selection_rule=(
            "among candidates that reject every predecessor residual and pass "
            "every panel/profile safety gate, maximize combined focus accepts, "
            "then all-safe accepts, then use declared candidate order"
        ),
        design_panel_roles=tuple(panel.panel_role for panel in selected),
        candidates=candidates,
        selected_feature_name=(
            None if winner is None else winner.feature_name
        ),
        selected_rejection_cutoff=(
            None if winner is None else winner.rejection_cutoff
        ),
        calibration_valid=winner is not None,
    )


def _target_q95_panel(
    *,
    seed: int,
    role: str,
    phase28: LocalSpatialResidualGuardResult,
    reference_count: int,
    surface_sample_count: int,
    full_protocol: bool,
) -> LocalSpatialPanel:
    predecessor_panel = _materialize_panel(
        _raw_panel(
            seed=seed,
            point_counts=TARGET_POINT_COUNTS,
            stresses=TARGET_STRESSES,
            reference_count=reference_count,
            repeats=TARGET_REPEATS,
            surface_sample_count=surface_sample_count,
            base_gate_config=None,
            shared_trend_config=None,
            matched_pair_config=phase28.matched_pair_config,
            stress_config=phase28.stress_config,
            profile_specs=GUARD_PROFILE_SPECS,
        ),
        panel_role=f"{role}_global_predecessor",
        seed=seed,
        model=phase28.predecessor_model,
        profile_specs=GUARD_PROFILE_SPECS,
        full_protocol=False,
    )
    return _materialize_local_panel(
        predecessor_panel,
        phase28.local_calibration,
        panel_role=f"{role}_q95_predecessor",
        full_protocol=full_protocol,
    )


def evaluate_tail_sensitive_local_guard(
    *,
    validation_a_seed: int = VALIDATION_A_SEED,
    validation_b_seed: int = VALIDATION_B_SEED,
    final_held_out_seed: int = FINAL_HELD_OUT_SEED,
    reference_count: int = 2048,
    surface_sample_count: int = 256,
    open_fresh: bool = True,
) -> TailSensitiveLocalGuardResult:
    seeds = (validation_a_seed, validation_b_seed, final_held_out_seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("validation and final base seeds must differ")
    seed_audit = audit_phase30_case_seed_disjointness(*seeds)
    if not seed_audit.passed:
        raise ValueError(
            "fresh case seeds must be mutually disjoint and disjoint from all "
            "prior full and targeted panels"
        )
    full_protocol = bool(
        seeds == (VALIDATION_A_SEED, VALIDATION_B_SEED, FINAL_HELD_OUT_SEED)
        and reference_count >= 2048
        and surface_sample_count >= 256
    )
    phase28 = evaluate_local_spatial_residual_guard(
        reference_count=reference_count,
        surface_sample_count=surface_sample_count,
        open_fresh=True,
    )
    phase28_reproduced = bool(
        phase28.design_reproduced
        and phase28.design_gate_passed
        and phase28.phase28_supported
        and phase28.validation_a is not None
        and phase28.validation_b is not None
        and phase28.final_held_out is not None
    )
    phase29_design_q95 = _target_q95_panel(
        seed=TAIL_DESIGN_SEED,
        role="phase29_validation_a",
        phase28=phase28,
        reference_count=reference_count,
        surface_sample_count=surface_sample_count,
        full_protocol=full_protocol,
    )
    phase29_failure_reproduced = bool(
        phase29_design_q95.unguarded_harmful_outlier_false_safe_count == 57
        and phase29_design_q95.predecessor_guarded_harmful_outlier_false_safe_count
        == 3
        and phase29_design_q95.guarded_harmful_outlier_false_safe_count == 3
        and not phase29_design_q95.panel_gate_passed
    )
    phase28_design_panels = (
        phase28.design_score_fit,
        phase28.design_cutoff_calibration,
        phase28.local_design_a,
        phase28.local_design_b,
        phase28.local_calibration_panel,
        phase28.validation_a,
        phase28.validation_b,
        phase28.final_held_out,
    )
    if any(panel is None for panel in phase28_design_panels):
        raise RuntimeError("complete Phase-28 panels are required")
    q95_design_panels = tuple(
        panel for panel in phase28_design_panels if panel is not None
    ) + (phase29_design_q95,)
    calibration = calibrate_tail_features(q95_design_panels)
    selected = select_tail_candidate(calibration.candidates)
    selected_name = None if selected is None else selected.feature_name
    selected_cutoff = None if selected is None else selected.rejection_cutoff
    design_panels = (
        ()
        if selected is None
        else tuple(
            _materialize_tail_panel(
                panel,
                feature_name=selected.feature_name,
                rejection_cutoff=selected.rejection_cutoff,
                panel_role=panel.panel_role.replace("_q95_predecessor", ""),
                full_protocol=full_protocol,
            )
            for panel in q95_design_panels
        )
    )
    design_reproduced = bool(
        full_protocol
        and phase28_reproduced
        and phase29_failure_reproduced
        and calibration.calibration_valid
        and selected is not None
        and selected_name == EXPECTED_SELECTED_FEATURE
        and selected.residual_harmful_case_count
        == EXPECTED_RESIDUAL_HARMFUL_COUNT
        and selected.limiting_tail_value is not None
        and math.isclose(
            selected.limiting_tail_value,
            EXPECTED_LIMITING_TAIL_VALUE,
            rel_tol=0.0,
            abs_tol=REPRODUCTION_TOLERANCE,
        )
        and selected_cutoff is not None
        and math.isclose(
            selected_cutoff,
            EXPECTED_TAIL_REJECTION_CUTOFF,
            rel_tol=0.0,
            abs_tol=REPRODUCTION_TOLERANCE,
        )
        and selected.original_focus_safe_accept_count
        == EXPECTED_ORIGINAL_FOCUS_COUNT
        and selected.predecessor_focus_safe_accept_count
        == EXPECTED_PREDECESSOR_FOCUS_COUNT
        and selected.combined_focus_safe_accept_count
        == EXPECTED_TAIL_RETAINED_FOCUS_COUNT
        and selected.original_safe_accept_count == EXPECTED_ORIGINAL_SAFE_COUNT
        and selected.predecessor_safe_accept_count
        == EXPECTED_PREDECESSOR_SAFE_COUNT
        and selected.combined_safe_accept_count
        == EXPECTED_TAIL_RETAINED_SAFE_COUNT
    )
    design_gate_passed = bool(
        design_reproduced
        and seed_audit.passed
        and len(design_panels) == 9
        and all(panel.panel_gate_passed for panel in design_panels)
    )
    fresh_panels: list[LocalSpatialPanel] = []

    def fresh_panel(seed: int, role: str) -> LocalSpatialPanel:
        if selected is None:
            raise RuntimeError("a selected tail feature is required")
        q95_panel = _target_q95_panel(
            seed=seed,
            role=role,
            phase28=phase28,
            reference_count=reference_count,
            surface_sample_count=surface_sample_count,
            full_protocol=full_protocol,
        )
        return _materialize_tail_panel(
            q95_panel,
            feature_name=selected.feature_name,
            rejection_cutoff=selected.rejection_cutoff,
            panel_role=role,
            full_protocol=full_protocol,
        )

    if open_fresh and design_gate_passed:
        validation_a = fresh_panel(validation_a_seed, "validation_a")
        fresh_panels.append(validation_a)
        if validation_a.panel_gate_passed:
            validation_b = fresh_panel(validation_b_seed, "validation_b")
            fresh_panels.append(validation_b)
            if validation_b.panel_gate_passed:
                final_panel = fresh_panel(final_held_out_seed, "final_held_out")
                fresh_panels.append(final_panel)
    validation_a = fresh_panels[0] if len(fresh_panels) >= 1 else None
    validation_b = fresh_panels[1] if len(fresh_panels) >= 2 else None
    final_panel = fresh_panels[2] if len(fresh_panels) >= 3 else None
    incremental = summarize_incremental_evidence(fresh_panels)
    supported = incremental.incremental_evidence_gate_passed
    return TailSensitiveLocalGuardResult(
        artifact_schema="pftf_alpha_tail_sensitive_local_guard_phase30/v1",
        role="synthetic_tail_sensitive_local_guard_transfer_audit",
        information_boundary=(
            "route adds the frozen maximum-to-q95 local-residual ratio to the "
            "Phase-28 score/q95 route; all inputs are observed primary and "
            "repeat coordinates, while endpoint truth is evaluation-only"
        ),
        frozen_predecessor="phase28_score_plus_q95_local_route",
        target_point_counts=TARGET_POINT_COUNTS,
        target_stresses=TARGET_STRESSES,
        target_repeats=TARGET_REPEATS,
        tail_design_seed=TAIL_DESIGN_SEED,
        validation_a_seed=validation_a_seed,
        validation_b_seed=validation_b_seed,
        final_held_out_seed=final_held_out_seed,
        reference_count=reference_count,
        surface_sample_count=surface_sample_count,
        case_seed_disjointness=seed_audit,
        phase28_reproduced_and_supported=phase28_reproduced,
        phase29_failure_reproduced=phase29_failure_reproduced,
        calibration=calibration,
        design_panels=design_panels,
        design_reproduced=design_reproduced,
        design_gate_passed=design_gate_passed,
        fresh_execution_requested=open_fresh,
        validation_a=validation_a,
        validation_b=validation_b,
        final_held_out=final_panel,
        incremental_evidence=incremental,
        phase30_supported=supported,
        tail_sensitive_local_guard_synthetic_supported=supported,
        real_correspondence_supported=False,
        real_paired_scan_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--design-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_tail_sensitive_local_guard(open_fresh=not args.design_only)
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result.phase30_supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
