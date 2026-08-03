"""Local/spatial residual guard transfer audit for Phase 28."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .focus_envelope_cutoff import (
    FINAL_HELD_OUT_SEED as PHASE27_FINAL_SEED,
)
from .focus_envelope_cutoff import (
    VALIDATION_A_SEED as LOCAL_CALIBRATION_SEED,
)
from .focus_envelope_cutoff import (
    VALIDATION_B_SEED as PHASE27_VALIDATION_B_SEED,
)
from .focus_envelope_cutoff import evaluate_focus_envelope_cutoff
from .frozen_partition_reconstruction import (
    FrozenPartitionCaseResult,
    FrozenPartitionPanel,
    _materialize_panel,
    _panel_case_seeds,
)
from .local_spatial_displacement import (
    LocalSpatialDisplacementConfig,
    LocalSpatialDisplacementEvidence,
)
from .local_surface_consensus import GeometryTopologyHarmEndpoint
from .matched_guard_signature import GUARD_PROFILE_SPECS, MatchedGuardModel
from .matched_pair_consistency import MatchedPairConfig
from .matched_pair_stress import (
    MatchedPairStressConfig,
    MatchedPairStressProfile,
    MatchedPairStressSpec,
    _raw_panel,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .sensor_stress import DEFAULT_POINT_COUNTS, DEFAULT_STRESSES, SensorStress
from .shared_trend_inference import SharedTrendConfig

LOCAL_DESIGN_A_SEED = 27500804
LOCAL_DESIGN_B_SEED = 27600804
VALIDATION_A_SEED = 28100804
VALIDATION_B_SEED = 28200804
FINAL_HELD_OUT_SEED = 28300804
PRIOR_BASE_SEEDS = tuple(index * 100000 + 804 for index in range(203, 260)) + (
    LOCAL_DESIGN_A_SEED,
    LOCAL_DESIGN_B_SEED,
    27700804,
    LOCAL_CALIBRATION_SEED,
    PHASE27_VALIDATION_B_SEED,
    PHASE27_FINAL_SEED,
)
LOCAL_FEATURE_NAME = "percentile95_local_residual"
EXPECTED_RESIDUAL_HARMFUL_COUNT = 1
EXPECTED_LIMITING_LOCAL_VALUE = 3.5441330652515526
EXPECTED_LOCAL_REJECTION_CUTOFF = 3.544133065251552
EXPECTED_LOCAL_ONLY_RETAINED_FOCUS_COUNT = 379
EXPECTED_COMBINED_RETAINED_FOCUS_COUNT = 378
EXPECTED_DESIGN_FOCUS_COUNT = 381
REPRODUCTION_TOLERANCE = 1.0e-15


@dataclass(frozen=True)
class Phase28CaseSeedAudit:
    case_seed_formula: str
    prior_base_seed_ranges: tuple[str, ...]
    panel_case_count: int
    validation_a_prior_overlap_count: int
    validation_b_prior_overlap_count: int
    final_prior_overlap_count: int
    validation_a_b_overlap_count: int
    validation_a_final_overlap_count: int
    validation_b_final_overlap_count: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalResidualCalibration:
    feature_name: str
    design_seeds: tuple[int, ...]
    residual_harmful_case_count: int
    limiting_case_seed: int | None
    limiting_profile: MatchedPairStressProfile | None
    limiting_stress: SensorStress | None
    limiting_point_count: int | None
    limiting_repeat: int | None
    limiting_local_value: float | None
    rejection_cutoff: float
    binary64_decrement: float | None
    focus_case_count: int
    local_only_retained_focus_count: int
    combined_retained_focus_count: int
    calibration_valid: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["limiting_profile"] = (
            None if self.limiting_profile is None else self.limiting_profile.value
        )
        payload["limiting_stress"] = (
            None if self.limiting_stress is None else self.limiting_stress.value
        )
        return payload


@dataclass(frozen=True)
class LocalSpatialGuardCase:
    profile: MatchedPairStressProfile
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    perturbation_seed: int
    model_score: float
    local_spatial_evidence: LocalSpatialDisplacementEvidence
    original_endpoint: GeometryTopologyHarmEndpoint
    routed_endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision
    predecessor_guarded_decision: SamplingGateDecision
    guarded_decision: SamplingGateDecision
    unguarded_safe_accept: bool
    predecessor_guarded_safe_accept: bool
    guarded_safe_accept: bool
    unguarded_harmful_outlier_false_safe: bool
    predecessor_guarded_harmful_outlier_false_safe: bool
    guarded_harmful_outlier_false_safe: bool
    introduced_routed_endpoint_harm_accept: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        payload["stress"] = self.stress.value
        payload["unguarded_decision"] = self.unguarded_decision.value
        payload["predecessor_guarded_decision"] = (
            self.predecessor_guarded_decision.value
        )
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class LocalSpatialProfileSummary:
    profile: MatchedPairStressProfile
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    predecessor_guarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    focus_unguarded_safe_accept_count: int
    focus_predecessor_guarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    all_stress_unguarded_safe_accept_count: int
    all_stress_predecessor_guarded_safe_accept_count: int
    all_stress_guarded_safe_accept_count: int
    introduced_routed_endpoint_harm_accept_count: int
    profile_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        return payload


@dataclass(frozen=True)
class LocalSpatialPanel:
    panel_role: str
    seed: int
    cases: tuple[LocalSpatialGuardCase, ...]
    profile_summaries: tuple[LocalSpatialProfileSummary, ...]
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    predecessor_guarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    focus_unguarded_safe_accept_count: int
    focus_predecessor_guarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    all_stress_unguarded_safe_accept_count: int
    all_stress_predecessor_guarded_safe_accept_count: int
    all_stress_guarded_safe_accept_count: int
    introduced_routed_endpoint_harm_accept_count: int
    full_protocol: bool
    panel_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "cases": [case.to_dict() for case in self.cases],
            "profile_summaries": [
                summary.to_dict() for summary in self.profile_summaries
            ],
        }


@dataclass(frozen=True)
class LocalSpatialResidualGuardResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    local_design_a_seed: int
    local_design_b_seed: int
    local_calibration_seed: int
    validation_a_seed: int
    validation_b_seed: int
    final_held_out_seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    profile_specs: tuple[MatchedPairStressSpec, ...]
    matched_pair_config: MatchedPairConfig
    stress_config: MatchedPairStressConfig
    local_config: LocalSpatialDisplacementConfig
    case_seed_disjointness: Phase28CaseSeedAudit
    predecessor_model: MatchedGuardModel
    local_calibration: LocalResidualCalibration
    design_score_fit: LocalSpatialPanel
    design_cutoff_calibration: LocalSpatialPanel
    local_design_a: LocalSpatialPanel
    local_design_b: LocalSpatialPanel
    local_calibration_panel: LocalSpatialPanel | None
    design_reproduced: bool
    design_gate_passed: bool
    fresh_execution_requested: bool
    validation_a: LocalSpatialPanel | None
    validation_b: LocalSpatialPanel | None
    final_held_out: LocalSpatialPanel | None
    phase28_supported: bool
    local_spatial_residual_guard_synthetic_supported: bool
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
            "local_design_a_seed": self.local_design_a_seed,
            "local_design_b_seed": self.local_design_b_seed,
            "local_calibration_seed": self.local_calibration_seed,
            "validation_a_seed": self.validation_a_seed,
            "validation_b_seed": self.validation_b_seed,
            "final_held_out_seed": self.final_held_out_seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "profile_specs": [
                {**asdict(spec), "profile": spec.profile.value}
                for spec in self.profile_specs
            ],
            "matched_pair_config": asdict(self.matched_pair_config),
            "stress_config": asdict(self.stress_config),
            "local_config": asdict(self.local_config),
            "case_seed_disjointness": self.case_seed_disjointness.to_dict(),
            "predecessor_model": self.predecessor_model.to_dict(),
            "local_calibration": self.local_calibration.to_dict(),
            "design_score_fit": self.design_score_fit.to_dict(),
            "design_cutoff_calibration": self.design_cutoff_calibration.to_dict(),
            "local_design_a": self.local_design_a.to_dict(),
            "local_design_b": self.local_design_b.to_dict(),
            "local_calibration_panel": (
                None
                if self.local_calibration_panel is None
                else self.local_calibration_panel.to_dict()
            ),
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
            "phase28_supported": self.phase28_supported,
            "local_spatial_residual_guard_synthetic_supported": (
                self.local_spatial_residual_guard_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "real_trimmed_reconstruction_supported": (
                self.real_trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def audit_phase28_case_seed_disjointness(
    validation_a_seed: int,
    validation_b_seed: int,
    final_held_out_seed: int,
    *,
    point_counts: tuple[int, ...] = DEFAULT_POINT_COUNTS,
    stresses: tuple[SensorStress, ...] = DEFAULT_STRESSES,
    repeats: int = 8,
) -> Phase28CaseSeedAudit:
    prior = frozenset().union(
        *(
            _panel_case_seeds(
                seed,
                point_counts=DEFAULT_POINT_COUNTS,
                stresses=DEFAULT_STRESSES,
                repeats=8,
            )
            for seed in PRIOR_BASE_SEEDS
        )
    )
    panels = tuple(
        _panel_case_seeds(
            seed,
            point_counts=point_counts,
            stresses=stresses,
            repeats=repeats,
        )
        for seed in (validation_a_seed, validation_b_seed, final_held_out_seed)
    )
    prior_overlaps = tuple(len(panel & prior) for panel in panels)
    mutual_overlaps = (
        len(panels[0] & panels[1]),
        len(panels[0] & panels[2]),
        len(panels[1] & panels[2]),
    )
    return Phase28CaseSeedAudit(
        case_seed_formula=(
            "base + count_index*1000003 + stress_index*100003 + repeat*10007"
        ),
        prior_base_seed_ranges=(
            "20300804--25900804",
            "27500804--28000804",
        ),
        panel_case_count=len(panels[0]),
        validation_a_prior_overlap_count=prior_overlaps[0],
        validation_b_prior_overlap_count=prior_overlaps[1],
        final_prior_overlap_count=prior_overlaps[2],
        validation_a_b_overlap_count=mutual_overlaps[0],
        validation_a_final_overlap_count=mutual_overlaps[1],
        validation_b_final_overlap_count=mutual_overlaps[2],
        passed=not any(prior_overlaps + mutual_overlaps),
    )


def _local_value(case: FrozenPartitionCaseResult) -> float:
    evidence = case.local_spatial_evidence
    if evidence is None:
        return math.inf
    return evidence.percentile95_local_residual


def calibrate_local_residual_cutoff(
    panels: Sequence[FrozenPartitionPanel],
) -> LocalResidualCalibration:
    cases = tuple(case for panel in panels for case in panel.cases)
    residual = tuple(
        case for case in cases if case.guarded_harmful_outlier_false_safe
    )
    focus = tuple(
        case
        for case in cases
        if case.unguarded_safe_accept
        and case.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    )
    limiting = min(residual, key=_local_value) if residual else None
    limiting_value = None if limiting is None else _local_value(limiting)
    valid = bool(
        residual
        and focus
        and limiting_value is not None
        and math.isfinite(limiting_value)
    )
    cutoff = (
        math.nextafter(limiting_value, -math.inf)
        if valid and limiting_value is not None
        else -sys.float_info.max
    )
    local_retained = sum(_local_value(case) < cutoff for case in focus)
    combined_retained = sum(
        case.guarded_safe_accept and _local_value(case) < cutoff for case in focus
    )
    return LocalResidualCalibration(
        feature_name=LOCAL_FEATURE_NAME,
        design_seeds=tuple(panel.seed for panel in panels),
        residual_harmful_case_count=len(residual),
        limiting_case_seed=None if limiting is None else limiting.seed,
        limiting_profile=None if limiting is None else limiting.profile,
        limiting_stress=None if limiting is None else limiting.stress,
        limiting_point_count=None if limiting is None else limiting.point_count,
        limiting_repeat=None if limiting is None else limiting.repeat,
        limiting_local_value=limiting_value,
        rejection_cutoff=cutoff,
        binary64_decrement=(
            None if limiting_value is None else limiting_value - cutoff
        ),
        focus_case_count=len(focus),
        local_only_retained_focus_count=local_retained,
        combined_retained_focus_count=combined_retained,
        calibration_valid=valid,
    )


def _materialize_local_case(
    case: FrozenPartitionCaseResult,
    calibration: LocalResidualCalibration,
) -> LocalSpatialGuardCase:
    evidence = case.local_spatial_evidence
    if evidence is None:
        raise ValueError("local/spatial evidence is required")
    unguarded_accept = case.unguarded_decision is SamplingGateDecision.ACCEPT
    predecessor_accept = (
        case.guarded_decision is SamplingGateDecision.ACCEPT
    )
    local_accept = bool(
        calibration.calibration_valid
        and evidence.percentile95_local_residual < calibration.rejection_cutoff
    )
    guarded_accept = bool(predecessor_accept and local_accept)
    guarded_decision = (
        SamplingGateDecision.ACCEPT
        if guarded_accept
        else (
            SamplingGateDecision.UNSUPPORTED
            if unguarded_accept
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
        local_spatial_evidence=evidence,
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


def _profile_summary(
    rows: Sequence[LocalSpatialGuardCase],
    profile: MatchedPairStressProfile,
    *,
    full_protocol: bool,
    calibration_valid: bool,
) -> LocalSpatialProfileSummary:
    selected = tuple(row for row in rows if row.profile is profile)
    focus = tuple(
        row
        for row in selected
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    )
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    predecessor_focus = sum(row.predecessor_guarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    all_safe = tuple(row for row in selected if row.unguarded_safe_accept)
    unguarded_harm = sum(
        row.unguarded_harmful_outlier_false_safe for row in selected
    )
    predecessor_harm = sum(
        row.predecessor_guarded_harmful_outlier_false_safe for row in selected
    )
    guarded_harm = sum(
        row.guarded_harmful_outlier_false_safe for row in selected
    )
    introduced_harm = sum(
        row.introduced_routed_endpoint_harm_accept for row in selected
    )
    retention = 0.0 if not unguarded_focus else guarded_focus / unguarded_focus
    return LocalSpatialProfileSummary(
        profile=profile,
        case_count=len(selected),
        unguarded_harmful_outlier_false_safe_count=unguarded_harm,
        predecessor_guarded_harmful_outlier_false_safe_count=predecessor_harm,
        guarded_harmful_outlier_false_safe_count=guarded_harm,
        focus_unguarded_safe_accept_count=unguarded_focus,
        focus_predecessor_guarded_safe_accept_count=predecessor_focus,
        focus_guarded_safe_accept_count=guarded_focus,
        focus_safe_accept_retention=retention,
        all_stress_unguarded_safe_accept_count=len(all_safe),
        all_stress_predecessor_guarded_safe_accept_count=sum(
            row.predecessor_guarded_safe_accept for row in all_safe
        ),
        all_stress_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in all_safe
        ),
        introduced_routed_endpoint_harm_accept_count=introduced_harm,
        profile_gate_passed=bool(
            full_protocol
            and calibration_valid
            and unguarded_harm > 0
            and guarded_harm == 0
            and retention >= 0.90
            and introduced_harm == 0
        ),
    )


def _materialize_local_panel(
    panel: FrozenPartitionPanel,
    calibration: LocalResidualCalibration,
    *,
    panel_role: str,
    full_protocol: bool,
) -> LocalSpatialPanel:
    rows = tuple(
        _materialize_local_case(case, calibration) for case in panel.cases
    )
    summaries = tuple(
        _profile_summary(
            rows,
            spec.profile,
            full_protocol=full_protocol,
            calibration_valid=calibration.calibration_valid,
        )
        for spec in GUARD_PROFILE_SPECS
    )
    focus = tuple(
        row
        for row in rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    )
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
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
        focus_unguarded_safe_accept_count=unguarded_focus,
        focus_predecessor_guarded_safe_accept_count=sum(
            row.predecessor_guarded_safe_accept for row in focus
        ),
        focus_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in focus
        ),
        focus_safe_accept_retention=(
            0.0
            if not unguarded_focus
            else sum(row.guarded_safe_accept for row in focus) / unguarded_focus
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


def evaluate_local_spatial_residual_guard(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    validation_a_seed: int = VALIDATION_A_SEED,
    validation_b_seed: int = VALIDATION_B_SEED,
    final_held_out_seed: int = FINAL_HELD_OUT_SEED,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
    matched_pair_config: MatchedPairConfig | None = None,
    stress_config: MatchedPairStressConfig | None = None,
    profile_specs: Sequence[MatchedPairStressSpec] = GUARD_PROFILE_SPECS,
    open_fresh: bool = True,
) -> LocalSpatialResidualGuardResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_matched = (
        MatchedPairConfig() if matched_pair_config is None else matched_pair_config
    )
    selected_stress = (
        MatchedPairStressConfig() if stress_config is None else stress_config
    )
    selected_profiles = tuple(profile_specs)
    local_config = LocalSpatialDisplacementConfig()
    fresh_seeds = (validation_a_seed, validation_b_seed, final_held_out_seed)
    if len(set(fresh_seeds)) != len(fresh_seeds):
        raise ValueError("validation and final base seeds must differ")
    seed_audit = audit_phase28_case_seed_disjointness(
        *fresh_seeds,
        point_counts=selected_counts,
        stresses=selected_stresses,
        repeats=repeats,
    )
    if not seed_audit.passed:
        raise ValueError(
            "validation/final case seeds must be mutually disjoint and disjoint "
            "from all prior and reserved panels"
        )
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and selected_profiles == GUARD_PROFILE_SPECS
        and repeats >= 8
        and reference_count >= 2048
        and surface_sample_count >= 256
        and fresh_seeds
        == (VALIDATION_A_SEED, VALIDATION_B_SEED, FINAL_HELD_OUT_SEED)
    )
    predecessor = evaluate_focus_envelope_cutoff(
        point_counts=selected_counts,
        stresses=selected_stresses,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        base_gate_config=base_gate_config,
        shared_trend_config=shared_trend_config,
        matched_pair_config=selected_matched,
        stress_config=selected_stress,
        profile_specs=selected_profiles,
    )
    calibration_inputs = [predecessor.cutoff_design_a, predecessor.cutoff_design_b]
    if predecessor.validation_a is not None:
        calibration_inputs.append(predecessor.validation_a)
    calibration = calibrate_local_residual_cutoff(calibration_inputs)

    design_score_fit = _materialize_local_panel(
        predecessor.design_score_fit,
        calibration,
        panel_role="design_score_fit",
        full_protocol=full_protocol,
    )
    design_cutoff_calibration = _materialize_local_panel(
        predecessor.design_cutoff_calibration,
        calibration,
        panel_role="design_cutoff_calibration",
        full_protocol=full_protocol,
    )
    local_design_a = _materialize_local_panel(
        predecessor.cutoff_design_a,
        calibration,
        panel_role="local_design_a",
        full_protocol=full_protocol,
    )
    local_design_b = _materialize_local_panel(
        predecessor.cutoff_design_b,
        calibration,
        panel_role="local_design_b",
        full_protocol=full_protocol,
    )
    local_calibration_panel = (
        None
        if predecessor.validation_a is None
        else _materialize_local_panel(
            predecessor.validation_a,
            calibration,
            panel_role="local_calibration",
            full_protocol=full_protocol,
        )
    )
    reproduced_limiting_value = (
        -sys.float_info.max
        if calibration.limiting_local_value is None
        else calibration.limiting_local_value
    )
    design_reproduced = bool(
        full_protocol
        and predecessor.design_reproduced
        and predecessor.design_gate_passed
        and predecessor.validation_a is not None
        and not predecessor.validation_a.panel_gate_passed
        and predecessor.validation_b is None
        and predecessor.final_held_out is None
        and not predecessor.phase27_supported
        and calibration.calibration_valid
        and calibration.residual_harmful_case_count
        == EXPECTED_RESIDUAL_HARMFUL_COUNT
        and calibration.focus_case_count == EXPECTED_DESIGN_FOCUS_COUNT
        and calibration.local_only_retained_focus_count
        == EXPECTED_LOCAL_ONLY_RETAINED_FOCUS_COUNT
        and calibration.combined_retained_focus_count
        == EXPECTED_COMBINED_RETAINED_FOCUS_COUNT
        and math.isclose(
            reproduced_limiting_value,
            EXPECTED_LIMITING_LOCAL_VALUE,
            rel_tol=0.0,
            abs_tol=REPRODUCTION_TOLERANCE,
        )
        and math.isclose(
            calibration.rejection_cutoff,
            EXPECTED_LOCAL_REJECTION_CUTOFF,
            rel_tol=0.0,
            abs_tol=REPRODUCTION_TOLERANCE,
        )
    )
    design_panels = (
        design_score_fit,
        design_cutoff_calibration,
        local_design_a,
        local_design_b,
    )
    design_gate_passed = bool(
        design_reproduced
        and seed_audit.passed
        and all(panel.panel_gate_passed for panel in design_panels)
        and local_calibration_panel is not None
        and local_calibration_panel.panel_gate_passed
    )

    validation_a: LocalSpatialPanel | None = None
    validation_b: LocalSpatialPanel | None = None
    final_panel: LocalSpatialPanel | None = None
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
        "profile_specs": selected_profiles,
    }

    def fresh_panel(seed: int, role: str) -> LocalSpatialPanel:
        predecessor_panel = _materialize_panel(
            _raw_panel(seed=seed, **common),
            panel_role=f"{role}_predecessor",
            seed=seed,
            model=predecessor.model,
            profile_specs=selected_profiles,
            full_protocol=full_protocol,
        )
        return _materialize_local_panel(
            predecessor_panel,
            calibration,
            panel_role=role,
            full_protocol=full_protocol,
        )

    if open_fresh and design_gate_passed:
        validation_a = fresh_panel(validation_a_seed, "validation_a")
        if validation_a.panel_gate_passed:
            validation_b = fresh_panel(validation_b_seed, "validation_b")
        if validation_b is not None and validation_b.panel_gate_passed:
            final_panel = fresh_panel(final_held_out_seed, "final_held_out")
    supported = bool(final_panel is not None and final_panel.panel_gate_passed)
    return LocalSpatialResidualGuardResult(
        artifact_schema="pftf_alpha_local_spatial_residual_guard_phase28/v1",
        role="synthetic_local_spatial_residual_guard_transfer_audit",
        information_boundary=(
            "route uses the frozen Phase-27 score plus primary-coordinate kNN "
            "and ordered matched displacement vectors; source labels, injected "
            "outlier identities, clean reference, and endpoint truth are hidden"
        ),
        frozen_predecessor="phase27_focus_envelope_transfer_failure",
        local_design_a_seed=LOCAL_DESIGN_A_SEED,
        local_design_b_seed=LOCAL_DESIGN_B_SEED,
        local_calibration_seed=LOCAL_CALIBRATION_SEED,
        validation_a_seed=validation_a_seed,
        validation_b_seed=validation_b_seed,
        final_held_out_seed=final_held_out_seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        profile_specs=selected_profiles,
        matched_pair_config=selected_matched,
        stress_config=selected_stress,
        local_config=local_config,
        case_seed_disjointness=seed_audit,
        predecessor_model=predecessor.model,
        local_calibration=calibration,
        design_score_fit=design_score_fit,
        design_cutoff_calibration=design_cutoff_calibration,
        local_design_a=local_design_a,
        local_design_b=local_design_b,
        local_calibration_panel=local_calibration_panel,
        design_reproduced=design_reproduced,
        design_gate_passed=design_gate_passed,
        fresh_execution_requested=open_fresh,
        validation_a=validation_a,
        validation_b=validation_b,
        final_held_out=final_panel,
        phase28_supported=supported,
        local_spatial_residual_guard_synthetic_supported=supported,
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
    result = evaluate_local_spatial_residual_guard(
        open_fresh=not args.design_only,
    )
    payload = result.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result.phase28_supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
