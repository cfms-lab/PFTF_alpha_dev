"""Multivariate observed-only cycle-signature audit for Phase 22."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .cycle_gated_assignment import (
    AssignmentCycle,
    CycleCandidateCase,
    CycleGatedCaseResult,
    CycleGatedProfileSummary,
    _candidate_cases,
    _materialize_case_with_selection,
    _profile_summary,
)
from .local_insertion_influence import InfluenceRectangle
from .matched_pair_consistency import MatchedPairConfig
from .matched_pair_stress import (
    DEFAULT_STRESS_SPECS,
    MatchedPairStressConfig,
    MatchedPairStressSpec,
)
from .sampling_gate import SamplingSufficiencyConfig
from .sensor_stress import DEFAULT_POINT_COUNTS, DEFAULT_STRESSES, SensorStress
from .shared_trend_inference import SharedTrendConfig
from .tangential_pair_confidence import (
    FROZEN_PHASE18_RECTANGLE,
    PairConfidenceConfig,
    _raw_panel,
)

FloatArray = NDArray[np.float64]

TRAINING_A_SEED = 23900804
DEVELOPMENT_B_SEED = 24000804
VALIDATION_A_SEED = 24100804
VALIDATION_B_SEED = 24200804
FINAL_HELD_OUT_SEED = 24300804
FORBIDDEN_PRIOR_SEEDS = frozenset(
    {
        23300804,
        23400804,
        23500804,
        23600804,
        23700804,
        23800804,
    }
)

LOG_RATIO_FLOOR = 1.0e-6
LOG_RATIO_LIMIT = 20.0
RIDGE_PENALTY = 1.0
MINIMUM_FEATURE_SCALE = 1.0e-12

SIGNATURE_FEATURE_NAMES = (
    "log_cycle_length",
    "tangent_relative_gain",
    "tangent_log_ratio_minimum",
    "tangent_log_ratio_median",
    "tangent_positive_fraction",
    "normal_log_ratio_minimum",
    "normal_log_ratio_median",
    "normal_positive_fraction",
    "total_log_ratio_minimum",
    "total_log_ratio_median",
    "total_positive_fraction",
    "log1p_presented_tangent_median",
    "log1p_presented_tangent_maximum",
    "log1p_assigned_tangent_median",
)


@dataclass(frozen=True)
class MultivariateCycleSignature:
    cycle: AssignmentCycle
    values: tuple[float, ...]

    @property
    def strictly_correcting(self) -> bool:
        return bool(
            self.cycle.truth_correct_after == len(self.cycle.rows)
            and self.cycle.truth_correct_after
            > self.cycle.truth_correct_before
        )


@dataclass(frozen=True)
class SignatureCandidateCase:
    candidate: CycleCandidateCase
    signatures: tuple[MultivariateCycleSignature, ...]


@dataclass(frozen=True)
class CycleSignatureModel:
    feature_names: tuple[str, ...]
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    cutoff: float
    ridge_penalty: float
    calibration_valid: bool
    training_cycle_count: int
    training_strictly_correcting_cycle_count: int
    training_unsafe_cycle_count: int
    accepted_training_strictly_correcting_cycle_count: int
    rejected_training_strictly_correcting_cycle_count: int
    accepted_training_unsafe_cycle_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultivariateCyclePanel:
    panel_role: str
    seed: int
    frozen_rectangle: InfluenceRectangle
    cases: tuple[CycleGatedCaseResult, ...]
    profile_summaries: tuple[CycleGatedProfileSummary, ...]
    case_count: int
    pair_count: int
    correct_assignment_count: int
    assignment_accuracy: float
    signature_cycle_count: int
    accepted_cycle_count: int
    accepted_strictly_correcting_cycle_count: int
    rejected_strictly_correcting_cycle_count: int
    accepted_unsafe_cycle_count: int
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
            "signature_cycle_count": self.signature_cycle_count,
            "accepted_cycle_count": self.accepted_cycle_count,
            "accepted_strictly_correcting_cycle_count": (
                self.accepted_strictly_correcting_cycle_count
            ),
            "rejected_strictly_correcting_cycle_count": (
                self.rejected_strictly_correcting_cycle_count
            ),
            "accepted_unsafe_cycle_count": self.accepted_unsafe_cycle_count,
            "unguarded_harmful_outlier_false_safe_count": (
                self.unguarded_harmful_outlier_false_safe_count
            ),
            "guarded_harmful_outlier_false_safe_count": (
                self.guarded_harmful_outlier_false_safe_count
            ),
            "focus_unguarded_safe_accept_count": (
                self.focus_unguarded_safe_accept_count
            ),
            "focus_guarded_safe_accept_count": (
                self.focus_guarded_safe_accept_count
            ),
            "focus_safe_accept_retention": self.focus_safe_accept_retention,
            "full_protocol": self.full_protocol,
            "panel_gate_passed": self.panel_gate_passed,
        }


@dataclass(frozen=True)
class MultivariateCycleSignatureResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    training_a_seed: int
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
    log_ratio_floor: float
    log_ratio_limit: float
    model: CycleSignatureModel
    training_a: MultivariateCyclePanel
    development_b: MultivariateCyclePanel | None
    development_screen_passed: bool
    validation_a: MultivariateCyclePanel | None
    validation_b: MultivariateCyclePanel | None
    final_held_out: MultivariateCyclePanel | None
    phase22_supported: bool
    multivariate_cycle_signature_synthetic_supported: bool
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
            "training_a_seed": self.training_a_seed,
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
            "log_ratio_floor": self.log_ratio_floor,
            "log_ratio_limit": self.log_ratio_limit,
            "model": self.model.to_dict(),
            "training_a": self.training_a.to_dict(),
            "development_b": (
                None
                if self.development_b is None
                else self.development_b.to_dict()
            ),
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
            "phase22_supported": self.phase22_supported,
            "multivariate_cycle_signature_synthetic_supported": (
                self.multivariate_cycle_signature_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def _log_cost_ratio(presented: FloatArray, assigned: FloatArray) -> FloatArray:
    values = np.log(
        (np.asarray(presented, dtype=np.float64) + LOG_RATIO_FLOOR)
        / (np.asarray(assigned, dtype=np.float64) + LOG_RATIO_FLOOR)
    )
    return np.clip(values, -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)


def cycle_signature(
    candidate: CycleCandidateCase,
    cycle: AssignmentCycle,
) -> MultivariateCycleSignature:
    rows = np.asarray(cycle.rows, dtype=np.int64)
    assignment = candidate.global_assignment
    presented_tangent = assignment.presented_costs[rows]
    assigned_tangent = assignment.assigned_costs[rows]
    presented_normal = assignment.presented_normal_costs[rows]
    assigned_normal = assignment.assigned_normal_costs[rows]
    presented_total = np.hypot(presented_tangent, presented_normal)
    assigned_total = np.hypot(assigned_tangent, assigned_normal)
    tangent_ratio = _log_cost_ratio(presented_tangent, assigned_tangent)
    normal_ratio = _log_cost_ratio(presented_normal, assigned_normal)
    total_ratio = _log_cost_ratio(presented_total, assigned_total)
    values = (
        math.log(float(rows.size)),
        cycle.relative_gain,
        float(np.min(tangent_ratio)),
        float(np.median(tangent_ratio)),
        float(np.mean(tangent_ratio > 0.0)),
        float(np.min(normal_ratio)),
        float(np.median(normal_ratio)),
        float(np.mean(normal_ratio > 0.0)),
        float(np.min(total_ratio)),
        float(np.median(total_ratio)),
        float(np.mean(total_ratio > 0.0)),
        math.log1p(float(np.median(presented_tangent))),
        math.log1p(float(np.max(presented_tangent))),
        math.log1p(float(np.median(assigned_tangent))),
    )
    if len(values) != len(SIGNATURE_FEATURE_NAMES):
        raise RuntimeError("cycle signature does not match its declared schema")
    if not np.all(np.isfinite(values)):
        raise ValueError("cycle signature must be finite")
    return MultivariateCycleSignature(cycle=cycle, values=values)


def _signature_cases(
    candidates: Sequence[CycleCandidateCase],
) -> tuple[SignatureCandidateCase, ...]:
    return tuple(
        SignatureCandidateCase(
            candidate=candidate,
            signatures=tuple(
                cycle_signature(candidate, cycle) for cycle in candidate.cycles
            ),
        )
        for candidate in candidates
    )


def score_cycle_signature(
    model: CycleSignatureModel,
    signature: MultivariateCycleSignature,
) -> float:
    values = np.asarray(signature.values, dtype=np.float64)
    center = np.asarray(model.feature_center, dtype=np.float64)
    scale = np.asarray(model.feature_scale, dtype=np.float64)
    coefficients = np.asarray(model.coefficients, dtype=np.float64)
    if values.shape != center.shape or scale.shape != center.shape:
        raise ValueError("model and cycle signature dimensions must agree")
    return float(model.intercept + ((values - center) / scale) @ coefficients)


def fit_cycle_signature_model(
    signatures: Sequence[MultivariateCycleSignature],
) -> CycleSignatureModel:
    rows = tuple(signatures)
    if not rows:
        raise ValueError("at least one cycle signature is required")
    features = np.asarray([row.values for row in rows], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(SIGNATURE_FEATURE_NAMES):
        raise ValueError("cycle signature matrix has the wrong shape")
    if not np.all(np.isfinite(features)):
        raise ValueError("cycle signature matrix must be finite")
    targets = np.asarray(
        [row.strictly_correcting for row in rows],
        dtype=np.float64,
    )
    safe_count = int(np.sum(targets))
    unsafe_count = len(rows) - safe_count
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < MINIMUM_FEATURE_SCALE, 1.0, scale)
    standardized = (features - center) / scale
    valid = bool(safe_count > 0 and unsafe_count > 0)
    if valid:
        design = np.column_stack((np.ones(len(rows)), standardized))
        penalty = np.diag(
            np.asarray((0.0,) + (RIDGE_PENALTY,) * features.shape[1])
        )
        beta = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ targets,
        )
        scores = np.asarray(
            [
                beta[0] + standardized[index] @ beta[1:]
                for index in range(len(rows))
            ],
            dtype=np.float64,
        )
        maximum_unsafe = float(np.max(scores[targets == 0.0]))
        cutoff = float(np.nextafter(maximum_unsafe, math.inf))
    else:
        beta = np.zeros(features.shape[1] + 1, dtype=np.float64)
        cutoff = np.finfo(float).max
        scores = np.zeros(len(rows), dtype=np.float64)
    accepted = scores >= cutoff
    accepted_safe = int(np.sum(accepted & (targets == 1.0)))
    accepted_unsafe = int(np.sum(accepted & (targets == 0.0)))
    return CycleSignatureModel(
        feature_names=SIGNATURE_FEATURE_NAMES,
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        intercept=float(beta[0]),
        coefficients=tuple(float(value) for value in beta[1:]),
        cutoff=cutoff,
        ridge_penalty=RIDGE_PENALTY,
        calibration_valid=valid,
        training_cycle_count=len(rows),
        training_strictly_correcting_cycle_count=safe_count,
        training_unsafe_cycle_count=unsafe_count,
        accepted_training_strictly_correcting_cycle_count=accepted_safe,
        rejected_training_strictly_correcting_cycle_count=(
            safe_count - accepted_safe
        ),
        accepted_training_unsafe_cycle_count=accepted_unsafe,
    )


def _materialize_panel(
    signature_cases: Sequence[SignatureCandidateCase],
    *,
    panel_role: str,
    seed: int,
    model: CycleSignatureModel,
    rectangle: InfluenceRectangle,
    matched_pair_config: MatchedPairConfig,
    profile_specs: tuple[MatchedPairStressSpec, ...],
    full_protocol: bool,
    assignment_accuracy_gate: float,
    mismatch_repair_gate: float,
) -> MultivariateCyclePanel:
    accepted_by_case: list[tuple[AssignmentCycle, ...]] = []
    for case in signature_cases:
        accepted_by_case.append(
            tuple(
                signature.cycle
                for signature in case.signatures
                if score_cycle_signature(model, signature) >= model.cutoff
            )
        )
    rows = tuple(
        _materialize_case_with_selection(
            case.candidate,
            accepted=accepted,
            rectangle=rectangle,
            matched_pair_config=matched_pair_config,
        )
        for case, accepted in zip(
            signature_cases,
            accepted_by_case,
            strict=True,
        )
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
    signatures = tuple(
        signature for case in signature_cases for signature in case.signatures
    )
    accepted_flags = np.asarray(
        [score_cycle_signature(model, row) >= model.cutoff for row in signatures],
        dtype=np.bool_,
    )
    strict_flags = np.asarray(
        [row.strictly_correcting for row in signatures],
        dtype=np.bool_,
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
    return MultivariateCyclePanel(
        panel_role=panel_role,
        seed=seed,
        frozen_rectangle=rectangle,
        cases=rows,
        profile_summaries=summaries,
        case_count=len(rows),
        pair_count=pair_count,
        correct_assignment_count=correct_count,
        assignment_accuracy=correct_count / pair_count,
        signature_cycle_count=len(signatures),
        accepted_cycle_count=int(np.sum(accepted_flags)),
        accepted_strictly_correcting_cycle_count=int(
            np.sum(accepted_flags & strict_flags)
        ),
        rejected_strictly_correcting_cycle_count=int(
            np.sum((~accepted_flags) & strict_flags)
        ),
        accepted_unsafe_cycle_count=int(np.sum(accepted_flags & (~strict_flags))),
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
            and model.calibration_valid
            and len(summaries) == len(profile_specs)
            and all(summary.profile_gate_passed for summary in summaries)
        ),
    )


def evaluate_multivariate_cycle_signature(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    training_a_seed: int = TRAINING_A_SEED,
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
) -> MultivariateCycleSignatureResult:
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
        training_a_seed,
        development_b_seed,
        validation_a_seed,
        validation_b_seed,
        final_held_out_seed,
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "training, development, validation, and final seeds must differ"
        )
    if any(seed in FORBIDDEN_PRIOR_SEEDS for seed in seeds):
        raise ValueError("Phase-20/21 opened or reserved seeds must not be reused")
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
            TRAINING_A_SEED,
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
    training_cases = _signature_cases(
        _candidate_cases(
            _raw_panel(seed=training_a_seed, **common),
            selected_confidence,
        )
    )
    model = fit_cycle_signature_model(
        tuple(
            signature
            for case in training_cases
            for signature in case.signatures
        )
    )
    panel_common = {
        "model": model,
        "rectangle": FROZEN_PHASE18_RECTANGLE,
        "matched_pair_config": selected_matched,
        "profile_specs": selected_profiles,
        "full_protocol": full_protocol,
        "assignment_accuracy_gate": assignment_accuracy_gate,
        "mismatch_repair_gate": mismatch_repair_gate,
    }
    training_a = _materialize_panel(
        training_cases,
        panel_role="training_a",
        seed=training_a_seed,
        **panel_common,
    )
    development_b: MultivariateCyclePanel | None = None
    validation_a: MultivariateCyclePanel | None = None
    validation_b: MultivariateCyclePanel | None = None
    final_panel: MultivariateCyclePanel | None = None
    if training_a.panel_gate_passed:
        development_b = _materialize_panel(
            _signature_cases(
                _candidate_cases(
                    _raw_panel(seed=development_b_seed, **common),
                    selected_confidence,
                )
            ),
            panel_role="development_b",
            seed=development_b_seed,
            **panel_common,
        )
    development_passed = bool(
        development_b is not None and development_b.panel_gate_passed
    )
    if development_passed:
        validation_a = _materialize_panel(
            _signature_cases(
                _candidate_cases(
                    _raw_panel(seed=validation_a_seed, **common),
                    selected_confidence,
                )
            ),
            panel_role="validation_a",
            seed=validation_a_seed,
            **panel_common,
        )
        if validation_a.panel_gate_passed:
            validation_b = _materialize_panel(
                _signature_cases(
                    _candidate_cases(
                        _raw_panel(seed=validation_b_seed, **common),
                        selected_confidence,
                    )
                ),
                panel_role="validation_b",
                seed=validation_b_seed,
                **panel_common,
            )
        if validation_b is not None and validation_b.panel_gate_passed:
            final_panel = _materialize_panel(
                _signature_cases(
                    _candidate_cases(
                        _raw_panel(seed=final_held_out_seed, **common),
                        selected_confidence,
                    )
                ),
                panel_role="final_held_out",
                seed=final_held_out_seed,
                **panel_common,
            )
    supported = bool(final_panel is not None and final_panel.panel_gate_passed)
    return MultivariateCycleSignatureResult(
        artifact_schema="pftf_alpha_multivariate_cycle_signature_phase22/v1",
        role="synthetic_supervised_multivariate_cycle_signature_audit",
        information_boundary=(
            "route uses presented coordinates, row alignment, candidate cycle "
            "cost signatures, and a model frozen on training A; source truth "
            "and endpoints are training/evaluation-only"
        ),
        frozen_predecessor=(
            "phase21_scalar_cycle_gain_overlap_and_development_failure"
        ),
        training_a_seed=training_a_seed,
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
        log_ratio_floor=LOG_RATIO_FLOOR,
        log_ratio_limit=LOG_RATIO_LIMIT,
        model=model,
        training_a=training_a,
        development_b=development_b,
        development_screen_passed=development_passed,
        validation_a=validation_a,
        validation_b=validation_b,
        final_held_out=final_panel,
        phase22_supported=supported,
        multivariate_cycle_signature_synthetic_supported=supported,
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
    result = evaluate_multivariate_cycle_signature(
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
