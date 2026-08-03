"""Case-level matched-displacement guard transfer audit for Phase 23."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .local_surface_consensus import GeometryTopologyHarmEndpoint
from .matched_pair_consistency import MatchedPairConfig, MatchedPairEvidence
from .matched_pair_stress import (
    DEFAULT_STRESS_SPECS,
    MatchedPairStressConfig,
    MatchedPairStressProfile,
    MatchedPairStressRawCase,
    MatchedPairStressSpec,
    _raw_panel,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .sensor_stress import DEFAULT_POINT_COUNTS, DEFAULT_STRESSES, SensorStress
from .shared_trend_inference import SharedTrendConfig

TRAINING_A_SEED = 24400804
DEVELOPMENT_B_SEED = 24500804
VALIDATION_A_SEED = 24600804
VALIDATION_B_SEED = 24700804
FINAL_HELD_OUT_SEED = 24800804
FORBIDDEN_PRIOR_SEEDS = frozenset(
    {
        23300804,
        23400804,
        23500804,
        23600804,
        23700804,
        23800804,
        23900804,
        24000804,
        24100804,
        24200804,
        24300804,
    }
)

GUARD_PROFILE_SPECS = DEFAULT_STRESS_SPECS[:3]
TAIL_RATIO_FLOOR = 1.0e-6
RIDGE_PENALTY = 1.0
MINIMUM_FEATURE_SCALE = 1.0e-12

SIGNATURE_FEATURE_NAMES = (
    "log_retained_pair_count",
    "retained_pair_fraction",
    "log1p_median_standardized_displacement",
    "log1p_percentile95_standardized_displacement",
    "log1p_peak_standardized_displacement",
    "log1p_support_standardized_displacement",
    "log1p_peak_to_support_ratio",
    "log1p_peak_support_gap",
    "log1p_normalized_maximum_centered_displacement",
    "log1p_axis_scale_anisotropy",
    "log1p_normalized_median_axis_scale",
    "log1p_normalized_displacement_location",
)


@dataclass(frozen=True)
class MatchedGuardSignature:
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            name: value
            for name, value in zip(
                SIGNATURE_FEATURE_NAMES,
                self.values,
                strict=True,
            )
        }


@dataclass(frozen=True)
class MatchedGuardModel:
    feature_names: tuple[str, ...]
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    rejection_cutoff: float
    ridge_penalty: float
    calibration_valid: bool
    training_case_count: int
    training_harmful_case_count: int
    training_safe_case_count: int
    rejected_training_harmful_case_count: int
    retained_training_harmful_case_count: int
    rejected_training_safe_case_count: int
    retained_training_safe_case_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MatchedGuardCaseResult:
    profile: MatchedPairStressProfile
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    perturbation_seed: int
    repeat_transient_outlier_count: int
    repeat_transient_outlier_index_sha256: str
    retained_pair_count: int
    missing_pair_count: int
    rotation_degrees: float
    rotation_axis: tuple[float, float, float]
    presented_pair_map_sha256: str
    signature: MatchedGuardSignature
    model_score: float
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
        payload["signature"] = self.signature.to_dict()
        payload["unguarded_decision"] = self.unguarded_decision.value
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class MatchedGuardProfileSummary:
    profile: MatchedPairStressProfile
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    all_stress_unguarded_safe_accept_count: int
    all_stress_guarded_safe_accept_count: int
    profile_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        return payload


@dataclass(frozen=True)
class MatchedGuardPanel:
    panel_role: str
    seed: int
    cases: tuple[MatchedGuardCaseResult, ...]
    profile_summaries: tuple[MatchedGuardProfileSummary, ...]
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    all_stress_unguarded_safe_accept_count: int
    all_stress_guarded_safe_accept_count: int
    full_protocol: bool
    panel_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_role": self.panel_role,
            "seed": self.seed,
            "cases": [case.to_dict() for case in self.cases],
            "profile_summaries": [
                summary.to_dict() for summary in self.profile_summaries
            ],
            "case_count": self.case_count,
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
            "all_stress_unguarded_safe_accept_count": (
                self.all_stress_unguarded_safe_accept_count
            ),
            "all_stress_guarded_safe_accept_count": (
                self.all_stress_guarded_safe_accept_count
            ),
            "full_protocol": self.full_protocol,
            "panel_gate_passed": self.panel_gate_passed,
        }


@dataclass(frozen=True)
class MatchedGuardSignatureResult:
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
    profile_specs: tuple[MatchedPairStressSpec, ...]
    tail_ratio_floor: float
    model: MatchedGuardModel
    training_a: MatchedGuardPanel
    development_b: MatchedGuardPanel | None
    development_screen_passed: bool
    validation_a: MatchedGuardPanel | None
    validation_b: MatchedGuardPanel | None
    final_held_out: MatchedGuardPanel | None
    phase23_supported: bool
    matched_guard_signature_synthetic_supported: bool
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
            "profile_specs": [
                {**asdict(spec), "profile": spec.profile.value}
                for spec in self.profile_specs
            ],
            "tail_ratio_floor": self.tail_ratio_floor,
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
            "phase23_supported": self.phase23_supported,
            "matched_guard_signature_synthetic_supported": (
                self.matched_guard_signature_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def matched_guard_signature(
    raw: MatchedPairStressRawCase,
) -> MatchedGuardSignature:
    evidence = raw.evidence
    characteristic_length = max(
        evidence.observed_characteristic_length,
        np.finfo(float).eps,
    )
    axis_scales = np.asarray(evidence.axis_scales, dtype=np.float64)
    displacement_location = np.asarray(
        evidence.displacement_location,
        dtype=np.float64,
    )
    peak = evidence.peak_standardized_displacement
    support = evidence.support_standardized_displacement
    values = (
        math.log(float(raw.retained_pair_count)),
        raw.retained_pair_count / raw.point_count,
        math.log1p(evidence.median_standardized_displacement),
        math.log1p(evidence.percentile95_standardized_displacement),
        math.log1p(peak),
        math.log1p(support),
        math.log1p(peak / (support + TAIL_RATIO_FLOOR)),
        math.log1p(max(peak - support, 0.0)),
        math.log1p(evidence.maximum_centered_displacement / characteristic_length),
        math.log1p(float(np.max(axis_scales) / np.min(axis_scales))),
        math.log1p(float(np.median(axis_scales) / characteristic_length)),
        math.log1p(
            float(np.linalg.norm(displacement_location) / characteristic_length)
        ),
    )
    if len(values) != len(SIGNATURE_FEATURE_NAMES):
        raise RuntimeError("matched guard signature does not match its schema")
    if not np.all(np.isfinite(values)):
        raise ValueError("matched guard signature must be finite")
    return MatchedGuardSignature(values=values)


def score_matched_guard_signature(
    model: MatchedGuardModel,
    signature: MatchedGuardSignature,
) -> float:
    values = np.asarray(signature.values, dtype=np.float64)
    center = np.asarray(model.feature_center, dtype=np.float64)
    scale = np.asarray(model.feature_scale, dtype=np.float64)
    coefficients = np.asarray(model.coefficients, dtype=np.float64)
    if values.shape != center.shape or scale.shape != center.shape:
        raise ValueError("model and matched guard signature dimensions must agree")
    return float(model.intercept + ((values - center) / scale) @ coefficients)


def fit_matched_guard_model(
    signatures: Sequence[MatchedGuardSignature],
    harmful_labels: Sequence[bool],
) -> MatchedGuardModel:
    rows = tuple(signatures)
    labels = np.asarray(tuple(harmful_labels), dtype=np.bool_)
    if labels.shape != (len(rows),):
        raise ValueError("signatures and harmful labels must align")
    if not rows:
        dimension = len(SIGNATURE_FEATURE_NAMES)
        return MatchedGuardModel(
            feature_names=SIGNATURE_FEATURE_NAMES,
            feature_center=(0.0,) * dimension,
            feature_scale=(1.0,) * dimension,
            intercept=0.0,
            coefficients=(0.0,) * dimension,
            rejection_cutoff=-np.finfo(float).max,
            ridge_penalty=RIDGE_PENALTY,
            calibration_valid=False,
            training_case_count=0,
            training_harmful_case_count=0,
            training_safe_case_count=0,
            rejected_training_harmful_case_count=0,
            retained_training_harmful_case_count=0,
            rejected_training_safe_case_count=0,
            retained_training_safe_case_count=0,
        )
    features = np.asarray([row.values for row in rows], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(SIGNATURE_FEATURE_NAMES):
        raise ValueError("matched guard signature matrix has the wrong shape")
    if not np.all(np.isfinite(features)):
        raise ValueError("matched guard signature matrix must be finite")
    harmful_count = int(np.sum(labels))
    safe_count = len(rows) - harmful_count
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < MINIMUM_FEATURE_SCALE, 1.0, scale)
    standardized = (features - center) / scale
    valid = bool(harmful_count > 0 and safe_count > 0)
    if valid:
        design = np.column_stack((np.ones(len(rows)), standardized))
        penalty = np.diag(
            np.asarray((0.0,) + (RIDGE_PENALTY,) * features.shape[1])
        )
        targets = labels.astype(np.float64)
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
        minimum_harmful = float(np.min(scores[labels]))
        cutoff = float(np.nextafter(minimum_harmful, -math.inf))
    else:
        beta = np.zeros(features.shape[1] + 1, dtype=np.float64)
        scores = np.zeros(len(rows), dtype=np.float64)
        cutoff = -np.finfo(float).max
    rejected = scores >= cutoff
    rejected_harmful = int(np.sum(rejected & labels))
    rejected_safe = int(np.sum(rejected & (~labels)))
    return MatchedGuardModel(
        feature_names=SIGNATURE_FEATURE_NAMES,
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        intercept=float(beta[0]),
        coefficients=tuple(float(value) for value in beta[1:]),
        rejection_cutoff=cutoff,
        ridge_penalty=RIDGE_PENALTY,
        calibration_valid=valid,
        training_case_count=len(rows),
        training_harmful_case_count=harmful_count,
        training_safe_case_count=safe_count,
        rejected_training_harmful_case_count=rejected_harmful,
        retained_training_harmful_case_count=harmful_count - rejected_harmful,
        rejected_training_safe_case_count=rejected_safe,
        retained_training_safe_case_count=safe_count - rejected_safe,
    )


def _materialize_case(
    raw: MatchedPairStressRawCase,
    model: MatchedGuardModel,
) -> MatchedGuardCaseResult:
    signature = matched_guard_signature(raw)
    score = score_matched_guard_signature(model, signature)
    unguarded_accept = raw.unguarded_decision is SamplingGateDecision.ACCEPT
    consistent = bool(model.calibration_valid and score < model.rejection_cutoff)
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
    return MatchedGuardCaseResult(
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
        retained_pair_count=raw.retained_pair_count,
        missing_pair_count=raw.missing_pair_count,
        rotation_degrees=raw.rotation_degrees,
        rotation_axis=raw.rotation_axis,
        presented_pair_map_sha256=raw.presented_pair_map_sha256,
        signature=signature,
        model_score=score,
        evidence=raw.evidence,
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
    rows: Sequence[MatchedGuardCaseResult],
    profile: MatchedPairStressProfile,
    *,
    full_protocol: bool,
    model_valid: bool,
) -> MatchedGuardProfileSummary:
    selected = [row for row in rows if row.profile is profile]
    focus = [
        row
        for row in selected
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    retention = 0.0 if not unguarded_focus else guarded_focus / unguarded_focus
    unguarded_harm = sum(
        row.unguarded_harmful_outlier_false_safe for row in selected
    )
    guarded_harm = sum(
        row.guarded_harmful_outlier_false_safe for row in selected
    )
    all_safe = [row for row in selected if row.unguarded_safe_accept]
    return MatchedGuardProfileSummary(
        profile=profile,
        case_count=len(selected),
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
        focus_safe_accept_retention=retention,
        all_stress_unguarded_safe_accept_count=len(all_safe),
        all_stress_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in all_safe
        ),
        profile_gate_passed=bool(
            full_protocol
            and model_valid
            and unguarded_harm > 0
            and guarded_harm == 0
            and retention >= 0.90
        ),
    )


def _materialize_panel(
    raw_rows: Sequence[MatchedPairStressRawCase],
    *,
    panel_role: str,
    seed: int,
    model: MatchedGuardModel,
    profile_specs: tuple[MatchedPairStressSpec, ...],
    full_protocol: bool,
) -> MatchedGuardPanel:
    rows = tuple(_materialize_case(raw, model) for raw in raw_rows)
    summaries = tuple(
        _profile_summary(
            rows,
            spec.profile,
            full_protocol=full_protocol,
            model_valid=model.calibration_valid,
        )
        for spec in profile_specs
    )
    focus = [
        row
        for row in rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    all_safe = [row for row in rows if row.unguarded_safe_accept]
    return MatchedGuardPanel(
        panel_role=panel_role,
        seed=seed,
        cases=rows,
        profile_summaries=summaries,
        case_count=len(rows),
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
        all_stress_unguarded_safe_accept_count=len(all_safe),
        all_stress_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in all_safe
        ),
        full_protocol=full_protocol,
        panel_gate_passed=bool(
            full_protocol
            and model.calibration_valid
            and len(summaries) == len(profile_specs)
            and all(summary.profile_gate_passed for summary in summaries)
        ),
    )


def evaluate_matched_guard_signature(
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
    profile_specs: Sequence[MatchedPairStressSpec] = GUARD_PROFILE_SPECS,
) -> MatchedGuardSignatureResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_matched = (
        MatchedPairConfig() if matched_pair_config is None else matched_pair_config
    )
    selected_stress = (
        MatchedPairStressConfig() if stress_config is None else stress_config
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
        raise ValueError("Phase-20--22 opened or reserved seeds must not be reused")
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
        and selected_profiles == GUARD_PROFILE_SPECS
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
        "profile_specs": selected_profiles,
    }
    training_raw = _raw_panel(seed=training_a_seed, **common)
    calibration_rows = tuple(
        row
        for row in training_raw
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
    )
    model = fit_matched_guard_model(
        tuple(matched_guard_signature(row) for row in calibration_rows),
        tuple(row.endpoint.geometry_topology_harm_present for row in calibration_rows),
    )
    panel_common = {
        "model": model,
        "profile_specs": selected_profiles,
        "full_protocol": full_protocol,
    }
    training_a = _materialize_panel(
        training_raw,
        panel_role="training_a",
        seed=training_a_seed,
        **panel_common,
    )
    development_b: MatchedGuardPanel | None = None
    validation_a: MatchedGuardPanel | None = None
    validation_b: MatchedGuardPanel | None = None
    final_panel: MatchedGuardPanel | None = None
    if training_a.panel_gate_passed:
        development_b = _materialize_panel(
            _raw_panel(seed=development_b_seed, **common),
            panel_role="development_b",
            seed=development_b_seed,
            **panel_common,
        )
    development_passed = bool(
        development_b is not None and development_b.panel_gate_passed
    )
    if development_passed:
        validation_a = _materialize_panel(
            _raw_panel(seed=validation_a_seed, **common),
            panel_role="validation_a",
            seed=validation_a_seed,
            **panel_common,
        )
        if validation_a.panel_gate_passed:
            validation_b = _materialize_panel(
                _raw_panel(seed=validation_b_seed, **common),
                panel_role="validation_b",
                seed=validation_b_seed,
                **panel_common,
            )
        if validation_b is not None and validation_b.panel_gate_passed:
            final_panel = _materialize_panel(
                _raw_panel(seed=final_held_out_seed, **common),
                panel_role="final_held_out",
                seed=final_held_out_seed,
                **panel_common,
            )
    supported = bool(final_panel is not None and final_panel.panel_gate_passed)
    return MatchedGuardSignatureResult(
        artifact_schema="pftf_alpha_matched_guard_signature_phase23/v1",
        role="synthetic_supervised_matched_guard_transfer_audit",
        information_boundary=(
            "route uses exact presented pairs, observed displacement-tail "
            "signature, upstream candidate decision, and a model frozen on "
            "training A; endpoints are training/evaluation-only"
        ),
        frozen_predecessor=(
            "phase22_exact_pair_missing_profile_guard_transfer_failure"
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
        profile_specs=selected_profiles,
        tail_ratio_floor=TAIL_RATIO_FLOOR,
        model=model,
        training_a=training_a,
        development_b=development_b,
        development_screen_passed=development_passed,
        validation_a=validation_a,
        validation_b=validation_b,
        final_held_out=final_panel,
        phase23_supported=supported,
        matched_guard_signature_synthetic_supported=supported,
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
    result = evaluate_matched_guard_signature(
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
