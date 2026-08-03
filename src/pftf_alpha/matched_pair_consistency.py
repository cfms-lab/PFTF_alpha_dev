"""Exact-correspondence matched-repeat consistency and Phase-17 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .conservative_influence_calibration import (
    InfluenceFeatureCohort,
    calibrate_dual_cohort_rectangle,
)
from .local_insertion_influence import InfluenceRectangle
from .local_surface_consensus import (
    GeometryTopologyHarmEndpoint,
    LocalSurfaceConsensusConfig,
    evaluate_geometry_topology_harm,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .sensor_stress import (
    DEFAULT_POINT_COUNTS,
    DEFAULT_STRESSES,
    SensorStress,
    evaluate_sensor_stress,
    make_sensor_stress_case,
    sensor_surface_height,
)
from .shared_trend_inference import (
    SharedTrendConfig,
    construct_shared_trend_surface,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CALIBRATION_A_SEED = 22400804
CALIBRATION_B_SEED = 22500804
FINAL_HELD_OUT_SEED = 22600804
REPLICATE_SEED_OFFSET = 500000009


@dataclass(frozen=True)
class MatchedPairConfig:
    """Frozen exact-correspondence score and simulator controls."""

    mad_consistency_factor: float = 1.4826
    minimum_axis_scale_fraction: float = 0.002
    harmful_distance_fraction: float = 0.025
    replicate_seed_offset: int = REPLICATE_SEED_OFFSET

    def __post_init__(self) -> None:
        if self.replicate_seed_offset <= 0:
            raise ValueError("replicate_seed_offset must be positive")
        for name, value in (
            ("mad_consistency_factor", self.mad_consistency_factor),
            ("minimum_axis_scale_fraction", self.minimum_axis_scale_fraction),
            ("harmful_distance_fraction", self.harmful_distance_fraction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MatchedRepeatObservation:
    points: FloatArray
    transient_outlier_indices: IntArray

    @property
    def transient_outlier_index_sha256(self) -> str:
        return hashlib.sha256(
            np.asarray(self.transient_outlier_indices, dtype="<i8").tobytes()
        ).hexdigest()


@dataclass(frozen=True)
class MatchedDisplacementScores:
    displacements: FloatArray
    displacement_location: FloatArray
    axis_scales: FloatArray
    point_scores: FloatArray
    observed_characteristic_length: float


@dataclass(frozen=True)
class MatchedPairEvidence:
    information_boundary: str
    primary_point_count: int
    repeat_point_count: int
    observed_characteristic_length: float
    displacement_location: tuple[float, float, float]
    axis_scales: tuple[float, float, float]
    median_standardized_displacement: float
    percentile95_standardized_displacement: float
    peak_standardized_displacement: float
    support_standardized_displacement: float
    leading_standardized_displacements: tuple[float, ...]
    maximum_centered_displacement: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MatchedRawCase:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    repeat_transient_outlier_count: int
    repeat_transient_outlier_index_sha256: str
    evidence: MatchedPairEvidence
    endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision


@dataclass(frozen=True)
class MatchedCaseResult:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    repeat_transient_outlier_count: int
    repeat_transient_outlier_index_sha256: str
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
        payload["stress"] = self.stress.value
        payload["unguarded_decision"] = self.unguarded_decision.value
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class MatchedPanelResult:
    panel_role: str
    seed: int
    rectangle: InfluenceRectangle | None
    cases: tuple[MatchedCaseResult, ...]
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int
    clean_local_bump_unguarded_safe_accept_count: int
    clean_local_bump_guarded_safe_accept_count: int
    clean_local_bump_safe_accept_retention: float
    all_stress_unguarded_safe_accept_count: int
    all_stress_guarded_safe_accept_count: int
    full_protocol: bool
    panel_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_role": self.panel_role,
            "seed": self.seed,
            "rectangle": None if self.rectangle is None else self.rectangle.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "case_count": self.case_count,
            "unguarded_harmful_outlier_false_safe_count": (
                self.unguarded_harmful_outlier_false_safe_count
            ),
            "guarded_harmful_outlier_false_safe_count": (
                self.guarded_harmful_outlier_false_safe_count
            ),
            "unguarded_provenance_violation_accept_count": (
                self.unguarded_provenance_violation_accept_count
            ),
            "guarded_provenance_violation_accept_count": (
                self.guarded_provenance_violation_accept_count
            ),
            "clean_local_bump_unguarded_safe_accept_count": (
                self.clean_local_bump_unguarded_safe_accept_count
            ),
            "clean_local_bump_guarded_safe_accept_count": (
                self.clean_local_bump_guarded_safe_accept_count
            ),
            "clean_local_bump_safe_accept_retention": (
                self.clean_local_bump_safe_accept_retention
            ),
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
class MatchedPairResult:
    artifact_schema: str
    role: str
    information_boundary: str
    simulator_boundary: str
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
    rectangle_selection_rule: str
    selected_rectangle: InfluenceRectangle | None
    calibration_a: MatchedPanelResult
    calibration_b: MatchedPanelResult
    final_held_out: MatchedPanelResult | None
    phase17_supported: bool
    exact_correspondence_synthetic_supported: bool
    real_correspondence_supported: bool
    real_paired_scan_supported: bool
    trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "simulator_boundary": self.simulator_boundary,
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
            "rectangle_selection_rule": self.rectangle_selection_rule,
            "selected_rectangle": (
                None
                if self.selected_rectangle is None
                else self.selected_rectangle.to_dict()
            ),
            "calibration_a": self.calibration_a.to_dict(),
            "calibration_b": self.calibration_b.to_dict(),
            "final_held_out": (
                None
                if self.final_held_out is None
                else self.final_held_out.to_dict()
            ),
            "phase17_supported": self.phase17_supported,
            "exact_correspondence_synthetic_supported": (
                self.exact_correspondence_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def make_matched_repeat_observation(
    primary_points: FloatArray,
    primary_source_labels: IntArray,
    stress: SensorStress | str,
    *,
    seed: int,
) -> MatchedRepeatObservation:
    """Generate one declared matched repeat; labels are simulator-only inputs."""

    primary = np.asarray(primary_points, dtype=np.float64)
    source_labels = np.asarray(primary_source_labels, dtype=np.int64)
    selected_stress = SensorStress(stress)
    if primary.ndim != 2 or primary.shape[1] != 3:
        raise ValueError("primary_points must have shape (n, 3)")
    if source_labels.shape != (primary.shape[0],):
        raise ValueError("primary_source_labels must align with primary points")
    if not set(np.unique(source_labels)).issubset({0, 1, 2}):
        raise ValueError("primary_source_labels must contain only zero, one, or two")
    if not np.all(np.isfinite(primary)):
        raise ValueError("primary_points must be finite")

    xy = primary[:, :2]
    lower = sensor_surface_height(selected_stress, xy, 0)
    upper = sensor_surface_height(selected_stress, xy, 1)
    surface_height = np.where(source_labels == 0, lower, upper)
    injected = source_labels == 2
    if np.any(injected):
        nearest_is_lower = (
            np.abs(primary[injected, 2] - lower[injected])
            <= np.abs(primary[injected, 2] - upper[injected])
        )
        surface_height[injected] = np.where(
            nearest_is_lower,
            lower[injected],
            upper[injected],
        )

    latent = np.column_stack((xy, surface_height))
    rng = np.random.default_rng(seed)
    if selected_stress is SensorStress.ANISOTROPIC_NOISE:
        noise_scales = np.asarray((0.006, 0.006, 0.040))
    else:
        noise_scales = np.asarray((0.010, 0.010, 0.010))
    repeat = latent + rng.normal(size=latent.shape) * noise_scales
    outlier_count = int(round(primary.shape[0] * selected_stress.outlier_fraction))
    if outlier_count:
        outlier_indices = np.sort(
            rng.choice(primary.shape[0], size=outlier_count, replace=False)
        ).astype(np.int64)
        repeat[outlier_indices, 2] = rng.uniform(-0.65, 0.95, size=outlier_count)
    else:
        outlier_indices = np.empty(0, dtype=np.int64)
    return MatchedRepeatObservation(
        points=repeat,
        transient_outlier_indices=outlier_indices,
    )


def matched_displacement_scores(
    primary_points: FloatArray,
    repeat_points: FloatArray,
    config: MatchedPairConfig | None = None,
) -> MatchedDisplacementScores:
    """Score aligned coordinate pairs without source or stress labels."""

    selected = MatchedPairConfig() if config is None else config
    primary = np.asarray(primary_points, dtype=np.float64)
    repeat = np.asarray(repeat_points, dtype=np.float64)
    if primary.ndim != 2 or primary.shape[1] != 3:
        raise ValueError("primary_points must have shape (n, 3)")
    if repeat.shape != primary.shape:
        raise ValueError("repeat_points must preserve exact one-to-one shape")
    if primary.shape[0] < 2:
        raise ValueError("at least two matched pairs are required")
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(repeat)):
        raise ValueError("matched points must be finite")

    pooled = np.vstack((primary, repeat))
    observed_characteristic_length = max(
        float(np.linalg.norm(np.ptp(pooled, axis=0))),
        np.finfo(float).eps,
    )
    displacements = primary - repeat
    location = np.median(displacements, axis=0)
    centered = displacements - location
    robust_scales = selected.mad_consistency_factor * np.median(
        np.abs(centered),
        axis=0,
    )
    minimum_scale = (
        selected.minimum_axis_scale_fraction * observed_characteristic_length
    )
    axis_scales = np.maximum(
        robust_scales,
        max(minimum_scale, np.finfo(float).eps),
    )
    point_scores = np.linalg.norm(centered / axis_scales, axis=1)
    return MatchedDisplacementScores(
        displacements=displacements,
        displacement_location=location,
        axis_scales=axis_scales,
        point_scores=point_scores,
        observed_characteristic_length=observed_characteristic_length,
    )


def estimate_matched_pair_evidence(
    primary_points: FloatArray,
    repeat_points: FloatArray,
    config: MatchedPairConfig | None = None,
) -> MatchedPairEvidence:
    scores = matched_displacement_scores(primary_points, repeat_points, config)
    descending = np.sort(scores.point_scores)[::-1]
    centered = scores.displacements - scores.displacement_location
    return MatchedPairEvidence(
        information_boundary=(
            "ordered_primary_and_repeat_coordinates_with_externally_asserted_"
            "exact_pair_identity_only"
        ),
        primary_point_count=int(np.asarray(primary_points).shape[0]),
        repeat_point_count=int(np.asarray(repeat_points).shape[0]),
        observed_characteristic_length=scores.observed_characteristic_length,
        displacement_location=tuple(
            float(value) for value in scores.displacement_location
        ),
        axis_scales=tuple(float(value) for value in scores.axis_scales),
        median_standardized_displacement=float(np.median(scores.point_scores)),
        percentile95_standardized_displacement=float(
            np.percentile(scores.point_scores, 95.0)
        ),
        peak_standardized_displacement=float(descending[0]),
        support_standardized_displacement=float(descending[1]),
        leading_standardized_displacements=tuple(
            float(value) for value in descending[:4]
        ),
        maximum_centered_displacement=float(
            np.max(np.linalg.norm(centered, axis=1))
        ),
    )


def _raw_panel(
    *,
    point_counts: tuple[int, ...],
    stresses: tuple[SensorStress, ...],
    reference_count: int,
    repeats: int,
    seed: int,
    surface_sample_count: int,
    base_gate_config: SamplingSufficiencyConfig | None,
    shared_trend_config: SharedTrendConfig | None,
    matched_pair_config: MatchedPairConfig,
) -> tuple[MatchedRawCase, ...]:
    base_result = evaluate_sensor_stress(
        point_counts=point_counts,
        stresses=stresses,
        reference_count=reference_count,
        repeats=repeats,
        seed=seed,
        surface_sample_count=surface_sample_count,
        base_gate_config=base_gate_config,
        shared_trend_config=shared_trend_config,
    )
    harm_config = LocalSurfaceConsensusConfig(
        harmful_distance_fraction=matched_pair_config.harmful_distance_fraction
    )
    rows: list[MatchedRawCase] = []
    for case_row in base_result.cases:
        primary = make_sensor_stress_case(
            case_row.stress,
            case_row.point_count,
            reference_count=max(reference_count, case_row.point_count),
            seed=case_row.seed,
        )
        primary_construction, _ = construct_shared_trend_surface(
            primary.points,
            shared_trend_config,
        )
        replicate_seed = case_row.seed + matched_pair_config.replicate_seed_offset
        matched_repeat = make_matched_repeat_observation(
            primary.points,
            primary.point_component_labels,
            case_row.stress,
            seed=replicate_seed,
        )
        evidence = estimate_matched_pair_evidence(
            primary.points,
            matched_repeat.points,
            matched_pair_config,
        )
        endpoint = evaluate_geometry_topology_harm(
            primary_construction.mesh,
            primary.reference_points,
            primary.point_component_labels,
            characteristic_length=primary.characteristic_length,
            config=harm_config,
        )
        rows.append(
            MatchedRawCase(
                stress=case_row.stress,
                point_count=case_row.point_count,
                repeat=case_row.repeat,
                seed=case_row.seed,
                replicate_seed=replicate_seed,
                repeat_transient_outlier_count=int(
                    matched_repeat.transient_outlier_indices.size
                ),
                repeat_transient_outlier_index_sha256=(
                    matched_repeat.transient_outlier_index_sha256
                ),
                evidence=evidence,
                endpoint=endpoint,
                unguarded_decision=case_row.candidate_decision,
            )
        )
    return tuple(rows)


def _features(row: MatchedRawCase) -> tuple[float, float]:
    return (
        row.evidence.peak_standardized_displacement,
        row.evidence.support_standardized_displacement,
    )


def _feature_cohort(rows: tuple[MatchedRawCase, ...]) -> InfluenceFeatureCohort:
    harmful = tuple(
        _features(row)
        for row in rows
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
        and row.stress.is_outlier_stress
        and row.endpoint.geometry_topology_harm_present
    )
    safe_rows = tuple(
        row
        for row in rows
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
        and not row.endpoint.geometry_topology_harm_present
    )
    focus_safe = tuple(
        _features(row)
        for row in safe_rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    )
    return InfluenceFeatureCohort(
        harmful=harmful,
        focus_safe=focus_safe,
        all_safe=tuple(_features(row) for row in safe_rows),
    )


def _materialize_panel(
    raw_rows: tuple[MatchedRawCase, ...],
    *,
    panel_role: str,
    seed: int,
    rectangle: InfluenceRectangle | None,
    full_protocol: bool,
) -> MatchedPanelResult:
    rows: list[MatchedCaseResult] = []
    for raw in raw_rows:
        unguarded_accept = raw.unguarded_decision is SamplingGateDecision.ACCEPT
        consistent = bool(
            rectangle is not None
            and raw.evidence.peak_standardized_displacement
            <= rectangle.peak_threshold
            and raw.evidence.support_standardized_displacement
            <= rectangle.support_threshold
        )
        if unguarded_accept and not consistent:
            guarded_decision = SamplingGateDecision.UNSUPPORTED
        else:
            guarded_decision = raw.unguarded_decision
        guarded_accept = guarded_decision is SamplingGateDecision.ACCEPT
        harmful_outlier = bool(
            raw.stress.is_outlier_stress
            and raw.endpoint.geometry_topology_harm_present
        )
        rows.append(
            MatchedCaseResult(
                stress=raw.stress,
                point_count=raw.point_count,
                repeat=raw.repeat,
                seed=raw.seed,
                replicate_seed=raw.replicate_seed,
                repeat_transient_outlier_count=raw.repeat_transient_outlier_count,
                repeat_transient_outlier_index_sha256=(
                    raw.repeat_transient_outlier_index_sha256
                ),
                evidence=raw.evidence,
                endpoint=raw.endpoint,
                unguarded_decision=raw.unguarded_decision,
                guarded_decision=guarded_decision,
                unguarded_safe_accept=bool(
                    unguarded_accept
                    and not raw.endpoint.geometry_topology_harm_present
                ),
                guarded_safe_accept=bool(
                    guarded_accept
                    and not raw.endpoint.geometry_topology_harm_present
                ),
                unguarded_harmful_outlier_false_safe=bool(
                    unguarded_accept and harmful_outlier
                ),
                guarded_harmful_outlier_false_safe=bool(
                    guarded_accept and harmful_outlier
                ),
                unguarded_provenance_violation_accept=bool(
                    unguarded_accept
                    and raw.endpoint.provenance_violation_present
                ),
                guarded_provenance_violation_accept=bool(
                    guarded_accept
                    and raw.endpoint.provenance_violation_present
                ),
            )
        )
    focus = [
        case
        for case in rows
        if case.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(case.unguarded_safe_accept for case in focus)
    guarded_focus = sum(case.guarded_safe_accept for case in focus)
    retention = 0.0 if not unguarded_focus else guarded_focus / unguarded_focus
    unguarded_harmful = sum(
        case.unguarded_harmful_outlier_false_safe for case in rows
    )
    guarded_harmful = sum(
        case.guarded_harmful_outlier_false_safe for case in rows
    )
    all_safe = [case for case in rows if case.unguarded_safe_accept]
    panel_passed = bool(
        full_protocol
        and rectangle is not None
        and unguarded_harmful > 0
        and guarded_harmful == 0
        and retention >= 0.90
    )
    return MatchedPanelResult(
        panel_role=panel_role,
        seed=seed,
        rectangle=rectangle,
        cases=tuple(rows),
        case_count=len(rows),
        unguarded_harmful_outlier_false_safe_count=unguarded_harmful,
        guarded_harmful_outlier_false_safe_count=guarded_harmful,
        unguarded_provenance_violation_accept_count=sum(
            case.unguarded_provenance_violation_accept for case in rows
        ),
        guarded_provenance_violation_accept_count=sum(
            case.guarded_provenance_violation_accept for case in rows
        ),
        clean_local_bump_unguarded_safe_accept_count=unguarded_focus,
        clean_local_bump_guarded_safe_accept_count=guarded_focus,
        clean_local_bump_safe_accept_retention=retention,
        all_stress_unguarded_safe_accept_count=len(all_safe),
        all_stress_guarded_safe_accept_count=sum(
            case.guarded_safe_accept for case in all_safe
        ),
        full_protocol=full_protocol,
        panel_gate_passed=panel_passed,
    )


def evaluate_matched_pair_consistency(
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
) -> MatchedPairResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_matched = (
        MatchedPairConfig() if matched_pair_config is None else matched_pair_config
    )
    seeds = (calibration_a_seed, calibration_b_seed, final_held_out_seed)
    if repeats < 1 or not selected_counts or not selected_stresses:
        raise ValueError("counts/stresses must be non-empty and repeats positive")
    if len(set(seeds)) != 3:
        raise ValueError("all calibration and held-out seeds must differ")
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and repeats == 8
        and reference_count == 2048
        and surface_sample_count == 256
        and seeds
        == (CALIBRATION_A_SEED, CALIBRATION_B_SEED, FINAL_HELD_OUT_SEED)
        and selected_matched == MatchedPairConfig()
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
    }
    calibration_a_raw = _raw_panel(seed=calibration_a_seed, **common)
    calibration_b_raw = _raw_panel(seed=calibration_b_seed, **common)
    rectangle = calibrate_dual_cohort_rectangle(
        _feature_cohort(calibration_a_raw),
        _feature_cohort(calibration_b_raw),
    )
    calibration_a = _materialize_panel(
        calibration_a_raw,
        panel_role="calibration_a",
        seed=calibration_a_seed,
        rectangle=rectangle,
        full_protocol=full_protocol,
    )
    calibration_b = _materialize_panel(
        calibration_b_raw,
        panel_role="calibration_b",
        seed=calibration_b_seed,
        rectangle=rectangle,
        full_protocol=full_protocol,
    )
    calibration_passed = bool(
        calibration_a.panel_gate_passed and calibration_b.panel_gate_passed
    )
    final_held_out: MatchedPanelResult | None = None
    if calibration_passed:
        final_raw = _raw_panel(seed=final_held_out_seed, **common)
        final_held_out = _materialize_panel(
            final_raw,
            panel_role="final_held_out",
            seed=final_held_out_seed,
            rectangle=rectangle,
            full_protocol=full_protocol,
        )
    supported = bool(
        calibration_passed
        and final_held_out is not None
        and final_held_out.panel_gate_passed
    )
    return MatchedPairResult(
        artifact_schema="pftf_alpha_matched_pair_consistency_phase17/v1",
        role="exact_correspondence_synthetic_upper_bound_guard",
        information_boundary=(
            "route uses ordered primary/repeat coordinates and externally asserted "
            "exact pair identity only; source labels, stress, analytic surfaces, "
            "and references are hidden"
        ),
        simulator_boundary=(
            "synthetic matched returns use evaluation-only source/stress truth to "
            "construct independently noisy same-location depth observations"
        ),
        frozen_predecessor=(
            "phase16_calibrations_22100804_22200804_studentization_negative"
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
        rectangle_selection_rule=(
            "zero harm in both calibration cohorts; maximize worst focus "
            "retention, total focus count, worst all-safe retention, total "
            "all-safe count, then peak and support thresholds"
        ),
        selected_rectangle=rectangle,
        calibration_a=calibration_a,
        calibration_b=calibration_b,
        final_held_out=final_held_out,
        phase17_supported=supported,
        exact_correspondence_synthetic_supported=supported,
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
    parser.add_argument(
        "--final-held-out-seed",
        type=int,
        default=FINAL_HELD_OUT_SEED,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_matched_pair_consistency(
        reference_count=args.reference,
        repeats=args.repeats,
        calibration_a_seed=args.calibration_a_seed,
        calibration_b_seed=args.calibration_b_seed,
        final_held_out_seed=args.final_held_out_seed,
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
