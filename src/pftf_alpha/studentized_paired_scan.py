"""Studentized paired-scan persistence and frozen Phase-16 audit."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

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
)
from .shared_trend_inference import (
    SharedTrendConfig,
    construct_shared_trend_surface,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CALIBRATION_A_SEED = 22100804
CALIBRATION_B_SEED = 22200804
FINAL_HELD_OUT_SEED = 22300804
REPLICATE_SEED_OFFSET = 400000009


@dataclass(frozen=True)
class StudentizedPairedConfig:
    """Frozen replicate-derived predictive uncertainty for Phase 16."""

    neighbor_counts: tuple[int, ...] = (12, 18, 24)
    mad_consistency_factor: float = 1.4826
    loo_quantile: float = 0.90
    minimum_scale_fraction: float = 0.04
    harmful_distance_fraction: float = 0.025
    replicate_seed_offset: int = REPLICATE_SEED_OFFSET

    def __post_init__(self) -> None:
        if not self.neighbor_counts:
            raise ValueError("neighbor_counts must be non-empty")
        if tuple(sorted(set(self.neighbor_counts))) != self.neighbor_counts:
            raise ValueError("neighbor_counts must be strictly increasing")
        if self.neighbor_counts[0] < 7:
            raise ValueError("quadratic fits require at least seven neighbors")
        if not 0.0 < self.loo_quantile < 1.0:
            raise ValueError("loo_quantile must lie strictly between zero and one")
        if self.replicate_seed_offset <= 0:
            raise ValueError("replicate_seed_offset must be positive")
        for name, value in (
            ("mad_consistency_factor", self.mad_consistency_factor),
            ("minimum_scale_fraction", self.minimum_scale_fraction),
            ("harmful_distance_fraction", self.harmful_distance_fraction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class StudentizedPairedScores:
    residuals_by_scale: FloatArray
    predictive_scales_by_scale: FloatArray
    query_leverages_by_scale: FloatArray
    standardized_residuals_by_scale: FloatArray
    best_residuals: FloatArray
    best_predictive_scales: FloatArray
    best_standardized_residuals: FloatArray
    selected_neighbor_counts: IntArray
    selected_query_leverages: FloatArray
    primary_to_replicate_layer_mapping: tuple[int, int]


@dataclass(frozen=True)
class StudentizedPairedEvidence:
    information_boundary: str
    primary_point_count: int
    replicate_point_count: int
    requested_neighbor_counts: tuple[int, ...]
    loo_quantile: float
    primary_to_replicate_layer_mapping: tuple[int, int]
    median_standardized_residual: float
    percentile95_standardized_residual: float
    peak_standardized_residual: float
    support_standardized_residual: float
    leading_standardized_residuals: tuple[float, ...]
    maximum_raw_residual: float
    maximum_selected_query_leverage: float
    selected_neighbor_count_histogram: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StudentizedRawCase:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    evidence: StudentizedPairedEvidence
    endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision


@dataclass(frozen=True)
class StudentizedCaseResult:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    evidence: StudentizedPairedEvidence
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
class StudentizedPanelResult:
    panel_role: str
    seed: int
    rectangle: InfluenceRectangle | None
    cases: tuple[StudentizedCaseResult, ...]
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
class StudentizedPairedResult:
    artifact_schema: str
    role: str
    information_boundary: str
    development_predecessor: str
    calibration_a_seed: int
    calibration_b_seed: int
    final_held_out_seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    studentized_config: StudentizedPairedConfig
    rectangle_selection_rule: str
    selected_rectangle: InfluenceRectangle | None
    calibration_a: StudentizedPanelResult
    calibration_b: StudentizedPanelResult
    final_held_out: StudentizedPanelResult | None
    phase16_supported: bool
    paired_synthetic_supported: bool
    real_paired_scan_supported: bool
    trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "development_predecessor": self.development_predecessor,
            "calibration_a_seed": self.calibration_a_seed,
            "calibration_b_seed": self.calibration_b_seed,
            "final_held_out_seed": self.final_held_out_seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "studentized_config": asdict(self.studentized_config),
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
            "phase16_supported": self.phase16_supported,
            "paired_synthetic_supported": self.paired_synthetic_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def _quadratic_design(coordinates: FloatArray) -> FloatArray:
    first = coordinates[:, 0]
    second = coordinates[:, 1]
    return np.column_stack(
        (
            np.ones(coordinates.shape[0]),
            first,
            second,
            first * first,
            first * second,
            second * second,
        )
    )


def _layer_mapping(
    primary_points: FloatArray,
    primary_labels: IntArray,
    replicate_points: FloatArray,
    replicate_labels: IntArray,
) -> tuple[int, int]:
    primary_centroids = tuple(
        np.mean(primary_points[primary_labels == layer], axis=0) for layer in (0, 1)
    )
    replicate_centroids = tuple(
        np.mean(replicate_points[replicate_labels == layer], axis=0)
        for layer in (0, 1)
    )
    identity_cost = sum(
        np.linalg.norm(primary_centroids[layer] - replicate_centroids[layer])
        for layer in (0, 1)
    )
    swap_cost = sum(
        np.linalg.norm(primary_centroids[layer] - replicate_centroids[1 - layer])
        for layer in (0, 1)
    )
    return (0, 1) if identity_cost <= swap_cost else (1, 0)


def studentized_paired_scores(
    primary_points: FloatArray,
    primary_labels: IntArray,
    replicate_points: FloatArray,
    replicate_labels: IntArray,
    config: StudentizedPairedConfig | None = None,
) -> StudentizedPairedScores:
    """Studentize primary-to-replicate residuals by predictive uncertainty."""

    selected = StudentizedPairedConfig() if config is None else config
    primary = np.asarray(primary_points, dtype=np.float64)
    replicate = np.asarray(replicate_points, dtype=np.float64)
    primary_layer_ids = np.asarray(primary_labels, dtype=np.int64)
    replicate_layer_ids = np.asarray(replicate_labels, dtype=np.int64)
    for name, points, labels in (
        ("primary", primary, primary_layer_ids),
        ("replicate", replicate, replicate_layer_ids),
    ):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{name}_points must have shape (n, 3)")
        if labels.shape != (points.shape[0],) or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{name}_labels must contain two aligned layers")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{name}_points must be finite")

    mapping = _layer_mapping(
        primary,
        primary_layer_ids,
        replicate,
        replicate_layer_ids,
    )
    shape = (primary.shape[0], len(selected.neighbor_counts))
    residuals = np.full(shape, np.nan, dtype=np.float64)
    predictive_scales = np.full(shape, np.nan, dtype=np.float64)
    query_leverages = np.full(shape, np.nan, dtype=np.float64)
    actual_counts = np.zeros(shape, dtype=np.int64)
    epsilon = np.finfo(float).eps

    for primary_layer in (0, 1):
        primary_indices = np.flatnonzero(primary_layer_ids == primary_layer)
        replicate_layer = mapping[primary_layer]
        repeat_layer_points = replicate[replicate_layer_ids == replicate_layer]
        if repeat_layer_points.shape[0] < 7:
            raise ValueError("each replicate layer requires at least seven points")
        available_counts = tuple(
            min(count, repeat_layer_points.shape[0])
            for count in selected.neighbor_counts
        )
        neighbor_rows = cKDTree(repeat_layer_points).query(
            primary[primary_indices],
            k=max(available_counts),
            workers=1,
        )[1]
        if neighbor_rows.ndim == 1:
            neighbor_rows = neighbor_rows[:, None]
        for local_index, ordered_neighbors in enumerate(neighbor_rows):
            global_index = int(primary_indices[local_index])
            point = primary[global_index]
            for scale_index, neighbor_count in enumerate(available_counts):
                neighbors = repeat_layer_points[ordered_neighbors[:neighbor_count]]
                center = np.mean(neighbors, axis=0)
                centered = neighbors - center
                eigenvectors = np.linalg.eigh(centered.T @ centered)[1]
                normal = eigenvectors[:, 0]
                tangent_basis = eigenvectors[:, 1:]
                tangent = centered @ tangent_basis
                heights = centered @ normal
                tangent_radius = max(
                    float(np.median(np.linalg.norm(tangent, axis=1))),
                    epsilon,
                )
                design = _quadratic_design(tangent / tangent_radius)
                gram_inverse = np.linalg.pinv(design.T @ design)
                coefficients = np.linalg.lstsq(design, heights, rcond=None)[0]
                fit_residuals = heights - design @ coefficients
                residual_center = float(np.median(fit_residuals))
                robust_scale = selected.mad_consistency_factor * float(
                    np.median(np.abs(fit_residuals - residual_center))
                )
                hat_diagonal = np.sum((design @ gram_inverse) * design, axis=1)
                loo_denominator = np.maximum(1.0 - hat_diagonal, epsilon)
                loo_errors = np.abs(fit_residuals / loo_denominator)
                loo_scale = float(
                    np.quantile(loo_errors, selected.loo_quantile)
                )
                base_scale = max(
                    robust_scale,
                    loo_scale,
                    selected.minimum_scale_fraction * tangent_radius,
                    epsilon,
                )
                point_centered = point - center
                point_tangent = (
                    point_centered @ tangent_basis / tangent_radius
                ).reshape(1, 2)
                point_design = _quadratic_design(point_tangent)[0]
                query_leverage = max(
                    float(point_design @ gram_inverse @ point_design),
                    0.0,
                )
                predictive_scale = base_scale * math.sqrt(1.0 + query_leverage)
                predicted_height = float(point_design @ coefficients)
                residual = abs(float(point_centered @ normal) - predicted_height)
                residuals[global_index, scale_index] = residual
                predictive_scales[global_index, scale_index] = predictive_scale
                query_leverages[global_index, scale_index] = query_leverage
                actual_counts[global_index, scale_index] = neighbor_count

    standardized = residuals / predictive_scales
    best_indices = np.nanargmin(standardized, axis=1)
    rows = np.arange(primary.shape[0])
    return StudentizedPairedScores(
        residuals_by_scale=residuals,
        predictive_scales_by_scale=predictive_scales,
        query_leverages_by_scale=query_leverages,
        standardized_residuals_by_scale=standardized,
        best_residuals=residuals[rows, best_indices],
        best_predictive_scales=predictive_scales[rows, best_indices],
        best_standardized_residuals=standardized[rows, best_indices],
        selected_neighbor_counts=actual_counts[rows, best_indices],
        selected_query_leverages=query_leverages[rows, best_indices],
        primary_to_replicate_layer_mapping=mapping,
    )


def estimate_studentized_paired_evidence(
    primary_points: FloatArray,
    primary_labels: IntArray,
    replicate_points: FloatArray,
    replicate_labels: IntArray,
    config: StudentizedPairedConfig | None = None,
) -> StudentizedPairedEvidence:
    selected = StudentizedPairedConfig() if config is None else config
    scores = studentized_paired_scores(
        primary_points,
        primary_labels,
        replicate_points,
        replicate_labels,
        selected,
    )
    standardized = scores.best_standardized_residuals
    descending = np.sort(standardized)[::-1]
    histogram = {
        str(count): int(np.sum(scores.selected_neighbor_counts == count))
        for count in sorted(set(scores.selected_neighbor_counts.tolist()))
    }
    return StudentizedPairedEvidence(
        information_boundary=(
            "primary_and_replicate_coordinates_with_inferred_layers_and_"
            "replicate_only_predictive_uncertainty"
        ),
        primary_point_count=int(np.asarray(primary_points).shape[0]),
        replicate_point_count=int(np.asarray(replicate_points).shape[0]),
        requested_neighbor_counts=selected.neighbor_counts,
        loo_quantile=selected.loo_quantile,
        primary_to_replicate_layer_mapping=(
            scores.primary_to_replicate_layer_mapping
        ),
        median_standardized_residual=float(np.median(standardized)),
        percentile95_standardized_residual=float(
            np.percentile(standardized, 95.0)
        ),
        peak_standardized_residual=float(descending[0]),
        support_standardized_residual=float(descending[1]),
        leading_standardized_residuals=tuple(
            float(value) for value in descending[:4]
        ),
        maximum_raw_residual=float(np.max(scores.best_residuals)),
        maximum_selected_query_leverage=float(
            np.max(scores.selected_query_leverages)
        ),
        selected_neighbor_count_histogram=histogram,
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
    studentized_config: StudentizedPairedConfig,
) -> tuple[StudentizedRawCase, ...]:
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
        harmful_distance_fraction=studentized_config.harmful_distance_fraction
    )
    rows: list[StudentizedRawCase] = []
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
        replicate_seed = case_row.seed + studentized_config.replicate_seed_offset
        replicate = make_sensor_stress_case(
            case_row.stress,
            case_row.point_count,
            reference_count=max(reference_count, case_row.point_count),
            seed=replicate_seed,
        )
        replicate_construction, _ = construct_shared_trend_surface(
            replicate.points,
            shared_trend_config,
        )
        evidence = estimate_studentized_paired_evidence(
            primary.points,
            primary_construction.inference.layer_ids,
            replicate.points,
            replicate_construction.inference.layer_ids,
            studentized_config,
        )
        endpoint = evaluate_geometry_topology_harm(
            primary_construction.mesh,
            primary.reference_points,
            primary.point_component_labels,
            characteristic_length=primary.characteristic_length,
            config=harm_config,
        )
        rows.append(
            StudentizedRawCase(
                stress=case_row.stress,
                point_count=case_row.point_count,
                repeat=case_row.repeat,
                seed=case_row.seed,
                replicate_seed=replicate_seed,
                evidence=evidence,
                endpoint=endpoint,
                unguarded_decision=case_row.candidate_decision,
            )
        )
    return tuple(rows)


def _features(row: StudentizedRawCase) -> tuple[float, float]:
    return (
        row.evidence.peak_standardized_residual,
        row.evidence.support_standardized_residual,
    )


def _feature_cohort(rows: tuple[StudentizedRawCase, ...]) -> InfluenceFeatureCohort:
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
    raw_rows: tuple[StudentizedRawCase, ...],
    *,
    panel_role: str,
    seed: int,
    rectangle: InfluenceRectangle | None,
    full_protocol: bool,
) -> StudentizedPanelResult:
    rows: list[StudentizedCaseResult] = []
    for raw in raw_rows:
        unguarded_accept = raw.unguarded_decision is SamplingGateDecision.ACCEPT
        persistent = bool(
            rectangle is not None
            and raw.evidence.peak_standardized_residual
            <= rectangle.peak_threshold
            and raw.evidence.support_standardized_residual
            <= rectangle.support_threshold
        )
        if unguarded_accept and not persistent:
            guarded_decision = SamplingGateDecision.UNSUPPORTED
        else:
            guarded_decision = raw.unguarded_decision
        guarded_accept = guarded_decision is SamplingGateDecision.ACCEPT
        harmful_outlier = bool(
            raw.stress.is_outlier_stress
            and raw.endpoint.geometry_topology_harm_present
        )
        rows.append(
            StudentizedCaseResult(
                stress=raw.stress,
                point_count=raw.point_count,
                repeat=raw.repeat,
                seed=raw.seed,
                replicate_seed=raw.replicate_seed,
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
    return StudentizedPanelResult(
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


def evaluate_studentized_paired_scan(
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
    studentized_config: StudentizedPairedConfig | None = None,
) -> StudentizedPairedResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_studentized = (
        StudentizedPairedConfig()
        if studentized_config is None
        else studentized_config
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
        and selected_studentized == StudentizedPairedConfig()
    )
    common = {
        "point_counts": selected_counts,
        "stresses": selected_stresses,
        "reference_count": reference_count,
        "repeats": repeats,
        "surface_sample_count": surface_sample_count,
        "base_gate_config": base_gate_config,
        "shared_trend_config": shared_trend_config,
        "studentized_config": selected_studentized,
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
    final_held_out: StudentizedPanelResult | None = None
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
    return StudentizedPairedResult(
        artifact_schema="pftf_alpha_studentized_paired_scan_phase16/v1",
        role="replicate_only_predictive_uncertainty_persistence_guard",
        information_boundary=(
            "route uses primary and replicate coordinates, independently inferred "
            "layers, and replicate-only LOO/leverage uncertainty; stress, source "
            "labels, and references are evaluation-only"
        ),
        development_predecessor=(
            "phase15_calibrations_21800804_21900804_development_only"
        ),
        calibration_a_seed=calibration_a_seed,
        calibration_b_seed=calibration_b_seed,
        final_held_out_seed=final_held_out_seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        studentized_config=selected_studentized,
        rectangle_selection_rule=(
            "zero harm in both calibration cohorts; maximize worst focus "
            "retention, total focus count, worst all-safe retention, total "
            "all-safe count, then peak and support thresholds"
        ),
        selected_rectangle=rectangle,
        calibration_a=calibration_a,
        calibration_b=calibration_b,
        final_held_out=final_held_out,
        phase16_supported=supported,
        paired_synthetic_supported=supported,
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
    result = evaluate_studentized_paired_scan(
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
