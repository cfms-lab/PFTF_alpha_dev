"""Local insertion influence and frozen Phase-12 safety audit."""

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

CALIBRATION_SEED = 21100804
HELD_OUT_SEED = 21200804


@dataclass(frozen=True)
class LocalInsertionInfluenceConfig:
    """Frozen observed-only representation for Phase 12."""

    neighbor_counts: tuple[int, ...] = (12, 18, 24)
    mad_consistency_factor: float = 1.4826
    minimum_scale_fraction: float = 0.04
    harmful_distance_fraction: float = 0.025

    def __post_init__(self) -> None:
        if not self.neighbor_counts:
            raise ValueError("neighbor_counts must be non-empty")
        if tuple(sorted(set(self.neighbor_counts))) != self.neighbor_counts:
            raise ValueError("neighbor_counts must be strictly increasing")
        if self.neighbor_counts[0] < 7:
            raise ValueError("quadratic fits require at least seven neighbors")
        for name, value in (
            ("mad_consistency_factor", self.mad_consistency_factor),
            ("minimum_scale_fraction", self.minimum_scale_fraction),
            ("harmful_distance_fraction", self.harmful_distance_fraction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class LocalInsertionInfluenceScores:
    """Per-point prediction shifts from inserting the omitted point."""

    prediction_shifts_by_scale: FloatArray
    local_scales_by_scale: FloatArray
    standardized_influences_by_scale: FloatArray
    best_prediction_shifts: FloatArray
    best_standardized_influences: FloatArray
    selected_neighbor_counts: IntArray


@dataclass(frozen=True)
class LocalInsertionInfluenceEvidence:
    information_boundary: str
    point_count: int
    requested_neighbor_counts: tuple[int, ...]
    median_standardized_influence: float
    percentile95_standardized_influence: float
    peak_standardized_influence: float
    support_standardized_influence: float
    leading_standardized_influences: tuple[float, ...]
    maximum_raw_prediction_shift: float
    selected_neighbor_count_histogram: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InfluenceRectangle:
    peak_threshold: float
    support_threshold: float
    retained_focus_safe_count: int
    retained_all_safe_count: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in ("peak_threshold", "support_threshold"):
            value = float(payload[name])
            payload[name] = value if math.isfinite(value) else "infinity"
        return payload


@dataclass(frozen=True)
class LocalInsertionInfluenceCaseResult:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    evidence: LocalInsertionInfluenceEvidence
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
class LocalInsertionInfluencePanelResult:
    panel_role: str
    seed: int
    influence_rectangle: InfluenceRectangle | None
    cases: tuple[LocalInsertionInfluenceCaseResult, ...]
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
            "influence_rectangle": (
                None
                if self.influence_rectangle is None
                else self.influence_rectangle.to_dict()
            ),
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
class LocalInsertionInfluenceResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    calibration_seed: int
    held_out_seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    influence_config: LocalInsertionInfluenceConfig
    rectangle_selection_rule: str
    calibration: LocalInsertionInfluencePanelResult
    held_out: LocalInsertionInfluencePanelResult | None
    phase12_supported: bool
    trimmed_reconstruction_supported: bool
    real_scan_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_predecessor": self.frozen_predecessor,
            "calibration_seed": self.calibration_seed,
            "held_out_seed": self.held_out_seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "influence_config": asdict(self.influence_config),
            "rectangle_selection_rule": self.rectangle_selection_rule,
            "calibration": self.calibration.to_dict(),
            "held_out": None if self.held_out is None else self.held_out.to_dict(),
            "phase12_supported": self.phase12_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "real_scan_supported": self.real_scan_supported,
            "deployment_supported": self.deployment_supported,
        }


@dataclass(frozen=True)
class LocalInsertionInfluenceRawCase:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    evidence: LocalInsertionInfluenceEvidence
    endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision


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


def local_insertion_influence_scores(
    points: FloatArray,
    inferred_labels: IntArray,
    config: LocalInsertionInfluenceConfig | None = None,
) -> LocalInsertionInfluenceScores:
    """Measure how much each omitted point changes its local quadratic fit."""

    selected = LocalInsertionInfluenceConfig() if config is None else config
    point_array = np.asarray(points, dtype=np.float64)
    labels = np.asarray(inferred_labels, dtype=np.int64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if labels.shape != (point_array.shape[0],) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("inferred_labels must contain two aligned layers")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must be finite")

    shape = (point_array.shape[0], len(selected.neighbor_counts))
    shifts = np.full(shape, np.nan, dtype=np.float64)
    local_scales = np.full(shape, np.nan, dtype=np.float64)
    actual_neighbor_counts = np.zeros(shape, dtype=np.int64)
    epsilon = np.finfo(float).eps

    for layer in (0, 1):
        indices = np.flatnonzero(labels == layer)
        if indices.size < 8:
            raise ValueError("each inferred layer requires at least eight points")
        layer_points = point_array[indices]
        available_counts = tuple(
            min(count, indices.size - 1) for count in selected.neighbor_counts
        )
        neighbor_rows = cKDTree(layer_points).query(
            layer_points,
            k=max(available_counts) + 1,
            workers=1,
        )[1][:, 1:]
        for local_index, ordered_neighbors in enumerate(neighbor_rows):
            global_index = int(indices[local_index])
            omitted = point_array[global_index]
            for scale_index, neighbor_count in enumerate(available_counts):
                neighbors = layer_points[ordered_neighbors[:neighbor_count]]
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
                baseline_coefficients = np.linalg.lstsq(
                    design,
                    heights,
                    rcond=None,
                )[0]
                baseline_residuals = heights - design @ baseline_coefficients
                residual_center = float(np.median(baseline_residuals))
                robust_scale = selected.mad_consistency_factor * float(
                    np.median(np.abs(baseline_residuals - residual_center))
                )
                scale = max(
                    robust_scale,
                    selected.minimum_scale_fraction * tangent_radius,
                    epsilon,
                )

                omitted_centered = omitted - center
                omitted_tangent = (
                    omitted_centered @ tangent_basis / tangent_radius
                ).reshape(1, 2)
                omitted_design = _quadratic_design(omitted_tangent)
                omitted_height = np.asarray(
                    [float(omitted_centered @ normal)],
                    dtype=np.float64,
                )
                augmented_coefficients = np.linalg.lstsq(
                    np.vstack((design, omitted_design)),
                    np.concatenate((heights, omitted_height)),
                    rcond=None,
                )[0]
                prediction_delta = design @ (
                    augmented_coefficients - baseline_coefficients
                )
                shift = float(np.sqrt(np.mean(prediction_delta * prediction_delta)))
                shifts[global_index, scale_index] = shift
                local_scales[global_index, scale_index] = scale
                actual_neighbor_counts[global_index, scale_index] = neighbor_count

    standardized = shifts / local_scales
    best_scale_indices = np.nanargmin(standardized, axis=1)
    rows = np.arange(point_array.shape[0])
    return LocalInsertionInfluenceScores(
        prediction_shifts_by_scale=shifts,
        local_scales_by_scale=local_scales,
        standardized_influences_by_scale=standardized,
        best_prediction_shifts=shifts[rows, best_scale_indices],
        best_standardized_influences=standardized[rows, best_scale_indices],
        selected_neighbor_counts=actual_neighbor_counts[rows, best_scale_indices],
    )


def estimate_local_insertion_influence(
    points: FloatArray,
    inferred_labels: IntArray,
    config: LocalInsertionInfluenceConfig | None = None,
) -> LocalInsertionInfluenceEvidence:
    """Retain the leading order statistics of local insertion influence."""

    selected = LocalInsertionInfluenceConfig() if config is None else config
    scores = local_insertion_influence_scores(points, inferred_labels, selected)
    standardized = scores.best_standardized_influences
    descending = np.sort(standardized)[::-1]
    leading = tuple(float(value) for value in descending[:4])
    histogram = {
        str(count): int(np.sum(scores.selected_neighbor_counts == count))
        for count in sorted(set(scores.selected_neighbor_counts.tolist()))
    }
    return LocalInsertionInfluenceEvidence(
        information_boundary="observed_coordinates_and_inferred_layers_only",
        point_count=standardized.size,
        requested_neighbor_counts=selected.neighbor_counts,
        median_standardized_influence=float(np.median(standardized)),
        percentile95_standardized_influence=float(
            np.percentile(standardized, 95.0)
        ),
        peak_standardized_influence=float(descending[0]),
        support_standardized_influence=float(descending[1]),
        leading_standardized_influences=leading,
        maximum_raw_prediction_shift=float(np.max(scores.best_prediction_shifts)),
        selected_neighbor_count_histogram=histogram,
    )


def _feature_array(
    values: Sequence[tuple[float, float]],
    *,
    name: str,
) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if not result.size:
        return np.empty((0, 2), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must contain (peak, support) pairs")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def calibrate_influence_rectangle(
    harmful_features: Sequence[tuple[float, float]],
    focus_safe_features: Sequence[tuple[float, float]],
    all_safe_features: Sequence[tuple[float, float]],
) -> InfluenceRectangle | None:
    """Select the frozen zero-calibration-harm rectangular accept region."""

    harmful = _feature_array(harmful_features, name="harmful_features")
    focus_safe = _feature_array(focus_safe_features, name="focus_safe_features")
    all_safe = _feature_array(all_safe_features, name="all_safe_features")
    if not harmful.shape[0]:
        return None
    peak_candidates = sorted(
        {math.inf}
        | {
            float(np.nextafter(value, -np.inf))
            for value in harmful[:, 0].tolist()
        }
    )
    support_candidates = sorted(
        {math.inf}
        | {
            float(np.nextafter(value, -np.inf))
            for value in harmful[:, 1].tolist()
        }
    )
    best: InfluenceRectangle | None = None
    best_key: tuple[int, int, float, float] | None = None
    for peak_threshold in peak_candidates:
        for support_threshold in support_candidates:
            harmful_accepted = np.logical_and(
                harmful[:, 0] <= peak_threshold,
                harmful[:, 1] <= support_threshold,
            )
            if np.any(harmful_accepted):
                continue
            focus_retained = int(
                np.sum(
                    np.logical_and(
                        focus_safe[:, 0] <= peak_threshold,
                        focus_safe[:, 1] <= support_threshold,
                    )
                )
            )
            all_retained = int(
                np.sum(
                    np.logical_and(
                        all_safe[:, 0] <= peak_threshold,
                        all_safe[:, 1] <= support_threshold,
                    )
                )
            )
            key = (
                focus_retained,
                all_retained,
                peak_threshold,
                support_threshold,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = InfluenceRectangle(
                    peak_threshold=peak_threshold,
                    support_threshold=support_threshold,
                    retained_focus_safe_count=focus_retained,
                    retained_all_safe_count=all_retained,
                )
    return best


def evaluate_local_insertion_influence_raw_panel(
    *,
    point_counts: tuple[int, ...],
    stresses: tuple[SensorStress, ...],
    reference_count: int,
    repeats: int,
    seed: int,
    surface_sample_count: int,
    base_gate_config: SamplingSufficiencyConfig | None,
    shared_trend_config: SharedTrendConfig | None,
    influence_config: LocalInsertionInfluenceConfig,
) -> tuple[LocalInsertionInfluenceRawCase, ...]:
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
        harmful_distance_fraction=influence_config.harmful_distance_fraction
    )
    rows: list[LocalInsertionInfluenceRawCase] = []
    for case_row in base_result.cases:
        case = make_sensor_stress_case(
            case_row.stress,
            case_row.point_count,
            reference_count=max(reference_count, case_row.point_count),
            seed=case_row.seed,
        )
        construction, _ = construct_shared_trend_surface(
            case.points,
            shared_trend_config,
        )
        evidence = estimate_local_insertion_influence(
            case.points,
            construction.inference.layer_ids,
            influence_config,
        )
        endpoint = evaluate_geometry_topology_harm(
            construction.mesh,
            case.reference_points,
            case.point_component_labels,
            characteristic_length=case.characteristic_length,
            config=harm_config,
        )
        rows.append(
            LocalInsertionInfluenceRawCase(
                stress=case_row.stress,
                point_count=case_row.point_count,
                repeat=case_row.repeat,
                seed=case_row.seed,
                evidence=evidence,
                endpoint=endpoint,
                unguarded_decision=case_row.candidate_decision,
            )
        )
    return tuple(rows)


def local_insertion_influence_features(
    row: LocalInsertionInfluenceRawCase,
) -> tuple[float, float]:
    return (
        row.evidence.peak_standardized_influence,
        row.evidence.support_standardized_influence,
    )


def materialize_local_insertion_influence_panel(
    raw_rows: tuple[LocalInsertionInfluenceRawCase, ...],
    *,
    panel_role: str,
    seed: int,
    rectangle: InfluenceRectangle | None,
    full_protocol: bool,
) -> LocalInsertionInfluencePanelResult:
    rows: list[LocalInsertionInfluenceCaseResult] = []
    for raw in raw_rows:
        unguarded_accept = raw.unguarded_decision is SamplingGateDecision.ACCEPT
        influence_supported = bool(
            rectangle is not None
            and raw.evidence.peak_standardized_influence
            <= rectangle.peak_threshold
            and raw.evidence.support_standardized_influence
            <= rectangle.support_threshold
        )
        if unguarded_accept and not influence_supported:
            guarded_decision = SamplingGateDecision.UNSUPPORTED
        else:
            guarded_decision = raw.unguarded_decision
        guarded_accept = guarded_decision is SamplingGateDecision.ACCEPT
        harmful_outlier = bool(
            raw.stress.is_outlier_stress
            and raw.endpoint.geometry_topology_harm_present
        )
        rows.append(
            LocalInsertionInfluenceCaseResult(
                stress=raw.stress,
                point_count=raw.point_count,
                repeat=raw.repeat,
                seed=raw.seed,
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
    panel_gate_passed = bool(
        full_protocol
        and rectangle is not None
        and unguarded_harmful > 0
        and guarded_harmful == 0
        and retention >= 0.90
    )
    return LocalInsertionInfluencePanelResult(
        panel_role=panel_role,
        seed=seed,
        influence_rectangle=rectangle,
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
        panel_gate_passed=panel_gate_passed,
    )


def evaluate_local_insertion_influence(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    calibration_seed: int = CALIBRATION_SEED,
    held_out_seed: int = HELD_OUT_SEED,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
    influence_config: LocalInsertionInfluenceConfig | None = None,
) -> LocalInsertionInfluenceResult:
    """Calibrate Phase 12 and conditionally execute its untouched held-out panel."""

    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_influence = (
        LocalInsertionInfluenceConfig()
        if influence_config is None
        else influence_config
    )
    if repeats < 1 or not selected_counts or not selected_stresses:
        raise ValueError("counts/stresses must be non-empty and repeats positive")
    if calibration_seed == held_out_seed:
        raise ValueError("calibration and held-out seeds must differ")
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and repeats == 8
        and reference_count == 2048
        and surface_sample_count == 256
        and calibration_seed == CALIBRATION_SEED
        and held_out_seed == HELD_OUT_SEED
    )

    calibration_raw = evaluate_local_insertion_influence_raw_panel(
        point_counts=selected_counts,
        stresses=selected_stresses,
        reference_count=reference_count,
        repeats=repeats,
        seed=calibration_seed,
        surface_sample_count=surface_sample_count,
        base_gate_config=base_gate_config,
        shared_trend_config=shared_trend_config,
        influence_config=selected_influence,
    )
    harmful_features = [
        local_insertion_influence_features(row)
        for row in calibration_raw
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
        and row.stress.is_outlier_stress
        and row.endpoint.geometry_topology_harm_present
    ]
    all_safe_rows = [
        row
        for row in calibration_raw
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
        and not row.endpoint.geometry_topology_harm_present
    ]
    focus_safe_features = [
        local_insertion_influence_features(row)
        for row in all_safe_rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    rectangle = calibrate_influence_rectangle(
        harmful_features,
        focus_safe_features,
        [local_insertion_influence_features(row) for row in all_safe_rows],
    )
    calibration = materialize_local_insertion_influence_panel(
        calibration_raw,
        panel_role="calibration",
        seed=calibration_seed,
        rectangle=rectangle,
        full_protocol=full_protocol,
    )

    held_out: LocalInsertionInfluencePanelResult | None = None
    if calibration.panel_gate_passed:
        held_out_raw = evaluate_local_insertion_influence_raw_panel(
            point_counts=selected_counts,
            stresses=selected_stresses,
            reference_count=reference_count,
            repeats=repeats,
            seed=held_out_seed,
            surface_sample_count=surface_sample_count,
            base_gate_config=base_gate_config,
            shared_trend_config=shared_trend_config,
            influence_config=selected_influence,
        )
        held_out = materialize_local_insertion_influence_panel(
            held_out_raw,
            panel_role="held_out",
            seed=held_out_seed,
            rectangle=rectangle,
            full_protocol=full_protocol,
        )

    supported = bool(
        calibration.panel_gate_passed
        and held_out is not None
        and held_out.panel_gate_passed
    )
    return LocalInsertionInfluenceResult(
        artifact_schema="pftf_alpha_local_insertion_influence_phase12/v1",
        role="set_valued_local_quadratic_insertion_influence_guard",
        information_boundary=(
            "route uses observed coordinates and inferred layers only; stress, "
            "source labels, and clean references are evaluation-only"
        ),
        frozen_predecessor="phase11_seeds_20900804_21000804_negative",
        calibration_seed=calibration_seed,
        held_out_seed=held_out_seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        influence_config=selected_influence,
        rectangle_selection_rule=(
            "zero calibration harm; maximize focus safe, then all safe, then "
            "peak threshold, then support threshold"
        ),
        calibration=calibration,
        held_out=held_out,
        phase12_supported=supported,
        trimmed_reconstruction_supported=False,
        real_scan_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--held-out-seed", type=int, default=HELD_OUT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_local_insertion_influence(
        reference_count=args.reference,
        repeats=args.repeats,
        calibration_seed=args.calibration_seed,
        held_out_seed=args.held_out_seed,
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
