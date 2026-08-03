"""Multiscale quadratic surface consensus and frozen Phase-11 audit."""

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
BoolArray = NDArray[np.bool_]

CALIBRATION_SEED = 20900804
HELD_OUT_SEED = 21000804


@dataclass(frozen=True)
class MultiscaleQuadraticConfig:
    """Frozen observed-only representation for Phase 11."""

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
class MultiscaleQuadraticScores:
    """Leave-one-out quadratic residuals at every requested scale."""

    residuals_by_scale: FloatArray
    local_scales_by_scale: FloatArray
    standardized_residuals_by_scale: FloatArray
    feasible_by_scale: BoolArray
    best_residuals: FloatArray
    best_standardized_residuals: FloatArray
    selected_neighbor_counts: IntArray


@dataclass(frozen=True)
class MultiscaleQuadraticEvidence:
    information_boundary: str
    point_count: int
    requested_neighbor_counts: tuple[int, ...]
    minimum_selected_neighbor_count: int
    median_standardized_residual: float
    percentile95_standardized_residual: float
    maximum_standardized_residual: float
    maximum_raw_residual: float
    selected_neighbor_count_histogram: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultiscaleConsensusCaseResult:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    evidence: MultiscaleQuadraticEvidence
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
class MultiscaleConsensusSummary:
    stress: SensorStress
    case_count: int
    unguarded_safe_accept_count: int
    guarded_safe_accept_count: int
    safe_accept_retention: float | None
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress"] = self.stress.value
        return payload


@dataclass(frozen=True)
class MultiscaleConsensusPanelResult:
    panel_role: str
    seed: int
    routing_threshold: float | None
    cases: tuple[MultiscaleConsensusCaseResult, ...]
    stress_summaries: tuple[MultiscaleConsensusSummary, ...]
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int
    clean_local_bump_unguarded_safe_accept_count: int
    clean_local_bump_guarded_safe_accept_count: int
    clean_local_bump_safe_accept_retention: float
    full_protocol: bool
    panel_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_role": self.panel_role,
            "seed": self.seed,
            "routing_threshold": self.routing_threshold,
            "cases": [case.to_dict() for case in self.cases],
            "stress_summaries": [row.to_dict() for row in self.stress_summaries],
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
            "full_protocol": self.full_protocol,
            "panel_gate_passed": self.panel_gate_passed,
        }


@dataclass(frozen=True)
class MultiscaleSurfaceConsensusResult:
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
    consensus_config: MultiscaleQuadraticConfig
    threshold_selection_rule: str
    calibration: MultiscaleConsensusPanelResult
    held_out: MultiscaleConsensusPanelResult | None
    phase11_supported: bool
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
            "consensus_config": asdict(self.consensus_config),
            "threshold_selection_rule": self.threshold_selection_rule,
            "calibration": self.calibration.to_dict(),
            "held_out": None if self.held_out is None else self.held_out.to_dict(),
            "phase11_supported": self.phase11_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "real_scan_supported": self.real_scan_supported,
            "deployment_supported": self.deployment_supported,
        }


@dataclass(frozen=True)
class _RawCase:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    evidence: MultiscaleQuadraticEvidence
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


def multiscale_quadratic_scores(
    points: FloatArray,
    inferred_labels: IntArray,
    config: MultiscaleQuadraticConfig | None = None,
) -> MultiscaleQuadraticScores:
    """Fit leave-one-out local quadratic surfaces at several spatial scales."""

    selected = MultiscaleQuadraticConfig() if config is None else config
    point_array = np.asarray(points, dtype=np.float64)
    labels = np.asarray(inferred_labels, dtype=np.int64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if labels.shape != (point_array.shape[0],) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("inferred_labels must contain two aligned layers")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must be finite")

    shape = (point_array.shape[0], len(selected.neighbor_counts))
    residuals = np.full(shape, np.nan, dtype=np.float64)
    scales = np.full(shape, np.nan, dtype=np.float64)
    feasible = np.zeros(shape, dtype=bool)
    epsilon = np.finfo(float).eps

    for layer in (0, 1):
        indices = np.flatnonzero(labels == layer)
        if indices.size < 8:
            raise ValueError("each inferred layer requires at least eight points")
        layer_points = point_array[indices]
        available_counts = tuple(
            min(count, indices.size - 1) for count in selected.neighbor_counts
        )
        largest_count = max(available_counts)
        neighbor_rows = cKDTree(layer_points).query(
            layer_points,
            k=largest_count + 1,
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
                normalized_tangent = tangent / tangent_radius
                design = _quadratic_design(normalized_tangent)
                coefficients = np.linalg.lstsq(
                    design,
                    heights,
                    rcond=None,
                )[0]
                fit_residuals = heights - design @ coefficients
                residual_center = float(np.median(fit_residuals))
                robust_scale = selected.mad_consistency_factor * float(
                    np.median(np.abs(fit_residuals - residual_center))
                )
                local_scale = max(
                    robust_scale,
                    selected.minimum_scale_fraction * tangent_radius,
                    epsilon,
                )
                omitted_centered = omitted - center
                omitted_tangent = (
                    omitted_centered @ tangent_basis / tangent_radius
                ).reshape(1, 2)
                predicted_height = float(
                    _quadratic_design(omitted_tangent)[0] @ coefficients
                )
                residual = abs(float(omitted_centered @ normal) - predicted_height)
                residuals[global_index, scale_index] = residual
                scales[global_index, scale_index] = local_scale
                feasible[global_index, scale_index] = True

    standardized = residuals / scales
    best_scale_indices = np.nanargmin(standardized, axis=1)
    rows = np.arange(point_array.shape[0])
    selected_counts_by_point = np.asarray(
        [selected.neighbor_counts[index] for index in best_scale_indices],
        dtype=np.int64,
    )
    return MultiscaleQuadraticScores(
        residuals_by_scale=residuals,
        local_scales_by_scale=scales,
        standardized_residuals_by_scale=standardized,
        feasible_by_scale=feasible,
        best_residuals=residuals[rows, best_scale_indices],
        best_standardized_residuals=standardized[rows, best_scale_indices],
        selected_neighbor_counts=selected_counts_by_point,
    )


def estimate_multiscale_quadratic_consensus(
    points: FloatArray,
    inferred_labels: IntArray,
    config: MultiscaleQuadraticConfig | None = None,
) -> MultiscaleQuadraticEvidence:
    """Summarize the observed-only multiscale quadratic case score."""

    selected = MultiscaleQuadraticConfig() if config is None else config
    scores = multiscale_quadratic_scores(points, inferred_labels, selected)
    standardized = scores.best_standardized_residuals
    histogram = {
        str(count): int(np.sum(scores.selected_neighbor_counts == count))
        for count in selected.neighbor_counts
    }
    return MultiscaleQuadraticEvidence(
        information_boundary="observed_coordinates_and_inferred_layers_only",
        point_count=standardized.size,
        requested_neighbor_counts=selected.neighbor_counts,
        minimum_selected_neighbor_count=int(
            np.min(scores.selected_neighbor_counts)
        ),
        median_standardized_residual=float(np.median(standardized)),
        percentile95_standardized_residual=float(
            np.percentile(standardized, 95.0)
        ),
        maximum_standardized_residual=float(np.max(standardized)),
        maximum_raw_residual=float(np.max(scores.best_residuals)),
        selected_neighbor_count_histogram=histogram,
    )


def calibrated_zero_harm_threshold(
    harmful_case_scores: Sequence[float],
) -> float | None:
    """Return the least restrictive threshold strictly below every harm score."""

    scores = np.asarray(harmful_case_scores, dtype=np.float64)
    if not scores.size:
        return None
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("harmful_case_scores must be finite and one-dimensional")
    if np.any(scores < 0.0):
        raise ValueError("harmful_case_scores must be non-negative")
    return float(np.nextafter(np.min(scores), -np.inf))


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
    consensus_config: MultiscaleQuadraticConfig,
) -> tuple[_RawCase, ...]:
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
        harmful_distance_fraction=consensus_config.harmful_distance_fraction
    )
    rows: list[_RawCase] = []
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
        evidence = estimate_multiscale_quadratic_consensus(
            case.points,
            construction.inference.layer_ids,
            consensus_config,
        )
        endpoint = evaluate_geometry_topology_harm(
            construction.mesh,
            case.reference_points,
            case.point_component_labels,
            characteristic_length=case.characteristic_length,
            config=harm_config,
        )
        rows.append(
            _RawCase(
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


def _summary(
    rows: Sequence[MultiscaleConsensusCaseResult],
    stress: SensorStress,
) -> MultiscaleConsensusSummary:
    selected = [case for case in rows if case.stress is stress]
    unguarded_safe = sum(case.unguarded_safe_accept for case in selected)
    guarded_safe = sum(case.guarded_safe_accept for case in selected)
    retention = None if not unguarded_safe else guarded_safe / unguarded_safe
    return MultiscaleConsensusSummary(
        stress=stress,
        case_count=len(selected),
        unguarded_safe_accept_count=unguarded_safe,
        guarded_safe_accept_count=guarded_safe,
        safe_accept_retention=retention,
        unguarded_harmful_outlier_false_safe_count=sum(
            case.unguarded_harmful_outlier_false_safe for case in selected
        ),
        guarded_harmful_outlier_false_safe_count=sum(
            case.guarded_harmful_outlier_false_safe for case in selected
        ),
        unguarded_provenance_violation_accept_count=sum(
            case.unguarded_provenance_violation_accept for case in selected
        ),
        guarded_provenance_violation_accept_count=sum(
            case.guarded_provenance_violation_accept for case in selected
        ),
    )


def _materialize_panel(
    raw_rows: tuple[_RawCase, ...],
    *,
    panel_role: str,
    seed: int,
    threshold: float | None,
    stresses: tuple[SensorStress, ...],
    full_protocol: bool,
) -> MultiscaleConsensusPanelResult:
    rows: list[MultiscaleConsensusCaseResult] = []
    for raw in raw_rows:
        unguarded_accept = raw.unguarded_decision is SamplingGateDecision.ACCEPT
        surface_consistent = bool(
            threshold is not None
            and raw.evidence.maximum_standardized_residual <= threshold
        )
        if unguarded_accept and not surface_consistent:
            guarded_decision = SamplingGateDecision.UNSUPPORTED
        else:
            guarded_decision = raw.unguarded_decision
        guarded_accept = guarded_decision is SamplingGateDecision.ACCEPT
        harmful_outlier = bool(
            raw.stress.is_outlier_stress
            and raw.endpoint.geometry_topology_harm_present
        )
        rows.append(
            MultiscaleConsensusCaseResult(
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

    clean_local = [
        case
        for case in rows
        if case.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_clean = sum(case.unguarded_safe_accept for case in clean_local)
    guarded_clean = sum(case.guarded_safe_accept for case in clean_local)
    retention = 0.0 if not unguarded_clean else guarded_clean / unguarded_clean
    unguarded_harmful = sum(
        case.unguarded_harmful_outlier_false_safe for case in rows
    )
    guarded_harmful = sum(
        case.guarded_harmful_outlier_false_safe for case in rows
    )
    panel_gate_passed = bool(
        full_protocol
        and threshold is not None
        and unguarded_harmful > 0
        and guarded_harmful == 0
        and retention >= 0.90
    )
    return MultiscaleConsensusPanelResult(
        panel_role=panel_role,
        seed=seed,
        routing_threshold=threshold,
        cases=tuple(rows),
        stress_summaries=tuple(_summary(rows, stress) for stress in stresses),
        case_count=len(rows),
        unguarded_harmful_outlier_false_safe_count=unguarded_harmful,
        guarded_harmful_outlier_false_safe_count=guarded_harmful,
        unguarded_provenance_violation_accept_count=sum(
            case.unguarded_provenance_violation_accept for case in rows
        ),
        guarded_provenance_violation_accept_count=sum(
            case.guarded_provenance_violation_accept for case in rows
        ),
        clean_local_bump_unguarded_safe_accept_count=unguarded_clean,
        clean_local_bump_guarded_safe_accept_count=guarded_clean,
        clean_local_bump_safe_accept_retention=retention,
        full_protocol=full_protocol,
        panel_gate_passed=panel_gate_passed,
    )


def evaluate_multiscale_surface_consensus(
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
    consensus_config: MultiscaleQuadraticConfig | None = None,
) -> MultiscaleSurfaceConsensusResult:
    """Calibrate Phase 11 and conditionally execute its untouched held-out panel."""

    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_consensus = (
        MultiscaleQuadraticConfig()
        if consensus_config is None
        else consensus_config
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

    calibration_raw = _raw_panel(
        point_counts=selected_counts,
        stresses=selected_stresses,
        reference_count=reference_count,
        repeats=repeats,
        seed=calibration_seed,
        surface_sample_count=surface_sample_count,
        base_gate_config=base_gate_config,
        shared_trend_config=shared_trend_config,
        consensus_config=selected_consensus,
    )
    harmful_scores = [
        row.evidence.maximum_standardized_residual
        for row in calibration_raw
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
        and row.stress.is_outlier_stress
        and row.endpoint.geometry_topology_harm_present
    ]
    threshold = calibrated_zero_harm_threshold(harmful_scores)
    calibration = _materialize_panel(
        calibration_raw,
        panel_role="calibration",
        seed=calibration_seed,
        threshold=threshold,
        stresses=selected_stresses,
        full_protocol=full_protocol,
    )

    held_out: MultiscaleConsensusPanelResult | None = None
    if calibration.panel_gate_passed:
        held_out_raw = _raw_panel(
            point_counts=selected_counts,
            stresses=selected_stresses,
            reference_count=reference_count,
            repeats=repeats,
            seed=held_out_seed,
            surface_sample_count=surface_sample_count,
            base_gate_config=base_gate_config,
            shared_trend_config=shared_trend_config,
            consensus_config=selected_consensus,
        )
        held_out = _materialize_panel(
            held_out_raw,
            panel_role="held_out",
            seed=held_out_seed,
            threshold=threshold,
            stresses=selected_stresses,
            full_protocol=full_protocol,
        )

    supported = bool(
        calibration.panel_gate_passed
        and held_out is not None
        and held_out.panel_gate_passed
    )
    return MultiscaleSurfaceConsensusResult(
        artifact_schema="pftf_alpha_multiscale_surface_consensus_phase11/v1",
        role="multiscale_quadratic_leave_one_out_calibrated_guard",
        information_boundary=(
            "route uses observed coordinates and inferred layers only; stress, "
            "source labels, and clean references are evaluation-only"
        ),
        frozen_predecessor="phase10_seed_20800804_negative",
        calibration_seed=calibration_seed,
        held_out_seed=held_out_seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        consensus_config=selected_consensus,
        threshold_selection_rule=(
            "nextafter(min(calibration accepted harmful case score), -infinity)"
        ),
        calibration=calibration,
        held_out=held_out,
        phase11_supported=supported,
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
    result = evaluate_multiscale_surface_consensus(
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
