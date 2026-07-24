"""Calibration-only freezing of P2 confidence and adaptive scale multipliers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from .adaptive import (
    AdaptiveCellFiltration,
    density_scaled_filtration,
    pca_anisotropic_filtration,
    pftf_confidence_fallback_filtration,
    pftf_local_metric_filtration,
)
from .baselines import BaselineID, BenchmarkConfig
from .filtration import AlphaFiltration
from .selection import ObjectiveTerms
from .surface import SurfaceEndpointMetrics, evaluate_surface
from .synthetic import SyntheticCase


@dataclass(frozen=True)
class CalibrationPoint:
    multiplier: float
    mean_objective: float
    mean_geometry: float
    mean_topology: float
    mean_complexity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveCalibrationResult:
    method: BaselineID
    multiplier: float
    candidate_count: int
    candidate_min: float
    candidate_max: float
    selected_mean_objective: float
    calibration_case_count: int
    curve: tuple[CalibrationPoint, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "multiplier": self.multiplier,
            "candidate_count": self.candidate_count,
            "candidate_min": self.candidate_min,
            "candidate_max": self.candidate_max,
            "selected_mean_objective": self.selected_mean_objective,
            "calibration_case_count": self.calibration_case_count,
            "curve": [point.to_dict() for point in self.curve],
        }


@dataclass(frozen=True)
class P2ConfidenceCalibrationResult:
    """Reference-free threshold selected from pooled calibration confidence."""

    threshold: float
    target_fallback_fraction: float
    achieved_fallback_fraction: float
    absolute_fraction_error: float
    calibration_case_count: int
    cell_count: int
    fallback_count: int
    per_case_fallback_min: float
    per_case_fallback_median: float
    per_case_fallback_max: float
    uses_reference_for_selection: bool = False

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _adaptive_for_case(
    case: SyntheticCase,
    method: BaselineID,
    config: BenchmarkConfig,
) -> AdaptiveCellFiltration:
    filtration = AlphaFiltration.from_points(case.points)
    if method is BaselineID.B4_DENSITY_SCALED:
        return density_scaled_filtration(
            filtration, k_neighbors=config.adaptive_k_neighbors
        )
    if method is BaselineID.B5_PCA_ANISOTROPIC:
        return pca_anisotropic_filtration(
            filtration,
            k_neighbors=config.adaptive_k_neighbors,
            max_normal_penalty=config.b5_max_normal_penalty,
        )
    if method is BaselineID.P1_PFTF_LOCAL_SPD:
        return pftf_local_metric_filtration(
            filtration,
            k_neighbors=config.adaptive_k_neighbors,
            relation_gain=config.p1_relation_gain,
            max_condition_number=config.p1_max_condition_number,
            density_contrast_scale=config.p1_density_contrast_scale,
            receiver_imbalance_weight=config.p1_receiver_imbalance_weight,
        )
    if method is BaselineID.P2_CONFIDENCE_FALLBACK:
        return pftf_confidence_fallback_filtration(
            filtration,
            k_neighbors=config.adaptive_k_neighbors,
            relation_gain=config.p1_relation_gain,
            max_condition_number=config.p1_max_condition_number,
            density_contrast_scale=config.p1_density_contrast_scale,
            receiver_imbalance_weight=config.p1_receiver_imbalance_weight,
            confidence_threshold=config.p2_confidence_threshold,
        )
    raise ValueError("adaptive calibration supports only B4, B5, P1, or P2")


def calibrate_p2_confidence_threshold(
    cases: Iterable[SyntheticCase],
    *,
    config: BenchmarkConfig,
    target_fallback_fraction: float = 0.25,
) -> P2ConfidenceCalibrationResult:
    """Freeze a P2 threshold without reading dense reference geometry.

    The threshold is placed between pooled P1 simplex-confidence order
    statistics so the requested fraction of calibration cells is routed to
    the trusted B4 guard whenever confidence ties permit it.
    """

    calibration_cases = tuple(cases)
    if not calibration_cases:
        raise ValueError("at least one calibration case is required")
    target = float(target_fallback_fraction)
    if not np.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target_fallback_fraction must lie strictly between 0 and 1")

    adaptives = tuple(
        _adaptive_for_case(case, BaselineID.P1_PFTF_LOCAL_SPD, config)
        for case in calibration_cases
    )
    confidence_by_case: list[np.ndarray] = []
    for adaptive in adaptives:
        if adaptive.cell_confidence is None:
            raise RuntimeError("P1 calibration did not provide simplex confidence")
        confidence_by_case.append(adaptive.cell_confidence)
    pooled = np.concatenate(confidence_by_case)
    target_count = min(
        pooled.size - 1,
        max(1, int(round(target * pooled.size))),
    )
    ordered = np.sort(pooled)
    lower = float(ordered[target_count - 1])
    upper = float(ordered[target_count])
    if lower < upper:
        threshold = 0.5 * (lower + upper)
    else:
        threshold = min(float(np.nextafter(lower, np.inf)), 1.0)

    per_case_fractions = np.asarray(
        [np.mean(confidence < threshold) for confidence in confidence_by_case],
        dtype=np.float64,
    )
    fallback_count = int(np.count_nonzero(pooled < threshold))
    achieved = fallback_count / pooled.size
    return P2ConfidenceCalibrationResult(
        threshold=threshold,
        target_fallback_fraction=target,
        achieved_fallback_fraction=achieved,
        absolute_fraction_error=abs(achieved - target),
        calibration_case_count=len(calibration_cases),
        cell_count=int(pooled.size),
        fallback_count=fallback_count,
        per_case_fallback_min=float(np.min(per_case_fractions)),
        per_case_fallback_median=float(np.median(per_case_fractions)),
        per_case_fallback_max=float(np.max(per_case_fractions)),
    )


def _endpoint(
    adaptive: AdaptiveCellFiltration,
    multiplier: float,
    case: SyntheticCase,
    config: BenchmarkConfig,
    *,
    seed: int,
) -> SurfaceEndpointMetrics:
    return evaluate_surface(
        adaptive.surface_at(multiplier),
        case.reference_points,
        expected_components=case.expected_components,
        characteristic_length=case.characteristic_length,
        expected_betti=case.expected_surface_betti,
        sample_count=config.surface_sample_count,
        threshold_fraction=config.fscore_threshold_fraction,
        seed=seed,
    )


def _terms(
    endpoints: SurfaceEndpointMetrics,
    *,
    maximum_faces: int,
) -> ObjectiveTerms:
    return ObjectiveTerms(
        geometry=(
            endpoints.normalized_chamfer_squared + endpoints.normalized_hausdorff
        ),
        topology=float(endpoints.component_error),
        stability=0.0,
        complexity=(
            endpoints.nonmanifold_edges / max(endpoints.edges, 1)
            + endpoints.faces / maximum_faces
        ),
    )


def calibrate_adaptive_multiplier(
    cases: Iterable[SyntheticCase],
    method: BaselineID | str,
    *,
    config: BenchmarkConfig,
    candidate_budget: int = 24,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.95,
) -> AdaptiveCalibrationResult:
    """Freeze one multiplier by mean calibration loss across all cases."""

    selected_method = BaselineID(method)
    calibration_cases = tuple(cases)
    if not calibration_cases:
        raise ValueError("at least one calibration case is required")
    if candidate_budget < 2:
        raise ValueError("candidate_budget must be at least two")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")

    adaptives = tuple(
        _adaptive_for_case(case, selected_method, config) for case in calibration_cases
    )
    pooled_scores = np.concatenate(
        [adaptive.critical_values() for adaptive in adaptives]
    )
    positive = pooled_scores[pooled_scores > 0.0]
    if positive.size == 0:
        raise ValueError("adaptive calibration has no positive scores")
    lower = float(np.quantile(positive, lower_quantile))
    upper = float(np.quantile(positive, upper_quantile))
    if upper <= lower:
        candidates = np.asarray([lower], dtype=np.float64)
    else:
        candidates = np.geomspace(lower, upper, num=candidate_budget)
        if lower <= 1.0 <= upper:
            candidates = np.unique(np.append(candidates, 1.0))

    endpoint_rows: list[list[SurfaceEndpointMetrics]] = []
    maximum_faces = [1 for _ in calibration_cases]
    for multiplier in candidates:
        row: list[SurfaceEndpointMetrics] = []
        for case_index, (case, adaptive) in enumerate(
            zip(calibration_cases, adaptives, strict=True)
        ):
            endpoints = _endpoint(
                adaptive,
                float(multiplier),
                case,
                config,
                seed=config.seed + case.seed + 90_000 + case_index,
            )
            maximum_faces[case_index] = max(maximum_faces[case_index], endpoints.faces)
            row.append(endpoints)
        endpoint_rows.append(row)

    curve: list[CalibrationPoint] = []
    for multiplier, row in zip(candidates, endpoint_rows, strict=True):
        terms = tuple(
            _terms(endpoints, maximum_faces=maximum_faces[case_index])
            for case_index, endpoints in enumerate(row)
        )
        curve.append(
            CalibrationPoint(
                multiplier=float(multiplier),
                mean_objective=float(
                    np.mean(
                        [
                            config.adaptive_weights.apply(case_terms)
                            for case_terms in terms
                        ]
                    )
                ),
                mean_geometry=float(
                    np.mean([case_terms.geometry for case_terms in terms])
                ),
                mean_topology=float(
                    np.mean([case_terms.topology for case_terms in terms])
                ),
                mean_complexity=float(
                    np.mean([case_terms.complexity for case_terms in terms])
                ),
            )
        )

    selected = min(
        curve,
        key=lambda point: (
            point.mean_objective,
            point.mean_complexity,
            point.multiplier,
        ),
    )
    return AdaptiveCalibrationResult(
        method=selected_method,
        multiplier=selected.multiplier,
        candidate_count=len(curve),
        candidate_min=float(candidates[0]),
        candidate_max=float(candidates[-1]),
        selected_mean_objective=selected.mean_objective,
        calibration_case_count=len(calibration_cases),
        curve=tuple(curve),
    )
