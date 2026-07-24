"""Calibration-only freezing of P2 confidence and adaptive scale multipliers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import rankdata

from .adaptive import (
    AdaptiveCellFiltration,
    boundary_bridge_localization,
    boundary_region_cut_intervention,
    bridge_penalized_filtration,
    density_scaled_filtration,
    geometric_bridge_risk,
    iterative_boundary_owner_intervention,
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


@dataclass(frozen=True)
class BridgePenaltyAblationPoint:
    """One evaluation-only point on the P2 bridge-penalty curve."""

    strength: float
    mean_objective: float
    mean_geometry: float
    mean_topology: float
    mean_complexity: float
    component_error_sum: int
    betti_error_sum: int
    labeled_false_bridge_edges: int
    labeled_false_bridge_faces: int
    selected_cell_count: int
    selected_flagged_cell_count: int
    selected_flagged_fraction: float
    selected_mean_risk: float
    objective_nonregression: bool
    geometry_nonregression: bool
    component_nonregression: bool
    betti_nonregression: bool
    bridge_edge_improved: bool
    bridge_face_improved: bool
    promotion_gate_passed: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class BridgePenaltyAblationResult:
    """Calibration-only audit; it never freezes or deploys a penalty."""

    base_method: BaselineID
    role: str
    penalty_formula: str
    calibration_case_count: int
    scale_multiplier: float
    candidate_count: int
    candidate_min: float
    candidate_max: float
    uses_reference_geometry_for_evaluation: bool
    uses_component_labels_for_evaluation: bool
    changes_benchmark_selection: bool
    selected_flagged_vs_false_bridge_spearman: float | None
    selected_mean_risk_vs_false_bridge_spearman: float | None
    promotion_supported: bool
    eligible_strengths: tuple[float, ...]
    curve: tuple[BridgePenaltyAblationPoint, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["base_method"] = self.base_method.value
        return payload


@dataclass(frozen=True)
class BoundaryOwnerInterventionPoint:
    """One calibration-only depth on the boundary-owner pruning curve."""

    rounds: int
    mean_objective: float
    mean_geometry: float
    mean_topology: float
    mean_complexity: float
    component_error_sum: int
    betti_error_sum: int
    labeled_false_bridge_edges: int
    labeled_false_bridge_faces: int
    initial_selected_cell_count: int
    final_selected_cell_count: int
    removed_cell_count: int
    removed_fraction: float
    executed_round_count: int
    boundary_recomputation_count: int
    remaining_flagged_face_count: int
    remaining_flagged_edge_count: int
    objective_nonregression: bool
    geometry_nonregression: bool
    component_nonregression: bool
    betti_nonregression: bool
    bridge_edge_improved: bool
    bridge_face_improved: bool
    promotion_gate_passed: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryOwnerInterventionAblationResult:
    """Calibration-only audit; no pruning depth is frozen or deployed."""

    base_method: BaselineID
    role: str
    intervention_rule: str
    risk_threshold: float
    calibration_case_count: int
    scale_multiplier: float
    candidate_count: int
    candidate_min: int
    candidate_max: int
    uses_reference_geometry_for_evaluation: bool
    uses_component_labels_for_evaluation: bool
    changes_benchmark_selection: bool
    recomputes_boundary_each_round: bool
    promotion_supported: bool
    eligible_rounds: tuple[int, ...]
    curve: tuple[BoundaryOwnerInterventionPoint, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["base_method"] = self.base_method.value
        return payload


@dataclass(frozen=True)
class BoundaryRegionCutAblationPoint:
    """One structural boundary-region/cut candidate on calibration."""

    strategy: str
    mean_objective: float
    mean_geometry: float
    mean_topology: float
    mean_complexity: float
    component_error_sum: int
    betti_error_sum: int
    labeled_false_bridge_edges: int
    labeled_false_bridge_faces: int
    risk_region_count: int
    largest_risk_region_face_count: int
    safe_boundary_component_count: int
    safe_backbone_cut_edge_count: int
    candidate_case_count: int
    candidate_face_count: int
    initial_selected_cell_count: int
    final_selected_cell_count: int
    removed_cell_count: int
    removed_fraction: float
    objective_nonregression: bool
    geometry_nonregression: bool
    component_nonregression: bool
    betti_nonregression: bool
    bridge_edge_improved: bool
    bridge_face_improved: bool
    promotion_gate_passed: bool

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryRegionCutAblationResult:
    """Calibration-only structural audit; no strategy is frozen implicitly."""

    base_method: BaselineID
    role: str
    risk_threshold: float
    region_adjacency: str
    safe_backbone_rule: str
    requested_strategies: tuple[str, ...]
    calibration_case_count: int
    scale_multiplier: float
    uses_reference_geometry_for_evaluation: bool
    uses_component_labels_for_evaluation: bool
    changes_benchmark_selection: bool
    promotion_supported: bool
    eligible_strategies: tuple[str, ...]
    curve: tuple[BoundaryRegionCutAblationPoint, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["base_method"] = self.base_method.value
        return payload


@dataclass(frozen=True)
class BoundaryBridgeCaseResult:
    """Evaluation-only localization summary for one frozen P2 output."""

    family: str
    split: str
    route: str
    normal_coherence: float
    selected_cell_count: int
    selected_dual_component_count: int
    selected_dual_edge_count: int
    selected_dual_bridge_edge_count: int
    selected_dual_articulation_cell_count: int
    boundary_face_count: int
    flagged_face_count: int
    labeled_mixed_face_count: int
    face_auc: float | None
    face_true_positive_count: int
    face_false_positive_count: int
    face_false_negative_count: int
    face_true_negative_count: int
    face_recall: float | None
    face_false_positive_rate: float | None
    boundary_edge_count: int
    flagged_edge_count: int
    labeled_mixed_edge_count: int
    edge_auc: float | None
    edge_true_positive_count: int
    edge_false_positive_count: int
    edge_false_negative_count: int
    edge_true_negative_count: int
    edge_recall: float | None
    edge_false_positive_rate: float | None
    dual_bottleneck_face_count: int
    labeled_mixed_dual_bottleneck_face_count: int
    dual_bottleneck_face_auc: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryBridgeLocalizationResult:
    """Frozen-split audit of boundary geometry and selected-cell dual cuts."""

    base_method: BaselineID
    role: str
    risk_formula: str
    risk_threshold: float
    evaluation_split: str
    case_count: int
    scale_multiplier: float
    uses_reference_geometry: bool
    uses_component_labels_for_evaluation: bool
    changes_benchmark_selection: bool
    pooled_boundary_face_count: int
    pooled_labeled_mixed_face_count: int
    pooled_flagged_face_count: int
    pooled_face_auc: float | None
    pooled_face_recall: float | None
    pooled_face_false_positive_rate: float | None
    pooled_boundary_edge_count: int
    pooled_labeled_mixed_edge_count: int
    pooled_flagged_edge_count: int
    pooled_edge_auc: float | None
    pooled_edge_recall: float | None
    pooled_edge_false_positive_rate: float | None
    pooled_dual_bottleneck_face_count: int
    pooled_dual_bottleneck_face_auc: float | None
    cases: tuple[BoundaryBridgeCaseResult, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["base_method"] = self.base_method.value
        return payload


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
        vertex_component_labels=case.point_component_labels,
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


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.ndim != 1 or second_array.shape != first_array.shape:
        raise ValueError("rank-correlation arrays must be aligned and one-dimensional")
    if first_array.size < 2:
        return None
    first_ranks = rankdata(first_array, method="average")
    second_ranks = rankdata(second_array, method="average")
    if np.ptp(first_ranks) == 0.0 or np.ptp(second_ranks) == 0.0:
        return None
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def evaluate_bridge_penalty_ablation(
    cases: Iterable[SyntheticCase],
    *,
    config: BenchmarkConfig,
    strengths: Iterable[float] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8),
) -> BridgePenaltyAblationResult:
    """Audit a label-free P2 soft penalty without selecting or deploying it.

    Geometry, component, Betti, and labeled false-bridge endpoints are used only
    to evaluate an explicit promotion gate on the calibration panel. The result
    never mutates ``config`` and never chooses a held-out penalty strength.
    """

    calibration_cases = tuple(cases)
    if not calibration_cases:
        raise ValueError("at least one calibration case is required")
    if config.p2_scale_multiplier is None:
        raise ValueError("bridge penalty ablation requires a frozen P2 multiplier")
    candidate_array = np.asarray(tuple(strengths), dtype=np.float64)
    if candidate_array.ndim != 1 or candidate_array.size < 2:
        raise ValueError("at least two bridge penalty strengths are required")
    if not np.all(np.isfinite(candidate_array)) or np.any(candidate_array < 0.0):
        raise ValueError("bridge penalty strengths must be finite and non-negative")
    candidates = np.unique(candidate_array)
    if candidates[0] != 0.0:
        raise ValueError("bridge penalty strengths must include zero as the baseline")

    inputs = []
    for case in calibration_cases:
        filtration = AlphaFiltration.from_points(case.points)
        base = pftf_confidence_fallback_filtration(
            filtration,
            k_neighbors=config.adaptive_k_neighbors,
            relation_gain=config.p1_relation_gain,
            max_condition_number=config.p1_max_condition_number,
            density_contrast_scale=config.p1_density_contrast_scale,
            receiver_imbalance_weight=config.p1_receiver_imbalance_weight,
            confidence_threshold=config.p2_confidence_threshold,
        )
        risk = geometric_bridge_risk(
            filtration,
            k_neighbors=config.adaptive_k_neighbors,
            normal_coherence_threshold=config.bridge_probe_normal_coherence_threshold,
            normal_edge_threshold=config.bridge_probe_normal_edge_threshold,
            length_edge_threshold=config.bridge_probe_length_edge_threshold,
        )
        inputs.append((case, base, risk))

    endpoint_rows: list[list[SurfaceEndpointMetrics]] = []
    selection_rows: list[tuple[int, int, float]] = []
    maximum_faces = [1 for _ in calibration_cases]
    scale_multiplier = float(config.p2_scale_multiplier)
    for strength in candidates:
        endpoints_for_strength: list[SurfaceEndpointMetrics] = []
        selected_count = 0
        selected_flagged_count = 0
        selected_risk_sum = 0.0
        for case_index, (case, base, risk) in enumerate(inputs):
            penalized = bridge_penalized_filtration(
                base,
                risk,
                strength=float(strength),
            )
            selected = penalized.scores <= scale_multiplier
            selected_count += int(np.count_nonzero(selected))
            selected_flagged_count += int(
                np.count_nonzero(selected & (risk.risk > 1.0))
            )
            selected_risk_sum += float(np.sum(risk.risk[selected]))
            endpoints = _endpoint(
                penalized,
                scale_multiplier,
                case,
                config,
                seed=config.seed + case.seed + 130_000 + case_index,
            )
            maximum_faces[case_index] = max(
                maximum_faces[case_index],
                endpoints.faces,
            )
            endpoints_for_strength.append(endpoints)
        endpoint_rows.append(endpoints_for_strength)
        selection_rows.append(
            (
                selected_count,
                selected_flagged_count,
                selected_risk_sum / max(selected_count, 1),
            )
        )

    raw_points: list[dict[str, float | int]] = []
    for strength, endpoints_row, selection_row in zip(
        candidates,
        endpoint_rows,
        selection_rows,
        strict=True,
    ):
        terms = tuple(
            _terms(endpoints, maximum_faces=maximum_faces[case_index])
            for case_index, endpoints in enumerate(endpoints_row)
        )
        betti_errors = [endpoints.betti_error for endpoints in endpoints_row]
        bridge_edges = [
            endpoints.labeled_false_bridge_edges for endpoints in endpoints_row
        ]
        bridge_faces = [
            endpoints.labeled_false_bridge_faces for endpoints in endpoints_row
        ]
        if any(value is None for value in betti_errors):
            raise RuntimeError("bridge penalty ablation requires Betti targets")
        if any(value is None for value in bridge_edges + bridge_faces):
            raise RuntimeError("bridge penalty ablation requires component labels")
        selected_count, selected_flagged_count, selected_mean_risk = selection_row
        raw_points.append(
            {
                "strength": float(strength),
                "mean_objective": float(
                    np.mean(
                        [
                            config.adaptive_weights.apply(case_terms)
                            for case_terms in terms
                        ]
                    )
                ),
                "mean_geometry": float(
                    np.mean([case_terms.geometry for case_terms in terms])
                ),
                "mean_topology": float(
                    np.mean([case_terms.topology for case_terms in terms])
                ),
                "mean_complexity": float(
                    np.mean([case_terms.complexity for case_terms in terms])
                ),
                "component_error_sum": int(
                    sum(endpoints.component_error for endpoints in endpoints_row)
                ),
                "betti_error_sum": int(sum(int(value) for value in betti_errors)),
                "labeled_false_bridge_edges": int(
                    sum(int(value) for value in bridge_edges)
                ),
                "labeled_false_bridge_faces": int(
                    sum(int(value) for value in bridge_faces)
                ),
                "selected_cell_count": selected_count,
                "selected_flagged_cell_count": selected_flagged_count,
                "selected_flagged_fraction": (
                    selected_flagged_count / max(selected_count, 1)
                ),
                "selected_mean_risk": selected_mean_risk,
            }
        )

    baseline = raw_points[0]
    tolerance = 64.0 * np.finfo(np.float64).eps
    curve: list[BridgePenaltyAblationPoint] = []
    for raw in raw_points:
        objective_nonregression = bool(
            raw["mean_objective"] <= baseline["mean_objective"] + tolerance
        )
        geometry_nonregression = bool(
            raw["mean_geometry"] <= baseline["mean_geometry"] + tolerance
        )
        component_nonregression = bool(
            raw["component_error_sum"] <= baseline["component_error_sum"]
        )
        betti_nonregression = bool(
            raw["betti_error_sum"] <= baseline["betti_error_sum"]
        )
        bridge_edge_improved = bool(
            raw["labeled_false_bridge_edges"] < baseline["labeled_false_bridge_edges"]
        )
        bridge_face_improved = bool(
            raw["labeled_false_bridge_faces"] < baseline["labeled_false_bridge_faces"]
        )
        promotion_gate_passed = bool(
            raw["strength"] > 0.0
            and objective_nonregression
            and geometry_nonregression
            and component_nonregression
            and betti_nonregression
            and bridge_edge_improved
            and bridge_face_improved
        )
        curve.append(
            BridgePenaltyAblationPoint(
                **raw,
                objective_nonregression=objective_nonregression,
                geometry_nonregression=geometry_nonregression,
                component_nonregression=component_nonregression,
                betti_nonregression=betti_nonregression,
                bridge_edge_improved=bridge_edge_improved,
                bridge_face_improved=bridge_face_improved,
                promotion_gate_passed=promotion_gate_passed,
            )
        )

    flagged_counts = np.asarray(
        [point.selected_flagged_cell_count for point in curve],
        dtype=np.float64,
    )
    selected_risks = np.asarray(
        [point.selected_mean_risk for point in curve],
        dtype=np.float64,
    )
    false_bridge_edges = np.asarray(
        [point.labeled_false_bridge_edges for point in curve],
        dtype=np.float64,
    )
    eligible_strengths = tuple(
        point.strength for point in curve if point.promotion_gate_passed
    )
    return BridgePenaltyAblationResult(
        base_method=BaselineID.P2_CONFIDENCE_FALLBACK,
        role="evaluation_only_no_selection",
        penalty_formula="score * (1 + strength * max(risk - 1, 0))",
        calibration_case_count=len(calibration_cases),
        scale_multiplier=scale_multiplier,
        candidate_count=len(curve),
        candidate_min=float(candidates[0]),
        candidate_max=float(candidates[-1]),
        uses_reference_geometry_for_evaluation=True,
        uses_component_labels_for_evaluation=True,
        changes_benchmark_selection=False,
        selected_flagged_vs_false_bridge_spearman=_rank_correlation(
            flagged_counts,
            false_bridge_edges,
        ),
        selected_mean_risk_vs_false_bridge_spearman=_rank_correlation(
            selected_risks,
            false_bridge_edges,
        ),
        promotion_supported=bool(eligible_strengths),
        eligible_strengths=eligible_strengths,
        curve=tuple(curve),
    )


def evaluate_boundary_owner_intervention(
    cases: Iterable[SyntheticCase],
    *,
    config: BenchmarkConfig,
    rounds: Iterable[int] = (0, 1, 2, 4),
    risk_threshold: float = 1.0,
) -> BoundaryOwnerInterventionAblationResult:
    """Audit iterative risky-boundary owner removal on calibration only.

    Candidate depths never alter ``config`` or held-out P2 selection. Reference
    geometry, topology targets, and component labels appear only in the
    promotion gate after each label-free intervention has been constructed.
    """

    calibration_cases = tuple(cases)
    if not calibration_cases:
        raise ValueError("at least one calibration case is required")
    if config.p2_scale_multiplier is None:
        raise ValueError("boundary owner intervention requires a frozen P2 multiplier")
    requested_rounds = tuple(rounds)
    if len(requested_rounds) < 2:
        raise ValueError("at least two intervention round counts are required")
    normalized_rounds: list[int] = []
    for value in requested_rounds:
        numeric = float(value)
        if (
            isinstance(value, bool)
            or not np.isfinite(numeric)
            or numeric < 0.0
            or not numeric.is_integer()
        ):
            raise ValueError(
                "intervention round counts must be finite non-negative integers"
            )
        normalized_rounds.append(int(numeric))
    candidates = np.unique(np.asarray(normalized_rounds, dtype=np.int64))
    if candidates.size < 2:
        raise ValueError("at least two distinct intervention depths are required")
    if candidates[0] != 0:
        raise ValueError("intervention round counts must include zero as the baseline")
    selected_risk_threshold = float(risk_threshold)
    if not np.isfinite(selected_risk_threshold) or selected_risk_threshold < 0.0:
        raise ValueError("risk_threshold must be finite and non-negative")

    inputs = [
        (
            case,
            _adaptive_for_case(
                case,
                BaselineID.P2_CONFIDENCE_FALLBACK,
                config,
            ),
        )
        for case in calibration_cases
    ]
    scale_multiplier = float(config.p2_scale_multiplier)
    endpoint_rows: list[list[SurfaceEndpointMetrics]] = []
    intervention_rows = []
    maximum_faces = [1 for _ in calibration_cases]
    for candidate_rounds in candidates:
        endpoints_for_depth: list[SurfaceEndpointMetrics] = []
        interventions_for_depth = []
        for case_index, (case, base) in enumerate(inputs):
            intervention = iterative_boundary_owner_intervention(
                base,
                scale_multiplier=scale_multiplier,
                max_rounds=int(candidate_rounds),
                risk_threshold=selected_risk_threshold,
                k_neighbors=config.adaptive_k_neighbors,
                normal_coherence_threshold=(
                    config.bridge_probe_normal_coherence_threshold
                ),
                normal_edge_threshold=config.bridge_probe_normal_edge_threshold,
                length_edge_threshold=config.bridge_probe_length_edge_threshold,
            )
            endpoints = _endpoint(
                intervention.filtration,
                scale_multiplier,
                case,
                config,
                seed=config.seed + case.seed + 170_000 + case_index,
            )
            maximum_faces[case_index] = max(
                maximum_faces[case_index],
                endpoints.faces,
            )
            endpoints_for_depth.append(endpoints)
            interventions_for_depth.append(intervention)
        endpoint_rows.append(endpoints_for_depth)
        intervention_rows.append(interventions_for_depth)

    raw_points: list[dict[str, float | int]] = []
    for candidate_rounds, endpoints_row, interventions_row in zip(
        candidates,
        endpoint_rows,
        intervention_rows,
        strict=True,
    ):
        terms = tuple(
            _terms(endpoints, maximum_faces=maximum_faces[case_index])
            for case_index, endpoints in enumerate(endpoints_row)
        )
        betti_errors = [endpoints.betti_error for endpoints in endpoints_row]
        bridge_edges = [
            endpoints.labeled_false_bridge_edges for endpoints in endpoints_row
        ]
        bridge_faces = [
            endpoints.labeled_false_bridge_faces for endpoints in endpoints_row
        ]
        if any(value is None for value in betti_errors):
            raise RuntimeError("boundary owner intervention requires Betti targets")
        if any(value is None for value in bridge_edges + bridge_faces):
            raise RuntimeError("boundary owner intervention requires component labels")
        initial_selected = sum(
            item.initial_selected_cell_count for item in interventions_row
        )
        final_selected = sum(
            item.final_selected_cell_count for item in interventions_row
        )
        removed_count = sum(
            int(item.removed_cell_indices.size) for item in interventions_row
        )
        raw_points.append(
            {
                "rounds": int(candidate_rounds),
                "mean_objective": float(
                    np.mean(
                        [
                            config.adaptive_weights.apply(case_terms)
                            for case_terms in terms
                        ]
                    )
                ),
                "mean_geometry": float(
                    np.mean([case_terms.geometry for case_terms in terms])
                ),
                "mean_topology": float(
                    np.mean([case_terms.topology for case_terms in terms])
                ),
                "mean_complexity": float(
                    np.mean([case_terms.complexity for case_terms in terms])
                ),
                "component_error_sum": int(
                    sum(endpoints.component_error for endpoints in endpoints_row)
                ),
                "betti_error_sum": int(sum(int(value) for value in betti_errors)),
                "labeled_false_bridge_edges": int(
                    sum(int(value) for value in bridge_edges)
                ),
                "labeled_false_bridge_faces": int(
                    sum(int(value) for value in bridge_faces)
                ),
                "initial_selected_cell_count": initial_selected,
                "final_selected_cell_count": final_selected,
                "removed_cell_count": removed_count,
                "removed_fraction": removed_count / max(initial_selected, 1),
                "executed_round_count": int(
                    sum(item.executed_rounds for item in interventions_row)
                ),
                "boundary_recomputation_count": int(
                    sum(item.boundary_recomputation_count for item in interventions_row)
                ),
                "remaining_flagged_face_count": int(
                    sum(item.final_flagged_face_count for item in interventions_row)
                ),
                "remaining_flagged_edge_count": int(
                    sum(item.final_flagged_edge_count for item in interventions_row)
                ),
            }
        )

    baseline = raw_points[0]
    tolerance = 64.0 * np.finfo(np.float64).eps
    curve: list[BoundaryOwnerInterventionPoint] = []
    for raw in raw_points:
        objective_nonregression = bool(
            raw["mean_objective"] <= baseline["mean_objective"] + tolerance
        )
        geometry_nonregression = bool(
            raw["mean_geometry"] <= baseline["mean_geometry"] + tolerance
        )
        component_nonregression = bool(
            raw["component_error_sum"] <= baseline["component_error_sum"]
        )
        betti_nonregression = bool(
            raw["betti_error_sum"] <= baseline["betti_error_sum"]
        )
        bridge_edge_improved = bool(
            raw["labeled_false_bridge_edges"] < baseline["labeled_false_bridge_edges"]
        )
        bridge_face_improved = bool(
            raw["labeled_false_bridge_faces"] < baseline["labeled_false_bridge_faces"]
        )
        promotion_gate_passed = bool(
            raw["rounds"] > 0
            and objective_nonregression
            and geometry_nonregression
            and component_nonregression
            and betti_nonregression
            and bridge_edge_improved
            and bridge_face_improved
        )
        curve.append(
            BoundaryOwnerInterventionPoint(
                **raw,
                objective_nonregression=objective_nonregression,
                geometry_nonregression=geometry_nonregression,
                component_nonregression=component_nonregression,
                betti_nonregression=betti_nonregression,
                bridge_edge_improved=bridge_edge_improved,
                bridge_face_improved=bridge_face_improved,
                promotion_gate_passed=promotion_gate_passed,
            )
        )

    eligible_rounds = tuple(
        point.rounds for point in curve if point.promotion_gate_passed
    )
    return BoundaryOwnerInterventionAblationResult(
        base_method=BaselineID.P2_CONFIDENCE_FALLBACK,
        role="calibration_only_evaluation_no_selection",
        intervention_rule=(
            "remove all unique owners of current boundary faces with risk above "
            "threshold; recompute boundary after every round"
        ),
        risk_threshold=selected_risk_threshold,
        calibration_case_count=len(calibration_cases),
        scale_multiplier=scale_multiplier,
        candidate_count=len(curve),
        candidate_min=int(candidates[0]),
        candidate_max=int(candidates[-1]),
        uses_reference_geometry_for_evaluation=True,
        uses_component_labels_for_evaluation=True,
        changes_benchmark_selection=False,
        recomputes_boundary_each_round=True,
        promotion_supported=bool(eligible_rounds),
        eligible_rounds=eligible_rounds,
        curve=tuple(curve),
    )


def evaluate_boundary_region_cut_ablation(
    cases: Iterable[SyntheticCase],
    *,
    config: BenchmarkConfig,
    strategies: Iterable[str] = (
        "baseline",
        "largest_risk_region",
        "safe_backbone_cut",
    ),
    risk_threshold: float = 1.0,
) -> BoundaryRegionCutAblationResult:
    """Audit connected risky regions and safe-backbone cuts on calibration."""

    calibration_cases = tuple(cases)
    if not calibration_cases:
        raise ValueError("at least one calibration case is required")
    if config.p2_scale_multiplier is None:
        raise ValueError("boundary region/cut ablation requires a frozen P2 multiplier")
    requested_strategies = tuple(strategies)
    allowed = {
        "baseline",
        "largest_risk_region",
        "safe_backbone_cut",
    }
    if len(requested_strategies) < 2:
        raise ValueError("at least two boundary region/cut strategies are required")
    if len(set(requested_strategies)) != len(requested_strategies):
        raise ValueError("boundary region/cut strategies must be unique")
    if requested_strategies[0] != "baseline":
        raise ValueError("the first boundary region/cut strategy must be baseline")
    if any(strategy not in allowed for strategy in requested_strategies):
        raise ValueError("invalid boundary region/cut strategy")
    selected_risk_threshold = float(risk_threshold)
    if not np.isfinite(selected_risk_threshold) or selected_risk_threshold < 0.0:
        raise ValueError("risk_threshold must be finite and non-negative")

    inputs = [
        (
            case,
            _adaptive_for_case(
                case,
                BaselineID.P2_CONFIDENCE_FALLBACK,
                config,
            ),
        )
        for case in calibration_cases
    ]
    scale_multiplier = float(config.p2_scale_multiplier)
    endpoint_rows: list[list[SurfaceEndpointMetrics]] = []
    intervention_rows = []
    maximum_faces = [1 for _ in calibration_cases]
    for strategy in requested_strategies:
        endpoints_for_strategy: list[SurfaceEndpointMetrics] = []
        interventions_for_strategy = []
        for case_index, (case, base) in enumerate(inputs):
            intervention = boundary_region_cut_intervention(
                base,
                scale_multiplier=scale_multiplier,
                strategy=strategy,
                risk_threshold=selected_risk_threshold,
                k_neighbors=config.adaptive_k_neighbors,
                normal_coherence_threshold=(
                    config.bridge_probe_normal_coherence_threshold
                ),
                normal_edge_threshold=config.bridge_probe_normal_edge_threshold,
                length_edge_threshold=config.bridge_probe_length_edge_threshold,
            )
            endpoints = _endpoint(
                intervention.filtration,
                scale_multiplier,
                case,
                config,
                seed=config.seed + case.seed + 230_000 + case_index,
            )
            maximum_faces[case_index] = max(
                maximum_faces[case_index],
                endpoints.faces,
            )
            endpoints_for_strategy.append(endpoints)
            interventions_for_strategy.append(intervention)
        endpoint_rows.append(endpoints_for_strategy)
        intervention_rows.append(interventions_for_strategy)

    raw_points: list[dict[str, float | int | str]] = []
    for strategy, endpoints_row, interventions_row in zip(
        requested_strategies,
        endpoint_rows,
        intervention_rows,
        strict=True,
    ):
        terms = tuple(
            _terms(endpoints, maximum_faces=maximum_faces[case_index])
            for case_index, endpoints in enumerate(endpoints_row)
        )
        betti_errors = [endpoints.betti_error for endpoints in endpoints_row]
        bridge_edges = [
            endpoints.labeled_false_bridge_edges for endpoints in endpoints_row
        ]
        bridge_faces = [
            endpoints.labeled_false_bridge_faces for endpoints in endpoints_row
        ]
        if any(value is None for value in betti_errors):
            raise RuntimeError("boundary region/cut ablation requires Betti targets")
        if any(value is None for value in bridge_edges + bridge_faces):
            raise RuntimeError("boundary region/cut ablation requires component labels")
        initial_selected = sum(
            item.initial_selected_cell_count for item in interventions_row
        )
        final_selected = sum(
            item.final_selected_cell_count for item in interventions_row
        )
        removed_count = sum(
            int(item.removed_cell_indices.size) for item in interventions_row
        )
        raw_points.append(
            {
                "strategy": strategy,
                "mean_objective": float(
                    np.mean(
                        [
                            config.adaptive_weights.apply(case_terms)
                            for case_terms in terms
                        ]
                    )
                ),
                "mean_geometry": float(
                    np.mean([case_terms.geometry for case_terms in terms])
                ),
                "mean_topology": float(
                    np.mean([case_terms.topology for case_terms in terms])
                ),
                "mean_complexity": float(
                    np.mean([case_terms.complexity for case_terms in terms])
                ),
                "component_error_sum": int(
                    sum(endpoints.component_error for endpoints in endpoints_row)
                ),
                "betti_error_sum": int(sum(int(value) for value in betti_errors)),
                "labeled_false_bridge_edges": int(
                    sum(int(value) for value in bridge_edges)
                ),
                "labeled_false_bridge_faces": int(
                    sum(int(value) for value in bridge_faces)
                ),
                "risk_region_count": int(
                    sum(
                        item.analysis.region_face_counts.size
                        for item in interventions_row
                    )
                ),
                "largest_risk_region_face_count": int(
                    max(
                        (
                            np.max(
                                item.analysis.region_face_counts,
                                initial=0,
                            )
                            for item in interventions_row
                        ),
                        default=0,
                    )
                ),
                "safe_boundary_component_count": int(
                    sum(
                        item.analysis.safe_boundary_component_count
                        for item in interventions_row
                    )
                ),
                "safe_backbone_cut_edge_count": int(
                    sum(
                        np.count_nonzero(item.analysis.flagged_edge_cut_mask)
                        for item in interventions_row
                    )
                ),
                "candidate_case_count": int(
                    sum(
                        item.removed_cell_indices.size > 0 for item in interventions_row
                    )
                ),
                "candidate_face_count": int(
                    sum(item.candidate_face_count for item in interventions_row)
                ),
                "initial_selected_cell_count": initial_selected,
                "final_selected_cell_count": final_selected,
                "removed_cell_count": removed_count,
                "removed_fraction": removed_count / max(initial_selected, 1),
            }
        )

    baseline = raw_points[0]
    tolerance = 64.0 * np.finfo(np.float64).eps
    curve: list[BoundaryRegionCutAblationPoint] = []
    for raw in raw_points:
        objective_nonregression = bool(
            raw["mean_objective"] <= baseline["mean_objective"] + tolerance
        )
        geometry_nonregression = bool(
            raw["mean_geometry"] <= baseline["mean_geometry"] + tolerance
        )
        component_nonregression = bool(
            raw["component_error_sum"] <= baseline["component_error_sum"]
        )
        betti_nonregression = bool(
            raw["betti_error_sum"] <= baseline["betti_error_sum"]
        )
        bridge_edge_improved = bool(
            raw["labeled_false_bridge_edges"] < baseline["labeled_false_bridge_edges"]
        )
        bridge_face_improved = bool(
            raw["labeled_false_bridge_faces"] < baseline["labeled_false_bridge_faces"]
        )
        promotion_gate_passed = bool(
            raw["strategy"] != "baseline"
            and objective_nonregression
            and geometry_nonregression
            and component_nonregression
            and betti_nonregression
            and bridge_edge_improved
            and bridge_face_improved
        )
        curve.append(
            BoundaryRegionCutAblationPoint(
                **raw,
                objective_nonregression=objective_nonregression,
                geometry_nonregression=geometry_nonregression,
                component_nonregression=component_nonregression,
                betti_nonregression=betti_nonregression,
                bridge_edge_improved=bridge_edge_improved,
                bridge_face_improved=bridge_face_improved,
                promotion_gate_passed=promotion_gate_passed,
            )
        )

    eligible_strategies = tuple(
        point.strategy for point in curve if point.promotion_gate_passed
    )
    return BoundaryRegionCutAblationResult(
        base_method=BaselineID.P2_CONFIDENCE_FALLBACK,
        role="calibration_only_evaluation_no_selection",
        risk_threshold=selected_risk_threshold,
        region_adjacency="flagged faces share a flagged boundary edge",
        safe_backbone_rule=(
            "remove flagged edges, label safe-edge vertex components, then retain "
            "flagged edges whose endpoints lie in distinct safe components"
        ),
        requested_strategies=requested_strategies,
        calibration_case_count=len(calibration_cases),
        scale_multiplier=scale_multiplier,
        uses_reference_geometry_for_evaluation=True,
        uses_component_labels_for_evaluation=True,
        changes_benchmark_selection=False,
        promotion_supported=bool(eligible_strategies),
        eligible_strategies=eligible_strategies,
        curve=tuple(curve),
    )


def _binary_localization_summary(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    *,
    threshold: float,
) -> tuple[float | None, int, int, int, int, float | None, float | None]:
    score_array = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positive_mask, dtype=bool)
    if score_array.ndim != 1 or positives.shape != score_array.shape:
        raise ValueError("localization scores and labels must be aligned vectors")
    flagged = score_array > threshold
    true_positive = int(np.count_nonzero(flagged & positives))
    false_positive = int(np.count_nonzero(flagged & ~positives))
    false_negative = int(np.count_nonzero(~flagged & positives))
    true_negative = int(np.count_nonzero(~flagged & ~positives))
    positive_count = true_positive + false_negative
    negative_count = false_positive + true_negative
    recall = None if positive_count == 0 else true_positive / positive_count
    false_positive_rate = (
        None if negative_count == 0 else false_positive / negative_count
    )
    auc = None
    if positive_count and negative_count:
        ranks = rankdata(score_array, method="average")
        positive_rank_sum = float(np.sum(ranks[positives]))
        auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
            positive_count * negative_count
        )
    return (
        None if auc is None else float(auc),
        true_positive,
        false_positive,
        false_negative,
        true_negative,
        None if recall is None else float(recall),
        None if false_positive_rate is None else float(false_positive_rate),
    )


def evaluate_boundary_bridge_localization(
    cases: Iterable[SyntheticCase],
    *,
    config: BenchmarkConfig,
) -> BoundaryBridgeLocalizationResult:
    """Evaluate label-free P2 boundary risks without changing P2 selection."""

    evaluation_cases = tuple(cases)
    if not evaluation_cases:
        raise ValueError("at least one evaluation case is required")
    if config.p2_scale_multiplier is None:
        raise ValueError("boundary bridge localization requires a frozen P2 multiplier")
    splits = {case.split.value for case in evaluation_cases}
    if len(splits) != 1:
        raise ValueError("boundary bridge localization requires one evaluation split")
    scale_multiplier = float(config.p2_scale_multiplier)

    case_results: list[BoundaryBridgeCaseResult] = []
    pooled_face_scores: list[np.ndarray] = []
    pooled_face_labels: list[np.ndarray] = []
    pooled_edge_scores: list[np.ndarray] = []
    pooled_edge_labels: list[np.ndarray] = []
    pooled_dual_scores: list[np.ndarray] = []
    for case in evaluation_cases:
        base = _adaptive_for_case(
            case,
            BaselineID.P2_CONFIDENCE_FALLBACK,
            config,
        )
        localization = boundary_bridge_localization(
            base,
            scale_multiplier=scale_multiplier,
            k_neighbors=config.adaptive_k_neighbors,
            normal_coherence_threshold=config.bridge_probe_normal_coherence_threshold,
            normal_edge_threshold=config.bridge_probe_normal_edge_threshold,
            length_edge_threshold=config.bridge_probe_length_edge_threshold,
        )
        face_labels = case.point_component_labels[localization.boundary_faces]
        mixed_faces = np.any(face_labels != face_labels[:, :1], axis=1)
        first_edge_labels = case.point_component_labels[
            localization.boundary_edges[:, 0]
        ]
        second_edge_labels = case.point_component_labels[
            localization.boundary_edges[:, 1]
        ]
        mixed_edges = first_edge_labels != second_edge_labels
        dual_scores = (
            localization.owner_articulation_mask.astype(np.float64)
            + localization.owner_dual_bridge_fraction
        )

        face_summary = _binary_localization_summary(
            localization.boundary_face_risk,
            mixed_faces,
            threshold=1.0,
        )
        edge_summary = _binary_localization_summary(
            localization.boundary_edge_risk,
            mixed_edges,
            threshold=1.0,
        )
        dual_summary = _binary_localization_summary(
            dual_scores,
            mixed_faces,
            threshold=0.0,
        )
        (
            face_auc,
            face_true_positive,
            face_false_positive,
            face_false_negative,
            face_true_negative,
            face_recall,
            face_false_positive_rate,
        ) = face_summary
        (
            edge_auc,
            edge_true_positive,
            edge_false_positive,
            edge_false_negative,
            edge_true_negative,
            edge_recall,
            edge_false_positive_rate,
        ) = edge_summary
        dual_bottleneck = dual_scores > 0.0
        case_results.append(
            BoundaryBridgeCaseResult(
                family=case.family.value,
                split=case.split.value,
                route=localization.route,
                normal_coherence=localization.normal_coherence,
                selected_cell_count=localization.selected_cell_count,
                selected_dual_component_count=(
                    localization.selected_dual_component_count
                ),
                selected_dual_edge_count=localization.selected_dual_edge_count,
                selected_dual_bridge_edge_count=(
                    localization.selected_dual_bridge_edge_count
                ),
                selected_dual_articulation_cell_count=(
                    localization.selected_dual_articulation_cell_count
                ),
                boundary_face_count=int(localization.boundary_faces.shape[0]),
                flagged_face_count=int(
                    np.count_nonzero(localization.boundary_face_risk > 1.0)
                ),
                labeled_mixed_face_count=int(np.count_nonzero(mixed_faces)),
                face_auc=face_auc,
                face_true_positive_count=face_true_positive,
                face_false_positive_count=face_false_positive,
                face_false_negative_count=face_false_negative,
                face_true_negative_count=face_true_negative,
                face_recall=face_recall,
                face_false_positive_rate=face_false_positive_rate,
                boundary_edge_count=int(localization.boundary_edges.shape[0]),
                flagged_edge_count=int(
                    np.count_nonzero(localization.boundary_edge_risk > 1.0)
                ),
                labeled_mixed_edge_count=int(np.count_nonzero(mixed_edges)),
                edge_auc=edge_auc,
                edge_true_positive_count=edge_true_positive,
                edge_false_positive_count=edge_false_positive,
                edge_false_negative_count=edge_false_negative,
                edge_true_negative_count=edge_true_negative,
                edge_recall=edge_recall,
                edge_false_positive_rate=edge_false_positive_rate,
                dual_bottleneck_face_count=int(np.count_nonzero(dual_bottleneck)),
                labeled_mixed_dual_bottleneck_face_count=int(
                    np.count_nonzero(dual_bottleneck & mixed_faces)
                ),
                dual_bottleneck_face_auc=dual_summary[0],
            )
        )
        pooled_face_scores.append(localization.boundary_face_risk)
        pooled_face_labels.append(mixed_faces)
        pooled_edge_scores.append(localization.boundary_edge_risk)
        pooled_edge_labels.append(mixed_edges)
        pooled_dual_scores.append(dual_scores)

    face_scores = np.concatenate(pooled_face_scores)
    face_labels = np.concatenate(pooled_face_labels)
    edge_scores = np.concatenate(pooled_edge_scores)
    edge_labels = np.concatenate(pooled_edge_labels)
    dual_scores = np.concatenate(pooled_dual_scores)
    face_summary = _binary_localization_summary(
        face_scores,
        face_labels,
        threshold=1.0,
    )
    edge_summary = _binary_localization_summary(
        edge_scores,
        edge_labels,
        threshold=1.0,
    )
    dual_summary = _binary_localization_summary(
        dual_scores,
        face_labels,
        threshold=0.0,
    )
    return BoundaryBridgeLocalizationResult(
        base_method=BaselineID.P2_CONFIDENCE_FALLBACK,
        role="evaluation_only_no_selection",
        risk_formula=(
            "route-specific boundary edge risk; face risk = maximum incident "
            "boundary edge risk"
        ),
        risk_threshold=1.0,
        evaluation_split=next(iter(splits)),
        case_count=len(evaluation_cases),
        scale_multiplier=scale_multiplier,
        uses_reference_geometry=False,
        uses_component_labels_for_evaluation=True,
        changes_benchmark_selection=False,
        pooled_boundary_face_count=int(face_scores.size),
        pooled_labeled_mixed_face_count=int(np.count_nonzero(face_labels)),
        pooled_flagged_face_count=int(np.count_nonzero(face_scores > 1.0)),
        pooled_face_auc=face_summary[0],
        pooled_face_recall=face_summary[5],
        pooled_face_false_positive_rate=face_summary[6],
        pooled_boundary_edge_count=int(edge_scores.size),
        pooled_labeled_mixed_edge_count=int(np.count_nonzero(edge_labels)),
        pooled_flagged_edge_count=int(np.count_nonzero(edge_scores > 1.0)),
        pooled_edge_auc=edge_summary[0],
        pooled_edge_recall=edge_summary[5],
        pooled_edge_false_positive_rate=edge_summary[6],
        pooled_dual_bottleneck_face_count=int(np.count_nonzero(dual_scores > 0.0)),
        pooled_dual_bottleneck_face_auc=dual_summary[0],
        cases=tuple(case_results),
    )
