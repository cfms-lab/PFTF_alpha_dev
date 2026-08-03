"""Calibrate and validate the frozen Phase-43 confidence-alpha comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .adaptive import (
    AdaptiveCellFiltration,
    density_scaled_filtration,
    pca_anisotropic_filtration,
)
from .confidence_alpha_field import (
    binary_confidence_subset,
    confidence_weighted_filtration,
    observed_point_confidence,
)
from .confidence_alpha_panel import (
    BINARY_CONFIDENCE_THRESHOLDS,
    CONTINUOUS_STRENGTHS,
    FSCORE_THRESHOLD_FRACTION,
    PROTOCOL_SCHEMA,
    SCALE_QUANTILES,
    SURFACE_SAMPLE_COUNT,
    ConfidenceAlphaCase,
    make_confidence_alpha_panel,
)
from .filtration import AlphaFiltration
from .surface import SurfaceEndpointMetrics, evaluate_surface
from .synthetic import PanelSplit

RESULT_SCHEMA = "pftf_alpha_confidence_alpha_benchmark_phase43/v1"
EXPECTED_PROTOCOL_SHA256 = (
    "a1e9e37b7b1e6bc203a061eabbce3647699d8104dd26c034270e69486b08853d"
)
K_NEIGHBORS = 12
B5_MAX_NORMAL_PENALTY = 4.0
METHOD_ORDER = (
    "anchor_density_B4",
    "fused_density_B4",
    "fused_pca_B5",
    "binary_confidence_deletion",
    "continuous_confidence_weighted_B4",
)


@dataclass(frozen=True)
class PreparedFiltration:
    case: ConfidenceAlphaCase
    method_id: str
    hyperparameter: float
    filtration: AdaptiveCellFiltration
    point_component_labels: np.ndarray
    retained_target_fraction: float
    target_confidence_mean: float


@dataclass(frozen=True)
class CalibrationCandidate:
    method_id: str
    hyperparameter: float
    scale_quantile: float
    score_threshold: float
    mean_geometry_loss: float
    mean_betti_error: float
    mean_objective: float
    mean_fscore: float


@dataclass(frozen=True)
class SelectedMethod:
    method_id: str
    hyperparameter: float
    scale_quantile: float
    score_threshold: float
    calibration_mean_geometry_loss: float
    calibration_mean_betti_error: float
    calibration_mean_objective: float
    calibration_mean_fscore: float


@dataclass(frozen=True)
class ConfidenceAlphaCaseResult:
    case_id: str
    family: str
    profile: str
    seed: int
    method_id: str
    hyperparameter: float
    score_threshold: float
    geometry_loss: float
    betti_error: int
    objective: float
    fscore: float
    precision: float
    recall: float
    connected_components: int
    betti_0: int
    betti_1: int
    betti_2: int
    faces: int
    nonmanifold_edges: int
    retained_target_fraction: float
    target_confidence_mean: float


@dataclass(frozen=True)
class MethodSummary:
    method_id: str
    case_count: int
    mean_geometry_loss: float
    mean_betti_error: float
    mean_objective: float
    mean_fscore: float
    mean_precision: float
    mean_recall: float
    mean_nonmanifold_fraction: float
    mean_retained_target_fraction: float
    repeat_stability: float


@dataclass(frozen=True)
class ConfidenceAlphaBenchmarkResult:
    artifact_schema: str
    protocol_path: str
    protocol_sha256: str
    calibration_case_count: int
    held_out_case_count: int
    selected_methods: tuple[SelectedMethod, ...]
    held_out_cases: tuple[ConfidenceAlphaCaseResult, ...]
    summaries: tuple[MethodSummary, ...]
    geometry_gate_passed: bool
    topology_gate_passed: bool
    stability_gate_passed: bool
    continuous_confidence_weighting_supported: bool
    bounded_simulated_confidence_filtration_supported: bool
    anchor_objective_dominance_supported: bool
    calibration_scale_boundary_reached: bool
    point_local_alpha_field_supported: bool
    topology_correctness_supported: bool
    classical_spatial_alpha_complex_supported: bool
    pftf_predicts_one_global_alpha_supported: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["selected_methods"] = [asdict(row) for row in self.selected_methods]
        payload["held_out_cases"] = [asdict(row) for row in self.held_out_cases]
        payload["summaries"] = [asdict(row) for row in self.summaries]
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol(path: str | Path) -> None:
    protocol = Path(path)
    if _sha256(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Phase-43 protocol SHA-256 mismatch")
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    if payload.get("artifact_schema") != PROTOCOL_SCHEMA:
        raise ValueError("Phase-43 protocol schema mismatch")


def _prepare(
    case: ConfidenceAlphaCase,
    method_id: str,
    hyperparameter: float,
) -> PreparedFiltration:
    confidence = observed_point_confidence(
        case.anchor_points,
        case.target_points,
        k_neighbors=K_NEIGHBORS,
    )
    target_confidence_mean = float(np.mean(confidence.target_confidence))
    if method_id == "anchor_density_B4":
        alpha = AlphaFiltration.from_points(case.anchor_points)
        adaptive = density_scaled_filtration(alpha, k_neighbors=K_NEIGHBORS)
        labels = case.point_component_labels[: case.anchor_points.shape[0]]
        retained_fraction = 0.0
    elif method_id == "fused_density_B4":
        alpha = AlphaFiltration.from_points(case.points)
        adaptive = density_scaled_filtration(alpha, k_neighbors=K_NEIGHBORS)
        labels = case.point_component_labels
        retained_fraction = 1.0
    elif method_id == "fused_pca_B5":
        alpha = AlphaFiltration.from_points(case.points)
        adaptive = pca_anisotropic_filtration(
            alpha,
            k_neighbors=K_NEIGHBORS,
            max_normal_penalty=B5_MAX_NORMAL_PENALTY,
        )
        labels = case.point_component_labels
        retained_fraction = 1.0
    elif method_id == "binary_confidence_deletion":
        points, retained = binary_confidence_subset(
            case.anchor_points,
            case.target_points,
            confidence.target_confidence,
            threshold=hyperparameter,
        )
        alpha = AlphaFiltration.from_points(points)
        adaptive = density_scaled_filtration(alpha, k_neighbors=K_NEIGHBORS)
        labels = np.concatenate(
            (
                case.point_component_labels[: case.anchor_points.shape[0]],
                case.point_component_labels[case.anchor_points.shape[0] :][retained],
            )
        )
        retained_fraction = float(np.mean(retained))
    elif method_id == "continuous_confidence_weighted_B4":
        alpha = AlphaFiltration.from_points(case.points)
        adaptive = confidence_weighted_filtration(
            alpha,
            confidence.point_confidence,
            k_neighbors=K_NEIGHBORS,
            penalty_strength=hyperparameter,
        )
        labels = case.point_component_labels
        retained_fraction = 1.0
    else:
        raise ValueError(f"unknown Phase-43 method: {method_id}")
    return PreparedFiltration(
        case=case,
        method_id=method_id,
        hyperparameter=float(hyperparameter),
        filtration=adaptive,
        point_component_labels=np.asarray(labels, dtype=np.int64),
        retained_target_fraction=retained_fraction,
        target_confidence_mean=target_confidence_mean,
    )


def _endpoint(
    prepared: PreparedFiltration,
    score_threshold: float,
    *,
    sample_count: int,
) -> SurfaceEndpointMetrics:
    mesh = prepared.filtration.surface_at(score_threshold)
    case = prepared.case
    return evaluate_surface(
        mesh,
        case.reference_points,
        expected_components=case.expected_components,
        characteristic_length=case.characteristic_length,
        sample_count=sample_count,
        threshold_fraction=FSCORE_THRESHOLD_FRACTION,
        seed=case.seed,
        expected_betti=case.expected_surface_betti,
        vertex_component_labels=prepared.point_component_labels,
    )


def _geometry(endpoint: SurfaceEndpointMetrics) -> float:
    return endpoint.normalized_chamfer_squared + endpoint.normalized_hausdorff


def _objective(endpoint: SurfaceEndpointMetrics) -> float:
    assert endpoint.betti_error is not None
    return _geometry(endpoint) + 0.05 * endpoint.betti_error


def _configuration_grid(
    *,
    continuous_strengths: Sequence[float],
    binary_thresholds: Sequence[float],
) -> tuple[tuple[str, float], ...]:
    return (
        ("anchor_density_B4", 0.0),
        ("fused_density_B4", 0.0),
        ("fused_pca_B5", B5_MAX_NORMAL_PENALTY),
        *(
            ("binary_confidence_deletion", float(value))
            for value in binary_thresholds
        ),
        *(
            ("continuous_confidence_weighted_B4", float(value))
            for value in continuous_strengths
        ),
    )


def calibrate_confidence_alpha_methods(
    cases: Sequence[ConfidenceAlphaCase],
    *,
    scale_quantiles: Sequence[float] = SCALE_QUANTILES,
    continuous_strengths: Sequence[float] = CONTINUOUS_STRENGTHS,
    binary_thresholds: Sequence[float] = BINARY_CONFIDENCE_THRESHOLDS,
    sample_count: int = SURFACE_SAMPLE_COUNT,
) -> tuple[SelectedMethod, ...]:
    if not cases:
        raise ValueError("calibration cases must not be empty")
    candidates: defaultdict[str, list[CalibrationCandidate]] = defaultdict(list)
    for method_id, hyperparameter in _configuration_grid(
        continuous_strengths=continuous_strengths,
        binary_thresholds=binary_thresholds,
    ):
        prepared = tuple(_prepare(case, method_id, hyperparameter) for case in cases)
        pooled_scores = np.concatenate(
            tuple(row.filtration.scores for row in prepared)
        )
        for quantile in scale_quantiles:
            threshold = float(np.quantile(pooled_scores, quantile))
            endpoints = tuple(
                _endpoint(row, threshold, sample_count=sample_count)
                for row in prepared
            )
            candidates[method_id].append(
                CalibrationCandidate(
                    method_id=method_id,
                    hyperparameter=hyperparameter,
                    scale_quantile=float(quantile),
                    score_threshold=threshold,
                    mean_geometry_loss=float(
                        np.mean([_geometry(endpoint) for endpoint in endpoints])
                    ),
                    mean_betti_error=float(
                        np.mean([float(endpoint.betti_error) for endpoint in endpoints])
                    ),
                    mean_objective=float(
                        np.mean([_objective(endpoint) for endpoint in endpoints])
                    ),
                    mean_fscore=float(
                        np.mean([endpoint.fscore for endpoint in endpoints])
                    ),
                )
            )
    selected = []
    for method_id in METHOD_ORDER:
        row = min(
            candidates[method_id],
            key=lambda item: (
                item.mean_objective,
                item.hyperparameter,
                item.score_threshold,
            ),
        )
        selected.append(
            SelectedMethod(
                method_id=row.method_id,
                hyperparameter=row.hyperparameter,
                scale_quantile=row.scale_quantile,
                score_threshold=row.score_threshold,
                calibration_mean_geometry_loss=row.mean_geometry_loss,
                calibration_mean_betti_error=row.mean_betti_error,
                calibration_mean_objective=row.mean_objective,
                calibration_mean_fscore=row.mean_fscore,
            )
        )
    return tuple(selected)


def _case_result(
    prepared: PreparedFiltration,
    selected: SelectedMethod,
    *,
    sample_count: int,
) -> ConfidenceAlphaCaseResult:
    endpoint = _endpoint(
        prepared,
        selected.score_threshold,
        sample_count=sample_count,
    )
    assert endpoint.betti_error is not None
    return ConfidenceAlphaCaseResult(
        case_id=prepared.case.case_id,
        family=prepared.case.family.value,
        profile=prepared.case.profile.value,
        seed=prepared.case.seed,
        method_id=selected.method_id,
        hyperparameter=selected.hyperparameter,
        score_threshold=selected.score_threshold,
        geometry_loss=_geometry(endpoint),
        betti_error=endpoint.betti_error,
        objective=_objective(endpoint),
        fscore=endpoint.fscore,
        precision=endpoint.precision,
        recall=endpoint.recall,
        connected_components=endpoint.connected_components,
        betti_0=endpoint.betti_0,
        betti_1=endpoint.betti_1,
        betti_2=endpoint.betti_2,
        faces=endpoint.faces,
        nonmanifold_edges=endpoint.nonmanifold_edges,
        retained_target_fraction=prepared.retained_target_fraction,
        target_confidence_mean=prepared.target_confidence_mean,
    )


def _stability(rows: Sequence[ConfidenceAlphaCaseResult]) -> float:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.family, row.profile)].append(row.objective)
    return float(np.mean([np.std(values) for values in grouped.values()]))


def _summary(
    method_id: str,
    rows: Sequence[ConfidenceAlphaCaseResult],
) -> MethodSummary:
    selected = tuple(row for row in rows if row.method_id == method_id)
    nonmanifold_fraction = [
        row.nonmanifold_edges / max(row.faces, 1) for row in selected
    ]
    return MethodSummary(
        method_id=method_id,
        case_count=len(selected),
        mean_geometry_loss=float(np.mean([row.geometry_loss for row in selected])),
        mean_betti_error=float(np.mean([row.betti_error for row in selected])),
        mean_objective=float(np.mean([row.objective for row in selected])),
        mean_fscore=float(np.mean([row.fscore for row in selected])),
        mean_precision=float(np.mean([row.precision for row in selected])),
        mean_recall=float(np.mean([row.recall for row in selected])),
        mean_nonmanifold_fraction=float(np.mean(nonmanifold_fraction)),
        mean_retained_target_fraction=float(
            np.mean([row.retained_target_fraction for row in selected])
        ),
        repeat_stability=_stability(selected),
    )


def evaluate_confidence_alpha_benchmark(
    protocol_path: str | Path,
    *,
    calibration_cases: Sequence[ConfidenceAlphaCase] | None = None,
    held_out_cases: Sequence[ConfidenceAlphaCase] | None = None,
    scale_quantiles: Sequence[float] = SCALE_QUANTILES,
    continuous_strengths: Sequence[float] = CONTINUOUS_STRENGTHS,
    binary_thresholds: Sequence[float] = BINARY_CONFIDENCE_THRESHOLDS,
    sample_count: int = SURFACE_SAMPLE_COUNT,
) -> ConfidenceAlphaBenchmarkResult:
    protocol = Path(protocol_path)
    verify_protocol(protocol)
    calibration = tuple(
        make_confidence_alpha_panel(PanelSplit.CALIBRATION)
        if calibration_cases is None
        else calibration_cases
    )
    held_out = tuple(
        make_confidence_alpha_panel(PanelSplit.HELD_OUT)
        if held_out_cases is None
        else held_out_cases
    )
    selected = calibrate_confidence_alpha_methods(
        calibration,
        scale_quantiles=scale_quantiles,
        continuous_strengths=continuous_strengths,
        binary_thresholds=binary_thresholds,
        sample_count=sample_count,
    )
    rows = []
    for method in selected:
        for case in held_out:
            prepared = _prepare(case, method.method_id, method.hyperparameter)
            rows.append(_case_result(prepared, method, sample_count=sample_count))
    cases = tuple(rows)
    summaries = tuple(_summary(method_id, cases) for method_id in METHOD_ORDER)
    by_method = {summary.method_id: summary for summary in summaries}
    continuous = by_method["continuous_confidence_weighted_B4"]
    fused = by_method["fused_density_B4"]
    binary = by_method["binary_confidence_deletion"]
    anchor = by_method["anchor_density_B4"]
    geometry_gate = continuous.mean_geometry_loss < min(
        fused.mean_geometry_loss, binary.mean_geometry_loss
    )
    topology_gate = continuous.mean_betti_error <= min(
        fused.mean_betti_error, binary.mean_betti_error
    )
    stability_gate = continuous.repeat_stability <= min(
        fused.repeat_stability, binary.repeat_stability
    )
    supported = geometry_gate and topology_gate and stability_gate
    return ConfidenceAlphaBenchmarkResult(
        artifact_schema=RESULT_SCHEMA,
        protocol_path=str(protocol),
        protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        calibration_case_count=len(calibration),
        held_out_case_count=len(held_out),
        selected_methods=selected,
        held_out_cases=cases,
        summaries=summaries,
        geometry_gate_passed=geometry_gate,
        topology_gate_passed=topology_gate,
        stability_gate_passed=stability_gate,
        continuous_confidence_weighting_supported=supported,
        bounded_simulated_confidence_filtration_supported=supported,
        anchor_objective_dominance_supported=(
            continuous.mean_objective < anchor.mean_objective
        ),
        calibration_scale_boundary_reached=any(
            method.scale_quantile == max(scale_quantiles) for method in selected
        ),
        point_local_alpha_field_supported=False,
        topology_correctness_supported=(
            supported and continuous.mean_betti_error == 0.0
        ),
        classical_spatial_alpha_complex_supported=False,
        pftf_predicts_one_global_alpha_supported=False,
        claim_boundary=(
            "The tested score is a closure-preserving confidence-weighted "
            "adaptive filtration, not a classical spatially varying alpha complex."
        ),
    )


def write_result(result: ConfidenceAlphaBenchmarkResult, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")
    return _sha256(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmark-out/confidence_alpha_panel_protocol_phase43.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/confidence_alpha_benchmark_phase43.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_confidence_alpha_benchmark(args.protocol)
    digest = write_result(result, args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    for summary in result.summaries:
        print(
            summary.method_id,
            f"geometry={summary.mean_geometry_loss:.6f}",
            f"betti={summary.mean_betti_error:.6f}",
            f"objective={summary.mean_objective:.6f}",
            f"stability={summary.repeat_stability:.6f}",
        )
    print(
        "continuous_confidence_weighting_supported="
        f"{result.continuous_confidence_weighting_supported}"
    )


if __name__ == "__main__":
    main()
