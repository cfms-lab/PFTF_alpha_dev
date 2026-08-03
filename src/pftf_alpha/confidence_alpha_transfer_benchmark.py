"""Phase-44 reference-free critical-gap transfer benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .adaptive import AdaptiveCellFiltration
from .confidence_alpha_benchmark import (
    B5_MAX_NORMAL_PENALTY,
    METHOD_ORDER,
    _endpoint,
    _geometry,
    _objective,
    _prepare,
)
from .confidence_alpha_transfer_panel import (
    BINARY_CONFIDENCE_THRESHOLDS,
    CONTINUOUS_STRENGTHS,
    MAXIMUM_SELECTED_CELL_FRACTION,
    MINIMUM_JOINT_WIN_FRACTION,
    MINIMUM_SELECTED_CELL_FRACTION,
    PROTOCOL_SCHEMA,
    SURFACE_SAMPLE_COUNT,
    ConfidenceAlphaTransferCase,
    make_confidence_alpha_transfer_panel,
)
from .synthetic import PanelSplit

RESULT_SCHEMA = "pftf_alpha_confidence_alpha_transfer_benchmark_phase44/v1"
EXPECTED_PROTOCOL_SHA256 = (
    "2bdd309855500e2e0bced3701ae4fffa3741358513d6e0d4a5310ed269a5f5a3"
)


@dataclass(frozen=True)
class CriticalGapSelection:
    threshold: float
    lower_critical_score: float
    upper_critical_score: float
    log_score_gap: float
    selected_cell_count: int
    selected_cell_fraction: float
    unique_critical_score_count: int


@dataclass(frozen=True)
class TransferSelectedMethod:
    method_id: str
    hyperparameter: float
    calibration_mean_geometry_loss: float
    calibration_mean_betti_error: float
    calibration_mean_objective: float
    calibration_mean_fscore: float
    calibration_mean_selected_cell_fraction: float
    calibration_mean_log_score_gap: float


@dataclass(frozen=True)
class TransferCaseResult:
    case_id: str
    family: str
    profile: str
    seed: int
    method_id: str
    hyperparameter: float
    score_threshold: float
    selected_cell_fraction: float
    log_score_gap: float
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
class TransferMethodSummary:
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
    mean_selected_cell_fraction: float
    repeat_stability: float


@dataclass(frozen=True)
class ConfidenceAlphaTransferBenchmarkResult:
    artifact_schema: str
    protocol_path: str
    protocol_sha256: str
    calibration_case_count: int
    held_out_case_count: int
    selected_methods: tuple[TransferSelectedMethod, ...]
    held_out_cases: tuple[TransferCaseResult, ...]
    summaries: tuple[TransferMethodSummary, ...]
    geometry_gate_passed: bool
    objective_gate_passed: bool
    topology_gate_passed: bool
    stability_gate_passed: bool
    b5_novelty_gate_passed: bool
    casewise_joint_win_count: int
    minimum_casewise_joint_win_count: int
    casewise_gate_passed: bool
    bounded_confidence_filtration_transfer_supported: bool
    point_local_alpha_field_supported: bool
    topology_correctness_supported: bool
    real_scan_transfer_supported: bool
    deployment_supported: bool
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
        raise ValueError("Phase-44 protocol SHA-256 mismatch")
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    if payload.get("artifact_schema") != PROTOCOL_SCHEMA:
        raise ValueError("Phase-44 protocol schema mismatch")


def complete_critical_gap_threshold(
    scores_or_filtration: ArrayLike | AdaptiveCellFiltration,
    *,
    minimum_selected_fraction: float = MINIMUM_SELECTED_CELL_FRACTION,
    maximum_selected_fraction: float = MAXIMUM_SELECTED_CELL_FRACTION,
) -> CriticalGapSelection:
    """Select the longest eligible adjacent log gap over all critical scores."""

    if not (
        math.isfinite(minimum_selected_fraction)
        and math.isfinite(maximum_selected_fraction)
        and 0.0 < minimum_selected_fraction < maximum_selected_fraction < 1.0
    ):
        raise ValueError("selected fractions must satisfy 0 < minimum < maximum < 1")
    raw_scores = (
        scores_or_filtration.scores
        if isinstance(scores_or_filtration, AdaptiveCellFiltration)
        else scores_or_filtration
    )
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.shape[0] < 2:
        raise ValueError(
            "critical scores must be a one-dimensional array of length >= 2"
        )
    if not np.all(np.isfinite(scores)) or np.any(scores <= 0.0):
        raise ValueError("critical scores must be finite and strictly positive")
    unique = np.unique(scores)
    if unique.shape[0] < 2:
        raise ValueError("at least two unique critical scores are required")

    candidates: list[tuple[float, int, float, float, float]] = []
    for index in range(unique.shape[0] - 1):
        lower = float(unique[index])
        upper = float(unique[index + 1])
        selected_count = int(np.count_nonzero(scores <= lower))
        selected_fraction = selected_count / scores.shape[0]
        if not (
            minimum_selected_fraction
            <= selected_fraction
            <= maximum_selected_fraction
        ):
            continue
        log_gap = math.log(upper) - math.log(lower)
        threshold = math.sqrt(lower * upper)
        candidates.append(
            (log_gap, selected_count, selected_fraction, threshold, lower)
        )
    if not candidates:
        raise ValueError("no adjacent critical-score gap lies in the eligible interval")
    log_gap, selected_count, selected_fraction, threshold, lower = max(
        candidates,
        key=lambda row: (row[0], -row[2], -row[3]),
    )
    lower_index = int(np.searchsorted(unique, lower))
    upper = float(unique[lower_index + 1])
    return CriticalGapSelection(
        threshold=threshold,
        lower_critical_score=lower,
        upper_critical_score=upper,
        log_score_gap=log_gap,
        selected_cell_count=selected_count,
        selected_cell_fraction=selected_fraction,
        unique_critical_score_count=int(unique.shape[0]),
    )


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


def calibrate_transfer_methods(
    cases: Sequence[ConfidenceAlphaTransferCase],
    *,
    continuous_strengths: Sequence[float] = CONTINUOUS_STRENGTHS,
    binary_thresholds: Sequence[float] = BINARY_CONFIDENCE_THRESHOLDS,
    sample_count: int = SURFACE_SAMPLE_COUNT,
) -> tuple[TransferSelectedMethod, ...]:
    if not cases:
        raise ValueError("calibration cases must not be empty")
    candidates: defaultdict[str, list[TransferSelectedMethod]] = defaultdict(list)
    for method_id, hyperparameter in _configuration_grid(
        continuous_strengths=continuous_strengths,
        binary_thresholds=binary_thresholds,
    ):
        endpoints = []
        selections = []
        for case in cases:
            prepared = _prepare(case, method_id, hyperparameter)
            selection = complete_critical_gap_threshold(prepared.filtration)
            endpoints.append(
                _endpoint(prepared, selection.threshold, sample_count=sample_count)
            )
            selections.append(selection)
        candidates[method_id].append(
            TransferSelectedMethod(
                method_id=method_id,
                hyperparameter=hyperparameter,
                calibration_mean_geometry_loss=float(
                    np.mean([_geometry(endpoint) for endpoint in endpoints])
                ),
                calibration_mean_betti_error=float(
                    np.mean([float(endpoint.betti_error) for endpoint in endpoints])
                ),
                calibration_mean_objective=float(
                    np.mean([_objective(endpoint) for endpoint in endpoints])
                ),
                calibration_mean_fscore=float(
                    np.mean([endpoint.fscore for endpoint in endpoints])
                ),
                calibration_mean_selected_cell_fraction=float(
                    np.mean([row.selected_cell_fraction for row in selections])
                ),
                calibration_mean_log_score_gap=float(
                    np.mean([row.log_score_gap for row in selections])
                ),
            )
        )
    return tuple(
        min(
            candidates[method_id],
            key=lambda row: (row.calibration_mean_objective, row.hyperparameter),
        )
        for method_id in METHOD_ORDER
    )


def _held_out_case_result(
    case: ConfidenceAlphaTransferCase,
    method: TransferSelectedMethod,
    *,
    sample_count: int,
) -> TransferCaseResult:
    prepared = _prepare(case, method.method_id, method.hyperparameter)
    selection = complete_critical_gap_threshold(prepared.filtration)
    endpoint = _endpoint(
        prepared,
        selection.threshold,
        sample_count=sample_count,
    )
    assert endpoint.betti_error is not None
    return TransferCaseResult(
        case_id=case.case_id,
        family=case.family.value,
        profile=case.profile.value,
        seed=case.seed,
        method_id=method.method_id,
        hyperparameter=method.hyperparameter,
        score_threshold=selection.threshold,
        selected_cell_fraction=selection.selected_cell_fraction,
        log_score_gap=selection.log_score_gap,
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


def _repeat_stability(rows: Sequence[TransferCaseResult]) -> float:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.family, row.profile)].append(row.objective)
    return float(np.mean([np.std(values) for values in grouped.values()]))


def _summary(
    method_id: str,
    rows: Sequence[TransferCaseResult],
) -> TransferMethodSummary:
    selected = tuple(row for row in rows if row.method_id == method_id)
    return TransferMethodSummary(
        method_id=method_id,
        case_count=len(selected),
        mean_geometry_loss=float(np.mean([row.geometry_loss for row in selected])),
        mean_betti_error=float(np.mean([row.betti_error for row in selected])),
        mean_objective=float(np.mean([row.objective for row in selected])),
        mean_fscore=float(np.mean([row.fscore for row in selected])),
        mean_precision=float(np.mean([row.precision for row in selected])),
        mean_recall=float(np.mean([row.recall for row in selected])),
        mean_nonmanifold_fraction=float(
            np.mean(
                [row.nonmanifold_edges / max(row.faces, 1) for row in selected]
            )
        ),
        mean_retained_target_fraction=float(
            np.mean([row.retained_target_fraction for row in selected])
        ),
        mean_selected_cell_fraction=float(
            np.mean([row.selected_cell_fraction for row in selected])
        ),
        repeat_stability=_repeat_stability(selected),
    )


def evaluate_confidence_alpha_transfer_benchmark(
    protocol_path: str | Path,
    *,
    calibration_cases: Sequence[ConfidenceAlphaTransferCase] | None = None,
    held_out_cases: Sequence[ConfidenceAlphaTransferCase] | None = None,
    continuous_strengths: Sequence[float] = CONTINUOUS_STRENGTHS,
    binary_thresholds: Sequence[float] = BINARY_CONFIDENCE_THRESHOLDS,
    sample_count: int = SURFACE_SAMPLE_COUNT,
) -> ConfidenceAlphaTransferBenchmarkResult:
    protocol = Path(protocol_path)
    verify_protocol(protocol)
    calibration = tuple(
        make_confidence_alpha_transfer_panel(PanelSplit.CALIBRATION)
        if calibration_cases is None
        else calibration_cases
    )
    held_out = tuple(
        make_confidence_alpha_transfer_panel(PanelSplit.HELD_OUT)
        if held_out_cases is None
        else held_out_cases
    )
    selected = calibrate_transfer_methods(
        calibration,
        continuous_strengths=continuous_strengths,
        binary_thresholds=binary_thresholds,
        sample_count=sample_count,
    )
    cases = tuple(
        _held_out_case_result(case, method, sample_count=sample_count)
        for method in selected
        for case in held_out
    )
    summaries = tuple(_summary(method_id, cases) for method_id in METHOD_ORDER)
    by_method = {row.method_id: row for row in summaries}
    anchor = by_method["anchor_density_B4"]
    fused = by_method["fused_density_B4"]
    b5 = by_method["fused_pca_B5"]
    binary = by_method["binary_confidence_deletion"]
    continuous = by_method["continuous_confidence_weighted_B4"]
    geometry_gate = continuous.mean_geometry_loss < min(
        anchor.mean_geometry_loss,
        fused.mean_geometry_loss,
        binary.mean_geometry_loss,
    )
    objective_gate = continuous.mean_objective < min(
        anchor.mean_objective,
        fused.mean_objective,
        binary.mean_objective,
    )
    topology_gate = continuous.mean_betti_error <= min(
        anchor.mean_betti_error,
        fused.mean_betti_error,
        binary.mean_betti_error,
    )
    stability_gate = continuous.repeat_stability <= min(
        fused.repeat_stability,
        binary.repeat_stability,
    )
    b5_gate = continuous.mean_objective < b5.mean_objective
    objective_by_method_case = {
        (row.method_id, row.case_id): row.objective for row in cases
    }
    joint_win_count = sum(
        objective_by_method_case[("continuous_confidence_weighted_B4", case.case_id)]
        < min(
            objective_by_method_case[("anchor_density_B4", case.case_id)],
            objective_by_method_case[("fused_density_B4", case.case_id)],
            objective_by_method_case[("binary_confidence_deletion", case.case_id)],
        )
        for case in held_out
    )
    minimum_joint_wins = math.ceil(MINIMUM_JOINT_WIN_FRACTION * len(held_out))
    casewise_gate = joint_win_count >= minimum_joint_wins
    supported = all(
        (
            geometry_gate,
            objective_gate,
            topology_gate,
            stability_gate,
            b5_gate,
            casewise_gate,
        )
    )
    return ConfidenceAlphaTransferBenchmarkResult(
        artifact_schema=RESULT_SCHEMA,
        protocol_path=str(protocol),
        protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        calibration_case_count=len(calibration),
        held_out_case_count=len(held_out),
        selected_methods=selected,
        held_out_cases=cases,
        summaries=summaries,
        geometry_gate_passed=geometry_gate,
        objective_gate_passed=objective_gate,
        topology_gate_passed=topology_gate,
        stability_gate_passed=stability_gate,
        b5_novelty_gate_passed=b5_gate,
        casewise_joint_win_count=joint_win_count,
        minimum_casewise_joint_win_count=minimum_joint_wins,
        casewise_gate_passed=casewise_gate,
        bounded_confidence_filtration_transfer_supported=supported,
        point_local_alpha_field_supported=False,
        topology_correctness_supported=(
            supported and continuous.mean_betti_error == 0.0
        ),
        real_scan_transfer_supported=False,
        deployment_supported=False,
        claim_boundary=(
            "This is a simulated transfer test of a confidence-weighted top-cell "
            "filtration, not a classical local-alpha complex or real deployment."
        ),
    )


def write_result(
    result: ConfidenceAlphaTransferBenchmarkResult,
    path: str | Path,
) -> str:
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
        default=Path(
            "benchmark-out/confidence_alpha_transfer_protocol_phase44.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark-out/confidence_alpha_transfer_benchmark_phase44.json"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_confidence_alpha_transfer_benchmark(args.protocol)
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
            f"selected={summary.mean_selected_cell_fraction:.6f}",
        )
    print(
        "bounded_confidence_filtration_transfer_supported="
        f"{result.bounded_confidence_filtration_transfer_supported}"
    )


if __name__ == "__main__":
    main()
