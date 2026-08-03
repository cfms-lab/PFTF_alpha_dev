"""Frozen Phase-45 confidence-aware regular/power alpha benchmark."""

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

from .confidence_alpha_benchmark import _endpoint, _geometry, _objective, _prepare
from .confidence_alpha_field import observed_point_confidence
from .confidence_alpha_transfer_benchmark import complete_critical_gap_threshold
from .confidence_alpha_transfer_panel import ConfidenceAlphaTransferCase
from .confidence_power_alpha import confidence_power_alpha_filtration
from .confidence_power_alpha_protocol import (
    CONFIDENCE_PENALTY_SCALES,
    FROZEN_BINARY_CONFIDENCE_THRESHOLD,
    FROZEN_CONTINUOUS_PENALTY_STRENGTH,
    M1_DENSITY_WEIGHT_SCALE,
    MAXIMUM_FALLBACK_FRACTION,
    MINIMUM_CASEWISE_JOINT_WIN_FRACTION,
    MINIMUM_CONNECTIVITY_CHANGE_FRACTION,
    PROTOCOL_SCHEMA,
    SURFACE_SAMPLE_COUNT,
    make_confidence_power_alpha_panel,
)
from .synthetic import PanelSplit
from .weighted_alpha import PointSubmersionError, weighted_alpha_filtration

RESULT_SCHEMA = "pftf_alpha_confidence_power_alpha_benchmark_phase45/v1"
EXPECTED_PROTOCOL_SHA256 = (
    "554b214bbc664041661634e8315c7bd56d87f10fb3825e4a410ad4e765cbe414"
)
K_NEIGHBORS = 12
METHOD_ORDER = (
    "anchor_density_B4",
    "fused_density_B4",
    "fused_pca_B5",
    "M1_density_power_alpha",
    "binary_confidence_deletion",
    "fixed_cell_continuous_confidence",
    "confidence_power_alpha",
)


@dataclass(frozen=True)
class PreparedPowerMethod:
    case: ConfidenceAlphaTransferCase
    method_id: str
    hyperparameter: float
    filtration: object
    point_component_labels: np.ndarray
    retained_target_fraction: float
    target_confidence_mean: float
    fallback_to_m1: bool
    connectivity_changed_from_m1: bool
    connectivity_jaccard_distance: float


@dataclass(frozen=True)
class PenaltyCalibrationResult:
    confidence_penalty_scale: float
    submerged_case_count: int
    valid: bool
    mean_geometry_loss: float | None
    mean_betti_error: float | None
    mean_objective: float | None
    mean_fscore: float | None
    mean_selected_cell_fraction: float | None


@dataclass(frozen=True)
class PowerAlphaCaseResult:
    case_id: str
    family: str
    profile: str
    seed: int
    method_id: str
    hyperparameter: float
    score_threshold: float
    selected_cell_fraction: float
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
    fallback_to_m1: bool
    connectivity_changed_from_m1: bool
    connectivity_jaccard_distance: float


@dataclass(frozen=True)
class PowerAlphaMethodSummary:
    method_id: str
    case_count: int
    mean_geometry_loss: float
    mean_betti_error: float
    mean_objective: float
    mean_fscore: float
    mean_precision: float
    mean_recall: float
    mean_selected_cell_fraction: float
    repeat_stability: float
    fallback_fraction: float
    connectivity_change_fraction: float
    mean_connectivity_jaccard_distance: float


@dataclass(frozen=True)
class ConfidencePowerAlphaBenchmarkResult:
    artifact_schema: str
    protocol_path: str
    protocol_sha256: str
    calibration_case_count: int
    held_out_case_count: int
    penalty_calibration: tuple[PenaltyCalibrationResult, ...]
    selected_confidence_penalty_scale: float
    held_out_cases: tuple[PowerAlphaCaseResult, ...]
    summaries: tuple[PowerAlphaMethodSummary, ...]
    geometry_gate_passed: bool
    objective_gate_passed: bool
    topology_gate_passed: bool
    stability_gate_passed: bool
    b5_novelty_gate_passed: bool
    casewise_joint_win_count: int
    minimum_casewise_joint_win_count: int
    casewise_gate_passed: bool
    connectivity_change_case_count: int
    minimum_connectivity_change_case_count: int
    connectivity_gate_passed: bool
    fallback_case_count: int
    maximum_fallback_case_count: int
    fallback_gate_passed: bool
    confidence_power_alpha_supported: bool
    exact_weighted_alpha_supported: bool
    pftf_trained_alpha_supported: bool
    point_local_alpha_field_supported: bool
    topology_correctness_supported: bool
    real_scan_transfer_supported: bool
    deployment_supported: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["penalty_calibration"] = [
            asdict(row) for row in self.penalty_calibration
        ]
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
        raise ValueError("Phase-45 protocol SHA-256 mismatch")
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    if payload.get("artifact_schema") != PROTOCOL_SCHEMA:
        raise ValueError("Phase-45 protocol schema mismatch")


def _canonical_cells(cells: np.ndarray) -> set[tuple[int, int, int, int]]:
    return {tuple(sorted(int(value) for value in cell)) for cell in cells}


def _connectivity_distance(left: np.ndarray, right: np.ndarray) -> float:
    first = _canonical_cells(left)
    second = _canonical_cells(right)
    union = first | second
    return 0.0 if not union else 1.0 - len(first & second) / len(union)


def _wrap_prepared(
    case: ConfidenceAlphaTransferCase,
    method_id: str,
    hyperparameter: float,
    prepared,
) -> PreparedPowerMethod:
    return PreparedPowerMethod(
        case=case,
        method_id=method_id,
        hyperparameter=hyperparameter,
        filtration=prepared.filtration,
        point_component_labels=prepared.point_component_labels,
        retained_target_fraction=prepared.retained_target_fraction,
        target_confidence_mean=prepared.target_confidence_mean,
        fallback_to_m1=False,
        connectivity_changed_from_m1=False,
        connectivity_jaccard_distance=0.0,
    )


def _prepare_m1(case: ConfidenceAlphaTransferCase) -> PreparedPowerMethod:
    confidence = observed_point_confidence(
        case.anchor_points, case.target_points, k_neighbors=K_NEIGHBORS
    )
    filtration = weighted_alpha_filtration(
        case.points,
        k_neighbors=K_NEIGHBORS,
        weight_scale=M1_DENSITY_WEIGHT_SCALE,
    )
    return PreparedPowerMethod(
        case=case,
        method_id="M1_density_power_alpha",
        hyperparameter=M1_DENSITY_WEIGHT_SCALE,
        filtration=filtration,
        point_component_labels=case.point_component_labels,
        retained_target_fraction=1.0,
        target_confidence_mean=float(np.mean(confidence.target_confidence)),
        fallback_to_m1=False,
        connectivity_changed_from_m1=False,
        connectivity_jaccard_distance=0.0,
    )


def _prepare_confidence_power(
    case: ConfidenceAlphaTransferCase,
    penalty_scale: float,
    *,
    allow_fallback: bool,
) -> PreparedPowerMethod:
    confidence = observed_point_confidence(
        case.anchor_points, case.target_points, k_neighbors=K_NEIGHBORS
    )
    m1 = weighted_alpha_filtration(
        case.points,
        k_neighbors=K_NEIGHBORS,
        weight_scale=M1_DENSITY_WEIGHT_SCALE,
    )
    try:
        filtration = confidence_power_alpha_filtration(
            case.points,
            confidence.point_confidence,
            k_neighbors=K_NEIGHBORS,
            density_weight_scale=M1_DENSITY_WEIGHT_SCALE,
            confidence_penalty_scale=penalty_scale,
        )
        fallback = False
    except PointSubmersionError:
        if not allow_fallback:
            raise
        filtration = m1
        fallback = True
    distance = _connectivity_distance(
        filtration.top_simplices, m1.top_simplices
    )
    return PreparedPowerMethod(
        case=case,
        method_id="confidence_power_alpha",
        hyperparameter=float(penalty_scale),
        filtration=filtration,
        point_component_labels=case.point_component_labels,
        retained_target_fraction=1.0,
        target_confidence_mean=float(np.mean(confidence.target_confidence)),
        fallback_to_m1=fallback,
        connectivity_changed_from_m1=distance > 0.0,
        connectivity_jaccard_distance=distance,
    )


def _prepare_method(
    case: ConfidenceAlphaTransferCase,
    method_id: str,
    selected_penalty_scale: float,
) -> PreparedPowerMethod:
    if method_id == "M1_density_power_alpha":
        return _prepare_m1(case)
    if method_id == "confidence_power_alpha":
        return _prepare_confidence_power(
            case, selected_penalty_scale, allow_fallback=True
        )
    if method_id == "fixed_cell_continuous_confidence":
        prepared = _prepare(
            case,
            "continuous_confidence_weighted_B4",
            FROZEN_CONTINUOUS_PENALTY_STRENGTH,
        )
        return _wrap_prepared(
            case,
            method_id,
            FROZEN_CONTINUOUS_PENALTY_STRENGTH,
            prepared,
        )
    hyperparameter = (
        FROZEN_BINARY_CONFIDENCE_THRESHOLD
        if method_id == "binary_confidence_deletion"
        else 0.0
    )
    prepared = _prepare(case, method_id, hyperparameter)
    return _wrap_prepared(case, method_id, hyperparameter, prepared)


def calibrate_confidence_power_penalty(
    cases: Sequence[ConfidenceAlphaTransferCase],
    *,
    penalty_scales: Sequence[float] = CONFIDENCE_PENALTY_SCALES,
    sample_count: int = SURFACE_SAMPLE_COUNT,
) -> tuple[tuple[PenaltyCalibrationResult, ...], float]:
    if not cases:
        raise ValueError("calibration cases must not be empty")
    rows = []
    for penalty_scale in penalty_scales:
        prepared_rows = []
        submerged = 0
        for case in cases:
            try:
                prepared_rows.append(
                    _prepare_confidence_power(
                        case, float(penalty_scale), allow_fallback=False
                    )
                )
            except PointSubmersionError:
                submerged += 1
        if submerged:
            rows.append(
                PenaltyCalibrationResult(
                    confidence_penalty_scale=float(penalty_scale),
                    submerged_case_count=submerged,
                    valid=False,
                    mean_geometry_loss=None,
                    mean_betti_error=None,
                    mean_objective=None,
                    mean_fscore=None,
                    mean_selected_cell_fraction=None,
                )
            )
            continue
        endpoints = []
        selected_fractions = []
        for prepared in prepared_rows:
            selection = complete_critical_gap_threshold(prepared.filtration)
            endpoints.append(
                _endpoint(prepared, selection.threshold, sample_count=sample_count)
            )
            selected_fractions.append(selection.selected_cell_fraction)
        rows.append(
            PenaltyCalibrationResult(
                confidence_penalty_scale=float(penalty_scale),
                submerged_case_count=0,
                valid=True,
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
                mean_selected_cell_fraction=float(np.mean(selected_fractions)),
            )
        )
    valid_rows = [row for row in rows if row.valid]
    if not valid_rows:
        raise ValueError("every Phase-45 confidence penalty submerges points")
    selected = min(
        valid_rows,
        key=lambda row: (
            float(row.mean_objective),
            row.confidence_penalty_scale,
        ),
    )
    return tuple(rows), selected.confidence_penalty_scale


def _case_result(
    prepared: PreparedPowerMethod,
    *,
    sample_count: int,
) -> PowerAlphaCaseResult:
    selection = complete_critical_gap_threshold(prepared.filtration)
    endpoint = _endpoint(
        prepared,
        selection.threshold,
        sample_count=sample_count,
    )
    assert endpoint.betti_error is not None
    case = prepared.case
    return PowerAlphaCaseResult(
        case_id=case.case_id,
        family=case.family.value,
        profile=case.profile.value,
        seed=case.seed,
        method_id=prepared.method_id,
        hyperparameter=prepared.hyperparameter,
        score_threshold=selection.threshold,
        selected_cell_fraction=selection.selected_cell_fraction,
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
        fallback_to_m1=prepared.fallback_to_m1,
        connectivity_changed_from_m1=prepared.connectivity_changed_from_m1,
        connectivity_jaccard_distance=prepared.connectivity_jaccard_distance,
    )


def _repeat_stability(rows: Sequence[PowerAlphaCaseResult]) -> float:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.family, row.profile)].append(row.objective)
    return float(np.mean([np.std(values) for values in grouped.values()]))


def _summary(
    method_id: str,
    rows: Sequence[PowerAlphaCaseResult],
) -> PowerAlphaMethodSummary:
    selected = tuple(row for row in rows if row.method_id == method_id)
    return PowerAlphaMethodSummary(
        method_id=method_id,
        case_count=len(selected),
        mean_geometry_loss=float(np.mean([row.geometry_loss for row in selected])),
        mean_betti_error=float(np.mean([row.betti_error for row in selected])),
        mean_objective=float(np.mean([row.objective for row in selected])),
        mean_fscore=float(np.mean([row.fscore for row in selected])),
        mean_precision=float(np.mean([row.precision for row in selected])),
        mean_recall=float(np.mean([row.recall for row in selected])),
        mean_selected_cell_fraction=float(
            np.mean([row.selected_cell_fraction for row in selected])
        ),
        repeat_stability=_repeat_stability(selected),
        fallback_fraction=float(np.mean([row.fallback_to_m1 for row in selected])),
        connectivity_change_fraction=float(
            np.mean([row.connectivity_changed_from_m1 for row in selected])
        ),
        mean_connectivity_jaccard_distance=float(
            np.mean([row.connectivity_jaccard_distance for row in selected])
        ),
    )


def evaluate_confidence_power_alpha_benchmark(
    protocol_path: str | Path,
    *,
    calibration_cases: Sequence[ConfidenceAlphaTransferCase] | None = None,
    held_out_cases: Sequence[ConfidenceAlphaTransferCase] | None = None,
    penalty_scales: Sequence[float] = CONFIDENCE_PENALTY_SCALES,
    sample_count: int = SURFACE_SAMPLE_COUNT,
) -> ConfidencePowerAlphaBenchmarkResult:
    protocol = Path(protocol_path)
    verify_protocol(protocol)
    calibration = tuple(
        make_confidence_power_alpha_panel(PanelSplit.CALIBRATION)
        if calibration_cases is None
        else calibration_cases
    )
    held_out = tuple(
        make_confidence_power_alpha_panel(PanelSplit.HELD_OUT)
        if held_out_cases is None
        else held_out_cases
    )
    penalty_rows, selected_penalty = calibrate_confidence_power_penalty(
        calibration,
        penalty_scales=penalty_scales,
        sample_count=sample_count,
    )
    cases = tuple(
        _case_result(
            _prepare_method(case, method_id, selected_penalty),
            sample_count=sample_count,
        )
        for method_id in METHOD_ORDER
        for case in held_out
    )
    summaries = tuple(_summary(method_id, cases) for method_id in METHOD_ORDER)
    by_method = {row.method_id: row for row in summaries}
    candidate = by_method["confidence_power_alpha"]
    anchor = by_method["anchor_density_B4"]
    fused = by_method["fused_density_B4"]
    b5 = by_method["fused_pca_B5"]
    m1 = by_method["M1_density_power_alpha"]
    binary = by_method["binary_confidence_deletion"]
    continuous = by_method["fixed_cell_continuous_confidence"]
    primary_comparators = (anchor, fused, m1, binary, continuous)
    geometry_gate = candidate.mean_geometry_loss < min(
        row.mean_geometry_loss for row in primary_comparators
    )
    objective_gate = candidate.mean_objective < min(
        row.mean_objective for row in primary_comparators
    )
    topology_gate = candidate.mean_betti_error <= min(
        row.mean_betti_error for row in primary_comparators
    )
    stability_gate = candidate.repeat_stability <= min(
        m1.repeat_stability,
        continuous.repeat_stability,
    )
    b5_gate = candidate.mean_objective < b5.mean_objective
    objectives = {(row.method_id, row.case_id): row.objective for row in cases}
    joint_wins = sum(
        objectives[("confidence_power_alpha", case.case_id)]
        < min(
            objectives[("M1_density_power_alpha", case.case_id)],
            objectives[("fixed_cell_continuous_confidence", case.case_id)],
        )
        for case in held_out
    )
    minimum_joint_wins = math.ceil(
        MINIMUM_CASEWISE_JOINT_WIN_FRACTION * len(held_out)
    )
    casewise_gate = joint_wins >= minimum_joint_wins
    connectivity_change_count = sum(
        row.connectivity_changed_from_m1
        for row in cases
        if row.method_id == "confidence_power_alpha"
    )
    minimum_connectivity_changes = math.ceil(
        MINIMUM_CONNECTIVITY_CHANGE_FRACTION * len(held_out)
    )
    connectivity_gate = connectivity_change_count >= minimum_connectivity_changes
    fallback_count = sum(
        row.fallback_to_m1
        for row in cases
        if row.method_id == "confidence_power_alpha"
    )
    maximum_fallback_count = math.floor(MAXIMUM_FALLBACK_FRACTION * len(held_out))
    fallback_gate = fallback_count <= maximum_fallback_count
    supported = all(
        (
            geometry_gate,
            objective_gate,
            topology_gate,
            stability_gate,
            b5_gate,
            casewise_gate,
            connectivity_gate,
            fallback_gate,
        )
    )
    return ConfidencePowerAlphaBenchmarkResult(
        artifact_schema=RESULT_SCHEMA,
        protocol_path=str(protocol),
        protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        calibration_case_count=len(calibration),
        held_out_case_count=len(held_out),
        penalty_calibration=penalty_rows,
        selected_confidence_penalty_scale=selected_penalty,
        held_out_cases=cases,
        summaries=summaries,
        geometry_gate_passed=geometry_gate,
        objective_gate_passed=objective_gate,
        topology_gate_passed=topology_gate,
        stability_gate_passed=stability_gate,
        b5_novelty_gate_passed=b5_gate,
        casewise_joint_win_count=joint_wins,
        minimum_casewise_joint_win_count=minimum_joint_wins,
        casewise_gate_passed=casewise_gate,
        connectivity_change_case_count=connectivity_change_count,
        minimum_connectivity_change_case_count=minimum_connectivity_changes,
        connectivity_gate_passed=connectivity_gate,
        fallback_case_count=fallback_count,
        maximum_fallback_case_count=maximum_fallback_count,
        fallback_gate_passed=fallback_gate,
        confidence_power_alpha_supported=supported,
        exact_weighted_alpha_supported=False,
        pftf_trained_alpha_supported=False,
        point_local_alpha_field_supported=False,
        topology_correctness_supported=(
            supported and candidate.mean_betti_error == 0.0
        ),
        real_scan_transfer_supported=False,
        deployment_supported=False,
        claim_boundary=(
            "The candidate changes regular-triangulation connectivity using "
            "observed confidence, but remains floating Qhull and is not a trained "
            "PFTF or local-SPD metric complex."
        ),
    )


def write_result(result: ConfidencePowerAlphaBenchmarkResult, path: str | Path) -> str:
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
        default=Path("benchmark-out/confidence_power_alpha_protocol_phase45.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/confidence_power_alpha_benchmark_phase45.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_confidence_power_alpha_benchmark(args.protocol)
    digest = write_result(result, args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(
        "selected_confidence_penalty_scale="
        f"{result.selected_confidence_penalty_scale}"
    )
    for summary in result.summaries:
        print(
            summary.method_id,
            f"geometry={summary.mean_geometry_loss:.6f}",
            f"betti={summary.mean_betti_error:.6f}",
            f"objective={summary.mean_objective:.6f}",
            f"stability={summary.repeat_stability:.6f}",
        )
    print(f"confidence_power_alpha_supported={result.confidence_power_alpha_supported}")


if __name__ == "__main__":
    main()
