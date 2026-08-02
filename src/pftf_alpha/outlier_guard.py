"""Robust residual outlier guard and frozen Phase-9 held-out evaluation."""

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

from .adaptive import local_neighborhood_geometry
from .sampling_gate import SamplingGateDecision
from .sensor_stress import (
    DEFAULT_POINT_COUNTS,
    DEFAULT_STRESSES,
    SensorStress,
    evaluate_sensor_stress,
    make_sensor_stress_case,
)
from .shared_trend_inference import infer_shared_trend_layers

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class OutlierDensityThreshold:
    maximum_point_count: int | None
    maximum_joint_score: float

    def __post_init__(self) -> None:
        if self.maximum_point_count is not None and self.maximum_point_count < 32:
            raise ValueError("maximum_point_count must be at least 32")
        if not math.isfinite(self.maximum_joint_score):
            raise ValueError("maximum_joint_score must be finite")
        if self.maximum_joint_score <= 0.0:
            raise ValueError("maximum_joint_score must be positive")


DEFAULT_OUTLIER_THRESHOLDS = (
    OutlierDensityThreshold(96, 9.20),
    OutlierDensityThreshold(160, 7.80),
    OutlierDensityThreshold(None, 9.65),
)


@dataclass(frozen=True)
class OutlierGuardConfig:
    k_neighbors: int = 12
    mad_consistency_factor: float = 1.4826
    density_thresholds: tuple[
        OutlierDensityThreshold, ...
    ] = DEFAULT_OUTLIER_THRESHOLDS

    def __post_init__(self) -> None:
        if self.k_neighbors < 3:
            raise ValueError("k_neighbors must be at least three")
        if not math.isfinite(self.mad_consistency_factor):
            raise ValueError("mad_consistency_factor must be finite")
        if self.mad_consistency_factor <= 0.0:
            raise ValueError("mad_consistency_factor must be positive")
        if (
            not self.density_thresholds
            or self.density_thresholds[-1].maximum_point_count is not None
        ):
            raise ValueError("density_thresholds must end with an open upper bin")


@dataclass(frozen=True)
class OutlierEvidence:
    information_boundary: str
    point_count: int
    k_neighbors: int
    robust_residual_scale: float
    maximum_studentized_residual: float
    maximum_local_density_ratio: float
    maximum_joint_score: float
    allowed_maximum_joint_score: float
    outlier_free: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OutlierGuardCaseResult:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    evidence: OutlierEvidence
    unguarded_decision: SamplingGateDecision
    guarded_decision: SamplingGateDecision
    true_safe_output: bool
    unguarded_safe_accept: bool
    guarded_safe_accept: bool
    unguarded_false_safe: bool
    guarded_false_safe: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress"] = self.stress.value
        payload["unguarded_decision"] = self.unguarded_decision.value
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class OutlierGuardSummary:
    group_kind: str
    group_name: str
    case_count: int
    unguarded_safe_accept_count: int
    guarded_safe_accept_count: int
    safe_accept_retention: float | None
    unguarded_false_safe_count: int
    guarded_false_safe_count: int
    group_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OutlierGuardResult:
    artifact_schema: str
    role: str
    information_boundary: str
    calibration_source: str
    seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    guard_config: OutlierGuardConfig
    cases: tuple[OutlierGuardCaseResult, ...]
    stress_summaries: tuple[OutlierGuardSummary, ...]
    density_summaries: tuple[OutlierGuardSummary, ...]
    case_count: int
    unguarded_safe_accept_count: int
    guarded_safe_accept_count: int
    safe_accept_retention: float
    unguarded_false_safe_count: int
    guarded_false_safe_count: int
    removed_false_safe_count: int
    phase9_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "calibration_source": self.calibration_source,
            "seed": self.seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "guard_config": {
                "k_neighbors": self.guard_config.k_neighbors,
                "mad_consistency_factor": (
                    self.guard_config.mad_consistency_factor
                ),
                "density_thresholds": [
                    asdict(row) for row in self.guard_config.density_thresholds
                ],
            },
            "cases": [case.to_dict() for case in self.cases],
            "stress_summaries": [row.to_dict() for row in self.stress_summaries],
            "density_summaries": [row.to_dict() for row in self.density_summaries],
            "case_count": self.case_count,
            "unguarded_safe_accept_count": self.unguarded_safe_accept_count,
            "guarded_safe_accept_count": self.guarded_safe_accept_count,
            "safe_accept_retention": self.safe_accept_retention,
            "unguarded_false_safe_count": self.unguarded_false_safe_count,
            "guarded_false_safe_count": self.guarded_false_safe_count,
            "removed_false_safe_count": self.removed_false_safe_count,
            "phase9_supported": self.phase9_supported,
            "deployment_supported": self.deployment_supported,
        }


def _threshold_for_count(
    point_count: int,
    config: OutlierGuardConfig,
) -> OutlierDensityThreshold:
    return next(
        row
        for row in config.density_thresholds
        if row.maximum_point_count is None or point_count <= row.maximum_point_count
    )


def estimate_outlier_evidence(
    points: FloatArray,
    inferred_labels: IntArray,
    config: OutlierGuardConfig | None = None,
) -> OutlierEvidence:
    """Score shared-trend residuals without using source labels or references."""

    selected = OutlierGuardConfig() if config is None else config
    point_array = np.asarray(points, dtype=np.float64)
    labels = np.asarray(inferred_labels, dtype=np.int64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if labels.shape != (point_array.shape[0],) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("inferred_labels must contain two aligned layers")

    geometry = local_neighborhood_geometry(
        point_array,
        k_neighbors=selected.k_neighbors,
    )
    normals = geometry.eigenvectors[:, :, 0]
    tensor = np.mean(normals[:, :, None] * normals[:, None, :], axis=0)
    frame = np.linalg.eigh(tensor)[1]
    coordinates = (point_array - np.mean(point_array, axis=0)) @ frame
    u, v, height = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    design = np.column_stack(
        (np.ones(point_array.shape[0]), u, v, u * u, u * v, v * v, labels)
    )
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = np.linalg.lstsq(design, height, rcond=None)[0]
    residuals = height - design @ coefficients
    leverage = np.sum((design @ inverse) * design, axis=1)
    centered = residuals - np.median(residuals)
    robust_scale = max(
        selected.mad_consistency_factor * np.median(np.abs(centered)),
        np.finfo(float).eps,
    )
    studentized = np.abs(centered) / (
        robust_scale * np.sqrt(np.maximum(1.0 - leverage, 1e-6))
    )
    selected_k = min(selected.k_neighbors, point_array.shape[0] - 1)
    distances = cKDTree(point_array).query(
        point_array,
        k=selected_k + 1,
        workers=1,
    )[0][:, 1:]
    local_scales = np.median(distances, axis=1)
    density_ratio = local_scales / max(
        float(np.median(local_scales)),
        np.finfo(float).eps,
    )
    joint = studentized * np.sqrt(density_ratio)
    threshold = _threshold_for_count(point_array.shape[0], selected)
    maximum_joint = float(np.max(joint))
    return OutlierEvidence(
        information_boundary="observed_coordinates_and_inferred_layers_only",
        point_count=point_array.shape[0],
        k_neighbors=selected_k,
        robust_residual_scale=float(robust_scale),
        maximum_studentized_residual=float(np.max(studentized)),
        maximum_local_density_ratio=float(np.max(density_ratio)),
        maximum_joint_score=maximum_joint,
        allowed_maximum_joint_score=threshold.maximum_joint_score,
        outlier_free=bool(maximum_joint <= threshold.maximum_joint_score),
    )


def route_with_outlier_guard(
    decision: SamplingGateDecision,
    evidence: OutlierEvidence,
) -> SamplingGateDecision:
    if decision is SamplingGateDecision.ACCEPT and not evidence.outlier_free:
        return SamplingGateDecision.UNSUPPORTED
    return decision


def _summary(
    rows: Sequence[OutlierGuardCaseResult],
    *,
    kind: str,
    name: str,
) -> OutlierGuardSummary:
    unguarded_safe = sum(case.unguarded_safe_accept for case in rows)
    guarded_safe = sum(case.guarded_safe_accept for case in rows)
    retention = None if not unguarded_safe else guarded_safe / unguarded_safe
    guarded_false = sum(case.guarded_false_safe for case in rows)
    return OutlierGuardSummary(
        group_kind=kind,
        group_name=name,
        case_count=len(rows),
        unguarded_safe_accept_count=unguarded_safe,
        guarded_safe_accept_count=guarded_safe,
        safe_accept_retention=retention,
        unguarded_false_safe_count=sum(case.unguarded_false_safe for case in rows),
        guarded_false_safe_count=guarded_false,
        group_gate_passed=bool(
            guarded_false == 0
            and (
                unguarded_safe < 8
                or (retention is not None and retention >= 0.85)
            )
        ),
    )


def evaluate_outlier_guard(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    seed: int = 20700804,
    surface_sample_count: int = 256,
    guard_config: OutlierGuardConfig | None = None,
) -> OutlierGuardResult:
    """Run the frozen Phase-9 outlier-guard held-out panel."""

    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_guard = OutlierGuardConfig() if guard_config is None else guard_config
    base_result = evaluate_sensor_stress(
        point_counts=selected_counts,
        stresses=selected_stresses,
        reference_count=reference_count,
        repeats=repeats,
        seed=seed,
        surface_sample_count=surface_sample_count,
    )
    results: list[OutlierGuardCaseResult] = []
    for case_row in base_result.cases:
        case = make_sensor_stress_case(
            case_row.stress,
            case_row.point_count,
            reference_count=max(reference_count, case_row.point_count),
            seed=case_row.seed,
        )
        inference = infer_shared_trend_layers(case.points)
        evidence = estimate_outlier_evidence(
            case.points,
            inference.inference.layer_ids,
            selected_guard,
        )
        guarded_decision = route_with_outlier_guard(
            case_row.candidate_decision,
            evidence,
        )
        guarded_accept = guarded_decision is SamplingGateDecision.ACCEPT
        results.append(
            OutlierGuardCaseResult(
                stress=case_row.stress,
                point_count=case_row.point_count,
                repeat=case_row.repeat,
                seed=case_row.seed,
                evidence=evidence,
                unguarded_decision=case_row.candidate_decision,
                guarded_decision=guarded_decision,
                true_safe_output=case_row.candidate_true_safe_output,
                unguarded_safe_accept=case_row.candidate_safe_accept,
                guarded_safe_accept=bool(
                    guarded_accept and case_row.candidate_true_safe_output
                ),
                unguarded_false_safe=case_row.candidate_false_safe,
                guarded_false_safe=bool(
                    guarded_accept and not case_row.candidate_true_safe_output
                ),
            )
        )

    stress_summaries = tuple(
        _summary(
            [case for case in results if case.stress is stress],
            kind="stress",
            name=stress.value,
        )
        for stress in selected_stresses
    )
    density_summaries = tuple(
        _summary(
            [case for case in results if case.point_count == point_count],
            kind="density",
            name=str(point_count),
        )
        for point_count in selected_counts
    )
    unguarded_safe = sum(case.unguarded_safe_accept for case in results)
    guarded_safe = sum(case.guarded_safe_accept for case in results)
    unguarded_false = sum(case.unguarded_false_safe for case in results)
    guarded_false = sum(case.guarded_false_safe for case in results)
    retention = 0.0 if not unguarded_safe else guarded_safe / unguarded_safe
    supported = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and repeats >= 8
        and len(results) == 216
        and unguarded_false > 0
        and guarded_false == 0
        and unguarded_false - guarded_false == unguarded_false
        and retention >= 0.90
        and all(row.group_gate_passed for row in stress_summaries)
        and all(row.group_gate_passed for row in density_summaries)
    )
    return OutlierGuardResult(
        artifact_schema="pftf_alpha_outlier_guard_phase9/v1",
        role="robust_residual_density_fail_closed_outlier_guard",
        information_boundary=(
            "guard uses observed coordinates and inferred layers only; stress "
            "identity, true source labels, and references are evaluation-only"
        ),
        calibration_source="phase8_seed_20600804",
        seed=seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        guard_config=selected_guard,
        cases=tuple(results),
        stress_summaries=stress_summaries,
        density_summaries=density_summaries,
        case_count=len(results),
        unguarded_safe_accept_count=unguarded_safe,
        guarded_safe_accept_count=guarded_safe,
        safe_accept_retention=retention,
        unguarded_false_safe_count=unguarded_false,
        guarded_false_safe_count=guarded_false,
        removed_false_safe_count=unguarded_false - guarded_false,
        phase9_supported=supported,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20700804)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_outlier_guard(
        reference_count=args.reference,
        repeats=args.repeats,
        seed=args.seed,
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
