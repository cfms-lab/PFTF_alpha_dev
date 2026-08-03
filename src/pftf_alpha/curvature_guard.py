"""Observed-only normal-coherence guard and frozen Phase-4b validation."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .adaptive import local_neighborhood_geometry
from .reacquisition import ReacquisitionConfig
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .surface import SurfaceEndpointMetrics, evaluate_surface
from .two_layer_boundary import DEFAULT_LEVELS, BoundaryAxis, make_boundary_case
from .two_layer_connectivity import (
    TwoLayerConstruction,
    construct_two_layer_surface,
    route_two_layer_output,
)


@dataclass(frozen=True)
class CurvatureGuardConfig:
    k_neighbors: int = 12
    minimum_normal_coherence: float = 0.82

    def __post_init__(self) -> None:
        if self.k_neighbors < 3:
            raise ValueError("k_neighbors must be at least three")
        if not math.isfinite(self.minimum_normal_coherence) or not (
            0.0 < self.minimum_normal_coherence <= 1.0
        ):
            raise ValueError("minimum_normal_coherence must lie in (0, 1]")


@dataclass(frozen=True)
class CurvatureGuardEvidence:
    information_boundary: str
    point_count: int
    k_neighbors: int
    normal_coherence: float
    minimum_normal_coherence: float
    model_adequate: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GuardCaseResult:
    axis: BoundaryAxis
    level: float
    repeat: int
    seed: int
    evidence: CurvatureGuardEvidence
    sampling_sufficient: bool
    base_decision: SamplingGateDecision
    guarded_decision: SamplingGateDecision
    constrained: SurfaceEndpointMetrics
    true_safe_output: bool
    base_safe_accept: bool
    guarded_safe_accept: bool
    base_false_safe: bool
    guarded_false_safe: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["axis"] = self.axis.value
        payload["base_decision"] = self.base_decision.value
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class GuardLevelSummary:
    axis: BoundaryAxis
    level: float
    case_count: int
    mean_normal_coherence: float
    base_accepted_count: int
    guarded_accepted_count: int
    base_safe_accept_count: int
    guarded_safe_accept_count: int
    base_false_safe_count: int
    guarded_false_safe_count: int
    guarded_safe_acceptance_rate: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["axis"] = self.axis.value
        return payload


@dataclass(frozen=True)
class CurvatureGuardResult:
    artifact_schema: str
    role: str
    information_boundary: str
    calibration_source: str
    calibration_false_safe_max_coherence: float
    calibration_anchor_min_coherence: float
    point_count: int
    reference_count: int
    repeats: int
    seed: int
    levels: Mapping[str, tuple[float, ...]]
    sampling_gate_config: SamplingSufficiencyConfig
    guard_config: CurvatureGuardConfig
    cases: tuple[GuardCaseResult, ...]
    level_summaries: tuple[GuardLevelSummary, ...]
    case_count: int
    base_safe_accept_count: int
    guarded_safe_accept_count: int
    safe_accept_retention: float
    base_false_safe_count: int
    guarded_false_safe_count: int
    removed_false_safe_count: int
    phase4b_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "calibration_source": self.calibration_source,
            "calibration_false_safe_max_coherence": (
                self.calibration_false_safe_max_coherence
            ),
            "calibration_anchor_min_coherence": (
                self.calibration_anchor_min_coherence
            ),
            "point_count": self.point_count,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "seed": self.seed,
            "levels": self.levels,
            "sampling_gate_config": asdict(self.sampling_gate_config),
            "guard_config": asdict(self.guard_config),
            "cases": [case.to_dict() for case in self.cases],
            "level_summaries": [row.to_dict() for row in self.level_summaries],
            "case_count": self.case_count,
            "base_safe_accept_count": self.base_safe_accept_count,
            "guarded_safe_accept_count": self.guarded_safe_accept_count,
            "safe_accept_retention": self.safe_accept_retention,
            "base_false_safe_count": self.base_false_safe_count,
            "guarded_false_safe_count": self.guarded_false_safe_count,
            "removed_false_safe_count": self.removed_false_safe_count,
            "phase4b_supported": self.phase4b_supported,
            "deployment_supported": self.deployment_supported,
        }


def estimate_curvature_guard(
    points: np.ndarray,
    config: CurvatureGuardConfig | None = None,
) -> CurvatureGuardEvidence:
    """Estimate global-normal model adequacy from observed coordinates only."""

    selected = CurvatureGuardConfig() if config is None else config
    point_array = np.asarray(points, dtype=np.float64)
    geometry = local_neighborhood_geometry(
        point_array,
        k_neighbors=selected.k_neighbors,
    )
    normals = geometry.eigenvectors[:, :, 0]
    orientation_tensor = np.mean(
        normals[:, :, None] * normals[:, None, :],
        axis=0,
    )
    coherence = float(np.linalg.eigvalsh(orientation_tensor)[-1])
    return CurvatureGuardEvidence(
        information_boundary="observed_point_coordinates_only",
        point_count=point_array.shape[0],
        k_neighbors=geometry.k_neighbors,
        normal_coherence=coherence,
        minimum_normal_coherence=selected.minimum_normal_coherence,
        model_adequate=coherence >= selected.minimum_normal_coherence,
    )


def route_with_curvature_guard(
    construction: TwoLayerConstruction,
    inferred_endpoints: SurfaceEndpointMetrics,
    evidence: CurvatureGuardEvidence,
) -> tuple[SamplingGateDecision, SamplingGateDecision]:
    """Return the frozen base decision and guarded fail-closed decision."""

    base = route_two_layer_output(construction, inferred_endpoints)
    guarded = base
    if base is SamplingGateDecision.ACCEPT and not evidence.model_adequate:
        guarded = SamplingGateDecision.UNSUPPORTED
    return base, guarded


def _summarize_level(
    cases: Sequence[GuardCaseResult],
    axis: BoundaryAxis,
    level: float,
) -> GuardLevelSummary:
    rows = [case for case in cases if case.axis is axis and case.level == level]
    if not rows:
        raise RuntimeError("every configured guard level must have results")
    return GuardLevelSummary(
        axis=axis,
        level=level,
        case_count=len(rows),
        mean_normal_coherence=float(
            np.mean([case.evidence.normal_coherence for case in rows])
        ),
        base_accepted_count=sum(
            case.base_decision is SamplingGateDecision.ACCEPT for case in rows
        ),
        guarded_accepted_count=sum(
            case.guarded_decision is SamplingGateDecision.ACCEPT for case in rows
        ),
        base_safe_accept_count=sum(case.base_safe_accept for case in rows),
        guarded_safe_accept_count=sum(case.guarded_safe_accept for case in rows),
        base_false_safe_count=sum(case.base_false_safe for case in rows),
        guarded_false_safe_count=sum(case.guarded_false_safe for case in rows),
        guarded_safe_acceptance_rate=float(
            np.mean([case.guarded_safe_accept for case in rows])
        ),
    )


def evaluate_curvature_guard(
    *,
    point_count: int = 160,
    reference_count: int = 2048,
    repeats: int = 8,
    seed: int = 20290804,
    levels: Mapping[BoundaryAxis | str, Sequence[float]] = DEFAULT_LEVELS,
    surface_sample_count: int = 256,
    sampling_gate_config: SamplingSufficiencyConfig | None = None,
    guard_config: CurvatureGuardConfig | None = None,
) -> CurvatureGuardResult:
    """Validate the frozen calibration-derived guard on a new held-out grid."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    selected_levels = {
        BoundaryAxis(axis): tuple(float(level) for level in axis_levels)
        for axis, axis_levels in levels.items()
    }
    if not selected_levels or any(not rows for rows in selected_levels.values()):
        raise ValueError("levels must contain at least one level per selected axis")
    selected_sampling = sampling_gate_config or SamplingSufficiencyConfig(
        minimum_separation_snr=3.0
    )
    selected_guard = guard_config or CurvatureGuardConfig(
        k_neighbors=selected_sampling.k_neighbors
    )
    if selected_guard.k_neighbors != selected_sampling.k_neighbors:
        raise ValueError("sampling and curvature guard k must match")
    reconstruction_config = ReacquisitionConfig(
        base_point_count=point_count,
        evaluation_reference_count=reference_count,
        candidate_pool_count=reference_count,
        added_point_counts=(1,),
        repeats=1,
        seed=seed,
        surface_sample_count=surface_sample_count,
        k_neighbors=selected_sampling.k_neighbors,
    )
    results: list[GuardCaseResult] = []
    for axis_index, (axis, axis_levels) in enumerate(selected_levels.items()):
        for level_index, level in enumerate(axis_levels):
            for repeat in range(repeats):
                case_seed = (
                    seed
                    + axis_index * 1_000_003
                    + level_index * 100_003
                    + repeat * 10_007
                )
                case = make_boundary_case(
                    axis,
                    level,
                    point_count=point_count,
                    reference_count=reference_count,
                    seed=case_seed,
                )
                construction = construct_two_layer_surface(
                    case.points,
                    selected_sampling,
                )
                constrained = evaluate_surface(
                    construction.mesh,
                    case.reference_points,
                    expected_components=2,
                    expected_betti=(2, 0, 0),
                    vertex_component_labels=case.point_component_labels,
                    characteristic_length=case.characteristic_length,
                    sample_count=surface_sample_count,
                    threshold_fraction=(
                        reconstruction_config.fscore_threshold_fraction
                    ),
                    seed=case_seed + 41,
                )
                inferred = evaluate_surface(
                    construction.mesh,
                    case.reference_points,
                    expected_components=2,
                    expected_betti=(2, 0, 0),
                    vertex_component_labels=construction.inference.layer_ids,
                    characteristic_length=case.characteristic_length,
                    sample_count=surface_sample_count,
                    threshold_fraction=(
                        reconstruction_config.fscore_threshold_fraction
                    ),
                    seed=case_seed + 41,
                )
                evidence = estimate_curvature_guard(case.points, selected_guard)
                base, guarded = route_with_curvature_guard(
                    construction,
                    inferred,
                    evidence,
                )
                true_safe = bool(
                    constrained.component_error == 0
                    and int(constrained.labeled_false_bridge_edges or 0) == 0
                    and int(constrained.labeled_false_bridge_faces or 0) == 0
                )
                base_accept = base is SamplingGateDecision.ACCEPT
                guarded_accept = guarded is SamplingGateDecision.ACCEPT
                results.append(
                    GuardCaseResult(
                        axis=axis,
                        level=level,
                        repeat=repeat,
                        seed=case_seed,
                        evidence=evidence,
                        sampling_sufficient=(
                            construction.inference.evidence.sampling_sufficient
                        ),
                        base_decision=base,
                        guarded_decision=guarded,
                        constrained=constrained,
                        true_safe_output=true_safe,
                        base_safe_accept=bool(base_accept and true_safe),
                        guarded_safe_accept=bool(guarded_accept and true_safe),
                        base_false_safe=bool(base_accept and not true_safe),
                        guarded_false_safe=bool(guarded_accept and not true_safe),
                    )
                )

    summaries = tuple(
        _summarize_level(results, axis, level)
        for axis, axis_levels in selected_levels.items()
        for level in axis_levels
    )
    base_safe = sum(case.base_safe_accept for case in results)
    guarded_safe = sum(case.guarded_safe_accept for case in results)
    base_false = sum(case.base_false_safe for case in results)
    guarded_false = sum(case.guarded_false_safe for case in results)
    retention = 0.0 if not base_safe else guarded_safe / base_safe
    anchors = {
        (BoundaryAxis.CURVATURE, 0.12),
        (BoundaryAxis.CURVATURE, 0.24),
        (BoundaryAxis.TILT_SPAN, 0.25),
        (BoundaryAxis.TILT_SPAN, 0.40),
        (BoundaryAxis.OVERLAP_OFFSET, 1.00),
        (BoundaryAxis.CONTACT_SEVERITY, 0.20),
    }
    anchor_rows = [row for row in summaries if (row.axis, row.level) in anchors]
    frozen_levels = {
        axis: tuple(axis_levels) for axis, axis_levels in DEFAULT_LEVELS.items()
    }
    supported = bool(
        selected_levels == frozen_levels
        and repeats >= 8
        and len(results) == 192
        and base_false > 0
        and guarded_false == 0
        and base_false - guarded_false == base_false
        and retention >= 0.90
        and len(anchor_rows) == len(anchors)
        and all(
            row.guarded_safe_acceptance_rate >= 0.75 for row in anchor_rows
        )
    )
    serialized_levels = {
        axis.value: axis_levels for axis, axis_levels in selected_levels.items()
    }
    return CurvatureGuardResult(
        artifact_schema="pftf_alpha_curvature_guard_phase4b/v1",
        role="observed_only_model_adequacy_fail_closed_guard",
        information_boundary=(
            "guard uses observed kNN PCA normal orientation tensor only; axis, "
            "level, true labels, and references are evaluation-only"
        ),
        calibration_source="phase4_seed_20280804",
        calibration_false_safe_max_coherence=0.7799,
        calibration_anchor_min_coherence=0.8294,
        point_count=point_count,
        reference_count=reference_count,
        repeats=repeats,
        seed=seed,
        levels=serialized_levels,
        sampling_gate_config=selected_sampling,
        guard_config=selected_guard,
        cases=tuple(results),
        level_summaries=summaries,
        case_count=len(results),
        base_safe_accept_count=base_safe,
        guarded_safe_accept_count=guarded_safe,
        safe_accept_retention=retention,
        base_false_safe_count=base_false,
        guarded_false_safe_count=guarded_false,
        removed_false_safe_count=base_false - guarded_false,
        phase4b_supported=supported,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--points", type=int, default=160)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20290804)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_curvature_guard(
        point_count=args.points,
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
