"""Frozen Phase-4 operating-boundary sweep for two-layer connectivity."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .reacquisition import ReacquisitionConfig, ReconstructionSnapshot, _reconstruct
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .surface import SurfaceEndpointMetrics, evaluate_surface
from .synthetic import PanelSplit, SyntheticCase, SyntheticFamily
from .two_layer_connectivity import (
    construct_two_layer_surface,
    route_two_layer_output,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class BoundaryAxis(StrEnum):
    CURVATURE = "curvature"
    TILT_SPAN = "tilt_span"
    OVERLAP_OFFSET = "overlap_offset"
    CONTACT_SEVERITY = "contact_severity"


DEFAULT_LEVELS: Mapping[BoundaryAxis, tuple[float, ...]] = {
    BoundaryAxis.CURVATURE: (0.00, 0.12, 0.24, 0.36, 0.48, 0.60),
    BoundaryAxis.TILT_SPAN: (0.00, 0.25, 0.40, 0.55, 0.70, 0.76),
    BoundaryAxis.OVERLAP_OFFSET: (0.00, 0.50, 1.00, 1.50, 2.00, 2.50),
    BoundaryAxis.CONTACT_SEVERITY: (0.00, 0.20, 0.40, 0.60, 0.70, 0.76),
}


@dataclass(frozen=True)
class BoundaryCaseResult:
    axis: BoundaryAxis
    level: float
    repeat: int
    seed: int
    sampling_sufficient: bool
    two_layer_identifiable: bool
    estimated_cross_knn_fraction: float
    true_cross_knn_fraction: float
    separation_snr: float
    decision: SamplingGateDecision
    b5: ReconstructionSnapshot
    constrained: SurfaceEndpointMetrics
    true_safe_output: bool
    safe_accept: bool
    false_safe: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["axis"] = self.axis.value
        payload["decision"] = self.decision.value
        return payload


@dataclass(frozen=True)
class BoundaryLevelSummary:
    axis: BoundaryAxis
    level: float
    case_count: int
    sampling_eligible_count: int
    accepted_count: int
    accepted_safe_count: int
    false_safe_count: int
    acceptance_rate: float
    safe_acceptance_rate: float
    mean_cross_knn_fraction: float
    mean_separation_snr: float
    mean_b5_fscore: float
    mean_constrained_fscore: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["axis"] = self.axis.value
        return payload


@dataclass(frozen=True)
class BoundaryAxisSummary:
    axis: BoundaryAxis
    last_reliable_level: float | None
    first_rejection_dominant_level: float | None
    transition_observed: bool
    all_levels_reliable: bool
    no_level_reliable: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["axis"] = self.axis.value
        return payload


@dataclass(frozen=True)
class TwoLayerBoundaryResult:
    artifact_schema: str
    role: str
    information_boundary: str
    point_count: int
    reference_count: int
    repeats: int
    seed: int
    levels: Mapping[str, tuple[float, ...]]
    gate_config: SamplingSufficiencyConfig
    cases: tuple[BoundaryCaseResult, ...]
    level_summaries: tuple[BoundaryLevelSummary, ...]
    axis_summaries: tuple[BoundaryAxisSummary, ...]
    case_count: int
    accepted_safe_count: int
    false_safe_count: int
    phase4_diagnostic_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "point_count": self.point_count,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "seed": self.seed,
            "levels": self.levels,
            "gate_config": asdict(self.gate_config),
            "cases": [case.to_dict() for case in self.cases],
            "level_summaries": [row.to_dict() for row in self.level_summaries],
            "axis_summaries": [row.to_dict() for row in self.axis_summaries],
            "case_count": self.case_count,
            "accepted_safe_count": self.accepted_safe_count,
            "false_safe_count": self.false_safe_count,
            "phase4_diagnostic_supported": self.phase4_diagnostic_supported,
            "deployment_supported": self.deployment_supported,
        }


def _surface_points(
    axis: BoundaryAxis,
    level: float,
    xy: FloatArray,
    layer: int,
) -> FloatArray:
    x, y = xy[:, 0], xy[:, 1]
    sign = -1.0 if layer == 0 else 1.0
    if axis is BoundaryAxis.CURVATURE:
        z = level * (x * x + y * y) + sign * 0.40
    elif axis is BoundaryAxis.TILT_SPAN:
        z = sign * 0.5 * (0.80 + level * x)
    elif axis is BoundaryAxis.CONTACT_SEVERITY:
        minimum_gap = 0.80 - level
        gap = minimum_gap + 0.5 * level * (x + 1.0)
        z = sign * 0.5 * gap
    else:
        z = np.full(x.shape, sign * 0.40)
    return np.column_stack((x, y, z))


def _characteristic_length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.ptp(points, axis=0)))


def make_boundary_case(
    axis: BoundaryAxis | str,
    level: float,
    *,
    point_count: int = 160,
    reference_count: int = 2048,
    seed: int = 0,
    noise: float = 0.01,
) -> SyntheticCase:
    """Generate one balanced two-layer case at a declared sweep level."""

    selected_axis = BoundaryAxis(axis)
    selected_level = float(level)
    if point_count < 16 or reference_count < point_count:
        raise ValueError("counts require reference_count >= point_count >= 16")
    if not math.isfinite(selected_level) or selected_level < 0.0:
        raise ValueError("level must be finite and non-negative")
    if selected_axis in {
        BoundaryAxis.TILT_SPAN,
        BoundaryAxis.CONTACT_SEVERITY,
    } and selected_level >= 0.80:
        raise ValueError("tilt/contact level must be below the base gap 0.80")
    if not math.isfinite(noise) or noise < 0.0:
        raise ValueError("noise must be finite and non-negative")
    observed_rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed + 1_000_003)

    def sample(
        count: int,
        rng: np.random.Generator,
        *,
        add_noise: bool,
    ) -> tuple[FloatArray, IntArray]:
        counts = (count // 2, count - count // 2)
        point_rows = []
        label_rows = []
        for layer, layer_count in enumerate(counts):
            xy = rng.uniform(-1.0, 1.0, size=(layer_count, 2))
            if selected_axis is BoundaryAxis.OVERLAP_OFFSET and layer == 1:
                xy[:, 0] += selected_level
            point_rows.append(_surface_points(selected_axis, selected_level, xy, layer))
            label_rows.append(np.full(layer_count, layer, dtype=np.int64))
        points = np.vstack(point_rows)
        if add_noise and noise > 0.0:
            points = points + rng.normal(scale=noise, size=points.shape)
        return points, np.concatenate(label_rows)

    observed, labels = sample(point_count, observed_rng, add_noise=True)
    reference, _ = sample(reference_count, reference_rng, add_noise=False)
    nominal_gap = max(0.80 - selected_level, 0.01)
    if selected_axis not in {
        BoundaryAxis.TILT_SPAN,
        BoundaryAxis.CONTACT_SEVERITY,
    }:
        nominal_gap = 0.80
    return SyntheticCase(
        family=SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.HELD_OUT,
        points=observed,
        reference_points=reference,
        expected_components=2,
        characteristic_length=_characteristic_length(reference),
        variation={"sheet_gap": nominal_gap, "noise": noise},
        seed=seed,
        expected_surface_betti=(2, 0, 0),
        point_component_labels=labels,
    )


def _level_summary(
    cases: Sequence[BoundaryCaseResult],
    axis: BoundaryAxis,
    level: float,
) -> BoundaryLevelSummary:
    rows = [case for case in cases if case.axis is axis and case.level == level]
    if not rows:
        raise RuntimeError("every configured axis level must have results")
    accepted_count = sum(
        case.decision is SamplingGateDecision.ACCEPT for case in rows
    )
    return BoundaryLevelSummary(
        axis=axis,
        level=level,
        case_count=len(rows),
        sampling_eligible_count=sum(case.sampling_sufficient for case in rows),
        accepted_count=accepted_count,
        accepted_safe_count=sum(case.safe_accept for case in rows),
        false_safe_count=sum(case.false_safe for case in rows),
        acceptance_rate=accepted_count / len(rows),
        safe_acceptance_rate=float(np.mean([case.safe_accept for case in rows])),
        mean_cross_knn_fraction=float(
            np.mean([case.estimated_cross_knn_fraction for case in rows])
        ),
        mean_separation_snr=float(np.mean([case.separation_snr for case in rows])),
        mean_b5_fscore=float(np.mean([case.b5.fscore for case in rows])),
        mean_constrained_fscore=float(
            np.mean([case.constrained.fscore for case in rows])
        ),
    )


def _axis_summary(
    levels: Sequence[BoundaryLevelSummary],
    axis: BoundaryAxis,
) -> BoundaryAxisSummary:
    rows = [row for row in levels if row.axis is axis]
    reliable = [
        row
        for row in rows
        if row.safe_acceptance_rate >= 0.75 and row.false_safe_count == 0
    ]
    last_reliable = None if not reliable else reliable[-1].level
    later_rows = (
        rows
        if last_reliable is None
        else [row for row in rows if row.level > last_reliable]
    )
    rejection = next(
        (row.level for row in later_rows if row.acceptance_rate <= 0.25),
        None,
    )
    return BoundaryAxisSummary(
        axis=axis,
        last_reliable_level=last_reliable,
        first_rejection_dominant_level=rejection,
        transition_observed=bool(last_reliable is not None and rejection is not None),
        all_levels_reliable=len(reliable) == len(rows),
        no_level_reliable=not reliable,
    )


def evaluate_two_layer_boundary(
    *,
    point_count: int = 160,
    reference_count: int = 2048,
    repeats: int = 8,
    seed: int = 20280804,
    levels: Mapping[BoundaryAxis | str, Sequence[float]] = DEFAULT_LEVELS,
    surface_sample_count: int = 256,
    gate_config: SamplingSufficiencyConfig | None = None,
) -> TwoLayerBoundaryResult:
    """Run the frozen boundary grid without fitting a threshold to its results."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    selected_levels = {
        BoundaryAxis(axis): tuple(float(level) for level in axis_levels)
        for axis, axis_levels in levels.items()
    }
    if not selected_levels or any(not rows for rows in selected_levels.values()):
        raise ValueError("levels must contain at least one level per selected axis")
    for rows in selected_levels.values():
        if tuple(sorted(rows)) != rows or len(set(rows)) != len(rows):
            raise ValueError("axis levels must be unique and increasing")
    selected_gate = gate_config or SamplingSufficiencyConfig(
        minimum_separation_snr=3.0
    )
    reconstruction_config = ReacquisitionConfig(
        base_point_count=point_count,
        evaluation_reference_count=reference_count,
        candidate_pool_count=reference_count,
        added_point_counts=(1,),
        repeats=1,
        seed=seed,
        surface_sample_count=surface_sample_count,
        k_neighbors=selected_gate.k_neighbors,
    )
    results: list[BoundaryCaseResult] = []
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
                b5, _ = _reconstruct(
                    case,
                    reconstruction_config,
                    evaluation_seed=case_seed + 31,
                )
                construction = construct_two_layer_surface(case.points, selected_gate)
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
                decision = route_two_layer_output(construction, inferred)
                true_safe = bool(
                    constrained.component_error == 0
                    and int(constrained.labeled_false_bridge_edges or 0) == 0
                    and int(constrained.labeled_false_bridge_faces or 0) == 0
                )
                accepted = decision is SamplingGateDecision.ACCEPT
                evidence = construction.inference.evidence
                results.append(
                    BoundaryCaseResult(
                        axis=axis,
                        level=level,
                        repeat=repeat,
                        seed=case_seed,
                        sampling_sufficient=evidence.sampling_sufficient,
                        two_layer_identifiable=evidence.two_layer_identifiable,
                        estimated_cross_knn_fraction=(
                            evidence.estimated_cross_knn_fraction
                        ),
                        true_cross_knn_fraction=b5.cross_knn_fraction,
                        separation_snr=evidence.separation_snr,
                        decision=decision,
                        b5=b5,
                        constrained=constrained,
                        true_safe_output=true_safe,
                        safe_accept=bool(accepted and true_safe),
                        false_safe=bool(accepted and not true_safe),
                    )
                )

    level_summaries = tuple(
        _level_summary(results, axis, level)
        for axis, axis_levels in selected_levels.items()
        for level in axis_levels
    )
    axis_summaries = tuple(
        _axis_summary(level_summaries, axis) for axis in selected_levels
    )
    false_safe_count = sum(case.false_safe for case in results)
    anchors = {
        (BoundaryAxis.CURVATURE, 0.12),
        (BoundaryAxis.TILT_SPAN, 0.25),
        (BoundaryAxis.OVERLAP_OFFSET, 1.00),
    }
    anchor_rows = [
        row for row in level_summaries if (row.axis, row.level) in anchors
    ]
    contact_endpoint = next(
        (
            row
            for row in level_summaries
            if row.axis is BoundaryAxis.CONTACT_SEVERITY
            and row.level == 0.76
        ),
        None,
    )
    frozen_levels = {
        axis: tuple(axis_levels) for axis, axis_levels in DEFAULT_LEVELS.items()
    }
    phase4_supported = bool(
        selected_levels == frozen_levels
        and repeats >= 8
        and len(results) == 192
        and false_safe_count == 0
        and len(anchor_rows) == 3
        and all(row.safe_acceptance_rate >= 0.75 for row in anchor_rows)
        and contact_endpoint is not None
        and contact_endpoint.safe_acceptance_rate <= 0.25
        and any(row.transition_observed for row in axis_summaries)
    )
    serialized_levels = {
        axis.value: axis_levels for axis, axis_levels in selected_levels.items()
    }
    return TwoLayerBoundaryResult(
        artifact_schema="pftf_alpha_two_layer_boundary_phase4/v1",
        role="operating_boundary_characterization",
        information_boundary=(
            "construction and routing use observed coordinates and inferred layers "
            "only; axis, level, true labels, and references are evaluation-only"
        ),
        point_count=point_count,
        reference_count=reference_count,
        repeats=repeats,
        seed=seed,
        levels=serialized_levels,
        gate_config=selected_gate,
        cases=tuple(results),
        level_summaries=level_summaries,
        axis_summaries=axis_summaries,
        case_count=len(results),
        accepted_safe_count=sum(case.safe_accept for case in results),
        false_safe_count=false_safe_count,
        phase4_diagnostic_supported=phase4_supported,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--points", type=int, default=160)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20280804)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_two_layer_boundary(
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
