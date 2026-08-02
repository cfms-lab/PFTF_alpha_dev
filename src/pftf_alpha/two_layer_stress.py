"""Frozen Phase-3 generalization stress for two-layer connectivity.

The panel includes four positive perturbations of separated two-layer geometry
and two negative geometries that must fail closed. Family declarations and true
labels are evaluation-only and never enter construction or routing.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
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


class TwoLayerStressFamily(StrEnum):
    ROTATED_PARALLEL = "rotated_parallel"
    CURVED_PARALLEL = "curved_parallel"
    TILTED_SEPARATION = "tilted_separation"
    PARTIAL_OVERLAP = "partial_overlap"
    NEAR_CONTACT = "near_contact"
    CROSSING = "crossing"


POSITIVE_FAMILIES = (
    TwoLayerStressFamily.ROTATED_PARALLEL,
    TwoLayerStressFamily.CURVED_PARALLEL,
    TwoLayerStressFamily.TILTED_SEPARATION,
    TwoLayerStressFamily.PARTIAL_OVERLAP,
)
NEGATIVE_FAMILIES = (
    TwoLayerStressFamily.NEAR_CONTACT,
    TwoLayerStressFamily.CROSSING,
)
DEFAULT_FAMILIES = POSITIVE_FAMILIES + NEGATIVE_FAMILIES


@dataclass(frozen=True)
class StressCaseResult:
    family: TwoLayerStressFamily
    declared_in_scope: bool
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
    inferred_component_error: int
    inferred_false_bridge_edges: int
    inferred_false_bridge_faces: int
    geometry_nonregression: bool
    true_safe_output: bool
    false_safe: bool
    out_of_scope_false_accept: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["family"] = self.family.value
        payload["decision"] = self.decision.value
        return payload


@dataclass(frozen=True)
class StressFamilySummary:
    family: TwoLayerStressFamily
    declared_in_scope: bool
    case_count: int
    sampling_eligible_count: int
    accepted_count: int
    accepted_safe_count: int
    false_safe_count: int
    out_of_scope_false_accept_count: int
    safe_acceptance_coverage: float | None
    mean_b5_fscore: float
    mean_constrained_fscore: float
    b5_betti_error_sum: int
    constrained_betti_error_sum: int
    family_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["family"] = self.family.value
        return payload


@dataclass(frozen=True)
class TwoLayerStressResult:
    artifact_schema: str
    role: str
    information_boundary: str
    point_count: int
    reference_count: int
    repeats: int
    seed: int
    families: tuple[TwoLayerStressFamily, ...]
    gate_config: SamplingSufficiencyConfig
    cases: tuple[StressCaseResult, ...]
    summaries: tuple[StressFamilySummary, ...]
    positive_eligible_case_count: int
    positive_accepted_safe_count: int
    positive_safe_acceptance_coverage: float
    negative_case_count: int
    negative_accept_count: int
    false_safe_count: int
    positive_mean_b5_fscore: float | None
    positive_mean_constrained_fscore: float | None
    positive_b5_betti_error_sum: int
    positive_constrained_betti_error_sum: int
    phase3_supported: bool
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
            "families": [family.value for family in self.families],
            "gate_config": asdict(self.gate_config),
            "cases": [case.to_dict() for case in self.cases],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "positive_eligible_case_count": self.positive_eligible_case_count,
            "positive_accepted_safe_count": self.positive_accepted_safe_count,
            "positive_safe_acceptance_coverage": (
                self.positive_safe_acceptance_coverage
            ),
            "negative_case_count": self.negative_case_count,
            "negative_accept_count": self.negative_accept_count,
            "false_safe_count": self.false_safe_count,
            "positive_mean_b5_fscore": self.positive_mean_b5_fscore,
            "positive_mean_constrained_fscore": (
                self.positive_mean_constrained_fscore
            ),
            "positive_b5_betti_error_sum": self.positive_b5_betti_error_sum,
            "positive_constrained_betti_error_sum": (
                self.positive_constrained_betti_error_sum
            ),
            "phase3_supported": self.phase3_supported,
            "deployment_supported": self.deployment_supported,
        }


def _rotation_matrix(rng: np.random.Generator) -> FloatArray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.35, 1.20)
    x, y, z = axis
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )


def _surface_coordinates(
    family: TwoLayerStressFamily,
    xy: FloatArray,
    layer: int,
) -> FloatArray:
    x, y = xy[:, 0], xy[:, 1]
    sign = -1.0 if layer == 0 else 1.0
    if family in {
        TwoLayerStressFamily.ROTATED_PARALLEL,
        TwoLayerStressFamily.PARTIAL_OVERLAP,
    }:
        z = np.full(x.shape, sign * 0.40)
    elif family is TwoLayerStressFamily.CURVED_PARALLEL:
        z = 0.12 * (x * x + y * y) + sign * 0.40
    elif family is TwoLayerStressFamily.TILTED_SEPARATION:
        separation = 0.80 + 0.25 * x
        z = 0.05 * y + sign * 0.5 * separation
    elif family is TwoLayerStressFamily.NEAR_CONTACT:
        separation = 0.04 + 0.38 * (x + 1.0)
        z = sign * 0.5 * separation
    else:
        z = sign * 0.40 * x
    return np.column_stack((x, y, z))


def _layer_xy(
    family: TwoLayerStressFamily,
    count: int,
    layer: int,
    rng: np.random.Generator,
) -> FloatArray:
    xy = rng.uniform(-1.0, 1.0, size=(count, 2))
    if family is TwoLayerStressFamily.PARTIAL_OVERLAP and layer == 1:
        xy[:, 0] += 1.0
    return xy


def _characteristic_length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.ptp(points, axis=0)))


def make_stress_case(
    family: TwoLayerStressFamily | str,
    *,
    point_count: int = 160,
    reference_count: int = 4096,
    seed: int = 0,
    noise: float = 0.01,
) -> SyntheticCase:
    """Generate a balanced two-layer stress case with evaluation-only labels."""

    selected = TwoLayerStressFamily(family)
    if point_count < 16 or reference_count < point_count:
        raise ValueError("counts require reference_count >= point_count >= 16")
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
        layers = []
        labels = []
        for layer, layer_count in enumerate(counts):
            xy = _layer_xy(selected, layer_count, layer, rng)
            layers.append(_surface_coordinates(selected, xy, layer))
            labels.append(np.full(layer_count, layer, dtype=np.int64))
        points = np.vstack(layers)
        if add_noise and noise > 0.0:
            points = points + rng.normal(scale=noise, size=points.shape)
        return points, np.concatenate(labels)

    observed, labels = sample(point_count, observed_rng, add_noise=True)
    reference, _ = sample(reference_count, reference_rng, add_noise=False)
    if selected is TwoLayerStressFamily.ROTATED_PARALLEL:
        rotation = _rotation_matrix(np.random.default_rng(seed + 2_000_003))
        observed = observed @ rotation.T
        reference = reference @ rotation.T
    nominal_gap = {
        TwoLayerStressFamily.NEAR_CONTACT: 0.04,
        TwoLayerStressFamily.CROSSING: 0.01,
    }.get(selected, 0.80)
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


def _summarize_family(
    cases: Sequence[StressCaseResult],
    family: TwoLayerStressFamily,
) -> StressFamilySummary:
    rows = [case for case in cases if case.family is family]
    if not rows:
        raise RuntimeError("every configured stress family must have results")
    declared_in_scope = family in POSITIVE_FAMILIES
    eligible = [case for case in rows if case.sampling_sufficient]
    accepted = [
        case for case in rows if case.decision is SamplingGateDecision.ACCEPT
    ]
    accepted_safe = [case for case in accepted if case.true_safe_output]
    coverage = None if not eligible else len(accepted_safe) / len(eligible)
    if declared_in_scope:
        passed = bool(
            len(eligible) >= 4
            and coverage is not None
            and coverage >= 0.75
            and len(accepted_safe) == len(accepted)
        )
    else:
        passed = not accepted
    return StressFamilySummary(
        family=family,
        declared_in_scope=declared_in_scope,
        case_count=len(rows),
        sampling_eligible_count=len(eligible),
        accepted_count=len(accepted),
        accepted_safe_count=len(accepted_safe),
        false_safe_count=sum(case.false_safe for case in rows),
        out_of_scope_false_accept_count=sum(
            case.out_of_scope_false_accept for case in rows
        ),
        safe_acceptance_coverage=coverage,
        mean_b5_fscore=float(np.mean([case.b5.fscore for case in rows])),
        mean_constrained_fscore=float(
            np.mean([case.constrained.fscore for case in rows])
        ),
        b5_betti_error_sum=sum(case.b5.betti_error for case in rows),
        constrained_betti_error_sum=sum(
            int(case.constrained.betti_error or 0) for case in rows
        ),
        family_gate_passed=passed,
    )


def evaluate_two_layer_stress(
    *,
    point_count: int = 160,
    reference_count: int = 4096,
    repeats: int = 8,
    seed: int = 20280803,
    families: Sequence[TwoLayerStressFamily | str] = DEFAULT_FAMILIES,
    surface_sample_count: int = 512,
    gate_config: SamplingSufficiencyConfig | None = None,
    fscore_nonregression_tolerance: float = 0.01,
) -> TwoLayerStressResult:
    """Execute the frozen Phase-3 stress panel without threshold fitting."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not math.isfinite(fscore_nonregression_tolerance) or (
        fscore_nonregression_tolerance < 0.0
    ):
        raise ValueError("F-score tolerance must be finite and non-negative")
    selected_families = tuple(TwoLayerStressFamily(family) for family in families)
    if not selected_families or len(set(selected_families)) != len(selected_families):
        raise ValueError("families must be non-empty and unique")
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
    results: list[StressCaseResult] = []
    for family_index, family in enumerate(selected_families):
        for repeat in range(repeats):
            case_seed = seed + family_index * 100_003 + repeat * 10_007
            case = make_stress_case(
                family,
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
                threshold_fraction=reconstruction_config.fscore_threshold_fraction,
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
                threshold_fraction=reconstruction_config.fscore_threshold_fraction,
                seed=case_seed + 41,
            )
            decision = route_two_layer_output(construction, inferred)
            true_safe = bool(
                constrained.component_error == 0
                and int(constrained.labeled_false_bridge_edges or 0) == 0
                and int(constrained.labeled_false_bridge_faces or 0) == 0
            )
            accepted = decision is SamplingGateDecision.ACCEPT
            declared_in_scope = family in POSITIVE_FAMILIES
            evidence = construction.inference.evidence
            results.append(
                StressCaseResult(
                    family=family,
                    declared_in_scope=declared_in_scope,
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
                    inferred_component_error=inferred.component_error,
                    inferred_false_bridge_edges=int(
                        inferred.labeled_false_bridge_edges or 0
                    ),
                    inferred_false_bridge_faces=int(
                        inferred.labeled_false_bridge_faces or 0
                    ),
                    geometry_nonregression=bool(
                        constrained.fscore + fscore_nonregression_tolerance
                        >= b5.fscore
                    ),
                    true_safe_output=true_safe,
                    false_safe=bool(accepted and not true_safe),
                    out_of_scope_false_accept=bool(
                        accepted and not declared_in_scope
                    ),
                )
            )

    summaries = tuple(
        _summarize_family(results, family) for family in selected_families
    )
    positive_eligible = [
        case
        for case in results
        if case.declared_in_scope and case.sampling_sufficient
    ]
    positive_accepted_safe = [
        case
        for case in positive_eligible
        if case.decision is SamplingGateDecision.ACCEPT and case.true_safe_output
    ]
    negative = [case for case in results if not case.declared_in_scope]
    negative_accepts = sum(
        case.decision is SamplingGateDecision.ACCEPT for case in negative
    )
    false_safe_count = sum(case.false_safe for case in results)
    coverage = (
        0.0
        if not positive_eligible
        else len(positive_accepted_safe) / len(positive_eligible)
    )
    mean_b5 = (
        None
        if not positive_eligible
        else float(np.mean([case.b5.fscore for case in positive_eligible]))
    )
    mean_constrained = (
        None
        if not positive_eligible
        else float(
            np.mean([case.constrained.fscore for case in positive_eligible])
        )
    )
    b5_betti = sum(case.b5.betti_error for case in positive_eligible)
    constrained_betti = sum(
        int(case.constrained.betti_error or 0) for case in positive_eligible
    )
    phase3_supported = bool(
        selected_families == DEFAULT_FAMILIES
        and repeats >= 8
        and all(summary.family_gate_passed for summary in summaries)
        and false_safe_count == 0
        and negative_accepts == 0
        and mean_b5 is not None
        and mean_constrained is not None
        and mean_constrained + fscore_nonregression_tolerance >= mean_b5
        and constrained_betti <= b5_betti
    )
    return TwoLayerStressResult(
        artifact_schema="pftf_alpha_two_layer_stress_phase3/v1",
        role="generalization_and_fail_closed_stress_test",
        information_boundary=(
            "construction and routing use observed coordinates and inferred layers "
            "only; family declarations, true labels, and references are evaluation-only"
        ),
        point_count=point_count,
        reference_count=reference_count,
        repeats=repeats,
        seed=seed,
        families=selected_families,
        gate_config=selected_gate,
        cases=tuple(results),
        summaries=summaries,
        positive_eligible_case_count=len(positive_eligible),
        positive_accepted_safe_count=len(positive_accepted_safe),
        positive_safe_acceptance_coverage=coverage,
        negative_case_count=len(negative),
        negative_accept_count=negative_accepts,
        false_safe_count=false_safe_count,
        positive_mean_b5_fscore=mean_b5,
        positive_mean_constrained_fscore=mean_constrained,
        positive_b5_betti_error_sum=b5_betti,
        positive_constrained_betti_error_sum=constrained_betti,
        phase3_supported=phase3_supported,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--points", type=int, default=160)
    parser.add_argument("--reference", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20280803)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_two_layer_stress(
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
