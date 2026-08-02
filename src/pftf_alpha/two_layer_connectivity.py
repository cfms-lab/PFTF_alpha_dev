"""Parallel-sheet constrained-connectivity baseline and Phase-2 evaluation.

This is a specialized baseline, not a general alpha-complex replacement.  It
uses observed-only layer inference, constructs one 2D Delaunay surface per
inferred layer, and never creates a cross-layer face.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError

from .reacquisition import ReacquisitionConfig, ReconstructionSnapshot, _reconstruct
from .sampling_gate import (
    ParallelLayerInference,
    SamplingGateDecision,
    SamplingSufficiencyConfig,
    infer_parallel_layers,
)
from .surface import SurfaceEndpointMetrics, SurfaceMesh, evaluate_surface
from .synthetic import PanelSplit, SyntheticFamily, make_synthetic_case

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TwoLayerConstruction:
    mesh: SurfaceMesh
    inference: ParallelLayerInference
    layer_point_counts: tuple[int, int]
    layer_face_counts: tuple[int, int]


@dataclass(frozen=True)
class TwoLayerCaseResult:
    repeat: int
    seed: int
    sheet_gap: float
    sampling_sufficient: bool
    estimated_cross_knn_fraction: float
    true_cross_knn_fraction: float
    layer_point_counts: tuple[int, int]
    layer_face_counts: tuple[int, int]
    decision: SamplingGateDecision
    b5: ReconstructionSnapshot
    constrained: SurfaceEndpointMetrics
    inferred_component_error: int
    inferred_false_bridge_edges: int
    inferred_false_bridge_faces: int
    geometry_nonregression: bool
    true_safe_output: bool
    false_safe: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload


@dataclass(frozen=True)
class TwoLayerGapSummary:
    sheet_gap: float
    case_count: int
    sampling_sufficient_count: int
    accepted_count: int
    true_safe_count: int
    false_safe_count: int
    mean_b5_fscore: float
    mean_constrained_fscore: float
    b5_component_error_sum: int
    constrained_component_error_sum: int
    b5_bridge_edge_sum: int
    constrained_bridge_edge_sum: int
    b5_betti_error_sum: int
    constrained_betti_error_sum: int
    geometry_nonregression_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TwoLayerExperimentResult:
    artifact_schema: str
    role: str
    information_boundary: str
    point_count: int
    reference_count: int
    repeats: int
    seed: int
    gaps: tuple[float, ...]
    gate_config: SamplingSufficiencyConfig
    cases: tuple[TwoLayerCaseResult, ...]
    summaries: tuple[TwoLayerGapSummary, ...]
    eligible_case_count: int
    accepted_case_count: int
    true_safe_case_count: int
    false_safe_count: int
    safe_acceptance_coverage: float
    eligible_mean_b5_fscore: float | None
    eligible_mean_constrained_fscore: float | None
    eligible_b5_betti_error_sum: int
    eligible_constrained_betti_error_sum: int
    phase2_supported: bool
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
            "gaps": self.gaps,
            "gate_config": asdict(self.gate_config),
            "cases": [case.to_dict() for case in self.cases],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "eligible_case_count": self.eligible_case_count,
            "accepted_case_count": self.accepted_case_count,
            "true_safe_case_count": self.true_safe_case_count,
            "false_safe_count": self.false_safe_count,
            "safe_acceptance_coverage": self.safe_acceptance_coverage,
            "eligible_mean_b5_fscore": self.eligible_mean_b5_fscore,
            "eligible_mean_constrained_fscore": (
                self.eligible_mean_constrained_fscore
            ),
            "eligible_b5_betti_error_sum": self.eligible_b5_betti_error_sum,
            "eligible_constrained_betti_error_sum": (
                self.eligible_constrained_betti_error_sum
            ),
            "phase2_supported": self.phase2_supported,
            "deployment_supported": self.deployment_supported,
        }


def construct_two_layer_surface(
    points: FloatArray,
    gate_config: SamplingSufficiencyConfig | None = None,
) -> TwoLayerConstruction:
    """Triangulate each inferred layer independently in its PCA tangent plane."""

    point_array = np.asarray(points, dtype=np.float64)
    inference = infer_parallel_layers(point_array, gate_config)
    return construct_two_layer_surface_from_inference(point_array, inference)


def construct_two_layer_surface_from_inference(
    points: FloatArray,
    inference: ParallelLayerInference,
) -> TwoLayerConstruction:
    """Triangulate externally inferred layers without changing their evidence."""

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if inference.layer_ids.shape != (point_array.shape[0],):
        raise ValueError("inference must align with points")
    all_faces: list[np.ndarray] = []
    point_counts: list[int] = []
    face_counts: list[int] = []
    for layer in range(2):
        indices = np.flatnonzero(inference.layer_ids == layer)
        point_counts.append(int(indices.size))
        if indices.size < 3:
            raise ValueError("each inferred layer requires at least three points")
        layer_points = point_array[indices]
        centered = layer_points - np.mean(layer_points, axis=0)
        basis = np.linalg.eigh(centered.T @ centered)[1][:, -2:]
        projected = centered @ basis
        try:
            local_faces = Delaunay(projected).simplices
        except QhullError as error:
            raise ValueError(f"layer {layer} 2D Delaunay failed: {error}") from error
        faces = indices[np.asarray(local_faces, dtype=np.int64)]
        all_faces.append(faces)
        face_counts.append(int(faces.shape[0]))
    mesh = SurfaceMesh(vertices=point_array, faces=np.vstack(all_faces))
    return TwoLayerConstruction(
        mesh=mesh,
        inference=inference,
        layer_point_counts=(point_counts[0], point_counts[1]),
        layer_face_counts=(face_counts[0], face_counts[1]),
    )


def route_two_layer_output(
    construction: TwoLayerConstruction,
    inferred_endpoints: SurfaceEndpointMetrics,
) -> SamplingGateDecision:
    evidence = construction.inference.evidence
    if not evidence.two_layer_identifiable:
        return SamplingGateDecision.UNSUPPORTED
    if not evidence.sampling_sufficient:
        return SamplingGateDecision.RESCAN_REQUIRED
    if (
        inferred_endpoints.component_error > 0
        or int(inferred_endpoints.labeled_false_bridge_edges or 0) > 0
        or int(inferred_endpoints.labeled_false_bridge_faces or 0) > 0
    ):
        return SamplingGateDecision.ALGORITHM_FAILURE
    return SamplingGateDecision.ACCEPT


def _summarize_gap(
    cases: Sequence[TwoLayerCaseResult],
    gap: float,
) -> TwoLayerGapSummary:
    rows = [case for case in cases if case.sheet_gap == gap]
    if not rows:
        raise RuntimeError("every configured gap must have results")
    return TwoLayerGapSummary(
        sheet_gap=gap,
        case_count=len(rows),
        sampling_sufficient_count=sum(case.sampling_sufficient for case in rows),
        accepted_count=sum(
            case.decision is SamplingGateDecision.ACCEPT for case in rows
        ),
        true_safe_count=sum(case.true_safe_output for case in rows),
        false_safe_count=sum(case.false_safe for case in rows),
        mean_b5_fscore=float(np.mean([case.b5.fscore for case in rows])),
        mean_constrained_fscore=float(
            np.mean([case.constrained.fscore for case in rows])
        ),
        b5_component_error_sum=sum(case.b5.component_error for case in rows),
        constrained_component_error_sum=sum(
            case.constrained.component_error for case in rows
        ),
        b5_bridge_edge_sum=sum(case.b5.labeled_false_bridge_edges for case in rows),
        constrained_bridge_edge_sum=sum(
            int(case.constrained.labeled_false_bridge_edges or 0) for case in rows
        ),
        b5_betti_error_sum=sum(case.b5.betti_error for case in rows),
        constrained_betti_error_sum=sum(
            int(case.constrained.betti_error or 0) for case in rows
        ),
        geometry_nonregression_count=sum(case.geometry_nonregression for case in rows),
    )


def evaluate_two_layer_connectivity(
    *,
    point_count: int = 96,
    reference_count: int = 2048,
    gaps: Sequence[float] = (0.18, 0.40, 0.60, 0.80, 1.20),
    repeats: int = 8,
    seed: int = 20280802,
    surface_sample_count: int = 512,
    gate_config: SamplingSufficiencyConfig | None = None,
    fscore_nonregression_tolerance: float = 0.01,
) -> TwoLayerExperimentResult:
    """Run the frozen Phase-2 B5 versus constrained-layer held-out panel."""

    selected_gate = (
        SamplingSufficiencyConfig(minimum_separation_snr=3.0)
        if gate_config is None
        else gate_config
    )
    selected_gaps = tuple(float(gap) for gap in gaps)
    if not selected_gaps or any(
        not math.isfinite(gap) or gap <= 0.0 for gap in selected_gaps
    ):
        raise ValueError("gaps must contain finite positive values")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if (
        not math.isfinite(fscore_nonregression_tolerance)
        or fscore_nonregression_tolerance < 0.0
    ):
        raise ValueError("fscore tolerance must be finite and non-negative")
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

    case_results: list[TwoLayerCaseResult] = []
    for gap_index, gap in enumerate(selected_gaps):
        for repeat in range(repeats):
            case_seed = seed + gap_index * 100_003 + repeat * 10_007
            case = make_synthetic_case(
                SyntheticFamily.OPPOSING_SHEETS,
                split=PanelSplit.HELD_OUT,
                point_count=point_count,
                reference_count=reference_count,
                seed=case_seed,
                variation_overrides={"sheet_gap": gap, "noise": 0.01},
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
                expected_components=case.expected_components,
                expected_betti=case.expected_surface_betti,
                vertex_component_labels=case.point_component_labels,
                characteristic_length=case.characteristic_length,
                sample_count=surface_sample_count,
                threshold_fraction=reconstruction_config.fscore_threshold_fraction,
                seed=case_seed + 41,
            )
            inferred_endpoints = evaluate_surface(
                construction.mesh,
                case.reference_points,
                expected_components=2,
                expected_betti=case.expected_surface_betti,
                vertex_component_labels=construction.inference.layer_ids,
                characteristic_length=case.characteristic_length,
                sample_count=surface_sample_count,
                threshold_fraction=reconstruction_config.fscore_threshold_fraction,
                seed=case_seed + 41,
            )
            decision = route_two_layer_output(construction, inferred_endpoints)
            geometry_nonregression = bool(
                constrained.fscore + fscore_nonregression_tolerance >= b5.fscore
            )
            true_safe = bool(
                constrained.component_error == 0
                and int(constrained.labeled_false_bridge_edges or 0) == 0
                and int(constrained.labeled_false_bridge_faces or 0) == 0
            )
            false_safe = bool(
                decision is SamplingGateDecision.ACCEPT and not true_safe
            )
            case_results.append(
                TwoLayerCaseResult(
                    repeat=repeat,
                    seed=case_seed,
                    sheet_gap=gap,
                    sampling_sufficient=(
                        construction.inference.evidence.sampling_sufficient
                    ),
                    estimated_cross_knn_fraction=(
                        construction.inference.evidence.estimated_cross_knn_fraction
                    ),
                    true_cross_knn_fraction=b5.cross_knn_fraction,
                    layer_point_counts=construction.layer_point_counts,
                    layer_face_counts=construction.layer_face_counts,
                    decision=decision,
                    b5=b5,
                    constrained=constrained,
                    inferred_component_error=inferred_endpoints.component_error,
                    inferred_false_bridge_edges=int(
                        inferred_endpoints.labeled_false_bridge_edges or 0
                    ),
                    inferred_false_bridge_faces=int(
                        inferred_endpoints.labeled_false_bridge_faces or 0
                    ),
                    geometry_nonregression=geometry_nonregression,
                    true_safe_output=true_safe,
                    false_safe=false_safe,
                )
            )

    summaries = tuple(
        _summarize_gap(case_results, gap) for gap in selected_gaps
    )
    eligible = [case for case in case_results if case.sampling_sufficient]
    accepted = [
        case for case in case_results if case.decision is SamplingGateDecision.ACCEPT
    ]
    true_safe = [case for case in eligible if case.true_safe_output]
    false_safe_count = sum(case.false_safe for case in case_results)
    safe_accepted = sum(case.true_safe_output for case in accepted)
    coverage = 0.0 if not true_safe else safe_accepted / len(true_safe)
    mean_b5_fscore = (
        None if not eligible else float(np.mean([case.b5.fscore for case in eligible]))
    )
    mean_constrained_fscore = (
        None
        if not eligible
        else float(np.mean([case.constrained.fscore for case in eligible]))
    )
    b5_betti_sum = sum(case.b5.betti_error for case in eligible)
    constrained_betti_sum = sum(
        int(case.constrained.betti_error or 0) for case in eligible
    )
    eligible_bridges = sum(
        int(case.constrained.labeled_false_bridge_edges or 0) for case in eligible
    )
    phase2_supported = bool(
        eligible
        and false_safe_count == 0
        and len(true_safe) == len(eligible)
        and coverage >= 0.80
        and mean_b5_fscore is not None
        and mean_constrained_fscore is not None
        and mean_constrained_fscore + fscore_nonregression_tolerance
        >= mean_b5_fscore
        and constrained_betti_sum <= b5_betti_sum
        and eligible_bridges == 0
    )
    return TwoLayerExperimentResult(
        artifact_schema="pftf_alpha_two_layer_connectivity_phase2/v1",
        role="specialized_parallel_sheet_connectivity_baseline",
        information_boundary=(
            "construction and routing use observed coordinates and inferred layers "
            "only; true labels and reference points are evaluation-only"
        ),
        point_count=point_count,
        reference_count=reference_count,
        repeats=repeats,
        seed=seed,
        gaps=selected_gaps,
        gate_config=selected_gate,
        cases=tuple(case_results),
        summaries=summaries,
        eligible_case_count=len(eligible),
        accepted_case_count=len(accepted),
        true_safe_case_count=len(true_safe),
        false_safe_count=false_safe_count,
        safe_acceptance_coverage=coverage,
        eligible_mean_b5_fscore=mean_b5_fscore,
        eligible_mean_constrained_fscore=mean_constrained_fscore,
        eligible_b5_betti_error_sum=b5_betti_sum,
        eligible_constrained_betti_error_sum=constrained_betti_sum,
        phase2_supported=phase2_supported,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument(
        "--gaps",
        type=float,
        nargs="+",
        default=[0.18, 0.40, 0.60, 0.80, 1.20],
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20280802)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_two_layer_connectivity(
        point_count=args.points,
        reference_count=args.reference,
        gaps=args.gaps,
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
