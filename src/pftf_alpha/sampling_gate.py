"""Observed-only sampling-sufficiency and residual-risk routing.

The Phase-1 gate is intentionally limited to approximately parallel two-layer
geometry.  It distinguishes a likely sampling deficit from a reconstruction
failure after sampling is adequate.  Unsupported geometry always fails closed.
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
from scipy.spatial import cKDTree

from .adaptive import local_neighborhood_geometry
from .reacquisition import ReacquisitionConfig, _reconstruct
from .synthetic import PanelSplit, SyntheticFamily, make_synthetic_case

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class SamplingGateDecision(StrEnum):
    RESCAN_REQUIRED = "rescan_required"
    ALGORITHM_FAILURE = "algorithm_failure_fail_closed"
    ACCEPT = "accept"
    UNSUPPORTED = "unsupported_geometry_fail_closed"


@dataclass(frozen=True)
class SamplingSufficiencyConfig:
    k_neighbors: int = 12
    cross_knn_threshold: float = 0.05
    minimum_cluster_fraction: float = 0.20
    minimum_separation_snr: float = 4.0
    risk_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.k_neighbors < 3:
            raise ValueError("k_neighbors must be at least three")
        for name, value in (
            ("cross_knn_threshold", self.cross_knn_threshold),
            ("minimum_cluster_fraction", self.minimum_cluster_fraction),
        ):
            if not math.isfinite(value) or not 0.0 < value < 0.5:
                raise ValueError(f"{name} must lie in (0, 0.5)")
        if (
            not math.isfinite(self.minimum_separation_snr)
            or self.minimum_separation_snr <= 0.0
        ):
            raise ValueError("minimum_separation_snr must be finite and positive")
        if not math.isfinite(self.risk_threshold) or self.risk_threshold < 0.0:
            raise ValueError("risk_threshold must be finite and non-negative")


@dataclass(frozen=True)
class SamplingSufficiencyEvidence:
    information_boundary: str
    point_count: int
    k_neighbors: int
    cluster_sizes: tuple[int, int]
    minimum_cluster_fraction: float
    estimated_layer_gap: float
    pooled_normal_spread: float
    separation_snr: float
    estimated_cross_knn_fraction: float
    cross_knn_threshold: float
    two_layer_identifiable: bool
    sampling_sufficient: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ParallelLayerInference:
    """Observed-only two-layer assignment plus its sufficiency evidence."""

    layer_ids: IntArray
    evidence: SamplingSufficiencyEvidence

    def __post_init__(self) -> None:
        labels = np.asarray(self.layer_ids, dtype=np.int64)
        if labels.shape != (self.evidence.point_count,):
            raise ValueError("layer_ids must align with the evidence point count")
        if np.any(labels < 0) or np.any(labels > 1):
            raise ValueError("layer_ids must contain only zero and one")
        object.__setattr__(self, "layer_ids", np.ascontiguousarray(labels))


@dataclass(frozen=True)
class SamplingGateCaseResult:
    repeat: int
    seed: int
    sheet_gap: float
    evidence: SamplingSufficiencyEvidence
    flagged_boundary_faces: int
    flagged_boundary_edges: int
    decision: SamplingGateDecision
    true_cross_knn_fraction: float
    true_sampling_sufficient: bool
    component_error: int
    labeled_false_bridge_edges: int
    labeled_false_bridge_faces: int
    expected_decision: SamplingGateDecision
    routing_correct: bool
    false_safe: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        payload["expected_decision"] = self.expected_decision.value
        return payload


@dataclass(frozen=True)
class SamplingGateGapSummary:
    sheet_gap: float
    case_count: int
    mean_estimated_cross_knn_fraction: float
    mean_true_cross_knn_fraction: float
    mean_absolute_cross_knn_error: float
    sampling_classification_accuracy: float
    routing_accuracy: float
    rescan_required_count: int
    algorithm_failure_count: int
    accept_count: int
    unsupported_count: int
    false_safe_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SamplingGateExperimentResult:
    artifact_schema: str
    role: str
    policy_information_boundary: str
    point_count: int
    reference_count: int
    repeats: int
    seed: int
    gaps: tuple[float, ...]
    gate_config: SamplingSufficiencyConfig
    cases: tuple[SamplingGateCaseResult, ...]
    summaries: tuple[SamplingGateGapSummary, ...]
    overall_sampling_classification_accuracy: float
    overall_routing_accuracy: float
    false_safe_count: int
    accept_count: int
    phase1_diagnostic_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "policy_information_boundary": self.policy_information_boundary,
            "point_count": self.point_count,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "seed": self.seed,
            "gaps": self.gaps,
            "gate_config": asdict(self.gate_config),
            "cases": [case.to_dict() for case in self.cases],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "overall_sampling_classification_accuracy": (
                self.overall_sampling_classification_accuracy
            ),
            "overall_routing_accuracy": self.overall_routing_accuracy,
            "false_safe_count": self.false_safe_count,
            "accept_count": self.accept_count,
            "phase1_diagnostic_supported": self.phase1_diagnostic_supported,
            "deployment_supported": self.deployment_supported,
        }


def _two_means(values: FloatArray) -> tuple[IntArray, tuple[float, float]]:
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size < 4:
        raise ValueError("two-layer coordinates must contain at least four values")
    centers = np.quantile(coordinates, [0.25, 0.75]).astype(np.float64)
    labels = np.zeros(coordinates.size, dtype=np.int64)
    for _ in range(64):
        distances = np.abs(coordinates[:, None] - centers[None, :])
        updated_labels = np.argmin(distances, axis=1).astype(np.int64)
        if np.any(np.bincount(updated_labels, minlength=2) == 0):
            break
        updated_centers = np.asarray(
            [np.mean(coordinates[updated_labels == index]) for index in range(2)]
        )
        if np.array_equal(labels, updated_labels) and np.allclose(
            centers,
            updated_centers,
            rtol=0.0,
            atol=4.0 * np.finfo(np.float64).eps,
        ):
            labels = updated_labels
            centers = updated_centers
            break
        labels = updated_labels
        centers = updated_centers
    order = np.argsort(centers)
    remap = np.empty(2, dtype=np.int64)
    remap[order] = np.arange(2, dtype=np.int64)
    sorted_centers = tuple(float(value) for value in centers[order])
    return remap[labels], sorted_centers


def infer_parallel_layers(
    points: FloatArray,
    config: SamplingSufficiencyConfig | None = None,
) -> ParallelLayerInference:
    """Infer two layers and sampling adequacy from observed coordinates only."""

    selected = SamplingSufficiencyConfig() if config is None else config
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if point_array.shape[0] <= selected.k_neighbors:
        raise ValueError("point count must exceed k_neighbors")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must be finite")

    centered = point_array - np.mean(point_array, axis=0)
    local_geometry = local_neighborhood_geometry(
        point_array,
        k_neighbors=selected.k_neighbors,
    )
    local_normals = local_geometry.eigenvectors[:, :, 0]
    orientation_tensor = np.mean(
        local_normals[:, :, None] * local_normals[:, None, :],
        axis=0,
    )
    normal = np.linalg.eigh(orientation_tensor)[1][:, -1]
    normal_coordinates = centered @ normal
    labels, centers = _two_means(normal_coordinates)
    cluster_counts = np.bincount(labels, minlength=2)
    cluster_fraction = float(np.min(cluster_counts) / point_array.shape[0])
    layer_gap = abs(centers[1] - centers[0])
    residuals = normal_coordinates - np.asarray(centers)[labels]
    pooled_spread = float(np.sqrt(np.mean(residuals**2)))
    separation_snr = layer_gap / max(
        pooled_spread,
        np.finfo(np.float64).eps,
    )

    selected_k = min(selected.k_neighbors, point_array.shape[0] - 1)
    neighbors = cKDTree(point_array).query(
        point_array,
        k=selected_k + 1,
        workers=1,
    )[1][:, 1:]
    estimated_cross_fraction = float(
        np.mean(labels[neighbors] != labels[:, None])
    )
    identifiable = bool(
        cluster_fraction >= selected.minimum_cluster_fraction
        and separation_snr >= selected.minimum_separation_snr
    )
    sufficient = bool(
        identifiable
        and estimated_cross_fraction <= selected.cross_knn_threshold
    )
    evidence = SamplingSufficiencyEvidence(
        information_boundary="observed_point_coordinates_only",
        point_count=point_array.shape[0],
        k_neighbors=selected_k,
        cluster_sizes=(int(cluster_counts[0]), int(cluster_counts[1])),
        minimum_cluster_fraction=cluster_fraction,
        estimated_layer_gap=layer_gap,
        pooled_normal_spread=pooled_spread,
        separation_snr=separation_snr,
        estimated_cross_knn_fraction=estimated_cross_fraction,
        cross_knn_threshold=selected.cross_knn_threshold,
        two_layer_identifiable=identifiable,
        sampling_sufficient=sufficient,
    )
    return ParallelLayerInference(layer_ids=labels, evidence=evidence)


def estimate_sampling_sufficiency(
    points: FloatArray,
    config: SamplingSufficiencyConfig | None = None,
) -> SamplingSufficiencyEvidence:
    """Estimate two-layer sampling adequacy from observed coordinates only."""

    return infer_parallel_layers(points, config).evidence


def route_sampling_gate(
    evidence: SamplingSufficiencyEvidence,
    *,
    flagged_boundary_faces: int,
    flagged_boundary_edges: int,
) -> SamplingGateDecision:
    """Route without allowing an unsupported or risky output to pass."""

    if flagged_boundary_faces < 0 or flagged_boundary_edges < 0:
        raise ValueError("flagged boundary counts must be non-negative")
    if not evidence.two_layer_identifiable:
        return SamplingGateDecision.UNSUPPORTED
    if not evidence.sampling_sufficient:
        return SamplingGateDecision.RESCAN_REQUIRED
    if flagged_boundary_faces > 0 or flagged_boundary_edges > 0:
        return SamplingGateDecision.ALGORITHM_FAILURE
    return SamplingGateDecision.ACCEPT


def _expected_decision(
    *,
    true_sampling_sufficient: bool,
    component_error: int,
    bridge_edges: int,
    bridge_faces: int,
) -> SamplingGateDecision:
    if not true_sampling_sufficient:
        return SamplingGateDecision.RESCAN_REQUIRED
    if component_error > 0 or bridge_edges > 0 or bridge_faces > 0:
        return SamplingGateDecision.ALGORITHM_FAILURE
    return SamplingGateDecision.ACCEPT


def _summarize_gap(
    cases: Sequence[SamplingGateCaseResult],
    gap: float,
) -> SamplingGateGapSummary:
    rows = [case for case in cases if case.sheet_gap == gap]
    if not rows:
        raise RuntimeError("every configured gap must have results")
    estimated = np.asarray(
        [case.evidence.estimated_cross_knn_fraction for case in rows]
    )
    actual = np.asarray([case.true_cross_knn_fraction for case in rows])
    decisions = [case.decision for case in rows]
    return SamplingGateGapSummary(
        sheet_gap=gap,
        case_count=len(rows),
        mean_estimated_cross_knn_fraction=float(np.mean(estimated)),
        mean_true_cross_knn_fraction=float(np.mean(actual)),
        mean_absolute_cross_knn_error=float(np.mean(np.abs(estimated - actual))),
        sampling_classification_accuracy=float(
            np.mean(
                [
                    case.evidence.sampling_sufficient
                    == case.true_sampling_sufficient
                    for case in rows
                ]
            )
        ),
        routing_accuracy=float(np.mean([case.routing_correct for case in rows])),
        rescan_required_count=decisions.count(
            SamplingGateDecision.RESCAN_REQUIRED
        ),
        algorithm_failure_count=decisions.count(
            SamplingGateDecision.ALGORITHM_FAILURE
        ),
        accept_count=decisions.count(SamplingGateDecision.ACCEPT),
        unsupported_count=decisions.count(SamplingGateDecision.UNSUPPORTED),
        false_safe_count=sum(case.false_safe for case in rows),
    )


def evaluate_sampling_sufficiency_gate(
    *,
    point_count: int = 96,
    reference_count: int = 2048,
    gaps: Sequence[float] = (0.18, 0.40, 0.60, 0.80, 1.20),
    repeats: int = 8,
    seed: int = 20260802,
    surface_sample_count: int = 512,
    gate_config: SamplingSufficiencyConfig | None = None,
) -> SamplingGateExperimentResult:
    """Evaluate the router on a frozen opposing-sheet gap sweep."""

    selected_gate = (
        SamplingSufficiencyConfig() if gate_config is None else gate_config
    )
    selected_gaps = tuple(float(gap) for gap in gaps)
    if not selected_gaps or any(
        not math.isfinite(gap) or gap <= 0.0 for gap in selected_gaps
    ):
        raise ValueError("gaps must contain finite positive values")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    reconstruction_config = ReacquisitionConfig(
        base_point_count=point_count,
        evaluation_reference_count=reference_count,
        candidate_pool_count=reference_count,
        added_point_counts=(1,),
        repeats=1,
        seed=seed,
        surface_sample_count=surface_sample_count,
        k_neighbors=selected_gate.k_neighbors,
        risk_threshold=selected_gate.risk_threshold,
    )

    case_results: list[SamplingGateCaseResult] = []
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
            snapshot, _ = _reconstruct(
                case,
                reconstruction_config,
                evaluation_seed=case_seed + 31,
            )
            evidence = estimate_sampling_sufficiency(case.points, selected_gate)
            decision = route_sampling_gate(
                evidence,
                flagged_boundary_faces=snapshot.flagged_boundary_faces,
                flagged_boundary_edges=snapshot.flagged_boundary_edges,
            )
            true_sufficient = bool(
                snapshot.cross_knn_fraction <= selected_gate.cross_knn_threshold
            )
            expected = _expected_decision(
                true_sampling_sufficient=true_sufficient,
                component_error=snapshot.component_error,
                bridge_edges=snapshot.labeled_false_bridge_edges,
                bridge_faces=snapshot.labeled_false_bridge_faces,
            )
            false_safe = bool(
                decision is SamplingGateDecision.ACCEPT
                and (
                    snapshot.component_error > 0
                    or snapshot.labeled_false_bridge_edges > 0
                    or snapshot.labeled_false_bridge_faces > 0
                )
            )
            case_results.append(
                SamplingGateCaseResult(
                    repeat=repeat,
                    seed=case_seed,
                    sheet_gap=gap,
                    evidence=evidence,
                    flagged_boundary_faces=snapshot.flagged_boundary_faces,
                    flagged_boundary_edges=snapshot.flagged_boundary_edges,
                    decision=decision,
                    true_cross_knn_fraction=snapshot.cross_knn_fraction,
                    true_sampling_sufficient=true_sufficient,
                    component_error=snapshot.component_error,
                    labeled_false_bridge_edges=snapshot.labeled_false_bridge_edges,
                    labeled_false_bridge_faces=snapshot.labeled_false_bridge_faces,
                    expected_decision=expected,
                    routing_correct=decision is expected,
                    false_safe=false_safe,
                )
            )
    summaries = tuple(
        _summarize_gap(case_results, gap) for gap in selected_gaps
    )
    sampling_accuracy = float(
        np.mean(
            [
                case.evidence.sampling_sufficient
                == case.true_sampling_sufficient
                for case in case_results
            ]
        )
    )
    routing_accuracy = float(
        np.mean([case.routing_correct for case in case_results])
    )
    false_safe_count = sum(case.false_safe for case in case_results)
    accept_count = sum(
        case.decision is SamplingGateDecision.ACCEPT for case in case_results
    )
    has_rescan = any(
        case.decision is SamplingGateDecision.RESCAN_REQUIRED
        for case in case_results
    )
    has_algorithm_failure = any(
        case.decision is SamplingGateDecision.ALGORITHM_FAILURE
        for case in case_results
    )
    diagnostic_supported = bool(
        false_safe_count == 0
        and sampling_accuracy >= 0.95
        and routing_accuracy >= 0.95
        and has_rescan
        and has_algorithm_failure
    )
    return SamplingGateExperimentResult(
        artifact_schema="pftf_alpha_sampling_sufficiency_gate_phase1/v1",
        role="synthetic_parallel_sheet_diagnostic_gate_only",
        policy_information_boundary="observed_point_coordinates_and_output_risk_only",
        point_count=point_count,
        reference_count=reference_count,
        repeats=repeats,
        seed=seed,
        gaps=selected_gaps,
        gate_config=selected_gate,
        cases=tuple(case_results),
        summaries=summaries,
        overall_sampling_classification_accuracy=sampling_accuracy,
        overall_routing_accuracy=routing_accuracy,
        false_safe_count=false_safe_count,
        accept_count=accept_count,
        phase1_diagnostic_supported=diagnostic_supported,
        deployment_supported=bool(diagnostic_supported and accept_count > 0),
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
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--cross-knn-threshold", type=float, default=0.05)
    parser.add_argument("--minimum-separation-snr", type=float, default=4.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gate_config = SamplingSufficiencyConfig(
        cross_knn_threshold=args.cross_knn_threshold,
        minimum_separation_snr=args.minimum_separation_snr,
    )
    result = evaluate_sampling_sufficiency_gate(
        point_count=args.points,
        reference_count=args.reference,
        gaps=args.gaps,
        repeats=args.repeats,
        seed=args.seed,
        surface_sample_count=args.surface_samples,
        gate_config=gate_config,
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
