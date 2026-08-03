"""Phase-0 validation of risk-localized point reacquisition.

This module deliberately implements a synthetic ROI-rescan proxy, not a full
next-best-view planner.  The policy sees only observed geometry, boundary-risk
localization, and candidate point positions.  Component labels and dense
reference geometry are reserved for endpoint evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .adaptive import (
    BoundaryBridgeLocalization,
    boundary_bridge_localization,
    pca_anisotropic_filtration,
)
from .filtration import AlphaFiltration
from .surface import evaluate_surface
from .synthetic import PanelSplit, SyntheticCase, SyntheticFamily, make_synthetic_case

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ReacquisitionPolicy(StrEnum):
    UNIFORM = "uniform"
    RISK_TARGETED = "risk_targeted"


@dataclass(frozen=True)
class ReacquisitionConfig:
    """Frozen Phase-0 budgets and reconstruction settings."""

    base_point_count: int = 48
    evaluation_reference_count: int = 4096
    candidate_pool_count: int = 4096
    added_point_counts: tuple[int, ...] = (12, 24, 36)
    repeats: int = 8
    seed: int = 20260724
    surface_sample_count: int = 1024
    fscore_threshold_fraction: float = 0.025
    k_neighbors: int = 12
    b5_scale_multiplier: float = 2.80293354289327
    b5_max_normal_penalty: float = 4.0
    risk_threshold: float = 1.0
    normal_coherence_threshold: float = 0.9
    normal_edge_threshold: float = 0.02
    length_edge_threshold: float = 1.8
    fscore_nonregression_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.base_point_count < 16:
            raise ValueError("base_point_count must be at least 16")
        if self.evaluation_reference_count < self.base_point_count:
            raise ValueError("evaluation_reference_count must cover the base points")
        if self.candidate_pool_count < max(self.added_point_counts, default=1):
            raise ValueError("candidate_pool_count must cover every added-point budget")
        if self.repeats < 1 or self.surface_sample_count < 1:
            raise ValueError("repeats and surface_sample_count must be positive")
        if self.k_neighbors < 3 or self.k_neighbors >= self.base_point_count:
            raise ValueError("k_neighbors must be in [3, base_point_count)")
        if not self.added_point_counts or any(
            isinstance(value, bool) or value < 1 for value in self.added_point_counts
        ):
            raise ValueError("added_point_counts must contain positive integers")
        if len(set(self.added_point_counts)) != len(self.added_point_counts):
            raise ValueError("added_point_counts must be unique")
        for name, value in (
            ("b5_scale_multiplier", self.b5_scale_multiplier),
            ("b5_max_normal_penalty", self.b5_max_normal_penalty),
            ("risk_threshold", self.risk_threshold),
            ("normal_coherence_threshold", self.normal_coherence_threshold),
            ("normal_edge_threshold", self.normal_edge_threshold),
            ("length_edge_threshold", self.length_edge_threshold),
            ("fscore_threshold_fraction", self.fscore_threshold_fraction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.fscore_nonregression_tolerance)
            or self.fscore_nonregression_tolerance < 0.0
        ):
            raise ValueError(
                "fscore_nonregression_tolerance must be finite and non-negative"
            )


@dataclass(frozen=True)
class ReconstructionSnapshot:
    fscore: float
    normalized_chamfer_squared: float
    normalized_hausdorff: float
    component_error: int
    betti_error: int
    labeled_false_bridge_edges: int
    labeled_false_bridge_faces: int
    boundary_edges: int
    nonmanifold_edges: int
    flagged_boundary_faces: int
    flagged_boundary_edges: int
    cross_knn_fraction: float
    median_knn_spacing: float
    gap_to_spacing_ratio: float
    risk_route: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReacquisitionTrial:
    repeat: int
    case_seed: int
    policy: ReacquisitionPolicy
    added_point_count: int
    selected_index_sha256: str
    risk_anchor_count: int
    fell_back_to_uniform: bool
    baseline: ReconstructionSnapshot
    reacquired: ReconstructionSnapshot

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["policy"] = self.policy.value
        return payload


@dataclass(frozen=True)
class BudgetComparison:
    added_point_count: int
    repeat_count: int
    uniform_mean_fscore: float
    targeted_mean_fscore: float
    uniform_mean_bridge_edges: float
    targeted_mean_bridge_edges: float
    uniform_mean_bridge_faces: float
    targeted_mean_bridge_faces: float
    uniform_mean_betti_error: float
    targeted_mean_betti_error: float
    uniform_mean_component_error: float
    targeted_mean_component_error: float
    uniform_mean_cross_knn_fraction: float
    targeted_mean_cross_knn_fraction: float
    targeted_bridge_edge_win_count: int
    targeted_bridge_face_win_count: int
    fscore_nonregression: bool
    topology_nonregression: bool
    component_nonregression: bool
    sampling_information_improved: bool
    bridge_edge_improved: bool
    bridge_face_improved: bool
    phase0_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReacquisitionExperimentResult:
    artifact_schema: str
    role: str
    family: str
    split: str
    reconstruction: str
    policy_information_boundary: str
    config: ReacquisitionConfig
    trials: tuple[ReacquisitionTrial, ...]
    comparisons: tuple[BudgetComparison, ...]
    phase0_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "family": self.family,
            "split": self.split,
            "reconstruction": self.reconstruction,
            "policy_information_boundary": self.policy_information_boundary,
            "config": asdict(self.config),
            "trials": [trial.to_dict() for trial in self.trials],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "phase0_supported": self.phase0_supported,
        }


def select_reacquisition_indices(
    candidate_points: FloatArray,
    *,
    budget: int,
    policy: ReacquisitionPolicy | str,
    seed: int,
    observed_points: FloatArray | None = None,
    localization: BoundaryBridgeLocalization | None = None,
    risk_threshold: float = 1.0,
) -> tuple[IntArray, int, bool]:
    """Select a fixed-size candidate subset without consulting labels.

    ``risk_targeted`` ranks candidate returns by tangent-plane distance to
    centroids of boundary faces whose risk exceeds the frozen threshold.  The
    normal coordinate is deliberately ignored so both nearby sheets can be
    sampled in a risky projected region.  This represents a local ROI rescan
    after those returns become available, not visibility-aware next-best-view
    planning.
    """

    candidates = np.asarray(candidate_points, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_points must have shape (n, 3)")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidate_points must be finite")
    if isinstance(budget, bool) or budget < 1 or budget > candidates.shape[0]:
        raise ValueError("budget must be between one and the candidate count")
    selected_policy = ReacquisitionPolicy(policy)
    rng = np.random.default_rng(seed)
    if selected_policy is ReacquisitionPolicy.UNIFORM:
        indices = rng.choice(candidates.shape[0], size=budget, replace=False)
        return np.sort(indices).astype(np.int64), 0, False

    if localization is None:
        raise ValueError("risk_targeted selection requires localization")
    if observed_points is None:
        raise ValueError("risk_targeted selection requires observed_points")
    observed = np.asarray(observed_points, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("observed_points must have shape (n, 3)")
    if localization.boundary_faces.size and (
        np.min(localization.boundary_faces) < 0
        or np.max(localization.boundary_faces) >= observed.shape[0]
    ):
        raise ValueError("localization boundary index is out of range")
    flagged = localization.boundary_face_risk > float(risk_threshold)
    anchors = np.mean(observed[localization.boundary_faces[flagged]], axis=1)
    if anchors.shape[0] == 0:
        indices = rng.choice(candidates.shape[0], size=budget, replace=False)
        return np.sort(indices).astype(np.int64), 0, True
    centered = observed - np.mean(observed, axis=0)
    covariance = centered.T @ centered / max(observed.shape[0], 1)
    normal = np.linalg.eigh(covariance)[1][:, 0]
    projected_candidates = candidates - np.outer(candidates @ normal, normal)
    projected_anchors = anchors - np.outer(anchors @ normal, normal)
    distances = np.linalg.norm(
        projected_anchors[:, None, :] - projected_candidates[None, :, :],
        axis=2,
    )
    orders = np.argsort(distances, axis=1, kind="stable")
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for neighbor_rank in range(candidates.shape[0]):
        for anchor_index in range(anchors.shape[0]):
            candidate_index = int(orders[anchor_index, neighbor_rank])
            if candidate_index in chosen_set:
                continue
            chosen.append(candidate_index)
            chosen_set.add(candidate_index)
            if len(chosen) == budget:
                break
        if len(chosen) == budget:
            break
    indices = np.asarray(chosen, dtype=np.int64)
    return np.sort(indices).astype(np.int64), int(anchors.shape[0]), False


def _select_indices_with_points(
    candidate_points: FloatArray,
    observed_points: FloatArray,
    *,
    budget: int,
    policy: ReacquisitionPolicy,
    seed: int,
    localization: BoundaryBridgeLocalization,
    risk_threshold: float,
) -> tuple[IntArray, int, bool]:
    if policy is ReacquisitionPolicy.UNIFORM:
        return select_reacquisition_indices(
            candidate_points,
            budget=budget,
            policy=policy,
            seed=seed,
        )
    return select_reacquisition_indices(
        candidate_points,
        budget=budget,
        policy=policy,
        seed=seed,
        observed_points=observed_points,
        localization=localization,
        risk_threshold=risk_threshold,
    )


def _split_reference_pool(
    case: SyntheticCase,
    config: ReacquisitionConfig,
) -> tuple[FloatArray, FloatArray]:
    count = config.candidate_pool_count
    return case.reference_points[:count], case.reference_points[count:]


def _case_with_points(
    base: SyntheticCase,
    points: FloatArray,
    labels: IntArray,
    evaluation_reference: FloatArray,
) -> SyntheticCase:
    return SyntheticCase(
        family=base.family,
        split=base.split,
        points=points,
        reference_points=evaluation_reference,
        expected_components=base.expected_components,
        characteristic_length=base.characteristic_length,
        variation=base.variation,
        seed=base.seed,
        expected_surface_betti=base.expected_surface_betti,
        point_component_labels=labels,
    )


def _reconstruct(
    case: SyntheticCase,
    config: ReacquisitionConfig,
    *,
    evaluation_seed: int,
) -> tuple[ReconstructionSnapshot, BoundaryBridgeLocalization]:
    filtration = AlphaFiltration.from_points(case.points)
    adaptive = pca_anisotropic_filtration(
        filtration,
        k_neighbors=config.k_neighbors,
        max_normal_penalty=config.b5_max_normal_penalty,
    )
    mesh = adaptive.surface_at(config.b5_scale_multiplier)
    endpoints = evaluate_surface(
        mesh,
        case.reference_points,
        expected_components=case.expected_components,
        expected_betti=case.expected_surface_betti,
        vertex_component_labels=case.point_component_labels,
        characteristic_length=case.characteristic_length,
        sample_count=config.surface_sample_count,
        threshold_fraction=config.fscore_threshold_fraction,
        seed=evaluation_seed,
    )
    localization = boundary_bridge_localization(
        adaptive,
        scale_multiplier=config.b5_scale_multiplier,
        k_neighbors=config.k_neighbors,
        normal_coherence_threshold=config.normal_coherence_threshold,
        normal_edge_threshold=config.normal_edge_threshold,
        length_edge_threshold=config.length_edge_threshold,
    )
    if endpoints.betti_error is None:
        raise RuntimeError("Phase-0 case requires a declared Betti target")
    if endpoints.labeled_false_bridge_edges is None:
        raise RuntimeError("Phase-0 case requires evaluation-only component labels")
    selected_k = min(config.k_neighbors, case.points.shape[0] - 1)
    distances, neighbors = cKDTree(case.points).query(
        case.points,
        k=selected_k + 1,
        workers=1,
    )
    cross_knn_fraction = float(
        np.mean(
            case.point_component_labels[neighbors[:, 1:]]
            != case.point_component_labels[:, None]
        )
    )
    median_knn_spacing = float(np.median(np.median(distances[:, 1:], axis=1)))
    sheet_gap = float(case.variation["sheet_gap"])
    return (
        ReconstructionSnapshot(
            fscore=endpoints.fscore,
            normalized_chamfer_squared=endpoints.normalized_chamfer_squared,
            normalized_hausdorff=endpoints.normalized_hausdorff,
            component_error=endpoints.component_error,
            betti_error=endpoints.betti_error,
            labeled_false_bridge_edges=endpoints.labeled_false_bridge_edges,
            labeled_false_bridge_faces=int(endpoints.labeled_false_bridge_faces or 0),
            boundary_edges=endpoints.boundary_edges,
            nonmanifold_edges=endpoints.nonmanifold_edges,
            flagged_boundary_faces=int(
                np.count_nonzero(
                    localization.boundary_face_risk > config.risk_threshold
                )
            ),
            flagged_boundary_edges=int(
                np.count_nonzero(
                    localization.boundary_edge_risk > config.risk_threshold
                )
            ),
            cross_knn_fraction=cross_knn_fraction,
            median_knn_spacing=median_knn_spacing,
            gap_to_spacing_ratio=sheet_gap / median_knn_spacing,
            risk_route=localization.route,
        ),
        localization,
    )


def _comparison(
    trials: Sequence[ReacquisitionTrial],
    budget: int,
    tolerance: float,
) -> BudgetComparison:
    uniform = sorted(
        (
            trial
            for trial in trials
            if trial.added_point_count == budget
            and trial.policy is ReacquisitionPolicy.UNIFORM
        ),
        key=lambda trial: trial.repeat,
    )
    targeted = sorted(
        (
            trial
            for trial in trials
            if trial.added_point_count == budget
            and trial.policy is ReacquisitionPolicy.RISK_TARGETED
        ),
        key=lambda trial: trial.repeat,
    )
    if [trial.repeat for trial in uniform] != [trial.repeat for trial in targeted]:
        raise RuntimeError("uniform and targeted trials must be paired")

    def mean(rows: Sequence[ReacquisitionTrial], name: str) -> float:
        return float(np.mean([getattr(row.reacquired, name) for row in rows]))

    uniform_fscore = mean(uniform, "fscore")
    targeted_fscore = mean(targeted, "fscore")
    uniform_edges = mean(uniform, "labeled_false_bridge_edges")
    targeted_edges = mean(targeted, "labeled_false_bridge_edges")
    uniform_faces = mean(uniform, "labeled_false_bridge_faces")
    targeted_faces = mean(targeted, "labeled_false_bridge_faces")
    uniform_betti = mean(uniform, "betti_error")
    targeted_betti = mean(targeted, "betti_error")
    uniform_components = mean(uniform, "component_error")
    targeted_components = mean(targeted, "component_error")
    uniform_cross_knn = mean(uniform, "cross_knn_fraction")
    targeted_cross_knn = mean(targeted, "cross_knn_fraction")
    fscore_nonregression = targeted_fscore + tolerance >= uniform_fscore
    topology_nonregression = targeted_betti <= uniform_betti
    component_nonregression = targeted_components <= uniform_components
    sampling_information_improved = targeted_cross_knn < uniform_cross_knn
    bridge_edge_improved = targeted_edges < uniform_edges
    bridge_face_improved = targeted_faces < uniform_faces
    return BudgetComparison(
        added_point_count=budget,
        repeat_count=len(uniform),
        uniform_mean_fscore=uniform_fscore,
        targeted_mean_fscore=targeted_fscore,
        uniform_mean_bridge_edges=uniform_edges,
        targeted_mean_bridge_edges=targeted_edges,
        uniform_mean_bridge_faces=uniform_faces,
        targeted_mean_bridge_faces=targeted_faces,
        uniform_mean_betti_error=uniform_betti,
        targeted_mean_betti_error=targeted_betti,
        uniform_mean_component_error=uniform_components,
        targeted_mean_component_error=targeted_components,
        uniform_mean_cross_knn_fraction=uniform_cross_knn,
        targeted_mean_cross_knn_fraction=targeted_cross_knn,
        targeted_bridge_edge_win_count=sum(
            target.reacquired.labeled_false_bridge_edges
            < control.reacquired.labeled_false_bridge_edges
            for control, target in zip(uniform, targeted, strict=True)
        ),
        targeted_bridge_face_win_count=sum(
            target.reacquired.labeled_false_bridge_faces
            < control.reacquired.labeled_false_bridge_faces
            for control, target in zip(uniform, targeted, strict=True)
        ),
        fscore_nonregression=fscore_nonregression,
        topology_nonregression=topology_nonregression,
        component_nonregression=component_nonregression,
        sampling_information_improved=sampling_information_improved,
        bridge_edge_improved=bridge_edge_improved,
        bridge_face_improved=bridge_face_improved,
        phase0_gate_passed=(
            fscore_nonregression
            and topology_nonregression
            and component_nonregression
            and targeted_components < uniform_components
            and bridge_edge_improved
            and bridge_face_improved
        ),
    )


def evaluate_risk_targeted_reacquisition(
    config: ReacquisitionConfig | None = None,
) -> ReacquisitionExperimentResult:
    """Run paired uniform/risk-targeted thin-gap ROI-rescan trials."""

    selected = ReacquisitionConfig() if config is None else config
    trials: list[ReacquisitionTrial] = []
    total_reference = (
        selected.candidate_pool_count + selected.evaluation_reference_count
    )
    for repeat in range(selected.repeats):
        case_seed = selected.seed + repeat * 10_007
        generated = make_synthetic_case(
            SyntheticFamily.OPPOSING_SHEETS,
            split=PanelSplit.HELD_OUT,
            point_count=selected.base_point_count,
            reference_count=total_reference,
            seed=case_seed,
        )
        candidate_pool, evaluation_reference = _split_reference_pool(
            generated, selected
        )
        base = _case_with_points(
            generated,
            generated.points,
            generated.point_component_labels,
            evaluation_reference,
        )
        baseline, localization = _reconstruct(
            base,
            selected,
            evaluation_seed=selected.seed + repeat * 100_003,
        )
        hidden_candidate_labels = (candidate_pool[:, 2] > 0.0).astype(np.int64)
        for budget in sorted(selected.added_point_counts):
            for policy_index, policy in enumerate(ReacquisitionPolicy):
                policy_seed = case_seed + budget * 1_009 + policy_index * 1_000_003
                indices, anchor_count, fallback = _select_indices_with_points(
                    candidate_pool,
                    base.points,
                    budget=budget,
                    policy=policy,
                    seed=policy_seed,
                    localization=localization,
                    risk_threshold=selected.risk_threshold,
                )
                noise_rng = np.random.default_rng(policy_seed + 17)
                added = candidate_pool[indices] + noise_rng.normal(
                    scale=float(base.variation.get("noise", 0.0)),
                    size=(budget, 3),
                )
                reacquired_case = _case_with_points(
                    base,
                    np.vstack((base.points, added)),
                    np.concatenate(
                        (base.point_component_labels, hidden_candidate_labels[indices])
                    ),
                    evaluation_reference,
                )
                reacquired, _ = _reconstruct(
                    reacquired_case,
                    selected,
                    evaluation_seed=policy_seed + 31,
                )
                digest = hashlib.sha256(
                    np.asarray(indices, dtype="<i8").tobytes()
                ).hexdigest()
                trials.append(
                    ReacquisitionTrial(
                        repeat=repeat,
                        case_seed=case_seed,
                        policy=policy,
                        added_point_count=budget,
                        selected_index_sha256=digest,
                        risk_anchor_count=anchor_count,
                        fell_back_to_uniform=fallback,
                        baseline=baseline,
                        reacquired=reacquired,
                    )
                )
    comparisons = tuple(
        _comparison(
            trials,
            budget,
            selected.fscore_nonregression_tolerance,
        )
        for budget in sorted(selected.added_point_counts)
    )
    return ReacquisitionExperimentResult(
        artifact_schema="pftf_alpha_risk_targeted_reacquisition_phase0/v1",
        role="synthetic_roi_rescan_feasibility_only",
        family=SyntheticFamily.OPPOSING_SHEETS.value,
        split=PanelSplit.HELD_OUT.value,
        reconstruction="B5_frozen_multiplier",
        policy_information_boundary=(
            "selection uses observed points, frozen boundary risk, and candidate "
            "positions only; component labels and evaluation reference are hidden"
        ),
        config=selected,
        trials=tuple(trials),
        comparisons=comparisons,
        phase0_supported=any(row.phase0_gate_passed for row in comparisons),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-points", type=int, default=48)
    parser.add_argument("--evaluation-reference", type=int, default=4096)
    parser.add_argument("--candidate-pool", type=int, default=4096)
    parser.add_argument("--added-points", type=int, nargs="+", default=[12, 24, 36])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ReacquisitionConfig(
        base_point_count=args.base_points,
        evaluation_reference_count=args.evaluation_reference,
        candidate_pool_count=args.candidate_pool,
        added_point_counts=tuple(args.added_points),
        repeats=args.repeats,
        surface_sample_count=args.surface_samples,
        seed=args.seed,
    )
    result = evaluate_risk_targeted_reacquisition(config)
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
