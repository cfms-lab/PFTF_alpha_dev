"""M1 — weighted / regular (power) alpha complex with density weights.

Unlike B4/B5/P1/P2, which select subcomplexes of one fixed isotropic Delaunay
triangulation, M1 changes the connectivity itself: it builds the regular
triangulation of weighted points (via a 4D lift) and then scores cells with the
same B4 density rule. Weight scale 0 reproduces B4 exactly, so M1 strictly
generalizes B4. This is a floating-point research construction (Qhull lift), not
an exact weighted-alpha complex; it keeps ``promotion_supported`` false. See
docs/M1_WEIGHTED_ALPHA_DESIGN.md.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .adaptive import (
    AdaptiveCellFiltration,
    density_scaled_filtration,
    knn_scales,
    pca_anisotropic_filtration,
)
from .baselines import BenchmarkConfig
from .filtration import AlphaFiltration
from .selection import ObjectiveTerms
from .surface import SurfaceEndpointMetrics, evaluate_surface
from .synthetic import PanelSplit, SyntheticCase, make_minimal_panel

FloatArray = np.ndarray
IntArray = np.ndarray


class PointSubmersionError(ValueError):
    """A weight scale removed at least one point from the regular triangulation."""


def regular_triangulation(points: FloatArray, weights: FloatArray) -> IntArray:
    """Regular (weighted) Delaunay tetrahedra via the lower 4D convex hull.

    ``weights = 0`` reproduces the ordinary Delaunay triangulation exactly.
    """

    point_array = np.asarray(points, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if weight_array.shape != (point_array.shape[0],):
        raise ValueError("weights must have shape (n,)")
    if not np.all(np.isfinite(weight_array)):
        raise ValueError("weights must be finite")
    heights = np.einsum("ij,ij->i", point_array, point_array) - weight_array
    lifted = np.hstack([point_array, heights[:, None]])
    try:
        hull = ConvexHull(lifted)
    except QhullError as error:
        raise ValueError(f"regular triangulation failed: {error}") from error
    lower = hull.equations[:, 3] < -1.0e-12
    cells = hull.simplices[lower]
    distinct = np.array([len(set(cell.tolist())) == 4 for cell in cells], dtype=bool)
    return np.ascontiguousarray(np.sort(cells[distinct], axis=1), dtype=np.int64)


def _weighted_orthoradius_squared(
    cell_points: FloatArray, cell_weights: FloatArray
) -> float:
    """Squared power (orthogonal) radius of one weighted tetrahedron.

    Reduces to the ordinary squared circumradius when all weights are zero, so
    ``weight_scale = 0`` keeps M1 identical to B4.
    """

    base = cell_points[0]
    matrix = 2.0 * (cell_points[1:] - base)
    lifted = np.einsum("ij,ij->i", cell_points, cell_points) - cell_weights
    rhs = lifted[1:] - lifted[0]
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return 0.0
    offset = center - base
    return float(np.dot(offset, offset) - cell_weights[0])


def weighted_alpha_filtration(
    points: FloatArray,
    *,
    k_neighbors: int,
    weight_scale: float,
) -> AdaptiveCellFiltration:
    """Regular-triangulation connectivity scored by the proper weighted (power)
    circumradius, normalized by local kNN spacing. ``weight_scale = 0`` equals B4.
    """

    if not np.isfinite(weight_scale) or weight_scale < 0.0:
        raise ValueError("weight_scale must be finite and non-negative")
    point_array = np.asarray(points, dtype=np.float64)
    spacing = knn_scales(point_array, k_neighbors=k_neighbors)
    weights = (float(weight_scale) * spacing) ** 2
    cells = regular_triangulation(point_array, weights)
    if np.unique(cells).size != point_array.shape[0]:
        raise PointSubmersionError(
            f"weight_scale={weight_scale} submerges points from the triangulation"
        )
    scores = np.empty(cells.shape[0], dtype=np.float64)
    for index, cell in enumerate(cells):
        radius_squared = _weighted_orthoradius_squared(
            point_array[cell], weights[cell]
        )
        simplex_scale = float(np.exp(np.mean(np.log(spacing[cell]))))
        scores[index] = math.sqrt(max(0.0, radius_squared)) / simplex_scale
    return AdaptiveCellFiltration(
        points=point_array,
        top_simplices=cells,
        scores=scores,
        method="M1_weighted_power_alpha",
        diagnostics={
            "k_neighbors": float(k_neighbors),
            "weight_scale": float(weight_scale),
        },
    )


def _objective_terms(
    endpoints: SurfaceEndpointMetrics, *, maximum_faces: int
) -> ObjectiveTerms:
    return ObjectiveTerms(
        geometry=(
            endpoints.normalized_chamfer_squared + endpoints.normalized_hausdorff
        ),
        topology=float(endpoints.component_error),
        stability=0.0,
        complexity=(
            endpoints.nonmanifold_edges / max(endpoints.edges, 1)
            + endpoints.faces / max(maximum_faces, 1)
        ),
    )


def _endpoints(
    adaptive: AdaptiveCellFiltration,
    multiplier: float,
    case: SyntheticCase,
    config: BenchmarkConfig,
    *,
    seed: int,
) -> SurfaceEndpointMetrics:
    return evaluate_surface(
        adaptive.surface_at(multiplier),
        case.reference_points,
        expected_components=case.expected_components,
        characteristic_length=case.characteristic_length,
        expected_betti=case.expected_surface_betti,
        vertex_component_labels=case.point_component_labels,
        sample_count=config.surface_sample_count,
        threshold_fraction=config.fscore_threshold_fraction,
        seed=seed,
    )


def _freeze_multiplier(
    adaptives: Sequence[AdaptiveCellFiltration],
    cases: Sequence[SyntheticCase],
    config: BenchmarkConfig,
    *,
    candidate_budget: int,
) -> float:
    pooled = np.concatenate([adaptive.critical_values() for adaptive in adaptives])
    positive = pooled[pooled > 0.0]
    if positive.size == 0:
        raise ValueError("no positive scores to calibrate")
    lower = float(np.quantile(positive, 0.02))
    upper = float(np.quantile(positive, 0.95))
    if upper <= lower:
        candidates = np.asarray([lower], dtype=np.float64)
    else:
        candidates = np.geomspace(lower, upper, num=candidate_budget)
    best_multiplier = float(candidates[0])
    best_objective = float("inf")
    for multiplier in candidates:
        objectives: list[float] = []
        for case_index, (case, adaptive) in enumerate(
            zip(cases, adaptives, strict=True)
        ):
            endpoints = _endpoints(
                adaptive,
                float(multiplier),
                case,
                config,
                seed=config.seed + case.seed + 90_000 + case_index,
            )
            terms = _objective_terms(endpoints, maximum_faces=max(endpoints.faces, 1))
            objectives.append(config.adaptive_weights.apply(terms))
        mean_objective = float(np.mean(objectives))
        if mean_objective < best_objective:
            best_objective = mean_objective
            best_multiplier = float(multiplier)
    return best_multiplier


@dataclass(frozen=True)
class MethodAggregate:
    method: str
    weight_scale: float | None
    frozen_multiplier: float
    mean_fscore: float
    mean_geometry_loss: float
    component_error_sum: int
    betti_error_sum: int
    labeled_false_bridge_edges_sum: int
    labeled_false_bridge_faces_sum: int
    nonmanifold_edges_sum: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _aggregate(
    method: str,
    weight_scale: float | None,
    adaptives: Sequence[AdaptiveCellFiltration],
    cases: Sequence[SyntheticCase],
    config: BenchmarkConfig,
    *,
    candidate_budget: int,
) -> MethodAggregate:
    multiplier = _freeze_multiplier(
        adaptives, cases, config, candidate_budget=candidate_budget
    )
    rows = [
        _endpoints(
            adaptive,
            multiplier,
            case,
            config,
            seed=config.seed + case.seed + 60_000,
        )
        for case, adaptive in zip(cases, adaptives, strict=True)
    ]
    return MethodAggregate(
        method=method,
        weight_scale=weight_scale,
        frozen_multiplier=multiplier,
        mean_fscore=float(np.mean([ep.fscore for ep in rows])),
        mean_geometry_loss=float(
            np.mean(
                [ep.normalized_chamfer_squared + ep.normalized_hausdorff for ep in rows]
            )
        ),
        component_error_sum=sum(ep.component_error for ep in rows),
        betti_error_sum=sum(int(ep.betti_error or 0) for ep in rows),
        labeled_false_bridge_edges_sum=sum(
            int(ep.labeled_false_bridge_edges or 0) for ep in rows
        ),
        labeled_false_bridge_faces_sum=sum(
            int(ep.labeled_false_bridge_faces or 0) for ep in rows
        ),
        nonmanifold_edges_sum=sum(ep.nonmanifold_edges for ep in rows),
    )


def _no_regression(candidate: MethodAggregate, baseline: MethodAggregate) -> bool:
    tol = 1.0e-12
    return (
        candidate.mean_fscore >= baseline.mean_fscore - tol
        and candidate.mean_geometry_loss <= baseline.mean_geometry_loss + tol
        and candidate.component_error_sum <= baseline.component_error_sum
        and candidate.betti_error_sum <= baseline.betti_error_sum
        and candidate.labeled_false_bridge_edges_sum
        <= baseline.labeled_false_bridge_edges_sum
        and candidate.labeled_false_bridge_faces_sum
        <= baseline.labeled_false_bridge_faces_sum
        and candidate.nonmanifold_edges_sum <= baseline.nonmanifold_edges_sum
    )


def _strictly_improves(candidate: MethodAggregate, baseline: MethodAggregate) -> bool:
    return (
        candidate.mean_geometry_loss < baseline.mean_geometry_loss
        or candidate.betti_error_sum < baseline.betti_error_sum
        or candidate.component_error_sum < baseline.component_error_sum
        or candidate.labeled_false_bridge_edges_sum
        < baseline.labeled_false_bridge_edges_sum
        or candidate.mean_fscore > baseline.mean_fscore
    )


@dataclass(frozen=True)
class M1AblationResult:
    weight_scales: tuple[float, ...]
    submerged_scales: tuple[float, ...]
    b4: MethodAggregate
    b5: MethodAggregate
    m1_by_scale: tuple[MethodAggregate, ...]
    frozen_weight_scale: float
    m1_promotes_over_baselines: bool
    m1_dominates_b4: bool
    b4_dominant_weight_scale: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": "pftf_alpha_m1_weighted_alpha_ablation/v1",
            "evaluation_role": "calibration_only_ablation",
            "construction": "regular_weighted_delaunay_via_lift_floating_qhull",
            "scoring": "weighted_power_circumradius_over_knn_spacing",
            "weight_model": "density_w_i=(scale*knn_spacing_i)^2",
            "weight_scales": list(self.weight_scales),
            "submerged_scales": list(self.submerged_scales),
            "b4_scale0_baseline": self.b4.to_dict(),
            "b5_baseline": self.b5.to_dict(),
            "m1_by_scale": [aggregate.to_dict() for aggregate in self.m1_by_scale],
            "frozen_weight_scale": self.frozen_weight_scale,
            "m1_promotes_over_baselines": self.m1_promotes_over_baselines,
            "m1_dominates_b4": self.m1_dominates_b4,
            "b4_dominant_weight_scale": self.b4_dominant_weight_scale,
            "promotion_supported": False,
            "claim_boundary": (
                "M1 is a floating-point regular-alpha construction; weight scale 0 "
                "equals B4. It changes connectivity, not exactness, does not resolve "
                "the thin-gap (opposing_sheets) bridge (left to M2), and does not by "
                "itself justify promotion."
            ),
        }


def evaluate_m1_ablation(
    *,
    point_count: int = 80,
    reference_count: int = 2048,
    candidate_budget: int = 12,
    weight_scales: Sequence[float] = (0.0, 0.125, 0.25, 0.375, 0.5),
    seed: int = 20_260_724,
) -> M1AblationResult:
    """Freeze one weight scale on calibration and compare M1 to B4 and B5."""

    config = BenchmarkConfig(seed=seed)
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=point_count,
        reference_count=reference_count,
        seed=seed,
    )
    filtrations = tuple(AlphaFiltration.from_points(case.points) for case in cases)
    b4 = _aggregate(
        "B4_density_scaled",
        0.0,
        tuple(
            density_scaled_filtration(filt, k_neighbors=config.adaptive_k_neighbors)
            for filt in filtrations
        ),
        cases,
        config,
        candidate_budget=candidate_budget,
    )
    b5 = _aggregate(
        "B5_pca_anisotropic",
        None,
        tuple(
            pca_anisotropic_filtration(
                filt,
                k_neighbors=config.adaptive_k_neighbors,
                max_normal_penalty=config.b5_max_normal_penalty,
            )
            for filt in filtrations
        ),
        cases,
        config,
        candidate_budget=candidate_budget,
    )

    m1_results: list[MethodAggregate] = []
    submerged: list[float] = []
    for scale in weight_scales:
        try:
            adaptives = tuple(
                weighted_alpha_filtration(
                    case.points,
                    k_neighbors=config.adaptive_k_neighbors,
                    weight_scale=float(scale),
                )
                for case in cases
            )
        except PointSubmersionError:
            submerged.append(float(scale))
            continue
        m1_results.append(
            _aggregate(
                "M1_weighted_alpha",
                float(scale),
                adaptives,
                cases,
                config,
                candidate_budget=candidate_budget,
            )
        )

    positive = [row for row in m1_results if row.weight_scale]
    qualifying = [
        row
        for row in positive
        if _no_regression(row, b4)
        and _no_regression(row, b5)
        and (_strictly_improves(row, b4) or _strictly_improves(row, b5))
    ]
    if qualifying:
        chosen = min(
            qualifying,
            key=lambda row: (row.mean_geometry_loss, row.weight_scale or 0.0),
        )
        frozen_scale = chosen.weight_scale or 0.0
        promotes = True
    else:
        frozen_scale = 0.0
        promotes = False

    b4_dominating = [
        row
        for row in positive
        if _no_regression(row, b4) and _strictly_improves(row, b4)
    ]
    b4_dominant = (
        min(b4_dominating, key=lambda row: (row.mean_geometry_loss, row.weight_scale))
        if b4_dominating
        else None
    )

    return M1AblationResult(
        weight_scales=tuple(float(scale) for scale in weight_scales),
        submerged_scales=tuple(submerged),
        b4=b4,
        b5=b5,
        m1_by_scale=tuple(m1_results),
        frozen_weight_scale=frozen_scale,
        m1_promotes_over_baselines=promotes,
        m1_dominates_b4=b4_dominant is not None,
        b4_dominant_weight_scale=(
            None if b4_dominant is None else b4_dominant.weight_scale
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the M1 weighted-alpha calibration ablation."
    )
    parser.add_argument("--point-count", type=int, default=80)
    parser.add_argument("--reference-count", type=int, default=2048)
    parser.add_argument("--candidate-budget", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20_260_724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/m1_weighted_alpha_ablation.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = evaluate_m1_ablation(
        point_count=args.point_count,
        reference_count=args.reference_count,
        candidate_budget=args.candidate_budget,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[m1] frozen_weight_scale={result.frozen_weight_scale} "
        f"promotes_over_baselines={result.m1_promotes_over_baselines}",
        flush=True,
    )
    print(f"Wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
