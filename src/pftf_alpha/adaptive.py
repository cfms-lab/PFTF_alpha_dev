"""Closure-preserving adaptive Delaunay methods B4, B5, P1, and P2."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .filtration import AlphaFiltration
from .geometry import intrinsic_circumsphere
from .metrics import metric_circumradius_squared
from .pftf import pftf_relation_field
from .surface import SurfaceMesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class LocalNeighborhoodGeometry:
    """kNN scale and PCA frame used by prior-art adaptive baselines."""

    scales: FloatArray
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    planarity: FloatArray
    k_neighbors: int


@dataclass(frozen=True)
class AdaptiveCellFiltration:
    """Dimensionless score for every Delaunay tetrahedron.

    A threshold selects top cells. Taking the complete closure of those cells
    guarantees a global simplicial subcomplex even though scores came from
    spatially varying local scales or metrics.
    """

    points: FloatArray
    top_simplices: IntArray
    scores: FloatArray
    method: str
    diagnostics: Mapping[str, float]
    cell_confidence: FloatArray | None = None
    fallback_mask: BoolArray | None = None
    guard_scores: FloatArray | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        cells = np.asarray(self.top_simplices, dtype=np.int64)
        scores = np.asarray(self.scores, dtype=np.float64)
        confidence = (
            None
            if self.cell_confidence is None
            else np.asarray(self.cell_confidence, dtype=np.float64)
        )
        fallback = (
            None
            if self.fallback_mask is None
            else np.asarray(self.fallback_mask, dtype=bool)
        )
        guard_scores = (
            None
            if self.guard_scores is None
            else np.asarray(self.guard_scores, dtype=np.float64)
        )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("adaptive filtration requires points with shape (n, 3)")
        if cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError("top_simplices must have shape (m, 4)")
        if scores.shape != (cells.shape[0],):
            raise ValueError("scores must have shape (m,)")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
            raise ValueError("adaptive scores must be finite and non-negative")
        if confidence is not None and (
            confidence.shape != scores.shape
            or not np.all(np.isfinite(confidence))
            or np.any((confidence < 0.0) | (confidence > 1.0))
        ):
            raise ValueError("cell_confidence must have score shape and lie in [0, 1]")
        if fallback is not None and fallback.shape != scores.shape:
            raise ValueError("fallback_mask must have the same shape as scores")
        if guard_scores is not None and (
            guard_scores.shape != scores.shape
            or not np.all(np.isfinite(guard_scores))
            or np.any(guard_scores < 0.0)
        ):
            raise ValueError(
                "guard_scores must have score shape and be finite and non-negative"
            )
        object.__setattr__(self, "points", np.ascontiguousarray(points))
        object.__setattr__(self, "top_simplices", np.ascontiguousarray(cells))
        object.__setattr__(self, "scores", np.ascontiguousarray(scores))
        if confidence is not None:
            object.__setattr__(
                self,
                "cell_confidence",
                np.ascontiguousarray(confidence),
            )
        if fallback is not None:
            object.__setattr__(
                self,
                "fallback_mask",
                np.ascontiguousarray(fallback),
            )
        if guard_scores is not None:
            object.__setattr__(
                self,
                "guard_scores",
                np.ascontiguousarray(guard_scores),
            )

    def critical_values(self) -> FloatArray:
        return np.unique(self.scores)

    def selected_cell_count(self, scale_multiplier: float) -> int:
        threshold = _finite_nonnegative(scale_multiplier, "scale_multiplier")
        return int(np.count_nonzero(self.scores <= threshold))

    def surface_at(self, scale_multiplier: float) -> SurfaceMesh:
        """Regularized boundary of selected tetrahedra and their closure."""
        threshold = _finite_nonnegative(scale_multiplier, "scale_multiplier")
        selected = self.scores <= threshold
        face_counts: Counter[tuple[int, int, int]] = Counter()
        for cell in self.top_simplices[selected]:
            ordered = tuple(sorted(int(vertex) for vertex in cell))
            face_counts.update(combinations(ordered, 3))
        boundary = sorted(
            face for face, incidence in face_counts.items() if incidence == 1
        )
        faces = (
            np.asarray(boundary, dtype=np.int64)
            if boundary
            else np.empty((0, 3), dtype=np.int64)
        )
        return SurfaceMesh(vertices=self.points, faces=faces)

    def diagnostics_at(self, scale_multiplier: float) -> dict[str, float]:
        """Combine static diagnostics with closure and guard evidence."""

        threshold = _finite_nonnegative(scale_multiplier, "scale_multiplier")
        selected = self.scores <= threshold
        selected_count = int(np.count_nonzero(selected))
        result = dict(self.diagnostics)
        result["selected_cell_count"] = float(selected_count)

        selected_cells = self.top_simplices[selected]
        closure_vertices: set[int] = set()
        closure_edges: set[tuple[int, int]] = set()
        face_counts: Counter[tuple[int, int, int]] = Counter()
        for cell in selected_cells:
            ordered = tuple(sorted(int(vertex) for vertex in cell))
            closure_vertices.update(ordered)
            closure_edges.update(combinations(ordered, 2))
            face_counts.update(combinations(ordered, 3))
        result.update(
            {
                "closure_vertex_count": float(len(closure_vertices)),
                "closure_edge_count": float(len(closure_edges)),
                "closure_face_count": float(len(face_counts)),
                "boundary_face_count": float(
                    sum(incidence == 1 for incidence in face_counts.values())
                ),
                "face_incidence_over_two_count": float(
                    sum(incidence > 2 for incidence in face_counts.values())
                ),
                "downward_closure_complete": 1.0,
            }
        )

        if self.cell_confidence is not None:
            selected_confidence = self.cell_confidence[selected]
            if selected_confidence.size:
                result.update(
                    {
                        "selected_confidence_min": float(np.min(selected_confidence)),
                        "selected_confidence_median": float(
                            np.median(selected_confidence)
                        ),
                        "selected_confidence_max": float(np.max(selected_confidence)),
                    }
                )
            else:
                result.update(
                    {
                        "selected_confidence_min": 0.0,
                        "selected_confidence_median": 0.0,
                        "selected_confidence_max": 0.0,
                    }
                )
        if self.fallback_mask is not None:
            fallback_count = int(np.count_nonzero(self.fallback_mask))
            selected_fallback_count = int(
                np.count_nonzero(self.fallback_mask & selected)
            )
            result["fallback_cell_count"] = float(fallback_count)
            result["fallback_fraction"] = fallback_count / max(len(self.scores), 1)
            result["selected_fallback_cell_count"] = float(selected_fallback_count)
            result["selected_fallback_fraction"] = selected_fallback_count / max(
                selected_count, 1
            )
        if self.guard_scores is not None:
            guarded_selected = selected
            if self.fallback_mask is not None:
                guarded_selected = guarded_selected & self.fallback_mask
            tolerance = (
                16.0 * np.finfo(np.float64).eps * np.maximum(self.guard_scores, 1.0)
            )
            selected_guard_violations = guarded_selected & (
                self.guard_scores > threshold + tolerance
            )
            guarded_selected_count = int(np.count_nonzero(guarded_selected))
            result["selected_guard_violation_count"] = float(
                np.count_nonzero(selected_guard_violations)
            )
            result["selected_guard_violation_fraction"] = float(
                np.count_nonzero(selected_guard_violations)
                / max(guarded_selected_count, 1)
            )
        return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_k(point_count: int, k_neighbors: int) -> int:
    if not 3 <= k_neighbors < point_count:
        raise ValueError("k_neighbors must satisfy 3 <= k_neighbors < point count")
    return int(k_neighbors)


def knn_scales(points: FloatArray, *, k_neighbors: int) -> FloatArray:
    """Median nonzero kNN distance at every observed point."""

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    selected_k = _validate_k(point_array.shape[0], k_neighbors)
    distances = cKDTree(point_array).query(point_array, k=selected_k + 1, workers=1)[0][
        :, 1:
    ]
    scales = np.median(distances, axis=1)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("kNN scales must be finite and positive")
    return np.ascontiguousarray(scales)


def local_neighborhood_geometry(
    points: FloatArray,
    *,
    k_neighbors: int,
) -> LocalNeighborhoodGeometry:
    """Fit local PCA frames without using reference geometry."""

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    selected_k = _validate_k(point_array.shape[0], k_neighbors)
    distances, neighbor_indices = cKDTree(point_array).query(
        point_array, k=selected_k + 1, workers=1
    )
    scales = np.median(distances[:, 1:], axis=1)
    eigenvalues = np.empty((point_array.shape[0], 3), dtype=np.float64)
    eigenvectors = np.empty((point_array.shape[0], 3, 3), dtype=np.float64)
    for point_index, indices in enumerate(neighbor_indices[:, 1:]):
        neighborhood = point_array[indices]
        centered = neighborhood - np.mean(neighborhood, axis=0)
        covariance = centered.T @ centered / selected_k
        values, vectors = np.linalg.eigh(covariance)
        eigenvalues[point_index] = np.maximum(values, 0.0)
        eigenvectors[point_index] = vectors

    denominator = np.maximum(eigenvalues[:, 2], np.finfo(np.float64).eps)
    planarity = np.clip(
        (eigenvalues[:, 1] - eigenvalues[:, 0]) / denominator,
        0.0,
        1.0,
    )
    return LocalNeighborhoodGeometry(
        scales=np.ascontiguousarray(scales),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        planarity=planarity,
        k_neighbors=selected_k,
    )


def density_scaled_filtration(
    filtration: AlphaFiltration,
    *,
    k_neighbors: int,
) -> AdaptiveCellFiltration:
    """B4 score: Euclidean circumradius divided by local kNN spacing."""

    if filtration.ambient_dimension != 3:
        raise ValueError("B4 requires a 3D Delaunay filtration")
    scales = knn_scales(filtration.points, k_neighbors=k_neighbors)
    scores = np.empty(filtration.top_simplices.shape[0], dtype=np.float64)
    for cell_index, cell in enumerate(filtration.top_simplices):
        radius_squared = intrinsic_circumsphere(filtration.points[cell]).radius_squared
        simplex_scale = float(np.exp(np.mean(np.log(scales[cell]))))
        scores[cell_index] = math.sqrt(radius_squared) / simplex_scale
    return AdaptiveCellFiltration(
        points=filtration.points,
        top_simplices=filtration.top_simplices,
        scores=scores,
        method="B4_knn_density_scaled",
        diagnostics={
            "k_neighbors": float(k_neighbors),
            "spacing_min": float(np.min(scales)),
            "spacing_median": float(np.median(scales)),
            "spacing_max": float(np.max(scales)),
        },
    )


def pca_anisotropic_filtration(
    filtration: AlphaFiltration,
    *,
    k_neighbors: int,
    max_normal_penalty: float,
) -> AdaptiveCellFiltration:
    """B5 score in a density-normalized PCA anisotropic SPD metric."""

    if filtration.ambient_dimension != 3:
        raise ValueError("B5 requires a 3D Delaunay filtration")
    if not math.isfinite(max_normal_penalty) or max_normal_penalty < 1.0:
        raise ValueError("max_normal_penalty must be finite and at least one")
    geometry = local_neighborhood_geometry(filtration.points, k_neighbors=k_neighbors)
    metrics = np.empty((filtration.points.shape[0], 3, 3), dtype=np.float64)
    penalties = 1.0 + (max_normal_penalty - 1.0) * geometry.planarity
    for point_index in range(filtration.points.shape[0]):
        eigenbasis = geometry.eigenvectors[point_index]
        metric_eigenvalues = np.array(
            [penalties[point_index] ** 2, 1.0, 1.0],
            dtype=np.float64,
        )
        metric = eigenbasis @ np.diag(metric_eigenvalues) @ eigenbasis.T
        metrics[point_index] = metric / geometry.scales[point_index] ** 2

    scores = np.empty(filtration.top_simplices.shape[0], dtype=np.float64)
    for cell_index, cell in enumerate(filtration.top_simplices):
        simplex_metric = np.mean(metrics[cell], axis=0)
        radius_squared = metric_circumradius_squared(
            filtration.points[cell], simplex_metric
        )
        scores[cell_index] = math.sqrt(radius_squared)
    return AdaptiveCellFiltration(
        points=filtration.points,
        top_simplices=filtration.top_simplices,
        scores=scores,
        method="B5_pca_anisotropic",
        diagnostics={
            "k_neighbors": float(k_neighbors),
            "max_normal_penalty": float(max_normal_penalty),
            "planarity_min": float(np.min(geometry.planarity)),
            "planarity_median": float(np.median(geometry.planarity)),
            "planarity_max": float(np.max(geometry.planarity)),
        },
    )


def pftf_local_metric_filtration(
    filtration: AlphaFiltration,
    *,
    k_neighbors: int,
    relation_gain: float,
    max_condition_number: float,
    density_contrast_scale: float,
    receiver_imbalance_weight: float,
) -> AdaptiveCellFiltration:
    """P1 score from a bounded directed-relation local SPD field.

    P1 uses confidence to blend uncertain point metrics toward the
    density-scaled identity. A hard confidence threshold and exact trusted
    fallback are intentionally reserved for P2.
    """

    if filtration.ambient_dimension != 3:
        raise ValueError("P1 requires a 3D Delaunay filtration")
    relation = pftf_relation_field(
        filtration.points,
        k_neighbors=k_neighbors,
        relation_gain=relation_gain,
        max_condition_number=max_condition_number,
        density_contrast_scale=density_contrast_scale,
        receiver_imbalance_weight=receiver_imbalance_weight,
    )
    field = relation.metric_field
    scores = np.empty(filtration.top_simplices.shape[0], dtype=np.float64)
    simplex_confidence = np.empty_like(scores)
    used_fallback = np.zeros(scores.shape[0], dtype=bool)
    for cell_index, cell in enumerate(filtration.top_simplices):
        simplex_scale = float(np.exp(np.mean(np.log(relation.scales[cell]))))
        decision = field.metric_for_simplex(
            cell,
            confidence_threshold=0.0,
            fallback_metric=np.eye(3, dtype=np.float64) / simplex_scale**2,
        )
        radius_squared = metric_circumradius_squared(
            filtration.points[cell],
            decision.metric,
            minimum_eigenvalue=field.minimum_eigenvalue,
        )
        scores[cell_index] = math.sqrt(radius_squared)
        simplex_confidence[cell_index] = decision.confidence
        used_fallback[cell_index] = decision.used_fallback

    normalized_metrics = field.matrices * relation.scales[:, None, None] ** 2
    metric_eigenvalues = np.linalg.eigvalsh(normalized_metrics)
    metric_condition = metric_eigenvalues[:, -1] / metric_eigenvalues[:, 0]
    return AdaptiveCellFiltration(
        points=filtration.points,
        top_simplices=filtration.top_simplices,
        scores=scores,
        method="P1_pftf_local_spd",
        diagnostics={
            "k_neighbors": float(k_neighbors),
            "relation_gain": float(relation_gain),
            "max_condition_number": float(max_condition_number),
            "density_contrast_scale": float(density_contrast_scale),
            "receiver_imbalance_weight": float(receiver_imbalance_weight),
            "point_confidence_min": float(np.min(field.confidence)),
            "point_confidence_median": float(np.median(field.confidence)),
            "point_confidence_max": float(np.max(field.confidence)),
            "simplex_confidence_min": float(np.min(simplex_confidence)),
            "simplex_confidence_median": float(np.median(simplex_confidence)),
            "simplex_confidence_max": float(np.max(simplex_confidence)),
            "relation_strength_median": float(np.median(relation.relation_strength)),
            "reciprocity_median": float(np.median(relation.reciprocity)),
            "metric_condition_max": float(np.max(metric_condition)),
            "fallback_fraction": float(np.mean(used_fallback)),
        },
        cell_confidence=simplex_confidence,
        fallback_mask=used_fallback,
    )


def pftf_confidence_fallback_filtration(
    filtration: AlphaFiltration,
    *,
    k_neighbors: int,
    relation_gain: float,
    max_condition_number: float,
    density_contrast_scale: float,
    receiver_imbalance_weight: float,
    confidence_threshold: float,
) -> AdaptiveCellFiltration:
    """P2 conservative guard using B4 on low-confidence P1 cells.

    For a low-confidence cell the combined score is the maximum of the P1 and
    trusted B4 scores. Therefore the cell must pass both tests under a shared
    threshold. This is a conservative SciPy prototype, not an exact CGAL
    fallback.
    """

    if not math.isfinite(confidence_threshold) or not (
        0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("confidence_threshold must lie in [0, 1]")
    p1 = pftf_local_metric_filtration(
        filtration,
        k_neighbors=k_neighbors,
        relation_gain=relation_gain,
        max_condition_number=max_condition_number,
        density_contrast_scale=density_contrast_scale,
        receiver_imbalance_weight=receiver_imbalance_weight,
    )
    trusted = density_scaled_filtration(
        filtration,
        k_neighbors=k_neighbors,
    )
    assert p1.cell_confidence is not None
    low_confidence = p1.cell_confidence < confidence_threshold
    scores = p1.scores.copy()
    scores[low_confidence] = np.maximum(
        p1.scores[low_confidence],
        trusted.scores[low_confidence],
    )
    tolerance = 16.0 * np.finfo(np.float64).eps * np.maximum(scores, 1.0)
    guard_violations = low_confidence & (
        (scores + tolerance < p1.scores) | (scores + tolerance < trusted.scores)
    )
    trusted_dominant = low_confidence & (trusted.scores > p1.scores)
    low_confidence_count = int(np.count_nonzero(low_confidence))
    diagnostics = dict(p1.diagnostics)
    diagnostics.update(
        {
            "confidence_threshold": float(confidence_threshold),
            "fallback_fraction": float(np.mean(low_confidence)),
            "fallback_score_dominant_fraction": float(
                np.count_nonzero(trusted_dominant) / max(low_confidence_count, 1)
            ),
            "fallback_guard_violation_count": float(np.count_nonzero(guard_violations)),
        }
    )
    return AdaptiveCellFiltration(
        points=filtration.points,
        top_simplices=filtration.top_simplices,
        scores=scores,
        method="P2_pftf_confidence_b4_guard",
        diagnostics=diagnostics,
        cell_confidence=p1.cell_confidence,
        fallback_mask=low_confidence,
        guard_scores=trusted.scores,
    )
