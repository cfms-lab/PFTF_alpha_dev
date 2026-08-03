"""Observed-only confidence and a continuous confidence-weighted filtration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from .adaptive import (
    AdaptiveCellFiltration,
    density_scaled_filtration,
    local_neighborhood_geometry,
)
from .filtration import AlphaFiltration

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ObservedPointConfidence:
    """Point confidence derived only from the two observed views."""

    point_confidence: FloatArray
    target_confidence: FloatArray
    nearest_anchor_distance: FloatArray
    anchor_plane_residual: FloatArray
    normal_alignment: FloatArray
    local_anchor_scale: FloatArray

    def __post_init__(self) -> None:
        point_confidence = np.asarray(self.point_confidence, dtype=np.float64)
        target_confidence = np.asarray(self.target_confidence, dtype=np.float64)
        nearest = np.asarray(self.nearest_anchor_distance, dtype=np.float64)
        residual = np.asarray(self.anchor_plane_residual, dtype=np.float64)
        alignment = np.asarray(self.normal_alignment, dtype=np.float64)
        local_scale = np.asarray(self.local_anchor_scale, dtype=np.float64)
        target_count = target_confidence.shape[0]
        for name, values in (
            ("nearest_anchor_distance", nearest),
            ("anchor_plane_residual", residual),
            ("normal_alignment", alignment),
            ("local_anchor_scale", local_scale),
        ):
            if values.shape != (target_count,):
                raise ValueError(f"{name} must match target confidence shape")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
        if point_confidence.ndim != 1 or point_confidence.shape[0] <= target_count:
            raise ValueError("point_confidence must contain anchor and target values")
        if np.any((point_confidence < 0.0) | (point_confidence > 1.0)):
            raise ValueError("point confidence must lie in [0, 1]")
        if np.any((target_confidence < 0.0) | (target_confidence > 1.0)):
            raise ValueError("target confidence must lie in [0, 1]")
        if np.any((alignment < 0.0) | (alignment > 1.0)):
            raise ValueError("normal_alignment must lie in [0, 1]")
        if np.any(local_scale <= 0.0):
            raise ValueError("local_anchor_scale must be positive")
        object.__setattr__(
            self, "point_confidence", np.ascontiguousarray(point_confidence)
        )
        object.__setattr__(
            self, "target_confidence", np.ascontiguousarray(target_confidence)
        )
        object.__setattr__(
            self, "nearest_anchor_distance", np.ascontiguousarray(nearest)
        )
        object.__setattr__(
            self, "anchor_plane_residual", np.ascontiguousarray(residual)
        )
        object.__setattr__(
            self, "normal_alignment", np.ascontiguousarray(alignment)
        )
        object.__setattr__(
            self, "local_anchor_scale", np.ascontiguousarray(local_scale)
        )


def observed_point_confidence(
    anchor_points: ArrayLike,
    target_points: ArrayLike,
    *,
    k_neighbors: int = 12,
) -> ObservedPointConfidence:
    """Measure continuous target-to-anchor agreement without truth access."""

    anchor = np.asarray(anchor_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    for name, points in (("anchor_points", anchor), ("target_points", target)):
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
            raise ValueError(f"{name} must have shape (n, 3) with n >= 4")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{name} must contain finite coordinates")
    if not isinstance(k_neighbors, int) or not 3 <= k_neighbors < min(
        anchor.shape[0], target.shape[0]
    ):
        raise ValueError("k_neighbors must be an integer valid for both views")

    anchor_geometry = local_neighborhood_geometry(
        anchor, k_neighbors=k_neighbors
    )
    target_geometry = local_neighborhood_geometry(
        target, k_neighbors=k_neighbors
    )
    tree = cKDTree(anchor)
    neighbor_distance, neighbor_index = tree.query(
        target, k=k_neighbors, workers=1
    )
    nearest = np.asarray(neighbor_distance[:, 0], dtype=np.float64)
    local_anchor_scale = np.exp(
        np.mean(np.log(anchor_geometry.scales[neighbor_index]), axis=1)
    )
    local_centroid = np.mean(anchor[neighbor_index], axis=1)
    covariance = np.einsum(
        "nki,nkj->nij",
        anchor[neighbor_index] - local_centroid[:, None, :],
        anchor[neighbor_index] - local_centroid[:, None, :],
    ) / float(k_neighbors)
    _, local_basis = np.linalg.eigh(covariance)
    anchor_normal = local_basis[:, :, 0]
    target_normal = target_geometry.eigenvectors[:, :, 0]
    normal_alignment = np.abs(np.einsum("ni,ni->n", anchor_normal, target_normal))
    plane_residual = np.abs(
        np.einsum("ni,ni->n", target - local_centroid, anchor_normal)
    )

    eps = np.finfo(np.float64).eps
    normalized_distance = nearest / np.maximum(local_anchor_scale, eps)
    normalized_residual = plane_residual / np.maximum(local_anchor_scale, eps)
    target_confidence = np.exp(
        -0.5 * normalized_distance**2 - 0.5 * normalized_residual**2
    ) * normal_alignment**2
    target_confidence = np.clip(target_confidence, 0.0, 1.0)
    point_confidence = np.concatenate(
        (np.ones(anchor.shape[0], dtype=np.float64), target_confidence)
    )
    return ObservedPointConfidence(
        point_confidence=point_confidence,
        target_confidence=target_confidence,
        nearest_anchor_distance=nearest,
        anchor_plane_residual=plane_residual,
        normal_alignment=normal_alignment,
        local_anchor_scale=local_anchor_scale,
    )


def confidence_weighted_filtration(
    filtration: AlphaFiltration,
    point_confidence: ArrayLike,
    *,
    k_neighbors: int = 12,
    penalty_strength: float,
) -> AdaptiveCellFiltration:
    """Delay low-confidence Delaunay cells without deleting their vertices."""

    confidence = np.asarray(point_confidence, dtype=np.float64)
    if confidence.shape != (filtration.points.shape[0],):
        raise ValueError("point_confidence must match filtration point count")
    if not np.all(np.isfinite(confidence)) or np.any(
        (confidence < 0.0) | (confidence > 1.0)
    ):
        raise ValueError("point_confidence must be finite and lie in [0, 1]")
    if not math.isfinite(penalty_strength) or penalty_strength < 0.0:
        raise ValueError("penalty_strength must be finite and non-negative")

    base = density_scaled_filtration(filtration, k_neighbors=k_neighbors)
    cells = filtration.top_simplices
    eps = np.finfo(np.float64).tiny
    cell_confidence = np.exp(
        np.mean(np.log(np.maximum(confidence[cells], eps)), axis=1)
    )
    scores = base.scores * (1.0 + penalty_strength * (1.0 - cell_confidence))
    diagnostics = dict(base.diagnostics)
    diagnostics.update(
        {
            "penalty_strength": float(penalty_strength),
            "point_confidence_min": float(np.min(confidence)),
            "point_confidence_median": float(np.median(confidence)),
            "point_confidence_max": float(np.max(confidence)),
            "cell_confidence_min": float(np.min(cell_confidence)),
            "cell_confidence_median": float(np.median(cell_confidence)),
            "cell_confidence_max": float(np.max(cell_confidence)),
            "score_increase_mean": float(np.mean(scores - base.scores)),
        }
    )
    return AdaptiveCellFiltration(
        points=filtration.points,
        top_simplices=cells,
        scores=scores,
        method="P43_continuous_confidence_weighted_B4",
        diagnostics=diagnostics,
        cell_confidence=cell_confidence,
        guard_scores=base.scores,
    )


def binary_confidence_subset(
    anchor_points: ArrayLike,
    target_points: ArrayLike,
    target_confidence: ArrayLike,
    *,
    threshold: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Binary-deletion comparator, kept separate from the continuous method."""

    anchor = np.asarray(anchor_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    confidence = np.asarray(target_confidence, dtype=np.float64)
    if confidence.shape != (target.shape[0],):
        raise ValueError("target_confidence must match target point count")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    retained = confidence >= threshold
    return np.vstack((anchor, target[retained])), retained
