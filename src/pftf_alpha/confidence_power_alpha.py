"""Confidence-aware regular triangulation with proper power-alpha scores."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .adaptive import AdaptiveCellFiltration, knn_scales
from .weighted_alpha import (
    PointSubmersionError,
    _weighted_orthoradius_squared,
    regular_triangulation,
)

FloatArray = NDArray[np.float64]


def confidence_power_weights(
    points: ArrayLike,
    point_confidence: ArrayLike,
    *,
    k_neighbors: int,
    density_weight_scale: float,
    confidence_penalty_scale: float,
) -> tuple[FloatArray, FloatArray]:
    """Return dimensionally consistent power weights and local spacing."""

    point_array = np.asarray(points, dtype=np.float64)
    confidence = np.asarray(point_confidence, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if confidence.shape != (point_array.shape[0],):
        raise ValueError("point_confidence must match the point count")
    if not np.all(np.isfinite(confidence)) or np.any(
        (confidence < 0.0) | (confidence > 1.0)
    ):
        raise ValueError("point_confidence must be finite and lie in [0, 1]")
    for name, value in (
        ("density_weight_scale", density_weight_scale),
        ("confidence_penalty_scale", confidence_penalty_scale),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    spacing = knn_scales(point_array, k_neighbors=k_neighbors)
    weights = spacing**2 * (
        density_weight_scale**2
        - confidence_penalty_scale**2 * (1.0 - confidence)
    )
    return np.ascontiguousarray(weights), np.ascontiguousarray(spacing)


def confidence_power_alpha_filtration(
    points: ArrayLike,
    point_confidence: ArrayLike,
    *,
    k_neighbors: int,
    density_weight_scale: float,
    confidence_penalty_scale: float,
) -> AdaptiveCellFiltration:
    """Build a confidence-aware regular complex and proper power filtration."""

    point_array = np.asarray(points, dtype=np.float64)
    confidence = np.asarray(point_confidence, dtype=np.float64)
    weights, spacing = confidence_power_weights(
        point_array,
        confidence,
        k_neighbors=k_neighbors,
        density_weight_scale=density_weight_scale,
        confidence_penalty_scale=confidence_penalty_scale,
    )
    cells = regular_triangulation(point_array, weights)
    if np.unique(cells).size != point_array.shape[0]:
        raise PointSubmersionError(
            "confidence power weights submerge points from the triangulation"
        )
    scores = np.empty(cells.shape[0], dtype=np.float64)
    for index, cell in enumerate(cells):
        radius_squared = _weighted_orthoradius_squared(
            point_array[cell], weights[cell]
        )
        simplex_scale = float(np.exp(np.mean(np.log(spacing[cell]))))
        scores[index] = math.sqrt(max(0.0, radius_squared)) / simplex_scale
    eps = np.finfo(np.float64).tiny
    cell_confidence = np.exp(
        np.mean(np.log(np.maximum(confidence[cells], eps)), axis=1)
    )
    return AdaptiveCellFiltration(
        points=point_array,
        top_simplices=cells,
        scores=scores,
        method="P45_confidence_regular_power_alpha",
        diagnostics={
            "k_neighbors": float(k_neighbors),
            "density_weight_scale": float(density_weight_scale),
            "confidence_penalty_scale": float(confidence_penalty_scale),
            "power_weight_min": float(np.min(weights)),
            "power_weight_median": float(np.median(weights)),
            "power_weight_max": float(np.max(weights)),
            "negative_power_weight_fraction": float(np.mean(weights < 0.0)),
            "point_confidence_min": float(np.min(confidence)),
            "point_confidence_median": float(np.median(confidence)),
            "point_confidence_max": float(np.max(confidence)),
        },
        cell_confidence=cell_confidence,
    )
