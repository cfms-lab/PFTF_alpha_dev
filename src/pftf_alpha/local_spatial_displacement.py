"""Observed local/spatial coherence features for matched displacements."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .matched_pair_consistency import (
    MatchedPairConfig,
    matched_displacement_scores,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class LocalSpatialDisplacementConfig:
    """Frozen neighborhood and support controls for observed-only evidence."""

    neighbor_count: int = 8
    peak_support_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.neighbor_count < 2:
            raise ValueError("neighbor_count must be at least two")
        if (
            not math.isfinite(self.peak_support_fraction)
            or not 0.0 < self.peak_support_fraction <= 1.0
        ):
            raise ValueError("peak_support_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class LocalSpatialDisplacementEvidence:
    """Case-level summaries derived from spatially adjacent matched pairs."""

    information_boundary: str
    point_count: int
    neighbor_count: int
    maximum_local_residual: float
    support_local_residual: float
    percentile95_local_residual: float
    maximum_local_score_excess: float
    peak_neighbor_score_support_fraction: float
    median_neighbor_radius_fraction: float
    peak_neighbor_radius_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _neighbor_indices(points: FloatArray, neighbor_count: int) -> IntArray:
    count = points.shape[0]
    selected_count = min(neighbor_count, count - 1)
    tree = cKDTree(points)
    _, queried = tree.query(points, k=selected_count + 1)
    queried = np.asarray(queried, dtype=np.int64)
    if queried.ndim == 1:
        queried = queried[:, None]
    result = np.empty((count, selected_count), dtype=np.int64)
    for index, row in enumerate(queried):
        selected = [int(value) for value in row if int(value) != index]
        if len(selected) < selected_count:
            distances = np.linalg.norm(points - points[index], axis=1)
            order = np.argsort(distances, kind="stable")
            selected = [int(value) for value in order if int(value) != index]
        result[index] = selected[:selected_count]
    return result


def estimate_local_spatial_displacement_evidence(
    primary_points: FloatArray,
    repeat_points: FloatArray,
    matched_pair_config: MatchedPairConfig | None = None,
    local_config: LocalSpatialDisplacementConfig | None = None,
) -> LocalSpatialDisplacementEvidence:
    """Summarize local displacement discontinuity without source labels."""

    primary = np.asarray(primary_points, dtype=np.float64)
    repeat = np.asarray(repeat_points, dtype=np.float64)
    if primary.ndim != 2 or primary.shape[1] != 3:
        raise ValueError("primary_points must have shape (n, 3)")
    if repeat.shape != primary.shape:
        raise ValueError("repeat_points must align with primary_points")
    if primary.shape[0] < 3:
        raise ValueError("at least three matched pairs are required")
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(repeat)):
        raise ValueError("matched points must be finite")

    selected = (
        LocalSpatialDisplacementConfig() if local_config is None else local_config
    )
    scores = matched_displacement_scores(
        primary,
        repeat,
        matched_pair_config,
    )
    neighbors = _neighbor_indices(primary, selected.neighbor_count)
    standardized_vectors = (
        scores.displacements - scores.displacement_location
    ) / scores.axis_scales
    neighbor_vectors = standardized_vectors[neighbors]
    neighbor_center = np.median(neighbor_vectors, axis=1)
    local_residuals = np.linalg.norm(
        standardized_vectors - neighbor_center,
        axis=1,
    )
    neighbor_scores = scores.point_scores[neighbors]
    neighbor_score_median = np.median(neighbor_scores, axis=1)
    score_excess = np.maximum(scores.point_scores - neighbor_score_median, 0.0)
    neighbor_radii = np.max(
        np.linalg.norm(primary[neighbors] - primary[:, None, :], axis=2),
        axis=1,
    )
    radius_scale = max(
        scores.observed_characteristic_length,
        np.finfo(float).eps,
    )
    descending = np.sort(local_residuals)[::-1]
    peak_index = int(np.argmax(local_residuals))
    peak_score = max(float(scores.point_scores[peak_index]), np.finfo(float).eps)
    support_threshold = selected.peak_support_fraction * peak_score
    values = np.asarray(
        (
            descending[0],
            descending[1],
            np.percentile(local_residuals, 95.0),
            np.max(score_excess),
            np.mean(neighbor_scores[peak_index] >= support_threshold),
            np.median(neighbor_radii) / radius_scale,
            neighbor_radii[peak_index] / radius_scale,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("local/spatial displacement evidence must be finite")
    return LocalSpatialDisplacementEvidence(
        information_boundary=(
            "ordered_primary_and_repeat_coordinates_with_presented_pair_order_"
            "and_primary_coordinate_knn_only; source_labels_and_endpoint_hidden"
        ),
        point_count=primary.shape[0],
        neighbor_count=neighbors.shape[1],
        maximum_local_residual=float(values[0]),
        support_local_residual=float(values[1]),
        percentile95_local_residual=float(values[2]),
        maximum_local_score_excess=float(values[3]),
        peak_neighbor_score_support_fraction=float(values[4]),
        median_neighbor_radius_fraction=float(values[5]),
        peak_neighbor_radius_fraction=float(values[6]),
    )
