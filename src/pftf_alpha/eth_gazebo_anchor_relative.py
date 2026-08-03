"""Anchor-relative local geometry evidence for ETH Gazebo reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .eth_gazebo_local_support import (
    LocalSupportEndpoint,
    SourceReconstructionInputs,
    _cell_summary,
    evaluate_local_support,
)
from .eth_gazebo_reconstruction_shadow import (
    ReconstructionEndpoint,
    _alpha_surface,
    _endpoint,
)

FloatArray = np.ndarray
PHASE41_MINIMUM_SUPPORT = 2
PHASE41_MAXIMUM_DISPERSION_METERS = 0.15
NORMAL_NEIGHBOR_COUNT = 12


@dataclass(frozen=True)
class AnchorRelativeRoute:
    phase41_candidate_cell_count: int
    anchor_relative_cell_count: int
    rejected_by_anchor_relative_count: int
    mean_nearest_anchor_distance_meters: float
    mean_anchor_plane_residual_meters: float
    mean_normal_alignment: float
    local_points: FloatArray


@dataclass(frozen=True)
class AnchorRelativeEndpoint:
    route: AnchorRelativeRoute
    endpoint: ReconstructionEndpoint


def _smallest_eigenvector(points: FloatArray) -> FloatArray:
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(points.shape[0], 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    return np.asarray(eigenvectors[:, 0], dtype=float)


def anchor_relative_route(
    inputs: SourceReconstructionInputs,
    *,
    maximum_nearest_anchor_distance_meters: float,
    maximum_anchor_plane_residual_meters: float,
    minimum_normal_alignment: float,
) -> AnchorRelativeRoute:
    """Filter Phase-41 cells by observed target-to-anchor local agreement."""

    if maximum_nearest_anchor_distance_meters <= 0.0:
        raise ValueError("maximum nearest-anchor distance must be positive")
    if maximum_anchor_plane_residual_meters <= 0.0:
        raise ValueError("maximum anchor-plane residual must be positive")
    if not 0.0 <= minimum_normal_alignment <= 1.0:
        raise ValueError("minimum normal alignment must be in [0, 1]")
    anchor_keys, _, anchor_centroids, _ = _cell_summary(
        inputs.anchor_points,
        inputs.lower,
    )
    target_keys, inverse, target_centroids, counts = _cell_summary(
        inputs.target_points,
        inputs.lower,
    )
    cell_provenance = np.unique(
        np.column_stack((inverse, inputs.target_provenance)),
        axis=0,
    )
    support = np.bincount(
        cell_provenance[:, 0],
        minlength=target_keys.shape[0],
    )
    squared_sum = np.bincount(
        inverse,
        weights=np.sum(inputs.target_points * inputs.target_points, axis=1),
        minlength=target_keys.shape[0],
    )
    dispersion = np.sqrt(
        np.maximum(
            squared_sum / counts
            - np.sum(target_centroids * target_centroids, axis=1),
            0.0,
        )
    )
    anchor_key_set = {tuple(key) for key in anchor_keys.tolist()}
    target_only = np.asarray(
        [tuple(key) not in anchor_key_set for key in target_keys.tolist()],
        dtype=bool,
    )
    phase41_mask = (
        target_only
        & (support >= PHASE41_MINIMUM_SUPPORT)
        & (dispersion <= PHASE41_MAXIMUM_DISPERSION_METERS)
    )
    phase41_centroids = target_centroids[phase41_mask]
    if phase41_centroids.shape[0] == 0:
        return AnchorRelativeRoute(
            phase41_candidate_cell_count=0,
            anchor_relative_cell_count=0,
            rejected_by_anchor_relative_count=0,
            mean_nearest_anchor_distance_meters=0.0,
            mean_anchor_plane_residual_meters=0.0,
            mean_normal_alignment=0.0,
            local_points=np.ascontiguousarray(anchor_centroids),
        )
    anchor_tree = cKDTree(anchor_centroids)
    target_tree = cKDTree(inputs.target_points)
    anchor_k = min(NORMAL_NEIGHBOR_COUNT, anchor_centroids.shape[0])
    target_k = min(NORMAL_NEIGHBOR_COUNT, inputs.target_points.shape[0])
    anchor_distances, anchor_indices = anchor_tree.query(
        phase41_centroids,
        k=anchor_k,
        workers=1,
    )
    _, target_indices = target_tree.query(
        phase41_centroids,
        k=target_k,
        workers=1,
    )
    anchor_distances = np.atleast_2d(anchor_distances)
    anchor_indices = np.atleast_2d(anchor_indices)
    target_indices = np.atleast_2d(target_indices)
    nearest = anchor_distances[:, 0]
    plane_residual = np.empty(phase41_centroids.shape[0], dtype=float)
    alignment = np.empty(phase41_centroids.shape[0], dtype=float)
    for index, (centroid, anchor_row, target_row) in enumerate(
        zip(
            phase41_centroids,
            anchor_indices,
            target_indices,
            strict=True,
        )
    ):
        anchor_neighbors = anchor_centroids[np.atleast_1d(anchor_row)]
        target_neighbors = inputs.target_points[np.atleast_1d(target_row)]
        anchor_normal = _smallest_eigenvector(anchor_neighbors)
        target_normal = _smallest_eigenvector(target_neighbors)
        anchor_center = np.mean(anchor_neighbors, axis=0)
        plane_residual[index] = abs(
            float((centroid - anchor_center) @ anchor_normal)
        )
        alignment[index] = abs(float(anchor_normal @ target_normal))
    accepted = (
        (nearest <= maximum_nearest_anchor_distance_meters)
        & (plane_residual <= maximum_anchor_plane_residual_meters)
        & (alignment >= minimum_normal_alignment)
    )
    local_points = np.ascontiguousarray(
        np.vstack((anchor_centroids, phase41_centroids[accepted]))
    )
    return AnchorRelativeRoute(
        phase41_candidate_cell_count=int(phase41_centroids.shape[0]),
        anchor_relative_cell_count=int(np.count_nonzero(accepted)),
        rejected_by_anchor_relative_count=int(np.count_nonzero(~accepted)),
        mean_nearest_anchor_distance_meters=float(np.mean(nearest)),
        mean_anchor_plane_residual_meters=float(np.mean(plane_residual)),
        mean_normal_alignment=float(np.mean(alignment)),
        local_points=local_points,
    )


def evaluate_anchor_relative(
    o3d: object,
    inputs: SourceReconstructionInputs,
    *,
    maximum_nearest_anchor_distance_meters: float,
    maximum_anchor_plane_residual_meters: float,
    minimum_normal_alignment: float,
    seed: int,
) -> AnchorRelativeEndpoint:
    route = anchor_relative_route(
        inputs,
        maximum_nearest_anchor_distance_meters=(
            maximum_nearest_anchor_distance_meters
        ),
        maximum_anchor_plane_residual_meters=(
            maximum_anchor_plane_residual_meters
        ),
        minimum_normal_alignment=minimum_normal_alignment,
    )
    endpoint = _endpoint(
        _alpha_surface(o3d, route.local_points),
        inputs.reference_points,
        characteristic_length=inputs.characteristic_length,
        seed=seed,
    )
    return AnchorRelativeEndpoint(route=route, endpoint=endpoint)


def evaluate_phase41_baseline(
    o3d: object,
    inputs: SourceReconstructionInputs,
    *,
    seed: int,
) -> LocalSupportEndpoint:
    return evaluate_local_support(
        o3d,
        inputs,
        minimum_support=PHASE41_MINIMUM_SUPPORT,
        maximum_dispersion_meters=PHASE41_MAXIMUM_DISPERSION_METERS,
        seed=seed,
    )
