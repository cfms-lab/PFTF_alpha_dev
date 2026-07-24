"""Directed-relation field construction for the P1 PFTF prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from .metrics import LocalMetricField

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def directed_scale_contrast(
    source_scale: float,
    receiver_scales: ArrayLike,
    *,
    contrast_scale: float,
) -> FloatArray:
    """Bounded directed message based on receiver/source sampling scale.

    Swapping source and receiver reverses the sign before aggregation. This
    typed message is kept separate from the SPD projection because a signed
    relation is not itself a valid distance metric.
    """

    source = float(source_scale)
    receivers = np.asarray(receiver_scales, dtype=np.float64)
    if not math.isfinite(source) or source <= 0.0:
        raise ValueError("source_scale must be finite and positive")
    if not np.all(np.isfinite(receivers)) or np.any(receivers <= 0.0):
        raise ValueError("receiver_scales must be finite and positive")
    if not math.isfinite(contrast_scale) or contrast_scale <= 0.0:
        raise ValueError("contrast_scale must be finite and positive")
    return np.tanh(np.log(receivers / source) / contrast_scale)


@dataclass(frozen=True)
class PFTFRelationField:
    """Auditable relation tensors and their safe local SPD projection."""

    metric_field: LocalMetricField
    scales: FloatArray
    relation_tensors: FloatArray
    relation_strength: FloatArray
    reciprocity: FloatArray
    neighbor_indices: IntArray
    k_neighbors: int
    relation_gain: float
    max_condition_number: float
    density_contrast_scale: float
    receiver_imbalance_weight: float

    def __post_init__(self) -> None:
        point_count = self.metric_field.matrices.shape[0]
        scales = np.asarray(self.scales, dtype=np.float64)
        tensors = np.asarray(self.relation_tensors, dtype=np.float64)
        strength = np.asarray(self.relation_strength, dtype=np.float64)
        reciprocity = np.asarray(self.reciprocity, dtype=np.float64)
        neighbors = np.asarray(self.neighbor_indices, dtype=np.int64)
        if scales.shape != (point_count,) or np.any(scales <= 0.0):
            raise ValueError("scales must be positive with shape (n,)")
        if tensors.shape != (point_count, 3, 3):
            raise ValueError("relation_tensors must have shape (n, 3, 3)")
        if strength.shape != (point_count,) or np.any(strength < 0.0):
            raise ValueError("relation_strength must be non-negative with shape (n,)")
        if reciprocity.shape != (point_count,) or np.any(
            (reciprocity < 0.0) | (reciprocity > 1.0)
        ):
            raise ValueError("reciprocity must lie in [0, 1] with shape (n,)")
        if neighbors.shape != (point_count, self.k_neighbors):
            raise ValueError("neighbor_indices must have shape (n, k_neighbors)")
        object.__setattr__(self, "scales", np.ascontiguousarray(scales))
        object.__setattr__(self, "relation_tensors", np.ascontiguousarray(tensors))
        object.__setattr__(self, "relation_strength", np.ascontiguousarray(strength))
        object.__setattr__(self, "reciprocity", np.ascontiguousarray(reciprocity))
        object.__setattr__(self, "neighbor_indices", np.ascontiguousarray(neighbors))


def pftf_relation_field(
    points: ArrayLike,
    *,
    k_neighbors: int,
    relation_gain: float = 2.0,
    max_condition_number: float = 9.0,
    density_contrast_scale: float = 0.5,
    receiver_imbalance_weight: float = 0.5,
    signal_floor: float = 0.05,
) -> PFTFRelationField:
    """Construct a bounded local SPD field from directed neighbor relations.

    The signed trace-free relation tensor is not used as a metric directly.
    Its eigensystem is mapped through bounded log-eigenvalues, then blended
    toward the density-scaled identity according to an unlabeled confidence.
    """

    point_array = np.asarray(points, dtype=np.float64)
    if (
        point_array.ndim != 2
        or point_array.shape[1] != 3
        or not np.all(np.isfinite(point_array))
    ):
        raise ValueError("points must be a finite array with shape (n, 3)")
    if not 3 <= k_neighbors < point_array.shape[0]:
        raise ValueError("k_neighbors must satisfy 3 <= k_neighbors < point count")
    for name, value in {
        "relation_gain": relation_gain,
        "density_contrast_scale": density_contrast_scale,
        "signal_floor": signal_floor,
    }.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(max_condition_number) or max_condition_number < 1.0:
        raise ValueError("max_condition_number must be finite and at least one")
    if not math.isfinite(receiver_imbalance_weight) or receiver_imbalance_weight < 0.0:
        raise ValueError("receiver_imbalance_weight must be finite and non-negative")

    distances, all_indices = cKDTree(point_array).query(
        point_array,
        k=k_neighbors + 1,
        workers=1,
    )
    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float64)
    neighbor_indices = np.asarray(all_indices[:, 1:], dtype=np.int64)
    scales = np.median(neighbor_distances, axis=1)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("kNN scales must be finite and positive")

    identity = np.eye(3, dtype=np.float64)
    neighbor_sets = [set(int(index) for index in row) for row in neighbor_indices]
    relation_tensors = np.empty((point_array.shape[0], 3, 3), dtype=np.float64)
    relation_strength = np.empty(point_array.shape[0], dtype=np.float64)
    reciprocity = np.empty(point_array.shape[0], dtype=np.float64)
    confidence = np.empty(point_array.shape[0], dtype=np.float64)
    metrics = np.empty((point_array.shape[0], 3, 3), dtype=np.float64)
    half_log_condition = 0.5 * math.log(max_condition_number)

    for point_index, receivers in enumerate(neighbor_indices):
        displacements = point_array[receivers] - point_array[point_index]
        local_distances = neighbor_distances[point_index]
        directions = displacements / local_distances[:, None]
        weights = np.exp(-0.5 * (local_distances / scales[point_index]) ** 2)
        weights /= np.sum(weights)

        messages = directed_scale_contrast(
            scales[point_index],
            scales[receivers],
            contrast_scale=density_contrast_scale,
        )
        dyads = np.einsum("ki,kj->kij", directions, directions)
        trace_free_dyads = dyads - identity[None, :, :] / 3.0
        density_relation = np.einsum(
            "k,kij->ij",
            weights * messages,
            trace_free_dyads,
        )
        receiver_flow = np.sum(weights[:, None] * directions, axis=0)
        flow_squared = float(receiver_flow @ receiver_flow)
        imbalance_relation = (
            np.outer(receiver_flow, receiver_flow) - flow_squared * identity / 3.0
        )
        relation = density_relation + receiver_imbalance_weight * imbalance_relation
        relation = 0.5 * (relation + relation.T)
        relation -= np.trace(relation) * identity / 3.0
        relation_tensors[point_index] = relation

        strength = float(np.linalg.norm(relation, ord="fro"))
        relation_strength[point_index] = strength
        reciprocal_fraction = float(
            np.mean(
                [point_index in neighbor_sets[int(receiver)] for receiver in receivers]
            )
        )
        reciprocity[point_index] = reciprocal_fraction
        distance_cv = float(np.std(local_distances) / np.mean(local_distances))
        signal_confidence = strength / (strength + signal_floor)
        point_confidence = float(
            np.clip(
                reciprocal_fraction * math.exp(-distance_cv) * signal_confidence,
                0.0,
                1.0,
            )
        )
        confidence[point_index] = point_confidence

        eigenvalues, eigenvectors = np.linalg.eigh(relation)
        centered = eigenvalues - float(np.mean(eigenvalues))
        log_metric_eigenvalues = np.clip(
            relation_gain * centered,
            -half_log_condition,
            half_log_condition,
        )
        anisotropic = (
            eigenvectors @ np.diag(np.exp(log_metric_eigenvalues)) @ eigenvectors.T
        )
        blended = (1.0 - point_confidence) * identity + point_confidence * anisotropic
        metrics[point_index] = blended / scales[point_index] ** 2

    minimum_metric_eigenvalue = float(np.min(np.linalg.eigvalsh(metrics)))
    metric_field = LocalMetricField(
        matrices=metrics,
        confidence=confidence,
        minimum_eigenvalue=max(
            np.finfo(np.float64).tiny,
            1.0e-8 * minimum_metric_eigenvalue,
        ),
    )
    return PFTFRelationField(
        metric_field=metric_field,
        scales=scales,
        relation_tensors=relation_tensors,
        relation_strength=relation_strength,
        reciprocity=reciprocity,
        neighbor_indices=neighbor_indices,
        k_neighbors=int(k_neighbors),
        relation_gain=float(relation_gain),
        max_condition_number=float(max_condition_number),
        density_contrast_scale=float(density_contrast_scale),
        receiver_imbalance_weight=float(receiver_imbalance_weight),
    )
