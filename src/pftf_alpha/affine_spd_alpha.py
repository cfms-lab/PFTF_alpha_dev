"""Globally compatible affine-SPD alpha construction.

One constant SPD matrix defines a coherent alpha complex by a single global
linear coordinate transform. A varying point metric field is accepted only
when every matrix agrees with the same constant matrix within declared
tolerances. This module deliberately does not invent a spatially varying
anisotropic Delaunay construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .filtration import AlphaFiltration
from .geometry import as_point_array
from .metrics import LocalMetricField, _validated_spd

FloatArray = NDArray[np.float64]


class IncompatibleLocalMetricError(ValueError):
    """The point metrics cannot be represented by one global affine map."""


@dataclass(frozen=True)
class GlobalMetricCompatibility:
    """Diagnostics for the constant-global-metric compatibility test."""

    compatible: bool
    maximum_absolute_deviation: float
    maximum_relative_deviation: float
    relative_tolerance: float
    absolute_tolerance: float
    reason: str


@dataclass(frozen=True)
class AffineSPDAlphaConstruction:
    """One global-SPD alpha filtration and its coordinate transform."""

    metric: FloatArray
    transform: FloatArray
    transformed_points: FloatArray
    filtration: AlphaFiltration


def audit_global_metric_compatibility(
    field: LocalMetricField,
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
) -> GlobalMetricCompatibility:
    """Test whether every point metric is one shared constant SPD matrix."""

    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be finite and non-negative")

    reference = field.matrices[0]
    deviations = np.linalg.norm(
        field.matrices - reference[None, :, :], axis=(1, 2)
    )
    reference_norm = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    maximum_absolute = float(np.max(deviations))
    maximum_relative = maximum_absolute / reference_norm
    compatible = bool(
        np.allclose(
            field.matrices,
            reference[None, :, :],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    )
    return GlobalMetricCompatibility(
        compatible=compatible,
        maximum_absolute_deviation=maximum_absolute,
        maximum_relative_deviation=maximum_relative,
        relative_tolerance=float(relative_tolerance),
        absolute_tolerance=float(absolute_tolerance),
        reason=(
            "shared_constant_spd_metric"
            if compatible
            else "point_metrics_require_more_than_one_global_affine_transform"
        ),
    )


def compatible_global_metric(
    field: LocalMetricField,
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
) -> FloatArray:
    """Return the shared metric or fail closed for a varying local field."""

    audit = audit_global_metric_compatibility(
        field,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if not audit.compatible:
        raise IncompatibleLocalMetricError(
            "local SPD field is not globally affine-compatible: "
            f"maximum relative deviation={audit.maximum_relative_deviation:.6g}"
        )
    return field.matrices[0].copy()


def global_affine_spd_alpha(
    points: ArrayLike,
    metric: ArrayLike,
    *,
    empty_ball_tolerance: float = 1.0e-12,
    qhull_options: str | None = None,
) -> AffineSPDAlphaConstruction:
    """Build the alpha filtration induced by one constant SPD metric.

    With row-vector coordinates and ``M = L L^T``, the map ``y = x L`` makes
    Euclidean distances in ``y`` equal metric distances in ``x``. Delaunay
    connectivity and every filtration value are therefore computed together
    in that one transformed coordinate system, then indexed on the original
    points for downstream surface reconstruction.
    """

    point_array = as_point_array(points)
    if not math.isfinite(empty_ball_tolerance) or empty_ball_tolerance < 0.0:
        raise ValueError("empty_ball_tolerance must be finite and non-negative")
    selected_metric = _validated_spd(metric, minimum_eigenvalue=1.0e-12)
    if selected_metric.shape != (point_array.shape[1], point_array.shape[1]):
        raise ValueError("point and metric dimensions do not match")

    transform = np.linalg.cholesky(selected_metric)
    transformed_points = np.ascontiguousarray(point_array @ transform)
    transformed_filtration = AlphaFiltration.from_points(
        transformed_points,
        empty_ball_tolerance=empty_ball_tolerance,
        qhull_options=qhull_options,
    )
    original_coordinate_filtration = AlphaFiltration(
        point_array,
        transformed_filtration.top_simplices.copy(),
        transformed_filtration.records,
    )
    return AffineSPDAlphaConstruction(
        metric=selected_metric,
        transform=transform,
        transformed_points=transformed_points,
        filtration=original_coordinate_filtration,
    )


def global_affine_spd_alpha_from_field(
    points: ArrayLike,
    field: LocalMetricField,
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
    empty_ball_tolerance: float = 1.0e-12,
    qhull_options: str | None = None,
) -> AffineSPDAlphaConstruction:
    """Build only when a point field reduces to one global affine metric."""

    point_array = as_point_array(points)
    if field.matrices.shape != (
        point_array.shape[0],
        point_array.shape[1],
        point_array.shape[1],
    ):
        raise ValueError("field shape must match the point cloud")
    metric = compatible_global_metric(
        field,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return global_affine_spd_alpha(
        point_array,
        metric,
        empty_ball_tolerance=empty_ball_tolerance,
        qhull_options=qhull_options,
    )
