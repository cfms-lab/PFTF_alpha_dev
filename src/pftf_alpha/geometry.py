"""Small, auditable geometry kernels used by the alpha filtration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Circumsphere:
    """Circumsphere constrained to a simplex's affine hull."""

    center: FloatArray
    radius_squared: float


def as_point_array(points: ArrayLike) -> FloatArray:
    """Validate a finite, unique 2D or 3D point cloud."""

    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n, 2) or (n, 3)")
    if result.shape[0] < result.shape[1] + 1:
        raise ValueError(
            f"at least {result.shape[1] + 1} points are required in {result.shape[1]}D"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("points must contain only finite coordinates")
    if np.unique(result, axis=0).shape[0] != result.shape[0]:
        raise ValueError("duplicate points are not supported")
    centered = result - result[0]
    if np.linalg.matrix_rank(centered) < result.shape[1]:
        raise ValueError("points do not span their ambient dimension")
    return np.ascontiguousarray(result)


def intrinsic_circumsphere(simplex_points: ArrayLike) -> Circumsphere:
    """Return the simplex circumsphere within its affine hull.

    ``simplex_points`` may describe a vertex, edge, triangle, or tetrahedron
    embedded in 2D or 3D.  Degenerate simplices are rejected instead of being
    assigned an unstable large radius.
    """

    points = np.asarray(simplex_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 1:
        raise ValueError("simplex_points must have shape (k + 1, dimension)")
    if not np.all(np.isfinite(points)):
        raise ValueError("simplex_points must contain only finite coordinates")

    simplex_dimension = points.shape[0] - 1
    ambient_dimension = points.shape[1]
    if simplex_dimension > ambient_dimension:
        raise ValueError("a simplex cannot exceed its ambient dimension")
    if simplex_dimension == 0:
        return Circumsphere(center=points[0].copy(), radius_squared=0.0)

    basis = points[1:] - points[0]
    gram = basis @ basis.T
    if np.linalg.matrix_rank(gram) < simplex_dimension:
        raise ValueError("degenerate simplex has no unique intrinsic circumsphere")

    rhs = 0.5 * np.diag(gram)
    coefficients = np.linalg.solve(gram, rhs)
    center = points[0] + coefficients @ basis
    delta = center - points[0]
    radius_squared = float(delta @ delta)
    if radius_squared < 0.0:
        raise ArithmeticError("computed a negative squared radius")
    return Circumsphere(center=center, radius_squared=radius_squared)
