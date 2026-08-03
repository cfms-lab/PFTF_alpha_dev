"""Alpha complexes induced by one globally declared coordinate map."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .filtration import AlphaFiltration
from .geometry import as_point_array

FloatArray = NDArray[np.float64]
JacobianFunction = Callable[[FloatArray], FloatArray]


class NonIntegrableJacobianError(ValueError):
    """A declared Jacobian field fails a necessary compatibility condition."""


class CoordinateMap3D(Protocol):
    """One explicit 3D coordinate map used by the spatial-alpha construction."""

    @property
    def name(self) -> str: ...

    def forward(self, points: ArrayLike) -> FloatArray: ...

    def inverse(self, points: ArrayLike) -> FloatArray: ...

    def jacobians(self, points: ArrayLike) -> FloatArray: ...


def _point_matrix(points: ArrayLike) -> FloatArray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError("points must contain only finite values")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class AffineCoordinateMap3D:
    """Invertible affine map with row-vector convention ``y=xL+t``."""

    factor: FloatArray
    offset: FloatArray

    def __post_init__(self) -> None:
        factor = np.asarray(self.factor, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64)
        if factor.shape != (3, 3) or not np.all(np.isfinite(factor)):
            raise ValueError("factor must be a finite (3, 3) matrix")
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("offset must be a finite (3,) vector")
        determinant = float(np.linalg.det(factor))
        if not math.isfinite(determinant) or determinant <= 0.0:
            raise ValueError("factor must have positive determinant")
        object.__setattr__(self, "factor", np.ascontiguousarray(factor))
        object.__setattr__(self, "offset", np.ascontiguousarray(offset))

    @property
    def name(self) -> str:
        return "affine_coordinate_map_3d"

    def forward(self, points: ArrayLike) -> FloatArray:
        point_array = _point_matrix(points)
        return np.ascontiguousarray(point_array @ self.factor + self.offset)

    def inverse(self, points: ArrayLike) -> FloatArray:
        point_array = _point_matrix(points)
        return np.ascontiguousarray(
            (point_array - self.offset) @ np.linalg.inv(self.factor)
        )

    def jacobians(self, points: ArrayLike) -> FloatArray:
        point_array = _point_matrix(points)
        return np.repeat(self.factor[None, :, :], len(point_array), axis=0)


@dataclass(frozen=True)
class QuadraticShearMap3D:
    """Global diffeomorphism ``(x,y,z) -> (x,y+s*x^2,z)``."""

    strength: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength):
            raise ValueError("strength must be finite")

    @property
    def name(self) -> str:
        return "quadratic_shear_map_3d"

    def forward(self, points: ArrayLike) -> FloatArray:
        point_array = _point_matrix(points)
        result = point_array.copy()
        result[:, 1] += float(self.strength) * point_array[:, 0] ** 2
        return np.ascontiguousarray(result)

    def inverse(self, points: ArrayLike) -> FloatArray:
        point_array = _point_matrix(points)
        result = point_array.copy()
        result[:, 1] -= float(self.strength) * point_array[:, 0] ** 2
        return np.ascontiguousarray(result)

    def jacobians(self, points: ArrayLike) -> FloatArray:
        point_array = _point_matrix(points)
        result = np.repeat(np.eye(3)[None, :, :], len(point_array), axis=0)
        result[:, 0, 1] = 2.0 * float(self.strength) * point_array[:, 0]
        return np.ascontiguousarray(result)


@dataclass(frozen=True)
class JacobianIntegrabilityAudit:
    """Necessary mixed-partial and orientation diagnostics for a field."""

    compatible: bool
    maximum_mixed_partial_residual: float
    minimum_jacobian_determinant: float
    finite_difference_step: float
    absolute_tolerance: float
    reason: str


@dataclass(frozen=True)
class IntegrableSpatialAlphaConstruction:
    """A spatial-SPD alpha filtration induced by one coordinate map."""

    coordinate_map_name: str
    transformed_points: FloatArray
    jacobians: FloatArray
    metrics: FloatArray
    filtration: AlphaFiltration
    maximum_inverse_roundtrip_error: float
    minimum_jacobian_determinant: float
    minimum_metric_eigenvalue: float
    maximum_relative_metric_variation: float


def numerical_jacobians(
    coordinate_map: CoordinateMap3D,
    points: ArrayLike,
    *,
    finite_difference_step: float,
) -> FloatArray:
    """Central-difference row Jacobians of one declared coordinate map."""

    point_array = _point_matrix(points)
    if not math.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive")
    result = np.empty((len(point_array), 3, 3), dtype=np.float64)
    for input_axis in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[input_axis] = finite_difference_step
        upper = coordinate_map.forward(point_array + delta)
        lower = coordinate_map.forward(point_array - delta)
        result[:, input_axis, :] = (
            upper - lower
        ) / (2.0 * finite_difference_step)
    return np.ascontiguousarray(result)


def audit_jacobian_integrability(
    jacobian_function: JacobianFunction,
    points: ArrayLike,
    *,
    finite_difference_step: float,
    absolute_tolerance: float,
    minimum_jacobian_determinant: float = 1.0e-12,
) -> JacobianIntegrabilityAudit:
    """Audit a necessary curl-free condition and positive orientation."""

    point_array = _point_matrix(points)
    if not math.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be finite and non-negative")
    if not math.isfinite(minimum_jacobian_determinant):
        raise ValueError("minimum_jacobian_determinant must be finite")

    base = np.asarray(jacobian_function(point_array), dtype=np.float64)
    if base.shape != (len(point_array), 3, 3) or not np.all(np.isfinite(base)):
        raise ValueError("jacobian_function must return finite shape (n, 3, 3)")
    derivatives = np.empty((3, len(point_array), 3, 3), dtype=np.float64)
    for coordinate in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[coordinate] = finite_difference_step
        upper = np.asarray(jacobian_function(point_array + delta), dtype=np.float64)
        lower = np.asarray(jacobian_function(point_array - delta), dtype=np.float64)
        if upper.shape != base.shape or lower.shape != base.shape:
            raise ValueError("jacobian_function changed output shape")
        derivatives[coordinate] = (
            upper - lower
        ) / (2.0 * finite_difference_step)

    maximum_residual = 0.0
    for first_axis in range(3):
        for second_axis in range(3):
            residual = np.abs(
                derivatives[second_axis, :, first_axis, :]
                - derivatives[first_axis, :, second_axis, :]
            )
            maximum_residual = max(maximum_residual, float(np.max(residual)))
    minimum_determinant = float(np.min(np.linalg.det(base)))
    compatible = bool(
        maximum_residual <= absolute_tolerance
        and minimum_determinant >= minimum_jacobian_determinant
    )
    if maximum_residual > absolute_tolerance:
        reason = "mixed_partial_residual_exceeds_tolerance"
    elif minimum_determinant < minimum_jacobian_determinant:
        reason = "jacobian_determinant_below_minimum"
    else:
        reason = "necessary_local_conditions_pass"
    return JacobianIntegrabilityAudit(
        compatible=compatible,
        maximum_mixed_partial_residual=maximum_residual,
        minimum_jacobian_determinant=minimum_determinant,
        finite_difference_step=float(finite_difference_step),
        absolute_tolerance=float(absolute_tolerance),
        reason=reason,
    )


def require_integrable_jacobian_field(
    jacobian_function: JacobianFunction,
    points: ArrayLike,
    *,
    finite_difference_step: float,
    absolute_tolerance: float,
    minimum_jacobian_determinant: float = 1.0e-12,
) -> JacobianIntegrabilityAudit:
    """Return the audit or fail closed when a necessary condition fails."""

    audit = audit_jacobian_integrability(
        jacobian_function,
        points,
        finite_difference_step=finite_difference_step,
        absolute_tolerance=absolute_tolerance,
        minimum_jacobian_determinant=minimum_jacobian_determinant,
    )
    if not audit.compatible:
        raise NonIntegrableJacobianError(
            "Jacobian field is not locally compatible: "
            f"reason={audit.reason}, residual="
            f"{audit.maximum_mixed_partial_residual:.6g}"
        )
    return audit


def coordinate_map_spatial_alpha(
    points: ArrayLike,
    coordinate_map: CoordinateMap3D,
    *,
    minimum_jacobian_determinant: float = 1.0e-12,
    inverse_roundtrip_tolerance: float = 1.0e-10,
    empty_ball_tolerance: float = 1.0e-12,
    qhull_options: str | None = None,
) -> IntegrableSpatialAlphaConstruction:
    """Build one alpha complex after a globally declared coordinate map."""

    point_array = as_point_array(points)
    if point_array.shape[1] != 3:
        raise ValueError("coordinate-map spatial alpha requires 3D points")
    if (
        not math.isfinite(minimum_jacobian_determinant)
        or minimum_jacobian_determinant <= 0.0
    ):
        raise ValueError("minimum_jacobian_determinant must be finite and positive")
    if (
        not math.isfinite(inverse_roundtrip_tolerance)
        or inverse_roundtrip_tolerance < 0.0
    ):
        raise ValueError("inverse_roundtrip_tolerance must be non-negative")

    transformed = as_point_array(coordinate_map.forward(point_array))
    recovered = _point_matrix(coordinate_map.inverse(transformed))
    maximum_roundtrip_error = float(np.max(np.abs(recovered - point_array)))
    if maximum_roundtrip_error > inverse_roundtrip_tolerance:
        raise ValueError(
            "coordinate map inverse roundtrip exceeds tolerance: "
            f"{maximum_roundtrip_error:.6g}"
        )

    jacobians = np.asarray(coordinate_map.jacobians(point_array), dtype=np.float64)
    if jacobians.shape != (len(point_array), 3, 3) or not np.all(
        np.isfinite(jacobians)
    ):
        raise ValueError("coordinate map must return finite Jacobians (n, 3, 3)")
    determinants = np.linalg.det(jacobians)
    minimum_determinant = float(np.min(determinants))
    if minimum_determinant < minimum_jacobian_determinant:
        raise ValueError(
            "coordinate map Jacobian determinant is below the declared minimum"
        )

    metrics = jacobians @ np.swapaxes(jacobians, -1, -2)
    metric_eigenvalues = np.linalg.eigvalsh(metrics)
    minimum_metric_eigenvalue = float(np.min(metric_eigenvalues))
    if minimum_metric_eigenvalue <= 0.0:
        raise ArithmeticError("Jacobian-induced metric is not positive definite")
    mean_metric = np.mean(metrics, axis=0)
    metric_deviations = np.linalg.norm(metrics - mean_metric[None, :, :], axis=(1, 2))
    metric_scale = max(float(np.linalg.norm(mean_metric)), np.finfo(float).tiny)
    maximum_metric_variation = float(np.max(metric_deviations)) / metric_scale

    transformed_filtration = AlphaFiltration.from_points(
        transformed,
        empty_ball_tolerance=empty_ball_tolerance,
        qhull_options=qhull_options,
    )
    original_coordinate_filtration = AlphaFiltration(
        point_array,
        transformed_filtration.top_simplices.copy(),
        transformed_filtration.records,
    )
    return IntegrableSpatialAlphaConstruction(
        coordinate_map_name=coordinate_map.name,
        transformed_points=transformed,
        jacobians=np.ascontiguousarray(jacobians),
        metrics=np.ascontiguousarray(metrics),
        filtration=original_coordinate_filtration,
        maximum_inverse_roundtrip_error=maximum_roundtrip_error,
        minimum_jacobian_determinant=minimum_determinant,
        minimum_metric_eigenvalue=minimum_metric_eigenvalue,
        maximum_relative_metric_variation=maximum_metric_variation,
    )
