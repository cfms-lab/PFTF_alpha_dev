"""SPD metric and soft-gate safety primitives for the future PFTF path."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import intrinsic_circumsphere

FloatArray = NDArray[np.float64]


def _validated_spd(matrix: ArrayLike, *, minimum_eigenvalue: float) -> FloatArray:
    result = np.asarray(matrix, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError("metric must be a square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError("metric must contain only finite values")
    if not np.allclose(result, result.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("metric must be symmetric")
    eigenvalues = np.linalg.eigvalsh(result)
    if float(eigenvalues[0]) < minimum_eigenvalue:
        raise ValueError(
            "metric is not safely positive definite: "
            f"minimum eigenvalue {eigenvalues[0]:.3e}"
        )
    return np.ascontiguousarray(0.5 * (result + result.T))


@dataclass(frozen=True)
class SimplexMetricDecision:
    """Metric selected for one simplex and whether it failed closed."""

    metric: FloatArray
    confidence: float
    used_fallback: bool
    reason: str


@dataclass(frozen=True)
class LocalMetricField:
    """Per-point SPD metrics with conservative confidence aggregation."""

    matrices: FloatArray
    confidence: FloatArray
    minimum_eigenvalue: float = 1.0e-8

    def __post_init__(self) -> None:
        matrices = np.asarray(self.matrices, dtype=np.float64)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("matrices must have shape (n, dimension, dimension)")
        if confidence.shape != (matrices.shape[0],):
            raise ValueError("confidence must have shape (n,)")
        if not math.isfinite(self.minimum_eigenvalue) or self.minimum_eigenvalue <= 0.0:
            raise ValueError("minimum_eigenvalue must be finite and positive")
        if not np.all(np.isfinite(confidence)) or np.any(
            (confidence < 0.0) | (confidence > 1.0)
        ):
            raise ValueError("confidence values must lie in [0, 1]")

        validated = np.stack(
            [
                _validated_spd(matrix, minimum_eigenvalue=self.minimum_eigenvalue)
                for matrix in matrices
            ]
        )
        object.__setattr__(self, "matrices", validated)
        object.__setattr__(self, "confidence", np.ascontiguousarray(confidence))

    @classmethod
    def from_factors(
        cls,
        factors: ArrayLike,
        confidence: ArrayLike,
        *,
        epsilon: float = 1.0e-6,
    ) -> LocalMetricField:
        """Construct ``M_i = L_i L_i^T + epsilon I`` safely."""

        factor_array = np.asarray(factors, dtype=np.float64)
        if factor_array.ndim != 3 or factor_array.shape[1] != factor_array.shape[2]:
            raise ValueError("factors must have shape (n, dimension, dimension)")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        identity = np.eye(factor_array.shape[1], dtype=np.float64)
        matrices = factor_array @ np.swapaxes(factor_array, -1, -2)
        matrices = matrices + epsilon * identity[None, :, :]
        return cls(
            matrices=matrices,
            confidence=np.asarray(confidence, dtype=np.float64),
            minimum_eigenvalue=epsilon * (1.0 - 1.0e-8),
        )

    def squared_distance(self, point_index: int, displacement: ArrayLike) -> float:
        """Evaluate one displacement in a point's local metric."""

        vector = np.asarray(displacement, dtype=np.float64)
        dimension = self.matrices.shape[1]
        if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"displacement must be a finite ({dimension},) vector")
        metric = self.matrices[int(point_index)]
        return float(vector @ metric @ vector)

    def metric_for_simplex(
        self,
        vertices: Iterable[int],
        *,
        confidence_threshold: float,
        fallback_metric: ArrayLike | None = None,
    ) -> SimplexMetricDecision:
        """Aggregate local metrics or fail closed to a trusted metric.

        The minimum vertex confidence is used as a conservative simplex
        confidence.  Arithmetic averaging preserves SPD and is intentionally
        only a local test; it does not assert global complex consistency.
        """

        indices = np.asarray(tuple(vertices), dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("vertices must contain at least one point index")
        if np.any(indices < 0) or np.any(indices >= self.matrices.shape[0]):
            raise IndexError("simplex vertex index is out of range")
        if (
            not math.isfinite(confidence_threshold)
            or not 0.0 <= confidence_threshold <= 1.0
        ):
            raise ValueError("confidence_threshold must lie in [0, 1]")

        simplex_confidence = float(np.min(self.confidence[indices]))
        if fallback_metric is None:
            fallback = np.eye(self.matrices.shape[1], dtype=np.float64)
        else:
            fallback = _validated_spd(
                fallback_metric, minimum_eigenvalue=self.minimum_eigenvalue
            )

        if simplex_confidence < confidence_threshold:
            return SimplexMetricDecision(
                metric=fallback,
                confidence=simplex_confidence,
                used_fallback=True,
                reason="confidence_below_threshold",
            )

        weights = self.confidence[indices]
        if float(np.sum(weights)) <= 0.0:
            return SimplexMetricDecision(
                metric=fallback,
                confidence=simplex_confidence,
                used_fallback=True,
                reason="zero_confidence_weight",
            )
        metric = np.average(self.matrices[indices], axis=0, weights=weights)
        metric = _validated_spd(metric, minimum_eigenvalue=self.minimum_eigenvalue)
        return SimplexMetricDecision(
            metric=metric,
            confidence=simplex_confidence,
            used_fallback=False,
            reason="pftf_metric",
        )


def metric_circumradius_squared(
    simplex_points: ArrayLike,
    metric: ArrayLike,
    *,
    minimum_eigenvalue: float = 1.0e-12,
) -> float:
    """Circumradius after applying a constant SPD metric to one simplex."""

    point_array = np.asarray(simplex_points, dtype=np.float64)
    if point_array.ndim != 2:
        raise ValueError("simplex_points must be a two-dimensional array")
    selected_metric = _validated_spd(metric, minimum_eigenvalue=minimum_eigenvalue)
    if point_array.shape[1] != selected_metric.shape[0]:
        raise ValueError("point and metric dimensions do not match")
    cholesky = np.linalg.cholesky(selected_metric)
    transformed = point_array @ cholesky
    return intrinsic_circumsphere(transformed).radius_squared


def hard_alpha_gate(alpha_squared: float, radius_squared: float) -> bool:
    """Exact comparison at the numerical values supplied by the caller."""

    if (
        not math.isfinite(alpha_squared)
        or not math.isfinite(radius_squared)
        or alpha_squared < 0.0
        or radius_squared < 0.0
    ):
        raise ValueError(
            "alpha_squared and radius_squared must be finite and non-negative"
        )
    return radius_squared <= alpha_squared


def soft_alpha_gate(
    alpha_squared: float,
    radius_squared: float,
    temperature: float,
) -> float:
    """Stable sigmoid relaxation of the hard alpha inclusion predicate."""

    if (
        not math.isfinite(alpha_squared)
        or not math.isfinite(radius_squared)
        or alpha_squared < 0.0
        or radius_squared < 0.0
    ):
        raise ValueError(
            "alpha_squared and radius_squared must be finite and non-negative"
        )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    argument = (alpha_squared - radius_squared) / temperature
    if argument >= 0.0:
        return 1.0 / (1.0 + math.exp(-argument))
    exponential = math.exp(argument)
    return exponential / (1.0 + exponential)
