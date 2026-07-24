"""Euclidean Delaunay alpha filtration for the G0-G1 baseline.

This module is deliberately a floating-point research baseline.  It does not
claim the exact-predicate guarantees required for the later CGAL evaluation
path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay, QhullError

from .conventions import AlphaConvention, alpha_to_squared_radius
from .geometry import as_point_array, intrinsic_circumsphere

IntArray = NDArray[np.int64]
Simplex = tuple[int, ...]


class BoundaryMode(StrEnum):
    """Treatment of lower-dimensional singular facets."""

    GENERAL = "general"
    REGULARIZED = "regularized"


@dataclass(frozen=True)
class SimplexRecord:
    """Filtration metadata for one Delaunay simplex."""

    vertices: Simplex
    alpha_squared: float
    is_gabriel: bool

    @property
    def dimension(self) -> int:
        return len(self.vertices) - 1


@dataclass(frozen=True)
class ComplexStatistics:
    """Cheap diagnostics available without a homology dependency."""

    alpha_squared: float
    simplex_counts: tuple[int, ...]
    connected_components: int
    euler_characteristic: int
    boundary_facets: int
    top_simplices: int


def _all_simplices(
    top_simplices: IntArray, ambient_dimension: int
) -> dict[int, set[Simplex]]:
    result: dict[int, set[Simplex]] = {
        dimension: set() for dimension in range(ambient_dimension + 1)
    }
    for cell in top_simplices:
        ordered = tuple(sorted(int(vertex) for vertex in cell))
        for size in range(1, ambient_dimension + 2):
            result[size - 1].update(combinations(ordered, size))
    return result


def _immediate_cofaces(
    simplices: dict[int, set[Simplex]], ambient_dimension: int
) -> dict[Simplex, tuple[Simplex, ...]]:
    result: dict[Simplex, tuple[Simplex, ...]] = {}
    for dimension in range(ambient_dimension):
        candidates = simplices[dimension + 1]
        for simplex in simplices[dimension]:
            simplex_set = set(simplex)
            result[simplex] = tuple(
                coface for coface in candidates if simplex_set.issubset(coface)
            )
    return result


class AlphaFiltration:
    """Finite Euclidean alpha filtration of a 2D or 3D point cloud."""

    def __init__(
        self,
        points: NDArray[np.float64],
        top_simplices: IntArray,
        records: Iterable[SimplexRecord],
    ) -> None:
        self.points = points
        self.top_simplices = top_simplices
        self.ambient_dimension = points.shape[1]
        self._records = {record.vertices: record for record in records}
        self._simplices_by_dimension: dict[int, tuple[Simplex, ...]] = {
            dimension: tuple(
                sorted(
                    simplex
                    for simplex, record in self._records.items()
                    if record.dimension == dimension
                )
            )
            for dimension in range(self.ambient_dimension + 1)
        }
        self._top_cofaces: dict[Simplex, tuple[Simplex, ...]] = {}
        top = self._simplices_by_dimension[self.ambient_dimension]
        for facet in self._simplices_by_dimension[self.ambient_dimension - 1]:
            facet_set = set(facet)
            self._top_cofaces[facet] = tuple(
                cell for cell in top if facet_set.issubset(cell)
            )

    @classmethod
    def from_points(
        cls,
        points: ArrayLike,
        *,
        empty_ball_tolerance: float = 1.0e-12,
        qhull_options: str | None = None,
    ) -> AlphaFiltration:
        """Construct the Delaunay filtration and its finite critical values."""

        point_array = as_point_array(points)
        if not np.isfinite(empty_ball_tolerance) or empty_ball_tolerance < 0.0:
            raise ValueError("empty_ball_tolerance must be finite and non-negative")

        ambient_dimension = point_array.shape[1]
        if point_array.shape[0] == ambient_dimension + 1:
            top_simplices = np.arange(ambient_dimension + 1, dtype=np.int64).reshape(
                1, -1
            )
        else:
            try:
                triangulation = Delaunay(point_array, qhull_options=qhull_options)
            except QhullError as error:
                raise ValueError(f"Delaunay triangulation failed: {error}") from error
            top_simplices = np.asarray(triangulation.simplices, dtype=np.int64)

        return cls.from_top_simplices(
            point_array,
            top_simplices,
            empty_ball_tolerance=empty_ball_tolerance,
        )

    @classmethod
    def from_top_simplices(
        cls,
        points: ArrayLike,
        top_simplices: ArrayLike,
        *,
        empty_ball_tolerance: float = 1.0e-12,
    ) -> AlphaFiltration:
        """Construct a filtration from caller-validated top connectivity.

        Filtration values remain floating-point circumsphere computations.  The
        caller is responsible for establishing that the supplied connectivity
        is a Delaunay triangulation; this method validates only the structural
        invariants needed by the filtration.
        """

        point_array = as_point_array(points)
        if not np.isfinite(empty_ball_tolerance) or empty_ball_tolerance < 0.0:
            raise ValueError("empty_ball_tolerance must be finite and non-negative")

        ambient_dimension = point_array.shape[1]
        raw_top_simplices = np.asarray(top_simplices)
        if (
            raw_top_simplices.ndim != 2
            or raw_top_simplices.shape[0] == 0
            or raw_top_simplices.shape[1] != ambient_dimension + 1
        ):
            raise ValueError(
                "top_simplices must have shape "
                f"(m, {ambient_dimension + 1}) with m >= 1"
            )
        if raw_top_simplices.dtype.kind not in "iu":
            raise ValueError("top_simplices must contain integer vertex indices")
        top_simplex_array = np.asarray(raw_top_simplices, dtype=np.int64)
        if np.any(top_simplex_array < 0) or np.any(
            top_simplex_array >= point_array.shape[0]
        ):
            raise ValueError("top_simplices contains an out-of-range vertex index")

        canonical_top_simplices = np.sort(top_simplex_array, axis=1)
        if np.any(np.diff(canonical_top_simplices, axis=1) == 0):
            raise ValueError("top_simplices contains a repeated vertex")
        if np.unique(canonical_top_simplices, axis=0).shape[0] != len(
            canonical_top_simplices
        ):
            raise ValueError("top_simplices contains duplicate cells")
        used_vertices = np.unique(canonical_top_simplices)
        if used_vertices.size != point_array.shape[0]:
            raise ValueError("top_simplices must use every input point")

        simplices = _all_simplices(top_simplex_array, ambient_dimension)
        cofaces = _immediate_cofaces(simplices, ambient_dimension)
        alpha_squared: dict[Simplex, float] = {}
        gabriel: dict[Simplex, bool] = {}

        for dimension in range(ambient_dimension, -1, -1):
            for simplex in sorted(simplices[dimension]):
                sphere = intrinsic_circumsphere(point_array[list(simplex)])
                other_indices = np.setdiff1d(
                    np.arange(point_array.shape[0], dtype=np.int64),
                    np.asarray(simplex, dtype=np.int64),
                    assume_unique=True,
                )
                if other_indices.size:
                    deltas = point_array[other_indices] - sphere.center
                    other_distances_squared = np.einsum("ij,ij->i", deltas, deltas)
                    comparison_scale = np.maximum(
                        1.0,
                        np.maximum(sphere.radius_squared, other_distances_squared),
                    )
                    is_empty = bool(
                        np.all(
                            other_distances_squared
                            >= sphere.radius_squared
                            - empty_ball_tolerance * comparison_scale
                        )
                    )
                else:
                    is_empty = True

                candidates: list[float] = []
                if is_empty or dimension == ambient_dimension:
                    candidates.append(sphere.radius_squared)
                candidates.extend(
                    alpha_squared[coface] for coface in cofaces.get(simplex, ())
                )
                if not candidates:
                    raise ArithmeticError(
                        f"could not assign a filtration value to simplex {simplex}"
                    )
                alpha_squared[simplex] = max(0.0, min(candidates))
                gabriel[simplex] = is_empty

        records = (
            SimplexRecord(
                vertices=simplex,
                alpha_squared=alpha_squared[simplex],
                is_gabriel=gabriel[simplex],
            )
            for dimension in range(ambient_dimension + 1)
            for simplex in sorted(simplices[dimension])
        )
        return cls(point_array, np.ascontiguousarray(top_simplex_array), records)

    @property
    def records(self) -> tuple[SimplexRecord, ...]:
        """All simplex records, ordered by dimension and vertex index."""

        return tuple(
            self._records[simplex]
            for dimension in range(self.ambient_dimension + 1)
            for simplex in self._simplices_by_dimension[dimension]
        )

    def critical_values(
        self,
        *,
        include_zero: bool = False,
        dimensions: Iterable[int] | None = None,
    ) -> NDArray[np.float64]:
        """Return sorted unique squared-radius values where the complex changes."""

        selected_dimensions = (
            set(range(self.ambient_dimension + 1))
            if dimensions is None
            else {int(dimension) for dimension in dimensions}
        )
        if not selected_dimensions.issubset(set(range(self.ambient_dimension + 1))):
            raise ValueError("dimensions contains an invalid simplex dimension")
        values = {
            record.alpha_squared
            for record in self._records.values()
            if record.dimension in selected_dimensions
            and (include_zero or record.alpha_squared > 0.0)
        }
        return np.asarray(sorted(values), dtype=np.float64)

    def simplices_at(
        self,
        alpha: float,
        *,
        convention: AlphaConvention | str = AlphaConvention.SQUARED_RADIUS,
    ) -> dict[int, IntArray]:
        """Return the alpha complex grouped by simplex dimension."""

        threshold = alpha_to_squared_radius(alpha, convention)
        result: dict[int, IntArray] = {}
        for dimension in range(self.ambient_dimension + 1):
            included = [
                simplex
                for simplex in self._simplices_by_dimension[dimension]
                if self._records[simplex].alpha_squared <= threshold
            ]
            if included:
                result[dimension] = np.asarray(included, dtype=np.int64)
            else:
                result[dimension] = np.empty((0, dimension + 1), dtype=np.int64)
        return result

    def boundary_facets_at(
        self,
        alpha: float,
        *,
        mode: BoundaryMode | str = BoundaryMode.REGULARIZED,
        convention: AlphaConvention | str = AlphaConvention.SQUARED_RADIUS,
    ) -> IntArray:
        """Return codimension-one facets on the general or regularized boundary."""

        threshold = alpha_to_squared_radius(alpha, convention)
        selected_mode = BoundaryMode(mode)
        facets: list[Simplex] = []
        for facet in self._simplices_by_dimension[self.ambient_dimension - 1]:
            if self._records[facet].alpha_squared > threshold:
                continue
            included_top_cofaces = sum(
                self._records[cell].alpha_squared <= threshold
                for cell in self._top_cofaces[facet]
            )
            if selected_mode is BoundaryMode.REGULARIZED:
                on_boundary = included_top_cofaces == 1
            else:
                on_boundary = included_top_cofaces < 2
            if on_boundary:
                facets.append(facet)
        if not facets:
            return np.empty((0, self.ambient_dimension), dtype=np.int64)
        return np.asarray(facets, dtype=np.int64)

    def statistics(
        self,
        alpha: float,
        *,
        boundary_mode: BoundaryMode | str = BoundaryMode.REGULARIZED,
        convention: AlphaConvention | str = AlphaConvention.SQUARED_RADIUS,
    ) -> ComplexStatistics:
        """Compute inexpensive topology and complexity diagnostics."""

        threshold = alpha_to_squared_radius(alpha, convention)
        complex_by_dimension = self.simplices_at(threshold)
        counts = tuple(
            int(complex_by_dimension[dimension].shape[0])
            for dimension in range(self.ambient_dimension + 1)
        )

        vertices = [int(vertex[0]) for vertex in complex_by_dimension[0]]
        parent = {vertex: vertex for vertex in vertices}

        def find(vertex: int) -> int:
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for edge in complex_by_dimension.get(1, np.empty((0, 2), dtype=np.int64)):
            union(int(edge[0]), int(edge[1]))

        components = len({find(vertex) for vertex in vertices})
        euler_characteristic = sum(
            count if dimension % 2 == 0 else -count
            for dimension, count in enumerate(counts)
        )
        boundary_facets = self.boundary_facets_at(threshold, mode=boundary_mode)
        return ComplexStatistics(
            alpha_squared=threshold,
            simplex_counts=counts,
            connected_components=components,
            euler_characteristic=euler_characteristic,
            boundary_facets=int(boundary_facets.shape[0]),
            top_simplices=counts[-1],
        )
