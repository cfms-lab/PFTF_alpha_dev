"""Exact-predicate preflight for the pending G4 construction backend.

The predicates in this module interpret every finite binary64 coordinate as
its exact rational value.  They audit supplied tetrahedral connectivity; they
do not construct a Delaunay triangulation and therefore do not provide an
exact-construction or CGAL certificate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .filtration import AlphaFiltration
from .geometry import as_point_array

IntArray = NDArray[np.int64]
IntegerPoint = tuple[int, int, int]


@dataclass(frozen=True)
class ExactPredicateCaseAudit:
    """Exact sign audit for one supplied tetrahedralization."""

    case_id: str
    point_count: int
    top_simplex_count: int
    unique_top_simplex_count: int
    coordinate_common_denominator_power: int
    duplicate_top_simplex_count: int
    exact_orientation_zero_count: int
    float_orientation_sign_disagreement_count: int
    interior_facet_count: int
    audited_interior_facet_count: int
    unresolved_interior_facet_count: int
    nonmanifold_facet_count: int
    interior_facet_side_violation_count: int
    exact_cospherical_interior_facet_count: int
    exact_local_delaunay_violation_count: int
    float_insphere_sign_disagreement_count: int
    predicate_consistent: bool
    unique_delaunay_combinatorics_supported: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExactPredicatePanelAudit:
    """Readiness-only exact-predicate audit across one requested split."""

    evaluation_split: str
    cases: tuple[ExactPredicateCaseAudit, ...]

    @property
    def all_predicates_consistent(self) -> bool:
        return all(case.predicate_consistent for case in self.cases)

    @property
    def all_unique_delaunay_combinatorics_supported(self) -> bool:
        return all(case.unique_delaunay_combinatorics_supported for case in self.cases)

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons = ["no_exact_construction_backend"]
        if not self.all_predicates_consistent:
            reasons.append("supplied_connectivity_failed_exact_predicate_audit")
        if not self.all_unique_delaunay_combinatorics_supported:
            reasons.append("unique_exact_delaunay_combinatorics_not_supported")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "readiness_audit_no_selection",
            "evaluation_split": self.evaluation_split,
            "coordinate_model": "binary64_values_as_exact_rationals",
            "triangulation_source": "SciPy_Qhull_floating_point",
            "predicates": ["orientation_3", "in_sphere_3"],
            "exact_construction_backend_integrated": False,
            "changes_benchmark_selection": False,
            "all_predicates_consistent": self.all_predicates_consistent,
            "all_unique_delaunay_combinatorics_supported": (
                self.all_unique_delaunay_combinatorics_supported
            ),
            "promotion_supported": False,
            "blocking_reasons": list(self.blocking_reasons),
            "totals": {
                "case_count": len(self.cases),
                "point_count": sum(case.point_count for case in self.cases),
                "top_simplex_count": sum(case.top_simplex_count for case in self.cases),
                "interior_facet_count": sum(
                    case.interior_facet_count for case in self.cases
                ),
                "exact_orientation_zero_count": sum(
                    case.exact_orientation_zero_count for case in self.cases
                ),
                "exact_cospherical_interior_facet_count": sum(
                    case.exact_cospherical_interior_facet_count for case in self.cases
                ),
                "exact_local_delaunay_violation_count": sum(
                    case.exact_local_delaunay_violation_count for case in self.cases
                ),
                "float_sign_disagreement_count": sum(
                    case.float_orientation_sign_disagreement_count
                    + case.float_insphere_sign_disagreement_count
                    for case in self.cases
                ),
            },
            "cases": [case.to_dict() for case in self.cases],
        }


def _sign(value: int | float) -> int:
    return int(value > 0) - int(value < 0)


def _determinant_sign(matrix: Sequence[Sequence[int]]) -> int:
    """Return an integer determinant sign using fraction-free elimination."""

    order = len(matrix)
    if order == 0 or any(len(row) != order for row in matrix):
        raise ValueError("matrix must be non-empty and square")
    work = [[int(value) for value in row] for row in matrix]
    row_sign = 1
    previous_pivot = 1

    for pivot_column in range(order - 1):
        pivot_row = next(
            (row for row in range(pivot_column, order) if work[row][pivot_column] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_column:
            work[pivot_column], work[pivot_row] = (
                work[pivot_row],
                work[pivot_column],
            )
            row_sign *= -1

        pivot = work[pivot_column][pivot_column]
        for row in range(pivot_column + 1, order):
            for column in range(pivot_column + 1, order):
                numerator = (
                    work[row][column] * pivot
                    - work[row][pivot_column] * work[pivot_column][column]
                )
                if pivot_column:
                    quotient, remainder = divmod(numerator, previous_pivot)
                    if remainder:
                        raise ArithmeticError("Bareiss division was not exact")
                    numerator = quotient
                work[row][column] = numerator
            work[row][pivot_column] = 0
        previous_pivot = pivot

    return _sign(row_sign * work[-1][-1])


def _integer_coordinates(
    points: NDArray[np.float64],
) -> tuple[tuple[IntegerPoint, ...], int]:
    ratios = [
        [float(coordinate).as_integer_ratio() for coordinate in point]
        for point in points
    ]
    common_denominator = max(
        denominator for point in ratios for _, denominator in point
    )
    if common_denominator & (common_denominator - 1):
        raise ArithmeticError("binary64 denominator was not a power of two")
    integer_points = tuple(
        tuple(
            numerator * (common_denominator // denominator)
            for numerator, denominator in point
        )
        for point in ratios
    )
    return integer_points, common_denominator.bit_length() - 1


def _orientation_sign(
    points: Sequence[IntegerPoint],
    cell: Sequence[int],
) -> int:
    base = points[cell[0]]
    matrix = [
        [points[vertex][axis] - base[axis] for axis in range(3)] for vertex in cell[1:]
    ]
    return _determinant_sign(matrix)


def _float_orientation_sign(
    points: NDArray[np.float64],
    cell: Sequence[int],
) -> int:
    simplex = points[np.asarray(cell, dtype=np.int64)]
    determinant = float(np.linalg.det(simplex[1:] - simplex[0]))
    return _sign(determinant)


def _insphere_sign(
    points: Sequence[IntegerPoint],
    cell: Sequence[int],
    query_vertex: int,
) -> int:
    orientation = _orientation_sign(points, cell)
    if orientation == 0:
        return 0
    rows = []
    for vertex in (*cell, query_vertex):
        point = points[vertex]
        rows.append([*point, sum(coordinate**2 for coordinate in point), 1])
    return -orientation * _determinant_sign(rows)


def _float_insphere_sign(
    points: NDArray[np.float64],
    cell: Sequence[int],
    query_vertex: int,
) -> int:
    orientation = _float_orientation_sign(points, cell)
    if orientation == 0:
        return 0
    vertices = np.asarray((*cell, query_vertex), dtype=np.int64)
    predicate_points = points[vertices]
    matrix = np.column_stack(
        (
            predicate_points,
            np.einsum("ij,ij->i", predicate_points, predicate_points),
            np.ones(5, dtype=np.float64),
        )
    )
    return -orientation * _sign(float(np.linalg.det(matrix)))


def _validated_top_simplices(
    top_simplices: ArrayLike,
    *,
    point_count: int,
) -> IntArray:
    cells = np.asarray(top_simplices, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 4 or cells.shape[0] == 0:
        raise ValueError("top_simplices must have non-empty shape (m, 4)")
    if np.any(cells < 0) or np.any(cells >= point_count):
        raise IndexError("top_simplices contains an out-of-range vertex")
    if any(len(set(int(vertex) for vertex in cell)) != 4 for cell in cells):
        raise ValueError("each top simplex must contain four distinct vertices")
    return np.ascontiguousarray(cells)


def audit_delaunay_predicates(
    case_id: str,
    points: ArrayLike,
    top_simplices: ArrayLike,
) -> ExactPredicateCaseAudit:
    """Audit exact signs for supplied 3D tetrahedral connectivity.

    Positive normalized in-sphere signs mean the neighboring opposite vertex
    lies strictly inside a cell's circumsphere and therefore violates the
    local Delaunay condition.  Zero identifies exact cospherical ambiguity.
    """

    if not case_id:
        raise ValueError("case_id must be non-empty")
    point_array = as_point_array(points)
    if point_array.shape[1] != 3:
        raise ValueError("exact Delaunay predicate audit requires 3D points")
    cells = _validated_top_simplices(
        top_simplices,
        point_count=point_array.shape[0],
    )
    integer_points, denominator_power = _integer_coordinates(point_array)

    canonical_cells = [tuple(sorted(int(vertex) for vertex in cell)) for cell in cells]
    unique_cells = tuple(dict.fromkeys(canonical_cells))
    duplicate_count = len(canonical_cells) - len(unique_cells)
    exact_orientation = {
        cell: _orientation_sign(integer_points, cell) for cell in unique_cells
    }
    float_orientation_disagreements = sum(
        exact_orientation[cell] != _float_orientation_sign(point_array, cell)
        for cell in unique_cells
    )

    facet_owners: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = (
        defaultdict(list)
    )
    for cell in unique_cells:
        for facet in combinations(cell, 3):
            facet_owners[facet].append(cell)

    interior_facet_count = 0
    audited_interior_facet_count = 0
    unresolved_interior_facet_count = 0
    nonmanifold_facet_count = 0
    side_violation_count = 0
    cospherical_count = 0
    local_violation_count = 0
    float_insphere_disagreement_count = 0

    for facet, owners in facet_owners.items():
        if len(owners) > 2:
            nonmanifold_facet_count += 1
            continue
        if len(owners) != 2:
            continue
        interior_facet_count += 1
        left, right = owners
        if exact_orientation[left] == 0 or exact_orientation[right] == 0:
            unresolved_interior_facet_count += 1
            continue
        left_opposite = next(vertex for vertex in left if vertex not in facet)
        right_opposite = next(vertex for vertex in right if vertex not in facet)
        left_side = _orientation_sign(
            integer_points,
            (*facet, left_opposite),
        )
        right_side = _orientation_sign(
            integer_points,
            (*facet, right_opposite),
        )
        if left_side == 0 or right_side == 0:
            unresolved_interior_facet_count += 1
            continue
        if left_side == right_side:
            side_violation_count += 1
            continue

        audited_interior_facet_count += 1
        exact_sign = _insphere_sign(integer_points, left, right_opposite)
        float_sign = _float_insphere_sign(point_array, left, right_opposite)
        if exact_sign == 0:
            cospherical_count += 1
        elif exact_sign > 0:
            local_violation_count += 1
        if exact_sign != float_sign:
            float_insphere_disagreement_count += 1

    orientation_zero_count = sum(sign == 0 for sign in exact_orientation.values())
    predicate_consistent = not any(
        (
            duplicate_count,
            orientation_zero_count,
            float_orientation_disagreements,
            unresolved_interior_facet_count,
            nonmanifold_facet_count,
            side_violation_count,
            local_violation_count,
            float_insphere_disagreement_count,
        )
    )
    unique_combinatorics = predicate_consistent and cospherical_count == 0
    return ExactPredicateCaseAudit(
        case_id=case_id,
        point_count=point_array.shape[0],
        top_simplex_count=cells.shape[0],
        unique_top_simplex_count=len(unique_cells),
        coordinate_common_denominator_power=denominator_power,
        duplicate_top_simplex_count=duplicate_count,
        exact_orientation_zero_count=orientation_zero_count,
        float_orientation_sign_disagreement_count=(float_orientation_disagreements),
        interior_facet_count=interior_facet_count,
        audited_interior_facet_count=audited_interior_facet_count,
        unresolved_interior_facet_count=unresolved_interior_facet_count,
        nonmanifold_facet_count=nonmanifold_facet_count,
        interior_facet_side_violation_count=side_violation_count,
        exact_cospherical_interior_facet_count=cospherical_count,
        exact_local_delaunay_violation_count=local_violation_count,
        float_insphere_sign_disagreement_count=(float_insphere_disagreement_count),
        predicate_consistent=predicate_consistent,
        unique_delaunay_combinatorics_supported=unique_combinatorics,
    )


def audit_exact_predicate_panel(
    cases: Iterable[tuple[str, ArrayLike]],
    *,
    evaluation_split: str,
) -> ExactPredicatePanelAudit:
    """Build the current SciPy triangulation and audit it without selection."""

    if not evaluation_split:
        raise ValueError("evaluation_split must be non-empty")
    materialized = tuple(cases)
    if not materialized:
        raise ValueError("cases must be non-empty")
    case_ids = [case_id for case_id, _ in materialized]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case identifiers must be unique")

    audits = []
    for case_id, points in materialized:
        filtration = AlphaFiltration.from_points(points)
        audits.append(
            audit_delaunay_predicates(
                case_id,
                filtration.points,
                filtration.top_simplices,
            )
        )
    return ExactPredicatePanelAudit(
        evaluation_split=evaluation_split,
        cases=tuple(audits),
    )
