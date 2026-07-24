"""Exact-rational audit of simplex filtration values on validated connectivity."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from .exact import IntegerPoint, _integer_coordinates
from .exact_backend import ExactConstructionCaseResult, ExactConstructionPanelResult
from .filtration import AlphaFiltration, Simplex, SimplexRecord
from .geometry import as_point_array
from .synthetic import SyntheticCase


@dataclass(frozen=True)
class ExactSimplexFiltrationRecord:
    """One exact simplex value before any floating-point conversion."""

    vertices: Simplex
    alpha_squared: Fraction
    is_gabriel: bool


@dataclass(frozen=True)
class ExactSimplexFiltration:
    """Exact values and a canonical digest for one validated triangulation."""

    records: tuple[ExactSimplexFiltrationRecord, ...]
    sha256: str


@dataclass(frozen=True)
class ExactRoundedFiltration:
    """Runtime filtration populated from correctly rounded exact values."""

    filtration: AlphaFiltration
    exact_filtration_sha256: str
    simplex_count: int
    exact_records: tuple[ExactSimplexFiltrationRecord, ...]


@dataclass(frozen=True)
class ExactFiltrationCaseAudit:
    """Exact-versus-floating filtration comparison for one benchmark case."""

    case_id: str
    status: str
    backend_accepted: bool
    backend_name: str | None
    backend_version: str | None
    backend_kernel: str | None
    point_count: int
    top_simplex_count: int
    simplex_count: int
    exact_gabriel_simplex_count: int
    float_gabriel_disagreement_count: int
    float_value_exact_match_count: int
    float_value_difference_count: int
    max_absolute_error: float | None
    max_relative_error: float | None
    max_ulp_difference: int | None
    exact_critical_value_count: int
    correctly_rounded_critical_value_count: int
    floating_critical_value_count: int
    exact_tie_group_count: int
    float_split_exact_tie_group_count: int
    adjacent_exact_order_violation_count: int
    exact_filtration_sha256: str | None
    rejection_reasons: tuple[str, ...]

    @property
    def audited(self) -> bool:
        return self.status == "audited"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExactFiltrationPanelAudit:
    """Panel audit that cannot alter primary benchmark selection."""

    evaluation_split: str
    backend_requested: bool
    requested_case_count: int
    accepted_backend_case_count: int
    cases: tuple[ExactFiltrationCaseAudit, ...]

    @property
    def audited_case_count(self) -> int:
        return sum(case.audited for case in self.cases)

    @property
    def value_difference_case_count(self) -> int:
        return sum(
            case.audited and case.float_value_difference_count > 0
            for case in self.cases
        )

    @property
    def gabriel_disagreement_case_count(self) -> int:
        return sum(
            case.audited and case.float_gabriel_disagreement_count > 0
            for case in self.cases
        )

    @property
    def order_violation_case_count(self) -> int:
        return sum(
            case.audited and case.adjacent_exact_order_violation_count > 0
            for case in self.cases
        )

    @property
    def total_simplex_count(self) -> int:
        return sum(case.simplex_count for case in self.cases if case.audited)

    @property
    def float_value_exact_match_count(self) -> int:
        return sum(
            case.float_value_exact_match_count for case in self.cases if case.audited
        )

    @property
    def float_value_difference_count(self) -> int:
        return sum(
            case.float_value_difference_count for case in self.cases if case.audited
        )

    @property
    def exact_tie_split_case_count(self) -> int:
        return sum(
            case.audited and case.float_split_exact_tie_group_count > 0
            for case in self.cases
        )

    @property
    def correctly_rounded_critical_count_mismatch_case_count(self) -> int:
        return sum(
            case.audited
            and case.exact_critical_value_count
            != case.correctly_rounded_critical_value_count
            for case in self.cases
        )

    @property
    def floating_critical_count_mismatch_case_count(self) -> int:
        return sum(
            case.audited
            and case.exact_critical_value_count != case.floating_critical_value_count
            for case in self.cases
        )

    @property
    def max_absolute_error(self) -> float | None:
        values = tuple(
            case.max_absolute_error
            for case in self.cases
            if case.audited and case.max_absolute_error is not None
        )
        return max(values, default=None)

    @property
    def max_relative_error(self) -> float | None:
        values = tuple(
            case.max_relative_error
            for case in self.cases
            if case.audited and case.max_relative_error is not None
        )
        return max(values, default=None)

    @property
    def max_ulp_difference(self) -> int | None:
        values = tuple(
            case.max_ulp_difference
            for case in self.cases
            if case.audited and case.max_ulp_difference is not None
        )
        return max(values, default=None)

    @property
    def all_accepted_cases_audited(self) -> bool:
        return (
            self.accepted_backend_case_count > 0
            and self.audited_case_count == self.accepted_backend_case_count
        )

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.backend_requested:
            reasons.append("no_exact_construction_backend")
        elif self.accepted_backend_case_count != self.requested_case_count:
            reasons.append("one_or_more_backend_results_rejected")
        if self.audited_case_count != self.accepted_backend_case_count:
            reasons.append("one_or_more_accepted_connectivities_not_audited")
        reasons.append("exact_filtration_values_audit_only_not_deployed")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "exact_rational_filtration_value_audit_no_selection",
            "evaluation_split": self.evaluation_split,
            "coordinate_model": "binary64_values_as_exact_rationals",
            "requested_case_count": self.requested_case_count,
            "accepted_backend_case_count": self.accepted_backend_case_count,
            "audited_case_count": self.audited_case_count,
            "value_difference_case_count": self.value_difference_case_count,
            "total_simplex_count": self.total_simplex_count,
            "float_value_exact_match_count": self.float_value_exact_match_count,
            "float_value_difference_count": self.float_value_difference_count,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "max_ulp_difference": self.max_ulp_difference,
            "exact_tie_split_case_count": self.exact_tie_split_case_count,
            "correctly_rounded_critical_count_mismatch_case_count": (
                self.correctly_rounded_critical_count_mismatch_case_count
            ),
            "floating_critical_count_mismatch_case_count": (
                self.floating_critical_count_mismatch_case_count
            ),
            "gabriel_disagreement_case_count": self.gabriel_disagreement_case_count,
            "order_violation_case_count": self.order_violation_case_count,
            "all_accepted_cases_audited": self.all_accepted_cases_audited,
            "exact_filtration_values_applied_to_primary": False,
            "primary_benchmark_results_changed": False,
            "selection_effect": "none",
            "promotion_supported": False,
            "blocking_reasons": list(self.blocking_reasons),
            "cases": [case.to_dict() for case in self.cases],
        }


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction())


def _solve_fraction_system(
    matrix: Sequence[Sequence[int]],
    right_hand_side: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    order = len(matrix)
    if order == 0 or len(right_hand_side) != order:
        raise ValueError("exact system must be non-empty and square")
    augmented = [
        [Fraction(value) for value in row] + [right_hand_side[index]]
        for index, row in enumerate(matrix)
    ]
    if any(len(row) != order + 1 for row in augmented):
        raise ValueError("exact system must be non-empty and square")

    for column in range(order):
        pivot_row = next(
            (row for row in range(column, order) if augmented[row][column] != 0),
            None,
        )
        if pivot_row is None:
            raise ArithmeticError("degenerate simplex has no exact circumsphere")
        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(order):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row],
                        augmented[column],
                        strict=True,
                    )
                ]
    return tuple(augmented[row][-1] for row in range(order))


def _exact_intrinsic_sphere(
    points: Sequence[IntegerPoint],
    simplex: Simplex,
) -> tuple[tuple[Fraction, Fraction, Fraction], Fraction]:
    base = points[simplex[0]]
    if len(simplex) == 1:
        return tuple(Fraction(value) for value in base), Fraction()

    basis = [
        tuple(points[vertex][axis] - base[axis] for axis in range(3))
        for vertex in simplex[1:]
    ]
    gram = [
        [sum(a * b for a, b in zip(left, right, strict=True)) for right in basis]
        for left in basis
    ]
    right_hand_side = tuple(
        Fraction(sum(value * value for value in vector), 2) for vector in basis
    )
    coefficients = _solve_fraction_system(gram, right_hand_side)
    center = tuple(
        Fraction(base[axis])
        + sum(
            (coefficients[index] * basis[index][axis] for index in range(len(basis))),
            Fraction(),
        )
        for axis in range(3)
    )
    radius_squared = _dot(
        tuple(center[axis] - base[axis] for axis in range(3)),
        tuple(center[axis] - base[axis] for axis in range(3)),
    )
    return center, radius_squared


def _squared_distance(
    point: IntegerPoint,
    center: Sequence[Fraction],
) -> Fraction:
    delta = tuple(Fraction(point[axis]) - center[axis] for axis in range(3))
    return _dot(delta, delta)


def _simplices_and_cofaces(
    top_simplices: Sequence[Sequence[int]],
) -> tuple[dict[int, set[Simplex]], dict[Simplex, tuple[Simplex, ...]]]:
    simplices = {dimension: set() for dimension in range(4)}
    for cell in top_simplices:
        ordered = tuple(sorted(int(vertex) for vertex in cell))
        for size in range(1, 5):
            simplices[size - 1].update(combinations(ordered, size))
    cofaces: dict[Simplex, tuple[Simplex, ...]] = {}
    for dimension in range(3):
        for simplex in simplices[dimension]:
            simplex_vertices = set(simplex)
            cofaces[simplex] = tuple(
                coface
                for coface in simplices[dimension + 1]
                if simplex_vertices.issubset(coface)
            )
    return simplices, cofaces


def _validated_top_simplices(
    top_simplices: Sequence[Sequence[int]],
    *,
    point_count: int,
) -> tuple[tuple[int, int, int, int], ...]:
    raw = np.asarray(top_simplices)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != 4:
        raise ValueError("top_simplices must have shape (m, 4) with m >= 1")
    if raw.dtype.kind not in "iu":
        raise ValueError("top_simplices must contain integer vertex indices")
    cells = np.asarray(raw, dtype=np.int64)
    if np.any(cells < 0) or np.any(cells >= point_count):
        raise ValueError("top_simplices contains an out-of-range vertex index")
    canonical = np.sort(cells, axis=1)
    if np.any(np.diff(canonical, axis=1) == 0):
        raise ValueError("top_simplices contains a repeated vertex")
    if np.unique(canonical, axis=0).shape[0] != len(canonical):
        raise ValueError("top_simplices contains duplicate cells")
    if np.unique(canonical).size != point_count:
        raise ValueError("top_simplices must use every input point")
    return tuple(tuple(int(vertex) for vertex in cell) for cell in canonical)


def exact_simplex_filtration(
    points: ArrayLike,
    top_simplices: Sequence[Sequence[int]],
) -> ExactSimplexFiltration:
    """Compute every simplex filtration value using exact rational arithmetic."""

    point_array = as_point_array(points)
    if point_array.shape[1] != 3:
        raise ValueError("exact filtration audit requires 3D points")
    validated_top_simplices = _validated_top_simplices(
        top_simplices,
        point_count=len(point_array),
    )
    integer_points, denominator_power = _integer_coordinates(point_array)
    coordinate_scale_squared = 1 << (2 * denominator_power)
    simplices, cofaces = _simplices_and_cofaces(validated_top_simplices)
    alpha_squared: dict[Simplex, Fraction] = {}
    gabriel: dict[Simplex, bool] = {}

    for dimension in range(3, -1, -1):
        for simplex in sorted(simplices[dimension]):
            center, scaled_radius_squared = _exact_intrinsic_sphere(
                integer_points,
                simplex,
            )
            is_empty = all(
                vertex in simplex
                or _squared_distance(point, center) >= scaled_radius_squared
                for vertex, point in enumerate(integer_points)
            )
            if dimension == 3 and not is_empty:
                raise ArithmeticError(
                    "validated top simplex does not have an exact empty sphere"
                )
            candidates = []
            if is_empty or dimension == 3:
                candidates.append(scaled_radius_squared / coordinate_scale_squared)
            candidates.extend(
                alpha_squared[coface] for coface in cofaces.get(simplex, ())
            )
            if not candidates:
                raise ArithmeticError(
                    f"could not assign an exact filtration value to simplex {simplex}"
                )
            alpha_squared[simplex] = min(candidates)
            gabriel[simplex] = is_empty

    records = tuple(
        ExactSimplexFiltrationRecord(
            vertices=simplex,
            alpha_squared=alpha_squared[simplex],
            is_gabriel=gabriel[simplex],
        )
        for dimension in range(4)
        for simplex in sorted(simplices[dimension])
    )
    digest_payload = [
        {
            "vertices": list(record.vertices),
            "numerator": str(record.alpha_squared.numerator),
            "denominator": str(record.alpha_squared.denominator),
            "is_gabriel": record.is_gabriel,
        }
        for record in records
    ]
    serialized = json.dumps(
        digest_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ExactSimplexFiltration(
        records=records,
        sha256=hashlib.sha256(serialized.encode("ascii")).hexdigest(),
    )


def exact_rounded_filtration(
    points: ArrayLike,
    top_simplices: Sequence[Sequence[int]],
) -> ExactRoundedFiltration:
    """Build a float runtime view from correctly rounded exact rational values."""

    point_array = as_point_array(points)
    if point_array.shape[1] != 3:
        raise ValueError("exact rounded filtration requires 3D points")
    validated_top_simplices = _validated_top_simplices(
        top_simplices,
        point_count=len(point_array),
    )
    exact = exact_simplex_filtration(point_array, validated_top_simplices)
    records = []
    for record in exact.records:
        try:
            alpha_squared = float(record.alpha_squared)
        except OverflowError as error:
            raise ArithmeticError(
                "exact filtration value is not finite when rounded"
            ) from error
        if not math.isfinite(alpha_squared) or alpha_squared < 0.0:
            raise ArithmeticError("exact filtration value is not finite when rounded")
        records.append(
            SimplexRecord(
                vertices=record.vertices,
                alpha_squared=alpha_squared,
                is_gabriel=record.is_gabriel,
            )
        )
    filtration = AlphaFiltration(
        point_array,
        np.ascontiguousarray(validated_top_simplices, dtype=np.int64),
        records,
    )
    return ExactRoundedFiltration(
        filtration=filtration,
        exact_filtration_sha256=exact.sha256,
        simplex_count=len(exact.records),
        exact_records=exact.records,
    )


def _ulp_difference(left: float, right: float) -> int:
    left_bits = int(np.asarray(left, dtype=np.float64).view(np.uint64))
    right_bits = int(np.asarray(right, dtype=np.float64).view(np.uint64))
    return abs(left_bits - right_bits)


def _rejected_case(
    case_id: str,
    point_count: int,
    construction: ExactConstructionCaseResult | None,
    reason: str,
) -> ExactFiltrationCaseAudit:
    return ExactFiltrationCaseAudit(
        case_id=case_id,
        status="not_audited",
        backend_accepted=bool(construction and construction.accepted),
        backend_name=None if construction is None else construction.backend_name,
        backend_version=None if construction is None else construction.backend_version,
        backend_kernel=None if construction is None else construction.backend_kernel,
        point_count=point_count,
        top_simplex_count=0 if construction is None else construction.top_simplex_count,
        simplex_count=0,
        exact_gabriel_simplex_count=0,
        float_gabriel_disagreement_count=0,
        float_value_exact_match_count=0,
        float_value_difference_count=0,
        max_absolute_error=None,
        max_relative_error=None,
        max_ulp_difference=None,
        exact_critical_value_count=0,
        correctly_rounded_critical_value_count=0,
        floating_critical_value_count=0,
        exact_tie_group_count=0,
        float_split_exact_tie_group_count=0,
        adjacent_exact_order_violation_count=0,
        exact_filtration_sha256=None,
        rejection_reasons=(reason,),
    )


def audit_exact_filtration_case(
    case_id: str,
    points: ArrayLike,
    construction: ExactConstructionCaseResult,
) -> ExactFiltrationCaseAudit:
    """Audit one host-validated exact connectivity without changing selection."""

    point_array = as_point_array(points)
    if not construction.accepted or construction.validated_top_simplices is None:
        return _rejected_case(
            case_id,
            len(point_array),
            construction,
            "backend_connectivity_not_accepted",
        )
    try:
        exact = exact_simplex_filtration(
            point_array,
            construction.validated_top_simplices,
        )
        floating = AlphaFiltration.from_top_simplices(
            point_array,
            construction.validated_top_simplices,
        )
    except (ArithmeticError, np.linalg.LinAlgError, ValueError) as error:
        return _rejected_case(
            case_id,
            len(point_array),
            construction,
            f"exact_filtration_audit_failed:{type(error).__name__}",
        )

    floating_by_simplex = {record.vertices: record for record in floating.records}
    absolute_errors = []
    relative_errors = []
    ulp_differences = []
    exact_matches = 0
    gabriel_disagreements = 0
    exact_groups: dict[Fraction, list[float]] = defaultdict(list)
    ordered = []
    for record in exact.records:
        floating_record = floating_by_simplex[record.vertices]
        try:
            correctly_rounded = float(record.alpha_squared)
        except OverflowError:
            return _rejected_case(
                case_id,
                len(point_array),
                construction,
                "exact_filtration_value_not_finite_when_rounded",
            )
        if not math.isfinite(correctly_rounded):
            return _rejected_case(
                case_id,
                len(point_array),
                construction,
                "exact_filtration_value_not_finite_when_rounded",
            )
        floating_value = floating_record.alpha_squared
        exact_matches += floating_value == correctly_rounded
        gabriel_disagreements += floating_record.is_gabriel != record.is_gabriel
        exact_float_value = Fraction.from_float(floating_value)
        absolute_error_fraction = abs(exact_float_value - record.alpha_squared)
        absolute_errors.append(float(absolute_error_fraction))
        relative_errors.append(
            0.0
            if record.alpha_squared == 0
            else float(absolute_error_fraction / record.alpha_squared)
        )
        ulp_differences.append(_ulp_difference(floating_value, correctly_rounded))
        exact_groups[record.alpha_squared].append(floating_value)
        ordered.append((record.alpha_squared, record.vertices, floating_value))

    ordered.sort()
    adjacent_order_violations = sum(
        left_exact < right_exact and left_float > right_float
        for (left_exact, _, left_float), (right_exact, _, right_float) in zip(
            ordered[:-1],
            ordered[1:],
            strict=True,
        )
    )
    exact_tie_groups = [values for values in exact_groups.values() if len(values) > 1]
    split_tie_groups = sum(len(set(values)) > 1 for values in exact_tie_groups)
    floating_values = [record.alpha_squared for record in floating.records]
    rounded_exact_values = [float(record.alpha_squared) for record in exact.records]
    simplex_count = len(exact.records)
    return ExactFiltrationCaseAudit(
        case_id=case_id,
        status="audited",
        backend_accepted=True,
        backend_name=construction.backend_name,
        backend_version=construction.backend_version,
        backend_kernel=construction.backend_kernel,
        point_count=len(point_array),
        top_simplex_count=construction.top_simplex_count,
        simplex_count=simplex_count,
        exact_gabriel_simplex_count=sum(record.is_gabriel for record in exact.records),
        float_gabriel_disagreement_count=gabriel_disagreements,
        float_value_exact_match_count=exact_matches,
        float_value_difference_count=simplex_count - exact_matches,
        max_absolute_error=max(absolute_errors, default=0.0),
        max_relative_error=max(relative_errors, default=0.0),
        max_ulp_difference=max(ulp_differences, default=0),
        exact_critical_value_count=len(exact_groups),
        correctly_rounded_critical_value_count=len(set(rounded_exact_values)),
        floating_critical_value_count=len(set(floating_values)),
        exact_tie_group_count=len(exact_tie_groups),
        float_split_exact_tie_group_count=split_tie_groups,
        adjacent_exact_order_violation_count=adjacent_order_violations,
        exact_filtration_sha256=exact.sha256,
        rejection_reasons=(),
    )


def evaluate_exact_filtration_panel(
    cases: Sequence[SyntheticCase],
    *,
    construction_result: ExactConstructionPanelResult,
) -> ExactFiltrationPanelAudit:
    """Audit exact simplex filtration values for every requested panel case."""

    construction_by_case = {
        result.case_id: result for result in construction_result.cases
    }
    audits = []
    for case in cases:
        case_id = case.family.value
        construction = construction_by_case.get(case_id)
        if construction is None:
            audits.append(
                _rejected_case(
                    case_id,
                    len(case.points),
                    None,
                    "backend_case_result_missing",
                )
            )
            continue
        audits.append(
            audit_exact_filtration_case(
                case_id,
                case.points,
                construction,
            )
        )
    return ExactFiltrationPanelAudit(
        evaluation_split=construction_result.evaluation_split,
        backend_requested=construction_result.backend_requested,
        requested_case_count=len(cases),
        accepted_backend_case_count=construction_result.accepted_case_count,
        cases=tuple(audits),
    )
