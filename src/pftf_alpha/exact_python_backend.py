"""Pure-Python exact 3D Delaunay backend for small audit panels.

The construction enumerates every four-point candidate and retains precisely
the tetrahedra whose circumspheres are empty under exact integer arithmetic.
It does not use SciPy/Qhull connectivity.  Exact empty-cosphere ambiguity fails
closed instead of choosing an undocumented symbolic perturbation.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from .exact_backend import (
    COORDINATE_MODEL,
    OPERATION,
    PROTOCOL_VERSION,
    _canonical_json,
)

IntegerPoint = tuple[int, int, int]
Cell = tuple[int, int, int, int]

BACKEND_NAME = "pftf_alpha_python_exact"
BACKEND_VERSION = "1"
BACKEND_KERNEL = "exact_integer_empty_sphere_enumeration_3"
MAX_EXACT_POINT_COUNT = 64
MAX_REQUEST_BYTES = 16 * 1024 * 1024


class ExactPythonBackendError(ValueError):
    """Fail-closed construction or request error with a stable reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ExactPythonConstruction:
    """Exact construction output and deterministic work counters."""

    point_count: int
    candidate_cell_count: int
    degenerate_candidate_count: int
    exact_power_test_count: int
    top_simplices: tuple[Cell, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "point_count": self.point_count,
            "candidate_cell_count": self.candidate_cell_count,
            "degenerate_candidate_count": self.degenerate_candidate_count,
            "exact_power_test_count": self.exact_power_test_count,
            "top_simplex_count": len(self.top_simplices),
        }


@dataclass(frozen=True)
class _SphereCertificate:
    base: IntegerPoint
    base_norm_squared: int
    center_denominator: int
    center_numerators: tuple[int, int, int]


def _sign(value: int) -> int:
    return int(value > 0) - int(value < 0)


def _determinant3(matrix: Sequence[Sequence[int]]) -> int:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("matrix must have shape (3, 3)")
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _norm_squared(point: IntegerPoint) -> int:
    return sum(coordinate * coordinate for coordinate in point)


def _sphere_certificate(
    points: Sequence[IntegerPoint],
    cell: Cell,
) -> _SphereCertificate | None:
    base = points[cell[0]]
    rows = [
        [2 * (points[vertex][axis] - base[axis]) for axis in range(3)]
        for vertex in cell[1:]
    ]
    right_hand_side = [
        _norm_squared(points[vertex]) - _norm_squared(base) for vertex in cell[1:]
    ]
    denominator = _determinant3(rows)
    if denominator == 0:
        return None

    numerators = []
    for column in range(3):
        replaced = [row.copy() for row in rows]
        for row_index in range(3):
            replaced[row_index][column] = right_hand_side[row_index]
        numerators.append(_determinant3(replaced))
    return _SphereCertificate(
        base=base,
        base_norm_squared=_norm_squared(base),
        center_denominator=denominator,
        center_numerators=tuple(numerators),
    )


def _sphere_power_sign(
    point: IntegerPoint,
    certificate: _SphereCertificate,
) -> int:
    """Return negative/zero/positive for inside/on/outside the circumsphere."""

    numerator = certificate.center_denominator * (
        _norm_squared(point) - certificate.base_norm_squared
    ) - 2 * sum(
        certificate.center_numerators[axis] * (point[axis] - certificate.base[axis])
        for axis in range(3)
    )
    return _sign(numerator) * _sign(certificate.center_denominator)


def exact_delaunay_tetrahedra(
    points: Sequence[IntegerPoint],
    *,
    max_point_count: int = MAX_EXACT_POINT_COUNT,
) -> ExactPythonConstruction:
    """Construct unique 3D Delaunay cells by exact empty-sphere enumeration."""

    materialized = tuple(
        tuple(int(coordinate) for coordinate in point) for point in points
    )
    if len(materialized) < 4:
        raise ExactPythonBackendError("at_least_four_points_required")
    if len(materialized) > max_point_count:
        raise ExactPythonBackendError("point_count_exceeds_exact_backend_limit")
    if any(len(point) != 3 for point in materialized):
        raise ExactPythonBackendError("points_must_be_three_dimensional")
    if len(set(materialized)) != len(materialized):
        raise ExactPythonBackendError("duplicate_points_not_supported")

    candidate_count = math.comb(len(materialized), 4)
    degenerate_count = 0
    power_test_count = 0
    cells = []
    for candidate in combinations(range(len(materialized)), 4):
        cell = tuple(candidate)
        certificate = _sphere_certificate(materialized, cell)
        if certificate is None:
            degenerate_count += 1
            continue

        empty = True
        cospherical_boundary = False
        for query_index, point in enumerate(materialized):
            if query_index in cell:
                continue
            power_test_count += 1
            power_sign = _sphere_power_sign(point, certificate)
            cospherical_boundary = cospherical_boundary or power_sign == 0
            if power_sign < 0:
                empty = False
                break
        if empty and cospherical_boundary:
            raise ExactPythonBackendError(
                "exact_empty_cospherical_ambiguity_not_supported"
            )
        if empty:
            cells.append(cell)

    if not cells:
        raise ExactPythonBackendError("exact_delaunay_construction_empty")
    used_vertices = {vertex for cell in cells for vertex in cell}
    if len(used_vertices) != len(materialized):
        raise ExactPythonBackendError("exact_delaunay_did_not_use_every_point")
    return ExactPythonConstruction(
        point_count=len(materialized),
        candidate_cell_count=candidate_count,
        degenerate_candidate_count=degenerate_count,
        exact_power_test_count=power_test_count,
        top_simplices=tuple(cells),
    )


def _require_integer_string(value: object, reason: str) -> int:
    if not isinstance(value, str) or not value:
        raise ExactPythonBackendError(reason)
    try:
        return int(value)
    except ValueError as error:
        raise ExactPythonBackendError(reason) from error


def decode_exact_request(
    request: Mapping[str, object],
) -> tuple[str, tuple[IntegerPoint, ...]]:
    """Decode protocol rational pairs to one common exact integer scale."""

    if type(request.get("schema_version")) is not int:
        raise ExactPythonBackendError("protocol_version_type_invalid")
    if request.get("schema_version") != PROTOCOL_VERSION:
        raise ExactPythonBackendError("protocol_version_mismatch")
    if request.get("operation") != OPERATION:
        raise ExactPythonBackendError("operation_mismatch")
    if request.get("coordinate_model") != COORDINATE_MODEL:
        raise ExactPythonBackendError("coordinate_model_mismatch")
    case_id = request.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ExactPythonBackendError("case_id_invalid")
    point_count = request.get("point_count")
    if type(point_count) is not int or point_count < 4:
        raise ExactPythonBackendError("point_count_invalid")
    if point_count > MAX_EXACT_POINT_COUNT:
        raise ExactPythonBackendError("point_count_exceeds_exact_backend_limit")
    encoded_points = request.get("points")
    if not isinstance(encoded_points, list) or len(encoded_points) != point_count:
        raise ExactPythonBackendError("points_shape_invalid")

    ratios: list[list[tuple[int, int]]] = []
    for encoded_point in encoded_points:
        if not isinstance(encoded_point, list) or len(encoded_point) != 3:
            raise ExactPythonBackendError("points_shape_invalid")
        point_ratios = []
        for encoded_coordinate in encoded_point:
            if not isinstance(encoded_coordinate, list) or len(encoded_coordinate) != 2:
                raise ExactPythonBackendError("coordinate_ratio_shape_invalid")
            numerator = _require_integer_string(
                encoded_coordinate[0],
                "coordinate_numerator_invalid",
            )
            denominator = _require_integer_string(
                encoded_coordinate[1],
                "coordinate_denominator_invalid",
            )
            if denominator <= 0 or denominator & (denominator - 1):
                raise ExactPythonBackendError(
                    "coordinate_denominator_not_positive_power_of_two"
                )
            point_ratios.append((numerator, denominator))
        ratios.append(point_ratios)

    common_denominator = max(
        denominator for point in ratios for _, denominator in point
    )
    integer_points = tuple(
        tuple(
            numerator * (common_denominator // denominator)
            for numerator, denominator in point
        )
        for point in ratios
    )
    return case_id, integer_points


def exact_backend_response(
    request: Mapping[str, object],
) -> dict[str, object]:
    """Construct one protocol response without using floating connectivity."""

    case_id, integer_points = decode_exact_request(request)
    construction = exact_delaunay_tetrahedra(integer_points)
    request_sha256 = hashlib.sha256(
        _canonical_json(request).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": PROTOCOL_VERSION,
        "operation": OPERATION,
        "case_id": case_id,
        "coordinate_model": COORDINATE_MODEL,
        "point_count": len(integer_points),
        "request_sha256": request_sha256,
        "backend": {
            "name": BACKEND_NAME,
            "version": BACKEND_VERSION,
            "kernel": BACKEND_KERNEL,
            "exact_construction": True,
        },
        "construction_diagnostics": construction.to_dict(),
        "top_simplices": [list(cell) for cell in construction.top_simplices],
    }


def main() -> int:
    raw_request = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw_request) > MAX_REQUEST_BYTES:
        print("backend_request_too_large", file=sys.stderr)
        return 2
    try:
        decoded = json.loads(raw_request.decode("ascii"))
        if not isinstance(decoded, Mapping):
            raise ExactPythonBackendError("backend_request_not_object")
        response = exact_backend_response(decoded)
    except (
        ExactPythonBackendError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        reason = getattr(error, "reason", "backend_request_invalid")
        print(reason, file=sys.stderr)
        return 2
    sys.stdout.write(_canonical_json(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
