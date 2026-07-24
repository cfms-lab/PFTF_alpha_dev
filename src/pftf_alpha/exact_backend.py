"""Fail-closed protocol for an optional exact 3D Delaunay backend.

The backend receives exact rational encodings of the input binary64 values and
returns tetrahedral connectivity.  The host validates protocol binding,
topological incidence, convex-hull support, exact volume coverage, and exact
orientation/in-sphere predicates.  Validated connectivity is not yet applied
to benchmark selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exact import (
    ExactPredicateCaseAudit,
    _integer_coordinates,
    _orientation_sign,
    _validated_top_simplices,
    audit_delaunay_predicates,
)
from .geometry import as_point_array

PROTOCOL_VERSION = 1
OPERATION = "delaunay_tetrahedralization_3"
COORDINATE_MODEL = "binary64_values_as_exact_rationals"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024

IntArray = NDArray[np.int64]
IntegerPoint = tuple[int, int, int]
Cell = tuple[int, int, int, int]
Facet = tuple[int, int, int]


@dataclass(frozen=True)
class ExactConstructionCaseResult:
    """Host validation result for one backend response."""

    case_id: str
    status: str
    accepted: bool
    protocol_valid: bool
    backend_name: str | None
    backend_version: str | None
    backend_kernel: str | None
    backend_attested_exact_construction: bool
    response_sha256: str | None
    point_count: int
    top_simplex_count: int
    used_point_count: int
    boundary_facet_count: int
    boundary_support_violation_count: int
    boundary_coplanar_point_count: int
    cell_volume_numerator: str | None
    hull_volume_numerator: str | None
    exact_volume_match: bool
    exact_predicate_audit: ExactPredicateCaseAudit | None
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExactConstructionPanelResult:
    """Requested-split backend handoff result without benchmark selection."""

    evaluation_split: str
    backend_requested: bool
    cases: tuple[ExactConstructionCaseResult, ...]

    @property
    def backend_responded_case_count(self) -> int:
        return sum(case.protocol_valid for case in self.cases)

    @property
    def accepted_case_count(self) -> int:
        return sum(case.accepted for case in self.cases)

    @property
    def backend_handoff_validated(self) -> bool:
        return bool(self.cases) and self.accepted_case_count == len(self.cases)

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        if not self.backend_requested:
            return ("no_exact_construction_backend",)
        reasons = []
        if not self.backend_handoff_validated:
            reasons.append("one_or_more_backend_results_rejected")
        reasons.append("exact_connectivity_not_applied_to_benchmark_selection")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "optional_backend_handoff_validation_no_selection",
            "evaluation_split": self.evaluation_split,
            "protocol_version": PROTOCOL_VERSION,
            "operation": OPERATION,
            "coordinate_model": COORDINATE_MODEL,
            "backend_requested": self.backend_requested,
            "backend_responded_case_count": self.backend_responded_case_count,
            "accepted_case_count": self.accepted_case_count,
            "backend_handoff_validated": self.backend_handoff_validated,
            "exact_construction_applied_to_benchmark": False,
            "changes_benchmark_selection": False,
            "promotion_supported": False,
            "blocking_reasons": list(self.blocking_reasons),
            "cases": [case.to_dict() for case in self.cases],
        }


class ExactBackendProtocolError(ValueError):
    """Stable protocol error carrying a machine-readable rejection reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def exact_construction_request(
    case_id: str,
    points: ArrayLike,
) -> tuple[dict[str, object], str]:
    """Encode input binary64 coordinates as exact numerator/denominator pairs."""

    if not case_id:
        raise ValueError("case_id must be non-empty")
    point_array = as_point_array(points)
    if point_array.shape[1] != 3:
        raise ValueError("exact construction backend requires 3D points")
    encoded_points = [
        [
            [str(numerator), str(denominator)]
            for numerator, denominator in (
                float(coordinate).as_integer_ratio() for coordinate in point
            )
        ]
        for point in point_array
    ]
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_VERSION,
        "operation": OPERATION,
        "case_id": case_id,
        "coordinate_model": COORDINATE_MODEL,
        "point_count": point_array.shape[0],
        "points": encoded_points,
    }
    serialized = _canonical_json(payload)
    digest = hashlib.sha256(serialized.encode("ascii")).hexdigest()
    return payload, digest


def _require_mapping(value: object, reason: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExactBackendProtocolError(reason)
    return value


def _require_string(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExactBackendProtocolError(reason)
    return value


def _parse_response(
    response: Mapping[str, object],
    *,
    case_id: str,
    point_count: int,
    request_sha256: str,
) -> tuple[str, str, str, bool, object]:
    if type(response.get("schema_version")) is not int:
        raise ExactBackendProtocolError("protocol_version_type_invalid")
    if response.get("schema_version") != PROTOCOL_VERSION:
        raise ExactBackendProtocolError("protocol_version_mismatch")
    if response.get("operation") != OPERATION:
        raise ExactBackendProtocolError("operation_mismatch")
    if response.get("case_id") != case_id:
        raise ExactBackendProtocolError("case_id_mismatch")
    if response.get("coordinate_model") != COORDINATE_MODEL:
        raise ExactBackendProtocolError("coordinate_model_mismatch")
    if type(response.get("point_count")) is not int:
        raise ExactBackendProtocolError("point_count_type_invalid")
    if response.get("point_count") != point_count:
        raise ExactBackendProtocolError("point_count_mismatch")
    if response.get("request_sha256") != request_sha256:
        raise ExactBackendProtocolError("request_sha256_mismatch")

    backend = _require_mapping(
        response.get("backend"),
        "backend_metadata_missing",
    )
    name = _require_string(backend.get("name"), "backend_name_missing")
    version = _require_string(backend.get("version"), "backend_version_missing")
    kernel = _require_string(backend.get("kernel"), "backend_kernel_missing")
    exact_attestation = backend.get("exact_construction")
    if not isinstance(exact_attestation, bool):
        raise ExactBackendProtocolError("exact_construction_attestation_invalid")
    if "top_simplices" not in response:
        raise ExactBackendProtocolError("top_simplices_missing")
    return name, version, kernel, exact_attestation, response["top_simplices"]


def _strict_response_cells(value: object) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ExactBackendProtocolError("top_simplices_not_nonempty_list")
    cells = []
    for cell in value:
        if not isinstance(cell, list) or len(cell) != 4:
            raise ExactBackendProtocolError("top_simplex_shape_invalid")
        if any(type(vertex) is not int for vertex in cell):
            raise ExactBackendProtocolError("top_simplex_index_type_invalid")
        cells.append(cell)
    return cells


def _determinant3(
    left: Sequence[int],
    middle: Sequence[int],
    right: Sequence[int],
) -> int:
    return (
        left[0] * (middle[1] * right[2] - middle[2] * right[1])
        - left[1] * (middle[0] * right[2] - middle[2] * right[0])
        + left[2] * (middle[0] * right[1] - middle[1] * right[0])
    )


def _tetrahedron_volume_numerator(
    points: Sequence[IntegerPoint],
    cell: Cell,
) -> int:
    base = points[cell[0]]
    edges = [
        tuple(points[vertex][axis] - base[axis] for axis in range(3))
        for vertex in cell[1:]
    ]
    return abs(_determinant3(edges[0], edges[1], edges[2]))


def _facet_owners(cells: Sequence[Cell]) -> dict[Facet, list[Cell]]:
    owners: dict[Facet, list[Cell]] = defaultdict(list)
    for cell in cells:
        for facet in combinations(cell, 3):
            owners[facet].append(cell)
    return owners


def _boundary_validation(
    points: Sequence[IntegerPoint],
    cells: Sequence[Cell],
) -> tuple[int, int, int, int]:
    """Return boundary count, support violations, coplanar pairs, hull volume."""

    boundary_facet_count = 0
    support_violations = 0
    coplanar_pairs = 0
    oriented_volume_sum = 0
    for facet, owners in _facet_owners(cells).items():
        if len(owners) != 1:
            continue
        boundary_facet_count += 1
        owner = owners[0]
        opposite = next(vertex for vertex in owner if vertex not in facet)
        interior_side = _orientation_sign(points, (*facet, opposite))
        if interior_side == 0:
            support_violations += 1
            continue
        for vertex in range(len(points)):
            if vertex in facet:
                continue
            side = _orientation_sign(points, (*facet, vertex))
            if side == 0:
                coplanar_pairs += 1
            elif side != interior_side:
                support_violations += 1

        first, second, third = facet
        if interior_side > 0:
            second, third = third, second
        oriented_volume_sum += _determinant3(
            points[first],
            points[second],
            points[third],
        )
    return (
        boundary_facet_count,
        support_violations,
        coplanar_pairs,
        abs(oriented_volume_sum),
    )


def _exact_connectivity_rejection_reasons(
    audit: ExactPredicateCaseAudit,
) -> list[str]:
    reasons = []
    if audit.duplicate_top_simplex_count:
        reasons.append("duplicate_top_simplices")
    if audit.exact_orientation_zero_count:
        reasons.append("degenerate_top_simplices")
    if audit.unresolved_interior_facet_count:
        reasons.append("unresolved_interior_facets")
    if audit.nonmanifold_facet_count:
        reasons.append("nonmanifold_facets")
    if audit.interior_facet_side_violation_count:
        reasons.append("interior_facet_side_violations")
    if audit.exact_local_delaunay_violation_count:
        reasons.append("exact_local_delaunay_violations")
    return reasons


def _failure_result(
    case_id: str,
    point_count: int,
    reason: str,
    *,
    response_sha256: str | None = None,
) -> ExactConstructionCaseResult:
    return ExactConstructionCaseResult(
        case_id=case_id,
        status="rejected",
        accepted=False,
        protocol_valid=False,
        backend_name=None,
        backend_version=None,
        backend_kernel=None,
        backend_attested_exact_construction=False,
        response_sha256=response_sha256,
        point_count=point_count,
        top_simplex_count=0,
        used_point_count=0,
        boundary_facet_count=0,
        boundary_support_violation_count=0,
        boundary_coplanar_point_count=0,
        cell_volume_numerator=None,
        hull_volume_numerator=None,
        exact_volume_match=False,
        exact_predicate_audit=None,
        rejection_reasons=(reason,),
    )


def validate_exact_construction_response(
    case_id: str,
    points: ArrayLike,
    response: Mapping[str, object],
    *,
    response_sha256: str | None = None,
) -> ExactConstructionCaseResult:
    """Validate one backend response with exact host-side invariants."""

    point_array = as_point_array(points)
    if point_array.shape[1] != 3:
        raise ValueError("exact construction response requires 3D points")
    _, request_sha256 = exact_construction_request(case_id, point_array)
    canonical_response = _canonical_json(response)
    digest = (
        response_sha256
        or hashlib.sha256(canonical_response.encode("ascii")).hexdigest()
    )

    try:
        name, version, kernel, exact_attestation, raw_cells = _parse_response(
            response,
            case_id=case_id,
            point_count=point_array.shape[0],
            request_sha256=request_sha256,
        )
        cells_array = _validated_top_simplices(
            _strict_response_cells(raw_cells),
            point_count=point_array.shape[0],
        )
    except ExactBackendProtocolError as error:
        return _failure_result(
            case_id,
            point_array.shape[0],
            error.reason,
            response_sha256=digest,
        )
    except (IndexError, TypeError, ValueError):
        return _failure_result(
            case_id,
            point_array.shape[0],
            "top_simplices_invalid",
            response_sha256=digest,
        )

    canonical_cells = tuple(
        dict.fromkeys(
            tuple(sorted(int(vertex) for vertex in cell)) for cell in cells_array
        )
    )
    used_vertices = {vertex for cell in canonical_cells for vertex in cell}
    integer_points, _ = _integer_coordinates(point_array)
    audit = audit_delaunay_predicates(case_id, point_array, cells_array)
    (
        boundary_facet_count,
        boundary_support_violations,
        boundary_coplanar_pairs,
        hull_volume_numerator,
    ) = _boundary_validation(integer_points, canonical_cells)
    cell_volume_numerator = sum(
        _tetrahedron_volume_numerator(integer_points, cell) for cell in canonical_cells
    )
    exact_volume_match = cell_volume_numerator == hull_volume_numerator

    rejection_reasons = _exact_connectivity_rejection_reasons(audit)
    if not exact_attestation:
        rejection_reasons.append("backend_did_not_attest_exact_construction")
    if len(used_vertices) != point_array.shape[0]:
        rejection_reasons.append("not_all_input_points_used")
    if boundary_facet_count == 0:
        rejection_reasons.append("boundary_facets_missing")
    if boundary_support_violations:
        rejection_reasons.append("boundary_support_violations")
    if not exact_volume_match:
        rejection_reasons.append("exact_volume_coverage_mismatch")

    accepted = not rejection_reasons
    return ExactConstructionCaseResult(
        case_id=case_id,
        status="accepted" if accepted else "rejected",
        accepted=accepted,
        protocol_valid=True,
        backend_name=name,
        backend_version=version,
        backend_kernel=kernel,
        backend_attested_exact_construction=exact_attestation,
        response_sha256=digest,
        point_count=point_array.shape[0],
        top_simplex_count=cells_array.shape[0],
        used_point_count=len(used_vertices),
        boundary_facet_count=boundary_facet_count,
        boundary_support_violation_count=boundary_support_violations,
        boundary_coplanar_point_count=boundary_coplanar_pairs,
        cell_volume_numerator=str(cell_volume_numerator),
        hull_volume_numerator=str(hull_volume_numerator),
        exact_volume_match=exact_volume_match,
        exact_predicate_audit=audit,
        rejection_reasons=tuple(rejection_reasons),
    )


def run_exact_construction_backend(
    command: Sequence[str],
    case_id: str,
    points: ArrayLike,
    *,
    timeout_seconds: float = 60.0,
) -> ExactConstructionCaseResult:
    """Run one explicitly supplied backend command and validate its response."""

    point_array = as_point_array(points)
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must contain non-empty strings")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    request, _ = exact_construction_request(case_id, point_array)
    request_text = _canonical_json(request)

    try:
        completed = subprocess.run(
            list(command),
            input=request_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failure_result(
            case_id,
            point_array.shape[0],
            "backend_timeout",
        )
    except (OSError, UnicodeError):
        return _failure_result(
            case_id,
            point_array.shape[0],
            "backend_execution_failed",
        )

    response_bytes = completed.stdout.encode("utf-8")
    response_sha256 = hashlib.sha256(response_bytes).hexdigest()
    if completed.returncode != 0:
        return _failure_result(
            case_id,
            point_array.shape[0],
            "backend_exit_nonzero",
            response_sha256=response_sha256,
        )
    if len(response_bytes) > MAX_RESPONSE_BYTES:
        return _failure_result(
            case_id,
            point_array.shape[0],
            "backend_response_too_large",
            response_sha256=response_sha256,
        )
    try:
        response: Any = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return _failure_result(
            case_id,
            point_array.shape[0],
            "backend_response_invalid_json",
            response_sha256=response_sha256,
        )
    if not isinstance(response, Mapping):
        return _failure_result(
            case_id,
            point_array.shape[0],
            "backend_response_not_object",
            response_sha256=response_sha256,
        )
    return validate_exact_construction_response(
        case_id,
        point_array,
        response,
        response_sha256=response_sha256,
    )


def evaluate_exact_construction_panel(
    cases: Iterable[tuple[str, ArrayLike]],
    *,
    evaluation_split: str,
    backend_command: Sequence[str] | None,
    timeout_seconds: float = 60.0,
) -> ExactConstructionPanelResult:
    """Evaluate an optional backend; absence is a recorded fail-closed result."""

    if not evaluation_split:
        raise ValueError("evaluation_split must be non-empty")
    materialized = tuple(cases)
    if not materialized:
        raise ValueError("cases must be non-empty")
    case_ids = [case_id for case_id, _ in materialized]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case identifiers must be unique")
    if backend_command is None:
        return ExactConstructionPanelResult(
            evaluation_split=evaluation_split,
            backend_requested=False,
            cases=(),
        )

    results = tuple(
        run_exact_construction_backend(
            backend_command,
            case_id,
            points,
            timeout_seconds=timeout_seconds,
        )
        for case_id, points in materialized
    )
    return ExactConstructionPanelResult(
        evaluation_split=evaluation_split,
        backend_requested=True,
        cases=results,
    )
