import numpy as np

from pftf_alpha.exact_backend import (
    COORDINATE_MODEL,
    OPERATION,
    exact_construction_request,
    validate_exact_construction_response,
)


def _points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )


def _response(points: np.ndarray) -> dict[str, object]:
    _, request_sha256 = exact_construction_request("strict_types", points)
    return {
        "schema_version": 1,
        "operation": OPERATION,
        "case_id": "strict_types",
        "coordinate_model": COORDINATE_MODEL,
        "point_count": len(points),
        "request_sha256": request_sha256,
        "backend": {
            "name": "protocol-test-fixture",
            "version": "1",
            "kernel": "fixture_exact_integer_kernel",
            "exact_construction": True,
        },
        "top_simplices": [[0, 1, 2, 3], [0, 1, 2, 4]],
    }


def test_protocol_rejects_float_cell_indices_before_numpy_conversion() -> None:
    points = _points()
    response = _response(points)
    response["top_simplices"] = [[0.0, 1, 2, 3], [0, 1, 2, 4]]

    result = validate_exact_construction_response(
        "strict_types",
        points,
        response,
    )

    assert not result.accepted
    assert not result.protocol_valid
    assert result.rejection_reasons == ("top_simplex_index_type_invalid",)


def test_protocol_rejects_boolean_schema_version() -> None:
    points = _points()
    response = _response(points)
    response["schema_version"] = True

    result = validate_exact_construction_response(
        "strict_types",
        points,
        response,
    )

    assert not result.accepted
    assert not result.protocol_valid
    assert result.rejection_reasons == ("protocol_version_type_invalid",)
