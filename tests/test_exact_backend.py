import hashlib
import json
import sys

import numpy as np

from pftf_alpha.exact_backend import (
    COORDINATE_MODEL,
    OPERATION,
    PROTOCOL_VERSION,
    evaluate_exact_construction_panel,
    exact_construction_request,
    run_exact_construction_backend,
    validate_exact_construction_response,
)


def _bipyramid_points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )


def _response(
    case_id: str,
    points: np.ndarray,
    cells: list[list[int]],
    *,
    exact_construction: bool = True,
) -> dict[str, object]:
    _, request_sha256 = exact_construction_request(case_id, points)
    return {
        "schema_version": PROTOCOL_VERSION,
        "operation": OPERATION,
        "case_id": case_id,
        "coordinate_model": COORDINATE_MODEL,
        "point_count": len(points),
        "request_sha256": request_sha256,
        "backend": {
            "name": "protocol-test-fixture",
            "version": "1",
            "kernel": "fixture_exact_integer_kernel",
            "exact_construction": exact_construction,
        },
        "top_simplices": cells,
    }


def test_request_encodes_binary64_values_as_exact_ratios() -> None:
    points = _bipyramid_points()
    points[0, 0] = 0.1

    request, digest = exact_construction_request("ratio_case", points)
    serialized = json.dumps(
        request,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    assert request["coordinate_model"] == COORDINATE_MODEL
    assert request["points"][0][0] == [
        "3602879701896397",
        "36028797018963968",
    ]
    assert digest == hashlib.sha256(serialized.encode("ascii")).hexdigest()


def test_validator_accepts_complete_exact_delaunay_connectivity() -> None:
    points = _bipyramid_points()
    response = _response(
        "bipyramid",
        points,
        [[0, 1, 2, 3], [0, 1, 2, 4]],
    )

    result = validate_exact_construction_response(
        "bipyramid",
        points,
        response,
    )

    assert result.accepted
    assert result.protocol_valid
    assert result.used_point_count == 5
    assert result.boundary_facet_count == 6
    assert result.boundary_support_violation_count == 0
    assert result.exact_volume_match
    assert result.cell_volume_numerator == result.hull_volume_numerator
    assert result.exact_predicate_audit is not None
    assert result.exact_predicate_audit.exact_local_delaunay_violation_count == 0
    assert result.rejection_reasons == ()


def test_validator_rejects_local_delaunay_violation() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.125, 0.125, -0.125],
        ]
    )
    response = _response(
        "local_violation",
        points,
        [[0, 1, 2, 3], [0, 1, 2, 4]],
    )

    result = validate_exact_construction_response(
        "local_violation",
        points,
        response,
    )

    assert not result.accepted
    assert "exact_local_delaunay_violations" in result.rejection_reasons


def test_validator_rejects_incomplete_point_coverage() -> None:
    points = _bipyramid_points()
    response = _response(
        "missing_point",
        points,
        [[0, 1, 2, 3]],
    )

    result = validate_exact_construction_response(
        "missing_point",
        points,
        response,
    )

    assert not result.accepted
    assert result.used_point_count == 4
    assert "not_all_input_points_used" in result.rejection_reasons
    assert "boundary_support_violations" in result.rejection_reasons


def test_validator_rejects_response_not_bound_to_request() -> None:
    points = _bipyramid_points()
    response = _response(
        "hash_mismatch",
        points,
        [[0, 1, 2, 3], [0, 1, 2, 4]],
    )
    response["request_sha256"] = "0" * 64

    result = validate_exact_construction_response(
        "hash_mismatch",
        points,
        response,
    )

    assert not result.accepted
    assert not result.protocol_valid
    assert result.rejection_reasons == ("request_sha256_mismatch",)


def test_subprocess_handoff_validates_protocol_fixture() -> None:
    points = _bipyramid_points()
    script = (
        "import hashlib,json,sys;"
        "raw=sys.stdin.read();"
        "request=json.loads(raw);"
        "response={'schema_version':1,"
        "'operation':'delaunay_tetrahedralization_3',"
        "'case_id':request['case_id'],"
        "'coordinate_model':request['coordinate_model'],"
        "'point_count':request['point_count'],"
        "'request_sha256':hashlib.sha256(raw.encode('ascii')).hexdigest(),"
        "'backend':{'name':'protocol-test-fixture','version':'1',"
        "'kernel':'fixture_exact_integer_kernel','exact_construction':True},"
        "'top_simplices':[[0,1,2,3],[0,1,2,4]]};"
        "sys.stdout.write(json.dumps(response,separators=(',',':')))"
    )

    result = run_exact_construction_backend(
        [sys.executable, "-c", script],
        "subprocess_fixture",
        points,
        timeout_seconds=10.0,
    )

    assert result.accepted
    assert result.backend_name == "protocol-test-fixture"
    assert result.response_sha256 is not None


def test_panel_without_backend_is_explicitly_fail_closed() -> None:
    result = evaluate_exact_construction_panel(
        [("bipyramid", _bipyramid_points())],
        evaluation_split="held_out",
        backend_command=None,
    )
    payload = result.to_dict()

    assert not payload["backend_requested"]
    assert payload["accepted_case_count"] == 0
    assert not payload["backend_handoff_validated"]
    assert not payload["exact_construction_applied_to_benchmark"]
    assert not payload["changes_benchmark_selection"]
    assert not payload["promotion_supported"]
    assert payload["blocking_reasons"] == ["no_exact_construction_backend"]
