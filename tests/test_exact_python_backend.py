import sys

import numpy as np
import pytest
import scipy.spatial

from pftf_alpha.exact_backend import (
    exact_construction_request,
    run_exact_construction_backend,
    validate_exact_construction_response,
)
from pftf_alpha.exact_python_backend import (
    BACKEND_KERNEL,
    BACKEND_NAME,
    ExactPythonBackendError,
    decode_exact_request,
    exact_backend_response,
    exact_delaunay_tetrahedra,
)


def _bipyramid_integer_points() -> tuple[tuple[int, int, int], ...]:
    return (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )


def _bipyramid_float_points() -> np.ndarray:
    return np.asarray(_bipyramid_integer_points(), dtype=np.float64)


def test_exact_enumeration_constructs_bipyramid_without_qhull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_qhull(*args, **kwargs):
        raise AssertionError(f"Qhull must not be called: {args!r} {kwargs!r}")

    monkeypatch.setattr(scipy.spatial, "Delaunay", forbidden_qhull)

    result = exact_delaunay_tetrahedra(_bipyramid_integer_points())

    assert result.candidate_cell_count == 5
    assert result.degenerate_candidate_count == 2
    assert result.exact_power_test_count > 0
    assert result.top_simplices == ((0, 1, 2, 3), (0, 1, 2, 4))


def test_exact_enumeration_fails_closed_on_empty_cosphere_ambiguity() -> None:
    cospherical = (
        (0, 0, 0),
        (2, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (2, 2, 2),
    )

    with pytest.raises(
        ExactPythonBackendError,
        match="exact_empty_cospherical_ambiguity_not_supported",
    ):
        exact_delaunay_tetrahedra(cospherical)


def test_nonempty_cospherical_candidate_does_not_trigger_ambiguity() -> None:
    points = (
        (5, 0, 0),
        (0, 5, 0),
        (0, 0, 5),
        (-3, -4, 0),
        (0, -3, -4),
        (0, 0, 0),
    )

    result = exact_delaunay_tetrahedra(points)

    assert result.candidate_cell_count == 15
    assert len(result.top_simplices) == 6
    assert all(5 in cell for cell in result.top_simplices)


def test_exact_enumeration_enforces_small_panel_limit() -> None:
    points = tuple((index, index * index, index * index * index) for index in range(65))

    with pytest.raises(
        ExactPythonBackendError,
        match="point_count_exceeds_exact_backend_limit",
    ):
        exact_delaunay_tetrahedra(points)


def test_protocol_response_is_accepted_by_independent_host_validation() -> None:
    points = _bipyramid_float_points()
    request, _ = exact_construction_request("exact_bipyramid", points)

    response = exact_backend_response(request)
    result = validate_exact_construction_response(
        "exact_bipyramid",
        points,
        response,
    )

    assert response["backend"]["name"] == BACKEND_NAME
    assert response["backend"]["kernel"] == BACKEND_KERNEL
    assert response["construction_diagnostics"]["candidate_cell_count"] == 5
    assert result.accepted
    assert result.validated_top_simplices == ((0, 1, 2, 3), (0, 1, 2, 4))


def test_request_decoder_rejects_non_power_of_two_denominator() -> None:
    points = _bipyramid_float_points()
    request, _ = exact_construction_request("bad_ratio", points)
    request["points"][0][0][1] = "3"

    with pytest.raises(
        ExactPythonBackendError,
        match="coordinate_denominator_not_positive_power_of_two",
    ):
        decode_exact_request(request)


def test_module_subprocess_satisfies_backend_protocol() -> None:
    result = run_exact_construction_backend(
        [sys.executable, "-m", "pftf_alpha.exact_python_backend"],
        "subprocess_exact_bipyramid",
        _bipyramid_float_points(),
        timeout_seconds=10.0,
    )

    assert result.accepted
    assert result.backend_name == BACKEND_NAME
    assert result.backend_kernel == BACKEND_KERNEL
    assert result.backend_attested_exact_construction
