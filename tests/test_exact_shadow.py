import numpy as np
import pytest
from scipy.spatial import Delaunay

from pftf_alpha.baselines import BaselineID, BenchmarkConfig, run_case_benchmarks
from pftf_alpha.exact_backend import (
    COORDINATE_MODEL,
    OPERATION,
    PROTOCOL_VERSION,
    ExactConstructionPanelResult,
    exact_construction_request,
    validate_exact_construction_response,
)
from pftf_alpha.exact_shadow import evaluate_exact_connectivity_shadow
from pftf_alpha.filtration import AlphaFiltration
from pftf_alpha.synthetic import PanelSplit, make_minimal_panel


def _held_out_case():
    return make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=16,
        reference_count=32,
        seed=919,
    )[0]


def _validated_qhull_result(case):
    case_id = case.family.value
    _, request_sha256 = exact_construction_request(case_id, case.points)
    response = {
        "schema_version": PROTOCOL_VERSION,
        "operation": OPERATION,
        "case_id": case_id,
        "coordinate_model": COORDINATE_MODEL,
        "point_count": len(case.points),
        "request_sha256": request_sha256,
        "backend": {
            "name": "shadow-test-fixture",
            "version": "1",
            "kernel": "fixture_attestation_only",
            "exact_construction": True,
        },
        "top_simplices": Delaunay(case.points).simplices.tolist(),
    }
    result = validate_exact_construction_response(case_id, case.points, response)
    assert result.accepted
    assert result.validated_top_simplices is not None
    return result


def test_from_top_simplices_reproduces_qhull_filtration() -> None:
    random = np.random.default_rng(20260724)
    points = random.normal(size=(16, 3))
    primary = AlphaFiltration.from_points(points)

    reconstructed = AlphaFiltration.from_top_simplices(
        points,
        primary.top_simplices[:, ::-1],
    )

    np.testing.assert_array_equal(
        reconstructed.top_simplices,
        primary.top_simplices[:, ::-1],
    )
    assert [
        (record.vertices, record.alpha_squared, record.is_gabriel)
        for record in reconstructed.records
    ] == [
        (record.vertices, record.alpha_squared, record.is_gabriel)
        for record in primary.records
    ]


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        (
            np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 4.0]]),
            "integer",
        ),
        (np.array([[0, 1, 2, 5], [0, 1, 2, 4]]), "out-of-range"),
        (np.array([[0, 1, 1, 3], [0, 1, 2, 4]]), "repeated"),
        (np.array([[0, 1, 2, 3], [3, 2, 1, 0]]), "duplicate"),
        (np.array([[0, 1, 2, 3]]), "every input point"),
    ],
)
def test_from_top_simplices_rejects_structural_invalidity(
    cells: np.ndarray,
    message: str,
) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )

    with pytest.raises(ValueError, match=message):
        AlphaFiltration.from_top_simplices(points, cells)


def test_validated_connectivity_runs_as_non_deployed_shadow() -> None:
    case = _held_out_case()
    config = BenchmarkConfig(surface_sample_count=24, seed=921)
    methods = (BaselineID.B1_FIXED_ALPHA,)
    primary = run_case_benchmarks(case, config=config, methods=methods)
    construction = ExactConstructionPanelResult(
        evaluation_split=PanelSplit.HELD_OUT.value,
        backend_requested=True,
        cases=(_validated_qhull_result(case),),
    )

    result = evaluate_exact_connectivity_shadow(
        (case,),
        (primary,),
        construction_result=construction,
        config=config,
        methods=methods,
    )
    payload = result.to_dict()

    assert payload["shadow_case_count"] == 1
    assert payload["output_difference_case_count"] == 0
    assert payload["all_accepted_cases_evaluated"]
    assert not payload["primary_benchmark_results_changed"]
    assert payload["selection_effect"] == "none"
    assert not payload["promotion_supported"]
    assert payload["blocking_reasons"] == ["exact_connectivity_shadow_not_deployed"]
    comparison = payload["cases"][0]
    assert comparison["connectivity_matches_primary"]
    assert comparison["all_nonruntime_outputs_match"]
    assert comparison["changed_methods"] == []
    assert comparison["shadow_report"]["results"][0]["method"] == "B1"


def test_missing_backend_leaves_shadow_and_primary_results_unchanged() -> None:
    case = _held_out_case()
    config = BenchmarkConfig(surface_sample_count=24, seed=922)
    methods = (BaselineID.B1_FIXED_ALPHA,)
    primary = run_case_benchmarks(case, config=config, methods=methods)
    primary_before = primary.to_dict()
    construction = ExactConstructionPanelResult(
        evaluation_split=PanelSplit.HELD_OUT.value,
        backend_requested=False,
        cases=(),
    )

    result = evaluate_exact_connectivity_shadow(
        (case,),
        (primary,),
        construction_result=construction,
        config=config,
        methods=methods,
    )
    payload = result.to_dict()

    assert payload["shadow_case_count"] == 0
    assert payload["output_difference_case_count"] == 0
    assert not payload["primary_benchmark_results_changed"]
    assert payload["blocking_reasons"] == [
        "no_exact_construction_backend",
        "exact_connectivity_shadow_not_deployed",
    ]
    assert not payload["cases"][0]["shadow_ran"]
    assert payload["cases"][0]["shadow_report"] is None
    assert primary.to_dict() == primary_before
