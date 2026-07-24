import json
import sys
from pathlib import Path

import numpy as np

from pftf_alpha.baselines import BaselineID, BenchmarkConfig
from pftf_alpha.filtration import AlphaFiltration
from pftf_alpha.g4_fallback import (
    EXACT_PROVENANCE,
    FALLBACK_PROVENANCE,
    G4CaseRouting,
    evaluate_g4_deployment_panel,
    route_case_filtration,
    run_g4_routed_case,
)
from pftf_alpha.synthetic import PanelSplit, SyntheticFamily, make_synthetic_case


def _small_case(point_count: int = 24):
    return make_synthetic_case(
        SyntheticFamily.TORUS,
        split=PanelSplit.CALIBRATION,
        point_count=point_count,
        reference_count=256,
        seed=7,
    )


def test_small_general_position_case_is_exact_certified_and_deployed() -> None:
    case = _small_case(24)
    filtration, routing = route_case_filtration("torus:small", case.points)

    assert routing.is_exact_certified
    assert routing.provenance == EXACT_PROVENANCE
    assert routing.failure_reason is None
    assert routing.exact_backend_requested
    assert routing.top_simplex_count == filtration.top_simplices.shape[0] > 0
    # The deployed connectivity is what the benchmark scores.
    report = run_g4_routed_case(
        case,
        config=BenchmarkConfig(seed=1),
        methods=[BaselineID.B4_DENSITY_SCALED, BaselineID.P2_CONFIDENCE_FALLBACK],
    )[0]
    assert len(report.results) == 2


def test_over_cap_fails_closed_to_floating() -> None:
    case = _small_case(96)  # > MAX_EXACT_POINT_COUNT
    filtration, routing = route_case_filtration("torus:big", case.points)

    assert not routing.is_exact_certified
    assert routing.provenance == FALLBACK_PROVENANCE
    assert routing.failure_reason == "point_count_exceeds_exact_backend_limit"
    assert not routing.exact_backend_requested
    # Fallback is the ordinary floating Qhull construction.
    qhull = AlphaFiltration.from_points(case.points)
    np.testing.assert_array_equal(filtration.top_simplices, qhull.top_simplices)


def test_cospherical_configuration_fails_closed() -> None:
    # The eight cube corners are exactly cospherical.
    cube = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.float64,
    )
    _, routing = route_case_filtration("cube", cube)

    assert not routing.is_exact_certified
    assert routing.failure_reason == "exact_empty_cospherical_ambiguity_not_supported"
    assert routing.exact_backend_requested


def test_duplicate_points_rejected_at_input() -> None:
    import pytest

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # Duplicates are refused by the shared input contract before any routing;
    # neither the exact backend nor the floating fallback can build from them,
    # so the whole call refuses rather than emitting an uncertified result.
    with pytest.raises(ValueError, match="duplicate"):
        route_case_filtration("dup", points)


def test_malformed_external_backend_fails_closed() -> None:
    case = _small_case(24)
    command = (sys.executable, "-c", "print('not-json')")
    filtration, routing = route_case_filtration(
        "torus:ext", case.points, backend_command=command
    )

    assert not routing.is_exact_certified
    assert routing.provenance == FALLBACK_PROVENANCE
    assert routing.failure_reason
    assert routing.exact_backend_requested
    qhull = AlphaFiltration.from_points(case.points)
    np.testing.assert_array_equal(filtration.top_simplices, qhull.top_simplices)


def test_routing_dataclass_rejects_inconsistent_certification() -> None:
    import pytest

    with pytest.raises(ValueError):
        G4CaseRouting(
            case_id="x", point_count=4, is_exact_certified=True,
            exact_backend_requested=True, provenance=FALLBACK_PROVENANCE,
            failure_reason=None, top_simplex_count=1,
        )
    with pytest.raises(ValueError):
        G4CaseRouting(
            case_id="x", point_count=4, is_exact_certified=False,
            exact_backend_requested=True, provenance=FALLBACK_PROVENANCE,
            failure_reason=None, top_simplex_count=1,
        )


def test_deployment_panel_artifact_and_invariant() -> None:
    cases = tuple(
        make_synthetic_case(
            family,
            split=PanelSplit.CALIBRATION,
            point_count=24,
            reference_count=256,
            seed=3,
        )
        for family in (SyntheticFamily.TORUS, SyntheticFamily.U_CONCAVITY)
    )
    result = evaluate_g4_deployment_panel(cases, config=BenchmarkConfig(seed=2))
    payload = result.to_dict()

    assert payload["artifact_schema"] == "pftf_alpha_g4_fail_closed/v1"
    assert payload["changes_benchmark_selection"] is True
    assert payload["promotion_supported"] is False
    assert payload["no_uncertified_result_labeled_exact"] is True
    # Every routing is internally consistent: certified xor recorded reason.
    for routing in result.routings:
        if routing.is_exact_certified:
            assert routing.provenance == EXACT_PROVENANCE
            assert routing.failure_reason is None
        else:
            assert routing.failure_reason


def test_routing_is_deterministic() -> None:
    case = _small_case(28)
    first = route_case_filtration("torus:det", case.points)[1]
    second = route_case_filtration("torus:det", case.points)[1]
    assert first.to_dict() == second.to_dict()


def test_cli_writes_artifact(tmp_path: Path) -> None:
    from pftf_alpha.g4_fallback import main

    output = tmp_path / "g4.json"
    exit_code = main(
        ["--point-count", "24", "--reference-count", "256", "--seed", "5",
         "--split", "calibration", "--output", str(output)]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_schema"] == "pftf_alpha_g4_fail_closed/v1"
    assert payload["case_count"] == 6
