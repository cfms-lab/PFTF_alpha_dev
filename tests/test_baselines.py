import math

import numpy as np

from pftf_alpha.baselines import (
    BaselineID,
    BenchmarkConfig,
    _plateau_persistence,
    run_case_benchmarks,
)
from pftf_alpha.synthetic import SyntheticFamily, make_synthetic_case


def test_b0_p2_runner_preserves_selection_information_boundary() -> None:
    case = make_synthetic_case(
        SyntheticFamily.DISCONNECTED_PARTS,
        point_count=32,
        reference_count=128,
        seed=91,
    )
    config = BenchmarkConfig(
        surface_sample_count=48,
        resample_repeats=1,
        b3_candidate_budget=5,
        seed=92,
    )
    report = run_case_benchmarks(case, config=config)

    assert [result.method for result in report.results] == list(BaselineID)
    results = {result.method: result for result in report.results}
    assert results[BaselineID.B0_CONVEX_HULL].alpha_squared is None
    assert results[BaselineID.B1_FIXED_ALPHA].candidate_count == 1
    assert results[BaselineID.B2_CRITICAL_ORACLE].candidate_count > 1
    assert (
        results[BaselineID.B2_CRITICAL_ORACLE].candidate_count
        == results[BaselineID.B2_CRITICAL_ORACLE].total_candidates_scanned
    )
    assert results[BaselineID.B2_CRITICAL_ORACLE].uses_reference_for_selection
    assert not results[BaselineID.B3_PERSISTENCE_STABILITY].uses_reference_for_selection
    assert results[BaselineID.B3_PERSISTENCE_STABILITY].candidate_count <= 5
    assert (
        results[BaselineID.B3_PERSISTENCE_STABILITY].total_candidates_scanned
        >= results[BaselineID.B3_PERSISTENCE_STABILITY].candidate_count
    )
    assert (
        results[BaselineID.B3_PERSISTENCE_STABILITY].alpha_radius_fraction is not None
    )
    for method in (
        BaselineID.B4_DENSITY_SCALED,
        BaselineID.B5_PCA_ANISOTROPIC,
        BaselineID.P1_PFTF_LOCAL_SPD,
        BaselineID.P2_CONFIDENCE_FALLBACK,
    ):
        adaptive = results[method]
        assert adaptive.uses_reference_for_selection
        assert adaptive.alpha_squared is None
        assert adaptive.selection_parameter_name == "local_scale_multiplier"
        assert adaptive.selection_parameter_value is not None
        assert adaptive.candidate_count > 1
        if method is BaselineID.P1_PFTF_LOCAL_SPD:
            assert adaptive.method_diagnostics is not None
            assert adaptive.method_diagnostics["metric_condition_max"] <= 9.0

        if method is BaselineID.P2_CONFIDENCE_FALLBACK:
            assert adaptive.method_diagnostics is not None
            assert adaptive.method_diagnostics["confidence_threshold"] == 0.5
            assert adaptive.method_diagnostics["fallback_guard_violation_count"] == 0
            assert (
                0.0 <= adaptive.method_diagnostics["selected_fallback_fraction"] <= 1.0
            )
    for result in report.results:
        assert result.runtime_seconds >= 0.0
        assert math.isfinite(result.endpoints.chamfer_squared)
        assert math.isfinite(result.endpoints.hausdorff)
        assert 0.0 <= result.endpoints.fscore <= 1.0


def test_runner_can_execute_b0_without_building_alpha_filtration() -> None:
    case = make_synthetic_case(
        SyntheticFamily.TORUS,
        point_count=24,
        reference_count=48,
        seed=3,
    )
    report = run_case_benchmarks(
        case,
        config=BenchmarkConfig(surface_sample_count=24),
        methods=[BaselineID.B0_CONVEX_HULL],
    )

    assert len(report.results) == 1
    assert report.results[0].method is BaselineID.B0_CONVEX_HULL


def test_frozen_local_multipliers_do_not_use_reference_for_selection() -> None:
    case = make_synthetic_case(
        SyntheticFamily.TORUS,
        point_count=24,
        reference_count=48,
        seed=15,
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        b4_scale_multiplier=1.5,
        b5_scale_multiplier=2.0,
        p1_scale_multiplier=2.5,
        p2_scale_multiplier=3.0,
    )
    report = run_case_benchmarks(
        case,
        config=config,
        methods=[
            BaselineID.B4_DENSITY_SCALED,
            BaselineID.B5_PCA_ANISOTROPIC,
            BaselineID.P1_PFTF_LOCAL_SPD,
            BaselineID.P2_CONFIDENCE_FALLBACK,
        ],
    )

    assert len(report.results) == 4
    for result in report.results:
        assert not result.uses_reference_for_selection
        assert result.selection_mode == "frozen_local_scale_multiplier"
        assert result.candidate_count == 1
        assert result.selection_parameter_value in (1.5, 2.0, 2.5, 3.0)


def test_terminal_convex_hull_plateau_is_not_rewarded() -> None:
    candidates = np.array([1.0, 4.0, 16.0, 64.0])
    signatures = [(2, 4), (1, 2), (1, 2), (1, 2)]

    persistence = _plateau_persistence(candidates, signatures)

    np.testing.assert_allclose(persistence, [1.0, 0.0, 0.0, 0.0])
