import math
from dataclasses import replace

import numpy as np

from pftf_alpha.baselines import (
    BaselineID,
    BenchmarkConfig,
    _binary_auc,
    _bridge_risk_probe,
    _plateau_persistence,
    run_case_benchmarks,
)
from pftf_alpha.filtration import AlphaFiltration
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
    probe = report.bridge_risk_probe
    assert probe is not None
    assert probe.selection_role == "evaluation_only"
    assert not probe.uses_reference_or_labels_for_risk
    assert probe.uses_component_labels_for_evaluation
    assert probe.cell_count == (
        probe.labeled_mixed_cell_count + probe.labeled_same_component_cell_count
    )
    assert probe.cell_count == (
        probe.labeled_true_positive_count
        + probe.labeled_false_positive_count
        + probe.labeled_false_negative_count
        + probe.labeled_true_negative_count
    )
    assert probe.labeled_auc is not None
    assert 0.0 <= probe.labeled_auc <= 1.0
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
    assert report.bridge_risk_probe is None


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


def test_binary_auc_uses_average_ranks_for_ties() -> None:
    scores = np.array([0.0, 1.0, 1.0, 2.0])
    positive_mask = np.array([False, False, True, True])

    assert _binary_auc(scores, positive_mask) == 0.875
    assert _binary_auc(scores, np.ones(4, dtype=np.bool_)) is None


def test_terminal_convex_hull_plateau_is_not_rewarded() -> None:
    candidates = np.array([1.0, 4.0, 16.0, 64.0])
    signatures = [(2, 4), (1, 2), (1, 2), (1, 2)]

    persistence = _plateau_persistence(candidates, signatures)

    np.testing.assert_allclose(persistence, [1.0, 0.0, 0.0, 0.0])


def test_bridge_risk_score_summary_does_not_depend_on_component_labels() -> None:
    case = make_synthetic_case(
        SyntheticFamily.DISCONNECTED_PARTS,
        point_count=32,
        reference_count=64,
        seed=403,
    )
    relabeled = replace(
        case,
        point_component_labels=np.arange(case.points.shape[0], dtype=np.int64) % 2,
    )
    filtration = AlphaFiltration.from_points(case.points)
    config = BenchmarkConfig(adaptive_k_neighbors=8)

    original = _bridge_risk_probe(filtration, case, config)
    changed = _bridge_risk_probe(filtration, relabeled, config)

    assert original.route == changed.route
    assert original.normal_coherence == changed.normal_coherence
    assert original.flagged_cell_count == changed.flagged_cell_count
    assert original.flagged_fraction == changed.flagged_fraction
    assert original.risk_min == changed.risk_min
    assert original.risk_median == changed.risk_median
    assert original.risk_max == changed.risk_max
    assert (
        original.labeled_mixed_cell_count,
        original.labeled_auc,
    ) != (
        changed.labeled_mixed_cell_count,
        changed.labeled_auc,
    )
