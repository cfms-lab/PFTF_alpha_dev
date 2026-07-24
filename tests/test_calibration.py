import math
from dataclasses import replace

import numpy as np
import pytest

from pftf_alpha.baselines import BaselineID, BenchmarkConfig, run_case_benchmarks
from pftf_alpha.calibration import (
    calibrate_adaptive_multiplier,
    calibrate_p2_confidence_threshold,
    evaluate_boundary_bridge_localization,
    evaluate_boundary_owner_intervention,
    evaluate_boundary_region_cut_ablation,
    evaluate_bridge_penalty_ablation,
)
from pftf_alpha.synthetic import (
    PanelSplit,
    SyntheticFamily,
    make_minimal_panel,
    make_synthetic_case,
)


@pytest.mark.parametrize(
    "method,field_name",
    [
        (BaselineID.B4_DENSITY_SCALED, "b4_scale_multiplier"),
        (BaselineID.B5_PCA_ANISOTROPIC, "b5_scale_multiplier"),
        (BaselineID.P1_PFTF_LOCAL_SPD, "p1_scale_multiplier"),
        (BaselineID.P2_CONFIDENCE_FALLBACK, "p2_scale_multiplier"),
    ],
)
def test_adaptive_calibration_freezes_one_panel_wide_multiplier(
    method: BaselineID,
    field_name: str,
) -> None:
    calibration_cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=731,
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        seed=732,
    )

    calibration = calibrate_adaptive_multiplier(
        calibration_cases,
        method,
        config=config,
        candidate_budget=4,
    )

    assert calibration.method is method
    assert calibration.calibration_case_count == len(calibration_cases)
    assert calibration.candidate_count >= 4
    assert calibration.candidate_min <= calibration.multiplier
    assert calibration.multiplier <= calibration.candidate_max
    assert math.isfinite(calibration.selected_mean_objective)

    frozen_config = replace(config, **{field_name: calibration.multiplier})
    held_out_case = make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=24,
        reference_count=48,
        seed=731,
    )[0]
    result = run_case_benchmarks(
        held_out_case,
        config=frozen_config,
        methods=[method],
    ).results[0]

    assert not result.uses_reference_for_selection
    assert result.selection_mode == "frozen_local_scale_multiplier"
    assert result.selection_parameter_value == calibration.multiplier


def test_adaptive_calibration_ignores_evaluation_only_component_labels() -> None:
    case = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=735,
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        seed=736,
    )

    original = calibrate_adaptive_multiplier(
        [case],
        BaselineID.B4_DENSITY_SCALED,
        config=config,
        candidate_budget=4,
    )
    relabeled_case = replace(
        case,
        point_component_labels=np.zeros_like(case.point_component_labels),
    )
    relabeled = calibrate_adaptive_multiplier(
        [relabeled_case],
        BaselineID.B4_DENSITY_SCALED,
        config=config,
        candidate_budget=4,
    )

    assert set(case.point_component_labels) == {0, 1}
    assert set(relabeled_case.point_component_labels) == {0}
    assert original == relabeled


def test_adaptive_calibration_rejects_nonadaptive_method() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=740,
    )

    with pytest.raises(ValueError, match="only B4, B5, P1, or P2"):
        calibrate_adaptive_multiplier(
            cases,
            BaselineID.B3_PERSISTENCE_STABILITY,
            config=BenchmarkConfig(surface_sample_count=24),
            candidate_budget=4,
        )


def test_p2_confidence_calibration_is_reference_free_and_deterministic() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=750,
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        seed=751,
    )

    first = calibrate_p2_confidence_threshold(
        cases,
        config=config,
        target_fallback_fraction=0.25,
    )
    altered_evaluation_data = tuple(
        replace(
            case,
            reference_points=np.zeros_like(case.reference_points),
            point_component_labels=np.zeros_like(case.point_component_labels),
        )
        for case in cases
    )
    second = calibrate_p2_confidence_threshold(
        altered_evaluation_data,
        config=config,
        target_fallback_fraction=0.25,
    )

    assert first == second
    assert not first.uses_reference_for_selection
    assert first.calibration_case_count == len(cases)
    assert 0.0 <= first.threshold <= 1.0
    assert first.achieved_fallback_fraction == pytest.approx(
        first.fallback_count / first.cell_count
    )
    assert first.per_case_fallback_min <= first.per_case_fallback_median
    assert first.per_case_fallback_median <= first.per_case_fallback_max


def test_p2_confidence_calibration_rejects_boundary_target() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=16,
        reference_count=24,
        seed=760,
    )
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        calibrate_p2_confidence_threshold(
            cases,
            config=BenchmarkConfig(),
            target_fallback_fraction=0.0,
        )


def test_bridge_penalty_ablation_is_evaluation_only_and_does_not_select() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=770,
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        p2_scale_multiplier=1.2,
        p2_confidence_threshold=0.3,
        seed=771,
    )

    result = evaluate_bridge_penalty_ablation(
        cases,
        config=config,
        strengths=[0.0, 0.2, 0.8],
    )

    assert result.base_method is BaselineID.P2_CONFIDENCE_FALLBACK
    assert result.role == "evaluation_only_no_selection"
    assert result.uses_reference_geometry_for_evaluation
    assert result.uses_component_labels_for_evaluation
    assert not result.changes_benchmark_selection
    assert result.candidate_count == 3
    assert [point.strength for point in result.curve] == [0.0, 0.2, 0.8]
    assert result.curve[0].selected_cell_count >= result.curve[-1].selected_cell_count
    assert not result.curve[0].promotion_gate_passed


def test_bridge_penalty_ablation_labels_affect_only_evaluation_fields() -> None:
    case = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=772,
    )
    relabeled = replace(
        case,
        point_component_labels=np.zeros_like(case.point_component_labels),
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        p2_scale_multiplier=1.2,
        p2_confidence_threshold=0.3,
        seed=773,
    )

    original = evaluate_bridge_penalty_ablation(
        [case],
        config=config,
        strengths=[0.0, 0.2],
    )
    changed = evaluate_bridge_penalty_ablation(
        [relabeled],
        config=config,
        strengths=[0.0, 0.2],
    )

    for first, second in zip(original.curve, changed.curve, strict=True):
        assert first.strength == second.strength
        assert first.mean_objective == second.mean_objective
        assert first.mean_geometry == second.mean_geometry
        assert first.mean_topology == second.mean_topology
        assert first.mean_complexity == second.mean_complexity
        assert first.component_error_sum == second.component_error_sum
        assert first.betti_error_sum == second.betti_error_sum
        assert first.selected_cell_count == second.selected_cell_count
        assert first.selected_flagged_cell_count == second.selected_flagged_cell_count
        assert first.selected_mean_risk == second.selected_mean_risk
        assert second.labeled_false_bridge_edges == 0
        assert second.labeled_false_bridge_faces == 0


def test_bridge_penalty_ablation_requires_frozen_p2_multiplier() -> None:
    case = make_synthetic_case(
        SyntheticFamily.TORUS,
        point_count=16,
        reference_count=24,
        seed=774,
    )

    with pytest.raises(ValueError, match="frozen P2 multiplier"):
        evaluate_bridge_penalty_ablation(
            [case],
            config=BenchmarkConfig(surface_sample_count=16),
            strengths=[0.0, 0.2],
        )


def test_boundary_bridge_localization_is_evaluation_only() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=24,
        reference_count=48,
        seed=780,
    )
    config = BenchmarkConfig(
        adaptive_k_neighbors=6,
        p2_scale_multiplier=1.2,
        p2_confidence_threshold=0.3,
        seed=781,
    )

    result = evaluate_boundary_bridge_localization(cases, config=config)

    assert result.base_method is BaselineID.P2_CONFIDENCE_FALLBACK
    assert result.role == "evaluation_only_no_selection"
    assert result.evaluation_split == PanelSplit.HELD_OUT.value
    assert not result.uses_reference_geometry
    assert result.uses_component_labels_for_evaluation
    assert not result.changes_benchmark_selection
    assert result.risk_threshold == 1.0
    assert result.case_count == len(cases)
    assert result.pooled_boundary_face_count == sum(
        case.boundary_face_count for case in result.cases
    )
    assert result.pooled_boundary_edge_count == sum(
        case.boundary_edge_count for case in result.cases
    )
    assert result.pooled_flagged_face_count == sum(
        case.flagged_face_count for case in result.cases
    )


def test_boundary_bridge_localization_labels_affect_only_evaluation() -> None:
    case = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.HELD_OUT,
        point_count=24,
        reference_count=48,
        seed=782,
    )
    relabeled = replace(
        case,
        point_component_labels=np.zeros_like(case.point_component_labels),
    )
    config = BenchmarkConfig(
        adaptive_k_neighbors=6,
        p2_scale_multiplier=1.2,
        p2_confidence_threshold=0.3,
        seed=783,
    )

    original = evaluate_boundary_bridge_localization([case], config=config)
    changed = evaluate_boundary_bridge_localization([relabeled], config=config)
    first = original.cases[0]
    second = changed.cases[0]

    for field in (
        "route",
        "normal_coherence",
        "selected_cell_count",
        "selected_dual_component_count",
        "selected_dual_edge_count",
        "selected_dual_bridge_edge_count",
        "selected_dual_articulation_cell_count",
        "boundary_face_count",
        "flagged_face_count",
        "boundary_edge_count",
        "flagged_edge_count",
        "dual_bottleneck_face_count",
    ):
        assert getattr(first, field) == getattr(second, field)
    assert second.labeled_mixed_face_count == 0
    assert second.labeled_mixed_edge_count == 0
    assert second.face_auc is None
    assert second.edge_auc is None


def test_boundary_bridge_localization_requires_frozen_p2_multiplier() -> None:
    case = make_synthetic_case(
        SyntheticFamily.TORUS,
        point_count=16,
        reference_count=24,
        seed=784,
    )

    with pytest.raises(ValueError, match="frozen P2 multiplier"):
        evaluate_boundary_bridge_localization(
            [case],
            config=BenchmarkConfig(adaptive_k_neighbors=6),
        )


def test_boundary_owner_intervention_is_calibration_only_and_does_not_select() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=790,
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        p2_scale_multiplier=1.2,
        p2_confidence_threshold=0.3,
        seed=791,
    )

    result = evaluate_boundary_owner_intervention(
        cases,
        config=config,
        rounds=[0, 1, 2],
    )

    assert result.base_method is BaselineID.P2_CONFIDENCE_FALLBACK
    assert result.role == "calibration_only_evaluation_no_selection"
    assert result.uses_reference_geometry_for_evaluation
    assert result.uses_component_labels_for_evaluation
    assert not result.changes_benchmark_selection
    assert result.recomputes_boundary_each_round
    assert result.candidate_count == 3
    assert [point.rounds for point in result.curve] == [0, 1, 2]
    baseline = result.curve[0]
    assert baseline.removed_cell_count == 0
    assert baseline.initial_selected_cell_count == baseline.final_selected_cell_count
    assert baseline.boundary_recomputation_count == len(cases)
    assert not baseline.promotion_gate_passed


def test_boundary_owner_intervention_labels_affect_only_evaluation_fields() -> None:
    case = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=792,
    )
    relabeled = replace(
        case,
        point_component_labels=np.zeros_like(case.point_component_labels),
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        p2_scale_multiplier=1.2,
        p2_confidence_threshold=0.3,
        seed=793,
    )

    original = evaluate_boundary_owner_intervention(
        [case],
        config=config,
        rounds=[0, 1],
    )
    changed = evaluate_boundary_owner_intervention(
        [relabeled],
        config=config,
        rounds=[0, 1],
    )

    for first, second in zip(original.curve, changed.curve, strict=True):
        for field in (
            "rounds",
            "mean_objective",
            "mean_geometry",
            "mean_topology",
            "mean_complexity",
            "component_error_sum",
            "betti_error_sum",
            "initial_selected_cell_count",
            "final_selected_cell_count",
            "removed_cell_count",
            "removed_fraction",
            "executed_round_count",
            "boundary_recomputation_count",
            "remaining_flagged_face_count",
            "remaining_flagged_edge_count",
        ):
            assert getattr(first, field) == getattr(second, field)
        assert second.labeled_false_bridge_edges == 0
        assert second.labeled_false_bridge_faces == 0


def test_boundary_owner_intervention_requires_frozen_p2_multiplier() -> None:
    case = make_synthetic_case(
        SyntheticFamily.TORUS,
        point_count=16,
        reference_count=24,
        seed=794,
    )

    with pytest.raises(ValueError, match="frozen P2 multiplier"):
        evaluate_boundary_owner_intervention(
            [case],
            config=BenchmarkConfig(adaptive_k_neighbors=6),
            rounds=[0, 1],
        )


def test_boundary_region_cut_ablation_is_calibration_only() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=800,
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        p2_scale_multiplier=1.2,
        p2_confidence_threshold=0.3,
        seed=801,
    )

    result = evaluate_boundary_region_cut_ablation(cases, config=config)

    assert result.base_method is BaselineID.P2_CONFIDENCE_FALLBACK
    assert result.role == "calibration_only_evaluation_no_selection"
    assert result.uses_reference_geometry_for_evaluation
    assert result.uses_component_labels_for_evaluation
    assert not result.changes_benchmark_selection
    assert result.requested_strategies == (
        "baseline",
        "largest_risk_region",
        "safe_backbone_cut",
    )
    assert [point.strategy for point in result.curve] == list(
        result.requested_strategies
    )
    baseline, largest, safe_cut = result.curve
    assert baseline.risk_region_count == largest.risk_region_count
    assert baseline.risk_region_count == safe_cut.risk_region_count
    assert baseline.safe_backbone_cut_edge_count == 0
    assert safe_cut.removed_cell_count == 0
    assert safe_cut.initial_selected_cell_count == safe_cut.final_selected_cell_count
    assert not any(point.promotion_gate_passed for point in result.curve)


def test_boundary_region_cut_labels_affect_only_evaluation_fields() -> None:
    case = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=802,
    )
    relabeled = replace(
        case,
        point_component_labels=np.zeros_like(case.point_component_labels),
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        p2_scale_multiplier=1.2,
        p2_confidence_threshold=0.3,
        seed=803,
    )

    original = evaluate_boundary_region_cut_ablation([case], config=config)
    changed = evaluate_boundary_region_cut_ablation([relabeled], config=config)

    for first, second in zip(original.curve, changed.curve, strict=True):
        for field in (
            "strategy",
            "mean_objective",
            "mean_geometry",
            "mean_topology",
            "mean_complexity",
            "component_error_sum",
            "betti_error_sum",
            "risk_region_count",
            "largest_risk_region_face_count",
            "safe_boundary_component_count",
            "safe_backbone_cut_edge_count",
            "candidate_case_count",
            "candidate_face_count",
            "initial_selected_cell_count",
            "final_selected_cell_count",
            "removed_cell_count",
            "removed_fraction",
        ):
            assert getattr(first, field) == getattr(second, field)
        assert second.labeled_false_bridge_edges == 0
        assert second.labeled_false_bridge_faces == 0


def test_boundary_region_cut_requires_frozen_p2_multiplier() -> None:
    case = make_synthetic_case(
        SyntheticFamily.TORUS,
        point_count=16,
        reference_count=24,
        seed=804,
    )

    with pytest.raises(ValueError, match="frozen P2 multiplier"):
        evaluate_boundary_region_cut_ablation(
            [case],
            config=BenchmarkConfig(adaptive_k_neighbors=6),
        )
