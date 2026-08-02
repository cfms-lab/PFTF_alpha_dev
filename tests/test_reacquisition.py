import numpy as np
import pytest

from pftf_alpha.adaptive import (
    boundary_bridge_localization,
    pca_anisotropic_filtration,
)
from pftf_alpha.filtration import AlphaFiltration
from pftf_alpha.reacquisition import (
    ReacquisitionConfig,
    ReacquisitionPolicy,
    evaluate_risk_targeted_reacquisition,
    select_reacquisition_indices,
)
from pftf_alpha.synthetic import PanelSplit, SyntheticFamily, make_synthetic_case


def test_uniform_selection_is_deterministic_and_budget_exact() -> None:
    candidates = np.random.default_rng(3).normal(size=(40, 3))
    first, anchors, fallback = select_reacquisition_indices(
        candidates,
        budget=9,
        policy=ReacquisitionPolicy.UNIFORM,
        seed=17,
    )
    second, _, _ = select_reacquisition_indices(
        candidates,
        budget=9,
        policy=ReacquisitionPolicy.UNIFORM,
        seed=17,
    )
    np.testing.assert_array_equal(first, second)
    assert len(first) == 9
    assert len(np.unique(first)) == 9
    assert anchors == 0
    assert fallback is False


def test_risk_targeted_selection_uses_observed_geometry_only() -> None:
    case = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.HELD_OUT,
        point_count=32,
        reference_count=128,
        seed=11,
    )
    filtration = AlphaFiltration.from_points(case.points)
    adaptive = pca_anisotropic_filtration(
        filtration,
        k_neighbors=8,
        max_normal_penalty=4.0,
    )
    localization = boundary_bridge_localization(
        adaptive,
        scale_multiplier=2.80293354289327,
        k_neighbors=8,
    )
    selected, anchor_count, fallback = select_reacquisition_indices(
        case.reference_points,
        budget=12,
        policy=ReacquisitionPolicy.RISK_TARGETED,
        seed=23,
        observed_points=case.points,
        localization=localization,
    )
    assert len(selected) == 12
    assert len(np.unique(selected)) == 12
    assert anchor_count > 0
    assert fallback is False


def test_phase0_smoke_is_paired_and_serializable() -> None:
    result = evaluate_risk_targeted_reacquisition(
        ReacquisitionConfig(
            base_point_count=32,
            evaluation_reference_count=128,
            candidate_pool_count=128,
            added_point_counts=(8,),
            repeats=1,
            surface_sample_count=64,
            k_neighbors=8,
            seed=29,
        )
    )
    payload = result.to_dict()
    assert payload["artifact_schema"].endswith("/v1")
    assert len(result.trials) == 2
    assert {trial.policy for trial in result.trials} == set(ReacquisitionPolicy)
    assert len(result.comparisons) == 1
    assert result.comparisons[0].repeat_count == 1
    assert "component labels" in result.policy_information_boundary


def test_config_rejects_candidate_pool_smaller_than_budget() -> None:
    with pytest.raises(ValueError):
        ReacquisitionConfig(candidate_pool_count=7, added_point_counts=(8,))
