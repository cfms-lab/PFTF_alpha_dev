import numpy as np
import pytest

from pftf_alpha.global_tangential_assignment import (
    evaluate_global_tangential_assignment,
    global_tangential_assignment,
)
from pftf_alpha.matched_pair_stress import DEFAULT_STRESS_SPECS
from pftf_alpha.sensor_stress import SensorStress


def test_global_assignment_recovers_swap_while_preserving_depth_excursion() -> None:
    axis = np.linspace(-1.0, 1.0, 5)
    xx, yy = np.meshgrid(axis, axis)
    primary = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    latent_repeat = primary.copy()
    latent_repeat[0, 2] = 0.75
    presented_repeat = latent_repeat.copy()
    presented_repeat[[1, -1]] = presented_repeat[[-1, 1]]

    result = global_tangential_assignment(primary, presented_repeat)

    expected = np.arange(primary.shape[0])
    expected[1] = primary.shape[0] - 1
    expected[-1] = 1
    assert np.array_equal(result.repeat_row_for_primary, expected)
    assert result.repeat_row_for_primary[0] == 0
    assert np.unique(result.repeat_row_for_primary).size == primary.shape[0]
    assert result.presented_normal_costs.shape == (primary.shape[0],)
    assert result.assigned_normal_costs.shape == (primary.shape[0],)
    assert np.all(result.presented_normal_costs >= 0.0)
    assert np.all(result.assigned_normal_costs >= 0.0)


def test_global_assignment_is_deterministic() -> None:
    rng = np.random.default_rng(47)
    primary = rng.normal(size=(32, 3))
    repeat = primary + rng.normal(scale=0.01, size=primary.shape)
    repeat[[3, 17]] = repeat[[17, 3]]

    first = global_tangential_assignment(primary, repeat)
    second = global_tangential_assignment(primary, repeat)

    assert np.array_equal(
        first.repeat_row_for_primary,
        second.repeat_row_for_primary,
    )
    assert np.array_equal(first.assigned_costs, second.assigned_costs)
    assert np.array_equal(
        first.presented_normal_costs,
        second.presented_normal_costs,
    )
    assert np.array_equal(
        first.assigned_normal_costs,
        second.assigned_normal_costs,
    )
    assert first.tie_perturbation_unit == second.tie_perturbation_unit


def test_phase20_reduced_panel_cannot_open_final() -> None:
    result = evaluate_global_tangential_assignment(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_a_seed=263,
        calibration_b_seed=269,
        final_held_out_seed=271,
        surface_sample_count=64,
    )
    assert result.calibration_a.case_count == len(DEFAULT_STRESS_SPECS)
    assert result.calibration_b.case_count == len(DEFAULT_STRESS_SPECS)
    assert result.calibration_a.full_protocol is False
    assert result.calibration_b.full_protocol is False
    assert result.final_held_out is None
    assert result.phase20_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase20_rejects_phase19_unopened_final_seed() -> None:
    with pytest.raises(ValueError, match="must not be reused"):
        evaluate_global_tangential_assignment(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            calibration_a_seed=23200804,
            calibration_b_seed=277,
            final_held_out_seed=281,
            surface_sample_count=64,
        )
