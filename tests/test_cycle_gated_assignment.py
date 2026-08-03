from typing import cast

import numpy as np
import pytest

from pftf_alpha.cycle_gated_assignment import (
    PHASE20_FINAL_HELD_OUT_SEED,
    AssignmentCycle,
    CycleCandidateCase,
    assignment_cycles,
    calibrate_cycle_gain_cutoff,
    evaluate_cycle_gated_assignment,
)
from pftf_alpha.global_tangential_assignment import (
    GlobalTangentialAssignment,
    global_tangential_assignment,
)
from pftf_alpha.matched_pair_stress import DEFAULT_STRESS_SPECS
from pftf_alpha.sensor_stress import SensorStress
from pftf_alpha.tangential_pair_confidence import TangentialPairRawCase


def test_assignment_cycles_recovers_a_truth_improving_swap() -> None:
    axis = np.linspace(-1.0, 1.0, 5)
    xx, yy = np.meshgrid(axis, axis)
    primary = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    latent_repeat = primary.copy()
    latent_repeat[0, 2] = 0.75
    presented_repeat = latent_repeat.copy()
    presented_repeat[[1, -1]] = presented_repeat[[-1, 1]]
    repeat_source_ids = np.arange(primary.shape[0])
    repeat_source_ids[[1, -1]] = repeat_source_ids[[-1, 1]]

    assignment = global_tangential_assignment(primary, presented_repeat)
    cycles = assignment_cycles(
        assignment,
        np.arange(primary.shape[0]),
        repeat_source_ids,
    )

    assert len(cycles) == 1
    assert set(cycles[0].rows) == {1, primary.shape[0] - 1}
    assert cycles[0].relative_gain > 0.0
    assert cycles[0].truth_correct_before == 0
    assert cycles[0].truth_correct_after == 2
    assert cycles[0].truth_improving is True


def test_cycle_cutoff_excludes_all_development_non_improving_cycles() -> None:
    cycles = (
        AssignmentCycle((0, 1), 0.20, 2, 1),
        AssignmentCycle((2, 3), 0.80, 0, 2),
        AssignmentCycle((4, 5), 0.10, 0, 2),
    )
    case = CycleCandidateCase(
        raw=cast(TangentialPairRawCase, None),
        global_assignment=cast(GlobalTangentialAssignment, None),
        cycles=cycles,
    )

    calibration = calibrate_cycle_gain_cutoff((case,))

    assert calibration.cutoff > 0.20
    assert calibration.maximum_non_improving_gain == 0.20
    assert calibration.accepted_non_improving_cycle_count == 0
    assert calibration.accepted_truth_improving_cycle_count == 1
    assert calibration.rejected_truth_improving_cycle_count == 1


def test_phase21_reduced_panel_cannot_open_fresh_seeds() -> None:
    result = evaluate_cycle_gated_assignment(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        development_a_seed=263,
        development_b_seed=269,
        validation_a_seed=271,
        validation_b_seed=277,
        final_held_out_seed=281,
        surface_sample_count=64,
    )

    assert result.development_a.case_count == len(DEFAULT_STRESS_SPECS)
    assert result.development_b.case_count == len(DEFAULT_STRESS_SPECS)
    assert result.development_a.full_protocol is False
    assert result.development_b.full_protocol is False
    assert result.development_screen_passed is False
    assert result.validation_a is None
    assert result.validation_b is None
    assert result.final_held_out is None
    assert result.phase21_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase21_rejects_phase20_unopened_final_seed() -> None:
    with pytest.raises(ValueError, match="must not be reused"):
        evaluate_cycle_gated_assignment(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            development_a_seed=263,
            development_b_seed=269,
            validation_a_seed=271,
            validation_b_seed=277,
            final_held_out_seed=PHASE20_FINAL_HELD_OUT_SEED,
            surface_sample_count=64,
        )
