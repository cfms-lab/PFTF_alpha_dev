from typing import cast

import numpy as np
import pytest

from pftf_alpha.cycle_gated_assignment import (
    AssignmentCycle,
    CycleCandidateCase,
    assignment_cycles,
)
from pftf_alpha.global_tangential_assignment import global_tangential_assignment
from pftf_alpha.matched_pair_stress import DEFAULT_STRESS_SPECS
from pftf_alpha.multivariate_cycle_signature import (
    FORBIDDEN_PRIOR_SEEDS,
    SIGNATURE_FEATURE_NAMES,
    MultivariateCycleSignature,
    cycle_signature,
    evaluate_multivariate_cycle_signature,
    fit_cycle_signature_model,
    score_cycle_signature,
)
from pftf_alpha.sensor_stress import SensorStress
from pftf_alpha.tangential_pair_confidence import TangentialPairRawCase


def _signature(
    value: float,
    *,
    strictly_correcting: bool,
) -> MultivariateCycleSignature:
    cycle = AssignmentCycle(
        rows=(0, 1),
        relative_gain=max(value, 0.0),
        truth_correct_before=0 if strictly_correcting else 2,
        truth_correct_after=2 if strictly_correcting else 1,
    )
    return MultivariateCycleSignature(
        cycle=cycle,
        values=(value,) * len(SIGNATURE_FEATURE_NAMES),
    )


def test_cycle_signature_is_finite_and_strictly_correcting_for_swap() -> None:
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
    candidate = CycleCandidateCase(
        raw=cast(TangentialPairRawCase, None),
        global_assignment=assignment,
        cycles=cycles,
    )

    signature = cycle_signature(candidate, cycles[0])

    assert len(signature.values) == len(SIGNATURE_FEATURE_NAMES)
    assert np.all(np.isfinite(signature.values))
    assert signature.strictly_correcting is True


def test_ridge_cutoff_rejects_every_unsafe_training_cycle() -> None:
    unsafe = (_signature(-2.0, strictly_correcting=False),)
    safe = (
        _signature(1.0, strictly_correcting=True),
        _signature(2.0, strictly_correcting=True),
    )

    model = fit_cycle_signature_model(unsafe + safe)

    assert model.calibration_valid is True
    assert model.accepted_training_unsafe_cycle_count == 0
    assert model.accepted_training_strictly_correcting_cycle_count > 0
    assert score_cycle_signature(model, unsafe[0]) < model.cutoff
    assert any(score_cycle_signature(model, row) >= model.cutoff for row in safe)


def test_phase22_reduced_panel_cannot_open_development_or_validation() -> None:
    result = evaluate_multivariate_cycle_signature(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        training_a_seed=263,
        development_b_seed=269,
        validation_a_seed=271,
        validation_b_seed=277,
        final_held_out_seed=281,
        surface_sample_count=64,
    )

    assert result.training_a.case_count == len(DEFAULT_STRESS_SPECS)
    assert result.training_a.full_protocol is False
    assert result.training_a.panel_gate_passed is False
    assert result.development_b is None
    assert result.development_screen_passed is False
    assert result.validation_a is None
    assert result.validation_b is None
    assert result.final_held_out is None
    assert result.phase22_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase22_rejects_all_prior_opened_or_reserved_seeds() -> None:
    forbidden = max(FORBIDDEN_PRIOR_SEEDS)
    with pytest.raises(ValueError, match="must not be reused"):
        evaluate_multivariate_cycle_signature(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            training_a_seed=263,
            development_b_seed=269,
            validation_a_seed=271,
            validation_b_seed=277,
            final_held_out_seed=forbidden,
            surface_sample_count=64,
        )
