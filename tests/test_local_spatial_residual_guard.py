from __future__ import annotations

import math

import pytest

from pftf_alpha.local_spatial_residual_guard import (
    EXPECTED_LIMITING_LOCAL_VALUE,
    EXPECTED_LOCAL_REJECTION_CUTOFF,
    FINAL_HELD_OUT_SEED,
    VALIDATION_A_SEED,
    VALIDATION_B_SEED,
    audit_phase28_case_seed_disjointness,
    evaluate_local_spatial_residual_guard,
)
from pftf_alpha.sensor_stress import SensorStress


def test_local_cutoff_is_one_float_below_design_residual() -> None:
    assert EXPECTED_LOCAL_REJECTION_CUTOFF == math.nextafter(
        EXPECTED_LIMITING_LOCAL_VALUE,
        -math.inf,
    )


def test_default_phase28_case_seeds_are_disjoint() -> None:
    audit = audit_phase28_case_seed_disjointness(
        VALIDATION_A_SEED,
        VALIDATION_B_SEED,
        FINAL_HELD_OUT_SEED,
    )

    assert audit.passed is True
    assert audit.panel_case_count == 216
    assert audit.validation_a_prior_overlap_count == 0
    assert audit.validation_a_b_overlap_count == 0


def test_phase28_rejects_prior_seed_reuse_before_evaluation() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_local_spatial_residual_guard(
            point_counts=(24,),
            stresses=(SensorStress.CONTROL,),
            reference_count=64,
            repeats=1,
            validation_a_seed=27800804,
            validation_b_seed=VALIDATION_B_SEED,
            final_held_out_seed=FINAL_HELD_OUT_SEED,
            surface_sample_count=32,
            open_fresh=False,
        )


def test_reduced_protocol_does_not_open_fresh_panels() -> None:
    result = evaluate_local_spatial_residual_guard(
        point_counts=(32,),
        stresses=(SensorStress.CONTROL, SensorStress.OUTLIERS_01),
        reference_count=64,
        repeats=1,
        surface_sample_count=32,
        open_fresh=False,
    )

    assert result.design_reproduced is False
    assert result.design_gate_passed is False
    assert result.validation_a is None
    assert result.validation_b is None
    assert result.final_held_out is None
    assert result.phase28_supported is False
