import pytest

from pftf_alpha.conservative_influence_calibration import (
    InfluenceFeatureCohort,
    calibrate_dual_cohort_rectangle,
    evaluate_conservative_influence_calibration,
)
from pftf_alpha.local_insertion_influence import LocalInsertionInfluenceConfig
from pftf_alpha.sensor_stress import SensorStress


def test_dual_cohort_rectangle_balances_zero_harm_retention() -> None:
    cohort_a = InfluenceFeatureCohort(
        harmful=((5.0, 1.0), (2.0, 5.0)),
        focus_safe=((4.0, 0.5), (1.5, 4.0), (1.0, 1.0), (4.0, 4.0)),
        all_safe=((4.0, 0.5), (1.5, 4.0), (1.0, 1.0), (4.0, 4.0)),
    )
    cohort_b = InfluenceFeatureCohort(
        harmful=((4.5, 1.2), (2.2, 4.5)),
        focus_safe=((4.0, 1.0), (2.0, 4.0), (1.0, 1.0), (3.0, 3.0)),
        all_safe=((4.0, 1.0), (2.0, 4.0), (1.0, 1.0), (3.0, 3.0)),
    )
    rectangle = calibrate_dual_cohort_rectangle(cohort_a, cohort_b)
    assert rectangle is not None
    assert rectangle.retained_focus_safe_count == 8
    for cohort in (cohort_a, cohort_b):
        for peak, support in cohort.harmful:
            assert not (
                peak <= rectangle.peak_threshold
                and support <= rectangle.support_threshold
            )


def test_dual_cohort_calibration_requires_harm_in_each_cohort() -> None:
    populated = InfluenceFeatureCohort(
        harmful=((2.0, 2.0),),
        focus_safe=((1.0, 1.0),),
        all_safe=((1.0, 1.0),),
    )
    empty = InfluenceFeatureCohort(
        harmful=(),
        focus_safe=((1.0, 1.0),),
        all_safe=((1.0, 1.0),),
    )
    assert calibrate_dual_cohort_rectangle(populated, empty) is None


def test_phase13_reduced_calibration_cannot_open_final_held_out() -> None:
    result = evaluate_conservative_influence_calibration(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_a_seed=59,
        calibration_b_seed=61,
        final_held_out_seed=67,
        surface_sample_count=64,
        influence_config=LocalInsertionInfluenceConfig(neighbor_counts=(8, 12)),
    )
    assert result.calibration_a.case_count == 1
    assert result.calibration_b.case_count == 1
    assert result.calibration_a.full_protocol is False
    assert result.calibration_b.full_protocol is False
    assert result.final_held_out is None
    assert result.phase13_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.real_scan_supported is False
    assert result.deployment_supported is False


def test_phase13_rejects_reused_seeds() -> None:
    with pytest.raises(ValueError, match="must differ"):
        evaluate_conservative_influence_calibration(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            calibration_a_seed=71,
            calibration_b_seed=71,
            final_held_out_seed=73,
            surface_sample_count=64,
        )
