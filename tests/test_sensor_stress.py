import numpy as np

from pftf_alpha.sensor_stress import (
    SensorStress,
    evaluate_sensor_stress,
    make_sensor_stress_case,
)


def test_sensor_generator_marks_outliers_and_occlusion() -> None:
    outlier_case = make_sensor_stress_case(
        SensorStress.OUTLIERS_05,
        96,
        reference_count=128,
        seed=7,
    )
    assert np.sum(outlier_case.point_component_labels == 2) == 5

    occluded = make_sensor_stress_case(
        SensorStress.UPPER_OCCLUSION,
        96,
        reference_count=128,
        seed=7,
    )
    upper = occluded.points[occluded.point_component_labels == 1]
    assert np.min(upper[:, 0]) > -0.25


def test_phase8_smoke_cannot_promote_a_reduced_panel() -> None:
    result = evaluate_sensor_stress(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        seed=29,
        surface_sample_count=64,
    )
    assert result.case_count == 1
    assert result.phase8_supported is False
    assert result.deployment_supported is False
