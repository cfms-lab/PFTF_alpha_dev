from pftf_alpha.outlier_guard import (
    estimate_outlier_evidence,
    evaluate_outlier_guard,
)
from pftf_alpha.sensor_stress import SensorStress, make_sensor_stress_case
from pftf_alpha.shared_trend_inference import infer_shared_trend_layers


def test_robust_score_rises_for_calibration_outliers() -> None:
    scores = []
    for stress in (SensorStress.CONTROL, SensorStress.OUTLIERS_05):
        case = make_sensor_stress_case(
            stress,
            160,
            reference_count=256,
            seed=41,
        )
        inference = infer_shared_trend_layers(case.points)
        evidence = estimate_outlier_evidence(
            case.points,
            inference.inference.layer_ids,
        )
        scores.append(evidence.maximum_joint_score)
    assert scores[1] > scores[0]


def test_phase9_smoke_cannot_promote_a_reduced_panel() -> None:
    result = evaluate_outlier_guard(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        seed=31,
        surface_sample_count=64,
    )
    assert result.case_count == 1
    assert result.phase9_supported is False
    assert result.deployment_supported is False
