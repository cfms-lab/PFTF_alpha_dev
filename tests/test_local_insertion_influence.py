import numpy as np

from pftf_alpha.local_insertion_influence import (
    InfluenceRectangle,
    LocalInsertionInfluenceConfig,
    calibrate_influence_rectangle,
    estimate_local_insertion_influence,
    evaluate_local_insertion_influence,
    local_insertion_influence_scores,
)
from pftf_alpha.sensor_stress import SensorStress, make_sensor_stress_case
from pftf_alpha.shared_trend_inference import infer_shared_trend_layers


def _parallel_quadratic_grids_with_outlier() -> tuple[np.ndarray, np.ndarray, int]:
    coordinates = np.linspace(-1.0, 1.0, 7)
    xy = np.asarray([(x, y) for x in coordinates for y in coordinates])
    height = 0.08 * xy[:, 0] ** 2 - 0.05 * xy[:, 1] ** 2
    lower = np.column_stack((xy, height - 0.4))
    upper = np.column_stack((xy, height + 0.4))
    outlier = np.asarray([[0.0, 0.0, 0.15]])
    points = np.vstack((lower, outlier, upper))
    labels = np.concatenate(
        (
            np.zeros(lower.shape[0] + 1, dtype=np.int64),
            np.ones(upper.shape[0], dtype=np.int64),
        )
    )
    return points, labels, lower.shape[0]


def test_insertion_influence_is_largest_for_isolated_off_surface_point() -> None:
    points, labels, outlier_index = _parallel_quadratic_grids_with_outlier()
    scores = local_insertion_influence_scores(points, labels)
    coherent = np.delete(scores.best_standardized_influences, outlier_index)
    assert scores.best_standardized_influences[outlier_index] > np.max(coherent)
    evidence = estimate_local_insertion_influence(points, labels)
    assert evidence.peak_standardized_influence == (
        scores.best_standardized_influences[outlier_index]
    )
    assert evidence.peak_standardized_influence >= (
        evidence.support_standardized_influence
    )


def test_insertion_influence_is_finite_on_local_bump() -> None:
    case = make_sensor_stress_case(
        SensorStress.LOCAL_BUMP,
        160,
        reference_count=256,
        seed=17,
    )
    inference = infer_shared_trend_layers(case.points)
    scores = local_insertion_influence_scores(
        case.points,
        inference.inference.layer_ids,
    )
    assert np.all(np.isfinite(scores.best_standardized_influences))
    assert set(np.unique(scores.selected_neighbor_counts)).issubset({12, 18, 24})


def test_rectangle_calibration_keeps_best_zero_harm_accept_region() -> None:
    harmful = [(5.0, 1.0), (2.0, 5.0)]
    focus_safe = [(4.0, 0.5), (1.5, 4.0), (1.0, 1.0), (4.0, 4.0)]
    rectangle = calibrate_influence_rectangle(
        harmful,
        focus_safe,
        focus_safe,
    )
    assert rectangle is not None
    assert rectangle.retained_focus_safe_count == 4
    assert rectangle.peak_threshold < 5.0
    assert rectangle.support_threshold < 5.0
    for peak, support in harmful:
        assert not (
            peak <= rectangle.peak_threshold
            and support <= rectangle.support_threshold
        )


def test_unbounded_rectangle_serializes_as_standard_json_value() -> None:
    rectangle = InfluenceRectangle(
        peak_threshold=0.5,
        support_threshold=float("inf"),
        retained_focus_safe_count=4,
        retained_all_safe_count=8,
    )
    assert rectangle.to_dict()["support_threshold"] == "infinity"


def test_phase12_reduced_calibration_cannot_open_held_out_panel() -> None:
    result = evaluate_local_insertion_influence(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_seed=47,
        held_out_seed=53,
        surface_sample_count=64,
        influence_config=LocalInsertionInfluenceConfig(neighbor_counts=(8, 12)),
    )
    assert result.calibration.case_count == 1
    assert result.calibration.full_protocol is False
    assert result.calibration.panel_gate_passed is False
    assert result.held_out is None
    assert result.phase12_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.real_scan_supported is False
    assert result.deployment_supported is False
