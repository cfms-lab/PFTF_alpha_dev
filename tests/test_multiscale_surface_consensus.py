import numpy as np

from pftf_alpha.multiscale_surface_consensus import (
    MultiscaleQuadraticConfig,
    calibrated_zero_harm_threshold,
    estimate_multiscale_quadratic_consensus,
    evaluate_multiscale_surface_consensus,
    multiscale_quadratic_scores,
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


def test_multiscale_quadratic_score_flags_isolated_off_surface_point() -> None:
    points, labels, outlier_index = _parallel_quadratic_grids_with_outlier()
    scores = multiscale_quadratic_scores(points, labels)
    coherent = np.delete(scores.best_standardized_residuals, outlier_index)
    assert scores.best_standardized_residuals[outlier_index] > 5.0
    assert scores.best_standardized_residuals[outlier_index] > np.max(coherent)
    evidence = estimate_multiscale_quadratic_consensus(points, labels)
    assert evidence.maximum_standardized_residual == (
        scores.best_standardized_residuals[outlier_index]
    )


def test_multiscale_scores_are_finite_on_local_bump() -> None:
    case = make_sensor_stress_case(
        SensorStress.LOCAL_BUMP,
        160,
        reference_count=256,
        seed=17,
    )
    inference = infer_shared_trend_layers(case.points)
    scores = multiscale_quadratic_scores(
        case.points,
        inference.inference.layer_ids,
    )
    assert np.all(np.isfinite(scores.best_standardized_residuals))
    assert set(np.unique(scores.selected_neighbor_counts)).issubset({12, 18, 24})


def test_calibration_threshold_is_strictly_below_minimum_harm_score() -> None:
    threshold = calibrated_zero_harm_threshold([7.0, 3.0, 5.0])
    assert threshold is not None
    assert threshold < 3.0
    assert np.nextafter(threshold, np.inf) == 3.0
    assert calibrated_zero_harm_threshold([]) is None


def test_phase11_reduced_calibration_cannot_open_held_out_panel() -> None:
    result = evaluate_multiscale_surface_consensus(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_seed=41,
        held_out_seed=43,
        surface_sample_count=64,
        consensus_config=MultiscaleQuadraticConfig(neighbor_counts=(8, 12)),
    )
    assert result.calibration.case_count == 1
    assert result.calibration.full_protocol is False
    assert result.calibration.panel_gate_passed is False
    assert result.held_out is None
    assert result.phase11_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.real_scan_supported is False
    assert result.deployment_supported is False
