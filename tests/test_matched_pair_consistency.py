import numpy as np
import pytest

from pftf_alpha.matched_pair_consistency import (
    MatchedPairConfig,
    evaluate_matched_pair_consistency,
    make_matched_repeat_observation,
    matched_displacement_scores,
)
from pftf_alpha.sensor_stress import SensorStress, make_sensor_stress_case


def test_matched_repeat_is_deterministic_and_preserves_pair_count() -> None:
    primary = make_sensor_stress_case(
        SensorStress.OUTLIERS_03,
        96,
        reference_count=128,
        seed=17,
    )
    first = make_matched_repeat_observation(
        primary.points,
        primary.point_component_labels,
        SensorStress.OUTLIERS_03,
        seed=23,
    )
    second = make_matched_repeat_observation(
        primary.points,
        primary.point_component_labels,
        SensorStress.OUTLIERS_03,
        seed=23,
    )
    assert first.points.shape == primary.points.shape
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(
        first.transient_outlier_indices,
        second.transient_outlier_indices,
    )
    assert first.transient_outlier_indices.size == 3
    assert (
        first.transient_outlier_index_sha256
        == second.transient_outlier_index_sha256
    )


def test_matched_displacement_localizes_one_transient_pair() -> None:
    axis = np.linspace(-1.0, 1.0, 32)
    primary = np.column_stack((axis, 0.2 * axis, 0.1 * axis**2))
    repeat = primary + np.asarray((0.01, -0.02, 0.005))
    transient_index = 7
    repeat[transient_index, 2] += 0.20
    scores = matched_displacement_scores(primary, repeat)
    assert np.all(np.isfinite(scores.point_scores))
    assert int(np.argmax(scores.point_scores)) == transient_index
    assert scores.point_scores[transient_index] > 10.0
    assert np.count_nonzero(scores.point_scores > 1.0) == 1


def test_phase17_reduced_panel_cannot_open_final_held_out() -> None:
    result = evaluate_matched_pair_consistency(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_a_seed=149,
        calibration_b_seed=151,
        final_held_out_seed=157,
        surface_sample_count=64,
    )
    assert result.calibration_a.case_count == 1
    assert result.calibration_b.case_count == 1
    assert result.calibration_a.full_protocol is False
    assert result.calibration_b.full_protocol is False
    assert result.final_held_out is None
    assert result.phase17_supported is False
    assert result.exact_correspondence_synthetic_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase17_rejects_reused_seeds() -> None:
    with pytest.raises(ValueError, match="must differ"):
        evaluate_matched_pair_consistency(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            calibration_a_seed=163,
            calibration_b_seed=163,
            final_held_out_seed=167,
            surface_sample_count=64,
            matched_pair_config=MatchedPairConfig(),
        )
