import numpy as np
import pytest

from pftf_alpha.paired_scan_persistence import (
    PairedScanPersistenceConfig,
    evaluate_paired_scan_persistence,
    paired_scan_persistence_scores,
)
from pftf_alpha.sensor_stress import SensorStress


def _paired_quadratic_layers() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    axis = np.linspace(-1.0, 1.0, 7)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    first = first.ravel()
    second = second.ravel()
    lower_height = 0.02 * first**2 - 0.01 * first * second
    upper_height = 0.45 + 0.01 * second**2
    lower = np.column_stack((first, second, lower_height))
    upper = np.column_stack((first, second, upper_height))
    points = np.vstack((lower, upper))
    labels = np.concatenate(
        (
            np.zeros(lower.shape[0], dtype=np.int64),
            np.ones(upper.shape[0], dtype=np.int64),
        )
    )
    swapped_labels = 1 - labels
    return points.copy(), labels, points.copy(), swapped_labels


def test_paired_scan_scores_align_swapped_layers_and_fit_coherent_surface() -> None:
    primary, primary_labels, replicate, replicate_labels = (
        _paired_quadratic_layers()
    )
    scores = paired_scan_persistence_scores(
        primary,
        primary_labels,
        replicate,
        replicate_labels,
        PairedScanPersistenceConfig(neighbor_counts=(8, 12)),
    )
    assert scores.primary_to_replicate_layer_mapping == (1, 0)
    assert np.all(np.isfinite(scores.best_standardized_residuals))
    assert np.max(scores.best_standardized_residuals) < 1e-3


def test_paired_scan_score_localizes_nonpersistent_primary_outlier() -> None:
    primary, primary_labels, replicate, replicate_labels = (
        _paired_quadratic_layers()
    )
    outlier_index = 24
    primary[outlier_index, 2] += 0.15
    scores = paired_scan_persistence_scores(
        primary,
        primary_labels,
        replicate,
        replicate_labels,
        PairedScanPersistenceConfig(neighbor_counts=(8, 12)),
    )
    assert int(np.argmax(scores.best_standardized_residuals)) == outlier_index
    assert scores.best_standardized_residuals[outlier_index] > 5.0


def test_phase15_reduced_panel_cannot_open_final_held_out() -> None:
    result = evaluate_paired_scan_persistence(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_a_seed=97,
        calibration_b_seed=101,
        final_held_out_seed=103,
        surface_sample_count=64,
        persistence_config=PairedScanPersistenceConfig(
            neighbor_counts=(8, 12)
        ),
    )
    assert result.calibration_a.case_count == 1
    assert result.calibration_b.case_count == 1
    assert result.calibration_a.full_protocol is False
    assert result.calibration_b.full_protocol is False
    assert result.final_held_out is None
    assert result.phase15_supported is False
    assert result.paired_synthetic_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase15_rejects_reused_seeds() -> None:
    with pytest.raises(ValueError, match="must differ"):
        evaluate_paired_scan_persistence(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            calibration_a_seed=107,
            calibration_b_seed=107,
            final_held_out_seed=109,
            surface_sample_count=64,
        )
