import numpy as np
import pytest

from pftf_alpha.paired_scan_persistence import (
    PairedScanPersistenceConfig,
    paired_scan_persistence_scores,
)
from pftf_alpha.sensor_stress import SensorStress
from pftf_alpha.studentized_paired_scan import (
    StudentizedPairedConfig,
    evaluate_studentized_paired_scan,
    studentized_paired_scores,
)


def _paired_curved_layers() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    axis = np.linspace(-1.0, 1.0, 7)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    first = first.ravel()
    second = second.ravel()
    bump = 0.25 * np.exp(-((first - 0.2) ** 2 + second**2) / 0.12)
    lower = np.column_stack((first, second, bump - 0.4))
    upper = np.column_stack((first, second, bump + 0.4))
    points = np.vstack((lower, upper))
    labels = np.concatenate(
        (
            np.zeros(lower.shape[0], dtype=np.int64),
            np.ones(upper.shape[0], dtype=np.int64),
        )
    )
    return points.copy(), labels, points.copy(), 1 - labels


def test_studentized_score_aligns_layers_and_expands_predictive_scale() -> None:
    primary, primary_labels, replicate, replicate_labels = _paired_curved_layers()
    plain = paired_scan_persistence_scores(
        primary,
        primary_labels,
        replicate,
        replicate_labels,
        PairedScanPersistenceConfig(neighbor_counts=(8, 12)),
    )
    studentized = studentized_paired_scores(
        primary,
        primary_labels,
        replicate,
        replicate_labels,
        StudentizedPairedConfig(neighbor_counts=(8, 12)),
    )
    assert studentized.primary_to_replicate_layer_mapping == (1, 0)
    assert np.all(np.isfinite(studentized.best_standardized_residuals))
    assert np.all(studentized.selected_query_leverages >= 0.0)
    assert np.all(
        studentized.best_standardized_residuals
        <= plain.best_standardized_residuals + 1e-12
    )


def test_studentized_score_localizes_nonpersistent_primary_outlier() -> None:
    primary, primary_labels, replicate, replicate_labels = _paired_curved_layers()
    outlier_index = 24
    primary[outlier_index, 2] += 0.20
    scores = studentized_paired_scores(
        primary,
        primary_labels,
        replicate,
        replicate_labels,
        StudentizedPairedConfig(neighbor_counts=(8, 12)),
    )
    assert int(np.argmax(scores.best_standardized_residuals)) == outlier_index
    assert scores.best_standardized_residuals[outlier_index] > 1.0


def test_phase16_reduced_panel_cannot_open_final_held_out() -> None:
    result = evaluate_studentized_paired_scan(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_a_seed=113,
        calibration_b_seed=127,
        final_held_out_seed=131,
        surface_sample_count=64,
        studentized_config=StudentizedPairedConfig(neighbor_counts=(8, 12)),
    )
    assert result.calibration_a.case_count == 1
    assert result.calibration_b.case_count == 1
    assert result.calibration_a.full_protocol is False
    assert result.calibration_b.full_protocol is False
    assert result.final_held_out is None
    assert result.phase16_supported is False
    assert result.paired_synthetic_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase16_rejects_reused_seeds() -> None:
    with pytest.raises(ValueError, match="must differ"):
        evaluate_studentized_paired_scan(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            calibration_a_seed=137,
            calibration_b_seed=137,
            final_held_out_seed=139,
            surface_sample_count=64,
        )
