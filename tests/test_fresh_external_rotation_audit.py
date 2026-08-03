import io
import math

import numpy as np
import pytest

from pftf_alpha.fresh_external_rotation_audit import (
    ETHLabeledRotationObservation,
    _parse_pose_labels,
    _relative_target_to_source,
    _rigid_errors,
    _summary,
)


def _pose_csv() -> bytes:
    header = "poseId, timestamp, " + ", ".join(
        f"T{row}{column}" for row in range(4) for column in range(4)
    )
    rows = []
    for index in range(31):
        matrix = np.eye(4)
        matrix[0, 3] = float(index)
        values = ", ".join(str(value) for value in matrix.ravel())
        rows.append(f"{index}, {index}.0, {values}")
    return (header + "\n" + "\n".join(rows) + "\n").encode("ascii")


def test_phase38_pose_parser_and_relative_direction() -> None:
    poses = _parse_pose_labels(io.BytesIO(_pose_csv()))
    relative = _relative_target_to_source(poses[0], poses[2])
    np.testing.assert_allclose(relative[:3, 3], [2.0, 0.0, 0.0])
    rotation, translation = _rigid_errors(relative, relative)
    assert rotation == pytest.approx(0.0)
    assert translation == pytest.approx(0.0)


def _observation(
    *,
    accepted: bool,
    correct: bool,
) -> ETHLabeledRotationObservation:
    return ETHLabeledRotationObservation(
        source_index=0,
        target_index=2,
        prediction_rotation_radians=0.0,
        scene_relative_rotation_percentile=0.1,
        guarded_accept=accepted,
        relative_rotation_error_degrees=0.0 if correct else 90.0,
        relative_translation_error_meters=0.0 if correct else 2.0,
        frozen_correct=correct,
    )


def test_phase38_summary_applies_unchanged_gates() -> None:
    rows = tuple(
        [*(_observation(accepted=True, correct=True) for _ in range(9))]
        + [_observation(accepted=False, correct=True)]
        + [*(_observation(accepted=True, correct=False) for _ in range(8))]
        + [*(_observation(accepted=False, correct=False) for _ in range(2))]
    )
    result = _summary(rows)
    assert result.correct_retention == pytest.approx(0.9)
    assert result.incorrect_rejection == pytest.approx(0.2)
    assert result.guarded_precision > result.base_precision
    assert result.fresh_scene_transfer_gate_passed is True


def test_phase38_rigid_error_detects_rotation_and_translation() -> None:
    prediction = np.eye(4)
    angle = math.radians(20.0)
    prediction[:2, :2] = [
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ]
    prediction[0, 3] = 0.4
    rotation, translation = _rigid_errors(prediction, np.eye(4))
    assert rotation == pytest.approx(20.0)
    assert translation == pytest.approx(0.4)
