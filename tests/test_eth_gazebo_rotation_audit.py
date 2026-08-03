import io

import numpy as np
import pytest

from pftf_alpha.eth_gazebo_rotation_audit import (
    GazeboLabeledObservation,
    _parse_gazebo_poses,
    summarize_gazebo,
)


def _pose_csv() -> bytes:
    header = "poseId, timestamp, " + ", ".join(
        f"T{row}{column}" for row in range(4) for column in range(4)
    )
    rows = []
    for index in range(32):
        matrix = np.eye(4)
        matrix[0, 3] = index
        values = ", ".join(str(value) for value in matrix.ravel())
        rows.append(f"{index}, {index}.0, {values}")
    return (header + "\n" + "\n".join(rows) + "\n").encode("ascii")


def _observation(accepted: bool, correct: bool) -> GazeboLabeledObservation:
    return GazeboLabeledObservation(
        source_index=0,
        target_index=2,
        prediction_rotation_radians=0.0,
        scene_relative_rotation_percentile=0.1,
        guarded_accept=accepted,
        relative_rotation_error_degrees=0.0 if correct else 90.0,
        relative_translation_error_meters=0.0 if correct else 2.0,
        frozen_correct=correct,
    )


def test_phase39_gazebo_pose_parser_requires_32_rows() -> None:
    poses = _parse_gazebo_poses(io.BytesIO(_pose_csv()))
    assert len(poses) == 32
    assert poses[-1][0, 3] == pytest.approx(31.0)


def test_phase39_gazebo_summary_applies_unchanged_gates() -> None:
    rows = tuple(
        [*(_observation(True, True) for _ in range(9))]
        + [_observation(False, True)]
        + [*(_observation(True, False) for _ in range(8))]
        + [*(_observation(False, False) for _ in range(2))]
    )
    result = summarize_gazebo(rows)
    assert result.correct_retention == pytest.approx(0.9)
    assert result.incorrect_rejection == pytest.approx(0.2)
    assert result.precision_improved is True
    assert result.fresh_scene_transfer_gate_passed is True
