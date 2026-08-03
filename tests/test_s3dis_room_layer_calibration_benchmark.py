from __future__ import annotations

from pathlib import Path

import numpy as np

from pftf_alpha.s3dis_room_layer_calibration import (
    evaluate_room_layer_calibration,
)
from pftf_alpha.s3dis_room_layer_calibration_benchmark import (
    build_room_layer_case,
)


def _write_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack((points, np.zeros_like(points))), fmt="%.8f")


def test_room_layer_case_is_balanced_disjoint_and_deterministic(tmp_path: Path) -> None:
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 40), np.linspace(-1.0, 1.0, 30))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    ceiling = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 3.0)))
    annotations = tmp_path / "Area_1" / "office_1" / "Annotations"
    _write_xyz(annotations / "floor_1.txt", floor)
    _write_xyz(annotations / "ceiling_1.txt", ceiling)
    calibration = evaluate_room_layer_calibration(tmp_path)
    pair = calibration.to_dict()["pairs"][0]

    case = build_room_layer_case(tmp_path, pair)
    repeated = build_room_layer_case(tmp_path, pair)

    assert case.points.shape == (160, 3)
    assert case.reference_points.shape == (1_024, 3)
    assert np.array_equal(case.points, repeated.points)
    assert np.array_equal(case.reference_points, repeated.reference_points)
    assert np.max(np.abs(case.points - repeated.points)) == 0.0
