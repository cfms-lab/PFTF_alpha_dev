from __future__ import annotations

from pathlib import Path

import numpy as np

from pftf_alpha.s3dis_two_layer_calibration import (
    evaluate_calibration_root,
    write_result,
)
from pftf_alpha.s3dis_two_layer_calibration_benchmark import build_case_from_pair


def _write_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack((points, np.zeros_like(points))), fmt="%.8f")


def test_build_case_uses_disjoint_balanced_coordinate_split(tmp_path: Path) -> None:
    grid = np.linspace(-1.0, 1.0, 30)
    x, y = np.meshgrid(grid, grid)
    board = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.05)))
    wall = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    annotations = tmp_path / "Area_1" / "office_1" / "Annotations"
    _write_xyz(annotations / "board_1.txt", board)
    _write_xyz(annotations / "wall_1.txt", wall)
    calibration = evaluate_calibration_root(tmp_path, max_fit_points=1_000)
    artifact = tmp_path / "calibration.json"
    write_result(calibration, artifact)

    case = build_case_from_pair(tmp_path, calibration.to_dict()["pairs"][0])
    repeated = build_case_from_pair(tmp_path, calibration.to_dict()["pairs"][0])

    assert case.points.shape == (160, 3)
    assert case.reference_points.shape == (1_024, 3)
    assert np.array_equal(case.points, repeated.points)
    assert np.array_equal(case.reference_points, repeated.reference_points)
    observed = {tuple(row) for row in case.points}
    reference = {tuple(row) for row in case.reference_points}
    assert observed.isdisjoint(reference)
