from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pftf_alpha.s3dis_two_layer_calibration import evaluate_calibration_root


def _write_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros_like(points)
    np.savetxt(path, np.column_stack((points, rgb)), fmt="%.8f")


def _plane_xy(z: float, count: int = 20) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, count), np.linspace(-0.5, 0.5, count))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def _plane_yz(x_value: float, count: int = 20) -> np.ndarray:
    y, z = np.meshgrid(np.linspace(-1.0, 1.0, count), np.linspace(-0.5, 0.5, count))
    return np.column_stack((np.full(y.size, x_value), y.ravel(), z.ravel()))


def test_calibration_selects_nearest_parallel_wall(tmp_path: Path) -> None:
    annotations = tmp_path / "Area_1" / "office_1" / "Annotations"
    _write_xyz(annotations / "board_1.txt", _plane_xy(0.05))
    _write_xyz(annotations / "wall_1.txt", _plane_xy(0.0))
    _write_xyz(annotations / "wall_2.txt", _plane_yz(3.0))

    result = evaluate_calibration_root(tmp_path, max_fit_points=500)

    assert result.calibration_geometry_audit_supported
    assert result.board_instance_count == 1
    pair = result.pairs[0]
    assert pair.selected_wall_path is not None
    assert pair.selected_wall_path.endswith("wall_1.txt")
    assert pair.selected_pair is not None
    assert pair.selected_pair.normal_angle_degrees == pytest.approx(0.0)
    assert pair.selected_pair.plane_gap == pytest.approx(0.05)
    assert pair.selected_pair.board_footprint_grid_support == pytest.approx(1.0)


def test_calibration_rejects_reserved_area5(tmp_path: Path) -> None:
    annotations = tmp_path / "Area_5" / "office_1" / "Annotations"
    _write_xyz(annotations / "board_1.txt", _plane_xy(0.05))
    _write_xyz(annotations / "wall_1.txt", _plane_xy(0.0))

    with pytest.raises(ValueError, match="reserved Area 5"):
        evaluate_calibration_root(tmp_path)
