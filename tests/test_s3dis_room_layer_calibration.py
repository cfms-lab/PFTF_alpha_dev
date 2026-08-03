from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pftf_alpha.s3dis_room_layer_calibration import (
    evaluate_room_layer_calibration,
)


def _write_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack((points, np.zeros_like(points))), fmt="%.8f")


def _plane(z: float) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 30), np.linspace(-1.0, 1.0, 20))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def test_room_layer_calibration_merges_fragments(tmp_path: Path) -> None:
    annotations = tmp_path / "Area_1" / "office_1" / "Annotations"
    floor = _plane(0.0)
    ceiling = _plane(3.0)
    _write_xyz(annotations / "floor_1.txt", floor[:300])
    _write_xyz(annotations / "floor_2.txt", floor[300:])
    _write_xyz(annotations / "ceiling_1.txt", ceiling)

    result = evaluate_room_layer_calibration(tmp_path)

    assert result.paired_room_count == 1
    pair = result.pairs[0]
    assert len(pair.floor_paths) == 2
    assert pair.normal_angle_degrees == pytest.approx(0.0)
    assert pair.plane_gap == pytest.approx(3.0)
    assert pair.bbox_overlap_fraction == pytest.approx(1.0)
    assert pair.floor_points_in_common_footprint == 600
    assert pair.ceiling_points_in_common_footprint == 600


def test_room_layer_calibration_rejects_area5(tmp_path: Path) -> None:
    annotations = tmp_path / "Area_5" / "office_1" / "Annotations"
    _write_xyz(annotations / "floor_1.txt", _plane(0.0))
    _write_xyz(annotations / "ceiling_1.txt", _plane(3.0))

    with pytest.raises(ValueError, match="reserved Area 5"):
        evaluate_room_layer_calibration(tmp_path)
