import numpy as np
import pytest

from pftf_alpha.eth_gazebo_anchor_relative import anchor_relative_route
from pftf_alpha.eth_gazebo_local_support import SourceReconstructionInputs


def _inputs() -> SourceReconstructionInputs:
    anchor = np.array(
        [(x, y, 0.0) for x in (0.0, 0.5, 1.0) for y in np.arange(0.0, 2.1, 0.5)]
    )
    target_centers = np.array(
        [(x, y, 0.0) for x in (1.6, 2.4) for y in (0.1, 0.9, 1.6)]
    )
    targets = np.vstack(
        (
            target_centers + np.array((0.02, 0.0, 0.0)),
            target_centers - np.array((0.02, 0.0, 0.0)),
        )
    )
    provenance = np.concatenate(
        (
            np.full(target_centers.shape[0], 2),
            np.full(target_centers.shape[0], 3),
        )
    )
    return SourceReconstructionInputs(
        source_index=0,
        pair_count=2,
        accepted_pair_count=2,
        rejected_pair_count=0,
        anchor_points=anchor,
        reference_points=anchor,
        target_points=targets,
        target_provenance=provenance,
        lower=np.zeros(3),
        upper=np.array((3.0, 3.0, 1.0)),
        characteristic_length=5.0,
    )


def test_anchor_relative_route_accepts_planar_aligned_cells() -> None:
    result = anchor_relative_route(
        _inputs(),
        maximum_nearest_anchor_distance_meters=2.0,
        maximum_anchor_plane_residual_meters=0.1,
        minimum_normal_alignment=0.9,
    )

    assert result.phase41_candidate_cell_count > 0
    assert result.anchor_relative_cell_count > 0
    assert result.mean_anchor_plane_residual_meters == pytest.approx(0.0)
    assert result.mean_normal_alignment == pytest.approx(1.0)


def test_anchor_relative_route_fails_closed_on_distance() -> None:
    result = anchor_relative_route(
        _inputs(),
        maximum_nearest_anchor_distance_meters=0.1,
        maximum_anchor_plane_residual_meters=0.1,
        minimum_normal_alignment=0.9,
    )

    assert result.phase41_candidate_cell_count > 0
    assert result.anchor_relative_cell_count == 0
    assert result.local_points.shape[0] < _inputs().anchor_points.shape[0]


@pytest.mark.parametrize(
    ("nearest", "plane", "alignment"),
    ((0.0, 0.1, 0.9), (1.0, 0.0, 0.9), (1.0, 0.1, 1.1)),
)
def test_anchor_relative_route_rejects_invalid_thresholds(
    nearest: float,
    plane: float,
    alignment: float,
) -> None:
    with pytest.raises(ValueError):
        anchor_relative_route(
            _inputs(),
            maximum_nearest_anchor_distance_meters=nearest,
            maximum_anchor_plane_residual_meters=plane,
            minimum_normal_alignment=alignment,
        )
