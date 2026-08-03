import numpy as np
import pytest

from pftf_alpha.eth_gazebo_local_support import local_support_route


def test_local_support_keeps_anchor_and_corroborated_target_only_cells() -> None:
    anchor = np.array(((0.10, 0.10, 0.10), (0.20, 0.10, 0.10)))
    targets = np.array(
        (
            (0.15, 0.10, 0.10),
            (0.85, 0.10, 0.10),
            (0.95, 0.10, 0.10),
            (1.60, 0.10, 0.10),
            (1.65, 0.10, 0.10),
        )
    )
    provenance = np.array((2, 2, 3, 4, 4))

    result = local_support_route(
        anchor,
        targets,
        provenance,
        np.zeros(3),
        minimum_support=2,
        maximum_dispersion_meters=0.15,
    )

    assert result.anchor_cell_count == 1
    assert result.target_cell_count == 3
    assert result.overlapping_target_cell_count == 1
    assert result.target_only_cell_count == 2
    assert result.corroborated_target_only_cell_count == 1
    assert result.rejected_target_only_cell_count == 1
    assert result.scan_fused_points.shape == (3, 3)
    assert result.local_points.shape == (2, 3)
    assert np.allclose(result.local_points[0], (0.15, 0.10, 0.10))
    assert np.allclose(result.local_points[1], (0.90, 0.10, 0.10))


def test_local_support_dispersion_can_fail_closed() -> None:
    anchor = np.array(((0.10, 0.10, 0.10),))
    targets = np.array(((0.80, 0.10, 0.10), (1.40, 0.10, 0.10)))
    provenance = np.array((2, 3))

    result = local_support_route(
        anchor,
        targets,
        provenance,
        np.zeros(3),
        minimum_support=2,
        maximum_dispersion_meters=0.15,
    )

    assert result.corroborated_target_only_cell_count == 0
    assert np.array_equal(result.local_points, anchor)


@pytest.mark.parametrize(
    ("support", "dispersion"),
    ((1, 0.15), (2, 0.0)),
)
def test_local_support_rejects_invalid_thresholds(
    support: int,
    dispersion: float,
) -> None:
    with pytest.raises(ValueError):
        local_support_route(
            np.array(((0.0, 0.0, 0.0),)),
            np.array(((1.0, 0.0, 0.0),)),
            np.array((2,)),
            np.zeros(3),
            minimum_support=support,
            maximum_dispersion_meters=dispersion,
        )
