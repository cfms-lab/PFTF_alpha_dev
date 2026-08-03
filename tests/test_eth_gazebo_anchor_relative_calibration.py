from pftf_alpha.eth_gazebo_anchor_relative_calibration import (
    CANDIDATE_GRID,
    DEVELOPMENT_SOURCE_INDICES,
    MINIMUM_ADDED_CELL_COUNT,
    candidate_id,
)


def test_phase42_development_grid_and_sources_are_frozen() -> None:
    assert DEVELOPMENT_SOURCE_INDICES == (0, 17)
    assert CANDIDATE_GRID == tuple(
        (nearest, plane, alignment)
        for nearest in (0.75, 1.00, 1.50)
        for plane in (0.15, 0.30, 0.50)
        for alignment in (0.0, 0.75)
    )
    assert MINIMUM_ADDED_CELL_COUNT == 3
    assert candidate_id(1.5, 0.5, 0.75) == "anchor_d150_p050_n075"
