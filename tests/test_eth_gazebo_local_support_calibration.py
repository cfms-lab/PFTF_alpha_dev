from pftf_alpha.eth_gazebo_local_support_calibration import (
    CANDIDATE_GRID,
    FSCORE_TOLERANCE,
    RECALL_TOLERANCE,
    candidate_id,
)


def test_phase41_calibration_grid_is_small_and_frozen() -> None:
    assert CANDIDATE_GRID == tuple(
        (support, dispersion)
        for support in (2, 3, 4)
        for dispersion in (0.15, 0.20, 0.25)
    )
    assert FSCORE_TOLERANCE == 0.025
    assert RECALL_TOLERANCE == 0.01
    assert candidate_id(2, 0.15) == "support02_dispersion0150mm"
