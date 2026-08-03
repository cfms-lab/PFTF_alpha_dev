import pytest

from pftf_alpha.eth_pipeline_calibration import (
    ETHCalibrationCandidate,
    candidate_grid,
    candidate_parameters,
    select_candidate,
)


def _candidate(
    candidate_id: str,
    voxel: float,
    use_icp: bool,
    correct_count: int,
) -> ETHCalibrationCandidate:
    return ETHCalibrationCandidate(
        candidate_id=candidate_id,
        voxel_size_meters=voxel,
        use_point_to_plane_icp=use_icp,
        parameters={},
        prediction_count=435,
        correct_count=correct_count,
        incorrect_count=435 - correct_count,
        rotation_threshold_pass_count=correct_count,
        translation_threshold_pass_count=correct_count,
        median_rotation_error_degrees=1.0,
        median_translation_error_meters=1.0,
        elapsed_seconds=1.0,
        predictions=(),
    )


def test_phase39_candidate_grid_is_exact_and_bounded() -> None:
    grid = candidate_grid()
    assert len(grid) == 8
    assert [row["candidate_id"] for row in grid] == [
        "fgr_v010",
        "fgr_icp_v010",
        "fgr_v020",
        "fgr_icp_v020",
        "fgr_v030",
        "fgr_icp_v030",
        "fgr_v050",
        "fgr_icp_v050",
    ]
    assert candidate_parameters(0.2, use_icp=True)[
        "icp_maximum_correspondence_distance_meters"
    ] == pytest.approx(0.3)


def test_phase39_selection_uses_correct_count_then_frozen_ties() -> None:
    candidates = (
        _candidate("large", 0.3, False, 8),
        _candidate("refined", 0.2, True, 8),
        _candidate("simple", 0.2, False, 8),
        _candidate("fewer", 0.1, False, 7),
    )
    assert select_candidate(candidates).candidate_id == "simple"


def test_phase39_zero_correct_is_not_viable() -> None:
    candidates = (_candidate("zero", 0.1, False, 0),)
    assert select_candidate(candidates) is None
