from pathlib import Path

import pytest

import pftf_alpha.eth_gazebo_fgr_icp_pipeline as pipeline
from pftf_alpha.eth_gazebo_validation_protocol import (
    EXPECTED_PAIR_COUNT,
    SCAN_COUNT,
    SELECTED_PARAMETERS,
)
from pftf_alpha.open3d_fgr_pipeline import nonconsecutive_fragment_pairs


def test_phase39_gazebo_generator_uses_frozen_pipeline() -> None:
    assert len(nonconsecutive_fragment_pairs(SCAN_COUNT)) == EXPECTED_PAIR_COUNT
    assert SELECTED_PARAMETERS["voxel_size_meters"] == 0.5
    assert SELECTED_PARAMETERS[
        "icp_maximum_correspondence_distance_meters"
    ] == 0.75
    assert SELECTED_PARAMETERS["icp_max_iterations"] == 50


def test_phase39_gazebo_protocol_hash_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "changed.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        pipeline._verify_protocol(path)
