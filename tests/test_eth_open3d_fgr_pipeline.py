import io
from pathlib import Path

import numpy as np
import pytest

import pftf_alpha.eth_open3d_fgr_pipeline as pipeline
from pftf_alpha.fresh_external_protocol import EXPECTED_PAIR_COUNT, SCAN_COUNT
from pftf_alpha.open3d_fgr_pipeline import nonconsecutive_fragment_pairs


def test_phase38_csv_loader_uses_only_xyz() -> None:
    stream = io.BytesIO(
        b"Time_in_sec,x,y,z,Intensities,2DscanId,PointId\n"
        b"1.0,2.0,3.0,4.0,-1.0,5,6\n"
        b"1.1,7.0,8.0,9.0,-1.0,10,11\n"
        b"1.2,12.0,13.0,14.0,-1.0,15,16\n"
    )

    points = pipeline._load_xyz(stream)

    np.testing.assert_allclose(
        points,
        [[2.0, 3.0, 4.0], [7.0, 8.0, 9.0], [12.0, 13.0, 14.0]],
    )


def test_phase38_csv_loader_rejects_changed_header() -> None:
    stream = io.BytesIO(b"x,y,z\n1,2,3\n")
    with pytest.raises(ValueError, match="header"):
        pipeline._load_xyz(stream)


def test_phase38_prediction_pair_universe_is_frozen() -> None:
    pairs = nonconsecutive_fragment_pairs(SCAN_COUNT)
    assert len(pairs) == EXPECTED_PAIR_COUNT == 435
    assert pairs[0] == (0, 2)
    assert pairs[-1] == (28, 30)


def test_phase38_protocol_hash_is_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "changed.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        pipeline._verify_protocol(path)
