from __future__ import annotations

import zipfile
from pathlib import Path

from pftf_alpha.s3dis_room_layer_intake import run_s3dis_room_layer_intake
from pftf_alpha.s3dis_two_layer_intake_protocol import ARCHIVE_NAME


def _write_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for index in range(1, 7):
            prefix = f"Stanford3dDataset_v1.2_Aligned_Version/Area_{index}/office_1"
            bundle.writestr(f"{prefix}/Annotations/floor_1.txt", f"{index} 0 0\n")
            bundle.writestr(
                f"{prefix}/Annotations/ceiling_1.txt", f"{index} 0 3\n"
            )
            bundle.writestr(f"{prefix}/Annotations/wall_1.txt", f"{index} 1 0\n")


def test_room_layer_intake_extracts_calibration_only(tmp_path: Path) -> None:
    archive = tmp_path / ARCHIVE_NAME
    extraction = tmp_path / "calibration"
    _write_archive(archive)

    result = run_s3dis_room_layer_intake(archive, extraction)

    assert result.floor_ceiling_intake_supported
    assert result.calibration_target_member_count == 10
    assert result.reserved_target_member_count == 2
    assert result.extracted_member_count == 10
    assert not result.reserved_content_opened
    assert not list(extraction.rglob("Area_5"))
    assert not list(extraction.rglob("wall*.txt"))
