from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from pftf_alpha.s3dis_two_layer_intake import (
    audit_s3dis_archive,
    extract_calibration_targets,
    run_s3dis_two_layer_intake,
)
from pftf_alpha.s3dis_two_layer_intake_protocol import ARCHIVE_NAME


def _write_archive(path: Path, *, unsafe: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for index in range(1, 7):
            prefix = f"Stanford3dDataset_v1.2_Aligned_Version/Area_{index}/office_1"
            bundle.writestr(f"{prefix}/Annotations/board_1.txt", f"{index} 0 0\n")
            bundle.writestr(f"{prefix}/Annotations/wall_1.txt", f"{index} 1 0\n")
            bundle.writestr(f"{prefix}/Annotations/chair_1.txt", f"{index} 2 0\n")
        if unsafe:
            bundle.writestr("../Area_1/Annotations/board_2.txt", "0 0 0\n")


def test_archive_audit_uses_metadata_and_counts_all_areas(tmp_path: Path) -> None:
    archive = tmp_path / ARCHIVE_NAME
    _write_archive(archive)

    inventory = audit_s3dis_archive(archive)

    assert inventory.member_count == 18
    assert inventory.calibration_target_member_count == 10
    assert inventory.reserved_target_member_count == 2
    assert inventory.target_member_counts["Area_5"] == {"board": 1, "wall": 1}


def test_calibration_extraction_never_opens_area5(tmp_path: Path) -> None:
    archive = tmp_path / ARCHIVE_NAME
    extraction = tmp_path / "calibration"
    _write_archive(archive)

    result = run_s3dis_two_layer_intake(archive, extraction)

    assert result.external_archive_intake_supported
    assert result.extracted_member_count == 10
    assert not result.reserved_content_opened
    assert not list(extraction.rglob("Area_5"))
    extracted = sorted(path.name for path in extraction.rglob("*.txt"))
    assert extracted == ["board_1.txt"] * 5 + ["wall_1.txt"] * 5


def test_extractor_rejects_unsafe_member_paths(tmp_path: Path) -> None:
    archive = tmp_path / ARCHIVE_NAME
    _write_archive(archive, unsafe=True)

    with pytest.raises(ValueError, match="unsafe ZIP member"):
        extract_calibration_targets(archive, tmp_path / "calibration")
