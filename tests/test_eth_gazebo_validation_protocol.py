from pathlib import Path

import pytest

import pftf_alpha.eth_gazebo_validation_protocol as protocol
from pftf_alpha.open3d_fgr_pipeline import nonconsecutive_fragment_pairs


def test_phase39_gazebo_inputs_and_pair_universe_are_frozen() -> None:
    assert len(protocol.SCAN_MEMBERS) == protocol.SCAN_COUNT == 32
    assert protocol.SCAN_MEMBERS[0].endswith("Hokuyo_0.csv")
    assert protocol.SCAN_MEMBERS[-1].endswith("Hokuyo_31.csv")
    assert len(nonconsecutive_fragment_pairs(protocol.SCAN_COUNT)) == 465
    assert protocol.EXPECTED_PAIR_COUNT == 465
    assert protocol.SELECTED_PARAMETERS["voxel_size_meters"] == 0.5
    assert protocol.SELECTED_PARAMETERS["use_point_to_plane_icp"] is True


def test_phase39_gazebo_directory_verifier_opens_no_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []

    class Entry:
        def __init__(self, filename: str) -> None:
            self.filename = filename

    class FakeZip:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> "FakeZip":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def infolist(self) -> list[Entry]:
            return [
                *(Entry(name) for name in protocol.SCAN_MEMBERS),
                Entry(protocol.LABEL_MEMBER),
            ]

        def open(self, member: str) -> None:
            opened.append(member)
            raise AssertionError("directory verification must not open members")

    class Stat:
        st_size = protocol.ARCHIVE_BYTE_COUNT

    path = Path(protocol.ARCHIVE_NAME)
    monkeypatch.setattr(Path, "stat", lambda self: Stat())
    monkeypatch.setattr(
        protocol,
        "_hashes",
        lambda value: (protocol.ARCHIVE_MD5, protocol.ARCHIVE_SHA256),
    )
    monkeypatch.setattr(protocol.zipfile, "ZipFile", FakeZip)

    result = protocol.verify_gazebo_archive_directory(path)

    assert opened == []
    assert result.label_member_content_opened is False
