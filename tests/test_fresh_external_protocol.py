from pathlib import Path

import pytest

import pftf_alpha.fresh_external_protocol as protocol


def test_phase38_pair_universe_and_frozen_inputs() -> None:
    assert protocol.SCAN_MEMBERS == tuple(
        f"plain_01-Sep-2011-16_39_18/csv_local/Hokuyo_{index}.csv"
        for index in range(31)
    )
    assert protocol.EXPECTED_PAIR_COUNT == 435
    assert protocol.MAX_RELATIVE_ROTATION_ERROR_DEGREES == 15.0
    assert protocol.MAX_RELATIVE_TRANSLATION_ERROR_METERS == 0.30
    assert protocol.phase37_parameters()["voxel_size_meters"] == 0.05


def test_phase38_archive_verifier_never_opens_label_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_members: list[str] = []

    class FakeEntry:
        def __init__(self, filename: str) -> None:
            self.filename = filename

    class FakeZip:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> "FakeZip":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def infolist(self) -> list[FakeEntry]:
            return [
                *(FakeEntry(name) for name in protocol.SCAN_MEMBERS),
                FakeEntry(protocol.LABEL_MEMBER),
            ]

        def open(self, member: str) -> None:
            opened_members.append(member)
            raise AssertionError("archive verifier must not open members")

    class FakeStat:
        st_size = protocol.ARCHIVE_BYTE_COUNT

    archive = Path(protocol.ARCHIVE_NAME)
    monkeypatch.setattr(Path, "stat", lambda self: FakeStat())
    monkeypatch.setattr(
        protocol,
        "_hashes",
        lambda path: (protocol.ARCHIVE_MD5, protocol.ARCHIVE_SHA256),
    )
    monkeypatch.setattr(protocol.zipfile, "ZipFile", FakeZip)

    result = protocol.verify_archive_directory(archive)

    assert opened_members == []
    assert result.label_member_name_present is True
    assert result.label_member_content_opened is False
