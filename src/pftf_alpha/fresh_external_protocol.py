"""Phase-38 preregistration for fresh ETH Mountain Plain validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .open3d_fgr_pipeline import phase37_parameters
from .scene_relative_rotation_guard import (
    MINIMUM_CORRECT_RETENTION,
    MINIMUM_INCORRECT_REJECTION,
    ROTATION_PERCENTILE_CUTOFF,
)

DATASET_NAME = "ETH Mountain Plain"
DATASET_DOI = "https://doi.org/10.3929/ethz-b-000721626"
DATASET_ITEM_UUID = "100a8c95-7d38-4e80-a3ce-0e211153a22b"
DATASET_LICENSE = "CC BY 4.0"
ARCHIVE_NAME = "plain_01-Sep-2011-16_39_18.zip"
ARCHIVE_BITSTREAM_UUID = "f493ec1b-b55c-43b7-be34-d230133fdd43"
ARCHIVE_BYTE_COUNT = 902_525_379
ARCHIVE_MD5 = "fb931a4ddf06720ec18774e2fdd0cc27"
ARCHIVE_SHA256 = (
    "d07ddd6f314c8caa2d91dea91646e8ca7c4ebdadd0139d6f9ca82fe12070d926"
)
ARCHIVE_ROOT = "plain_01-Sep-2011-16_39_18"
SCAN_COUNT = 31
SCAN_MEMBERS = tuple(
    f"{ARCHIVE_ROOT}/csv_local/Hokuyo_{index}.csv"
    for index in range(SCAN_COUNT)
)
LABEL_MEMBER = f"{ARCHIVE_ROOT}/leica/pose_scanner_leica.csv"
EXPECTED_PAIR_COUNT = 435
MAX_RELATIVE_ROTATION_ERROR_DEGREES = 15.0
MAX_RELATIVE_TRANSLATION_ERROR_METERS = 0.30


@dataclass(frozen=True)
class Phase38ArchiveDirectoryVerification:
    archive_path: str
    byte_count: int
    md5: str
    sha256: str
    scan_member_count: int
    scan_members_present: bool
    label_member_name_present: bool
    label_member_content_opened: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Phase38Preregistration:
    artifact_schema: str
    role: str
    dataset_name: str
    dataset_doi: str
    dataset_item_uuid: str
    dataset_license: str
    archive_name: str
    archive_bitstream_uuid: str
    archive_verification: Phase38ArchiveDirectoryVerification
    scan_members: tuple[str, ...]
    label_member: str
    scan_order: str
    pair_universe: str
    expected_pair_count: int
    pipeline_name: str
    pipeline_parameters: dict[str, object]
    prediction_matrix_convention: str
    correctness_rule: str
    maximum_relative_rotation_error_degrees: float
    maximum_relative_translation_error_meters: float
    rotation_percentile_cutoff: float
    minimum_correct_retention: float
    minimum_incorrect_rejection: float
    panel_gate: str
    label_boundary: str
    label_values_accessed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scan_members"] = list(self.scan_members)
        return payload


def _hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def verify_archive_directory(
    archive_path: str | Path,
) -> Phase38ArchiveDirectoryVerification:
    """Verify the container and member names without opening label content."""

    path = Path(archive_path)
    if path.name != ARCHIVE_NAME:
        raise ValueError(f"unexpected archive name: {path.name}")
    byte_count = path.stat().st_size
    if byte_count != ARCHIVE_BYTE_COUNT:
        raise ValueError("ETH archive byte count mismatch")
    md5, sha256 = _hashes(path)
    if md5 != ARCHIVE_MD5:
        raise ValueError("ETH archive MD5 mismatch")
    if sha256 != ARCHIVE_SHA256:
        raise ValueError("ETH archive SHA-256 mismatch")
    with zipfile.ZipFile(path) as source:
        names = frozenset(item.filename for item in source.infolist())
    missing = set(SCAN_MEMBERS).difference(names)
    if missing:
        raise ValueError(f"missing ETH scan members: {sorted(missing)}")
    if LABEL_MEMBER not in names:
        raise ValueError("missing ETH label member name")
    return Phase38ArchiveDirectoryVerification(
        archive_path=str(path),
        byte_count=byte_count,
        md5=md5,
        sha256=sha256,
        scan_member_count=len(SCAN_MEMBERS),
        scan_members_present=True,
        label_member_name_present=True,
        label_member_content_opened=False,
    )


def preregister_phase38(
    archive_path: str | Path,
) -> Phase38Preregistration:
    verification = verify_archive_directory(archive_path)
    return Phase38Preregistration(
        artifact_schema="pftf_alpha_fresh_external_protocol_phase38/v1",
        role="pre_label_fixed_protocol",
        dataset_name=DATASET_NAME,
        dataset_doi=DATASET_DOI,
        dataset_item_uuid=DATASET_ITEM_UUID,
        dataset_license=DATASET_LICENSE,
        archive_name=ARCHIVE_NAME,
        archive_bitstream_uuid=ARCHIVE_BITSTREAM_UUID,
        archive_verification=verification,
        scan_members=SCAN_MEMBERS,
        label_member=LABEL_MEMBER,
        scan_order="integer suffix 0 through 30",
        pair_universe=(
            "all source<target scan pairs with target-source>1; no overlap, "
            "pose, prediction-quality, or label filtering"
        ),
        expected_pair_count=EXPECTED_PAIR_COUNT,
        pipeline_name="open3d_0.19.0_fpfh_fast_global_registration",
        pipeline_parameters=phase37_parameters(),
        prediction_matrix_convention=(
            "for pair (source,target), the stored matrix maps target-index "
            "local coordinates into source-index local coordinates"
        ),
        correctness_rule=(
            "strict conjunction: relative rotation error < 15 degrees and "
            "relative translation error < 0.30 meters"
        ),
        maximum_relative_rotation_error_degrees=(
            MAX_RELATIVE_ROTATION_ERROR_DEGREES
        ),
        maximum_relative_translation_error_meters=(
            MAX_RELATIVE_TRANSLATION_ERROR_METERS
        ),
        rotation_percentile_cutoff=ROTATION_PERCENTILE_CUTOFF,
        minimum_correct_retention=MINIMUM_CORRECT_RETENTION,
        minimum_incorrect_rejection=MINIMUM_INCORRECT_REJECTION,
        panel_gate=(
            "base contains correct and incorrect predictions; guarded "
            "precision improves; correct retention >= 0.90; incorrect "
            "rejection >= 0.10"
        ),
        label_boundary=(
            "the outer archive was downloaded, byte-hashed, and its ZIP "
            "central-directory member names were enumerated; no label member "
            "was opened, decompressed, decoded, or numerically inspected. "
            "The prediction generator may open only the 31 frozen Hokuyo "
            "members and must finish the complete prediction artifact before "
            "a separate evaluator opens the frozen Leica label member"
        ),
        label_values_accessed=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("benchmark-data/eth_mountain_plain") / ARCHIVE_NAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/fresh_external_protocol_phase38.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preregister_phase38(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
