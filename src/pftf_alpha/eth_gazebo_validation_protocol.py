"""Phase-39 preregistration for untouched ETH Gazebo Summer validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .eth_pipeline_calibration import candidate_parameters
from .fresh_external_protocol import (
    MAX_RELATIVE_ROTATION_ERROR_DEGREES,
    MAX_RELATIVE_TRANSLATION_ERROR_METERS,
)
from .scene_relative_rotation_guard import (
    MINIMUM_CORRECT_RETENTION,
    MINIMUM_INCORRECT_REJECTION,
    ROTATION_PERCENTILE_CUTOFF,
)

SCENE_NAME = "ETH Gazebo Summer"
DATASET_DOI = "https://doi.org/10.3929/ethz-b-000721626"
DATASET_LICENSE = "CC BY 4.0"
ARCHIVE_NAME = "gazebo_summer_04-Aug-2011-16_13_22.zip"
ARCHIVE_BITSTREAM_UUID = "20e4fcd9-42d2-470e-8f90-598690fe65e2"
ARCHIVE_BYTE_COUNT = 1_332_460_435
ARCHIVE_MD5 = "94f59356d881a67d2ce74937133c3246"
ARCHIVE_SHA256 = (
    "614052861d6b599c576209965e504682ca78767ac0d4acc112565cf467acb579"
)
ARCHIVE_ROOT = "gazebo_summer_04-Aug-2011-16_13_22"
SCAN_COUNT = 32
SCAN_MEMBERS = tuple(
    f"{ARCHIVE_ROOT}/csv_local/Hokuyo_{index}.csv"
    for index in range(SCAN_COUNT)
)
LABEL_MEMBER = f"{ARCHIVE_ROOT}/leica/pose_scanner_leica.csv"
EXPECTED_PAIR_COUNT = 465
CALIBRATION_ARTIFACT_SHA256 = (
    "1001b214f6b69be4bfe21bade1a0100a7bb89e357b58796edb00031c077228d5"
)
SELECTED_CANDIDATE_ID = "fgr_icp_v050"
SELECTED_PARAMETERS = candidate_parameters(0.50, use_icp=True)


@dataclass(frozen=True)
class GazeboArchiveDirectoryVerification:
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
class GazeboValidationPreregistration:
    artifact_schema: str
    role: str
    scene_name: str
    dataset_doi: str
    dataset_license: str
    archive_name: str
    archive_bitstream_uuid: str
    archive_verification: GazeboArchiveDirectoryVerification
    scan_members: tuple[str, ...]
    label_member: str
    expected_pair_count: int
    pair_universe: str
    calibration_artifact_path: str
    calibration_artifact_sha256: str
    selected_candidate_id: str
    selected_parameters: dict[str, object]
    correctness_rule: str
    rotation_percentile_cutoff: float
    minimum_correct_retention: float
    minimum_incorrect_rejection: float
    label_boundary: str
    validation_label_values_accessed: bool

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_gazebo_archive_directory(
    archive_path: str | Path,
) -> GazeboArchiveDirectoryVerification:
    path = Path(archive_path)
    if path.name != ARCHIVE_NAME:
        raise ValueError("unexpected Gazebo Summer archive name")
    if path.stat().st_size != ARCHIVE_BYTE_COUNT:
        raise ValueError("Gazebo Summer archive byte count mismatch")
    md5, sha256 = _hashes(path)
    if md5 != ARCHIVE_MD5 or sha256 != ARCHIVE_SHA256:
        raise ValueError("Gazebo Summer archive hash mismatch")
    with zipfile.ZipFile(path) as source:
        names = frozenset(item.filename for item in source.infolist())
    if not set(SCAN_MEMBERS).issubset(names):
        raise ValueError("Gazebo Summer scan member set mismatch")
    if LABEL_MEMBER not in names:
        raise ValueError("Gazebo Summer label member name missing")
    return GazeboArchiveDirectoryVerification(
        archive_path=str(path),
        byte_count=ARCHIVE_BYTE_COUNT,
        md5=md5,
        sha256=sha256,
        scan_member_count=SCAN_COUNT,
        scan_members_present=True,
        label_member_name_present=True,
        label_member_content_opened=False,
    )


def _verify_calibration(path: Path) -> Mapping[str, object]:
    if _sha256(path) != CALIBRATION_ARTIFACT_SHA256:
        raise ValueError("Phase-39 calibration artifact SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase-39 calibration artifact must be an object")
    expected = {
        "artifact_schema": "pftf_alpha_eth_pipeline_calibration_phase39/v1",
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_parameters": SELECTED_PARAMETERS,
        "calibration_viable": True,
        "p90_guard_used_for_selection": False,
        "fresh_validation_label_values_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase-39 calibration mismatch: {key}")
    return payload


def preregister_gazebo_validation(
    archive_path: str | Path,
    calibration_path: str | Path,
) -> GazeboValidationPreregistration:
    verification = verify_gazebo_archive_directory(archive_path)
    calibration = Path(calibration_path)
    _verify_calibration(calibration)
    return GazeboValidationPreregistration(
        artifact_schema="pftf_alpha_eth_gazebo_validation_protocol_phase39/v1",
        role="pre_label_frozen_external_validation_protocol",
        scene_name=SCENE_NAME,
        dataset_doi=DATASET_DOI,
        dataset_license=DATASET_LICENSE,
        archive_name=ARCHIVE_NAME,
        archive_bitstream_uuid=ARCHIVE_BITSTREAM_UUID,
        archive_verification=verification,
        scan_members=SCAN_MEMBERS,
        label_member=LABEL_MEMBER,
        expected_pair_count=EXPECTED_PAIR_COUNT,
        pair_universe=(
            "all source<target pairs with target-source>1; no overlap, pose, "
            "fitness, result, or label filtering"
        ),
        calibration_artifact_path=str(calibration),
        calibration_artifact_sha256=CALIBRATION_ARTIFACT_SHA256,
        selected_candidate_id=SELECTED_CANDIDATE_ID,
        selected_parameters=SELECTED_PARAMETERS,
        correctness_rule=(
            f"strict RRE < {MAX_RELATIVE_ROTATION_ERROR_DEGREES:g} degrees "
            f"and strict RTE < {MAX_RELATIVE_TRANSLATION_ERROR_METERS:.2f} m"
        ),
        rotation_percentile_cutoff=ROTATION_PERCENTILE_CUTOFF,
        minimum_correct_retention=MINIMUM_CORRECT_RETENTION,
        minimum_incorrect_rejection=MINIMUM_INCORRECT_REJECTION,
        label_boundary=(
            "the outer archive was downloaded, byte-hashed, and its central "
            "directory names enumerated; no Gazebo pose member was opened or "
            "decoded. The generator may open exactly the 32 Hokuyo members. "
            "Predictions and p90 decisions must be committed before a separate "
            "evaluator may open the frozen Leica pose member"
        ),
        validation_label_values_accessed=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("benchmark-data/eth_gazebo_summer") / ARCHIVE_NAME,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("benchmark-out/eth_pipeline_calibration_phase39.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_validation_protocol_phase39.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preregister_gazebo_validation(args.archive, args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
