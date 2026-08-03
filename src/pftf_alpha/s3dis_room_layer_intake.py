"""Extract calibration-only S3DIS floor and ceiling annotation members."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .s3dis_room_layer_protocol import TARGET_CLASSES
from .s3dis_two_layer_intake import EXPECTED_AREAS, _sha256
from .s3dis_two_layer_intake_protocol import (
    ARCHIVE_NAME,
    CALIBRATION_AREAS,
    RESERVED_HELD_OUT_AREA,
)

RESULT_SCHEMA = "pftf_alpha_s3dis_room_layer_intake_phase51b/v1"


@dataclass(frozen=True)
class S3DISRoomLayerIntakeResult:
    artifact_schema: str
    role: str
    archive_path: str
    archive_bytes: int
    archive_sha256: str
    calibration_areas: tuple[str, ...]
    reserved_held_out_area: str
    target_classes: tuple[str, ...]
    target_member_counts: dict[str, dict[str, int]]
    calibration_target_member_count: int
    reserved_target_member_count: int
    extracted_root: str
    extracted_member_count: int
    extracted_uncompressed_bytes: int
    reserved_content_opened: bool
    floor_ceiling_intake_supported: bool
    real_scan_supported: bool
    held_out_validation_supported: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in ("calibration_areas", "target_classes"):
            payload[name] = list(payload[name])
        return payload


def _parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {name}")
    return path.parts


def _area(parts: tuple[str, ...]) -> str | None:
    matches = [part for part in parts if part in EXPECTED_AREAS]
    if len(matches) > 1:
        raise ValueError(f"ambiguous area in ZIP member: {'/'.join(parts)}")
    return matches[0] if matches else None


def _room_layer(parts: tuple[str, ...]) -> str | None:
    if "Annotations" not in parts or not parts:
        return None
    name = PurePosixPath(parts[-1])
    if name.suffix.lower() != ".txt":
        return None
    prefix = name.stem.split("_", maxsplit=1)[0].lower()
    return prefix if prefix in TARGET_CLASSES else None


def _selected_infos(bundle: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    selected: list[zipfile.ZipInfo] = []
    for info in bundle.infolist():
        parts = _parts(info.filename)
        area = _area(parts)
        if (
            area in CALIBRATION_AREAS
            and _room_layer(parts) is not None
            and not info.is_dir()
        ):
            selected.append(info)
    return sorted(selected, key=lambda info: info.filename)


def run_s3dis_room_layer_intake(
    archive_path: str | Path,
    extraction_root: str | Path,
) -> S3DISRoomLayerIntakeResult:
    archive = Path(archive_path)
    if archive.name != ARCHIVE_NAME:
        raise ValueError(f"expected archive name {ARCHIVE_NAME!r}")
    counts = {
        area: {target: 0 for target in TARGET_CLASSES} for area in EXPECTED_AREAS
    }
    root = Path(extraction_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted_count = 0
    extracted_bytes = 0
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        for info in infos:
            parts = _parts(info.filename)
            area = _area(parts)
            target = _room_layer(parts)
            if area is not None and target is not None and not info.is_dir():
                counts[area][target] += 1
        for info in _selected_infos(bundle):
            parts = _parts(info.filename)
            if _area(parts) == RESERVED_HELD_OUT_AREA:
                raise AssertionError("reserved Area 5 reached extraction path")
            destination = root.joinpath(*parts).resolve()
            if root not in destination.parents:
                raise ValueError(f"unsafe extraction destination: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != info.file_size:
                    raise FileExistsError(
                        f"existing extraction has wrong size: {destination}"
                    )
            else:
                with bundle.open(info) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            extracted_count += 1
            extracted_bytes += info.file_size
    calibration_count = sum(
        counts[area][target]
        for area in CALIBRATION_AREAS
        for target in TARGET_CLASSES
    )
    reserved_count = sum(
        counts[RESERVED_HELD_OUT_AREA][target] for target in TARGET_CLASSES
    )
    intake_supported = bool(
        calibration_count > 0
        and reserved_count > 0
        and extracted_count == calibration_count
    )
    return S3DISRoomLayerIntakeResult(
        artifact_schema=RESULT_SCHEMA,
        role="metadata_audit_and_calibration_only_floor_ceiling_extraction",
        archive_path=str(archive.resolve()),
        archive_bytes=archive.stat().st_size,
        archive_sha256=_sha256(archive),
        calibration_areas=CALIBRATION_AREAS,
        reserved_held_out_area=RESERVED_HELD_OUT_AREA,
        target_classes=TARGET_CLASSES,
        target_member_counts=counts,
        calibration_target_member_count=calibration_count,
        reserved_target_member_count=reserved_count,
        extracted_root=str(root),
        extracted_member_count=extracted_count,
        extracted_uncompressed_bytes=extracted_bytes,
        reserved_content_opened=False,
        floor_ceiling_intake_supported=intake_supported,
        real_scan_supported=False,
        held_out_validation_supported=False,
        claim_boundary=(
            "a true intake flag verifies calibration extraction only; floor--"
            "ceiling efficacy and Area-5 transfer remain untested"
        ),
    )


def write_result(result: S3DISRoomLayerIntakeResult, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    output.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/s3dis_room_layer_intake_phase51b.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_s3dis_room_layer_intake(args.archive, args.extract_root)
    digest = write_result(result, args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(
        "floor_ceiling_intake_supported="
        f"{str(result.floor_ceiling_intake_supported).lower()} "
        f"calibration_members={result.extracted_member_count} "
        f"reserved_metadata_members={result.reserved_target_member_count} "
        f"reserved_content_opened={str(result.reserved_content_opened).lower()}"
    )


if __name__ == "__main__":
    main()
