"""Audit S3DIS metadata and extract calibration-only wall--board instances."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .s3dis_two_layer_intake_protocol import (
    ARCHIVE_NAME,
    CALIBRATION_AREAS,
    RESERVED_HELD_OUT_AREA,
    TARGET_CLASSES,
)

RESULT_SCHEMA = "pftf_alpha_s3dis_two_layer_intake_phase51/v1"
EXPECTED_AREAS = tuple(f"Area_{index}" for index in range(1, 7))


@dataclass(frozen=True)
class S3DISArchiveInventory:
    archive_path: str
    archive_name: str
    archive_bytes: int
    archive_sha256: str
    member_metadata_sha256: str
    member_count: int
    area_member_counts: dict[str, int]
    target_member_counts: dict[str, dict[str, int]]
    calibration_target_member_count: int
    reserved_target_member_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class S3DISTwoLayerIntakeResult:
    artifact_schema: str
    role: str
    inventory: S3DISArchiveInventory
    calibration_areas: tuple[str, ...]
    reserved_held_out_area: str
    extracted_root: str
    extracted_member_count: int
    extracted_uncompressed_bytes: int
    reserved_content_opened: bool
    extraction_rule: str
    external_archive_intake_supported: bool
    real_scan_supported: bool
    held_out_validation_supported: bool
    pftf_superiority_supported: bool
    deployment_supported: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["inventory"] = self.inventory.to_dict()
        payload["calibration_areas"] = list(self.calibration_areas)
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {name}")
    return path.parts


def _area_from_parts(parts: tuple[str, ...]) -> str | None:
    matches = [part for part in parts if part in EXPECTED_AREAS]
    if len(matches) > 1:
        raise ValueError(f"ambiguous area in ZIP member: {'/'.join(parts)}")
    return matches[0] if matches else None


def _target_class(parts: tuple[str, ...]) -> str | None:
    if "Annotations" not in parts or not parts:
        return None
    name = PurePosixPath(parts[-1])
    if name.suffix.lower() != ".txt":
        return None
    prefix = name.stem.split("_", maxsplit=1)[0].lower()
    return prefix if prefix in TARGET_CLASSES else None


def _metadata_digest(infos: list[zipfile.ZipInfo]) -> str:
    digest = hashlib.sha256()
    for info in sorted(infos, key=lambda item: item.filename):
        row = (
            f"{info.filename}\0{info.compress_size}\0{info.file_size}\n"
        ).encode()
        digest.update(row)
    return digest.hexdigest()


def audit_s3dis_archive(path: str | Path) -> S3DISArchiveInventory:
    """Read ZIP metadata only and report wall/board member counts by area."""

    archive = Path(path)
    if archive.name != ARCHIVE_NAME:
        raise ValueError(f"expected archive name {ARCHIVE_NAME!r}")
    area_member_counts = {area: 0 for area in EXPECTED_AREAS}
    target_member_counts = {
        area: {target: 0 for target in TARGET_CLASSES} for area in EXPECTED_AREAS
    }
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        for info in infos:
            parts = _safe_parts(info.filename)
            area = _area_from_parts(parts)
            if area is None:
                continue
            area_member_counts[area] += 1
            target = _target_class(parts)
            if target is not None and not info.is_dir():
                target_member_counts[area][target] += 1
    missing = [area for area, count in area_member_counts.items() if count == 0]
    if missing:
        raise ValueError(f"archive is missing S3DIS areas: {missing}")
    calibration_count = sum(
        target_member_counts[area][target]
        for area in CALIBRATION_AREAS
        for target in TARGET_CLASSES
    )
    reserved_count = sum(
        target_member_counts[RESERVED_HELD_OUT_AREA][target]
        for target in TARGET_CLASSES
    )
    return S3DISArchiveInventory(
        archive_path=str(archive.resolve()),
        archive_name=archive.name,
        archive_bytes=archive.stat().st_size,
        archive_sha256=_sha256(archive),
        member_metadata_sha256=_metadata_digest(infos),
        member_count=len(infos),
        area_member_counts=area_member_counts,
        target_member_counts=target_member_counts,
        calibration_target_member_count=calibration_count,
        reserved_target_member_count=reserved_count,
    )


def _calibration_targets(bundle: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    selected: list[zipfile.ZipInfo] = []
    for info in bundle.infolist():
        parts = _safe_parts(info.filename)
        area = _area_from_parts(parts)
        if area == RESERVED_HELD_OUT_AREA and _target_class(parts) is not None:
            continue
        if (
            area in CALIBRATION_AREAS
            and _target_class(parts) is not None
            and not info.is_dir()
        ):
            selected.append(info)
    return sorted(selected, key=lambda info: info.filename)


def extract_calibration_targets(
    archive_path: str | Path,
    output_root: str | Path,
) -> tuple[int, int]:
    """Extract only calibration wall/board annotations, never Area 5 contents."""

    archive = Path(archive_path)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted_count = 0
    extracted_bytes = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in _calibration_targets(bundle):
            parts = _safe_parts(info.filename)
            area = _area_from_parts(parts)
            if area == RESERVED_HELD_OUT_AREA:
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
    return extracted_count, extracted_bytes


def run_s3dis_two_layer_intake(
    archive_path: str | Path,
    extraction_root: str | Path,
) -> S3DISTwoLayerIntakeResult:
    inventory = audit_s3dis_archive(archive_path)
    extracted_count, extracted_bytes = extract_calibration_targets(
        archive_path,
        extraction_root,
    )
    intake_supported = bool(
        inventory.calibration_target_member_count > 0
        and inventory.reserved_target_member_count > 0
        and extracted_count == inventory.calibration_target_member_count
    )
    return S3DISTwoLayerIntakeResult(
        artifact_schema=RESULT_SCHEMA,
        role="metadata_audit_and_calibration_only_extraction",
        inventory=inventory,
        calibration_areas=CALIBRATION_AREAS,
        reserved_held_out_area=RESERVED_HELD_OUT_AREA,
        extracted_root=str(Path(extraction_root).resolve()),
        extracted_member_count=extracted_count,
        extracted_uncompressed_bytes=extracted_bytes,
        reserved_content_opened=False,
        extraction_rule=(
            "open only calibration-area Annotations/board*.txt and "
            "Annotations/wall*.txt members; never open an Area-5 member"
        ),
        external_archive_intake_supported=intake_supported,
        real_scan_supported=False,
        held_out_validation_supported=False,
        pftf_superiority_supported=False,
        deployment_supported=False,
        claim_boundary=(
            "a true intake flag verifies archive structure and leakage-controlled "
            "calibration extraction only; it is not a real-scan efficacy result"
        ),
    )


def write_result(result: S3DISTwoLayerIntakeResult, path: str | Path) -> str:
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
        default=Path("benchmark-out/s3dis_two_layer_intake_phase51.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_s3dis_two_layer_intake(args.archive, args.extract_root)
    digest = write_result(result, args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(
        "external_archive_intake_supported="
        f"{str(result.external_archive_intake_supported).lower()} "
        f"calibration_members={result.extracted_member_count} "
        f"reserved_metadata_members={result.inventory.reserved_target_member_count} "
        f"reserved_content_opened={str(result.reserved_content_opened).lower()}"
    )


if __name__ == "__main__":
    main()
