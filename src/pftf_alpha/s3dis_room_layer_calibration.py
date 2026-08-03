"""Calibration-only geometry audit for S3DIS floor--ceiling room pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .s3dis_two_layer_calibration import (
    DEFAULT_MAX_FIT_POINTS,
    DEFAULT_MAX_SPACING_POINTS,
    PlaneSummary,
    load_xyz,
    summarize_plane,
)
from .s3dis_two_layer_intake_protocol import (
    CALIBRATION_AREAS,
    RESERVED_HELD_OUT_AREA,
)

RESULT_SCHEMA = "pftf_alpha_s3dis_room_layer_calibration_phase51b/v1"


@dataclass(frozen=True)
class RoomLayerPairCalibration:
    area: str
    room: str
    annotation_path: str
    floor_paths: tuple[str, ...]
    ceiling_paths: tuple[str, ...]
    floor: PlaneSummary
    ceiling: PlaneSummary
    normal_angle_degrees: float
    plane_gap: float
    gap_to_median_spacing: float
    gap_to_residual_snr: float
    bbox_overlap_fraction: float
    common_footprint_extent_u: float
    common_footprint_extent_v: float
    floor_points_in_common_footprint: int
    ceiling_points_in_common_footprint: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["floor_paths"] = list(self.floor_paths)
        payload["ceiling_paths"] = list(self.ceiling_paths)
        payload["floor"] = self.floor.to_dict()
        payload["ceiling"] = self.ceiling.to_dict()
        return payload


@dataclass(frozen=True)
class S3DISRoomLayerCalibrationResult:
    artifact_schema: str
    role: str
    calibration_root: str
    calibration_areas: tuple[str, ...]
    reserved_held_out_area: str
    reserved_content_opened: bool
    annotation_directory_count: int
    missing_floor_count: int
    missing_ceiling_count: int
    paired_room_count: int
    pairs: tuple[RoomLayerPairCalibration, ...]
    calibration_geometry_audit_supported: bool
    eligibility_frozen: bool
    real_scan_supported: bool
    held_out_validation_supported: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["calibration_areas"] = list(self.calibration_areas)
        payload["pairs"] = [pair.to_dict() for pair in self.pairs]
        return payload


def _load_merged(paths: list[Path]) -> np.ndarray:
    if not paths:
        raise ValueError("cannot merge an empty point-file set")
    return np.ascontiguousarray(np.vstack([load_xyz(path) for path in paths]))


def _area_and_room(
    annotation_path: Path,
    *,
    allowed_areas: tuple[str, ...] = CALIBRATION_AREAS,
) -> tuple[str, str]:
    areas = [part for part in annotation_path.parts if part.startswith("Area_")]
    if len(areas) != 1:
        raise ValueError(f"cannot identify one area from {annotation_path}")
    area = areas[0]
    if area == RESERVED_HELD_OUT_AREA and area not in allowed_areas:
        raise ValueError("reserved Area 5 content may not enter calibration")
    if area not in allowed_areas:
        raise ValueError(f"area is outside the allowed set: {area}")
    return area, annotation_path.parent.name


def _bbox(points: np.ndarray, center: np.ndarray, axes: np.ndarray) -> np.ndarray:
    uv = (points - center) @ axes
    return np.vstack((np.min(uv, axis=0), np.max(uv, axis=0)))


def _room_pair(
    root: Path,
    annotation_path: Path,
    floor_paths: list[Path],
    ceiling_paths: list[Path],
    *,
    max_fit_points: int,
    max_spacing_points: int,
    allowed_areas: tuple[str, ...] = CALIBRATION_AREAS,
) -> RoomLayerPairCalibration:
    area, room = _area_and_room(annotation_path, allowed_areas=allowed_areas)
    floor_points = _load_merged(floor_paths)
    ceiling_points = _load_merged(ceiling_paths)
    floor = summarize_plane(
        floor_points,
        max_fit_points=max_fit_points,
        max_spacing_points=max_spacing_points,
    )
    ceiling = summarize_plane(
        ceiling_points,
        max_fit_points=max_fit_points,
        max_spacing_points=max_spacing_points,
    )
    floor_normal = np.asarray(floor.normal)
    ceiling_normal = np.asarray(ceiling.normal)
    cosine = float(np.clip(abs(floor_normal @ ceiling_normal), 0.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    floor_center = np.asarray(floor.centroid)
    ceiling_center = np.asarray(ceiling.centroid)
    gap = abs(float((ceiling_center - floor_center) @ floor_normal))
    spacing = max(floor.median_spacing, ceiling.median_spacing, np.finfo(float).eps)
    residual_scale = math.hypot(floor.residual_rms, ceiling.residual_rms)
    axes = np.column_stack((np.asarray(floor.tangent_u), np.asarray(floor.tangent_v)))
    floor_bbox = _bbox(floor_points, floor_center, axes)
    ceiling_bbox = _bbox(ceiling_points, floor_center, axes)
    common_lower = np.maximum(floor_bbox[0], ceiling_bbox[0])
    common_upper = np.minimum(floor_bbox[1], ceiling_bbox[1])
    common_extent = np.maximum(common_upper - common_lower, 0.0)
    floor_extent = np.maximum(floor_bbox[1] - floor_bbox[0], 0.0)
    ceiling_extent = np.maximum(ceiling_bbox[1] - ceiling_bbox[0], 0.0)
    common_area = float(np.prod(common_extent))
    smaller_area = min(float(np.prod(floor_extent)), float(np.prod(ceiling_extent)))
    overlap = common_area / max(smaller_area, np.finfo(float).eps)
    floor_uv = (floor_points - floor_center) @ axes
    ceiling_uv = (ceiling_points - floor_center) @ axes
    floor_inside = np.all(
        (floor_uv >= common_lower) & (floor_uv <= common_upper), axis=1
    )
    ceiling_inside = np.all(
        (ceiling_uv >= common_lower) & (ceiling_uv <= common_upper), axis=1
    )
    return RoomLayerPairCalibration(
        area=area,
        room=room,
        annotation_path=annotation_path.relative_to(root).as_posix(),
        floor_paths=tuple(path.relative_to(root).as_posix() for path in floor_paths),
        ceiling_paths=tuple(
            path.relative_to(root).as_posix() for path in ceiling_paths
        ),
        floor=floor,
        ceiling=ceiling,
        normal_angle_degrees=angle,
        plane_gap=gap,
        gap_to_median_spacing=gap / spacing,
        gap_to_residual_snr=gap / max(residual_scale, np.finfo(float).eps),
        bbox_overlap_fraction=overlap,
        common_footprint_extent_u=float(common_extent[0]),
        common_footprint_extent_v=float(common_extent[1]),
        floor_points_in_common_footprint=int(np.count_nonzero(floor_inside)),
        ceiling_points_in_common_footprint=int(np.count_nonzero(ceiling_inside)),
    )


def evaluate_room_layer_calibration(
    root: str | Path,
    *,
    max_fit_points: int = DEFAULT_MAX_FIT_POINTS,
    max_spacing_points: int = DEFAULT_MAX_SPACING_POINTS,
) -> S3DISRoomLayerCalibrationResult:
    calibration_root = Path(root).resolve()
    annotation_paths = sorted(
        path for path in calibration_root.rglob("Annotations") if path.is_dir()
    )
    pairs: list[RoomLayerPairCalibration] = []
    missing_floor = 0
    missing_ceiling = 0
    for annotation_path in annotation_paths:
        _area_and_room(annotation_path)
        floor_paths = sorted(annotation_path.glob("floor*.txt"))
        ceiling_paths = sorted(annotation_path.glob("ceiling*.txt"))
        if not floor_paths:
            missing_floor += 1
        if not ceiling_paths:
            missing_ceiling += 1
        if not floor_paths or not ceiling_paths:
            continue
        pairs.append(
            _room_pair(
                calibration_root,
                annotation_path,
                floor_paths,
                ceiling_paths,
                max_fit_points=max_fit_points,
                max_spacing_points=max_spacing_points,
            )
        )
    return S3DISRoomLayerCalibrationResult(
        artifact_schema=RESULT_SCHEMA,
        role="calibration_only_floor_ceiling_geometry_audit",
        calibration_root=str(calibration_root),
        calibration_areas=CALIBRATION_AREAS,
        reserved_held_out_area=RESERVED_HELD_OUT_AREA,
        reserved_content_opened=False,
        annotation_directory_count=len(annotation_paths),
        missing_floor_count=missing_floor,
        missing_ceiling_count=missing_ceiling,
        paired_room_count=len(pairs),
        pairs=tuple(pairs),
        calibration_geometry_audit_supported=bool(pairs),
        eligibility_frozen=False,
        real_scan_supported=False,
        held_out_validation_supported=False,
        claim_boundary=(
            "room geometry summaries are calibration-only and cannot support a "
            "real held-out efficacy claim"
        ),
    )


def write_result(result: S3DISRoomLayerCalibrationResult, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    output.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/s3dis_room_layer_calibration_phase51b.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_room_layer_calibration(args.calibration_root)
    digest = write_result(result, args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(
        f"rooms={result.annotation_directory_count} "
        f"paired={result.paired_room_count} "
        f"missing_floor={result.missing_floor_count} "
        f"missing_ceiling={result.missing_ceiling_count} "
        f"reserved_content_opened={str(result.reserved_content_opened).lower()}"
    )


if __name__ == "__main__":
    main()
