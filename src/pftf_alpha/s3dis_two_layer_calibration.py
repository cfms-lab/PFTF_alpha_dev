"""Calibration-only geometry audit for S3DIS wall--board instance pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .s3dis_two_layer_intake_protocol import (
    CALIBRATION_AREAS,
    RESERVED_HELD_OUT_AREA,
)

FloatArray = NDArray[np.float64]
RESULT_SCHEMA = "pftf_alpha_s3dis_two_layer_calibration_phase51/v1"
DEFAULT_MAX_FIT_POINTS = 20_000
DEFAULT_MAX_SPACING_POINTS = 4_096
PARALLEL_PREFILTER_DEGREES = 30.0


@dataclass(frozen=True)
class PlaneSummary:
    point_count: int
    fit_point_count: int
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]
    tangent_u: tuple[float, float, float]
    tangent_v: tuple[float, float, float]
    residual_rms: float
    residual_p95: float
    median_spacing: float
    tangent_extent_u: float
    tangent_extent_v: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WallCandidateSummary:
    wall_path: str
    normal_angle_degrees: float
    plane_gap: float
    board_diagonal: float
    gap_to_board_diagonal: float
    gap_to_median_spacing: float
    wall_points_in_board_footprint: int
    board_footprint_grid_support: float
    pairing_prefilter_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BoardPairCalibration:
    area: str
    room: str
    board_path: str
    board: PlaneSummary
    wall_candidate_count: int
    selected_wall_path: str | None
    selected_wall: PlaneSummary | None
    selected_pair: WallCandidateSummary | None
    candidates: tuple[WallCandidateSummary, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["board"] = self.board.to_dict()
        payload["selected_wall"] = (
            None if self.selected_wall is None else self.selected_wall.to_dict()
        )
        payload["selected_pair"] = (
            None if self.selected_pair is None else self.selected_pair.to_dict()
        )
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload


@dataclass(frozen=True)
class S3DISTwoLayerCalibrationResult:
    artifact_schema: str
    role: str
    calibration_root: str
    calibration_areas: tuple[str, ...]
    reserved_held_out_area: str
    reserved_content_opened: bool
    max_fit_points: int
    max_spacing_points: int
    parallel_prefilter_degrees: float
    board_instance_count: int
    paired_board_count: int
    pairs: tuple[BoardPairCalibration, ...]
    calibration_geometry_audit_supported: bool
    pair_eligibility_frozen: bool
    real_scan_supported: bool
    held_out_validation_supported: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["calibration_areas"] = list(self.calibration_areas)
        payload["pairs"] = [pair.to_dict() for pair in self.pairs]
        return payload


def load_xyz(path: str | Path) -> FloatArray:
    points = np.loadtxt(path, dtype=np.float64, usecols=(0, 1, 2), ndmin=2)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError(f"point file has invalid shape: {path}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"point file has non-finite coordinates: {path}")
    return np.ascontiguousarray(points)


def _even_sample(points: FloatArray, maximum: int) -> FloatArray:
    if points.shape[0] <= maximum:
        return points
    indices = np.linspace(0, points.shape[0] - 1, maximum, dtype=np.int64)
    return np.ascontiguousarray(points[indices])


def summarize_plane(
    points: FloatArray,
    *,
    max_fit_points: int = DEFAULT_MAX_FIT_POINTS,
    max_spacing_points: int = DEFAULT_MAX_SPACING_POINTS,
) -> PlaneSummary:
    fit = _even_sample(np.asarray(points, dtype=np.float64), max_fit_points)
    center = np.mean(fit, axis=0)
    centered = fit - center
    covariance = centered.T @ centered / fit.shape[0]
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    tangent_u = vectors[:, order[0]]
    tangent_v = vectors[:, order[1]]
    normal = vectors[:, order[2]]
    residuals = np.abs(centered @ normal)
    spacing_points = _even_sample(fit, max_spacing_points)
    distances, _ = cKDTree(spacing_points).query(spacing_points, k=2, workers=1)
    median_spacing = float(np.median(distances[:, 1]))
    uv = np.column_stack((centered @ tangent_u, centered @ tangent_v))
    extents = np.ptp(uv, axis=0)
    return PlaneSummary(
        point_count=int(points.shape[0]),
        fit_point_count=int(fit.shape[0]),
        centroid=tuple(float(value) for value in center),
        normal=tuple(float(value) for value in normal),
        tangent_u=tuple(float(value) for value in tangent_u),
        tangent_v=tuple(float(value) for value in tangent_v),
        residual_rms=float(np.sqrt(np.mean(residuals**2))),
        residual_p95=float(np.quantile(residuals, 0.95)),
        median_spacing=median_spacing,
        tangent_extent_u=float(extents[0]),
        tangent_extent_v=float(extents[1]),
    )


def _grid_support(board_uv: FloatArray, wall_uv: FloatArray, bins: int = 8) -> float:
    lower = np.min(board_uv, axis=0)
    upper = np.max(board_uv, axis=0)
    extent = upper - lower
    if np.any(extent <= 0.0) or wall_uv.shape[0] == 0:
        return 0.0
    scaled = (wall_uv - lower) / extent
    inside = np.all((scaled >= 0.0) & (scaled <= 1.0), axis=1)
    if not np.any(inside):
        return 0.0
    cells = np.minimum((scaled[inside] * bins).astype(np.int64), bins - 1)
    occupied = np.unique(cells[:, 0] * bins + cells[:, 1]).size
    return float(occupied / (bins * bins))


def compare_wall_candidate(
    board_points: FloatArray,
    board: PlaneSummary,
    wall_points: FloatArray,
    wall: PlaneSummary,
    wall_path: str,
) -> WallCandidateSummary:
    board_normal = np.asarray(board.normal)
    wall_normal = np.asarray(wall.normal)
    cosine = float(np.clip(abs(board_normal @ wall_normal), 0.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    board_center = np.asarray(board.centroid)
    wall_center = np.asarray(wall.centroid)
    gap = abs(float((board_center - wall_center) @ wall_normal))
    tangent_u = np.asarray(board.tangent_u)
    tangent_v = np.asarray(board.tangent_v)
    board_centered = board_points - board_center
    wall_centered = wall_points - board_center
    board_uv = np.column_stack(
        (board_centered @ tangent_u, board_centered @ tangent_v)
    )
    wall_uv = np.column_stack(
        (wall_centered @ tangent_u, wall_centered @ tangent_v)
    )
    lower = np.min(board_uv, axis=0)
    upper = np.max(board_uv, axis=0)
    inside = np.all((wall_uv >= lower) & (wall_uv <= upper), axis=1)
    diagonal = math.hypot(board.tangent_extent_u, board.tangent_extent_v)
    spacing = max(board.median_spacing, wall.median_spacing, np.finfo(float).eps)
    return WallCandidateSummary(
        wall_path=wall_path,
        normal_angle_degrees=angle,
        plane_gap=gap,
        board_diagonal=diagonal,
        gap_to_board_diagonal=gap / max(diagonal, np.finfo(float).eps),
        gap_to_median_spacing=gap / spacing,
        wall_points_in_board_footprint=int(np.count_nonzero(inside)),
        board_footprint_grid_support=_grid_support(board_uv, wall_uv),
        pairing_prefilter_passed=bool(angle <= PARALLEL_PREFILTER_DEGREES),
    )


def _area_and_room(path: Path) -> tuple[str, str]:
    areas = [part for part in path.parts if part.startswith("Area_")]
    if len(areas) != 1:
        raise ValueError(f"cannot identify one area from {path}")
    area = areas[0]
    if area == RESERVED_HELD_OUT_AREA:
        raise ValueError("reserved Area 5 content may not enter calibration")
    if area not in CALIBRATION_AREAS:
        raise ValueError(f"unexpected calibration area: {area}")
    annotation_index = path.parts.index("Annotations")
    return area, path.parts[annotation_index - 1]


def evaluate_calibration_root(
    root: str | Path,
    *,
    max_fit_points: int = DEFAULT_MAX_FIT_POINTS,
    max_spacing_points: int = DEFAULT_MAX_SPACING_POINTS,
) -> S3DISTwoLayerCalibrationResult:
    calibration_root = Path(root).resolve()
    board_paths = sorted(calibration_root.rglob("Annotations/board*.txt"))
    pairs: list[BoardPairCalibration] = []
    for board_path in board_paths:
        area, room = _area_and_room(board_path)
        wall_paths = sorted(board_path.parent.glob("wall*.txt"))
        board_points = load_xyz(board_path)
        board_summary = summarize_plane(
            board_points,
            max_fit_points=max_fit_points,
            max_spacing_points=max_spacing_points,
        )
        wall_rows: list[tuple[WallCandidateSummary, PlaneSummary]] = []
        for wall_path in wall_paths:
            wall_points = load_xyz(wall_path)
            wall_summary = summarize_plane(
                wall_points,
                max_fit_points=max_fit_points,
                max_spacing_points=max_spacing_points,
            )
            relative = wall_path.relative_to(calibration_root).as_posix()
            comparison = compare_wall_candidate(
                board_points,
                board_summary,
                wall_points,
                wall_summary,
                relative,
            )
            wall_rows.append((comparison, wall_summary))
        eligible = [row for row in wall_rows if row[0].pairing_prefilter_passed]
        selected_pool = eligible if eligible else wall_rows
        selected = min(
            selected_pool,
            key=lambda row: (row[0].plane_gap, row[0].normal_angle_degrees),
            default=None,
        )
        pairs.append(
            BoardPairCalibration(
                area=area,
                room=room,
                board_path=board_path.relative_to(calibration_root).as_posix(),
                board=board_summary,
                wall_candidate_count=len(wall_rows),
                selected_wall_path=None if selected is None else selected[0].wall_path,
                selected_wall=None if selected is None else selected[1],
                selected_pair=None if selected is None else selected[0],
                candidates=tuple(row[0] for row in wall_rows),
            )
        )
    paired_count = sum(pair.selected_pair is not None for pair in pairs)
    return S3DISTwoLayerCalibrationResult(
        artifact_schema=RESULT_SCHEMA,
        role="calibration_only_wall_board_geometry_audit",
        calibration_root=str(calibration_root),
        calibration_areas=CALIBRATION_AREAS,
        reserved_held_out_area=RESERVED_HELD_OUT_AREA,
        reserved_content_opened=False,
        max_fit_points=max_fit_points,
        max_spacing_points=max_spacing_points,
        parallel_prefilter_degrees=PARALLEL_PREFILTER_DEGREES,
        board_instance_count=len(pairs),
        paired_board_count=paired_count,
        pairs=tuple(pairs),
        calibration_geometry_audit_supported=bool(pairs and paired_count == len(pairs)),
        pair_eligibility_frozen=False,
        real_scan_supported=False,
        held_out_validation_supported=False,
        claim_boundary=(
            "calibration summaries may define a later frozen eligibility rule; "
            "they are not held-out efficacy evidence"
        ),
    )


def write_result(result: S3DISTwoLayerCalibrationResult, path: str | Path) -> str:
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
        default=Path("benchmark-out/s3dis_two_layer_calibration_phase51.json"),
    )
    parser.add_argument("--max-fit-points", type=int, default=DEFAULT_MAX_FIT_POINTS)
    parser.add_argument(
        "--max-spacing-points", type=int, default=DEFAULT_MAX_SPACING_POINTS
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_calibration_root(
        args.calibration_root,
        max_fit_points=args.max_fit_points,
        max_spacing_points=args.max_spacing_points,
    )
    digest = write_result(result, args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(
        f"boards={result.board_instance_count} paired={result.paired_board_count} "
        f"reserved_content_opened={str(result.reserved_content_opened).lower()}"
    )


if __name__ == "__main__":
    main()
