"""Calibrate Phase-42 anchor-relative evidence on opened Gazebo sources."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .eth_gazebo_anchor_relative import (
    AnchorRelativeEndpoint,
    evaluate_anchor_relative,
    evaluate_phase41_baseline,
)
from .eth_gazebo_local_support import (
    evaluate_anchor_and_scan_baselines,
    load_hokuyo_scans,
    prepare_source_inputs,
)
from .eth_gazebo_local_support_calibration import (
    FSCORE_TOLERANCE,
    RECALL_TOLERANCE,
)
from .eth_gazebo_reconstruction_protocol import (
    DECISION_SCHEMA,
    EXPECTED_DECISION_SHA256,
    EXPECTED_PREDICTION_SHA256,
    PREDICTION_SCHEMA,
    SOURCE_VOXEL_METERS,
)
from .eth_gazebo_reconstruction_shadow import (
    ReconstructionEndpoint,
    _load_locked_json,
    _voxel_downsample,
)
from .eth_gazebo_validation_protocol import (
    ARCHIVE_NAME,
    verify_gazebo_archive_directory,
)
from .eth_open3d_fgr_pipeline import _load_open3d

DEVELOPMENT_SOURCE_INDICES = (0, 17)
CALIBRATION_SEED = 20_260_807
MINIMUM_ADDED_CELL_COUNT = 3
CANDIDATE_GRID = tuple(
    (nearest, plane, alignment)
    for nearest in (0.75, 1.00, 1.50)
    for plane in (0.15, 0.30, 0.50)
    for alignment in (0.0, 0.75)
)


def candidate_id(nearest: float, plane: float, alignment: float) -> str:
    return (
        f"anchor_d{round(100 * nearest):03d}_p{round(100 * plane):03d}_"
        f"n{round(100 * alignment):03d}"
    )


@dataclass(frozen=True)
class AnchorRelativeDevelopmentCase:
    source_index: int
    anchor_baseline: ReconstructionEndpoint
    phase41_baseline: ReconstructionEndpoint

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["anchor_baseline"] = self.anchor_baseline.to_dict()
        payload["phase41_baseline"] = self.phase41_baseline.to_dict()
        return payload


@dataclass(frozen=True)
class AnchorRelativeCandidateCase:
    source_index: int
    added_cell_count: int
    endpoint: ReconstructionEndpoint

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["endpoint"] = self.endpoint.to_dict()
        return payload


@dataclass(frozen=True)
class AnchorRelativeCalibrationPoint:
    candidate_id: str
    maximum_nearest_anchor_distance_meters: float
    maximum_anchor_plane_residual_meters: float
    minimum_normal_alignment: float
    cases: tuple[AnchorRelativeCandidateCase, ...]
    mean_geometry_loss: float
    mean_fscore: float
    mean_recall: float
    every_case_exercised: bool
    mean_geometry_beats_both_baselines: bool
    mean_fscore_within_anchor_tolerance: bool
    mean_recall_within_anchor_tolerance: bool
    eligible: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["cases"] = [case.to_dict() for case in self.cases]
        return payload


@dataclass(frozen=True)
class AnchorRelativeCalibration:
    artifact_schema: str
    role: str
    archive_path: str
    archive_sha256: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    decision_artifact_path: str
    decision_artifact_sha256: str
    development_source_indices: tuple[int, ...]
    development_references_previously_opened: bool
    candidate_grid: tuple[tuple[float, float, float], ...]
    minimum_added_cell_count: int
    fscore_tolerance: float
    recall_tolerance: float
    baseline_cases: tuple[AnchorRelativeDevelopmentCase, ...]
    mean_anchor_geometry_loss: float
    mean_phase41_geometry_loss: float
    mean_anchor_fscore: float
    mean_anchor_recall: float
    candidates: tuple[AnchorRelativeCalibrationPoint, ...]
    selected_candidate_id: str | None
    selected_maximum_nearest_anchor_distance_meters: float | None
    selected_maximum_anchor_plane_residual_meters: float | None
    selected_minimum_normal_alignment: float | None
    calibration_viable: bool
    topology_used_for_selection: bool
    unopened_validation_references_accessed: bool
    registration_label_values_accessed: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_grid"] = [list(row) for row in self.candidate_grid]
        payload["baseline_cases"] = [
            case.to_dict() for case in self.baseline_cases
        ]
        payload["candidates"] = [row.to_dict() for row in self.candidates]
        return payload


def calibrate_anchor_relative(
    archive_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
) -> AnchorRelativeCalibration:
    archive = Path(archive_path)
    prediction_file = Path(prediction_path)
    decision_file = Path(decision_path)
    verification = verify_gazebo_archive_directory(archive)
    predictions = _load_locked_json(
        prediction_file,
        expected_sha256=EXPECTED_PREDICTION_SHA256,
        expected_schema=PREDICTION_SCHEMA,
    )
    decisions = _load_locked_json(
        decision_file,
        expected_sha256=EXPECTED_DECISION_SHA256,
        expected_schema=DECISION_SCHEMA,
    )
    if predictions.get("validation_label_member_opened") is not False:
        raise ValueError("prediction artifact crossed the registration label boundary")
    if decisions.get("validation_label_values_accessed") is not False:
        raise ValueError("decision artifact crossed the registration label boundary")
    raw_predictions = predictions.get("predictions")
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_predictions, list) or not isinstance(raw_decisions, list):
        raise ValueError("Gazebo prediction or decision rows are missing")
    prediction_by_source: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    accept_by_pair: dict[tuple[int, int], bool] = {}
    for prediction, decision in zip(raw_predictions, raw_decisions, strict=True):
        pair = (
            int(prediction["source_index"]),
            int(prediction["target_index"]),
        )
        decision_pair = (
            int(decision["source_index"]),
            int(decision["target_index"]),
        )
        if pair != decision_pair:
            raise ValueError("Gazebo prediction and decision pair order differs")
        prediction_by_source[pair[0]].append(prediction)
        accept_by_pair[pair] = bool(decision["guarded_accept"])
    o3d = _load_open3d()
    raw_scans, _ = load_hokuyo_scans(archive)
    downsampled_scans = tuple(
        _voxel_downsample(o3d, points, SOURCE_VOXEL_METERS)
        for points in raw_scans
    )
    inputs_by_source = {
        source: prepare_source_inputs(
            o3d,
            raw_scans,
            downsampled_scans,
            prediction_by_source[source],
            accept_by_pair,
            source_index=source,
        )
        for source in DEVELOPMENT_SOURCE_INDICES
    }
    baselines: list[AnchorRelativeDevelopmentCase] = []
    for source, inputs in inputs_by_source.items():
        seed = CALIBRATION_SEED + source
        anchor, _, _ = evaluate_anchor_and_scan_baselines(
            o3d,
            inputs,
            seed=seed,
        )
        phase41 = evaluate_phase41_baseline(o3d, inputs, seed=seed)
        baselines.append(
            AnchorRelativeDevelopmentCase(
                source_index=source,
                anchor_baseline=anchor,
                phase41_baseline=phase41.endpoint,
            )
        )
    mean_anchor_geometry = float(
        np.mean([case.anchor_baseline.geometry_loss for case in baselines])
    )
    mean_phase41_geometry = float(
        np.mean([case.phase41_baseline.geometry_loss for case in baselines])
    )
    mean_anchor_fscore = float(
        np.mean([case.anchor_baseline.fscore for case in baselines])
    )
    mean_anchor_recall = float(
        np.mean([case.anchor_baseline.recall for case in baselines])
    )
    rows: list[AnchorRelativeCalibrationPoint] = []
    for nearest, plane, alignment in CANDIDATE_GRID:
        case_rows: list[AnchorRelativeCandidateCase] = []
        for source, inputs in inputs_by_source.items():
            result: AnchorRelativeEndpoint = evaluate_anchor_relative(
                o3d,
                inputs,
                maximum_nearest_anchor_distance_meters=nearest,
                maximum_anchor_plane_residual_meters=plane,
                minimum_normal_alignment=alignment,
                seed=CALIBRATION_SEED + source,
            )
            case_rows.append(
                AnchorRelativeCandidateCase(
                    source_index=source,
                    added_cell_count=result.route.anchor_relative_cell_count,
                    endpoint=result.endpoint,
                )
            )
        mean_geometry = float(
            np.mean([case.endpoint.geometry_loss for case in case_rows])
        )
        mean_fscore = float(np.mean([case.endpoint.fscore for case in case_rows]))
        mean_recall = float(np.mean([case.endpoint.recall for case in case_rows]))
        exercised = all(
            case.added_cell_count >= MINIMUM_ADDED_CELL_COUNT for case in case_rows
        )
        geometry_beats = mean_geometry < min(
            mean_anchor_geometry,
            mean_phase41_geometry,
        )
        fscore_close = mean_fscore >= mean_anchor_fscore - FSCORE_TOLERANCE
        recall_close = mean_recall >= mean_anchor_recall - RECALL_TOLERANCE
        rows.append(
            AnchorRelativeCalibrationPoint(
                candidate_id=candidate_id(nearest, plane, alignment),
                maximum_nearest_anchor_distance_meters=nearest,
                maximum_anchor_plane_residual_meters=plane,
                minimum_normal_alignment=alignment,
                cases=tuple(case_rows),
                mean_geometry_loss=mean_geometry,
                mean_fscore=mean_fscore,
                mean_recall=mean_recall,
                every_case_exercised=exercised,
                mean_geometry_beats_both_baselines=geometry_beats,
                mean_fscore_within_anchor_tolerance=fscore_close,
                mean_recall_within_anchor_tolerance=recall_close,
                eligible=exercised
                and geometry_beats
                and fscore_close
                and recall_close,
            )
        )
    eligible = [row for row in rows if row.eligible]
    selected = (
        None
        if not eligible
        else min(
            eligible,
            key=lambda row: (
                row.mean_geometry_loss,
                -row.minimum_normal_alignment,
                row.maximum_nearest_anchor_distance_meters,
                row.maximum_anchor_plane_residual_meters,
            ),
        )
    )
    return AnchorRelativeCalibration(
        artifact_schema="pftf_alpha_eth_gazebo_anchor_relative_calibration_phase42/v1",
        role="opened_failure_anchor_relative_calibration",
        archive_path=str(archive),
        archive_sha256=verification.sha256,
        prediction_artifact_path=str(prediction_file),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        decision_artifact_path=str(decision_file),
        decision_artifact_sha256=EXPECTED_DECISION_SHA256,
        development_source_indices=DEVELOPMENT_SOURCE_INDICES,
        development_references_previously_opened=True,
        candidate_grid=CANDIDATE_GRID,
        minimum_added_cell_count=MINIMUM_ADDED_CELL_COUNT,
        fscore_tolerance=FSCORE_TOLERANCE,
        recall_tolerance=RECALL_TOLERANCE,
        baseline_cases=tuple(baselines),
        mean_anchor_geometry_loss=mean_anchor_geometry,
        mean_phase41_geometry_loss=mean_phase41_geometry,
        mean_anchor_fscore=mean_anchor_fscore,
        mean_anchor_recall=mean_anchor_recall,
        candidates=tuple(rows),
        selected_candidate_id=(None if selected is None else selected.candidate_id),
        selected_maximum_nearest_anchor_distance_meters=(
            None
            if selected is None
            else selected.maximum_nearest_anchor_distance_meters
        ),
        selected_maximum_anchor_plane_residual_meters=(
            None
            if selected is None
            else selected.maximum_anchor_plane_residual_meters
        ),
        selected_minimum_normal_alignment=(
            None if selected is None else selected.minimum_normal_alignment
        ),
        calibration_viable=selected is not None,
        topology_used_for_selection=False,
        unopened_validation_references_accessed=False,
        registration_label_values_accessed=False,
        claim_boundary=(
            "Selection uses only opened sources 0 and 17. Sources 25-27 remain "
            "endpoint-unopened. Anchor distance, plane residual, and unsigned PCA "
            "normal alignment are observed-only filters layered after Phase 41; "
            "this is not a spatially varying alpha complex."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("benchmark-data/eth_gazebo_summer") / ARCHIVE_NAME,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_predictions_phase39.json"),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_rotation_decisions_phase39.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_anchor_relative_calibration_phase42.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = calibrate_anchor_relative(
        args.archive,
        args.predictions,
        args.decisions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
