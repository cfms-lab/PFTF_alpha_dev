"""Calibrate Phase-41 local support on the opened Gazebo source 0."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .eth_gazebo_local_support import (
    evaluate_anchor_and_scan_baselines,
    evaluate_local_support,
    load_hokuyo_scans,
    prepare_source_inputs,
)
from .eth_gazebo_reconstruction_protocol import (
    ALPHA_METERS,
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

DEVELOPMENT_SOURCE_INDEX = 0
CALIBRATION_SEED = 20_260_805
FSCORE_TOLERANCE = 0.025
RECALL_TOLERANCE = 0.01
CANDIDATE_GRID = tuple(
    (support, dispersion)
    for support in (2, 3, 4)
    for dispersion in (0.15, 0.20, 0.25)
)


def candidate_id(support: int, dispersion: float) -> str:
    return f"support{support:02d}_dispersion{round(1000 * dispersion):04d}mm"


@dataclass(frozen=True)
class LocalSupportCalibrationPoint:
    candidate_id: str
    minimum_support: int
    maximum_dispersion_meters: float
    corroborated_target_only_cell_count: int
    rejected_target_only_cell_count: int
    endpoint: ReconstructionEndpoint
    geometry_beats_both_baselines: bool
    fscore_within_anchor_tolerance: bool
    recall_within_anchor_tolerance: bool
    eligible: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["endpoint"] = self.endpoint.to_dict()
        return payload


@dataclass(frozen=True)
class LocalSupportCalibration:
    artifact_schema: str
    role: str
    archive_path: str
    archive_sha256: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    decision_artifact_path: str
    decision_artifact_sha256: str
    development_source_index: int
    development_reference_previously_opened: bool
    candidate_grid: tuple[tuple[int, float], ...]
    fscore_tolerance: float
    recall_tolerance: float
    alpha_meters: float
    anchor_baseline: ReconstructionEndpoint
    scan_fused_baseline: ReconstructionEndpoint
    candidates: tuple[LocalSupportCalibrationPoint, ...]
    selected_candidate_id: str | None
    selected_minimum_support: int | None
    selected_maximum_dispersion_meters: float | None
    calibration_viable: bool
    topology_used_for_selection: bool
    phase40_validation_references_accessed: bool
    registration_label_values_accessed: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_grid"] = [list(row) for row in self.candidate_grid]
        payload["anchor_baseline"] = self.anchor_baseline.to_dict()
        payload["scan_fused_baseline"] = self.scan_fused_baseline.to_dict()
        payload["candidates"] = [row.to_dict() for row in self.candidates]
        return payload


def calibrate_local_support(
    archive_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
) -> LocalSupportCalibration:
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
    inputs = prepare_source_inputs(
        o3d,
        raw_scans,
        downsampled_scans,
        prediction_by_source[DEVELOPMENT_SOURCE_INDEX],
        accept_by_pair,
        source_index=DEVELOPMENT_SOURCE_INDEX,
    )
    seed = CALIBRATION_SEED
    anchor, scan, _ = evaluate_anchor_and_scan_baselines(o3d, inputs, seed=seed)
    rows: list[LocalSupportCalibrationPoint] = []
    for support, dispersion in CANDIDATE_GRID:
        evaluated = evaluate_local_support(
            o3d,
            inputs,
            minimum_support=support,
            maximum_dispersion_meters=dispersion,
            seed=seed,
        )
        endpoint = evaluated.endpoint
        geometry_beats = endpoint.geometry_loss < min(
            anchor.geometry_loss,
            scan.geometry_loss,
        )
        fscore_close = endpoint.fscore >= anchor.fscore - FSCORE_TOLERANCE
        recall_close = endpoint.recall >= anchor.recall - RECALL_TOLERANCE
        rows.append(
            LocalSupportCalibrationPoint(
                candidate_id=candidate_id(support, dispersion),
                minimum_support=support,
                maximum_dispersion_meters=dispersion,
                corroborated_target_only_cell_count=(
                    evaluated.route.corroborated_target_only_cell_count
                ),
                rejected_target_only_cell_count=(
                    evaluated.route.rejected_target_only_cell_count
                ),
                endpoint=endpoint,
                geometry_beats_both_baselines=geometry_beats,
                fscore_within_anchor_tolerance=fscore_close,
                recall_within_anchor_tolerance=recall_close,
                eligible=(
                    geometry_beats
                    and fscore_close
                    and recall_close
                    and evaluated.route.corroborated_target_only_cell_count > 0
                ),
            )
        )
    eligible = [row for row in rows if row.eligible]
    selected = (
        None
        if not eligible
        else min(
            eligible,
            key=lambda row: (
                row.endpoint.geometry_loss,
                -row.minimum_support,
                row.maximum_dispersion_meters,
            ),
        )
    )
    return LocalSupportCalibration(
        artifact_schema="pftf_alpha_eth_gazebo_local_support_calibration_phase41/v1",
        role="opened_development_source_local_support_calibration",
        archive_path=str(archive),
        archive_sha256=verification.sha256,
        prediction_artifact_path=str(prediction_file),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        decision_artifact_path=str(decision_file),
        decision_artifact_sha256=EXPECTED_DECISION_SHA256,
        development_source_index=DEVELOPMENT_SOURCE_INDEX,
        development_reference_previously_opened=True,
        candidate_grid=CANDIDATE_GRID,
        fscore_tolerance=FSCORE_TOLERANCE,
        recall_tolerance=RECALL_TOLERANCE,
        alpha_meters=ALPHA_METERS,
        anchor_baseline=anchor,
        scan_fused_baseline=scan,
        candidates=tuple(rows),
        selected_candidate_id=(None if selected is None else selected.candidate_id),
        selected_minimum_support=(
            None if selected is None else selected.minimum_support
        ),
        selected_maximum_dispersion_meters=(
            None if selected is None else selected.maximum_dispersion_meters
        ),
        calibration_viable=selected is not None,
        topology_used_for_selection=False,
        phase40_validation_references_accessed=False,
        registration_label_values_accessed=False,
        claim_boundary=(
            "Candidate selection uses only the already-open development source 0. "
            "It does not consume Phase-40 validation endpoints or registration "
            "correctness labels. The rule is point-local input routing at a fixed "
            "alpha, not a spatially varying alpha complex."
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
        default=Path("benchmark-out/eth_gazebo_local_support_calibration_phase41.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = calibrate_local_support(
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
