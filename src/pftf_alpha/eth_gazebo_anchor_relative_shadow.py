"""Evaluate the preregistered Phase-42 anchor-relative reconstruction shadow."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .eth_gazebo_anchor_relative import (
    evaluate_anchor_relative,
    evaluate_phase41_baseline,
)
from .eth_gazebo_anchor_relative_protocol import (
    CALIBRATION_SCHEMA,
    CALIBRATION_SHA256,
    FSCORE_TOLERANCE,
    PROTOCOL_SCHEMA,
    RECALL_TOLERANCE,
    SELECTED_CANDIDATE_ID,
    SELECTED_MAXIMUM_ANCHOR_PLANE_RESIDUAL_METERS,
    SELECTED_MAXIMUM_NEAREST_ANCHOR_DISTANCE_METERS,
    SELECTED_MINIMUM_NORMAL_ALIGNMENT,
    VALIDATION_SEED,
    VALIDATION_SOURCE_INDICES,
    _load_locked_json,
)
from .eth_gazebo_local_support import (
    evaluate_anchor_and_scan_baselines,
    load_hokuyo_scans,
    prepare_source_inputs,
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
    _voxel_downsample,
)
from .eth_gazebo_validation_protocol import (
    ARCHIVE_NAME,
    ARCHIVE_SHA256,
    verify_gazebo_archive_directory,
)
from .eth_open3d_fgr_pipeline import _load_open3d

EXPECTED_PROTOCOL_SHA256 = (
    "1b304d0c62251e1f572ab65295f5903b29e820ee1c9557edf8b6c54424b3efac"
)


@dataclass(frozen=True)
class AnchorRelativeValidationCase:
    source_index: int
    pair_count: int
    phase41_candidate_cell_count: int
    anchor_relative_cell_count: int
    rejected_by_anchor_relative_count: int
    anchor_baseline: ReconstructionEndpoint
    phase41_baseline: ReconstructionEndpoint
    anchor_relative: ReconstructionEndpoint
    geometry_margin_vs_anchor: float
    geometry_margin_vs_phase41: float
    fscore_margin_vs_anchor: float
    recall_margin_vs_anchor: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["anchor_baseline"] = self.anchor_baseline.to_dict()
        payload["phase41_baseline"] = self.phase41_baseline.to_dict()
        payload["anchor_relative"] = self.anchor_relative.to_dict()
        return payload


@dataclass(frozen=True)
class AnchorRelativeValidationSummary:
    case_count: int
    pair_count: int
    phase41_candidate_cell_count: int
    anchor_relative_cell_count: int
    rejected_by_anchor_relative_count: int
    mean_anchor_geometry_loss: float
    mean_phase41_geometry_loss: float
    mean_anchor_relative_geometry_loss: float
    mean_anchor_fscore: float
    mean_phase41_fscore: float
    mean_anchor_relative_fscore: float
    mean_anchor_recall: float
    mean_phase41_recall: float
    mean_anchor_relative_recall: float
    mean_anchor_components: float
    mean_phase41_components: float
    mean_anchor_relative_components: float
    mean_anchor_betti_1: float
    mean_phase41_betti_1: float
    mean_anchor_relative_betti_1: float
    mean_anchor_nonmanifold_edge_fraction: float
    mean_phase41_nonmanifold_edge_fraction: float
    mean_anchor_relative_nonmanifold_edge_fraction: float
    geometry_win_vs_anchor_count: int
    geometry_win_vs_phase41_count: int
    all_meshes_materialized: bool
    every_case_exercised_anchor_relative: bool
    mean_geometry_beats_both_baselines: bool
    mean_fscore_within_anchor_tolerance: bool
    mean_recall_within_anchor_tolerance: bool
    anchor_relative_shadow_supported: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GazeboAnchorRelativeShadow:
    artifact_schema: str
    role: str
    protocol_artifact_path: str
    protocol_artifact_sha256: str
    calibration_artifact_path: str
    calibration_artifact_sha256: str
    archive_path: str
    archive_sha256: str
    opened_archive_members: tuple[str, ...]
    validation_source_indices: tuple[int, ...]
    selected_candidate_id: str
    cases: tuple[AnchorRelativeValidationCase, ...]
    summary: AnchorRelativeValidationSummary
    prior_validation_references_accessed: bool
    validation_reference_values_accessed_by_evaluator: bool
    registration_label_values_accessed: bool
    anchor_relative_shadow_supported: bool
    point_local_alpha_field_supported: bool
    topology_correctness_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["opened_archive_members"] = list(self.opened_archive_members)
        payload["validation_source_indices"] = list(
            self.validation_source_indices
        )
        payload["cases"] = [case.to_dict() for case in self.cases]
        payload["summary"] = self.summary.to_dict()
        return payload


def _verify_protocol(path: Path) -> Mapping[str, object]:
    payload = _load_locked_json(
        path,
        expected_sha256=EXPECTED_PROTOCOL_SHA256,
        expected_schema=PROTOCOL_SCHEMA,
    )
    expected = {
        "archive_sha256": ARCHIVE_SHA256,
        "calibration_artifact_sha256": CALIBRATION_SHA256,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_maximum_nearest_anchor_distance_meters": (
            SELECTED_MAXIMUM_NEAREST_ANCHOR_DISTANCE_METERS
        ),
        "selected_maximum_anchor_plane_residual_meters": (
            SELECTED_MAXIMUM_ANCHOR_PLANE_RESIDUAL_METERS
        ),
        "selected_minimum_normal_alignment": SELECTED_MINIMUM_NORMAL_ALIGNMENT,
        "prior_validation_references_accessed": False,
        "validation_reference_values_accessed": False,
        "registration_label_values_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase-42 protocol mismatch: {key}")
    raw_sources = payload.get("validation_sources")
    if not isinstance(raw_sources, list):
        raise ValueError("Phase-42 validation sources are missing")
    if tuple(int(row["source_index"]) for row in raw_sources) != (
        VALIDATION_SOURCE_INDICES
    ):
        raise ValueError("Phase-42 protocol validation source set mismatch")
    return payload


def _summarize(
    cases: Sequence[AnchorRelativeValidationCase],
) -> AnchorRelativeValidationSummary:
    if not cases:
        raise ValueError("Phase-42 validation requires cases")

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=float)))

    anchor_geometry = mean([case.anchor_baseline.geometry_loss for case in cases])
    phase41_geometry = mean([case.phase41_baseline.geometry_loss for case in cases])
    relative_geometry = mean(
        [case.anchor_relative.geometry_loss for case in cases]
    )
    anchor_fscore = mean([case.anchor_baseline.fscore for case in cases])
    phase41_fscore = mean([case.phase41_baseline.fscore for case in cases])
    relative_fscore = mean([case.anchor_relative.fscore for case in cases])
    anchor_recall = mean([case.anchor_baseline.recall for case in cases])
    phase41_recall = mean([case.phase41_baseline.recall for case in cases])
    relative_recall = mean([case.anchor_relative.recall for case in cases])
    all_meshes = all(
        case.anchor_baseline.faces > 0
        and case.phase41_baseline.faces > 0
        and case.anchor_relative.faces > 0
        for case in cases
    )
    exercised = all(case.anchor_relative_cell_count > 0 for case in cases)
    geometry_beats = relative_geometry < min(anchor_geometry, phase41_geometry)
    fscore_close = relative_fscore >= anchor_fscore - FSCORE_TOLERANCE
    recall_close = relative_recall >= anchor_recall - RECALL_TOLERANCE
    return AnchorRelativeValidationSummary(
        case_count=len(cases),
        pair_count=sum(case.pair_count for case in cases),
        phase41_candidate_cell_count=sum(
            case.phase41_candidate_cell_count for case in cases
        ),
        anchor_relative_cell_count=sum(
            case.anchor_relative_cell_count for case in cases
        ),
        rejected_by_anchor_relative_count=sum(
            case.rejected_by_anchor_relative_count for case in cases
        ),
        mean_anchor_geometry_loss=anchor_geometry,
        mean_phase41_geometry_loss=phase41_geometry,
        mean_anchor_relative_geometry_loss=relative_geometry,
        mean_anchor_fscore=anchor_fscore,
        mean_phase41_fscore=phase41_fscore,
        mean_anchor_relative_fscore=relative_fscore,
        mean_anchor_recall=anchor_recall,
        mean_phase41_recall=phase41_recall,
        mean_anchor_relative_recall=relative_recall,
        mean_anchor_components=mean(
            [float(case.anchor_baseline.connected_components) for case in cases]
        ),
        mean_phase41_components=mean(
            [float(case.phase41_baseline.connected_components) for case in cases]
        ),
        mean_anchor_relative_components=mean(
            [float(case.anchor_relative.connected_components) for case in cases]
        ),
        mean_anchor_betti_1=mean(
            [float(case.anchor_baseline.betti_1) for case in cases]
        ),
        mean_phase41_betti_1=mean(
            [float(case.phase41_baseline.betti_1) for case in cases]
        ),
        mean_anchor_relative_betti_1=mean(
            [float(case.anchor_relative.betti_1) for case in cases]
        ),
        mean_anchor_nonmanifold_edge_fraction=mean(
            [case.anchor_baseline.nonmanifold_edge_fraction for case in cases]
        ),
        mean_phase41_nonmanifold_edge_fraction=mean(
            [case.phase41_baseline.nonmanifold_edge_fraction for case in cases]
        ),
        mean_anchor_relative_nonmanifold_edge_fraction=mean(
            [case.anchor_relative.nonmanifold_edge_fraction for case in cases]
        ),
        geometry_win_vs_anchor_count=sum(
            case.geometry_margin_vs_anchor > 0.0 for case in cases
        ),
        geometry_win_vs_phase41_count=sum(
            case.geometry_margin_vs_phase41 > 0.0 for case in cases
        ),
        all_meshes_materialized=all_meshes,
        every_case_exercised_anchor_relative=exercised,
        mean_geometry_beats_both_baselines=geometry_beats,
        mean_fscore_within_anchor_tolerance=fscore_close,
        mean_recall_within_anchor_tolerance=recall_close,
        anchor_relative_shadow_supported=(
            all_meshes
            and exercised
            and geometry_beats
            and fscore_close
            and recall_close
        ),
    )


def evaluate_anchor_relative_shadow(
    protocol_path: str | Path,
    calibration_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
    archive_path: str | Path,
) -> GazeboAnchorRelativeShadow:
    protocol_file = Path(protocol_path)
    calibration_file = Path(calibration_path)
    prediction_file = Path(prediction_path)
    decision_file = Path(decision_path)
    archive_file = Path(archive_path)
    _verify_protocol(protocol_file)
    _load_locked_json(
        calibration_file,
        expected_sha256=CALIBRATION_SHA256,
        expected_schema=CALIBRATION_SCHEMA,
    )
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
        if pair != (
            int(decision["source_index"]),
            int(decision["target_index"]),
        ):
            raise ValueError("Gazebo prediction and decision pair order differs")
        prediction_by_source[pair[0]].append(prediction)
        accept_by_pair[pair] = bool(decision["guarded_accept"])
    verification = verify_gazebo_archive_directory(archive_file)
    o3d = _load_open3d()
    raw_scans, opened_members = load_hokuyo_scans(archive_file)
    downsampled_scans = tuple(
        _voxel_downsample(o3d, points, SOURCE_VOXEL_METERS)
        for points in raw_scans
    )
    cases: list[AnchorRelativeValidationCase] = []
    for source_index in VALIDATION_SOURCE_INDICES:
        inputs = prepare_source_inputs(
            o3d,
            raw_scans,
            downsampled_scans,
            prediction_by_source[source_index],
            accept_by_pair,
            source_index=source_index,
        )
        seed = VALIDATION_SEED + source_index
        anchor, _, _ = evaluate_anchor_and_scan_baselines(
            o3d,
            inputs,
            seed=seed,
        )
        phase41 = evaluate_phase41_baseline(o3d, inputs, seed=seed)
        relative = evaluate_anchor_relative(
            o3d,
            inputs,
            maximum_nearest_anchor_distance_meters=(
                SELECTED_MAXIMUM_NEAREST_ANCHOR_DISTANCE_METERS
            ),
            maximum_anchor_plane_residual_meters=(
                SELECTED_MAXIMUM_ANCHOR_PLANE_RESIDUAL_METERS
            ),
            minimum_normal_alignment=SELECTED_MINIMUM_NORMAL_ALIGNMENT,
            seed=seed,
        )
        cases.append(
            AnchorRelativeValidationCase(
                source_index=source_index,
                pair_count=inputs.pair_count,
                phase41_candidate_cell_count=(
                    relative.route.phase41_candidate_cell_count
                ),
                anchor_relative_cell_count=(
                    relative.route.anchor_relative_cell_count
                ),
                rejected_by_anchor_relative_count=(
                    relative.route.rejected_by_anchor_relative_count
                ),
                anchor_baseline=anchor,
                phase41_baseline=phase41.endpoint,
                anchor_relative=relative.endpoint,
                geometry_margin_vs_anchor=(
                    anchor.geometry_loss - relative.endpoint.geometry_loss
                ),
                geometry_margin_vs_phase41=(
                    phase41.endpoint.geometry_loss - relative.endpoint.geometry_loss
                ),
                fscore_margin_vs_anchor=relative.endpoint.fscore - anchor.fscore,
                recall_margin_vs_anchor=relative.endpoint.recall - anchor.recall,
            )
        )
        print(
            f"[ETH Gazebo Phase 42] source {source_index}: "
            f"{relative.route.anchor_relative_cell_count} anchor-relative cells",
            flush=True,
        )
    frozen_cases = tuple(cases)
    summary = _summarize(frozen_cases)
    return GazeboAnchorRelativeShadow(
        artifact_schema="pftf_alpha_eth_gazebo_anchor_relative_shadow_phase42/v1",
        role="post_protocol_anchor_relative_reconstruction_shadow",
        protocol_artifact_path=str(protocol_file),
        protocol_artifact_sha256=EXPECTED_PROTOCOL_SHA256,
        calibration_artifact_path=str(calibration_file),
        calibration_artifact_sha256=CALIBRATION_SHA256,
        archive_path=str(archive_file),
        archive_sha256=verification.sha256,
        opened_archive_members=opened_members,
        validation_source_indices=VALIDATION_SOURCE_INDICES,
        selected_candidate_id=SELECTED_CANDIDATE_ID,
        cases=frozen_cases,
        summary=summary,
        prior_validation_references_accessed=False,
        validation_reference_values_accessed_by_evaluator=True,
        registration_label_values_accessed=False,
        anchor_relative_shadow_supported=summary.anchor_relative_shadow_supported,
        point_local_alpha_field_supported=False,
        topology_correctness_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
        claim_boundary=(
            "This validates anchor-relative observed-only point routing on three "
            "small late-sequence Gazebo sources at fixed alpha. It is not a local "
            "alpha complex, full-scene or topology truth, deployed trimming, or "
            "deployment evidence."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_anchor_relative_protocol_phase42.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_anchor_relative_calibration_phase42.json"),
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
        "--archive",
        type=Path,
        default=Path("benchmark-data/eth_gazebo_summer") / ARCHIVE_NAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_anchor_relative_shadow_phase42.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_anchor_relative_shadow(
        args.protocol,
        args.calibration,
        args.predictions,
        args.decisions,
        args.archive,
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
