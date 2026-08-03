"""Evaluate the preregistered Phase-41 point-local support shadow."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .eth_gazebo_local_support import (
    evaluate_anchor_and_scan_baselines,
    evaluate_local_support,
    load_hokuyo_scans,
    prepare_source_inputs,
)
from .eth_gazebo_local_support_protocol import (
    CALIBRATION_SCHEMA,
    CALIBRATION_SHA256,
    FSCORE_TOLERANCE,
    PROTOCOL_SCHEMA,
    RECALL_TOLERANCE,
    SELECTED_CANDIDATE_ID,
    SELECTED_MAXIMUM_DISPERSION_METERS,
    SELECTED_MINIMUM_SUPPORT,
    VALIDATION_SEED,
    VALIDATION_SOURCE_INDICES,
    _load_locked_json,
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
    "f3c48a60657a8877f4cde330a8f917b99bd1a3521ed5ffd7682565145fce60a8"
)


@dataclass(frozen=True)
class LocalSupportValidationCase:
    source_index: int
    pair_count: int
    accepted_pair_count: int
    rejected_pair_count: int
    anchor_cell_count: int
    target_only_cell_count: int
    corroborated_target_only_cell_count: int
    rejected_target_only_cell_count: int
    mean_target_support: float
    mean_target_dispersion_meters: float
    anchor_baseline: ReconstructionEndpoint
    scan_fused_baseline: ReconstructionEndpoint
    local_support: ReconstructionEndpoint
    geometry_margin_vs_anchor: float
    geometry_margin_vs_scan: float
    fscore_margin_vs_anchor: float
    recall_margin_vs_anchor: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["anchor_baseline"] = self.anchor_baseline.to_dict()
        payload["scan_fused_baseline"] = self.scan_fused_baseline.to_dict()
        payload["local_support"] = self.local_support.to_dict()
        return payload


@dataclass(frozen=True)
class LocalSupportValidationSummary:
    case_count: int
    pair_count: int
    corroborated_target_only_cell_count: int
    rejected_target_only_cell_count: int
    mean_anchor_geometry_loss: float
    mean_scan_geometry_loss: float
    mean_local_geometry_loss: float
    mean_anchor_fscore: float
    mean_scan_fscore: float
    mean_local_fscore: float
    mean_anchor_recall: float
    mean_scan_recall: float
    mean_local_recall: float
    mean_anchor_components: float
    mean_scan_components: float
    mean_local_components: float
    mean_anchor_betti_1: float
    mean_scan_betti_1: float
    mean_local_betti_1: float
    mean_anchor_nonmanifold_edge_fraction: float
    mean_scan_nonmanifold_edge_fraction: float
    mean_local_nonmanifold_edge_fraction: float
    geometry_win_vs_anchor_count: int
    geometry_win_vs_scan_count: int
    all_meshes_materialized: bool
    every_case_exercised_local_support: bool
    mean_geometry_beats_both_baselines: bool
    mean_fscore_within_anchor_tolerance: bool
    mean_recall_within_anchor_tolerance: bool
    local_support_shadow_supported: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GazeboLocalSupportShadow:
    artifact_schema: str
    role: str
    protocol_artifact_path: str
    protocol_artifact_sha256: str
    calibration_artifact_path: str
    calibration_artifact_sha256: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    decision_artifact_path: str
    decision_artifact_sha256: str
    archive_path: str
    archive_sha256: str
    opened_archive_members: tuple[str, ...]
    validation_source_indices: tuple[int, ...]
    selected_candidate_id: str
    selected_minimum_support: int
    selected_maximum_dispersion_meters: float
    cases: tuple[LocalSupportValidationCase, ...]
    summary: LocalSupportValidationSummary
    phase40_validation_references_accessed: bool
    validation_reference_values_accessed_by_evaluator: bool
    registration_label_values_accessed: bool
    local_support_shadow_supported: bool
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
        "prediction_artifact_sha256": EXPECTED_PREDICTION_SHA256,
        "decision_artifact_sha256": EXPECTED_DECISION_SHA256,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_minimum_support": SELECTED_MINIMUM_SUPPORT,
        "selected_maximum_dispersion_meters": (
            SELECTED_MAXIMUM_DISPERSION_METERS
        ),
        "phase40_validation_references_accessed": False,
        "validation_reference_values_accessed": False,
        "registration_label_values_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase-41 protocol mismatch: {key}")
    raw_sources = payload.get("validation_sources")
    if not isinstance(raw_sources, list):
        raise ValueError("Phase-41 protocol validation sources are missing")
    if tuple(int(row["source_index"]) for row in raw_sources) != (
        VALIDATION_SOURCE_INDICES
    ):
        raise ValueError("Phase-41 protocol validation source set mismatch")
    return payload


def _summarize(
    cases: Sequence[LocalSupportValidationCase],
) -> LocalSupportValidationSummary:
    if not cases:
        raise ValueError("Phase-41 validation requires cases")

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=float)))

    anchor_geometry = mean([case.anchor_baseline.geometry_loss for case in cases])
    scan_geometry = mean([case.scan_fused_baseline.geometry_loss for case in cases])
    local_geometry = mean([case.local_support.geometry_loss for case in cases])
    anchor_fscore = mean([case.anchor_baseline.fscore for case in cases])
    scan_fscore = mean([case.scan_fused_baseline.fscore for case in cases])
    local_fscore = mean([case.local_support.fscore for case in cases])
    anchor_recall = mean([case.anchor_baseline.recall for case in cases])
    scan_recall = mean([case.scan_fused_baseline.recall for case in cases])
    local_recall = mean([case.local_support.recall for case in cases])
    all_meshes = all(
        case.anchor_baseline.faces > 0
        and case.scan_fused_baseline.faces > 0
        and case.local_support.faces > 0
        for case in cases
    )
    exercised = all(case.corroborated_target_only_cell_count > 0 for case in cases)
    geometry_beats = local_geometry < min(anchor_geometry, scan_geometry)
    fscore_close = local_fscore >= anchor_fscore - FSCORE_TOLERANCE
    recall_close = local_recall >= anchor_recall - RECALL_TOLERANCE
    return LocalSupportValidationSummary(
        case_count=len(cases),
        pair_count=sum(case.pair_count for case in cases),
        corroborated_target_only_cell_count=sum(
            case.corroborated_target_only_cell_count for case in cases
        ),
        rejected_target_only_cell_count=sum(
            case.rejected_target_only_cell_count for case in cases
        ),
        mean_anchor_geometry_loss=anchor_geometry,
        mean_scan_geometry_loss=scan_geometry,
        mean_local_geometry_loss=local_geometry,
        mean_anchor_fscore=anchor_fscore,
        mean_scan_fscore=scan_fscore,
        mean_local_fscore=local_fscore,
        mean_anchor_recall=anchor_recall,
        mean_scan_recall=scan_recall,
        mean_local_recall=local_recall,
        mean_anchor_components=mean(
            [float(case.anchor_baseline.connected_components) for case in cases]
        ),
        mean_scan_components=mean(
            [float(case.scan_fused_baseline.connected_components) for case in cases]
        ),
        mean_local_components=mean(
            [float(case.local_support.connected_components) for case in cases]
        ),
        mean_anchor_betti_1=mean(
            [float(case.anchor_baseline.betti_1) for case in cases]
        ),
        mean_scan_betti_1=mean(
            [float(case.scan_fused_baseline.betti_1) for case in cases]
        ),
        mean_local_betti_1=mean(
            [float(case.local_support.betti_1) for case in cases]
        ),
        mean_anchor_nonmanifold_edge_fraction=mean(
            [case.anchor_baseline.nonmanifold_edge_fraction for case in cases]
        ),
        mean_scan_nonmanifold_edge_fraction=mean(
            [case.scan_fused_baseline.nonmanifold_edge_fraction for case in cases]
        ),
        mean_local_nonmanifold_edge_fraction=mean(
            [case.local_support.nonmanifold_edge_fraction for case in cases]
        ),
        geometry_win_vs_anchor_count=sum(
            case.geometry_margin_vs_anchor > 0.0 for case in cases
        ),
        geometry_win_vs_scan_count=sum(
            case.geometry_margin_vs_scan > 0.0 for case in cases
        ),
        all_meshes_materialized=all_meshes,
        every_case_exercised_local_support=exercised,
        mean_geometry_beats_both_baselines=geometry_beats,
        mean_fscore_within_anchor_tolerance=fscore_close,
        mean_recall_within_anchor_tolerance=recall_close,
        local_support_shadow_supported=(
            all_meshes
            and exercised
            and geometry_beats
            and fscore_close
            and recall_close
        ),
    )


def evaluate_local_support_shadow(
    protocol_path: str | Path,
    calibration_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
    archive_path: str | Path,
) -> GazeboLocalSupportShadow:
    protocol_file = Path(protocol_path)
    calibration_file = Path(calibration_path)
    prediction_file = Path(prediction_path)
    decision_file = Path(decision_path)
    archive_file = Path(archive_path)
    _verify_protocol(protocol_file)
    calibration = _load_locked_json(
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
    if calibration.get("phase40_validation_references_accessed") is not False:
        raise ValueError("calibration crossed the Phase-40 reference boundary")
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
    verification = verify_gazebo_archive_directory(archive_file)
    o3d = _load_open3d()
    raw_scans, opened_members = load_hokuyo_scans(archive_file)
    downsampled_scans = tuple(
        _voxel_downsample(o3d, points, SOURCE_VOXEL_METERS)
        for points in raw_scans
    )
    cases: list[LocalSupportValidationCase] = []
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
        anchor, scan, base_route = evaluate_anchor_and_scan_baselines(
            o3d,
            inputs,
            seed=seed,
        )
        local = evaluate_local_support(
            o3d,
            inputs,
            minimum_support=SELECTED_MINIMUM_SUPPORT,
            maximum_dispersion_meters=SELECTED_MAXIMUM_DISPERSION_METERS,
            seed=seed,
        )
        cases.append(
            LocalSupportValidationCase(
                source_index=source_index,
                pair_count=inputs.pair_count,
                accepted_pair_count=inputs.accepted_pair_count,
                rejected_pair_count=inputs.rejected_pair_count,
                anchor_cell_count=base_route.anchor_cell_count,
                target_only_cell_count=base_route.target_only_cell_count,
                corroborated_target_only_cell_count=(
                    local.route.corroborated_target_only_cell_count
                ),
                rejected_target_only_cell_count=(
                    local.route.rejected_target_only_cell_count
                ),
                mean_target_support=local.route.mean_target_support,
                mean_target_dispersion_meters=(
                    local.route.mean_target_dispersion_meters
                ),
                anchor_baseline=anchor,
                scan_fused_baseline=scan,
                local_support=local.endpoint,
                geometry_margin_vs_anchor=(
                    anchor.geometry_loss - local.endpoint.geometry_loss
                ),
                geometry_margin_vs_scan=(
                    scan.geometry_loss - local.endpoint.geometry_loss
                ),
                fscore_margin_vs_anchor=local.endpoint.fscore - anchor.fscore,
                recall_margin_vs_anchor=local.endpoint.recall - anchor.recall,
            )
        )
        print(
            f"[ETH Gazebo Phase 41] source {source_index}: "
            f"{local.route.corroborated_target_only_cell_count} supported cells",
            flush=True,
        )
    frozen_cases = tuple(cases)
    summary = _summarize(frozen_cases)
    return GazeboLocalSupportShadow(
        artifact_schema="pftf_alpha_eth_gazebo_local_support_shadow_phase41/v1",
        role="post_protocol_point_local_support_reconstruction_shadow",
        protocol_artifact_path=str(protocol_file),
        protocol_artifact_sha256=EXPECTED_PROTOCOL_SHA256,
        calibration_artifact_path=str(calibration_file),
        calibration_artifact_sha256=CALIBRATION_SHA256,
        prediction_artifact_path=str(prediction_file),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        decision_artifact_path=str(decision_file),
        decision_artifact_sha256=EXPECTED_DECISION_SHA256,
        archive_path=str(archive_file),
        archive_sha256=verification.sha256,
        opened_archive_members=opened_members,
        validation_source_indices=VALIDATION_SOURCE_INDICES,
        selected_candidate_id=SELECTED_CANDIDATE_ID,
        selected_minimum_support=SELECTED_MINIMUM_SUPPORT,
        selected_maximum_dispersion_meters=SELECTED_MAXIMUM_DISPERSION_METERS,
        cases=frozen_cases,
        summary=summary,
        phase40_validation_references_accessed=False,
        validation_reference_values_accessed_by_evaluator=True,
        registration_label_values_accessed=False,
        local_support_shadow_supported=summary.local_support_shadow_supported,
        point_local_alpha_field_supported=False,
        topology_correctness_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
        claim_boundary=(
            "This validates observed-only point-local input routing at one fixed "
            "alpha on seven endpoint-unopened Gazebo sources. It does not construct "
            "a spatially varying alpha complex, establish full-scene or topology "
            "truth, identify physical correspondences, deploy trimming, or support "
            "deployment."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_local_support_protocol_phase41.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_local_support_calibration_phase41.json"),
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
        default=Path("benchmark-out/eth_gazebo_local_support_shadow_phase41.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_local_support_shadow(
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
