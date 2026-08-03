"""Phase-40 preregistration for the ETH Gazebo reconstruction shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .eth_gazebo_validation_protocol import (
    ARCHIVE_NAME,
    LABEL_MEMBER,
    verify_gazebo_archive_directory,
)

EXPECTED_PREDICTION_SHA256 = (
    "ed25ac05393d3a9270bef04e99bf79870b8eddd4c0ba6cb0e45d7bff2931900e"
)
EXPECTED_DECISION_SHA256 = (
    "20dcacaed83575d7c997657d61de2a5e797cfb7d5a3fd3d2ecaea0e070a5f6fb"
)
PREDICTION_SCHEMA = "pftf_alpha_eth_gazebo_predictions_phase39/v1"
DECISION_SCHEMA = "pftf_alpha_eth_gazebo_rotation_decisions_phase39/v1"
PROTOCOL_SCHEMA = "pftf_alpha_eth_gazebo_reconstruction_protocol_phase40/v1"
DEVELOPMENT_SOURCE_INDEX = 0
VALIDATION_SOURCE_INDICES = (
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    19,
    20,
)
HELDOUT_MODULUS = 5
HELDOUT_REMAINDER = 0
SOURCE_VOXEL_METERS = 0.50
REFERENCE_VOXEL_METERS = 0.25
FUSION_VOXEL_METERS = 0.75
ALPHA_METERS = 1.00
ROI_LOWER_QUANTILE = 0.005
ROI_UPPER_QUANTILE = 0.995
ROI_MARGIN_METERS = 1.0
SURFACE_SAMPLE_COUNT = 4096
DISTANCE_THRESHOLD_FRACTION = 0.01
RECALL_NONREGRESSION_TOLERANCE = 0.01
EVALUATION_SEED = 20_260_804


@dataclass(frozen=True)
class ReconstructionSourcePlan:
    source_index: int
    pair_count: int
    accepted_count: int
    rejected_count: int


@dataclass(frozen=True)
class GazeboReconstructionPreregistration:
    artifact_schema: str
    role: str
    archive_path: str
    archive_sha256: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    decision_artifact_path: str
    decision_artifact_sha256: str
    development_source_index: int
    validation_sources: tuple[ReconstructionSourcePlan, ...]
    source_selection_rule: str
    heldout_split_rule: str
    source_voxel_meters: float
    reference_voxel_meters: float
    fusion_voxel_meters: float
    alpha_meters: float
    alpha_selection_rule: str
    roi_rule: str
    surface_sample_count: int
    distance_threshold_fraction: float
    geometry_loss_rule: str
    geometry_support_rule: str
    topology_role: str
    label_boundary: str
    registration_label_values_accessed: bool
    validation_reference_values_accessed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["validation_sources"] = [
            asdict(source) for source in self.validation_sources
        ]
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_locked_json(
    path: Path,
    *,
    expected_sha256: str,
    expected_schema: str,
) -> Mapping[str, object]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"Gazebo artifact SHA-256 mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gazebo artifact must be a JSON object")
    if payload.get("artifact_schema") != expected_schema:
        raise ValueError("Gazebo artifact schema mismatch")
    return payload


def _source_plans(decisions: Sequence[Mapping[str, object]]) -> tuple[
    ReconstructionSourcePlan, ...
]:
    totals: Counter[int] = Counter()
    accepted: Counter[int] = Counter()
    for row in decisions:
        source = int(row["source_index"])
        totals[source] += 1
        accepted[source] += bool(row["guarded_accept"])
    plans = tuple(
        ReconstructionSourcePlan(
            source_index=source,
            pair_count=totals[source],
            accepted_count=accepted[source],
            rejected_count=totals[source] - accepted[source],
        )
        for source in sorted(totals)
        if source != DEVELOPMENT_SOURCE_INDEX
        and accepted[source] > 0
        and totals[source] - accepted[source] > 0
    )
    if tuple(plan.source_index for plan in plans) != VALIDATION_SOURCE_INDICES:
        raise ValueError("Phase-40 validation source set mismatch")
    return plans


def preregister_gazebo_reconstruction(
    archive_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
) -> GazeboReconstructionPreregistration:
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
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("Gazebo decision rows are missing")
    plans = _source_plans(raw_decisions)
    return GazeboReconstructionPreregistration(
        artifact_schema=PROTOCOL_SCHEMA,
        role="pre_validation_real_alpha_reconstruction_shadow_protocol",
        archive_path=str(archive),
        archive_sha256=verification.sha256,
        prediction_artifact_path=str(prediction_file),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        decision_artifact_path=str(decision_file),
        decision_artifact_sha256=EXPECTED_DECISION_SHA256,
        development_source_index=DEVELOPMENT_SOURCE_INDEX,
        validation_sources=plans,
        source_selection_rule=(
            "all non-development source indices having at least one frozen p90 "
            "accept and at least one frozen p90 reject; registration labels are unused"
        ),
        heldout_split_rule=(
            "within each source Hokuyo row order, index mod 5 == 0 is evaluation-only; "
            "the other rows form the source observation"
        ),
        source_voxel_meters=SOURCE_VOXEL_METERS,
        reference_voxel_meters=REFERENCE_VOXEL_METERS,
        fusion_voxel_meters=FUSION_VOXEL_METERS,
        alpha_meters=ALPHA_METERS,
        alpha_selection_rule=(
            "fixed at twice the frozen Phase-39 0.50 m registration voxel; no "
            "validation-reference alpha search"
        ),
        roi_rule=(
            "per-axis source-observation quantiles [0.005, 0.995], expanded by 1 m; "
            "source reference and all fused inputs are cropped to the same ROI"
        ),
        surface_sample_count=SURFACE_SAMPLE_COUNT,
        distance_threshold_fraction=DISTANCE_THRESHOLD_FRACTION,
        geometry_loss_rule=(
            "normalized symmetric squared Chamfer plus normalized Hausdorff "
            "against the source-view heldout points"
        ),
        geometry_support_rule=(
            "mean guard geometry loss < mean baseline loss, mean guard F-score >= "
            "mean baseline F-score, and mean guard recall >= baseline recall - 0.01"
        ),
        topology_role=(
            "connected components, Betti numbers, and nonmanifold edges are "
            "descriptive because no full-scene topology ground truth is supplied"
        ),
        label_boundary=(
            f"only the {len(plans)} validation Hokuyo source/reference splits, other "
            "Hokuyo inputs, and hash-locked pre-label Phase-39 artifacts may be read; "
            f"the registration label member {LABEL_MEMBER!r} must not be opened"
        ),
        registration_label_values_accessed=False,
        validation_reference_values_accessed=False,
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
        default=Path(
            "benchmark-out/eth_gazebo_reconstruction_protocol_phase40.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preregister_gazebo_reconstruction(
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
