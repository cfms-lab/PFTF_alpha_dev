"""Audit frozen Phase-39 Gazebo decisions after first label access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .eth_gazebo_fgr_icp_pipeline import EXPECTED_PROTOCOL_SHA256
from .eth_gazebo_rotation_decisions import EXPECTED_PREDICTION_SHA256
from .eth_gazebo_validation_protocol import (
    EXPECTED_PAIR_COUNT,
    LABEL_MEMBER,
    MAX_RELATIVE_ROTATION_ERROR_DEGREES,
    MAX_RELATIVE_TRANSLATION_ERROR_METERS,
    SCAN_COUNT,
    verify_gazebo_archive_directory,
)
from .fresh_external_rotation_audit import (
    _relative_target_to_source,
    _rigid_errors,
)
from .open3d_fgr_pipeline import nonconsecutive_fragment_pairs
from .scene_relative_rotation_guard import (
    MINIMUM_CORRECT_RETENTION,
    MINIMUM_INCORRECT_REJECTION,
    ROTATION_PERCENTILE_CUTOFF,
)

EXPECTED_DECISION_SHA256 = (
    "20dcacaed83575d7c997657d61de2a5e797cfb7d5a3fd3d2ecaea0e070a5f6fb"
)
PRELABEL_COMMIT = "032a3c39d3ef0908ce86fc742ceaeb37a618ad45"


@dataclass(frozen=True)
class GazeboLabeledObservation:
    source_index: int
    target_index: int
    prediction_rotation_radians: float
    scene_relative_rotation_percentile: float
    guarded_accept: bool
    relative_rotation_error_degrees: float
    relative_translation_error_meters: float
    frozen_correct: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GazeboValidationSummary:
    scene_name: str
    prediction_count: int
    base_correct_count: int
    base_incorrect_count: int
    base_precision: float
    guarded_accepted_count: int
    guarded_correct_count: int
    guarded_incorrect_count: int
    guarded_precision: float
    precision_gain_percentage_points: float
    correct_retention: float
    incorrect_rejection: float
    has_correct_and_incorrect_predictions: bool
    precision_improved: bool
    correct_retention_gate_passed: bool
    incorrect_rejection_gate_passed: bool
    fresh_scene_transfer_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GazeboRotationAudit:
    artifact_schema: str
    role: str
    protocol_artifact_path: str
    protocol_artifact_sha256: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    decision_artifact_path: str
    decision_artifact_sha256: str
    prelabel_commit: str
    archive_path: str
    archive_sha256: str
    label_member: str
    label_execution_order: str
    correctness_rule: str
    rotation_percentile_cutoff: float
    minimum_correct_retention: float
    minimum_incorrect_rejection: float
    observations: tuple[GazeboLabeledObservation, ...]
    summary: GazeboValidationSummary
    validation_label_values_accessed_by_evaluator: bool
    prelabel_prediction_artifact_verified: bool
    prelabel_decision_artifact_verified: bool
    fresh_label_blind_validation_supported: bool
    calibrated_external_pipeline_transfer_supported: bool
    fresh_external_rotation_guard_transfer_supported: bool
    independent_algorithm_implementation_supported: bool
    real_registration_labels_supported: bool
    real_correspondence_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "observations": [row.to_dict() for row in self.observations],
            "summary": self.summary.to_dict(),
        }


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


def _parse_gazebo_poses(stream: BinaryIO) -> tuple[np.ndarray, ...]:
    with io.TextIOWrapper(stream, encoding="ascii", newline="") as text:
        reader = csv.reader(text, skipinitialspace=True)
        header = next(reader)
        expected_header = [
            "poseId",
            "timestamp",
            *(f"T{row}{column}" for row in range(4) for column in range(4)),
        ]
        if header != expected_header:
            raise ValueError("unexpected Gazebo Leica pose header")
        poses: list[np.ndarray] = []
        for expected_index, row in enumerate(reader):
            if len(row) != 18 or int(row[0]) != expected_index:
                raise ValueError("Gazebo Leica pose rows are malformed")
            matrix = np.asarray([float(value) for value in row[2:]], dtype=float)
            matrix = matrix.reshape(4, 4)
            if not np.all(np.isfinite(matrix)):
                raise ValueError("Gazebo Leica pose contains non-finite values")
            if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
                raise ValueError("Gazebo Leica pose is not affine")
            if not np.allclose(
                matrix[:3, :3].T @ matrix[:3, :3],
                np.eye(3),
                atol=2e-6,
            ):
                raise ValueError("Gazebo Leica rotation is not orthogonal")
            poses.append(matrix)
    if len(poses) != SCAN_COUNT:
        raise ValueError("Gazebo Leica pose count mismatch")
    return tuple(poses)


def summarize_gazebo(
    observations: tuple[GazeboLabeledObservation, ...],
) -> GazeboValidationSummary:
    correct = sum(row.frozen_correct for row in observations)
    incorrect = len(observations) - correct
    accepted = tuple(row for row in observations if row.guarded_accept)
    guarded_correct = sum(row.frozen_correct for row in accepted)
    guarded_incorrect = len(accepted) - guarded_correct
    base_precision = correct / len(observations)
    guarded_precision = guarded_correct / len(accepted)
    retention = guarded_correct / correct if correct else 0.0
    rejection = (incorrect - guarded_incorrect) / incorrect if incorrect else 0.0
    has_both = correct > 0 and incorrect > 0
    precision_improved = guarded_precision > base_precision
    retention_passed = retention >= MINIMUM_CORRECT_RETENTION
    rejection_passed = rejection >= MINIMUM_INCORRECT_REJECTION
    passed = has_both and precision_improved and retention_passed and rejection_passed
    return GazeboValidationSummary(
        scene_name="ETH Gazebo Summer",
        prediction_count=len(observations),
        base_correct_count=correct,
        base_incorrect_count=incorrect,
        base_precision=base_precision,
        guarded_accepted_count=len(accepted),
        guarded_correct_count=guarded_correct,
        guarded_incorrect_count=guarded_incorrect,
        guarded_precision=guarded_precision,
        precision_gain_percentage_points=(
            100.0 * (guarded_precision - base_precision)
        ),
        correct_retention=retention,
        incorrect_rejection=rejection,
        has_correct_and_incorrect_predictions=has_both,
        precision_improved=precision_improved,
        correct_retention_gate_passed=retention_passed,
        incorrect_rejection_gate_passed=rejection_passed,
        fresh_scene_transfer_gate_passed=passed,
    )


def evaluate_gazebo_rotation_audit(
    protocol_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
    archive_path: str | Path,
) -> GazeboRotationAudit:
    protocol_file = Path(protocol_path)
    prediction_file = Path(prediction_path)
    decision_file = Path(decision_path)
    archive_file = Path(archive_path)
    protocol = _load_locked_json(
        protocol_file,
        expected_sha256=EXPECTED_PROTOCOL_SHA256,
        expected_schema="pftf_alpha_eth_gazebo_validation_protocol_phase39/v1",
    )
    predictions = _load_locked_json(
        prediction_file,
        expected_sha256=EXPECTED_PREDICTION_SHA256,
        expected_schema="pftf_alpha_eth_gazebo_predictions_phase39/v1",
    )
    decisions = _load_locked_json(
        decision_file,
        expected_sha256=EXPECTED_DECISION_SHA256,
        expected_schema=(
            "pftf_alpha_eth_gazebo_rotation_decisions_phase39/v1"
        ),
    )
    if protocol.get("validation_label_values_accessed") is not False:
        raise ValueError("Gazebo protocol label flag mismatch")
    if predictions.get("validation_label_member_opened") is not False:
        raise ValueError("Gazebo prediction label flag mismatch")
    if decisions.get("validation_label_values_accessed") is not False:
        raise ValueError("Gazebo decision label flag mismatch")
    if predictions.get("complete_prediction_set_materialized") is not True:
        raise ValueError("Gazebo predictions are incomplete")
    if decisions.get("complete_decision_set_materialized") is not True:
        raise ValueError("Gazebo decisions are incomplete")
    raw_predictions = predictions.get("predictions")
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_predictions, list) or not isinstance(raw_decisions, list):
        raise ValueError("Gazebo rows are missing")
    if len(raw_predictions) != EXPECTED_PAIR_COUNT:
        raise ValueError("Gazebo prediction count mismatch")
    if len(raw_decisions) != EXPECTED_PAIR_COUNT:
        raise ValueError("Gazebo decision count mismatch")
    expected_pairs = nonconsecutive_fragment_pairs(SCAN_COUNT)
    prediction_by_pair = {
        (int(row["source_index"]), int(row["target_index"])): row
        for row in raw_predictions
    }
    decision_by_pair = {
        (int(row["source_index"]), int(row["target_index"])): row
        for row in raw_decisions
    }
    if tuple(prediction_by_pair) != expected_pairs:
        raise ValueError("Gazebo prediction pair order mismatch")
    if tuple(decision_by_pair) != expected_pairs:
        raise ValueError("Gazebo decision pair order mismatch")
    verification = verify_gazebo_archive_directory(archive_file)
    with zipfile.ZipFile(archive_file) as source:
        with source.open(LABEL_MEMBER) as stream:
            poses = _parse_gazebo_poses(stream)
    observations: list[GazeboLabeledObservation] = []
    for pair in expected_pairs:
        prediction = prediction_by_pair[pair]
        decision = decision_by_pair[pair]
        ground_truth = _relative_target_to_source(poses[pair[0]], poses[pair[1]])
        rotation_error, translation_error = _rigid_errors(
            prediction["target_to_source_matrix"],
            ground_truth,
        )
        correct = (
            rotation_error < MAX_RELATIVE_ROTATION_ERROR_DEGREES
            and translation_error < MAX_RELATIVE_TRANSLATION_ERROR_METERS
        )
        observations.append(
            GazeboLabeledObservation(
                source_index=pair[0],
                target_index=pair[1],
                prediction_rotation_radians=float(
                    decision["prediction_rotation_radians"]
                ),
                scene_relative_rotation_percentile=float(
                    decision["scene_relative_rotation_percentile"]
                ),
                guarded_accept=bool(decision["guarded_accept"]),
                relative_rotation_error_degrees=rotation_error,
                relative_translation_error_meters=translation_error,
                frozen_correct=correct,
            )
        )
    frozen = tuple(observations)
    summary = summarize_gazebo(frozen)
    supported = summary.fresh_scene_transfer_gate_passed
    return GazeboRotationAudit(
        artifact_schema="pftf_alpha_eth_gazebo_rotation_audit_phase39/v1",
        role="post_label_audit_of_committed_prelabel_gazebo_decisions",
        protocol_artifact_path=str(protocol_file),
        protocol_artifact_sha256=EXPECTED_PROTOCOL_SHA256,
        prediction_artifact_path=str(prediction_file),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        decision_artifact_path=str(decision_file),
        decision_artifact_sha256=EXPECTED_DECISION_SHA256,
        prelabel_commit=PRELABEL_COMMIT,
        archive_path=str(archive_file),
        archive_sha256=verification.sha256,
        label_member=LABEL_MEMBER,
        label_execution_order=(
            "verify committed protocol, complete prediction set, and complete "
            "p90 decisions by hash; only then open the single Gazebo Leica "
            "pose member and join frozen correctness labels"
        ),
        correctness_rule="strict RRE < 15 degrees and strict RTE < 0.30 meters",
        rotation_percentile_cutoff=ROTATION_PERCENTILE_CUTOFF,
        minimum_correct_retention=MINIMUM_CORRECT_RETENTION,
        minimum_incorrect_rejection=MINIMUM_INCORRECT_REJECTION,
        observations=frozen,
        summary=summary,
        validation_label_values_accessed_by_evaluator=True,
        prelabel_prediction_artifact_verified=True,
        prelabel_decision_artifact_verified=True,
        fresh_label_blind_validation_supported=supported,
        calibrated_external_pipeline_transfer_supported=supported,
        fresh_external_rotation_guard_transfer_supported=supported,
        independent_algorithm_implementation_supported=False,
        real_registration_labels_supported=True,
        real_correspondence_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_validation_protocol_phase39.json"),
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
        default=(
            Path("benchmark-data/eth_gazebo_summer")
            / "gazebo_summer_04-Aug-2011-16_13_22.zip"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_rotation_audit_phase39.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_gazebo_rotation_audit(
        args.protocol,
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
