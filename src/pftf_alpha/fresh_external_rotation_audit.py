"""Evaluate frozen Phase-38 ETH decisions after label materialization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .eth_open3d_fgr_pipeline import EXPECTED_PROTOCOL_SHA256
from .eth_rotation_decisions import EXPECTED_PREDICTION_SHA256
from .fresh_external_protocol import (
    EXPECTED_PAIR_COUNT,
    LABEL_MEMBER,
    MAX_RELATIVE_ROTATION_ERROR_DEGREES,
    MAX_RELATIVE_TRANSLATION_ERROR_METERS,
    SCAN_COUNT,
    verify_archive_directory,
)
from .open3d_fgr_pipeline import nonconsecutive_fragment_pairs
from .scene_relative_rotation_guard import (
    MINIMUM_CORRECT_RETENTION,
    MINIMUM_INCORRECT_REJECTION,
    ROTATION_PERCENTILE_CUTOFF,
)

EXPECTED_DECISION_SHA256 = (
    "26f069fa77841dfb446185d01809a062b242af3f1517e605d68105aab43850c0"
)
PRELABEL_PREDICTION_COMMIT = "dffe72dd2c2e10a4233f49e25cd3aa98c6ee837c"


@dataclass(frozen=True)
class ETHLabeledRotationObservation:
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
class ETHFreshExternalSummary:
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
class ETHFreshExternalRotationAudit:
    artifact_schema: str
    role: str
    preregistration_artifact_path: str
    preregistration_artifact_sha256: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    decision_artifact_path: str
    decision_artifact_sha256: str
    prelabel_prediction_commit: str
    archive_path: str
    archive_sha256: str
    label_member: str
    label_execution_order: str
    matrix_convention: str
    correctness_rule: str
    maximum_relative_rotation_error_degrees: float
    maximum_relative_translation_error_meters: float
    rotation_percentile_cutoff: float
    minimum_correct_retention: float
    minimum_incorrect_rejection: float
    observations: tuple[ETHLabeledRotationObservation, ...]
    summary: ETHFreshExternalSummary
    label_values_accessed_by_evaluator: bool
    prelabel_prediction_artifact_verified: bool
    prelabel_decision_artifact_verified: bool
    fresh_label_blind_validation_supported: bool
    fresh_external_pipeline_transfer_supported: bool
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


def _load_hash_locked_json(
    path: Path,
    *,
    expected_sha256: str,
    expected_schema: str,
) -> Mapping[str, object]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"artifact SHA-256 mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact must be a JSON object")
    if payload.get("artifact_schema") != expected_schema:
        raise ValueError(f"artifact schema mismatch: {path}")
    return payload


def _parse_pose_labels(stream: BinaryIO) -> tuple[np.ndarray, ...]:
    with io.TextIOWrapper(stream, encoding="ascii", newline="") as text:
        reader = csv.reader(text, skipinitialspace=True)
        header = next(reader)
        expected_header = [
            "poseId",
            "timestamp",
            *(f"T{row}{column}" for row in range(4) for column in range(4)),
        ]
        if header != expected_header:
            raise ValueError("unexpected ETH Leica pose header")
        poses: list[np.ndarray] = []
        for expected_index, row in enumerate(reader):
            if len(row) != 18:
                raise ValueError("unexpected ETH Leica pose row width")
            if int(row[0]) != expected_index:
                raise ValueError("ETH Leica pose IDs are not contiguous")
            matrix = np.asarray([float(value) for value in row[2:]], dtype=float)
            matrix = matrix.reshape(4, 4)
            if not np.all(np.isfinite(matrix)):
                raise ValueError("ETH Leica pose contains non-finite values")
            if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
                raise ValueError("ETH Leica pose is not affine")
            if not np.allclose(
                matrix[:3, :3].T @ matrix[:3, :3],
                np.eye(3),
                atol=2e-6,
            ):
                raise ValueError("ETH Leica rotation is not orthogonal")
            poses.append(matrix)
    if len(poses) != SCAN_COUNT:
        raise ValueError("ETH Leica pose count mismatch")
    return tuple(poses)


def _relative_target_to_source(
    global_from_source: np.ndarray,
    global_from_target: np.ndarray,
) -> np.ndarray:
    return np.linalg.inv(global_from_source) @ global_from_target


def _rigid_errors(
    prediction_target_to_source: object,
    ground_truth_target_to_source: np.ndarray,
) -> tuple[float, float]:
    prediction = np.asarray(prediction_target_to_source, dtype=np.float64)
    if prediction.shape != (4, 4):
        raise ValueError("prediction transform must be 4x4")
    delta = np.linalg.inv(ground_truth_target_to_source) @ prediction
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    rotation_degrees = math.degrees(math.acos(cosine))
    translation_meters = float(np.linalg.norm(delta[:3, 3]))
    return rotation_degrees, translation_meters


def _summary(
    observations: tuple[ETHLabeledRotationObservation, ...],
) -> ETHFreshExternalSummary:
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
    return ETHFreshExternalSummary(
        scene_name="ETH Mountain Plain",
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


def evaluate_fresh_external_rotation_audit(
    preregistration_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
    archive_path: str | Path,
) -> ETHFreshExternalRotationAudit:
    preregistration_file = Path(preregistration_path)
    prediction_file = Path(prediction_path)
    decision_file = Path(decision_path)
    archive_file = Path(archive_path)
    protocol = _load_hash_locked_json(
        preregistration_file,
        expected_sha256=EXPECTED_PROTOCOL_SHA256,
        expected_schema="pftf_alpha_fresh_external_protocol_phase38/v1",
    )
    predictions = _load_hash_locked_json(
        prediction_file,
        expected_sha256=EXPECTED_PREDICTION_SHA256,
        expected_schema="pftf_alpha_eth_open3d_fgr_predictions_phase38/v1",
    )
    decisions = _load_hash_locked_json(
        decision_file,
        expected_sha256=EXPECTED_DECISION_SHA256,
        expected_schema="pftf_alpha_eth_rotation_decisions_phase38/v1",
    )
    expected_input_flags = {
        "protocol": protocol.get("label_values_accessed") is False,
        "predictions": (
            predictions.get("complete_prediction_set_materialized") is True
            and predictions.get("ground_truth_label_member_opened") is False
        ),
        "decisions": (
            decisions.get("complete_decision_set_materialized") is True
            and decisions.get("label_values_accessed") is False
        ),
    }
    if not all(expected_input_flags.values()):
        raise ValueError("Phase-38 pre-label provenance flags mismatch")
    raw_predictions = predictions.get("predictions")
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_predictions, list) or not isinstance(raw_decisions, list):
        raise ValueError("Phase-38 prediction or decision rows missing")
    if len(raw_predictions) != EXPECTED_PAIR_COUNT:
        raise ValueError("Phase-38 prediction count mismatch")
    if len(raw_decisions) != EXPECTED_PAIR_COUNT:
        raise ValueError("Phase-38 decision count mismatch")
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
        raise ValueError("Phase-38 prediction pair order mismatch")
    if tuple(decision_by_pair) != expected_pairs:
        raise ValueError("Phase-38 decision pair order mismatch")
    verification = verify_archive_directory(archive_file)
    with zipfile.ZipFile(archive_file) as source:
        with source.open(LABEL_MEMBER) as stream:
            poses = _parse_pose_labels(stream)
    observations: list[ETHLabeledRotationObservation] = []
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
            ETHLabeledRotationObservation(
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
    frozen_observations = tuple(observations)
    summary = _summary(frozen_observations)
    supported = summary.fresh_scene_transfer_gate_passed
    return ETHFreshExternalRotationAudit(
        artifact_schema="pftf_alpha_fresh_external_rotation_audit_phase38/v1",
        role="post_label_audit_of_hash_locked_prelabel_decisions",
        preregistration_artifact_path=str(preregistration_file),
        preregistration_artifact_sha256=EXPECTED_PROTOCOL_SHA256,
        prediction_artifact_path=str(prediction_file),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        decision_artifact_path=str(decision_file),
        decision_artifact_sha256=EXPECTED_DECISION_SHA256,
        prelabel_prediction_commit=PRELABEL_PREDICTION_COMMIT,
        archive_path=str(archive_file),
        archive_sha256=verification.sha256,
        label_member=LABEL_MEMBER,
        label_execution_order=(
            "verify hash-locked preregistration, complete predictions, and "
            "complete p90 decisions; only then open the single frozen Leica "
            "pose member and join correctness labels"
        ),
        matrix_convention=(
            "ETH poses map local scanner coordinates to global coordinates; "
            "ground-truth target-to-source is inv(global_from_source) @ "
            "global_from_target"
        ),
        correctness_rule=(
            "strict RRE < 15 degrees and strict RTE < 0.30 meters"
        ),
        maximum_relative_rotation_error_degrees=(
            MAX_RELATIVE_ROTATION_ERROR_DEGREES
        ),
        maximum_relative_translation_error_meters=(
            MAX_RELATIVE_TRANSLATION_ERROR_METERS
        ),
        rotation_percentile_cutoff=ROTATION_PERCENTILE_CUTOFF,
        minimum_correct_retention=MINIMUM_CORRECT_RETENTION,
        minimum_incorrect_rejection=MINIMUM_INCORRECT_REJECTION,
        observations=frozen_observations,
        summary=summary,
        label_values_accessed_by_evaluator=True,
        prelabel_prediction_artifact_verified=True,
        prelabel_decision_artifact_verified=True,
        fresh_label_blind_validation_supported=supported,
        fresh_external_pipeline_transfer_supported=supported,
        independent_algorithm_implementation_supported=False,
        real_registration_labels_supported=True,
        real_correspondence_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("benchmark-out/fresh_external_protocol_phase38.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("benchmark-out/eth_open3d_fgr_predictions_phase38.json"),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("benchmark-out/eth_rotation_decisions_phase38.json"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=(
            Path("benchmark-data/eth_mountain_plain")
            / "plain_01-Sep-2011-16_39_18.zip"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/fresh_external_rotation_audit_phase38.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_fresh_external_rotation_audit(
        args.preregistration,
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
