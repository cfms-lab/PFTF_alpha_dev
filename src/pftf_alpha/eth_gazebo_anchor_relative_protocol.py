"""Phase-42 preregistration for reserved anchor-relative Gazebo endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .eth_gazebo_local_support_calibration import (
    FSCORE_TOLERANCE,
    RECALL_TOLERANCE,
)
from .eth_gazebo_local_support_protocol import (
    VALIDATION_SOURCE_INDICES as PHASE41_VALIDATION_SOURCE_INDICES,
)
from .eth_gazebo_reconstruction_protocol import (
    DECISION_SCHEMA,
    EXPECTED_DECISION_SHA256,
    EXPECTED_PREDICTION_SHA256,
    PREDICTION_SCHEMA,
)
from .eth_gazebo_reconstruction_protocol import (
    VALIDATION_SOURCE_INDICES as PHASE40_VALIDATION_SOURCE_INDICES,
)
from .eth_gazebo_validation_protocol import (
    ARCHIVE_NAME,
    LABEL_MEMBER,
    verify_gazebo_archive_directory,
)

CALIBRATION_SCHEMA = (
    "pftf_alpha_eth_gazebo_anchor_relative_calibration_phase42/v1"
)
CALIBRATION_SHA256 = (
    "ddf119166d119e376acc21b0d73ba078616d6356a9a0f2093944dd7b1fb2f16f"
)
PROTOCOL_SCHEMA = "pftf_alpha_eth_gazebo_anchor_relative_protocol_phase42/v1"
DEVELOPMENT_SOURCE_INDICES = (0, 17)
VALIDATION_SOURCE_INDICES = (25, 26, 27)
MINIMUM_DIRECT_PAIR_COUNT = 3
SELECTED_CANDIDATE_ID = "anchor_d150_p050_n075"
SELECTED_MAXIMUM_NEAREST_ANCHOR_DISTANCE_METERS = 1.5
SELECTED_MAXIMUM_ANCHOR_PLANE_RESIDUAL_METERS = 0.5
SELECTED_MINIMUM_NORMAL_ALIGNMENT = 0.75
VALIDATION_SEED = 20_260_808


@dataclass(frozen=True)
class AnchorRelativeValidationSource:
    source_index: int
    pair_count: int
    accepted_pair_count: int
    rejected_pair_count: int


@dataclass(frozen=True)
class AnchorRelativeValidationProtocol:
    artifact_schema: str
    role: str
    archive_path: str
    archive_sha256: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    decision_artifact_path: str
    decision_artifact_sha256: str
    calibration_artifact_path: str
    calibration_artifact_sha256: str
    development_source_indices: tuple[int, ...]
    excluded_phase40_validation_sources: tuple[int, ...]
    excluded_phase41_validation_sources: tuple[int, ...]
    validation_sources: tuple[AnchorRelativeValidationSource, ...]
    source_selection_rule: str
    selected_candidate_id: str
    selected_maximum_nearest_anchor_distance_meters: float
    selected_maximum_anchor_plane_residual_meters: float
    selected_minimum_normal_alignment: float
    fscore_tolerance: float
    recall_tolerance: float
    validation_gate: str
    topology_role: str
    label_boundary: str
    prior_validation_references_accessed: bool
    validation_reference_values_accessed: bool
    registration_label_values_accessed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["development_source_indices"] = list(
            self.development_source_indices
        )
        payload["excluded_phase40_validation_sources"] = list(
            self.excluded_phase40_validation_sources
        )
        payload["excluded_phase41_validation_sources"] = list(
            self.excluded_phase41_validation_sources
        )
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


def _validation_sources(
    decisions: Sequence[Mapping[str, object]],
) -> tuple[AnchorRelativeValidationSource, ...]:
    totals: Counter[int] = Counter()
    accepted: Counter[int] = Counter()
    for row in decisions:
        source = int(row["source_index"])
        totals[source] += 1
        accepted[source] += bool(row["guarded_accept"])
    excluded = {
        *DEVELOPMENT_SOURCE_INDICES,
        *PHASE40_VALIDATION_SOURCE_INDICES,
        *PHASE41_VALIDATION_SOURCE_INDICES,
    }
    rows = tuple(
        AnchorRelativeValidationSource(
            source_index=source,
            pair_count=totals[source],
            accepted_pair_count=accepted[source],
            rejected_pair_count=totals[source] - accepted[source],
        )
        for source in sorted(totals)
        if source not in excluded and totals[source] >= MINIMUM_DIRECT_PAIR_COUNT
    )
    if tuple(row.source_index for row in rows) != VALIDATION_SOURCE_INDICES:
        raise ValueError("Phase-42 validation source set mismatch")
    if any(row.rejected_pair_count != 0 for row in rows):
        raise ValueError("Phase-42 reserved source decisions changed")
    return rows


def preregister_anchor_relative_validation(
    archive_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
    calibration_path: str | Path,
) -> AnchorRelativeValidationProtocol:
    archive = Path(archive_path)
    prediction_file = Path(prediction_path)
    decision_file = Path(decision_path)
    calibration_file = Path(calibration_path)
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
    calibration = _load_locked_json(
        calibration_file,
        expected_sha256=CALIBRATION_SHA256,
        expected_schema=CALIBRATION_SCHEMA,
    )
    if predictions.get("validation_label_member_opened") is not False:
        raise ValueError("prediction artifact crossed the registration label boundary")
    if decisions.get("validation_label_values_accessed") is not False:
        raise ValueError("decision artifact crossed the registration label boundary")
    expected_calibration = {
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_maximum_nearest_anchor_distance_meters": (
            SELECTED_MAXIMUM_NEAREST_ANCHOR_DISTANCE_METERS
        ),
        "selected_maximum_anchor_plane_residual_meters": (
            SELECTED_MAXIMUM_ANCHOR_PLANE_RESIDUAL_METERS
        ),
        "selected_minimum_normal_alignment": SELECTED_MINIMUM_NORMAL_ALIGNMENT,
        "calibration_viable": True,
        "topology_used_for_selection": False,
        "unopened_validation_references_accessed": False,
        "registration_label_values_accessed": False,
    }
    for key, value in expected_calibration.items():
        if calibration.get(key) != value:
            raise ValueError(f"Phase-42 calibration mismatch: {key}")
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("Gazebo decision rows are missing")
    sources = _validation_sources(raw_decisions)
    return AnchorRelativeValidationProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="pre_validation_anchor_relative_geometry_protocol",
        archive_path=str(archive),
        archive_sha256=verification.sha256,
        prediction_artifact_path=str(prediction_file),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        decision_artifact_path=str(decision_file),
        decision_artifact_sha256=EXPECTED_DECISION_SHA256,
        calibration_artifact_path=str(calibration_file),
        calibration_artifact_sha256=CALIBRATION_SHA256,
        development_source_indices=DEVELOPMENT_SOURCE_INDICES,
        excluded_phase40_validation_sources=PHASE40_VALIDATION_SOURCE_INDICES,
        excluded_phase41_validation_sources=PHASE41_VALIDATION_SOURCE_INDICES,
        validation_sources=sources,
        source_selection_rule=(
            "all remaining sources with at least three direct predictions after "
            "excluding development and Phase-40/41 endpoint panels"
        ),
        selected_candidate_id=SELECTED_CANDIDATE_ID,
        selected_maximum_nearest_anchor_distance_meters=(
            SELECTED_MAXIMUM_NEAREST_ANCHOR_DISTANCE_METERS
        ),
        selected_maximum_anchor_plane_residual_meters=(
            SELECTED_MAXIMUM_ANCHOR_PLANE_RESIDUAL_METERS
        ),
        selected_minimum_normal_alignment=SELECTED_MINIMUM_NORMAL_ALIGNMENT,
        fscore_tolerance=FSCORE_TOLERANCE,
        recall_tolerance=RECALL_TOLERANCE,
        validation_gate=(
            "all meshes materialize, every case adds at least one anchor-relative "
            "cell, mean geometry is below both anchor and Phase-41 means, mean "
            "F-score is at least anchor minus 0.025, and mean recall is at least "
            "anchor minus 0.01"
        ),
        topology_role="descriptive only; excluded from validation support",
        label_boundary=(
            "the evaluator may open only Hokuyo point-cloud members, never "
            f"{LABEL_MEMBER!r}; all prior endpoint panels and registration labels "
            "are excluded"
        ),
        prior_validation_references_accessed=False,
        validation_reference_values_accessed=False,
        registration_label_values_accessed=False,
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
        "--calibration",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_anchor_relative_calibration_phase42.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_anchor_relative_protocol_phase42.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preregister_anchor_relative_validation(
        args.archive,
        args.predictions,
        args.decisions,
        args.calibration,
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
