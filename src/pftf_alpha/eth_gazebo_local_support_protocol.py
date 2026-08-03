"""Phase-41 preregistration for reserved Gazebo local-support endpoints."""

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
    "pftf_alpha_eth_gazebo_local_support_calibration_phase41/v1"
)
CALIBRATION_SHA256 = (
    "dfc37f2bbc011e89646bd5a9a89744b9e065d78026b1dcdb58f900f64b18ecae"
)
PROTOCOL_SCHEMA = "pftf_alpha_eth_gazebo_local_support_protocol_phase41/v1"
DEVELOPMENT_SOURCE_INDEX = 0
VALIDATION_SOURCE_INDICES = (1, 17, 18, 21, 22, 23, 24)
MINIMUM_DIRECT_PAIR_COUNT = 6
SELECTED_CANDIDATE_ID = "support02_dispersion0150mm"
SELECTED_MINIMUM_SUPPORT = 2
SELECTED_MAXIMUM_DISPERSION_METERS = 0.15
VALIDATION_SEED = 20_260_806


@dataclass(frozen=True)
class LocalSupportValidationSource:
    source_index: int
    pair_count: int
    accepted_pair_count: int
    rejected_pair_count: int


@dataclass(frozen=True)
class LocalSupportValidationProtocol:
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
    development_source_index: int
    excluded_phase40_validation_sources: tuple[int, ...]
    validation_sources: tuple[LocalSupportValidationSource, ...]
    source_selection_rule: str
    selected_candidate_id: str
    selected_minimum_support: int
    selected_maximum_dispersion_meters: float
    fscore_tolerance: float
    recall_tolerance: float
    support_rule: str
    validation_gate: str
    topology_role: str
    label_boundary: str
    phase40_validation_references_accessed: bool
    validation_reference_values_accessed: bool
    registration_label_values_accessed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["excluded_phase40_validation_sources"] = list(
            self.excluded_phase40_validation_sources
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
) -> tuple[LocalSupportValidationSource, ...]:
    totals: Counter[int] = Counter()
    accepted: Counter[int] = Counter()
    for row in decisions:
        source = int(row["source_index"])
        totals[source] += 1
        accepted[source] += bool(row["guarded_accept"])
    excluded = {DEVELOPMENT_SOURCE_INDEX, *PHASE40_VALIDATION_SOURCE_INDICES}
    rows = tuple(
        LocalSupportValidationSource(
            source_index=source,
            pair_count=totals[source],
            accepted_pair_count=accepted[source],
            rejected_pair_count=totals[source] - accepted[source],
        )
        for source in sorted(totals)
        if source not in excluded and totals[source] >= MINIMUM_DIRECT_PAIR_COUNT
    )
    if tuple(row.source_index for row in rows) != VALIDATION_SOURCE_INDICES:
        raise ValueError("Phase-41 validation source set mismatch")
    if any(row.rejected_pair_count != 0 for row in rows):
        raise ValueError("Phase-41 reserved source decisions changed")
    return rows


def preregister_local_support_validation(
    archive_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
    calibration_path: str | Path,
) -> LocalSupportValidationProtocol:
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
    calibration_expected = {
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_minimum_support": SELECTED_MINIMUM_SUPPORT,
        "selected_maximum_dispersion_meters": (
            SELECTED_MAXIMUM_DISPERSION_METERS
        ),
        "calibration_viable": True,
        "topology_used_for_selection": False,
        "phase40_validation_references_accessed": False,
        "registration_label_values_accessed": False,
    }
    for key, value in calibration_expected.items():
        if calibration.get(key) != value:
            raise ValueError(f"Phase-41 calibration mismatch: {key}")
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("Gazebo decision rows are missing")
    sources = _validation_sources(raw_decisions)
    return LocalSupportValidationProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="pre_validation_point_local_support_protocol",
        archive_path=str(archive),
        archive_sha256=verification.sha256,
        prediction_artifact_path=str(prediction_file),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        decision_artifact_path=str(decision_file),
        decision_artifact_sha256=EXPECTED_DECISION_SHA256,
        calibration_artifact_path=str(calibration_file),
        calibration_artifact_sha256=CALIBRATION_SHA256,
        development_source_index=DEVELOPMENT_SOURCE_INDEX,
        excluded_phase40_validation_sources=PHASE40_VALIDATION_SOURCE_INDICES,
        validation_sources=sources,
        source_selection_rule=(
            "all non-development, non-Phase-40 sources with at least six direct "
            "target predictions; the frozen set is 1,17,18,21,22,23,24"
        ),
        selected_candidate_id=SELECTED_CANDIDATE_ID,
        selected_minimum_support=SELECTED_MINIMUM_SUPPORT,
        selected_maximum_dispersion_meters=(
            SELECTED_MAXIMUM_DISPERSION_METERS
        ),
        fscore_tolerance=FSCORE_TOLERANCE,
        recall_tolerance=RECALL_TOLERANCE,
        support_rule=(
            "retain every anchor voxel; add a target-only 0.75 m voxel only when "
            "at least two distinct accepted scans contribute and RMS dispersion "
            "is at most 0.15 m"
        ),
        validation_gate=(
            "all meshes materialize, every case exercises at least one added "
            "target-only cell, mean local geometry loss is below both mean "
            "baselines, mean F-score is at least anchor minus 0.025, and mean "
            "recall is at least anchor minus 0.01"
        ),
        topology_role=(
            "topology endpoints are descriptive and excluded from validation support"
        ),
        label_boundary=(
            "the evaluator may open only Hokuyo point-cloud members, never "
            f"{LABEL_MEMBER!r}; "
            "Phase-40 validation endpoints and registration labels are excluded"
        ),
        phase40_validation_references_accessed=False,
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
        default=Path("benchmark-out/eth_gazebo_local_support_calibration_phase41.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_local_support_protocol_phase41.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preregister_local_support_validation(
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
