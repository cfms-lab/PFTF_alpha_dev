"""Materialize Phase-39 Gazebo rotation decisions before labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .eth_gazebo_validation_protocol import EXPECTED_PAIR_COUNT
from .scene_relative_rotation_guard import (
    ROTATION_PERCENTILE_CUTOFF,
    empirical_midrank_percentiles,
    prediction_rotation_radians,
)

EXPECTED_PREDICTION_SHA256 = (
    "ed25ac05393d3a9270bef04e99bf79870b8eddd4c0ba6cb0e45d7bff2931900e"
)


@dataclass(frozen=True)
class GazeboBlindDecision:
    source_index: int
    target_index: int
    prediction_rotation_radians: float
    scene_relative_rotation_percentile: float
    guarded_accept: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GazeboBlindDecisionArtifact:
    artifact_schema: str
    role: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    rotation_percentile_cutoff: float
    percentile_method: str
    tie_policy: str
    expected_pair_count: int
    decisions: tuple[GazeboBlindDecision, ...]
    accepted_count: int
    rejected_count: int
    complete_decision_set_materialized: bool
    label_boundary: str
    validation_label_values_accessed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "decisions": [row.to_dict() for row in self.decisions],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_predictions(path: Path) -> Mapping[str, object]:
    if _sha256(path) != EXPECTED_PREDICTION_SHA256:
        raise ValueError("Gazebo prediction artifact SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gazebo prediction artifact must be an object")
    expected = {
        "artifact_schema": "pftf_alpha_eth_gazebo_predictions_phase39/v1",
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "complete_prediction_set_materialized": True,
        "validation_label_member_opened": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Gazebo prediction mismatch: {key}")
    return payload


def materialize_gazebo_decisions(
    prediction_path: str | Path,
) -> GazeboBlindDecisionArtifact:
    path = Path(prediction_path)
    payload = _load_predictions(path)
    raw_predictions = payload.get("predictions")
    if not isinstance(raw_predictions, list):
        raise ValueError("Gazebo predictions are missing")
    if len(raw_predictions) != EXPECTED_PAIR_COUNT:
        raise ValueError("Gazebo prediction count mismatch")
    angles = tuple(
        prediction_rotation_radians(row["target_to_source_matrix"])
        for row in raw_predictions
    )
    percentiles = empirical_midrank_percentiles(angles)
    decisions = tuple(
        GazeboBlindDecision(
            source_index=int(row["source_index"]),
            target_index=int(row["target_index"]),
            prediction_rotation_radians=angle,
            scene_relative_rotation_percentile=float(percentile),
            guarded_accept=bool(percentile < ROTATION_PERCENTILE_CUTOFF),
        )
        for row, angle, percentile in zip(
            raw_predictions,
            angles,
            percentiles,
            strict=True,
        )
    )
    accepted = sum(row.guarded_accept for row in decisions)
    return GazeboBlindDecisionArtifact(
        artifact_schema="pftf_alpha_eth_gazebo_rotation_decisions_phase39/v1",
        role="pre_label_complete_rotation_decisions",
        prediction_artifact_path=str(path),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        rotation_percentile_cutoff=ROTATION_PERCENTILE_CUTOFF,
        percentile_method="within-scene empirical midrank percentile",
        tie_policy="strict percentile < 0.90; equal rotations share midrank",
        expected_pair_count=EXPECTED_PAIR_COUNT,
        decisions=decisions,
        accepted_count=accepted,
        rejected_count=len(decisions) - accepted,
        complete_decision_set_materialized=(
            len(decisions) == EXPECTED_PAIR_COUNT
        ),
        label_boundary=(
            "program accepts only the hash-locked prediction JSON and has no "
            "archive or label path argument"
        ),
        validation_label_values_accessed=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_predictions_phase39.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_rotation_decisions_phase39.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_gazebo_decisions(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
