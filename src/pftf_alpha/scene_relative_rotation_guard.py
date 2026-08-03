"""Phase-34 scene-relative rotation-tail guard design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

EXPECTED_PHASE32_SHA256 = (
    "b7653adda0f0b93f14fda54bb57a4559c4a00863e4f22702b4bc14650442cb4d"
)
EXPECTED_PHASE33_SHA256 = (
    "961773176cd10cd41e6054b2b898f02af2c9c357a28401911c05189b2bedd5fa"
)
ROTATION_PERCENTILE_CUTOFF = 0.90
MINIMUM_CORRECT_RETENTION = 0.90
MINIMUM_INCORRECT_REJECTION = 0.10

DESIGN_SCENES = (
    "7-scenes-redkitchen",
    "sun3d-hotel_umd-maryland_hotel3",
)
UNTOUCHED_VALIDATION_SCENES = (
    "sun3d-home_at-home_at_scan1_2013_jan_1",
    "sun3d-home_md-home_md_scan9_2012_sep_30",
    "sun3d-hotel_uc-scan3",
    "sun3d-hotel_umd-maryland_hotel1",
    "sun3d-mit_76_studyroom-76-1studyroom2",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
)


@dataclass(frozen=True)
class BlindRotationObservation:
    scene_name: str
    source_index: int
    target_index: int
    prediction_rotation_radians: float
    scene_relative_rotation_percentile: float
    guarded_accept: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LabeledRotationObservation:
    blind: BlindRotationObservation
    official_correct: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **self.blind.to_dict(),
            "official_correct": self.official_correct,
        }


@dataclass(frozen=True)
class RotationGuardSceneSummary:
    scene_name: str
    prediction_count: int
    ground_truth_overlap_pair_count: int
    base_correct_count: int
    base_incorrect_count: int
    base_precision: float
    base_recall: float
    guarded_accepted_count: int
    guarded_correct_count: int
    guarded_incorrect_count: int
    guarded_precision: float
    guarded_recall: float
    correct_retention: float
    incorrect_rejection: float
    precision_improved: bool
    correct_retention_gate_passed: bool
    incorrect_rejection_gate_passed: bool
    scene_design_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SceneRelativeRotationGuardResult:
    artifact_schema: str
    role: str
    feature_name: str
    information_boundary: str
    label_blind_execution_order: str
    selection_history: str
    phase32_artifact_path: str
    phase32_artifact_sha256: str
    phase33_artifact_path: str
    phase33_artifact_sha256: str
    design_scenes: tuple[str, ...]
    untouched_validation_scenes: tuple[str, ...]
    rotation_percentile_cutoff: float
    minimum_correct_retention: float
    minimum_incorrect_rejection: float
    observations: tuple[LabeledRotationObservation, ...]
    scene_summaries: tuple[RotationGuardSceneSummary, ...]
    phase34_design_supported: bool
    held_out_validation_artifacts_accessed: bool
    held_out_validation_supported: bool
    cross_scene_real_registration_supported: bool
    real_correspondence_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "feature_name": self.feature_name,
            "information_boundary": self.information_boundary,
            "label_blind_execution_order": self.label_blind_execution_order,
            "selection_history": self.selection_history,
            "phase32_artifact_path": self.phase32_artifact_path,
            "phase32_artifact_sha256": self.phase32_artifact_sha256,
            "phase33_artifact_path": self.phase33_artifact_path,
            "phase33_artifact_sha256": self.phase33_artifact_sha256,
            "design_scenes": list(self.design_scenes),
            "untouched_validation_scenes": list(
                self.untouched_validation_scenes
            ),
            "rotation_percentile_cutoff": self.rotation_percentile_cutoff,
            "minimum_correct_retention": self.minimum_correct_retention,
            "minimum_incorrect_rejection": self.minimum_incorrect_rejection,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "scene_summaries": [
                summary.to_dict() for summary in self.scene_summaries
            ],
            "phase34_design_supported": self.phase34_design_supported,
            "held_out_validation_artifacts_accessed": (
                self.held_out_validation_artifacts_accessed
            ),
            "held_out_validation_supported": (
                self.held_out_validation_supported
            ),
            "cross_scene_real_registration_supported": (
                self.cross_scene_real_registration_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_trimmed_reconstruction_supported": (
                self.real_trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_schema: str,
) -> tuple[Path, Mapping[str, object]]:
    resolved = Path(path)
    observed_sha256 = _sha256(resolved)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"artifact SHA-256 mismatch: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact must contain a JSON object")
    if payload.get("artifact_schema") != expected_schema:
        raise ValueError("artifact schema mismatch")
    if payload.get("real_registration_labels_supported") is not True:
        raise ValueError("design artifact must contain real registration labels")
    return resolved, payload


def prediction_rotation_radians(matrix: object) -> float:
    """Return the principal rotation angle of one proper rigid transform."""

    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("prediction matrix must be finite and 4x4")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
        raise ValueError("prediction matrix must be affine")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("prediction rotation must be orthogonal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
        raise ValueError("prediction rotation must be proper")
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def empirical_midrank_percentiles(values: Sequence[float]) -> np.ndarray:
    """Return deterministic midrank percentiles in the open interval (0, 1)."""

    selected = np.asarray(tuple(values), dtype=np.float64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(selected)):
        raise ValueError("rank values must be finite")
    _, inverse, counts = np.unique(
        selected,
        return_inverse=True,
        return_counts=True,
    )
    starts = np.cumsum(np.concatenate(([0], counts[:-1])))
    midranks = starts + (counts + 1.0) / 2.0
    return np.ascontiguousarray((midranks[inverse] - 0.5) / selected.size)


def _blind_scene_observations(
    scene_name: str,
    payload: Mapping[str, object],
) -> tuple[BlindRotationObservation, ...]:
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("design artifact has no observations")
    angles = tuple(
        prediction_rotation_radians(observation["prediction_matrix"])
        for observation in observations
    )
    percentiles = empirical_midrank_percentiles(angles)
    return tuple(
        BlindRotationObservation(
            scene_name=scene_name,
            source_index=int(observation["source_index"]),
            target_index=int(observation["target_index"]),
            prediction_rotation_radians=angle,
            scene_relative_rotation_percentile=float(percentile),
            guarded_accept=bool(percentile < ROTATION_PERCENTILE_CUTOFF),
        )
        for observation, angle, percentile in zip(
            observations,
            angles,
            percentiles,
            strict=True,
        )
    )


def _join_scene_labels(
    blind: Sequence[BlindRotationObservation],
    payload: Mapping[str, object],
) -> tuple[LabeledRotationObservation, ...]:
    raw = payload["observations"]
    if len(blind) != len(raw):
        raise ValueError("blind observations and labels must align")
    return tuple(
        LabeledRotationObservation(
            blind=observation,
            official_correct=bool(source["official_correct"]),
        )
        for observation, source in zip(blind, raw, strict=True)
    )


def _scene_summary(
    scene_name: str,
    observations: Sequence[LabeledRotationObservation],
    ground_truth_overlap_pair_count: int,
) -> RotationGuardSceneSummary:
    if ground_truth_overlap_pair_count <= 0:
        raise ValueError("ground-truth overlap pair count must be positive")
    rows = tuple(observations)
    base_correct = sum(row.official_correct for row in rows)
    base_incorrect = len(rows) - base_correct
    if base_correct == 0 or base_incorrect == 0:
        raise ValueError("scene must contain correct and incorrect predictions")
    guarded = tuple(row for row in rows if row.blind.guarded_accept)
    guarded_correct = sum(row.official_correct for row in guarded)
    guarded_incorrect = len(guarded) - guarded_correct
    base_precision = base_correct / len(rows)
    guarded_precision = guarded_correct / len(guarded) if guarded else 0.0
    correct_retention = guarded_correct / base_correct
    incorrect_rejection = 1.0 - guarded_incorrect / base_incorrect
    precision_improved = bool(guarded_precision > base_precision)
    correct_gate = bool(correct_retention >= MINIMUM_CORRECT_RETENTION)
    incorrect_gate = bool(
        incorrect_rejection >= MINIMUM_INCORRECT_REJECTION
    )
    return RotationGuardSceneSummary(
        scene_name=scene_name,
        prediction_count=len(rows),
        ground_truth_overlap_pair_count=ground_truth_overlap_pair_count,
        base_correct_count=base_correct,
        base_incorrect_count=base_incorrect,
        base_precision=base_precision,
        base_recall=base_correct / ground_truth_overlap_pair_count,
        guarded_accepted_count=len(guarded),
        guarded_correct_count=guarded_correct,
        guarded_incorrect_count=guarded_incorrect,
        guarded_precision=guarded_precision,
        guarded_recall=guarded_correct / ground_truth_overlap_pair_count,
        correct_retention=correct_retention,
        incorrect_rejection=incorrect_rejection,
        precision_improved=precision_improved,
        correct_retention_gate_passed=correct_gate,
        incorrect_rejection_gate_passed=incorrect_gate,
        scene_design_gate_passed=bool(
            precision_improved and correct_gate and incorrect_gate
        ),
    )


def evaluate_scene_relative_rotation_guard(
    phase32_artifact: str | Path,
    phase33_artifact: str | Path,
) -> SceneRelativeRotationGuardResult:
    """Fit no parameters; evaluate the frozen 90th-percentile design rule."""

    phase32_path, phase32 = _verified_artifact(
        phase32_artifact,
        expected_sha256=EXPECTED_PHASE32_SHA256,
        expected_schema="pftf_alpha_threedmatch_registration_guard_phase32/v1",
    )
    phase33_path, phase33 = _verified_artifact(
        phase33_artifact,
        expected_sha256=EXPECTED_PHASE33_SHA256,
        expected_schema="pftf_alpha_threedmatch_transfer_audit_phase33/v1",
    )
    if phase32.get("dataset_name") != DESIGN_SCENES[0]:
        raise ValueError("Phase-32 design scene identity mismatch")
    phase33_scene = phase33.get("scene")
    if not isinstance(phase33_scene, dict):
        raise ValueError("Phase-33 scene identity is missing")
    if phase33_scene.get("scene_name") != DESIGN_SCENES[1]:
        raise ValueError("Phase-33 design scene identity mismatch")

    # Both complete decision sets are materialized before either label is read.
    blind_by_scene = (
        _blind_scene_observations(DESIGN_SCENES[0], phase32),
        _blind_scene_observations(DESIGN_SCENES[1], phase33),
    )
    labeled_by_scene = (
        _join_scene_labels(blind_by_scene[0], phase32),
        _join_scene_labels(blind_by_scene[1], phase33),
    )
    payloads = (phase32, phase33)
    summaries = tuple(
        _scene_summary(
            scene_name,
            labeled,
            int(payload["ground_truth_overlap_pair_count"]),
        )
        for scene_name, labeled, payload in zip(
            DESIGN_SCENES,
            labeled_by_scene,
            payloads,
            strict=True,
        )
    )
    design_supported = all(
        summary.scene_design_gate_passed for summary in summaries
    )
    return SceneRelativeRotationGuardResult(
        artifact_schema="pftf_alpha_scene_relative_rotation_guard_phase34/v1",
        role="opened_scene_design_only_before_untouched_validation",
        feature_name="scene_relative_prediction_rotation_midrank_percentile",
        information_boundary=(
            "external_prediction_rotation_matrices_and_the_unlabeled_within_"
            "scene_prediction_set_only_for_guard_decisions; opened_design_"
            "labels_joined_after_both_scene_decision_sets; no_validation_"
            "archive_or_label_access"
        ),
        label_blind_execution_order=(
            "materialize_rotation_angles_midrank_percentiles_and_guard_"
            "decisions_for_both_design_scenes_then_join_design_labels"
        ),
        selection_history=(
            "chosen during Phase-34 design after absolute summary-ridge and "
            "spatial-footprint ridge candidates failed bidirectional transfer; "
            "not preregistered evidence and requires untouched validation"
        ),
        phase32_artifact_path=str(phase32_path),
        phase32_artifact_sha256=_sha256(phase32_path),
        phase33_artifact_path=str(phase33_path),
        phase33_artifact_sha256=_sha256(phase33_path),
        design_scenes=DESIGN_SCENES,
        untouched_validation_scenes=UNTOUCHED_VALIDATION_SCENES,
        rotation_percentile_cutoff=ROTATION_PERCENTILE_CUTOFF,
        minimum_correct_retention=MINIMUM_CORRECT_RETENTION,
        minimum_incorrect_rejection=MINIMUM_INCORRECT_REJECTION,
        observations=tuple(
            observation
            for scene_rows in labeled_by_scene
            for observation in scene_rows
        ),
        scene_summaries=summaries,
        phase34_design_supported=design_supported,
        held_out_validation_artifacts_accessed=False,
        held_out_validation_supported=False,
        cross_scene_real_registration_supported=False,
        real_correspondence_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase32-artifact",
        type=Path,
        default=Path(
            "benchmark-out/threedmatch_registration_guard_phase32.json"
        ),
    )
    parser.add_argument(
        "--phase33-artifact",
        type=Path,
        default=Path("benchmark-out/threedmatch_transfer_audit_phase33.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/scene_relative_rotation_guard_phase34.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_scene_relative_rotation_guard(
        args.phase32_artifact,
        args.phase33_artifact,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
