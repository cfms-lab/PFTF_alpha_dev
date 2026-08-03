"""Phase-35 held-out validation of the frozen scene-relative rotation guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .scene_relative_rotation_guard import (
    MINIMUM_CORRECT_RETENTION,
    MINIMUM_INCORRECT_REJECTION,
    ROTATION_PERCENTILE_CUTOFF,
    UNTOUCHED_VALIDATION_SCENES,
    BlindRotationObservation,
    empirical_midrank_percentiles,
    prediction_rotation_radians,
)
from .threedmatch_redkitchen import (
    RegistrationInfoEntry,
    RegistrationLogEntry,
    ThreeDMatchArchiveVerification,
    official_transformation_error,
    parse_registration_info,
    parse_registration_log,
)
from .threedmatch_registration_guard import OFFICIAL_ERROR_THRESHOLD_SQUARED

EXPECTED_PHASE34_SHA256 = (
    "805f056fdf50c80aa89fd74d1bba67968ab8405279b481bf9051494f655ea9d8"
)
OFFICIAL_BENCHMARK_URL = "https://3dmatch.cs.princeton.edu/"
ARCHIVE_URL_ROOT = (
    "https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/"
    "scene-fragments"
)
SUN3D_LICENSE_BOUNDARY = (
    "the accessed 3DMatch and SUN3D pages request dataset citation but do not "
    "state an explicit SUN3D data license; do not redistribute archives"
)


@dataclass(frozen=True)
class ThreeDMatchEvaluationSpec:
    """Exact identity of one evaluation-only held-out scene archive."""

    scene_name: str
    evaluation_name: str
    archive_name: str
    url: str
    md5: str
    sha256: str
    dataset_source: str = "SUN3D via the official 3DMatch benchmark"
    dataset_license_boundary: str = SUN3D_LICENSE_BOUNDARY

    def __post_init__(self) -> None:
        text_fields = (
            self.scene_name,
            self.evaluation_name,
            self.archive_name,
            self.url,
            self.dataset_source,
            self.dataset_license_boundary,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("evaluation specification fields must be non-empty")
        for label, value, length in (
            ("MD5", self.md5, 32),
            ("SHA-256", self.sha256, 64),
        ):
            if len(value) != length or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{label} must be lowercase hexadecimal")

    @property
    def expected_members(self) -> tuple[str, ...]:
        return (
            f"{self.evaluation_name}/",
            f"{self.evaluation_name}/3dmatch.log",
            f"{self.evaluation_name}/gt.info",
            f"{self.evaluation_name}/gt.log",
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _evaluation_spec(
    scene_name: str,
    *,
    md5: str,
    sha256: str,
) -> ThreeDMatchEvaluationSpec:
    evaluation_name = f"{scene_name}-evaluation"
    archive_name = f"{evaluation_name}.zip"
    return ThreeDMatchEvaluationSpec(
        scene_name=scene_name,
        evaluation_name=evaluation_name,
        archive_name=archive_name,
        url=f"{ARCHIVE_URL_ROOT}/{archive_name}",
        md5=md5,
        sha256=sha256,
    )


PHASE35_VALIDATION_SCENES = (
    _evaluation_spec(
        "sun3d-home_at-home_at_scan1_2013_jan_1",
        md5="ce04d973763ea01a86c042dffc356724",
        sha256=(
            "0645890e675444a7b6a49e1e3ac1c443a55a761b65fb275dd5308980a200f757"
        ),
    ),
    _evaluation_spec(
        "sun3d-home_md-home_md_scan9_2012_sep_30",
        md5="657fbdd5d3f75313b017ffee505cb4d8",
        sha256=(
            "272b83fcb74cb0faedad4c9f614eb9eedac83bc000de60591422d3daa332cecf"
        ),
    ),
    _evaluation_spec(
        "sun3d-hotel_uc-scan3",
        md5="9d6dc696247d5f462ac08b6cbc3a479e",
        sha256=(
            "cc07c279a355756167dcd850184031f59578b56ed916c732020c47ef7420e957"
        ),
    ),
    _evaluation_spec(
        "sun3d-hotel_umd-maryland_hotel1",
        md5="f5a4488f41ec5ce2004d77063e8bf5e5",
        sha256=(
            "1a7822280320c5b6652f30584735ce518af4f3bfd311f2e0a98696d2bbf93c70"
        ),
    ),
    _evaluation_spec(
        "sun3d-mit_76_studyroom-76-1studyroom2",
        md5="bcdd48d415d54b9e51b3f1998ea7b2f9",
        sha256=(
            "f3f49cb14224e777dbb64244a6efae9e43a2d07fc69469e04fd650832535b14e"
        ),
    ),
    _evaluation_spec(
        "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
        md5="d11b00d76ae0b253abb4435ec31c62ea",
        sha256=(
            "4dce89e4d24ba883422cf6b782497889a9ed02f01e5ea0c7f21d19590e128660"
        ),
    ),
)


@dataclass(frozen=True)
class ValidationRotationObservation:
    blind: BlindRotationObservation
    ground_truth_overlap_pair: bool
    official_transformation_error: float | None
    official_correct: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **self.blind.to_dict(),
            "ground_truth_overlap_pair": self.ground_truth_overlap_pair,
            "official_transformation_error": self.official_transformation_error,
            "official_correct": self.official_correct,
        }


@dataclass(frozen=True)
class ValidationRotationSceneSummary:
    scene_name: str
    raw_prediction_count: int
    eligible_prediction_count: int
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
    precision_gain_percentage_points: float
    correct_retention: float
    incorrect_rejection: float
    precision_improved: bool
    correct_retention_gate_passed: bool
    incorrect_rejection_gate_passed: bool
    scene_validation_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SceneRelativeRotationValidationResult:
    artifact_schema: str
    role: str
    dataset_name: str
    dataset_source_url: str
    feature_name: str
    information_boundary: str
    label_blind_execution_order: str
    frozen_validation_contract: str
    phase34_artifact_path: str
    phase34_artifact_sha256: str
    validation_scene_specs: tuple[ThreeDMatchEvaluationSpec, ...]
    evaluation_archives: tuple[ThreeDMatchArchiveVerification, ...]
    rotation_percentile_cutoff: float
    minimum_correct_retention: float
    minimum_incorrect_rejection: float
    official_error_threshold_squared: float
    observations: tuple[ValidationRotationObservation, ...]
    scene_summaries: tuple[ValidationRotationSceneSummary, ...]
    phase34_design_supported: bool
    held_out_validation_artifacts_accessed: bool
    phase35_validation_supported: bool
    held_out_validation_supported: bool
    cross_scene_real_registration_supported: bool
    real_registration_labels_supported: bool
    real_correspondence_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "dataset_name": self.dataset_name,
            "dataset_source_url": self.dataset_source_url,
            "feature_name": self.feature_name,
            "information_boundary": self.information_boundary,
            "label_blind_execution_order": self.label_blind_execution_order,
            "frozen_validation_contract": self.frozen_validation_contract,
            "phase34_artifact_path": self.phase34_artifact_path,
            "phase34_artifact_sha256": self.phase34_artifact_sha256,
            "validation_scene_specs": [
                scene.to_dict() for scene in self.validation_scene_specs
            ],
            "evaluation_archives": [
                archive.to_dict() for archive in self.evaluation_archives
            ],
            "rotation_percentile_cutoff": self.rotation_percentile_cutoff,
            "minimum_correct_retention": self.minimum_correct_retention,
            "minimum_incorrect_rejection": self.minimum_incorrect_rejection,
            "official_error_threshold_squared": (
                self.official_error_threshold_squared
            ),
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
            "phase35_validation_supported": self.phase35_validation_supported,
            "held_out_validation_supported": self.held_out_validation_supported,
            "cross_scene_real_registration_supported": (
                self.cross_scene_real_registration_supported
            ),
            "real_registration_labels_supported": (
                self.real_registration_labels_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_trimmed_reconstruction_supported": (
                self.real_trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_phase34_artifact(
    path: str | Path,
    scenes: Sequence[ThreeDMatchEvaluationSpec],
) -> tuple[Path, Mapping[str, object]]:
    resolved = Path(path)
    observed_sha256 = _hash_file(resolved, "sha256")
    if observed_sha256 != EXPECTED_PHASE34_SHA256:
        raise ValueError(f"Phase-34 artifact SHA-256 mismatch: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase-34 artifact must contain a JSON object")
    expected = {
        "artifact_schema": "pftf_alpha_scene_relative_rotation_guard_phase34/v1",
        "feature_name": "scene_relative_prediction_rotation_midrank_percentile",
        "rotation_percentile_cutoff": ROTATION_PERCENTILE_CUTOFF,
        "minimum_correct_retention": MINIMUM_CORRECT_RETENTION,
        "minimum_incorrect_rejection": MINIMUM_INCORRECT_REJECTION,
        "phase34_design_supported": True,
        "held_out_validation_artifacts_accessed": False,
        "held_out_validation_supported": False,
        "cross_scene_real_registration_supported": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase-34 frozen field mismatch: {key}")
    scene_names = tuple(scene.scene_name for scene in scenes)
    if tuple(payload.get("untouched_validation_scenes", ())) != scene_names:
        raise ValueError("Phase-34 validation scene panel mismatch")
    if scene_names != UNTOUCHED_VALIDATION_SCENES:
        raise ValueError("Phase-35 scene order differs from the frozen panel")
    return resolved, payload


def verify_evaluation_archive(
    archive_path: str | Path,
    scene: ThreeDMatchEvaluationSpec,
) -> ThreeDMatchArchiveVerification:
    """Verify exact archive identity and its central-directory allowlist."""

    resolved = Path(archive_path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    md5 = _hash_file(resolved, "md5")
    sha256 = _hash_file(resolved, "sha256")
    if md5 != scene.md5:
        raise ValueError(f"{scene.scene_name} evaluation archive MD5 mismatch")
    if sha256 != scene.sha256:
        raise ValueError(
            f"{scene.scene_name} evaluation archive SHA-256 mismatch"
        )
    with zipfile.ZipFile(resolved) as archive:
        infos = archive.infolist()
        members = tuple(info.filename for info in infos)
        if set(members) != set(scene.expected_members) or len(members) != len(
            scene.expected_members
        ):
            raise ValueError(
                f"unexpected {scene.scene_name} evaluation archive members"
            )
        for info in infos:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe evaluation archive path: {info.filename}")
            if info.is_dir() != info.filename.endswith("/"):
                raise ValueError(f"ambiguous archive member: {info.filename}")
    return ThreeDMatchArchiveVerification(
        role="evaluation",
        archive_path=str(resolved),
        byte_count=resolved.stat().st_size,
        md5=md5,
        sha256=sha256,
        file_count=sum(not info.is_dir() for info in infos),
        verified=True,
    )


def _fetch_evaluation_archive(
    data_root: Path,
    scene: ThreeDMatchEvaluationSpec,
) -> Path:
    target = data_root / scene.archive_name
    if target.exists():
        verify_evaluation_archive(target, scene)
        return target
    data_root.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"stale partial download requires review: {partial}")
    try:
        with urllib.request.urlopen(scene.url, timeout=120) as response:  # noqa: S310
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        verify_evaluation_archive(partial, scene)
        os.replace(partial, target)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return target


def fetch_phase35_evaluation_archives(
    data_root: str | Path,
    scenes: Sequence[ThreeDMatchEvaluationSpec] = PHASE35_VALIDATION_SCENES,
) -> tuple[Path, ...]:
    """Fetch only the evaluation archives needed by the rotation-only rule."""

    root = Path(data_root)
    return tuple(_fetch_evaluation_archive(root, scene) for scene in scenes)


def _read_archive_member(
    archive_path: Path,
    scene: ThreeDMatchEvaluationSpec,
    member_name: str,
) -> str:
    member = f"{scene.evaluation_name}/{member_name}"
    with zipfile.ZipFile(archive_path) as archive:
        try:
            payload = archive.read(member)
        except KeyError as error:
            raise ValueError(f"missing evaluation member: {member}") from error
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"evaluation member is not ASCII: {member}") from error


def _blind_validation_scene(
    scene: ThreeDMatchEvaluationSpec,
    predictions: Sequence[RegistrationLogEntry],
) -> tuple[BlindRotationObservation, ...]:
    angles = tuple(
        prediction_rotation_radians(prediction.source_to_target_matrix)
        for prediction in predictions
    )
    percentiles = empirical_midrank_percentiles(angles)
    return tuple(
        BlindRotationObservation(
            scene_name=scene.scene_name,
            source_index=prediction.source_index,
            target_index=prediction.target_index,
            prediction_rotation_radians=angle,
            scene_relative_rotation_percentile=float(percentile),
            guarded_accept=bool(percentile < ROTATION_PERCENTILE_CUTOFF),
        )
        for prediction, angle, percentile in zip(
            predictions,
            angles,
            percentiles,
            strict=True,
        )
    )


def _read_labeled_members(
    archive_path: Path,
    scene: ThreeDMatchEvaluationSpec,
) -> tuple[tuple[RegistrationLogEntry, ...], tuple[RegistrationInfoEntry, ...]]:
    ground_truth = parse_registration_log(
        _read_archive_member(archive_path, scene, "gt.log")
    )
    information = parse_registration_info(
        _read_archive_member(archive_path, scene, "gt.info")
    )
    return ground_truth, information


def _label_validation_scene(
    blind: Sequence[BlindRotationObservation],
    predictions: Sequence[RegistrationLogEntry],
    ground_truth: Sequence[RegistrationLogEntry],
    information: Sequence[RegistrationInfoEntry],
) -> tuple[tuple[ValidationRotationObservation, ...], int]:
    eligible_ground_truth = {
        entry.pair: entry
        for entry in ground_truth
        if entry.target_index - entry.source_index > 1
    }
    eligible_information = {
        entry.pair: entry
        for entry in information
        if entry.target_index - entry.source_index > 1
    }
    if eligible_ground_truth.keys() != eligible_information.keys():
        raise ValueError("eligible ground-truth log and info pairs disagree")
    if len(blind) != len(predictions):
        raise ValueError("blind observations and predictions must align")
    labeled = []
    for observation, prediction in zip(blind, predictions, strict=True):
        if observation.source_index != prediction.source_index or (
            observation.target_index != prediction.target_index
        ):
            raise ValueError("blind observations and prediction pairs disagree")
        pair = prediction.pair
        if pair in eligible_ground_truth:
            error = official_transformation_error(
                eligible_ground_truth[pair],
                prediction,
                eligible_information[pair],
            )
            correct = bool(error <= OFFICIAL_ERROR_THRESHOLD_SQUARED)
        else:
            error = None
            correct = False
        labeled.append(
            ValidationRotationObservation(
                blind=observation,
                ground_truth_overlap_pair=pair in eligible_ground_truth,
                official_transformation_error=error,
                official_correct=correct,
            )
        )
    return tuple(labeled), len(eligible_ground_truth)


def _scene_summary(
    scene_name: str,
    *,
    raw_prediction_count: int,
    observations: Sequence[ValidationRotationObservation],
    ground_truth_overlap_pair_count: int,
) -> ValidationRotationSceneSummary:
    rows = tuple(observations)
    base_correct = sum(row.official_correct for row in rows)
    base_incorrect = len(rows) - base_correct
    if not rows or base_correct == 0 or base_incorrect == 0:
        raise ValueError("validation scene requires correct and incorrect predictions")
    if ground_truth_overlap_pair_count <= 0:
        raise ValueError("ground-truth overlap pair count must be positive")
    guarded = tuple(row for row in rows if row.blind.guarded_accept)
    guarded_correct = sum(row.official_correct for row in guarded)
    guarded_incorrect = len(guarded) - guarded_correct
    base_precision = base_correct / len(rows)
    guarded_precision = guarded_correct / len(guarded) if guarded else 0.0
    correct_retention = guarded_correct / base_correct
    incorrect_rejection = 1.0 - guarded_incorrect / base_incorrect
    precision_improved = bool(guarded_precision > base_precision)
    correct_gate = bool(correct_retention >= MINIMUM_CORRECT_RETENTION)
    incorrect_gate = bool(incorrect_rejection >= MINIMUM_INCORRECT_REJECTION)
    validation_gate = bool(precision_improved and correct_gate and incorrect_gate)
    return ValidationRotationSceneSummary(
        scene_name=scene_name,
        raw_prediction_count=raw_prediction_count,
        eligible_prediction_count=len(rows),
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
        precision_gain_percentage_points=(
            100.0 * (guarded_precision - base_precision)
        ),
        correct_retention=correct_retention,
        incorrect_rejection=incorrect_rejection,
        precision_improved=precision_improved,
        correct_retention_gate_passed=correct_gate,
        incorrect_rejection_gate_passed=incorrect_gate,
        scene_validation_gate_passed=validation_gate,
    )


def evaluate_scene_relative_rotation_validation(
    data_root: str | Path,
    phase34_artifact: str | Path,
    *,
    scenes: Sequence[ThreeDMatchEvaluationSpec] = PHASE35_VALIDATION_SCENES,
) -> SceneRelativeRotationValidationResult:
    """Run the frozen six-scene panel without revising the Phase-34 rule."""

    selected_scenes = tuple(scenes)
    phase34_path, phase34 = _verified_phase34_artifact(
        phase34_artifact,
        selected_scenes,
    )
    root = Path(data_root)
    archive_paths = tuple(root / scene.archive_name for scene in selected_scenes)
    verifications = tuple(
        verify_evaluation_archive(path, scene)
        for path, scene in zip(archive_paths, selected_scenes, strict=True)
    )

    raw_predictions_by_scene = tuple(
        parse_registration_log(
            _read_archive_member(path, scene, "3dmatch.log")
        )
        for path, scene in zip(archive_paths, selected_scenes, strict=True)
    )
    eligible_predictions_by_scene = tuple(
        tuple(
            prediction
            for prediction in predictions
            if prediction.target_index - prediction.source_index > 1
        )
        for predictions in raw_predictions_by_scene
    )

    # All six complete decision sets exist before any gt.log or gt.info is read.
    blind_by_scene = tuple(
        _blind_validation_scene(scene, predictions)
        for scene, predictions in zip(
            selected_scenes,
            eligible_predictions_by_scene,
            strict=True,
        )
    )

    labeled_members_by_scene = tuple(
        _read_labeled_members(path, scene)
        for path, scene in zip(archive_paths, selected_scenes, strict=True)
    )
    labeled_and_counts = tuple(
        _label_validation_scene(blind, predictions, ground_truth, information)
        for blind, predictions, (ground_truth, information) in zip(
            blind_by_scene,
            eligible_predictions_by_scene,
            labeled_members_by_scene,
            strict=True,
        )
    )
    summaries = tuple(
        _scene_summary(
            scene.scene_name,
            raw_prediction_count=len(raw_predictions),
            observations=labeled,
            ground_truth_overlap_pair_count=ground_truth_count,
        )
        for scene, raw_predictions, (labeled, ground_truth_count) in zip(
            selected_scenes,
            raw_predictions_by_scene,
            labeled_and_counts,
            strict=True,
        )
    )
    validation_supported = all(
        summary.scene_validation_gate_passed for summary in summaries
    )
    return SceneRelativeRotationValidationResult(
        artifact_schema=(
            "pftf_alpha_scene_relative_rotation_validation_phase35/v1"
        ),
        role="frozen_six_scene_held_out_validation_panel",
        dataset_name="3DMatch geometric registration benchmark",
        dataset_source_url=OFFICIAL_BENCHMARK_URL,
        feature_name="scene_relative_prediction_rotation_midrank_percentile",
        information_boundary=(
            "exact_archive_hashes_and_member_metadata_verified_then_only_all_"
            "six_external_3dmatch_prediction_logs_read_to_materialize_every_"
            "rotation_midrank_decision_before_any_gt_log_or_gt_info_member_is_"
            "decompressed_or_decoded; no_fragment_coordinates_accessed"
        ),
        label_blind_execution_order=(
            "verify_phase34_identity_then_verify_all_archive_identities_then_"
            "read_all_prediction_logs_then_materialize_all_six_blind_decision_"
            "sets_then_read_and_join_all_official_labels"
        ),
        frozen_validation_contract=(
            "no_change_to_feature_rotation_cutoff_midrank_tie_handling_or_"
            "scene_gates_after_validation_access; every_scene_must_pass"
        ),
        phase34_artifact_path=str(phase34_path),
        phase34_artifact_sha256=_hash_file(phase34_path, "sha256"),
        validation_scene_specs=selected_scenes,
        evaluation_archives=verifications,
        rotation_percentile_cutoff=ROTATION_PERCENTILE_CUTOFF,
        minimum_correct_retention=MINIMUM_CORRECT_RETENTION,
        minimum_incorrect_rejection=MINIMUM_INCORRECT_REJECTION,
        official_error_threshold_squared=OFFICIAL_ERROR_THRESHOLD_SQUARED,
        observations=tuple(
            observation
            for labeled, _ in labeled_and_counts
            for observation in labeled
        ),
        scene_summaries=summaries,
        phase34_design_supported=bool(phase34["phase34_design_supported"]),
        held_out_validation_artifacts_accessed=True,
        phase35_validation_supported=validation_supported,
        held_out_validation_supported=validation_supported,
        cross_scene_real_registration_supported=bool(
            phase34["phase34_design_supported"] and validation_supported
        ),
        real_registration_labels_supported=True,
        real_correspondence_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("benchmark-data/3dmatch_phase35_evaluation"),
    )
    parser.add_argument(
        "--phase34-artifact",
        type=Path,
        default=Path("benchmark-out/scene_relative_rotation_guard_phase34.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark-out/scene_relative_rotation_validation_phase35.json"
        ),
    )
    parser.add_argument("--fetch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.fetch:
        fetch_phase35_evaluation_archives(args.data_root)
    result = evaluate_scene_relative_rotation_validation(
        args.data_root,
        args.phase34_artifact,
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
