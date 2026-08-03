"""Phase-36 transfer of the frozen rotation guard to FPFH and Spin-Images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .scene_relative_rotation_guard import (
    MINIMUM_CORRECT_RETENTION,
    MINIMUM_INCORRECT_REJECTION,
    ROTATION_PERCENTILE_CUTOFF,
    empirical_midrank_percentiles,
    prediction_rotation_radians,
)
from .threedmatch_redkitchen import (
    RegistrationInfoEntry,
    RegistrationLogEntry,
    official_transformation_error,
    parse_registration_info,
    parse_registration_log,
)
from .threedmatch_registration_guard import OFFICIAL_ERROR_THRESHOLD_SQUARED

EXPECTED_PHASE35_SHA256 = (
    "c07cb04e82ef597f5c7480fad1181fd3d8141e2d7fbc2ab8d0a3c4e644179372"
)
SOURCE_REPOSITORY = "https://github.com/andyzeng/3dmatch-toolbox"
SOURCE_REPOSITORY_COMMIT = "4c6b2f613adb8bdcc9a62cb04134b7e1379b1a36"
METHODS = ("fpfh", "spin")
SOURCE_LICENSE_BOUNDARY = (
    "the toolbox source code is Simplified BSD; no separate license was "
    "identified for the committed benchmark log files, so do not redistribute "
    "the logs"
)


@dataclass(frozen=True)
class SourceFileSpec:
    file_name: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.file_name.strip() or self.byte_count <= 0:
            raise ValueError("source file identity must be non-empty and positive")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("source file SHA-256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticMethodSceneSpec:
    scene_name: str
    evaluation_name: str
    fpfh_log: SourceFileSpec
    spin_log: SourceFileSpec
    ground_truth_log: SourceFileSpec
    ground_truth_info: SourceFileSpec

    def __post_init__(self) -> None:
        if not self.scene_name.strip() or not self.evaluation_name.strip():
            raise ValueError("scene names must be non-empty")
        expected_names = {
            self.fpfh_log.file_name: "fpfh.log",
            self.spin_log.file_name: "spin.log",
            self.ground_truth_log.file_name: "gt.log",
            self.ground_truth_info.file_name: "gt.info",
        }
        if any(observed != expected for observed, expected in expected_names.items()):
            raise ValueError("scene source file names do not match their roles")

    def file_spec(self, role: str) -> SourceFileSpec:
        choices = {
            "fpfh": self.fpfh_log,
            "spin": self.spin_log,
            "ground_truth_log": self.ground_truth_log,
            "ground_truth_info": self.ground_truth_info,
        }
        try:
            return choices[role]
        except KeyError as error:
            raise ValueError(f"unknown source file role: {role}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_name": self.scene_name,
            "evaluation_name": self.evaluation_name,
            "fpfh_log": self.fpfh_log.to_dict(),
            "spin_log": self.spin_log.to_dict(),
            "ground_truth_log": self.ground_truth_log.to_dict(),
            "ground_truth_info": self.ground_truth_info.to_dict(),
        }


def _file(file_name: str, byte_count: int, sha256: str) -> SourceFileSpec:
    return SourceFileSpec(
        file_name=file_name,
        byte_count=byte_count,
        sha256=sha256,
    )


PHASE36_SCENES = (
    SyntheticMethodSceneSpec(
        scene_name="iclnuim-livingroom1",
        evaluation_name="iclnuim-livingroom1-evaluation",
        fpfh_log=_file(
            "fpfh.log",
            141_925,
            "3ccc001ea46a4e37ab52e8df9bcd1521523b7c23b5d412094ff185197ee174ea",
        ),
        spin_log=_file(
            "spin.log",
            125_398,
            "3beb87f061fcd1c82b821943f1652a6a06ccab23b8785198258ed9f7bd4e5d42",
        ),
        ground_truth_log=_file(
            "gt.log",
            59_507,
            "5bee3ead05841d418b6e5533efc269c34ecc7c22e2e33e82de60c288e4907b3f",
        ),
        ground_truth_info=_file(
            "gt.info",
            168_730,
            "830154126ba81e58e99092b5084669adcd1fe811b45a84346e89dbebbfffcd28",
        ),
    ),
    SyntheticMethodSceneSpec(
        scene_name="iclnuim-livingroom2",
        evaluation_name="iclnuim-livingroom2-evaluation",
        fpfh_log=_file(
            "fpfh.log",
            82_106,
            "90e824fe015807038a7af1a232d57efc4a6d22bc3b3b1f28225f6c9bf99ca8ab",
        ),
        spin_log=_file(
            "spin.log",
            90_146,
            "bb8c9052389f39fe638f1560b6cca443a1549104ebc4cd2a74ffd8ba18742e75",
        ),
        ground_truth_log=_file(
            "gt.log",
            36_197,
            "111ea5894862bd14b91ac3fcda28e6cb6325f77e4fef79f7cb757071a919022c",
        ),
        ground_truth_info=_file(
            "gt.info",
            100_990,
            "2be9e746e262099a5cf0c4b13aac984ec3675f6cef0864e89106c229ee143d15",
        ),
    ),
    SyntheticMethodSceneSpec(
        scene_name="iclnuim-office1",
        evaluation_name="iclnuim-office1-evaluation",
        fpfh_log=_file(
            "fpfh.log",
            103_132,
            "a05ccbf35ef28caa6eae81f3821360616c1b54883c095cc3d9bbf241abea0d33",
        ),
        spin_log=_file(
            "spin.log",
            96_949,
            "8e16c5de61a381c0dacb13a44c337e9609c67e0cab88ebe78cd3f5b9ee29dd7f",
        ),
        ground_truth_log=_file(
            "gt.log",
            42_164,
            "6404bcc429483bc499f9961f0536a50c9d1a70726bc75e756e476bfed5051626",
        ),
        ground_truth_info=_file(
            "gt.info",
            118_816,
            "ffd354c2ece1e094ce797ad520dd82d9328dd24f1cd96519bc835038e764c39e",
        ),
    ),
    SyntheticMethodSceneSpec(
        scene_name="iclnuim-office2",
        evaluation_name="iclnuim-office2-evaluation",
        fpfh_log=_file(
            "fpfh.log",
            115_400,
            "b695d47f657dc61b277ef88c2cd7d4f126136f0295f37061e9c357971d111631",
        ),
        spin_log=_file(
            "spin.log",
            116_169,
            "3e15a45d664554096a358d181a5405ebc49c923e91bbadfd9ffa5a98d000dfdb",
        ),
        ground_truth_log=_file(
            "gt.log",
            32_988,
            "f586d4d93ac9f05edd7a58a1ba90a4ef7e5bd8017657637bc77a25ac1e49a509",
        ),
        ground_truth_info=_file(
            "gt.info",
            92_998,
            "1ec3f8fadf73e96bee9b6a69a11b8bf366ced268a2a8c48204468b5f6a242c2c",
        ),
    ),
)


@dataclass(frozen=True)
class SourceFileVerification:
    scene_name: str
    role: str
    file_path: str
    byte_count: int
    sha256: str
    verified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BlindMethodRotationObservation:
    scene_name: str
    method_name: str
    source_index: int
    target_index: int
    prediction_rotation_radians: float
    block_relative_rotation_percentile: float
    guarded_accept: bool

    @property
    def pair(self) -> tuple[int, int]:
        return (self.source_index, self.target_index)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LabeledMethodRotationObservation:
    blind: BlindMethodRotationObservation
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
class MethodSceneRotationSummary:
    scene_name: str
    method_name: str
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
    block_transfer_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IndependentMethodRotationTransferResult:
    artifact_schema: str
    role: str
    benchmark_name: str
    source_repository: str
    source_repository_commit: str
    source_license_boundary: str
    phase35_artifact_path: str
    phase35_artifact_sha256: str
    feature_name: str
    method_independence_boundary: str
    information_boundary: str
    label_blind_execution_order: str
    frozen_transfer_contract: str
    methods: tuple[str, ...]
    scene_specs: tuple[SyntheticMethodSceneSpec, ...]
    source_file_verifications: tuple[SourceFileVerification, ...]
    rotation_percentile_cutoff: float
    minimum_correct_retention: float
    minimum_incorrect_rejection: float
    official_error_threshold_squared: float
    observations: tuple[LabeledMethodRotationObservation, ...]
    block_summaries: tuple[MethodSceneRotationSummary, ...]
    phase35_predecessor_supported: bool
    phase36_panel_supported: bool
    independent_method_transfer_supported: bool
    independent_end_to_end_pipeline_transfer_supported: bool
    cross_benchmark_transfer_supported: bool
    prediction_log_provenance_verified: bool
    external_method_generation_reproduced: bool
    synthetic_registration_labels_supported: bool
    real_correspondence_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "benchmark_name": self.benchmark_name,
            "source_repository": self.source_repository,
            "source_repository_commit": self.source_repository_commit,
            "source_license_boundary": self.source_license_boundary,
            "phase35_artifact_path": self.phase35_artifact_path,
            "phase35_artifact_sha256": self.phase35_artifact_sha256,
            "feature_name": self.feature_name,
            "method_independence_boundary": self.method_independence_boundary,
            "information_boundary": self.information_boundary,
            "label_blind_execution_order": self.label_blind_execution_order,
            "frozen_transfer_contract": self.frozen_transfer_contract,
            "methods": list(self.methods),
            "scene_specs": [scene.to_dict() for scene in self.scene_specs],
            "source_file_verifications": [
                verification.to_dict()
                for verification in self.source_file_verifications
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
            "block_summaries": [
                summary.to_dict() for summary in self.block_summaries
            ],
            "phase35_predecessor_supported": self.phase35_predecessor_supported,
            "phase36_panel_supported": self.phase36_panel_supported,
            "independent_method_transfer_supported": (
                self.independent_method_transfer_supported
            ),
            "independent_end_to_end_pipeline_transfer_supported": (
                self.independent_end_to_end_pipeline_transfer_supported
            ),
            "cross_benchmark_transfer_supported": (
                self.cross_benchmark_transfer_supported
            ),
            "prediction_log_provenance_verified": (
                self.prediction_log_provenance_verified
            ),
            "external_method_generation_reproduced": (
                self.external_method_generation_reproduced
            ),
            "synthetic_registration_labels_supported": (
                self.synthetic_registration_labels_supported
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


def _verified_phase35_artifact(
    path: str | Path,
) -> tuple[Path, Mapping[str, object]]:
    resolved = Path(path)
    if _sha256(resolved) != EXPECTED_PHASE35_SHA256:
        raise ValueError(f"Phase-35 artifact SHA-256 mismatch: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase-35 artifact must contain a JSON object")
    expected = {
        "artifact_schema": (
            "pftf_alpha_scene_relative_rotation_validation_phase35/v1"
        ),
        "feature_name": "scene_relative_prediction_rotation_midrank_percentile",
        "rotation_percentile_cutoff": ROTATION_PERCENTILE_CUTOFF,
        "minimum_correct_retention": MINIMUM_CORRECT_RETENTION,
        "minimum_incorrect_rejection": MINIMUM_INCORRECT_REJECTION,
        "phase35_validation_supported": True,
        "held_out_validation_supported": True,
        "cross_scene_real_registration_supported": True,
        "real_registration_labels_supported": True,
        "real_correspondence_supported": False,
        "real_trimmed_reconstruction_supported": False,
        "deployment_supported": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase-35 frozen field mismatch: {key}")
    return resolved, payload


def _verify_source_file(
    root: Path,
    scene: SyntheticMethodSceneSpec,
    role: str,
) -> SourceFileVerification:
    spec = scene.file_spec(role)
    path = root / scene.evaluation_name / spec.file_name
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != spec.byte_count:
        raise ValueError(f"{scene.scene_name} {role} byte count mismatch")
    observed_sha256 = _sha256(path)
    if observed_sha256 != spec.sha256:
        raise ValueError(f"{scene.scene_name} {role} SHA-256 mismatch")
    return SourceFileVerification(
        scene_name=scene.scene_name,
        role=role,
        file_path=str(path),
        byte_count=spec.byte_count,
        sha256=observed_sha256,
        verified=True,
    )


def _read_ascii(path: Path) -> str:
    return path.read_text(encoding="ascii")


def _prediction_path(
    root: Path,
    scene: SyntheticMethodSceneSpec,
    method: str,
) -> Path:
    return root / scene.evaluation_name / scene.file_spec(method).file_name


def _blind_block(
    scene_name: str,
    method_name: str,
    predictions: Sequence[RegistrationLogEntry],
) -> tuple[BlindMethodRotationObservation, ...]:
    angles = tuple(
        prediction_rotation_radians(prediction.source_to_target_matrix)
        for prediction in predictions
    )
    percentiles = empirical_midrank_percentiles(angles)
    return tuple(
        BlindMethodRotationObservation(
            scene_name=scene_name,
            method_name=method_name,
            source_index=prediction.source_index,
            target_index=prediction.target_index,
            prediction_rotation_radians=angle,
            block_relative_rotation_percentile=float(percentile),
            guarded_accept=bool(percentile < ROTATION_PERCENTILE_CUTOFF),
        )
        for prediction, angle, percentile in zip(
            predictions,
            angles,
            percentiles,
            strict=True,
        )
    )


def _read_scene_labels(
    root: Path,
    scene: SyntheticMethodSceneSpec,
) -> tuple[tuple[RegistrationLogEntry, ...], tuple[RegistrationInfoEntry, ...]]:
    scene_root = root / scene.evaluation_name
    ground_truth = parse_registration_log(
        _read_ascii(scene_root / scene.ground_truth_log.file_name)
    )
    information = parse_registration_info(
        _read_ascii(scene_root / scene.ground_truth_info.file_name)
    )
    return ground_truth, information


def _label_block(
    blind: Sequence[BlindMethodRotationObservation],
    predictions: Sequence[RegistrationLogEntry],
    ground_truth: Sequence[RegistrationLogEntry],
    information: Sequence[RegistrationInfoEntry],
) -> tuple[tuple[LabeledMethodRotationObservation, ...], int]:
    ground_truth_map = {
        entry.pair: entry
        for entry in ground_truth
        if entry.target_index - entry.source_index > 1
    }
    information_map = {
        entry.pair: entry
        for entry in information
        if entry.target_index - entry.source_index > 1
    }
    if ground_truth_map.keys() != information_map.keys():
        raise ValueError("eligible ground-truth log and info pairs disagree")
    if len(blind) != len(predictions):
        raise ValueError("blind observations and predictions must align")
    labeled = []
    for observation, prediction in zip(blind, predictions, strict=True):
        if observation.pair != prediction.pair:
            raise ValueError("blind observations and prediction pairs disagree")
        pair = prediction.pair
        if pair in ground_truth_map:
            error = official_transformation_error(
                ground_truth_map[pair],
                prediction,
                information_map[pair],
            )
            correct = bool(error <= OFFICIAL_ERROR_THRESHOLD_SQUARED)
        else:
            error = None
            correct = False
        labeled.append(
            LabeledMethodRotationObservation(
                blind=observation,
                ground_truth_overlap_pair=pair in ground_truth_map,
                official_transformation_error=error,
                official_correct=correct,
            )
        )
    return tuple(labeled), len(ground_truth_map)


def _block_summary(
    scene_name: str,
    method_name: str,
    *,
    raw_prediction_count: int,
    observations: Sequence[LabeledMethodRotationObservation],
    ground_truth_overlap_pair_count: int,
) -> MethodSceneRotationSummary:
    rows = tuple(observations)
    base_correct = sum(row.official_correct for row in rows)
    base_incorrect = len(rows) - base_correct
    if not rows or base_correct == 0 or base_incorrect == 0:
        raise ValueError("transfer block requires correct and incorrect predictions")
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
    block_gate = bool(precision_improved and correct_gate and incorrect_gate)
    return MethodSceneRotationSummary(
        scene_name=scene_name,
        method_name=method_name,
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
        block_transfer_gate_passed=block_gate,
    )


def evaluate_independent_method_rotation_transfer(
    fragments_root: str | Path,
    phase35_artifact: str | Path,
    *,
    scenes: Sequence[SyntheticMethodSceneSpec] = PHASE36_SCENES,
) -> IndependentMethodRotationTransferResult:
    """Evaluate all eight frozen method-scene blocks before revising nothing."""

    selected_scenes = tuple(scenes)
    if tuple(scene.scene_name for scene in selected_scenes) != tuple(
        scene.scene_name for scene in PHASE36_SCENES
    ):
        raise ValueError("Phase-36 scene panel differs from the frozen panel")
    phase35_path, phase35 = _verified_phase35_artifact(phase35_artifact)
    root = Path(fragments_root)
    roles = (*METHODS, "ground_truth_log", "ground_truth_info")

    # Exact identities are checked before any source file is decoded.
    verifications = tuple(
        _verify_source_file(root, scene, role)
        for scene in selected_scenes
        for role in roles
    )

    blocks = tuple(
        (scene, method)
        for scene in selected_scenes
        for method in METHODS
    )
    raw_predictions_by_block = tuple(
        parse_registration_log(
            _read_ascii(_prediction_path(root, scene, method))
        )
        for scene, method in blocks
    )
    eligible_predictions_by_block = tuple(
        tuple(
            prediction
            for prediction in predictions
            if prediction.target_index - prediction.source_index > 1
        )
        for predictions in raw_predictions_by_block
    )

    # All eight decision sets exist before the first gt.log or gt.info decode.
    blind_by_block = tuple(
        _blind_block(scene.scene_name, method, predictions)
        for (scene, method), predictions in zip(
            blocks,
            eligible_predictions_by_block,
            strict=True,
        )
    )

    labels_by_scene = tuple(
        _read_scene_labels(root, scene) for scene in selected_scenes
    )
    labels_by_scene_name = {
        scene.scene_name: labels
        for scene, labels in zip(selected_scenes, labels_by_scene, strict=True)
    }
    labeled_and_counts = tuple(
        _label_block(
            blind,
            predictions,
            *labels_by_scene_name[scene.scene_name],
        )
        for (scene, _), blind, predictions in zip(
            blocks,
            blind_by_block,
            eligible_predictions_by_block,
            strict=True,
        )
    )
    summaries = tuple(
        _block_summary(
            scene.scene_name,
            method,
            raw_prediction_count=len(raw_predictions),
            observations=labeled,
            ground_truth_overlap_pair_count=ground_truth_count,
        )
        for (scene, method), raw_predictions, (labeled, ground_truth_count) in zip(
            blocks,
            raw_predictions_by_block,
            labeled_and_counts,
            strict=True,
        )
    )
    panel_supported = all(summary.block_transfer_gate_passed for summary in summaries)
    return IndependentMethodRotationTransferResult(
        artifact_schema="pftf_alpha_independent_method_rotation_transfer_phase36/v1",
        role="frozen_cross_method_cross_benchmark_transfer_panel",
        benchmark_name="ICL-NUIM synthetic geometric registration benchmark",
        source_repository=SOURCE_REPOSITORY,
        source_repository_commit=SOURCE_REPOSITORY_COMMIT,
        source_license_boundary=SOURCE_LICENSE_BOUNDARY,
        phase35_artifact_path=str(phase35_path),
        phase35_artifact_sha256=_sha256(phase35_path),
        feature_name="scene_relative_prediction_rotation_midrank_percentile",
        method_independence_boundary=(
            "fpfh_and_spin_images_are_distinct_descriptor_methods_but_the_"
            "committed_predictions_share_the_3dmatch_toolbox_ransac_"
            "registration_and_log_generation_pipeline"
        ),
        information_boundary=(
            "all_committed_source_file_sizes_and_hashes_verified_then_all_"
            "eight_fpfh_and_spin_prediction_logs_decoded_and_all_rotation_"
            "midrank_decisions_materialized_before_any_gt_log_or_gt_info_is_"
            "decoded"
        ),
        label_blind_execution_order=(
            "verify_phase35_and_all_source_identities_then_decode_all_eight_"
            "prediction_logs_then_materialize_all_eight_blind_decision_sets_"
            "then_decode_and_join_the_four_scene_label_pairs"
        ),
        frozen_transfer_contract=(
            "four_scenes_times_two_methods; unchanged_feature_p90_midrank_ties_"
            "and_gates; every_method_scene_block_must_pass; no_post_label_"
            "revision"
        ),
        methods=METHODS,
        scene_specs=selected_scenes,
        source_file_verifications=verifications,
        rotation_percentile_cutoff=ROTATION_PERCENTILE_CUTOFF,
        minimum_correct_retention=MINIMUM_CORRECT_RETENTION,
        minimum_incorrect_rejection=MINIMUM_INCORRECT_REJECTION,
        official_error_threshold_squared=OFFICIAL_ERROR_THRESHOLD_SQUARED,
        observations=tuple(
            observation
            for labeled, _ in labeled_and_counts
            for observation in labeled
        ),
        block_summaries=summaries,
        phase35_predecessor_supported=bool(
            phase35["phase35_validation_supported"]
        ),
        phase36_panel_supported=panel_supported,
        independent_method_transfer_supported=panel_supported,
        independent_end_to_end_pipeline_transfer_supported=False,
        cross_benchmark_transfer_supported=panel_supported,
        prediction_log_provenance_verified=all(
            verification.verified for verification in verifications
        ),
        external_method_generation_reproduced=False,
        synthetic_registration_labels_supported=True,
        real_correspondence_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fragments-root",
        type=Path,
        default=Path(
            "benchmark-data/3dmatch-toolbox-phase36/data/fragments"
        ),
    )
    parser.add_argument(
        "--phase35-artifact",
        type=Path,
        default=Path(
            "benchmark-out/scene_relative_rotation_validation_phase35.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark-out/independent_method_rotation_transfer_phase36.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_independent_method_rotation_transfer(
        args.fragments_root,
        args.phase35_artifact,
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
