"""Phase-37 fixed-parameter audit of independently generated registrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .open3d_fgr_pipeline import (
    OPEN3D_VERSION,
    PHASE37_SCENE_INPUTS,
    Phase37SceneInput,
    nonconsecutive_fragment_pairs,
    phase37_parameters,
)
from .scene_relative_rotation_guard import (
    MINIMUM_CORRECT_RETENTION,
    MINIMUM_INCORRECT_REJECTION,
    ROTATION_PERCENTILE_CUTOFF,
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
    verify_redkitchen_archive,
)
from .threedmatch_registration_guard import OFFICIAL_ERROR_THRESHOLD_SQUARED
from .threedmatch_scene import (
    MARYLAND_HOTEL3_SPEC,
    verify_threedmatch_scene_archive,
)

EXPECTED_PHASE36_SHA256 = (
    "9157e15adccdf8dea98e14f96124f826389d3c35bc3ddf04bbcc51e0a00ec24d"
)


@dataclass(frozen=True)
class PipelineAuditObservation:
    blind: BlindRotationObservation
    prediction_fitness: float
    prediction_inlier_rmse: float
    prediction_correspondence_count: int
    ground_truth_overlap_pair: bool
    official_transformation_error: float | None
    official_correct: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **self.blind.to_dict(),
            "prediction_fitness": self.prediction_fitness,
            "prediction_inlier_rmse": self.prediction_inlier_rmse,
            "prediction_correspondence_count": (self.prediction_correspondence_count),
            "ground_truth_overlap_pair": self.ground_truth_overlap_pair,
            "official_transformation_error": self.official_transformation_error,
            "official_correct": self.official_correct,
        }


@dataclass(frozen=True)
class IndependentPipelineSceneSummary:
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
    precision_gain_percentage_points: float
    correct_retention: float
    incorrect_rejection: float
    has_correct_and_incorrect_predictions: bool
    precision_improved: bool
    correct_retention_gate_passed: bool
    incorrect_rejection_gate_passed: bool
    scene_transfer_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IndependentPipelineRotationAuditResult:
    artifact_schema: str
    role: str
    pipeline_name: str
    evidence_boundary: str
    label_execution_order: str
    parameter_selection_history: str
    phase36_artifact_path: str
    phase36_artifact_sha256: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    prediction_artifact_schema: str
    generation_correction_history: str
    evaluation_archives: tuple[ThreeDMatchArchiveVerification, ...]
    rotation_percentile_cutoff: float
    minimum_correct_retention: float
    minimum_incorrect_rejection: float
    official_error_threshold_squared: float
    observations: tuple[PipelineAuditObservation, ...]
    scene_summaries: tuple[IndependentPipelineSceneSummary, ...]
    external_method_generation_reproduced: bool
    independently_generated_prediction_artifact_supported: bool
    phase37_fixed_parameter_audit_supported: bool
    independent_end_to_end_pipeline_transfer_supported: bool
    fresh_label_blind_validation_supported: bool
    independent_algorithm_implementation_supported: bool
    real_registration_labels_supported: bool
    real_correspondence_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "pipeline_name": self.pipeline_name,
            "evidence_boundary": self.evidence_boundary,
            "label_execution_order": self.label_execution_order,
            "parameter_selection_history": self.parameter_selection_history,
            "phase36_artifact_path": self.phase36_artifact_path,
            "phase36_artifact_sha256": self.phase36_artifact_sha256,
            "prediction_artifact_path": self.prediction_artifact_path,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_artifact_schema": self.prediction_artifact_schema,
            "generation_correction_history": self.generation_correction_history,
            "evaluation_archives": [
                archive.to_dict() for archive in self.evaluation_archives
            ],
            "rotation_percentile_cutoff": self.rotation_percentile_cutoff,
            "minimum_correct_retention": self.minimum_correct_retention,
            "minimum_incorrect_rejection": self.minimum_incorrect_rejection,
            "official_error_threshold_squared": (self.official_error_threshold_squared),
            "observations": [row.to_dict() for row in self.observations],
            "scene_summaries": [summary.to_dict() for summary in self.scene_summaries],
            "external_method_generation_reproduced": (
                self.external_method_generation_reproduced
            ),
            "independently_generated_prediction_artifact_supported": (
                self.independently_generated_prediction_artifact_supported
            ),
            "phase37_fixed_parameter_audit_supported": (
                self.phase37_fixed_parameter_audit_supported
            ),
            "independent_end_to_end_pipeline_transfer_supported": (
                self.independent_end_to_end_pipeline_transfer_supported
            ),
            "fresh_label_blind_validation_supported": (
                self.fresh_label_blind_validation_supported
            ),
            "independent_algorithm_implementation_supported": (
                self.independent_algorithm_implementation_supported
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def _verify_phase36(path: Path) -> Mapping[str, object]:
    if _sha256(path) != EXPECTED_PHASE36_SHA256:
        raise ValueError("Phase-36 artifact SHA-256 mismatch")
    payload = _load_json_object(path)
    expected = {
        "artifact_schema": (
            "pftf_alpha_independent_method_rotation_transfer_phase36/v1"
        ),
        "phase36_panel_supported": True,
        "independent_method_transfer_supported": True,
        "independent_end_to_end_pipeline_transfer_supported": False,
        "external_method_generation_reproduced": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase-36 predecessor mismatch: {key}")
    return payload


def _prediction_entry(
    raw: Mapping[str, object],
    scene: Phase37SceneInput,
) -> RegistrationLogEntry:
    expected_keys = {
        "source_index",
        "target_index",
        "fragment_count",
        "pair_random_seed",
        "benchmark_target_to_source_matrix",
        "fitness",
        "inlier_rmse",
        "correspondence_count",
    }
    if raw.keys() != expected_keys:
        raise ValueError("prediction row schema mismatch")
    return RegistrationLogEntry(
        source_index=int(raw["source_index"]),
        target_index=int(raw["target_index"]),
        fragment_count=int(raw["fragment_count"]),
        source_to_target_matrix=raw["benchmark_target_to_source_matrix"],
    )


def _verify_prediction_artifact(
    path: Path,
) -> tuple[Mapping[str, object], tuple[tuple[RegistrationLogEntry, ...], ...]]:
    payload = _load_json_object(path)
    expected_scalars = {
        "artifact_schema": "pftf_alpha_open3d_fgr_predictions_phase37/v2",
        "pipeline_name": "open3d_0.19.0_fpfh_fast_global_registration",
        "open3d_version": OPEN3D_VERSION,
        "parameters": phase37_parameters(),
        "external_method_generation_reproduced": True,
        "ground_truth_artifacts_accessed_by_generator": False,
    }
    for key, value in expected_scalars.items():
        if payload.get(key) != value:
            raise ValueError(f"prediction artifact protocol mismatch: {key}")
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list) or len(raw_scenes) != len(PHASE37_SCENE_INPUTS):
        raise ValueError("prediction artifact scene panel mismatch")
    scene_entries = []
    for raw_scene, scene in zip(
        raw_scenes,
        PHASE37_SCENE_INPUTS,
        strict=True,
    ):
        if not isinstance(raw_scene, dict):
            raise ValueError("prediction scene must be an object")
        raw_identity = raw_scene.get("scene")
        if raw_identity != scene.to_dict():
            raise ValueError(f"prediction scene identity mismatch: {scene.scene_name}")
        if raw_scene.get("pair_universe") != (
            "all_source_lt_target_pairs_with_target_minus_source_gt_1"
        ):
            raise ValueError("prediction pair universe mismatch")
        expected_pairs = nonconsecutive_fragment_pairs(scene.fragment_count)
        if raw_scene.get("expected_pair_count") != len(expected_pairs):
            raise ValueError("prediction pair count contract mismatch")
        raw_predictions = raw_scene.get("predictions")
        if not isinstance(raw_predictions, list):
            raise ValueError("predictions must be a list")
        entries = tuple(
            _prediction_entry(raw, scene)
            for raw in raw_predictions
            if isinstance(raw, dict)
        )
        if len(entries) != len(raw_predictions):
            raise ValueError("prediction row must be an object")
        if tuple(entry.pair for entry in entries) != expected_pairs:
            raise ValueError("prediction pairs are incomplete or out of order")
        if any(entry.fragment_count != scene.fragment_count for entry in entries):
            raise ValueError("prediction fragment count mismatch")
        scene_entries.append(entries)
    return payload, tuple(scene_entries)


def _blind_scene(
    scene: Phase37SceneInput,
    predictions: Sequence[RegistrationLogEntry],
) -> tuple[BlindRotationObservation, ...]:
    angles = tuple(
        prediction_rotation_radians(entry.source_to_target_matrix)
        for entry in predictions
    )
    percentiles = empirical_midrank_percentiles(angles)
    return tuple(
        BlindRotationObservation(
            scene_name=scene.scene_name,
            source_index=entry.source_index,
            target_index=entry.target_index,
            prediction_rotation_radians=angle,
            scene_relative_rotation_percentile=float(percentile),
            guarded_accept=bool(percentile < ROTATION_PERCENTILE_CUTOFF),
        )
        for entry, angle, percentile in zip(
            predictions,
            angles,
            percentiles,
            strict=True,
        )
    )


def _verify_evaluation_archive(
    root: Path,
    scene: Phase37SceneInput,
) -> ThreeDMatchArchiveVerification:
    path = root / scene.evaluation_archive_name
    if scene == PHASE37_SCENE_INPUTS[0]:
        return verify_redkitchen_archive(path, role="evaluation")
    if scene == PHASE37_SCENE_INPUTS[1]:
        return verify_threedmatch_scene_archive(
            path,
            MARYLAND_HOTEL3_SPEC,
            role="evaluation",
        )
    raise ValueError("unsupported Phase-37 scene")


def _read_evaluation_member(
    root: Path,
    scene: Phase37SceneInput,
    member_name: str,
) -> str:
    path = root / scene.evaluation_archive_name
    member = f"{scene.evaluation_name}/{member_name}"
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(member)
    return payload.decode("ascii")


def _labels(
    root: Path,
    scene: Phase37SceneInput,
) -> tuple[tuple[RegistrationLogEntry, ...], tuple[RegistrationInfoEntry, ...]]:
    return (
        parse_registration_log(_read_evaluation_member(root, scene, "gt.log")),
        parse_registration_info(_read_evaluation_member(root, scene, "gt.info")),
    )


def _label_scene(
    raw_scene: Mapping[str, object],
    blind: Sequence[BlindRotationObservation],
    predictions: Sequence[RegistrationLogEntry],
    ground_truth: Sequence[RegistrationLogEntry],
    information: Sequence[RegistrationInfoEntry],
) -> tuple[tuple[PipelineAuditObservation, ...], int]:
    gt = {
        entry.pair: entry
        for entry in ground_truth
        if entry.target_index - entry.source_index > 1
    }
    info = {
        entry.pair: entry
        for entry in information
        if entry.target_index - entry.source_index > 1
    }
    if gt.keys() != info.keys():
        raise ValueError("ground-truth log and information pairs disagree")
    raw_predictions = raw_scene.get("predictions")
    if not isinstance(raw_predictions, list) or not (
        len(raw_predictions) == len(blind) == len(predictions)
    ):
        raise ValueError("prediction representations do not align")
    labeled = []
    for raw, observation, prediction in zip(
        raw_predictions,
        blind,
        predictions,
        strict=True,
    ):
        if not isinstance(raw, dict):
            raise ValueError("prediction row must be an object")
        pair = prediction.pair
        if pair in gt:
            error = official_transformation_error(gt[pair], prediction, info[pair])
            correct = bool(error <= OFFICIAL_ERROR_THRESHOLD_SQUARED)
        else:
            error = None
            correct = False
        fitness = float(raw["fitness"])
        inlier_rmse = float(raw["inlier_rmse"])
        correspondence_count = int(raw["correspondence_count"])
        if (
            not math.isfinite(fitness)
            or not math.isfinite(inlier_rmse)
            or fitness < 0.0
            or inlier_rmse < 0.0
            or correspondence_count < 0
        ):
            raise ValueError("prediction diagnostics must be finite and nonnegative")
        labeled.append(
            PipelineAuditObservation(
                blind=observation,
                prediction_fitness=fitness,
                prediction_inlier_rmse=inlier_rmse,
                prediction_correspondence_count=correspondence_count,
                ground_truth_overlap_pair=pair in gt,
                official_transformation_error=error,
                official_correct=correct,
            )
        )
    return tuple(labeled), len(gt)


def _summary(
    scene_name: str,
    observations: Sequence[PipelineAuditObservation],
    ground_truth_count: int,
) -> IndependentPipelineSceneSummary:
    rows = tuple(observations)
    if not rows or ground_truth_count <= 0:
        raise ValueError("scene requires predictions and ground-truth overlaps")
    base_correct = sum(row.official_correct for row in rows)
    base_incorrect = len(rows) - base_correct
    guarded = tuple(row for row in rows if row.blind.guarded_accept)
    guarded_correct = sum(row.official_correct for row in guarded)
    guarded_incorrect = len(guarded) - guarded_correct
    base_precision = base_correct / len(rows)
    guarded_precision = guarded_correct / len(guarded) if guarded else 0.0
    correct_retention = guarded_correct / base_correct if base_correct else 0.0
    incorrect_rejection = (
        1.0 - guarded_incorrect / base_incorrect if base_incorrect else 0.0
    )
    has_both = bool(base_correct > 0 and base_incorrect > 0)
    precision_improved = bool(has_both and guarded_precision > base_precision)
    correct_gate = bool(has_both and correct_retention >= MINIMUM_CORRECT_RETENTION)
    incorrect_gate = bool(
        has_both and incorrect_rejection >= MINIMUM_INCORRECT_REJECTION
    )
    passed = bool(precision_improved and correct_gate and incorrect_gate)
    return IndependentPipelineSceneSummary(
        scene_name=scene_name,
        prediction_count=len(rows),
        ground_truth_overlap_pair_count=ground_truth_count,
        base_correct_count=base_correct,
        base_incorrect_count=base_incorrect,
        base_precision=base_precision,
        base_recall=base_correct / ground_truth_count,
        guarded_accepted_count=len(guarded),
        guarded_correct_count=guarded_correct,
        guarded_incorrect_count=guarded_incorrect,
        guarded_precision=guarded_precision,
        guarded_recall=guarded_correct / ground_truth_count,
        precision_gain_percentage_points=(100.0 * (guarded_precision - base_precision)),
        correct_retention=correct_retention,
        incorrect_rejection=incorrect_rejection,
        has_correct_and_incorrect_predictions=has_both,
        precision_improved=precision_improved,
        correct_retention_gate_passed=correct_gate,
        incorrect_rejection_gate_passed=incorrect_gate,
        scene_transfer_gate_passed=passed,
    )


def evaluate_independent_pipeline_rotation_audit(
    prediction_artifact: str | Path,
    phase36_artifact: str | Path,
    redkitchen_root: str | Path,
    maryland_root: str | Path,
) -> IndependentPipelineRotationAuditResult:
    """Apply the unchanged Phase-34 p90 rule after predictions are materialized."""

    prediction_path = Path(prediction_artifact)
    phase36_path = Path(phase36_artifact)
    _verify_phase36(phase36_path)
    prediction_payload, predictions_by_scene = _verify_prediction_artifact(
        prediction_path
    )
    raw_scenes = prediction_payload["scenes"]
    assert isinstance(raw_scenes, list)

    # Every guard decision exists before either evaluation archive is decoded.
    blind_by_scene = tuple(
        _blind_scene(scene, predictions)
        for scene, predictions in zip(
            PHASE37_SCENE_INPUTS,
            predictions_by_scene,
            strict=True,
        )
    )
    roots = (Path(redkitchen_root), Path(maryland_root))
    verifications = tuple(
        _verify_evaluation_archive(root, scene)
        for root, scene in zip(roots, PHASE37_SCENE_INPUTS, strict=True)
    )
    labels_by_scene = tuple(
        _labels(root, scene)
        for root, scene in zip(roots, PHASE37_SCENE_INPUTS, strict=True)
    )
    labeled_and_counts = tuple(
        _label_scene(raw_scene, blind, predictions, ground_truth, information)
        for raw_scene, blind, predictions, (ground_truth, information) in zip(
            raw_scenes,
            blind_by_scene,
            predictions_by_scene,
            labels_by_scene,
            strict=True,
        )
    )
    summaries = tuple(
        _summary(scene.scene_name, labeled, ground_truth_count)
        for scene, (labeled, ground_truth_count) in zip(
            PHASE37_SCENE_INPUTS,
            labeled_and_counts,
            strict=True,
        )
    )
    supported = all(summary.scene_transfer_gate_passed for summary in summaries)
    observations = tuple(row for labeled, _ in labeled_and_counts for row in labeled)
    return IndependentPipelineRotationAuditResult(
        artifact_schema="pftf_alpha_independent_pipeline_rotation_audit_phase37/v1",
        role="opened_label_fixed_parameter_end_to_end_pipeline_transfer_audit",
        pipeline_name="open3d_0.19.0_fpfh_fast_global_registration",
        evidence_boundary=(
            "independently executed Open3D FPFH+FGR predictions on official "
            "3DMatch fragments; both scene labels were opened in earlier "
            "phases, so this is not fresh held-out validation"
        ),
        label_execution_order=(
            "generate and save both complete all-pair prediction sets without "
            "evaluation inputs; materialize both p90 decision sets; then "
            "verify and decode gt.log and gt.info"
        ),
        parameter_selection_history=(
            "Open3D 0.19.0 official tutorial parameters and the unchanged "
            "Phase-34 p90 guard were fixed before Phase-37 result generation; "
            "no Phase-37 label-based tuning"
        ),
        phase36_artifact_path=str(phase36_path),
        phase36_artifact_sha256=_sha256(phase36_path),
        prediction_artifact_path=str(prediction_path),
        prediction_artifact_sha256=_sha256(prediction_path),
        prediction_artifact_schema=str(prediction_payload["artifact_schema"]),
        generation_correction_history=str(
            prediction_payload["generation_correction_history"]
        ),
        evaluation_archives=verifications,
        rotation_percentile_cutoff=ROTATION_PERCENTILE_CUTOFF,
        minimum_correct_retention=MINIMUM_CORRECT_RETENTION,
        minimum_incorrect_rejection=MINIMUM_INCORRECT_REJECTION,
        official_error_threshold_squared=OFFICIAL_ERROR_THRESHOLD_SQUARED,
        observations=observations,
        scene_summaries=summaries,
        external_method_generation_reproduced=True,
        independently_generated_prediction_artifact_supported=True,
        phase37_fixed_parameter_audit_supported=supported,
        independent_end_to_end_pipeline_transfer_supported=supported,
        fresh_label_blind_validation_supported=False,
        independent_algorithm_implementation_supported=False,
        real_registration_labels_supported=True,
        real_correspondence_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-artifact",
        type=Path,
        default=Path("benchmark-out/open3d_fgr_predictions_phase37.json"),
    )
    parser.add_argument(
        "--phase36-artifact",
        type=Path,
        default=Path("benchmark-out/independent_method_rotation_transfer_phase36.json"),
    )
    parser.add_argument(
        "--redkitchen-root",
        type=Path,
        default=Path("benchmark-data/3dmatch_redkitchen"),
    )
    parser.add_argument(
        "--maryland-root",
        type=Path,
        default=Path("benchmark-data/3dmatch_maryland_hotel3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/independent_pipeline_rotation_audit_phase37.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_independent_pipeline_rotation_audit(
        args.prediction_artifact,
        args.phase36_artifact,
        args.redkitchen_root,
        args.maryland_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scene_summaries": [
                    summary.to_dict() for summary in result.scene_summaries
                ],
                "phase37_fixed_parameter_audit_supported": (
                    result.phase37_fixed_parameter_audit_supported
                ),
                "independent_end_to_end_pipeline_transfer_supported": (
                    result.independent_end_to_end_pipeline_transfer_supported
                ),
                "fresh_label_blind_validation_supported": (
                    result.fresh_label_blind_validation_supported
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
