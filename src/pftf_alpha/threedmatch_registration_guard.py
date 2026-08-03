"""Phase-32 frozen guard benchmark on 3DMatch redkitchen registrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .local_spatial_displacement import (
    estimate_local_spatial_displacement_evidence,
)
from .local_spatial_residual_guard import EXPECTED_LOCAL_REJECTION_CUTOFF
from .matched_guard_signature import (
    MatchedGuardModel,
    matched_guard_signature_from_evidence,
    score_matched_guard_signature,
)
from .matched_pair_consistency import estimate_matched_pair_evidence
from .open3d_demo_icp import transform_points
from .open3d_real_pair_intake import load_phase28_predecessor_model
from .tail_sensitive_local_guard import (
    EXPECTED_TAIL_REJECTION_CUTOFF,
    tail_feature_value,
)
from .threedmatch_redkitchen import (
    EVALUATION_NAME,
    FRAGMENT_COUNT,
    SCENE_NAME,
    RegistrationInfoEntry,
    RegistrationLogEntry,
    ThreeDMatchArchiveVerification,
    official_transformation_error,
    prepare_redkitchen_data,
    read_binary_ply_xyz,
    read_registration_info,
    read_registration_log,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_DISTANCE_THRESHOLDS = (0.02, 0.05)
DEFAULT_MAXIMUM_POINTS_PER_FRAGMENT = 10_000
DEFAULT_PATCH_SIZE = 96
DEFAULT_PATCH_COUNT = 4
OFFICIAL_ERROR_THRESHOLD_SQUARED = 0.04
MINIMUM_CORRECT_RETENTION = 0.90
MINIMUM_INCORRECT_REJECTION_FRACTION = 0.10
MINIMUM_TAIL_PREDECESSOR_CORRECT_RETENTION = 0.90


@dataclass(frozen=True)
class PatchGuardObservation:
    patch_index: int
    pair_count: int
    presented_pair_map_sha256: str
    model_score: float
    global_passed: bool
    percentile95_local_residual: float
    local_passed: bool
    isolated_tail_ratio: float
    tail_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BlindThresholdObservation:
    maximum_correspondence_distance: float
    reciprocal_match_count: int
    source_coverage_fraction: float
    target_coverage_fraction: float
    median_correspondence_distance: float | None
    percentile95_correspondence_distance: float | None
    patch_supported: bool
    patches: tuple[PatchGuardObservation, ...]
    global_route_passed: bool
    global_local_route_passed: bool
    global_local_tail_route_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "patches": [patch.to_dict() for patch in self.patches],
        }


@dataclass(frozen=True)
class BlindRegistrationObservation:
    source_index: int
    target_index: int
    fragment_count: int
    prediction_matrix: FloatArray
    source_raw_point_count: int
    target_raw_point_count: int
    source_sampled_point_count: int
    target_sampled_point_count: int
    thresholds: tuple[BlindThresholdObservation, ...]

    @property
    def pair(self) -> tuple[int, int]:
        return (self.source_index, self.target_index)


@dataclass(frozen=True)
class LabeledRegistrationObservation:
    blind: BlindRegistrationObservation
    ground_truth_overlap_pair: bool
    official_transformation_error: float | None
    official_correct: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "source_index": self.blind.source_index,
            "target_index": self.blind.target_index,
            "fragment_count": self.blind.fragment_count,
            "prediction_matrix": self.blind.prediction_matrix.tolist(),
            "source_raw_point_count": self.blind.source_raw_point_count,
            "target_raw_point_count": self.blind.target_raw_point_count,
            "source_sampled_point_count": self.blind.source_sampled_point_count,
            "target_sampled_point_count": self.blind.target_sampled_point_count,
            "thresholds": [
                threshold.to_dict() for threshold in self.blind.thresholds
            ],
            "ground_truth_overlap_pair": self.ground_truth_overlap_pair,
            "official_transformation_error": (
                self.official_transformation_error
            ),
            "official_correct": self.official_correct,
        }


@dataclass(frozen=True)
class RegistrationRouteSummary:
    route_name: str
    accepted_prediction_count: int
    correct_accepted_count: int
    incorrect_accepted_count: int
    precision: float
    recall: float
    base_correct_retention: float
    base_incorrect_rejection_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdRegistrationSummary:
    maximum_correspondence_distance: float
    base: RegistrationRouteSummary
    global_only: RegistrationRouteSummary
    global_local: RegistrationRouteSummary
    global_local_tail: RegistrationRouteSummary
    tail_predecessor_correct_count: int
    tail_retained_predecessor_correct_count: int
    tail_predecessor_correct_retention: float
    tail_rejected_predecessor_incorrect_count: int
    precision_improved: bool
    correct_retention_gate_passed: bool
    incorrect_rejection_gate_passed: bool
    tail_incremental_gate_passed: bool
    threshold_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ThreeDMatchRegistrationGuardResult:
    artifact_schema: str
    role: str
    dataset_name: str
    dataset_license_boundary: str
    information_boundary: str
    label_blind_execution_order: str
    fragment_archive: ThreeDMatchArchiveVerification
    evaluation_archive: ThreeDMatchArchiveVerification
    phase28_artifact_path: str
    phase28_artifact_sha256: str
    prediction_log_name: str
    raw_prediction_count: int
    eligible_prediction_count: int
    ground_truth_overlap_pair_count: int
    official_error_threshold_squared: float
    maximum_points_per_fragment: int
    patch_size: int
    patch_count: int
    distance_thresholds: tuple[float, ...]
    global_signature_rejection_cutoff: float
    local_percentile95_rejection_cutoff: float
    isolated_tail_ratio_rejection_cutoff: float
    observations: tuple[LabeledRegistrationObservation, ...]
    threshold_summaries: tuple[ThresholdRegistrationSummary, ...]
    real_registration_labels_supported: bool
    phase32_supported: bool
    tail_sensitive_real_registration_supported: bool
    real_correspondence_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "dataset_name": self.dataset_name,
            "dataset_license_boundary": self.dataset_license_boundary,
            "information_boundary": self.information_boundary,
            "label_blind_execution_order": self.label_blind_execution_order,
            "fragment_archive": self.fragment_archive.to_dict(),
            "evaluation_archive": self.evaluation_archive.to_dict(),
            "phase28_artifact_path": self.phase28_artifact_path,
            "phase28_artifact_sha256": self.phase28_artifact_sha256,
            "prediction_log_name": self.prediction_log_name,
            "raw_prediction_count": self.raw_prediction_count,
            "eligible_prediction_count": self.eligible_prediction_count,
            "ground_truth_overlap_pair_count": (
                self.ground_truth_overlap_pair_count
            ),
            "official_error_threshold_squared": (
                self.official_error_threshold_squared
            ),
            "maximum_points_per_fragment": self.maximum_points_per_fragment,
            "patch_size": self.patch_size,
            "patch_count": self.patch_count,
            "distance_thresholds": list(self.distance_thresholds),
            "global_signature_rejection_cutoff": (
                self.global_signature_rejection_cutoff
            ),
            "local_percentile95_rejection_cutoff": (
                self.local_percentile95_rejection_cutoff
            ),
            "isolated_tail_ratio_rejection_cutoff": (
                self.isolated_tail_ratio_rejection_cutoff
            ),
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "threshold_summaries": [
                summary.to_dict() for summary in self.threshold_summaries
            ],
            "real_registration_labels_supported": (
                self.real_registration_labels_supported
            ),
            "phase32_supported": self.phase32_supported,
            "tail_sensitive_real_registration_supported": (
                self.tail_sensitive_real_registration_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_trimmed_reconstruction_supported": (
                self.real_trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


@dataclass(frozen=True)
class _SampledFragment:
    raw_point_count: int
    points: FloatArray
    original_indices: IntArray


@dataclass(frozen=True)
class SceneRegistrationGuardEvaluation:
    """Shared label-blind result before phase-specific claim packaging."""

    scene_name: str
    evaluation_name: str
    phase28_artifact_path: str
    phase28_artifact_sha256: str
    model: MatchedGuardModel
    raw_prediction_count: int
    eligible_prediction_count: int
    ground_truth_overlap_pair_count: int
    maximum_points_per_fragment: int
    patch_size: int
    patch_count: int
    distance_thresholds: tuple[float, ...]
    observations: tuple[LabeledRegistrationObservation, ...]
    threshold_summaries: tuple[ThresholdRegistrationSummary, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_fragment_sample(
    points: FloatArray,
    maximum_points: int,
) -> tuple[FloatArray, IntArray]:
    """Retain a deterministic, order-spanning subset without labels."""

    selected = np.asarray(points, dtype=np.float64)
    if selected.ndim != 2 or selected.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if maximum_points < 96:
        raise ValueError("maximum_points must be at least 96")
    if selected.shape[0] <= maximum_points:
        indices = np.arange(selected.shape[0], dtype=np.int64)
    else:
        indices = np.linspace(
            0,
            selected.shape[0] - 1,
            maximum_points,
            dtype=np.int64,
        )
    return np.ascontiguousarray(selected[indices]), indices


def _farthest_anchor_indices(points: FloatArray, count: int) -> IntArray:
    if count <= 0 or count > points.shape[0]:
        raise ValueError("anchor count must lie within the point count")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    anchors = [int(order[0])]
    selected = np.zeros(points.shape[0], dtype=np.bool_)
    selected[anchors[0]] = True
    minimum_squared = np.sum((points - points[anchors[0]]) ** 2, axis=1)
    for _ in range(1, count):
        candidates = np.where(selected, -np.inf, minimum_squared)
        next_index = int(np.argmax(candidates))
        anchors.append(next_index)
        selected[next_index] = True
        squared = np.sum((points - points[next_index]) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, squared)
    return np.asarray(anchors, dtype=np.int64)


def _pair_map_sha256(source_indices: IntArray, target_indices: IntArray) -> str:
    mapping = np.column_stack((source_indices, target_indices)).astype("<i8")
    return hashlib.sha256(mapping.tobytes()).hexdigest()


def _patches(
    aligned_source: FloatArray,
    target: FloatArray,
    source_indices: IntArray,
    target_indices: IntArray,
    source_original_indices: IntArray,
    target_original_indices: IntArray,
    *,
    patch_size: int,
    patch_count: int,
    model: MatchedGuardModel,
) -> tuple[PatchGuardObservation, ...]:
    matched_source = aligned_source[source_indices]
    matched_target = target[target_indices]
    anchors = _farthest_anchor_indices(matched_source, patch_count)
    tree = cKDTree(matched_source)
    patches = []
    for patch_index, anchor_index in enumerate(anchors):
        _, selected = tree.query(
            matched_source[anchor_index],
            k=patch_size,
            workers=1,
        )
        selected = np.sort(np.asarray(selected, dtype=np.int64).reshape(-1))
        primary = matched_source[selected]
        repeat = matched_target[selected]
        evidence = replace(
            estimate_matched_pair_evidence(primary, repeat),
            information_boundary=(
                "prediction_aligned_reciprocal_nearest_neighbor_candidates; "
                "registration_ground_truth_hidden"
            ),
        )
        signature = matched_guard_signature_from_evidence(
            evidence,
            retained_pair_count=patch_size,
            point_count=patch_size,
        )
        model_score = score_matched_guard_signature(model, signature)
        local = estimate_local_spatial_displacement_evidence(primary, repeat)
        tail_ratio = tail_feature_value(local, "isolated_tail_ratio")
        patches.append(
            PatchGuardObservation(
                patch_index=patch_index,
                pair_count=patch_size,
                presented_pair_map_sha256=_pair_map_sha256(
                    source_original_indices[source_indices[selected]],
                    target_original_indices[target_indices[selected]],
                ),
                model_score=model_score,
                global_passed=bool(model_score < model.rejection_cutoff),
                percentile95_local_residual=(
                    local.percentile95_local_residual
                ),
                local_passed=bool(
                    local.percentile95_local_residual
                    < EXPECTED_LOCAL_REJECTION_CUTOFF
                ),
                isolated_tail_ratio=tail_ratio,
                tail_passed=bool(
                    tail_ratio < EXPECTED_TAIL_REJECTION_CUTOFF
                ),
            )
        )
    return tuple(patches)


def _blind_observation(
    prediction: RegistrationLogEntry,
    source: _SampledFragment,
    target: _SampledFragment,
    target_tree: cKDTree,
    *,
    distance_thresholds: tuple[float, ...],
    patch_size: int,
    patch_count: int,
    model: MatchedGuardModel,
) -> BlindRegistrationObservation:
    aligned_source = transform_points(
        source.points,
        prediction.source_to_target_matrix,
    )
    forward_distances, target_indices = target_tree.query(
        aligned_source,
        k=1,
        workers=1,
    )
    source_tree = cKDTree(aligned_source)
    _, backward_indices = source_tree.query(target.points, k=1, workers=1)
    forward_distances = np.asarray(forward_distances, dtype=np.float64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    backward_indices = np.asarray(backward_indices, dtype=np.int64)
    all_source_indices = np.arange(aligned_source.shape[0], dtype=np.int64)
    reciprocal = backward_indices[target_indices] == all_source_indices
    threshold_observations = []
    for maximum_distance in distance_thresholds:
        selected = reciprocal & (forward_distances <= maximum_distance)
        source_indices = all_source_indices[selected]
        selected_targets = target_indices[selected]
        distances = forward_distances[selected]
        supported = bool(
            distances.size >= patch_size
            and distances.size >= patch_count
        )
        patches = (
            _patches(
                aligned_source,
                target.points,
                source_indices,
                selected_targets,
                source.original_indices,
                target.original_indices,
                patch_size=patch_size,
                patch_count=patch_count,
                model=model,
            )
            if supported
            else ()
        )
        global_pass = bool(supported and all(p.global_passed for p in patches))
        local_pass = bool(global_pass and all(p.local_passed for p in patches))
        tail_pass = bool(local_pass and all(p.tail_passed for p in patches))
        threshold_observations.append(
            BlindThresholdObservation(
                maximum_correspondence_distance=maximum_distance,
                reciprocal_match_count=int(distances.size),
                source_coverage_fraction=float(
                    distances.size / source.points.shape[0]
                ),
                target_coverage_fraction=float(
                    distances.size / target.points.shape[0]
                ),
                median_correspondence_distance=(
                    None if distances.size == 0 else float(np.median(distances))
                ),
                percentile95_correspondence_distance=(
                    None
                    if distances.size == 0
                    else float(np.percentile(distances, 95.0))
                ),
                patch_supported=supported,
                patches=patches,
                global_route_passed=global_pass,
                global_local_route_passed=local_pass,
                global_local_tail_route_passed=tail_pass,
            )
        )
    return BlindRegistrationObservation(
        source_index=prediction.source_index,
        target_index=prediction.target_index,
        fragment_count=prediction.fragment_count,
        prediction_matrix=prediction.source_to_target_matrix,
        source_raw_point_count=source.raw_point_count,
        target_raw_point_count=target.raw_point_count,
        source_sampled_point_count=source.points.shape[0],
        target_sampled_point_count=target.points.shape[0],
        thresholds=tuple(threshold_observations),
    )


def _label_observations(
    blind: Sequence[BlindRegistrationObservation],
    ground_truth: Sequence[RegistrationLogEntry],
    information: Sequence[RegistrationInfoEntry],
) -> tuple[LabeledRegistrationObservation, ...]:
    gt_map = {
        entry.pair: entry
        for entry in ground_truth
        if entry.target_index - entry.source_index > 1
    }
    info_map = {
        entry.pair: entry
        for entry in information
        if entry.target_index - entry.source_index > 1
    }
    if gt_map.keys() != info_map.keys():
        raise ValueError("eligible ground-truth log and info pairs disagree")
    labeled = []
    for observation in blind:
        pair = observation.pair
        if pair not in gt_map:
            error = None
            correct = False
        else:
            prediction = RegistrationLogEntry(
                source_index=observation.source_index,
                target_index=observation.target_index,
                fragment_count=observation.fragment_count,
                source_to_target_matrix=observation.prediction_matrix,
            )
            error = official_transformation_error(
                gt_map[pair],
                prediction,
                info_map[pair],
            )
            correct = bool(error <= OFFICIAL_ERROR_THRESHOLD_SQUARED)
        labeled.append(
            LabeledRegistrationObservation(
                blind=observation,
                ground_truth_overlap_pair=pair in gt_map,
                official_transformation_error=error,
                official_correct=correct,
            )
        )
    return tuple(labeled)


def _route_summary(
    observations: Sequence[LabeledRegistrationObservation],
    *,
    route_name: str,
    threshold_index: int,
    ground_truth_pair_count: int,
    base_correct_count: int,
    base_incorrect_count: int,
) -> RegistrationRouteSummary:
    def accepted(observation: LabeledRegistrationObservation) -> bool:
        threshold = observation.blind.thresholds[threshold_index]
        routes = {
            "base": True,
            "global_only": threshold.global_route_passed,
            "global_local": threshold.global_local_route_passed,
            "global_local_tail": threshold.global_local_tail_route_passed,
        }
        return routes[route_name]

    selected = tuple(item for item in observations if accepted(item))
    correct = sum(item.official_correct for item in selected)
    incorrect = len(selected) - correct
    precision = correct / len(selected) if selected else 0.0
    recall = correct / ground_truth_pair_count
    correct_retention = correct / base_correct_count
    incorrect_rejection = 1.0 - incorrect / base_incorrect_count
    return RegistrationRouteSummary(
        route_name=route_name,
        accepted_prediction_count=len(selected),
        correct_accepted_count=correct,
        incorrect_accepted_count=incorrect,
        precision=precision,
        recall=recall,
        base_correct_retention=correct_retention,
        base_incorrect_rejection_fraction=incorrect_rejection,
    )


def _threshold_summary(
    observations: Sequence[LabeledRegistrationObservation],
    *,
    threshold_index: int,
    ground_truth_pair_count: int,
) -> ThresholdRegistrationSummary:
    base_correct = sum(item.official_correct for item in observations)
    base_incorrect = len(observations) - base_correct
    if base_correct == 0 or base_incorrect == 0:
        raise ValueError("benchmark requires both correct and incorrect predictions")
    summaries = {
        route: _route_summary(
            observations,
            route_name=route,
            threshold_index=threshold_index,
            ground_truth_pair_count=ground_truth_pair_count,
            base_correct_count=base_correct,
            base_incorrect_count=base_incorrect,
        )
        for route in (
            "base",
            "global_only",
            "global_local",
            "global_local_tail",
        )
    }
    predecessor = summaries["global_local"]
    full = summaries["global_local_tail"]
    tail_retention = (
        full.correct_accepted_count / predecessor.correct_accepted_count
        if predecessor.correct_accepted_count
        else 0.0
    )
    tail_rejected_incorrect = (
        predecessor.incorrect_accepted_count - full.incorrect_accepted_count
    )
    precision_improved = bool(full.precision > summaries["base"].precision)
    correct_gate = bool(
        full.base_correct_retention >= MINIMUM_CORRECT_RETENTION
    )
    incorrect_gate = bool(
        full.base_incorrect_rejection_fraction
        >= MINIMUM_INCORRECT_REJECTION_FRACTION
    )
    tail_gate = bool(
        tail_retention >= MINIMUM_TAIL_PREDECESSOR_CORRECT_RETENTION
        and tail_rejected_incorrect > 0
    )
    return ThresholdRegistrationSummary(
        maximum_correspondence_distance=(
            observations[0]
            .blind.thresholds[threshold_index]
            .maximum_correspondence_distance
        ),
        base=summaries["base"],
        global_only=summaries["global_only"],
        global_local=predecessor,
        global_local_tail=full,
        tail_predecessor_correct_count=predecessor.correct_accepted_count,
        tail_retained_predecessor_correct_count=full.correct_accepted_count,
        tail_predecessor_correct_retention=tail_retention,
        tail_rejected_predecessor_incorrect_count=tail_rejected_incorrect,
        precision_improved=precision_improved,
        correct_retention_gate_passed=correct_gate,
        incorrect_rejection_gate_passed=incorrect_gate,
        tail_incremental_gate_passed=tail_gate,
        threshold_gate_passed=bool(
            precision_improved and correct_gate and incorrect_gate and tail_gate
        ),
    )


def evaluate_extracted_threedmatch_scene_guard(
    data_root: str | Path,
    phase28_artifact: str | Path,
    *,
    scene_name: str,
    evaluation_name: str,
    fragment_count: int,
    distance_thresholds: Sequence[float] = DEFAULT_DISTANCE_THRESHOLDS,
    maximum_points_per_fragment: int = DEFAULT_MAXIMUM_POINTS_PER_FRAGMENT,
    patch_size: int = DEFAULT_PATCH_SIZE,
    patch_count: int = DEFAULT_PATCH_COUNT,
) -> SceneRegistrationGuardEvaluation:
    """Evaluate one extracted scene, materializing all guards before labels."""

    root = Path(data_root)
    thresholds = tuple(float(value) for value in distance_thresholds)
    if (
        not thresholds
        or tuple(sorted(set(thresholds))) != thresholds
        or any(not math.isfinite(value) or value <= 0.0 for value in thresholds)
    ):
        raise ValueError("distance thresholds must be unique, positive, and sorted")
    if fragment_count <= 0:
        raise ValueError("fragment_count must be positive")
    if patch_size < 9 or patch_count <= 0 or patch_count > patch_size:
        raise ValueError("invalid patch size or patch count")
    model_path = Path(phase28_artifact)
    model = load_phase28_predecessor_model(model_path)
    prediction_path = root / evaluation_name / "3dmatch.log"
    predictions = read_registration_log(prediction_path)
    eligible_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.target_index - prediction.source_index > 1
    )
    fragments: dict[int, _SampledFragment] = {}
    for index in range(fragment_count):
        points = read_binary_ply_xyz(
            root / scene_name / f"cloud_bin_{index}.ply"
        )
        sampled, original_indices = deterministic_fragment_sample(
            points,
            maximum_points_per_fragment,
        )
        fragments[index] = _SampledFragment(
            raw_point_count=points.shape[0],
            points=sampled,
            original_indices=original_indices,
        )
    target_trees = {
        index: cKDTree(fragment.points)
        for index, fragment in fragments.items()
    }

    # Ground truth is intentionally not read until every blind observation exists.
    blind_observations = tuple(
        _blind_observation(
            prediction,
            fragments[prediction.source_index],
            fragments[prediction.target_index],
            target_trees[prediction.target_index],
            distance_thresholds=thresholds,
            patch_size=patch_size,
            patch_count=patch_count,
            model=model,
        )
        for prediction in eligible_predictions
    )

    evaluation_root = root / evaluation_name
    ground_truth = read_registration_log(evaluation_root / "gt.log")
    information = read_registration_info(evaluation_root / "gt.info")
    labeled = _label_observations(blind_observations, ground_truth, information)
    ground_truth_pair_count = sum(
        entry.target_index - entry.source_index > 1 for entry in ground_truth
    )
    summaries = tuple(
        _threshold_summary(
            labeled,
            threshold_index=index,
            ground_truth_pair_count=ground_truth_pair_count,
        )
        for index in range(len(thresholds))
    )
    return SceneRegistrationGuardEvaluation(
        scene_name=scene_name,
        evaluation_name=evaluation_name,
        phase28_artifact_path=str(model_path),
        phase28_artifact_sha256=_sha256(model_path),
        model=model,
        raw_prediction_count=len(predictions),
        eligible_prediction_count=len(eligible_predictions),
        ground_truth_overlap_pair_count=ground_truth_pair_count,
        maximum_points_per_fragment=maximum_points_per_fragment,
        patch_size=patch_size,
        patch_count=patch_count,
        distance_thresholds=thresholds,
        observations=labeled,
        threshold_summaries=summaries,
    )


def evaluate_threedmatch_registration_guard(
    data_root: str | Path,
    phase28_artifact: str | Path,
    *,
    distance_thresholds: Sequence[float] = DEFAULT_DISTANCE_THRESHOLDS,
    maximum_points_per_fragment: int = DEFAULT_MAXIMUM_POINTS_PER_FRAGMENT,
    patch_size: int = DEFAULT_PATCH_SIZE,
    patch_count: int = DEFAULT_PATCH_COUNT,
) -> ThreeDMatchRegistrationGuardResult:
    """Run blind observations first, then join official labels for scoring."""

    root = Path(data_root)
    fragment_archive, evaluation_archive = prepare_redkitchen_data(root)
    scene = evaluate_extracted_threedmatch_scene_guard(
        root,
        phase28_artifact,
        scene_name=SCENE_NAME,
        evaluation_name=EVALUATION_NAME,
        fragment_count=FRAGMENT_COUNT,
        distance_thresholds=distance_thresholds,
        maximum_points_per_fragment=maximum_points_per_fragment,
        patch_size=patch_size,
        patch_count=patch_count,
    )
    phase32_supported = all(
        summary.threshold_gate_passed for summary in scene.threshold_summaries
    )
    tail_supported = all(
        summary.tail_incremental_gate_passed
        for summary in scene.threshold_summaries
    )
    return ThreeDMatchRegistrationGuardResult(
        artifact_schema="pftf_alpha_threedmatch_registration_guard_phase32/v1",
        role="real_fragment_registration_guard_benchmark",
        dataset_name=SCENE_NAME,
        dataset_license_boundary="7-Scenes data are non-commercial-use only",
        information_boundary=(
            "fragment_coordinates_and_external_3dmatch_prediction_transforms_"
            "only_during_guard_execution; gt_log_and_gt_info_joined_after_all_"
            "blind_observations_for_evaluation_only"
        ),
        label_blind_execution_order=(
            "materialize_all_prediction_coordinate_guard_observations_then_"
            "read_and_join_ground_truth"
        ),
        fragment_archive=fragment_archive,
        evaluation_archive=evaluation_archive,
        phase28_artifact_path=scene.phase28_artifact_path,
        phase28_artifact_sha256=scene.phase28_artifact_sha256,
        prediction_log_name="3dmatch.log",
        raw_prediction_count=scene.raw_prediction_count,
        eligible_prediction_count=scene.eligible_prediction_count,
        ground_truth_overlap_pair_count=scene.ground_truth_overlap_pair_count,
        official_error_threshold_squared=OFFICIAL_ERROR_THRESHOLD_SQUARED,
        maximum_points_per_fragment=scene.maximum_points_per_fragment,
        patch_size=scene.patch_size,
        patch_count=scene.patch_count,
        distance_thresholds=scene.distance_thresholds,
        global_signature_rejection_cutoff=scene.model.rejection_cutoff,
        local_percentile95_rejection_cutoff=(
            EXPECTED_LOCAL_REJECTION_CUTOFF
        ),
        isolated_tail_ratio_rejection_cutoff=(
            EXPECTED_TAIL_REJECTION_CUTOFF
        ),
        observations=scene.observations,
        threshold_summaries=scene.threshold_summaries,
        real_registration_labels_supported=bool(
            scene.ground_truth_overlap_pair_count > 0
            and len(scene.observations) > 0
        ),
        phase32_supported=phase32_supported,
        tail_sensitive_real_registration_supported=tail_supported,
        real_correspondence_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("benchmark-data/3dmatch_redkitchen"),
    )
    parser.add_argument(
        "--phase28-artifact",
        type=Path,
        default=Path(
            "benchmark-out/local_spatial_residual_guard_phase28.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark-out/threedmatch_registration_guard_phase32.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_threedmatch_registration_guard(
        args.data_root,
        args.phase28_artifact,
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
