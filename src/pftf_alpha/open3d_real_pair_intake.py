"""Phase-31 observed local-evidence intake on real Open3D scan pairs."""

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
    LocalSpatialDisplacementEvidence,
    estimate_local_spatial_displacement_evidence,
)
from .local_spatial_residual_guard import EXPECTED_LOCAL_REJECTION_CUTOFF
from .matched_guard_signature import (
    MatchedGuardModel,
    MatchedGuardSignature,
    matched_guard_signature_from_evidence,
    score_matched_guard_signature,
)
from .matched_pair_consistency import (
    MatchedPairEvidence,
    estimate_matched_pair_evidence,
)
from .open3d_demo_icp import (
    ARCHIVE_NAME,
    DATASET_URL,
    DemoICPArchiveVerification,
    TransformationLogEntry,
    extract_demo_icp_archive,
    read_binary_pcd_xyz,
    read_transformation_log,
    transform_points,
)
from .tail_sensitive_local_guard import (
    EXPECTED_TAIL_REJECTION_CUTOFF,
    tail_feature_value,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_DISTANCE_THRESHOLDS = (0.02, 0.05)
DEFAULT_PATCH_SIZE = 96
DEFAULT_PATCH_COUNT = 8


@dataclass(frozen=True)
class AlignmentDirectionDiagnostic:
    direct_logged_median_nn_distance: float
    inverse_logged_median_nn_distance: float
    direct_logged_within_2cm_fraction: float
    inverse_logged_within_2cm_fraction: float
    inverse_direction_better: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CorrespondenceThresholdSummary:
    maximum_distance: float
    reciprocal_match_count: int
    source_coverage_fraction: float
    target_coverage_fraction: float
    median_distance: float
    percentile95_distance: float
    maximum_distance_observed: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RealPairPatchObservation:
    source_index: int
    target_index: int
    maximum_correspondence_distance: float
    patch_index: int
    pair_count: int
    presented_pair_map_sha256: str
    median_correspondence_distance: float
    percentile95_correspondence_distance: float
    maximum_correspondence_distance_observed: float
    matched_evidence: MatchedPairEvidence
    signature: MatchedGuardSignature
    model_score: float
    global_signature_below_cutoff: bool
    local_spatial_evidence: LocalSpatialDisplacementEvidence
    local_percentile95_below_cutoff: bool
    isolated_tail_ratio: float
    isolated_tail_ratio_below_cutoff: bool
    observational_guard_stack_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["matched_evidence"] = self.matched_evidence.to_dict()
        payload["signature"] = self.signature.to_dict()
        payload["local_spatial_evidence"] = self.local_spatial_evidence.to_dict()
        return payload


@dataclass(frozen=True)
class RealScanPairObservation:
    source_index: int
    target_index: int
    source_point_count: int
    target_point_count: int
    information_count_from_log: int
    logged_matrix: FloatArray
    source_to_target_matrix: FloatArray
    direction_diagnostic: AlignmentDirectionDiagnostic
    threshold_summaries: tuple[CorrespondenceThresholdSummary, ...]
    patches: tuple[RealPairPatchObservation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "source_point_count": self.source_point_count,
            "target_point_count": self.target_point_count,
            "information_count_from_log": self.information_count_from_log,
            "logged_matrix": self.logged_matrix.tolist(),
            "source_to_target_matrix": self.source_to_target_matrix.tolist(),
            "direction_diagnostic": self.direction_diagnostic.to_dict(),
            "threshold_summaries": [
                summary.to_dict() for summary in self.threshold_summaries
            ],
            "patches": [patch.to_dict() for patch in self.patches],
        }


@dataclass(frozen=True)
class Open3DRealPairIntakeResult:
    artifact_schema: str
    role: str
    dataset_url: str
    dataset_license: str
    information_boundary: str
    correspondence_method: str
    archive: DemoICPArchiveVerification
    phase28_artifact_path: str
    phase28_artifact_sha256: str
    global_signature_rejection_cutoff: float
    local_percentile95_rejection_cutoff: float
    isolated_tail_ratio_rejection_cutoff: float
    distance_thresholds: tuple[float, ...]
    patch_size: int
    patch_count_per_pair_and_threshold: int
    scan_pairs: tuple[RealScanPairObservation, ...]
    patch_observation_count: int
    observational_guard_stack_pass_count: int
    real_paired_scan_intake_supported: bool
    real_correspondence_supported: bool
    real_paired_scan_guard_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "dataset_url": self.dataset_url,
            "dataset_license": self.dataset_license,
            "information_boundary": self.information_boundary,
            "correspondence_method": self.correspondence_method,
            "archive": self.archive.to_dict(),
            "phase28_artifact_path": self.phase28_artifact_path,
            "phase28_artifact_sha256": self.phase28_artifact_sha256,
            "global_signature_rejection_cutoff": (
                self.global_signature_rejection_cutoff
            ),
            "local_percentile95_rejection_cutoff": (
                self.local_percentile95_rejection_cutoff
            ),
            "isolated_tail_ratio_rejection_cutoff": (
                self.isolated_tail_ratio_rejection_cutoff
            ),
            "distance_thresholds": list(self.distance_thresholds),
            "patch_size": self.patch_size,
            "patch_count_per_pair_and_threshold": (
                self.patch_count_per_pair_and_threshold
            ),
            "scan_pairs": [pair.to_dict() for pair in self.scan_pairs],
            "patch_observation_count": self.patch_observation_count,
            "observational_guard_stack_pass_count": (
                self.observational_guard_stack_pass_count
            ),
            "real_paired_scan_intake_supported": (
                self.real_paired_scan_intake_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_guard_supported": (
                self.real_paired_scan_guard_supported
            ),
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


def load_phase28_predecessor_model(path: str | Path) -> MatchedGuardModel:
    """Load the frozen Phase-27 score model carried by the Phase-28 artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_schema") != (
        "pftf_alpha_local_spatial_residual_guard_phase28/v1"
    ):
        raise ValueError("unexpected Phase-28 artifact schema")
    model_payload = payload.get("predecessor_model")
    if not isinstance(model_payload, dict):
        raise ValueError("Phase-28 artifact has no predecessor_model")
    tuple_fields = {
        "feature_names",
        "feature_center",
        "feature_scale",
        "coefficients",
    }
    normalized = {
        key: tuple(value) if key in tuple_fields else value
        for key, value in model_payload.items()
    }
    model = MatchedGuardModel(**normalized)
    if not model.calibration_valid:
        raise ValueError("Phase-28 predecessor model must be calibrated")
    return model


def _nearest_distances(source: FloatArray, target: FloatArray) -> FloatArray:
    distances, _ = cKDTree(target).query(source, k=1, workers=1)
    return np.asarray(distances, dtype=np.float64)


def _direction_diagnostic(
    source: FloatArray,
    target: FloatArray,
    entry: TransformationLogEntry,
) -> AlignmentDirectionDiagnostic:
    direct_distances = _nearest_distances(
        transform_points(source, entry.logged_matrix),
        target,
    )
    inverse_distances = _nearest_distances(
        transform_points(source, entry.source_to_target_matrix),
        target,
    )
    direct_median = float(np.median(direct_distances))
    inverse_median = float(np.median(inverse_distances))
    direct_fraction = float(np.mean(direct_distances <= 0.02))
    inverse_fraction = float(np.mean(inverse_distances <= 0.02))
    return AlignmentDirectionDiagnostic(
        direct_logged_median_nn_distance=direct_median,
        inverse_logged_median_nn_distance=inverse_median,
        direct_logged_within_2cm_fraction=direct_fraction,
        inverse_logged_within_2cm_fraction=inverse_fraction,
        inverse_direction_better=bool(
            inverse_median < direct_median and inverse_fraction > direct_fraction
        ),
    )


def _reciprocal_matches(
    source: FloatArray,
    target: FloatArray,
    maximum_distance: float,
) -> tuple[IntArray, IntArray, FloatArray]:
    target_tree = cKDTree(target)
    forward_distances, target_indices = target_tree.query(
        source,
        k=1,
        workers=1,
    )
    source_tree = cKDTree(source)
    _, backward_indices = source_tree.query(target, k=1, workers=1)
    source_indices = np.arange(source.shape[0], dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    forward_distances = np.asarray(forward_distances, dtype=np.float64)
    backward_indices = np.asarray(backward_indices, dtype=np.int64)
    reciprocal = backward_indices[target_indices] == source_indices
    selected = reciprocal & (forward_distances <= maximum_distance)
    return (
        source_indices[selected],
        target_indices[selected],
        forward_distances[selected],
    )


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


def _patch_observations(
    aligned_source: FloatArray,
    target: FloatArray,
    source_indices: IntArray,
    target_indices: IntArray,
    distances: FloatArray,
    *,
    entry: TransformationLogEntry,
    maximum_distance: float,
    patch_size: int,
    patch_count: int,
    model: MatchedGuardModel,
) -> tuple[RealPairPatchObservation, ...]:
    matched_source = aligned_source[source_indices]
    matched_target = target[target_indices]
    if matched_source.shape[0] < patch_size:
        raise ValueError("too few reciprocal matches for the requested patch size")
    anchors = _farthest_anchor_indices(
        matched_source,
        min(patch_count, matched_source.shape[0]),
    )
    tree = cKDTree(matched_source)
    observations = []
    for patch_index, anchor_index in enumerate(anchors):
        _, selected = tree.query(
            matched_source[anchor_index],
            k=patch_size,
            workers=1,
        )
        selected = np.asarray(selected, dtype=np.int64).reshape(-1)
        selected.sort()
        primary = matched_source[selected]
        repeat = matched_target[selected]
        patch_distances = distances[selected]
        evidence = replace(
            estimate_matched_pair_evidence(primary, repeat),
            information_boundary=(
                "ordered_metadata_aligned_reciprocal_nearest_neighbor_"
                "candidates_only; physical_point_identity_unknown"
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
        global_pass = bool(model_score < model.rejection_cutoff)
        local_pass = bool(
            local.percentile95_local_residual
            < EXPECTED_LOCAL_REJECTION_CUTOFF
        )
        tail_pass = bool(tail_ratio < EXPECTED_TAIL_REJECTION_CUTOFF)
        observations.append(
            RealPairPatchObservation(
                source_index=entry.source_index,
                target_index=entry.target_index,
                maximum_correspondence_distance=maximum_distance,
                patch_index=patch_index,
                pair_count=patch_size,
                presented_pair_map_sha256=_pair_map_sha256(
                    source_indices[selected],
                    target_indices[selected],
                ),
                median_correspondence_distance=float(
                    np.median(patch_distances)
                ),
                percentile95_correspondence_distance=float(
                    np.percentile(patch_distances, 95.0)
                ),
                maximum_correspondence_distance_observed=float(
                    np.max(patch_distances)
                ),
                matched_evidence=evidence,
                signature=signature,
                model_score=model_score,
                global_signature_below_cutoff=global_pass,
                local_spatial_evidence=local,
                local_percentile95_below_cutoff=local_pass,
                isolated_tail_ratio=tail_ratio,
                isolated_tail_ratio_below_cutoff=tail_pass,
                observational_guard_stack_passed=bool(
                    global_pass and local_pass and tail_pass
                ),
            )
        )
    return tuple(observations)


def _evaluate_pair(
    data_root: Path,
    entry: TransformationLogEntry,
    *,
    distance_thresholds: tuple[float, ...],
    patch_size: int,
    patch_count: int,
    model: MatchedGuardModel,
) -> RealScanPairObservation:
    source = read_binary_pcd_xyz(
        data_root / f"cloud_bin_{entry.source_index}.pcd"
    )
    target = read_binary_pcd_xyz(
        data_root / f"cloud_bin_{entry.target_index}.pcd"
    )
    diagnostic = _direction_diagnostic(source, target, entry)
    if not diagnostic.inverse_direction_better:
        raise ValueError(
            "inverse log direction did not pass the frozen alignment check"
        )
    aligned_source = transform_points(source, entry.source_to_target_matrix)
    summaries = []
    patches = []
    for threshold in distance_thresholds:
        source_indices, target_indices, distances = _reciprocal_matches(
            aligned_source,
            target,
            threshold,
        )
        if distances.size == 0:
            raise ValueError("no reciprocal correspondences passed the threshold")
        summaries.append(
            CorrespondenceThresholdSummary(
                maximum_distance=threshold,
                reciprocal_match_count=int(distances.size),
                source_coverage_fraction=float(distances.size / source.shape[0]),
                target_coverage_fraction=float(distances.size / target.shape[0]),
                median_distance=float(np.median(distances)),
                percentile95_distance=float(np.percentile(distances, 95.0)),
                maximum_distance_observed=float(np.max(distances)),
            )
        )
        patches.extend(
            _patch_observations(
                aligned_source,
                target,
                source_indices,
                target_indices,
                distances,
                entry=entry,
                maximum_distance=threshold,
                patch_size=patch_size,
                patch_count=patch_count,
                model=model,
            )
        )
    return RealScanPairObservation(
        source_index=entry.source_index,
        target_index=entry.target_index,
        source_point_count=source.shape[0],
        target_point_count=target.shape[0],
        information_count_from_log=entry.information_count,
        logged_matrix=entry.logged_matrix,
        source_to_target_matrix=entry.source_to_target_matrix,
        direction_diagnostic=diagnostic,
        threshold_summaries=tuple(summaries),
        patches=tuple(patches),
    )


def evaluate_open3d_real_pair_intake(
    data_root: str | Path,
    phase28_artifact: str | Path,
    *,
    distance_thresholds: Sequence[float] = DEFAULT_DISTANCE_THRESHOLDS,
    patch_size: int = DEFAULT_PATCH_SIZE,
    patch_count: int = DEFAULT_PATCH_COUNT,
) -> Open3DRealPairIntakeResult:
    """Evaluate observed-only guard features without asserting true identity."""

    root = Path(data_root)
    selected_thresholds = tuple(float(value) for value in distance_thresholds)
    invalid_threshold = any(
        not math.isfinite(value) or value <= 0.0
        for value in selected_thresholds
    )
    if (
        not selected_thresholds
        or invalid_threshold
        or tuple(sorted(set(selected_thresholds))) != selected_thresholds
    ):
        raise ValueError("distance thresholds must be unique, positive, and sorted")
    if patch_size < 9 or patch_count <= 0:
        raise ValueError("patch_size must be at least 9 and patch_count positive")
    verification = extract_demo_icp_archive(root / ARCHIVE_NAME, root)
    artifact_path = Path(phase28_artifact)
    model = load_phase28_predecessor_model(artifact_path)
    entries = read_transformation_log(root / "init.log")
    pairs = tuple(
        _evaluate_pair(
            root,
            entry,
            distance_thresholds=selected_thresholds,
            patch_size=patch_size,
            patch_count=patch_count,
            model=model,
        )
        for entry in entries
    )
    patches = tuple(patch for pair in pairs for patch in pair.patches)
    direction_supported = all(
        pair.direction_diagnostic.inverse_direction_better for pair in pairs
    )
    return Open3DRealPairIntakeResult(
        artifact_schema="pftf_alpha_open3d_real_pair_intake_phase31/v1",
        role="real_paired_scan_observed_local_evidence_intake",
        dataset_url=DATASET_URL,
        dataset_license=(
            "CC BY 3.0 as stated by the official Open3D DemoICPPointClouds API"
        ),
        information_boundary=(
            "verified_real_scan_coordinates_and_Open3D_transform_metadata; "
            "candidate_pairs_are_distance_gated_reciprocal_nearest_neighbors; "
            "physical_point_identity_and_geometry_topology_endpoint_are_unknown"
        ),
        correspondence_method=(
            "inverse_logged_source_to_target_transform_then_reciprocal_1NN"
        ),
        archive=verification,
        phase28_artifact_path=str(artifact_path),
        phase28_artifact_sha256=_sha256(artifact_path),
        global_signature_rejection_cutoff=model.rejection_cutoff,
        local_percentile95_rejection_cutoff=(
            EXPECTED_LOCAL_REJECTION_CUTOFF
        ),
        isolated_tail_ratio_rejection_cutoff=(
            EXPECTED_TAIL_REJECTION_CUTOFF
        ),
        distance_thresholds=selected_thresholds,
        patch_size=patch_size,
        patch_count_per_pair_and_threshold=patch_count,
        scan_pairs=pairs,
        patch_observation_count=len(patches),
        observational_guard_stack_pass_count=sum(
            patch.observational_guard_stack_passed for patch in patches
        ),
        real_paired_scan_intake_supported=bool(
            verification.verified and direction_supported and len(pairs) == 2
        ),
        real_correspondence_supported=False,
        real_paired_scan_guard_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("benchmark-data/open3d_demo_icp"),
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
        default=Path("benchmark-out/open3d_real_pair_intake_phase31.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_open3d_real_pair_intake(
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
