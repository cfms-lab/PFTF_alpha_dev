"""Phase-33 unchanged-guard transfer audit on a second 3DMatch scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .local_spatial_residual_guard import EXPECTED_LOCAL_REJECTION_CUTOFF
from .tail_sensitive_local_guard import EXPECTED_TAIL_REJECTION_CUTOFF
from .threedmatch_redkitchen import ThreeDMatchArchiveVerification
from .threedmatch_registration_guard import (
    DEFAULT_DISTANCE_THRESHOLDS,
    DEFAULT_MAXIMUM_POINTS_PER_FRAGMENT,
    DEFAULT_PATCH_COUNT,
    DEFAULT_PATCH_SIZE,
    OFFICIAL_ERROR_THRESHOLD_SQUARED,
    LabeledRegistrationObservation,
    ThresholdRegistrationSummary,
    evaluate_extracted_threedmatch_scene_guard,
)
from .threedmatch_scene import (
    MARYLAND_HOTEL3_SPEC,
    ThreeDMatchSceneSpec,
    fetch_threedmatch_scene_archives,
    prepare_threedmatch_scene_data,
)

EXPECTED_GLOBAL_REJECTION_CUTOFF = 0.18181536333942858
EXPECTED_REFERENCE_SCHEMA = (
    "pftf_alpha_threedmatch_registration_guard_phase32/v1"
)
EXPECTED_REFERENCE_SHA256 = (
    "b7653adda0f0b93f14fda54bb57a4559c4a00863e4f22702b4bc14650442cb4d"
)


@dataclass(frozen=True)
class FrozenTransferProtocol:
    reference_artifact_path: str
    reference_artifact_sha256: str
    reference_dataset_name: str
    reference_phase32_supported: bool
    reference_tail_supported: bool
    phase28_artifact_sha256: str
    distance_thresholds: tuple[float, ...]
    maximum_points_per_fragment: int
    patch_size: int
    patch_count: int
    global_signature_rejection_cutoff: float
    local_percentile95_rejection_cutoff: float
    isolated_tail_ratio_rejection_cutoff: float
    official_error_threshold_squared: float
    protocol_identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_artifact_path": self.reference_artifact_path,
            "reference_artifact_sha256": self.reference_artifact_sha256,
            "reference_dataset_name": self.reference_dataset_name,
            "reference_phase32_supported": self.reference_phase32_supported,
            "reference_tail_supported": self.reference_tail_supported,
            "phase28_artifact_sha256": self.phase28_artifact_sha256,
            "distance_thresholds": list(self.distance_thresholds),
            "maximum_points_per_fragment": self.maximum_points_per_fragment,
            "patch_size": self.patch_size,
            "patch_count": self.patch_count,
            "global_signature_rejection_cutoff": (
                self.global_signature_rejection_cutoff
            ),
            "local_percentile95_rejection_cutoff": (
                self.local_percentile95_rejection_cutoff
            ),
            "isolated_tail_ratio_rejection_cutoff": (
                self.isolated_tail_ratio_rejection_cutoff
            ),
            "official_error_threshold_squared": (
                self.official_error_threshold_squared
            ),
            "protocol_identity_sha256": self.protocol_identity_sha256,
        }


@dataclass(frozen=True)
class ThreeDMatchTransferAuditResult:
    artifact_schema: str
    role: str
    scene: ThreeDMatchSceneSpec
    fragment_archive: ThreeDMatchArchiveVerification
    evaluation_archive: ThreeDMatchArchiveVerification
    frozen_protocol: FrozenTransferProtocol
    information_boundary: str
    label_blind_execution_order: str
    raw_prediction_count: int
    eligible_prediction_count: int
    ground_truth_overlap_pair_count: int
    observations: tuple[LabeledRegistrationObservation, ...]
    threshold_summaries: tuple[ThresholdRegistrationSummary, ...]
    phase33_audit_completed: bool
    second_scene_guard_supported: bool
    second_scene_tail_supported: bool
    negative_transfer_reproduced: bool
    cross_scene_guard_supported: bool
    tail_sensitive_real_registration_supported: bool
    real_registration_labels_supported: bool
    real_correspondence_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "scene": self.scene.to_dict(),
            "fragment_archive": self.fragment_archive.to_dict(),
            "evaluation_archive": self.evaluation_archive.to_dict(),
            "frozen_protocol": self.frozen_protocol.to_dict(),
            "information_boundary": self.information_boundary,
            "label_blind_execution_order": self.label_blind_execution_order,
            "raw_prediction_count": self.raw_prediction_count,
            "eligible_prediction_count": self.eligible_prediction_count,
            "ground_truth_overlap_pair_count": (
                self.ground_truth_overlap_pair_count
            ),
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "threshold_summaries": [
                summary.to_dict() for summary in self.threshold_summaries
            ],
            "phase33_audit_completed": self.phase33_audit_completed,
            "second_scene_guard_supported": self.second_scene_guard_supported,
            "second_scene_tail_supported": self.second_scene_tail_supported,
            "negative_transfer_reproduced": self.negative_transfer_reproduced,
            "cross_scene_guard_supported": self.cross_scene_guard_supported,
            "tail_sensitive_real_registration_supported": (
                self.tail_sensitive_real_registration_supported
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


def _finite_float(mapping: Mapping[str, object], key: str) -> float:
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"reference {key} must be finite")
    return value


def load_frozen_transfer_protocol(
    reference_phase32_artifact: str | Path,
) -> FrozenTransferProtocol:
    """Validate the opened Phase-32 artifact and freeze its exact protocol."""

    path = Path(reference_phase32_artifact)
    reference_sha256 = _sha256(path)
    if reference_sha256 != EXPECTED_REFERENCE_SHA256:
        raise ValueError("reference Phase-32 artifact SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reference Phase-32 artifact must contain an object")
    if payload.get("artifact_schema") != EXPECTED_REFERENCE_SCHEMA:
        raise ValueError("reference Phase-32 artifact schema mismatch")
    if payload.get("dataset_name") != "7-scenes-redkitchen":
        raise ValueError("reference artifact must be the opened redkitchen run")
    for key in (
        "phase32_supported",
        "tail_sensitive_real_registration_supported",
    ):
        if not isinstance(payload.get(key), bool):
            raise ValueError(f"reference {key} must be Boolean")
    thresholds = tuple(float(value) for value in payload["distance_thresholds"])
    expected_scalars = {
        "maximum_points_per_fragment": DEFAULT_MAXIMUM_POINTS_PER_FRAGMENT,
        "patch_size": DEFAULT_PATCH_SIZE,
        "patch_count": DEFAULT_PATCH_COUNT,
        "official_error_threshold_squared": OFFICIAL_ERROR_THRESHOLD_SQUARED,
        "global_signature_rejection_cutoff": (
            EXPECTED_GLOBAL_REJECTION_CUTOFF
        ),
        "local_percentile95_rejection_cutoff": (
            EXPECTED_LOCAL_REJECTION_CUTOFF
        ),
        "isolated_tail_ratio_rejection_cutoff": (
            EXPECTED_TAIL_REJECTION_CUTOFF
        ),
    }
    if thresholds != DEFAULT_DISTANCE_THRESHOLDS:
        raise ValueError("reference distance thresholds are not frozen defaults")
    for key, expected in expected_scalars.items():
        observed = _finite_float(payload, key)
        if observed != float(expected):
            raise ValueError(f"reference {key} does not match the frozen value")
    phase28_sha256 = str(payload["phase28_artifact_sha256"])
    canonical_protocol = {
        "distance_thresholds": list(thresholds),
        **expected_scalars,
        "phase28_artifact_sha256": phase28_sha256,
    }
    protocol_sha256 = hashlib.sha256(
        json.dumps(
            canonical_protocol,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return FrozenTransferProtocol(
        reference_artifact_path=str(path),
        reference_artifact_sha256=reference_sha256,
        reference_dataset_name=str(payload["dataset_name"]),
        reference_phase32_supported=bool(payload["phase32_supported"]),
        reference_tail_supported=bool(
            payload["tail_sensitive_real_registration_supported"]
        ),
        phase28_artifact_sha256=phase28_sha256,
        distance_thresholds=thresholds,
        maximum_points_per_fragment=int(
            expected_scalars["maximum_points_per_fragment"]
        ),
        patch_size=int(expected_scalars["patch_size"]),
        patch_count=int(expected_scalars["patch_count"]),
        global_signature_rejection_cutoff=float(
            expected_scalars["global_signature_rejection_cutoff"]
        ),
        local_percentile95_rejection_cutoff=float(
            expected_scalars["local_percentile95_rejection_cutoff"]
        ),
        isolated_tail_ratio_rejection_cutoff=float(
            expected_scalars["isolated_tail_ratio_rejection_cutoff"]
        ),
        official_error_threshold_squared=float(
            expected_scalars["official_error_threshold_squared"]
        ),
        protocol_identity_sha256=protocol_sha256,
    )


def evaluate_threedmatch_transfer_audit(
    data_root: str | Path,
    phase28_artifact: str | Path,
    reference_phase32_artifact: str | Path,
    *,
    scene: ThreeDMatchSceneSpec = MARYLAND_HOTEL3_SPEC,
) -> ThreeDMatchTransferAuditResult:
    """Run the unchanged Phase-32 route on one independently labeled scene."""

    protocol = load_frozen_transfer_protocol(reference_phase32_artifact)
    phase28_path = Path(phase28_artifact)
    if _sha256(phase28_path) != protocol.phase28_artifact_sha256:
        raise ValueError("Phase-28 model differs from the Phase-32 reference")
    fragment_archive, evaluation_archive = prepare_threedmatch_scene_data(
        data_root,
        scene,
    )
    evaluation = evaluate_extracted_threedmatch_scene_guard(
        data_root,
        phase28_path,
        scene_name=scene.scene_name,
        evaluation_name=scene.evaluation_name,
        fragment_count=scene.fragment_count,
        distance_thresholds=protocol.distance_thresholds,
        maximum_points_per_fragment=protocol.maximum_points_per_fragment,
        patch_size=protocol.patch_size,
        patch_count=protocol.patch_count,
    )
    if evaluation.model.rejection_cutoff != (
        protocol.global_signature_rejection_cutoff
    ):
        raise ValueError("loaded Phase-28 model cutoff differs from Phase 32")
    scene_supported = all(
        summary.threshold_gate_passed
        for summary in evaluation.threshold_summaries
    )
    scene_tail_supported = all(
        summary.tail_incremental_gate_passed
        for summary in evaluation.threshold_summaries
    )
    cross_scene_supported = bool(
        protocol.reference_phase32_supported and scene_supported
    )
    cross_scene_tail_supported = bool(
        protocol.reference_tail_supported and scene_tail_supported
    )
    return ThreeDMatchTransferAuditResult(
        artifact_schema="pftf_alpha_threedmatch_transfer_audit_phase33/v1",
        role="unchanged_negative_route_second_scene_transfer_audit",
        scene=scene,
        fragment_archive=fragment_archive,
        evaluation_archive=evaluation_archive,
        frozen_protocol=protocol,
        information_boundary=(
            "second_scene_fragment_coordinates_and_external_3dmatch_prediction_"
            "transforms_only_during_guard_execution; gt_log_and_gt_info_joined_"
            "after_all_blind_observations_for_evaluation_only; no_threshold_"
            "selection_or_retuning"
        ),
        label_blind_execution_order=(
            "validate_phase32_protocol_then_materialize_all_second_scene_"
            "prediction_coordinate_guard_observations_then_read_and_join_"
            "second_scene_ground_truth"
        ),
        raw_prediction_count=evaluation.raw_prediction_count,
        eligible_prediction_count=evaluation.eligible_prediction_count,
        ground_truth_overlap_pair_count=(
            evaluation.ground_truth_overlap_pair_count
        ),
        observations=evaluation.observations,
        threshold_summaries=evaluation.threshold_summaries,
        phase33_audit_completed=True,
        second_scene_guard_supported=scene_supported,
        second_scene_tail_supported=scene_tail_supported,
        negative_transfer_reproduced=bool(
            not protocol.reference_phase32_supported and not scene_supported
        ),
        cross_scene_guard_supported=cross_scene_supported,
        tail_sensitive_real_registration_supported=(
            cross_scene_tail_supported
        ),
        real_registration_labels_supported=bool(
            evaluation.ground_truth_overlap_pair_count > 0
            and evaluation.eligible_prediction_count > 0
        ),
        real_correspondence_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("benchmark-data/3dmatch_maryland_hotel3"),
    )
    parser.add_argument(
        "--phase28-artifact",
        type=Path,
        default=Path(
            "benchmark-out/local_spatial_residual_guard_phase28.json"
        ),
    )
    parser.add_argument(
        "--reference-phase32-artifact",
        type=Path,
        default=Path(
            "benchmark-out/threedmatch_registration_guard_phase32.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark-out/threedmatch_transfer_audit_phase33.json"
        ),
    )
    parser.add_argument("--download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.download:
        fetch_threedmatch_scene_archives(args.data_root, MARYLAND_HOTEL3_SPEC)
    result = evaluate_threedmatch_transfer_audit(
        args.data_root,
        args.phase28_artifact,
        args.reference_phase32_artifact,
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
