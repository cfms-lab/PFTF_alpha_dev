"""Frozen Phase-44 transfer panel for confidence-weighted filtration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .confidence_alpha_panel import (
    ReferenceSurfaceFamily,
    _rotation_matrix,
    _surface_points,
)
from .synthetic import PanelSplit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROTOCOL_SCHEMA = "pftf_alpha_confidence_alpha_transfer_panel_phase44/v1"
REFERENCE_POINT_COUNT = 1536
CALIBRATION_SEEDS = (44_001,)
HELD_OUT_SEEDS = (44_101, 44_102, 44_103)
CONTINUOUS_STRENGTHS = (0.5, 1.0, 2.0, 4.0)
BINARY_CONFIDENCE_THRESHOLDS = (0.25, 0.5, 0.75)
MINIMUM_SELECTED_CELL_FRACTION = 0.50
MAXIMUM_SELECTED_CELL_FRACTION = 0.98
SURFACE_SAMPLE_COUNT = 768
FSCORE_THRESHOLD_FRACTION = 0.05
MINIMUM_JOINT_WIN_FRACTION = 2.0 / 3.0


class TransferStressProfile(StrEnum):
    DENSITY_SHIFT = "density_shift"
    TARGET_OCCLUSION = "target_occlusion"
    LOCAL_NONRIGID_WARP = "local_nonrigid_warp"


_PROFILE_COUNTS: dict[TransferStressProfile, tuple[int, int]] = {
    TransferStressProfile.DENSITY_SHIFT: (48, 96),
    TransferStressProfile.TARGET_OCCLUSION: (72, 72),
    TransferStressProfile.LOCAL_NONRIGID_WARP: (72, 72),
}

_EXPECTED_BETTI: dict[ReferenceSurfaceFamily, tuple[int, int, int]] = {
    ReferenceSurfaceFamily.SPHERE: (1, 0, 1),
    ReferenceSurfaceFamily.TORUS: (1, 2, 1),
    ReferenceSurfaceFamily.DISCONNECTED_SPHERES: (2, 0, 2),
}


@dataclass(frozen=True)
class ConfidenceAlphaTransferCase:
    family: ReferenceSurfaceFamily
    profile: TransferStressProfile
    split: PanelSplit
    seed: int
    anchor_points: FloatArray
    target_points: FloatArray
    reference_points: FloatArray
    point_component_labels: IntArray
    expected_surface_betti: tuple[int, int, int]
    characteristic_length: float
    rigid_rotation_degrees: float
    rigid_translation_fraction: float
    observation_noise: float
    occlusion_retained_fraction: float
    local_warp_fraction: float

    def __post_init__(self) -> None:
        anchor = np.asarray(self.anchor_points, dtype=np.float64)
        target = np.asarray(self.target_points, dtype=np.float64)
        reference = np.asarray(self.reference_points, dtype=np.float64)
        for name, points in (
            ("anchor_points", anchor),
            ("target_points", target),
            ("reference_points", reference),
        ):
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 16:
                raise ValueError(f"{name} must have shape (n, 3) with n >= 16")
            if not np.all(np.isfinite(points)):
                raise ValueError(f"{name} must contain finite coordinates")
        points = np.vstack((anchor, target))
        if np.unique(points, axis=0).shape[0] != points.shape[0]:
            raise ValueError("combined observation points must be unique")
        labels = np.asarray(self.point_component_labels, dtype=np.int64)
        if labels.shape != (points.shape[0],):
            raise ValueError("point_component_labels must match the point count")
        expected = tuple(int(value) for value in self.expected_surface_betti)
        if expected != _EXPECTED_BETTI[self.family]:
            raise ValueError("expected topology does not match the analytic family")
        if np.any(labels < 0) or np.any(labels >= expected[0]):
            raise ValueError("component labels are outside the expected range")
        if (
            not math.isfinite(self.characteristic_length)
            or self.characteristic_length <= 0
        ):
            raise ValueError("characteristic_length must be finite and positive")
        object.__setattr__(self, "anchor_points", np.ascontiguousarray(anchor))
        object.__setattr__(self, "target_points", np.ascontiguousarray(target))
        object.__setattr__(self, "reference_points", np.ascontiguousarray(reference))
        object.__setattr__(
            self, "point_component_labels", np.ascontiguousarray(labels)
        )
        object.__setattr__(self, "expected_surface_betti", expected)

    @property
    def points(self) -> FloatArray:
        return np.vstack((self.anchor_points, self.target_points))

    @property
    def expected_components(self) -> int:
        return self.expected_surface_betti[0]

    @property
    def case_id(self) -> str:
        return (
            f"{self.split.value}_{self.family.value}_{self.profile.value}_{self.seed}"
        )


@dataclass(frozen=True)
class ConfidenceAlphaTransferProtocol:
    artifact_schema: str
    role: str
    families: tuple[str, ...]
    profiles: tuple[dict[str, float | int | str], ...]
    reference_point_count: int
    calibration_seeds: tuple[int, ...]
    held_out_seeds: tuple[int, ...]
    calibration_case_count: int
    held_out_case_count: int
    continuous_strengths: tuple[float, ...]
    binary_confidence_thresholds: tuple[float, ...]
    minimum_selected_cell_fraction: float
    maximum_selected_cell_fraction: float
    surface_sample_count: int
    fscore_threshold_fraction: float
    critical_score_selection: str
    calibration_objective: str
    frozen_comparators: tuple[str, ...]
    validation_gate: str
    reference_boundary: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "families",
            "profiles",
            "calibration_seeds",
            "held_out_seeds",
            "continuous_strengths",
            "binary_confidence_thresholds",
            "frozen_comparators",
        ):
            payload[key] = list(payload[key])
        return payload


def _occluded_surface_points(
    family: ReferenceSurfaceFamily,
    count: int,
    rng: np.random.Generator,
    view_direction: FloatArray,
) -> tuple[FloatArray, IntArray]:
    candidates, labels = _surface_points(family, 2 * count, rng)
    visibility = candidates @ view_direction
    selected = np.argsort(visibility, kind="stable")[-count:]
    return candidates[selected], labels[selected]


def _local_warp(
    points: FloatArray,
    rng: np.random.Generator,
    *,
    characteristic_length: float,
    warp_fraction: float,
) -> FloatArray:
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    center = points[int(rng.integers(0, points.shape[0]))]
    radius = 0.30 * characteristic_length
    distance_squared = np.sum((points - center) ** 2, axis=1)
    weight = np.exp(-distance_squared / max(radius**2, np.finfo(float).eps))
    return points + (
        warp_fraction * characteristic_length * weight[:, None] * direction
    )


def make_confidence_alpha_transfer_case(
    family: ReferenceSurfaceFamily | str,
    profile: TransferStressProfile | str,
    *,
    split: PanelSplit | str,
    seed: int,
    reference_point_count: int = REFERENCE_POINT_COUNT,
) -> ConfidenceAlphaTransferCase:
    selected_family = ReferenceSurfaceFamily(family)
    selected_profile = TransferStressProfile(profile)
    selected_split = PanelSplit(split)
    if selected_split not in (PanelSplit.CALIBRATION, PanelSplit.HELD_OUT):
        raise ValueError("Phase-44 panel permits calibration or held_out only")
    if reference_point_count < 96:
        raise ValueError("reference_point_count must be at least 96")

    anchor_count, target_count = _PROFILE_COUNTS[selected_profile]
    anchor_rng = np.random.default_rng(seed)
    target_rng = np.random.default_rng(seed + 1_000_003)
    reference_rng = np.random.default_rng(seed + 2_000_003)
    transform_rng = np.random.default_rng(seed + 3_000_003)
    reference, _ = _surface_points(
        selected_family, reference_point_count, reference_rng
    )
    characteristic_length = float(np.linalg.norm(np.ptp(reference, axis=0)))
    anchor, anchor_labels = _surface_points(
        selected_family, anchor_count, anchor_rng
    )
    view_direction = transform_rng.normal(size=3)
    view_direction /= np.linalg.norm(view_direction)
    if selected_profile is TransferStressProfile.TARGET_OCCLUSION:
        target, target_labels = _occluded_surface_points(
            selected_family,
            target_count,
            target_rng,
            view_direction,
        )
        occlusion_retained_fraction = 0.5
    else:
        target, target_labels = _surface_points(
            selected_family, target_count, target_rng
        )
        occlusion_retained_fraction = 1.0

    observation_noise = 0.005
    rigid_rotation_degrees = 3.0
    rigid_translation_fraction = 0.035
    local_warp_fraction = (
        0.08
        if selected_profile is TransferStressProfile.LOCAL_NONRIGID_WARP
        else 0.0
    )
    anchor += anchor_rng.normal(scale=observation_noise, size=anchor.shape)
    target += target_rng.normal(scale=observation_noise, size=target.shape)
    if local_warp_fraction > 0.0:
        target = _local_warp(
            target,
            transform_rng,
            characteristic_length=characteristic_length,
            warp_fraction=local_warp_fraction,
        )
    rotation_axis = transform_rng.normal(size=3)
    target = target @ _rotation_matrix(
        rotation_axis, rigid_rotation_degrees
    ).T
    translation_direction = transform_rng.normal(size=3)
    translation_direction /= np.linalg.norm(translation_direction)
    target += (
        rigid_translation_fraction
        * characteristic_length
        * translation_direction
    )
    return ConfidenceAlphaTransferCase(
        family=selected_family,
        profile=selected_profile,
        split=selected_split,
        seed=int(seed),
        anchor_points=anchor,
        target_points=target,
        reference_points=reference,
        point_component_labels=np.concatenate((anchor_labels, target_labels)),
        expected_surface_betti=_EXPECTED_BETTI[selected_family],
        characteristic_length=characteristic_length,
        rigid_rotation_degrees=rigid_rotation_degrees,
        rigid_translation_fraction=rigid_translation_fraction,
        observation_noise=observation_noise,
        occlusion_retained_fraction=occlusion_retained_fraction,
        local_warp_fraction=local_warp_fraction,
    )


def make_confidence_alpha_transfer_panel(
    split: PanelSplit | str,
) -> tuple[ConfidenceAlphaTransferCase, ...]:
    selected_split = PanelSplit(split)
    if selected_split not in (PanelSplit.CALIBRATION, PanelSplit.HELD_OUT):
        raise ValueError("Phase-44 panel permits calibration or held_out only")
    seeds = (
        CALIBRATION_SEEDS
        if selected_split is PanelSplit.CALIBRATION
        else HELD_OUT_SEEDS
    )
    return tuple(
        make_confidence_alpha_transfer_case(
            family,
            profile,
            split=selected_split,
            seed=seed,
        )
        for family in ReferenceSurfaceFamily
        for profile in TransferStressProfile
        for seed in seeds
    )


def preregister_confidence_alpha_transfer_panel() -> ConfidenceAlphaTransferProtocol:
    profiles = tuple(
        {
            "profile": profile.value,
            "anchor_point_count": _PROFILE_COUNTS[profile][0],
            "target_point_count": _PROFILE_COUNTS[profile][1],
            "occlusion_retained_fraction": (
                0.5 if profile is TransferStressProfile.TARGET_OCCLUSION else 1.0
            ),
            "local_warp_fraction": (
                0.08
                if profile is TransferStressProfile.LOCAL_NONRIGID_WARP
                else 0.0
            ),
        }
        for profile in TransferStressProfile
    )
    block_count = len(ReferenceSurfaceFamily) * len(TransferStressProfile)
    return ConfidenceAlphaTransferProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="pre_method_confidence_filtration_transfer_protocol",
        families=tuple(family.value for family in ReferenceSurfaceFamily),
        profiles=profiles,
        reference_point_count=REFERENCE_POINT_COUNT,
        calibration_seeds=CALIBRATION_SEEDS,
        held_out_seeds=HELD_OUT_SEEDS,
        calibration_case_count=block_count * len(CALIBRATION_SEEDS),
        held_out_case_count=block_count * len(HELD_OUT_SEEDS),
        continuous_strengths=CONTINUOUS_STRENGTHS,
        binary_confidence_thresholds=BINARY_CONFIDENCE_THRESHOLDS,
        minimum_selected_cell_fraction=MINIMUM_SELECTED_CELL_FRACTION,
        maximum_selected_cell_fraction=MAXIMUM_SELECTED_CELL_FRACTION,
        surface_sample_count=SURFACE_SAMPLE_COUNT,
        fscore_threshold_fraction=FSCORE_THRESHOLD_FRACTION,
        critical_score_selection=(
            "scan every finite unique top-cell score; among adjacent scores whose "
            "lower selected-cell fraction lies in [0.50, 0.98], choose the "
            "largest log-score gap and use its geometric midpoint; ties prefer "
            "the lower selected fraction and lower threshold"
        ),
        calibration_objective=(
            "select only continuous strength and binary confidence threshold by "
            "minimum calibration mean normalized Chamfer-squared + normalized "
            "Hausdorff + 0.05 * Betti L1 error; ties prefer lower parameter"
        ),
        frozen_comparators=(
            "anchor_density_B4",
            "fused_density_B4",
            "fused_pca_B5",
            "binary_confidence_deletion",
        ),
        validation_gate=(
            "continuous weighting must have lower held-out mean geometry and "
            "objective than anchor B4, fused B4, and binary deletion; no larger "
            "mean Betti error than all three; repeat stability no larger than "
            "fused B4 and binary deletion; lower objective than B5; and joint "
            "casewise objective wins over anchor/fused/binary in at least 2/3 "
            "of held-out cases"
        ),
        reference_boundary=(
            "reference points, family/component labels, profile identity, and "
            "applied perturbations are evaluation-only; critical-gap selection "
            "and point confidence use observed coordinates only"
        ),
        claim_boundary=(
            "a positive result supports transfer of the bounded confidence-"
            "weighted filtration only; it is not a classical local-alpha complex, "
            "a learned global alpha, real-scan evidence, or deployment evidence"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_confidence_alpha_transfer_panel().to_dict(),
        indent=2,
        sort_keys=True,
    ) + "\n"
    output.write_text(text, encoding="utf-8")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark-out/confidence_alpha_transfer_protocol_phase44.json"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_protocol(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
