"""Frozen analytic multi-view panel for Phase-43 confidence-alpha research."""

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

from .synthetic import PanelSplit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROTOCOL_SCHEMA = "pftf_alpha_confidence_alpha_panel_phase43/v1"
POINTS_PER_VIEW = 72
REFERENCE_POINT_COUNT = 1536
CALIBRATION_SEEDS = (43_001,)
HELD_OUT_SEEDS = (43_101, 43_102, 43_103)
SCALE_QUANTILES = (0.08, 0.12, 0.18, 0.26, 0.36, 0.50, 0.68, 0.84)
CONTINUOUS_STRENGTHS = (0.5, 1.0, 2.0, 4.0)
BINARY_CONFIDENCE_THRESHOLDS = (0.25, 0.5, 0.75)
SURFACE_SAMPLE_COUNT = 768
FSCORE_THRESHOLD_FRACTION = 0.05


class ReferenceSurfaceFamily(StrEnum):
    SPHERE = "sphere"
    TORUS = "torus"
    DISCONNECTED_SPHERES = "disconnected_spheres"


class MisregistrationProfile(StrEnum):
    MILD = "mild"
    COHERENT = "coherent"


_EXPECTED_BETTI: dict[ReferenceSurfaceFamily, tuple[int, int, int]] = {
    ReferenceSurfaceFamily.SPHERE: (1, 0, 1),
    ReferenceSurfaceFamily.TORUS: (1, 2, 1),
    ReferenceSurfaceFamily.DISCONNECTED_SPHERES: (2, 0, 2),
}

_PROFILE_PARAMETERS: dict[MisregistrationProfile, tuple[float, float, float]] = {
    # rotation degrees, translation / characteristic length, observation noise
    MisregistrationProfile.MILD: (2.0, 0.025, 0.003),
    MisregistrationProfile.COHERENT: (6.0, 0.075, 0.005),
}


@dataclass(frozen=True)
class ConfidenceAlphaCase:
    """Observed anchor/target views and evaluation-only analytic truth."""

    family: ReferenceSurfaceFamily
    profile: MisregistrationProfile
    split: PanelSplit
    seed: int
    anchor_points: FloatArray
    target_points: FloatArray
    reference_points: FloatArray
    point_view_labels: IntArray
    point_component_labels: IntArray
    expected_surface_betti: tuple[int, int, int]
    characteristic_length: float
    rotation_degrees: float
    translation_fraction: float
    observation_noise: float

    def __post_init__(self) -> None:
        anchor = np.asarray(self.anchor_points, dtype=np.float64)
        target = np.asarray(self.target_points, dtype=np.float64)
        reference = np.asarray(self.reference_points, dtype=np.float64)
        for name, values in (
            ("anchor_points", anchor),
            ("target_points", target),
            ("reference_points", reference),
        ):
            if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 16:
                raise ValueError(f"{name} must have shape (n, 3) with n >= 16")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite coordinates")
        points = np.vstack((anchor, target))
        if np.unique(points, axis=0).shape[0] != points.shape[0]:
            raise ValueError("combined observation points must be unique")
        view_labels = np.asarray(self.point_view_labels, dtype=np.int64)
        component_labels = np.asarray(self.point_component_labels, dtype=np.int64)
        if view_labels.shape != (points.shape[0],):
            raise ValueError("point_view_labels must match the combined point count")
        if component_labels.shape != (points.shape[0],):
            raise ValueError(
                "point_component_labels must match the combined point count"
            )
        if not np.array_equal(
            view_labels,
            np.concatenate(
                (
                    np.zeros(anchor.shape[0], dtype=np.int64),
                    np.ones(target.shape[0], dtype=np.int64),
                )
            ),
        ):
            raise ValueError("point_view_labels must identify anchor then target")
        expected_betti = tuple(int(value) for value in self.expected_surface_betti)
        if expected_betti != _EXPECTED_BETTI[self.family]:
            raise ValueError("expected topology does not match the analytic family")
        if np.any(component_labels < 0) or np.any(
            component_labels >= expected_betti[0]
        ):
            raise ValueError("component labels are outside the expected range")
        if (
            not math.isfinite(self.characteristic_length)
            or self.characteristic_length <= 0
        ):
            raise ValueError("characteristic_length must be finite and positive")
        object.__setattr__(self, "anchor_points", np.ascontiguousarray(anchor))
        object.__setattr__(self, "target_points", np.ascontiguousarray(target))
        object.__setattr__(self, "reference_points", np.ascontiguousarray(reference))
        object.__setattr__(self, "point_view_labels", np.ascontiguousarray(view_labels))
        object.__setattr__(
            self, "point_component_labels", np.ascontiguousarray(component_labels)
        )
        object.__setattr__(self, "expected_surface_betti", expected_betti)

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
class ConfidenceAlphaPanelProtocol:
    artifact_schema: str
    role: str
    families: tuple[str, ...]
    profiles: tuple[dict[str, float | str], ...]
    points_per_view: int
    reference_point_count: int
    calibration_seeds: tuple[int, ...]
    held_out_seeds: tuple[int, ...]
    calibration_case_count: int
    held_out_case_count: int
    scale_quantiles: tuple[float, ...]
    continuous_strengths: tuple[float, ...]
    binary_confidence_thresholds: tuple[float, ...]
    surface_sample_count: int
    fscore_threshold_fraction: float
    calibration_objective: str
    frozen_comparators: tuple[str, ...]
    validation_gate: str
    stability_endpoint: str
    reference_boundary: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "families",
            "profiles",
            "calibration_seeds",
            "held_out_seeds",
            "scale_quantiles",
            "continuous_strengths",
            "binary_confidence_thresholds",
            "frozen_comparators",
        ):
            payload[key] = list(payload[key])
        return payload


def _unit_sphere(
    count: int, rng: np.random.Generator, *, radius: float = 1.0
) -> FloatArray:
    z = rng.uniform(-1.0, 1.0, size=count)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return radius * np.column_stack(
        (radial * np.cos(angle), radial * np.sin(angle), z)
    )


def _surface_points(
    family: ReferenceSurfaceFamily,
    count: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray]:
    if family is ReferenceSurfaceFamily.SPHERE:
        return _unit_sphere(count, rng), np.zeros(count, dtype=np.int64)
    if family is ReferenceSurfaceFamily.TORUS:
        major_angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
        minor_angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
        minor_radius = 0.35
        radial = 1.0 + minor_radius * np.cos(minor_angle)
        points = np.column_stack(
            (
                radial * np.cos(major_angle),
                radial * np.sin(major_angle),
                minor_radius * np.sin(minor_angle),
            )
        )
        return points, np.zeros(count, dtype=np.int64)

    left_count = count // 2
    right_count = count - left_count
    left = _unit_sphere(left_count, rng, radius=0.65)
    right = _unit_sphere(right_count, rng, radius=0.65)
    left[:, 0] -= 1.2
    right[:, 0] += 1.2
    return (
        np.vstack((left, right)),
        np.concatenate(
            (
                np.zeros(left_count, dtype=np.int64),
                np.ones(right_count, dtype=np.int64),
            )
        ),
    )


def _rotation_matrix(axis: FloatArray, angle_degrees: float) -> FloatArray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = np.deg2rad(angle_degrees)
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def make_confidence_alpha_case(
    family: ReferenceSurfaceFamily | str,
    profile: MisregistrationProfile | str,
    *,
    split: PanelSplit | str,
    seed: int,
    points_per_view: int = POINTS_PER_VIEW,
    reference_point_count: int = REFERENCE_POINT_COUNT,
) -> ConfidenceAlphaCase:
    """Create one deterministic case without using truth in observed features."""

    selected_family = ReferenceSurfaceFamily(family)
    selected_profile = MisregistrationProfile(profile)
    selected_split = PanelSplit(split)
    if selected_split not in (PanelSplit.CALIBRATION, PanelSplit.HELD_OUT):
        raise ValueError("Phase-43 panel permits calibration or held_out only")
    if points_per_view < 16:
        raise ValueError("points_per_view must be at least 16")
    if reference_point_count < points_per_view:
        raise ValueError("reference_point_count must be at least points_per_view")

    anchor_rng = np.random.default_rng(seed)
    target_rng = np.random.default_rng(seed + 1_000_003)
    reference_rng = np.random.default_rng(seed + 2_000_003)
    transform_rng = np.random.default_rng(seed + 3_000_003)
    anchor, anchor_labels = _surface_points(
        selected_family, points_per_view, anchor_rng
    )
    target, target_labels = _surface_points(
        selected_family, points_per_view, target_rng
    )
    reference, _ = _surface_points(
        selected_family, reference_point_count, reference_rng
    )
    characteristic_length = float(np.linalg.norm(np.ptp(reference, axis=0)))
    rotation_degrees, translation_fraction, observation_noise = (
        _PROFILE_PARAMETERS[selected_profile]
    )
    anchor += anchor_rng.normal(scale=observation_noise, size=anchor.shape)
    target += target_rng.normal(scale=observation_noise, size=target.shape)
    axis = transform_rng.normal(size=3)
    direction = transform_rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    target = target @ _rotation_matrix(axis, rotation_degrees).T
    target += translation_fraction * characteristic_length * direction

    return ConfidenceAlphaCase(
        family=selected_family,
        profile=selected_profile,
        split=selected_split,
        seed=int(seed),
        anchor_points=anchor,
        target_points=target,
        reference_points=reference,
        point_view_labels=np.concatenate(
            (
                np.zeros(points_per_view, dtype=np.int64),
                np.ones(points_per_view, dtype=np.int64),
            )
        ),
        point_component_labels=np.concatenate((anchor_labels, target_labels)),
        expected_surface_betti=_EXPECTED_BETTI[selected_family],
        characteristic_length=characteristic_length,
        rotation_degrees=rotation_degrees,
        translation_fraction=translation_fraction,
        observation_noise=observation_noise,
    )


def make_confidence_alpha_panel(
    split: PanelSplit | str,
) -> tuple[ConfidenceAlphaCase, ...]:
    selected_split = PanelSplit(split)
    seeds = (
        CALIBRATION_SEEDS
        if selected_split is PanelSplit.CALIBRATION
        else HELD_OUT_SEEDS
    )
    if selected_split not in (PanelSplit.CALIBRATION, PanelSplit.HELD_OUT):
        raise ValueError("Phase-43 panel permits calibration or held_out only")
    return tuple(
        make_confidence_alpha_case(
            family,
            profile,
            split=selected_split,
            seed=seed,
        )
        for family in ReferenceSurfaceFamily
        for profile in MisregistrationProfile
        for seed in seeds
    )


def preregister_confidence_alpha_panel() -> ConfidenceAlphaPanelProtocol:
    profile_rows = tuple(
        {
            "profile": profile.value,
            "rotation_degrees": _PROFILE_PARAMETERS[profile][0],
            "translation_fraction": _PROFILE_PARAMETERS[profile][1],
            "observation_noise": _PROFILE_PARAMETERS[profile][2],
        }
        for profile in MisregistrationProfile
    )
    block_count = len(ReferenceSurfaceFamily) * len(MisregistrationProfile)
    return ConfidenceAlphaPanelProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="pre_method_analytic_multiview_protocol",
        families=tuple(family.value for family in ReferenceSurfaceFamily),
        profiles=profile_rows,
        points_per_view=POINTS_PER_VIEW,
        reference_point_count=REFERENCE_POINT_COUNT,
        calibration_seeds=CALIBRATION_SEEDS,
        held_out_seeds=HELD_OUT_SEEDS,
        calibration_case_count=block_count * len(CALIBRATION_SEEDS),
        held_out_case_count=block_count * len(HELD_OUT_SEEDS),
        scale_quantiles=SCALE_QUANTILES,
        continuous_strengths=CONTINUOUS_STRENGTHS,
        binary_confidence_thresholds=BINARY_CONFIDENCE_THRESHOLDS,
        surface_sample_count=SURFACE_SAMPLE_COUNT,
        fscore_threshold_fraction=FSCORE_THRESHOLD_FRACTION,
        calibration_objective=(
            "minimize calibration mean normalized_chamfer_squared + "
            "normalized_hausdorff + 0.05 * Betti_L1_error; ties prefer lower "
            "complexity parameter then lower score threshold"
        ),
        frozen_comparators=(
            "anchor_density_B4",
            "fused_density_B4",
            "fused_pca_B5",
            "binary_confidence_deletion",
        ),
        validation_gate=(
            "continuous confidence weighting must have lower held-out mean "
            "geometry loss than fused B4 and binary deletion, no larger mean "
            "Betti error than either, and objective repeat standard deviation "
            "no larger than the better of those two; B5 remains a mandatory "
            "reported novelty baseline"
        ),
        stability_endpoint=(
            "standard deviation of per-case objective over three held-out seeds "
            "within each family/profile block, averaged over blocks"
        ),
        reference_boundary=(
            "analytic reference points, family labels, perturbation values, and "
            "expected Betti numbers are evaluation-only and may not enter point "
            "confidence or filtration scores"
        ),
        claim_boundary=(
            "a positive result supports only a bounded confidence-weighted "
            "adaptive filtration; it is not a classical spatially varying alpha "
            "complex and does not imply that PFTF predicts one global alpha"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_confidence_alpha_panel().to_dict(),
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
        default=Path("benchmark-out/confidence_alpha_panel_protocol_phase43.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_protocol(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
