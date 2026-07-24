"""Deterministic 3D synthetic cases from the first benchmark plan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SyntheticFamily(StrEnum):
    U_CONCAVITY = "u_concavity"
    OPPOSING_SHEETS = "opposing_sheets"
    TORUS = "torus"
    DISCONNECTED_PARTS = "disconnected_parts"
    SHARP_CREASE = "sharp_crease"
    MISSING_PATCH = "missing_patch"


_EXPECTED_SURFACE_BETTI: dict[SyntheticFamily, tuple[int, int, int]] = {
    SyntheticFamily.U_CONCAVITY: (1, 1, 0),
    SyntheticFamily.OPPOSING_SHEETS: (2, 0, 0),
    SyntheticFamily.TORUS: (1, 2, 1),
    SyntheticFamily.DISCONNECTED_PARTS: (2, 0, 2),
    SyntheticFamily.SHARP_CREASE: (1, 0, 0),
    SyntheticFamily.MISSING_PATCH: (1, 0, 1),
}


class PanelSplit(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    HELD_OUT = "held_out"


@dataclass(frozen=True)
class SyntheticCase:
    """Observed points plus a dense reference used only for evaluation."""

    family: SyntheticFamily
    split: PanelSplit
    points: FloatArray
    reference_points: FloatArray
    expected_components: int
    characteristic_length: float
    variation: Mapping[str, float]
    seed: int
    expected_surface_betti: tuple[int, int, int]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        reference = np.asarray(self.reference_points, dtype=np.float64)
        for name, values in (("points", points), ("reference_points", reference)):
            if values.ndim != 2 or values.shape[1] != 3:
                raise ValueError(f"{name} must have shape (n, 3)")
            if values.shape[0] < 4 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain at least four finite points")
        if np.unique(points, axis=0).shape[0] != points.shape[0]:
            raise ValueError("synthetic observed points must be unique")
        if self.expected_components < 1:
            raise ValueError("expected_components must be positive")
        raw_expected_betti = tuple(self.expected_surface_betti)
        if len(raw_expected_betti) != 3 or any(
            not isinstance(value, (int, np.integer)) or value < 0
            for value in raw_expected_betti
        ):
            raise ValueError(
                "expected_surface_betti must contain three non-negative integers"
            )
        expected_betti = tuple(int(value) for value in raw_expected_betti)
        if expected_betti[0] != self.expected_components:
            raise ValueError("expected_surface_betti[0] must equal expected_components")
        if (
            not math.isfinite(self.characteristic_length)
            or self.characteristic_length <= 0.0
        ):
            raise ValueError("characteristic_length must be finite and positive")
        object.__setattr__(self, "expected_surface_betti", expected_betti)
        object.__setattr__(self, "points", np.ascontiguousarray(points))
        object.__setattr__(self, "reference_points", np.ascontiguousarray(reference))


_SPLIT_VARIATIONS: dict[PanelSplit, dict[str, float]] = {
    PanelSplit.TRAIN: {
        "opening_width": 0.9,
        "sheet_gap": 0.34,
        "torus_minor_radius": 0.38,
        "part_separation": 3.2,
        "crease_angle_degrees": 80.0,
        "cap_height": 0.60,
        "noise": 0.002,
    },
    PanelSplit.CALIBRATION: {
        "opening_width": 0.75,
        "sheet_gap": 0.26,
        "torus_minor_radius": 0.33,
        "part_separation": 2.9,
        "crease_angle_degrees": 60.0,
        "cap_height": 0.40,
        "noise": 0.004,
    },
    PanelSplit.HELD_OUT: {
        "opening_width": 0.55,
        "sheet_gap": 0.18,
        "torus_minor_radius": 0.26,
        "part_separation": 2.5,
        "crease_angle_degrees": 38.0,
        "cap_height": 0.15,
        "noise": 0.010,
    },
}


def _sphere_points(
    count: int,
    rng: np.random.Generator,
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
) -> FloatArray:
    z = rng.uniform(-1.0, 1.0, size=count)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    unit = np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))
    return radius * unit + np.asarray(center, dtype=np.float64)


def _torus_points(
    count: int,
    rng: np.random.Generator,
    *,
    major_radius: float,
    minor_radius: float,
) -> FloatArray:
    major_angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
    minor_angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radial = major_radius + minor_radius * np.cos(minor_angle)
    return np.column_stack(
        (
            radial * np.cos(major_angle),
            radial * np.sin(major_angle),
            minor_radius * np.sin(minor_angle),
        )
    )


def _opposing_sheet_points(
    count: int, rng: np.random.Generator, *, gap: float
) -> FloatArray:
    sheet = rng.integers(0, 2, size=count)
    xy = rng.uniform(-1.0, 1.0, size=(count, 2))
    z = np.where(sheet == 0, -0.5 * gap, 0.5 * gap)
    return np.column_stack((xy, z))


def _crease_points(
    count: int, rng: np.random.Generator, *, angle_degrees: float
) -> FloatArray:
    side = rng.choice(np.array([-1.0, 1.0]), size=count)
    along_crease = rng.uniform(-1.0, 1.0, size=count)
    away = rng.uniform(0.0, 1.0, size=count)
    half_angle = np.deg2rad(angle_degrees) * 0.5
    return np.column_stack(
        (
            along_crease,
            away * np.cos(half_angle),
            side * away * np.sin(half_angle),
        )
    )


def _u_side_points(
    count: int, rng: np.random.Generator, *, opening_width: float
) -> FloatArray:
    half_opening = 0.5 * opening_width
    polygon = np.asarray(
        [
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [half_opening, 1.0],
            [half_opening, -0.35],
            [-half_opening, -0.35],
            [-half_opening, 1.0],
        ],
        dtype=np.float64,
    )
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    lengths = np.linalg.norm(ends - starts, axis=1)
    edge_indices = rng.choice(len(starts), size=count, p=lengths / np.sum(lengths))
    fraction = rng.uniform(0.0, 1.0, size=(count, 1))
    xy = starts[edge_indices] + fraction * (ends[edge_indices] - starts[edge_indices])
    z = rng.uniform(-0.5, 0.5, size=count)
    return np.column_stack((xy, z))


def _add_observation_noise(
    points: FloatArray, rng: np.random.Generator, noise: float
) -> FloatArray:
    if noise == 0.0:
        return points.copy()
    return points + rng.normal(scale=noise, size=points.shape)


def _characteristic_length(reference_points: FloatArray) -> float:
    extent = np.ptp(reference_points, axis=0)
    return float(np.linalg.norm(extent))


def make_synthetic_case(
    family: SyntheticFamily | str,
    *,
    split: PanelSplit | str = PanelSplit.CALIBRATION,
    point_count: int = 96,
    reference_count: int = 2048,
    seed: int = 0,
) -> SyntheticCase:
    """Generate one deterministic observed/reference pair."""

    selected_family = SyntheticFamily(family)
    selected_split = PanelSplit(split)
    if point_count < 16:
        raise ValueError("point_count must be at least 16")
    if reference_count < point_count:
        raise ValueError("reference_count must be at least point_count")

    variation = dict(_SPLIT_VARIATIONS[selected_split])
    observed_rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed + 1_000_003)
    expected_surface_betti = _EXPECTED_SURFACE_BETTI[selected_family]
    expected_components = expected_surface_betti[0]

    if selected_family is SyntheticFamily.U_CONCAVITY:
        key = "opening_width"
        observed = _u_side_points(
            point_count, observed_rng, opening_width=variation[key]
        )
        reference = _u_side_points(
            reference_count, reference_rng, opening_width=variation[key]
        )
    elif selected_family is SyntheticFamily.OPPOSING_SHEETS:
        key = "sheet_gap"
        observed = _opposing_sheet_points(point_count, observed_rng, gap=variation[key])
        reference = _opposing_sheet_points(
            reference_count, reference_rng, gap=variation[key]
        )
    elif selected_family is SyntheticFamily.TORUS:
        key = "torus_minor_radius"
        observed = _torus_points(
            point_count,
            observed_rng,
            major_radius=1.0,
            minor_radius=variation[key],
        )
        reference = _torus_points(
            reference_count,
            reference_rng,
            major_radius=1.0,
            minor_radius=variation[key],
        )
    elif selected_family is SyntheticFamily.DISCONNECTED_PARTS:
        key = "part_separation"
        observed_count_left = point_count // 2
        reference_count_left = reference_count // 2
        offset = 0.5 * variation[key]
        observed = np.vstack(
            (
                _sphere_points(
                    observed_count_left,
                    observed_rng,
                    center=(-offset, 0.0, 0.0),
                    radius=0.7,
                ),
                _sphere_points(
                    point_count - observed_count_left,
                    observed_rng,
                    center=(offset, 0.0, 0.0),
                    radius=0.7,
                ),
            )
        )
        reference = np.vstack(
            (
                _sphere_points(
                    reference_count_left,
                    reference_rng,
                    center=(-offset, 0.0, 0.0),
                    radius=0.7,
                ),
                _sphere_points(
                    reference_count - reference_count_left,
                    reference_rng,
                    center=(offset, 0.0, 0.0),
                    radius=0.7,
                ),
            )
        )
    elif selected_family is SyntheticFamily.SHARP_CREASE:
        key = "crease_angle_degrees"
        observed = _crease_points(
            point_count, observed_rng, angle_degrees=variation[key]
        )
        reference = _crease_points(
            reference_count, reference_rng, angle_degrees=variation[key]
        )
    else:
        key = "cap_height"
        candidates = _sphere_points(point_count * 4, observed_rng)
        observed = candidates[candidates[:, 2] <= variation[key]][:point_count]
        if observed.shape[0] < point_count:
            raise RuntimeError("could not sample enough points outside missing cap")
        reference = _sphere_points(reference_count, reference_rng)

    observed = _add_observation_noise(observed, observed_rng, variation["noise"])
    relevant_variation = {
        key: variation[key],
        "noise": variation["noise"],
    }
    return SyntheticCase(
        family=selected_family,
        split=selected_split,
        points=observed,
        reference_points=reference,
        expected_components=expected_components,
        characteristic_length=_characteristic_length(reference),
        variation=relevant_variation,
        seed=seed,
        expected_surface_betti=expected_surface_betti,
    )


def make_minimal_panel(
    *,
    split: PanelSplit | str = PanelSplit.CALIBRATION,
    point_count: int = 96,
    reference_count: int = 2048,
    seed: int = 0,
) -> tuple[SyntheticCase, ...]:
    """Generate all six required synthetic families with frozen derived seeds."""

    return tuple(
        make_synthetic_case(
            family,
            split=split,
            point_count=point_count,
            reference_count=reference_count,
            seed=seed + 10_007 * index,
        )
        for index, family in enumerate(SyntheticFamily)
    )
