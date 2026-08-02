"""Frozen Phase-5 density/noise/shape transfer test for the curvature guard."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .curvature_guard import (
    CurvatureGuardConfig,
    CurvatureGuardEvidence,
    estimate_curvature_guard,
    route_with_curvature_guard,
)
from .reacquisition import ReacquisitionConfig
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .surface import evaluate_surface
from .synthetic import PanelSplit, SyntheticCase, SyntheticFamily
from .two_layer_connectivity import construct_two_layer_surface

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ShiftGeometry(StrEnum):
    PARABOLOID_024 = "paraboloid_024"
    PARABOLOID_036 = "paraboloid_036"
    ASYMMETRIC_CONVERGING = "asymmetric_converging"
    SADDLE_024 = "saddle_024"
    CYLINDER_036 = "cylinder_036"


DEFAULT_GEOMETRIES = tuple(ShiftGeometry)


@dataclass(frozen=True)
class SamplingProfile:
    point_count: int
    noise: float

    @property
    def name(self) -> str:
        return f"n{self.point_count}_noise{self.noise:.3f}"

    def __post_init__(self) -> None:
        if self.point_count < 16:
            raise ValueError("point_count must be at least 16")
        if not math.isfinite(self.noise) or self.noise < 0.0:
            raise ValueError("noise must be finite and non-negative")


DEFAULT_PROFILES = tuple(
    SamplingProfile(point_count=point_count, noise=noise)
    for point_count in (96, 160, 256)
    for noise in (0.005, 0.010, 0.025)
)


@dataclass(frozen=True)
class ShiftCaseResult:
    profile: str
    point_count: int
    noise: float
    geometry: ShiftGeometry
    repeat: int
    seed: int
    evidence: CurvatureGuardEvidence
    base_decision: SamplingGateDecision
    guarded_decision: SamplingGateDecision
    true_safe_output: bool
    base_safe_accept: bool
    guarded_safe_accept: bool
    base_false_safe: bool
    guarded_false_safe: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["geometry"] = self.geometry.value
        payload["base_decision"] = self.base_decision.value
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class ShiftGroupSummary:
    group_kind: str
    group_name: str
    case_count: int
    mean_normal_coherence: float
    base_accept_count: int
    guarded_accept_count: int
    base_safe_accept_count: int
    guarded_safe_accept_count: int
    safe_accept_retention: float | None
    base_false_safe_count: int
    guarded_false_safe_count: int
    group_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GuardDomainShiftResult:
    artifact_schema: str
    role: str
    information_boundary: str
    seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    profiles: tuple[SamplingProfile, ...]
    geometries: tuple[ShiftGeometry, ...]
    sampling_gate_config: SamplingSufficiencyConfig
    guard_config: CurvatureGuardConfig
    cases: tuple[ShiftCaseResult, ...]
    profile_summaries: tuple[ShiftGroupSummary, ...]
    geometry_summaries: tuple[ShiftGroupSummary, ...]
    case_count: int
    base_safe_accept_count: int
    guarded_safe_accept_count: int
    safe_accept_retention: float
    base_false_safe_count: int
    guarded_false_safe_count: int
    removed_false_safe_count: int
    phase5_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "seed": self.seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "profiles": [
                asdict(profile) | {"name": profile.name}
                for profile in self.profiles
            ],
            "geometries": [geometry.value for geometry in self.geometries],
            "sampling_gate_config": asdict(self.sampling_gate_config),
            "guard_config": asdict(self.guard_config),
            "cases": [case.to_dict() for case in self.cases],
            "profile_summaries": [row.to_dict() for row in self.profile_summaries],
            "geometry_summaries": [row.to_dict() for row in self.geometry_summaries],
            "case_count": self.case_count,
            "base_safe_accept_count": self.base_safe_accept_count,
            "guarded_safe_accept_count": self.guarded_safe_accept_count,
            "safe_accept_retention": self.safe_accept_retention,
            "base_false_safe_count": self.base_false_safe_count,
            "guarded_false_safe_count": self.guarded_false_safe_count,
            "removed_false_safe_count": self.removed_false_safe_count,
            "phase5_supported": self.phase5_supported,
            "deployment_supported": self.deployment_supported,
        }


def _height(
    geometry: ShiftGeometry,
    xy: FloatArray,
    layer: int,
) -> FloatArray:
    x, y = xy[:, 0], xy[:, 1]
    radius_squared = x * x + y * y
    if geometry is ShiftGeometry.PARABOLOID_024:
        middle = 0.24 * radius_squared
    elif geometry is ShiftGeometry.PARABOLOID_036:
        middle = 0.36 * radius_squared
    elif geometry is ShiftGeometry.SADDLE_024:
        middle = 0.24 * (x * x - y * y)
    elif geometry is ShiftGeometry.CYLINDER_036:
        middle = 0.36 * x * x
    else:
        curvature = 0.36 if layer == 0 else 0.12
        middle = curvature * radius_squared
    offset = -0.40 if layer == 0 else 0.40
    return middle + offset


def _characteristic_length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.ptp(points, axis=0)))


def make_shift_case(
    geometry: ShiftGeometry | str,
    profile: SamplingProfile,
    *,
    reference_count: int = 2048,
    seed: int = 0,
) -> SyntheticCase:
    """Generate one balanced domain-shift case with evaluation-only labels."""

    selected = ShiftGeometry(geometry)
    if reference_count < profile.point_count:
        raise ValueError("reference_count must be at least point_count")
    observed_rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed + 1_000_003)

    def sample(
        count: int,
        rng: np.random.Generator,
        *,
        add_noise: bool,
    ) -> tuple[FloatArray, IntArray]:
        counts = (count // 2, count - count // 2)
        point_rows = []
        label_rows = []
        for layer, layer_count in enumerate(counts):
            xy = rng.uniform(-1.0, 1.0, size=(layer_count, 2))
            z = _height(selected, xy, layer)
            point_rows.append(np.column_stack((xy, z)))
            label_rows.append(np.full(layer_count, layer, dtype=np.int64))
        points = np.vstack(point_rows)
        if add_noise and profile.noise > 0.0:
            points = points + rng.normal(scale=profile.noise, size=points.shape)
        return points, np.concatenate(label_rows)

    observed, labels = sample(profile.point_count, observed_rng, add_noise=True)
    reference, _ = sample(reference_count, reference_rng, add_noise=False)
    sheet_gap = (
        0.32 if selected is ShiftGeometry.ASYMMETRIC_CONVERGING else 0.80
    )
    return SyntheticCase(
        family=SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.HELD_OUT,
        points=observed,
        reference_points=reference,
        expected_components=2,
        characteristic_length=_characteristic_length(reference),
        variation={"sheet_gap": sheet_gap, "noise": profile.noise},
        seed=seed,
        expected_surface_betti=(2, 0, 0),
        point_component_labels=labels,
    )


def _group_summary(
    rows: Sequence[ShiftCaseResult],
    *,
    kind: str,
    name: str,
) -> ShiftGroupSummary:
    if not rows:
        raise RuntimeError("every domain-shift group must have cases")
    base_safe = sum(case.base_safe_accept for case in rows)
    guarded_safe = sum(case.guarded_safe_accept for case in rows)
    retention = None if not base_safe else guarded_safe / base_safe
    guarded_false = sum(case.guarded_false_safe for case in rows)
    passed = bool(
        guarded_false == 0
        and (base_safe < 4 or (retention is not None and retention >= 0.75))
    )
    return ShiftGroupSummary(
        group_kind=kind,
        group_name=name,
        case_count=len(rows),
        mean_normal_coherence=float(
            np.mean([case.evidence.normal_coherence for case in rows])
        ),
        base_accept_count=sum(
            case.base_decision is SamplingGateDecision.ACCEPT for case in rows
        ),
        guarded_accept_count=sum(
            case.guarded_decision is SamplingGateDecision.ACCEPT for case in rows
        ),
        base_safe_accept_count=base_safe,
        guarded_safe_accept_count=guarded_safe,
        safe_accept_retention=retention,
        base_false_safe_count=sum(case.base_false_safe for case in rows),
        guarded_false_safe_count=guarded_false,
        group_gate_passed=passed,
    )


def evaluate_guard_domain_shift(
    *,
    profiles: Sequence[SamplingProfile] = DEFAULT_PROFILES,
    geometries: Sequence[ShiftGeometry | str] = DEFAULT_GEOMETRIES,
    reference_count: int = 2048,
    repeats: int = 8,
    seed: int = 20300804,
    surface_sample_count: int = 256,
    sampling_gate_config: SamplingSufficiencyConfig | None = None,
    guard_config: CurvatureGuardConfig | None = None,
) -> GuardDomainShiftResult:
    """Run the frozen Phase-5 transfer panel without threshold tuning."""

    selected_profiles = tuple(profiles)
    selected_geometries = tuple(ShiftGeometry(value) for value in geometries)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not selected_profiles or not selected_geometries:
        raise ValueError("profiles and geometries must be non-empty")
    if len({profile.name for profile in selected_profiles}) != len(selected_profiles):
        raise ValueError("profile names must be unique")
    selected_sampling = sampling_gate_config or SamplingSufficiencyConfig(
        minimum_separation_snr=3.0
    )
    selected_guard = guard_config or CurvatureGuardConfig(
        k_neighbors=selected_sampling.k_neighbors
    )
    reconstruction_config = ReacquisitionConfig(
        base_point_count=max(profile.point_count for profile in selected_profiles),
        evaluation_reference_count=reference_count,
        candidate_pool_count=reference_count,
        added_point_counts=(1,),
        repeats=1,
        seed=seed,
        surface_sample_count=surface_sample_count,
        k_neighbors=selected_sampling.k_neighbors,
    )
    results: list[ShiftCaseResult] = []
    for profile_index, profile in enumerate(selected_profiles):
        for geometry_index, geometry in enumerate(selected_geometries):
            for repeat in range(repeats):
                case_seed = (
                    seed
                    + profile_index * 1_000_003
                    + geometry_index * 100_003
                    + repeat * 10_007
                )
                case = make_shift_case(
                    geometry,
                    profile,
                    reference_count=reference_count,
                    seed=case_seed,
                )
                construction = construct_two_layer_surface(
                    case.points,
                    selected_sampling,
                )
                constrained = evaluate_surface(
                    construction.mesh,
                    case.reference_points,
                    expected_components=2,
                    expected_betti=(2, 0, 0),
                    vertex_component_labels=case.point_component_labels,
                    characteristic_length=case.characteristic_length,
                    sample_count=surface_sample_count,
                    threshold_fraction=(
                        reconstruction_config.fscore_threshold_fraction
                    ),
                    seed=case_seed + 41,
                )
                inferred = evaluate_surface(
                    construction.mesh,
                    case.reference_points,
                    expected_components=2,
                    expected_betti=(2, 0, 0),
                    vertex_component_labels=construction.inference.layer_ids,
                    characteristic_length=case.characteristic_length,
                    sample_count=surface_sample_count,
                    threshold_fraction=(
                        reconstruction_config.fscore_threshold_fraction
                    ),
                    seed=case_seed + 41,
                )
                evidence = estimate_curvature_guard(case.points, selected_guard)
                base, guarded = route_with_curvature_guard(
                    construction,
                    inferred,
                    evidence,
                )
                true_safe = bool(
                    constrained.component_error == 0
                    and int(constrained.labeled_false_bridge_edges or 0) == 0
                    and int(constrained.labeled_false_bridge_faces or 0) == 0
                )
                base_accept = base is SamplingGateDecision.ACCEPT
                guarded_accept = guarded is SamplingGateDecision.ACCEPT
                results.append(
                    ShiftCaseResult(
                        profile=profile.name,
                        point_count=profile.point_count,
                        noise=profile.noise,
                        geometry=geometry,
                        repeat=repeat,
                        seed=case_seed,
                        evidence=evidence,
                        base_decision=base,
                        guarded_decision=guarded,
                        true_safe_output=true_safe,
                        base_safe_accept=bool(base_accept and true_safe),
                        guarded_safe_accept=bool(guarded_accept and true_safe),
                        base_false_safe=bool(base_accept and not true_safe),
                        guarded_false_safe=bool(guarded_accept and not true_safe),
                    )
                )

    profile_summaries = tuple(
        _group_summary(
            [case for case in results if case.profile == profile.name],
            kind="profile",
            name=profile.name,
        )
        for profile in selected_profiles
    )
    geometry_summaries = tuple(
        _group_summary(
            [case for case in results if case.geometry is geometry],
            kind="geometry",
            name=geometry.value,
        )
        for geometry in selected_geometries
    )
    base_safe = sum(case.base_safe_accept for case in results)
    guarded_safe = sum(case.guarded_safe_accept for case in results)
    base_false = sum(case.base_false_safe for case in results)
    guarded_false = sum(case.guarded_false_safe for case in results)
    retention = 0.0 if not base_safe else guarded_safe / base_safe
    supported = bool(
        selected_profiles == DEFAULT_PROFILES
        and selected_geometries == DEFAULT_GEOMETRIES
        and repeats >= 8
        and len(results) == 360
        and base_false > 0
        and guarded_false == 0
        and base_false - guarded_false == base_false
        and retention >= 0.90
        and all(row.group_gate_passed for row in profile_summaries)
    )
    return GuardDomainShiftResult(
        artifact_schema="pftf_alpha_curvature_guard_domain_shift_phase5/v1",
        role="frozen_density_noise_shape_transfer_test",
        information_boundary=(
            "construction and guard use observed coordinates only; profile, "
            "geometry, true labels, and references are evaluation-only"
        ),
        seed=seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        profiles=selected_profiles,
        geometries=selected_geometries,
        sampling_gate_config=selected_sampling,
        guard_config=selected_guard,
        cases=tuple(results),
        profile_summaries=profile_summaries,
        geometry_summaries=geometry_summaries,
        case_count=len(results),
        base_safe_accept_count=base_safe,
        guarded_safe_accept_count=guarded_safe,
        safe_accept_retention=retention,
        base_false_safe_count=base_false,
        guarded_false_safe_count=guarded_false,
        removed_false_safe_count=base_false - guarded_false,
        phase5_supported=supported,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20300804)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_guard_domain_shift(
        reference_count=args.reference,
        repeats=args.repeats,
        seed=args.seed,
        surface_sample_count=args.surface_samples,
    )
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
