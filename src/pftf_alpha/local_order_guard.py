"""Density-normalized local layer-order guard and Phase-6 held-out test."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .adaptive import local_neighborhood_geometry
from .guard_domain_shift import (
    DEFAULT_GEOMETRIES,
    DEFAULT_PROFILES,
    SamplingProfile,
    ShiftGeometry,
    make_shift_case,
)
from .reacquisition import ReacquisitionConfig
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .surface import SurfaceEndpointMetrics, evaluate_surface
from .two_layer_connectivity import (
    TwoLayerConstruction,
    construct_two_layer_surface,
    route_two_layer_output,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DensityBinThreshold:
    maximum_point_count: int | None
    minimum_normal_coherence: float
    minimum_local_order_margin: float

    def __post_init__(self) -> None:
        if self.maximum_point_count is not None and self.maximum_point_count < 16:
            raise ValueError("maximum_point_count must be at least 16")
        for name, value in (
            ("minimum_normal_coherence", self.minimum_normal_coherence),
            ("minimum_local_order_margin", self.minimum_local_order_margin),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


DEFAULT_DENSITY_BINS = (
    DensityBinThreshold(96, 0.75, 0.150),
    DensityBinThreshold(160, 0.80, 0.195),
    DensityBinThreshold(None, 0.80, 0.220),
)


@dataclass(frozen=True)
class LocalOrderGuardConfig:
    k_neighbors: int = 12
    margin_quantile: float = 0.05
    density_bins: tuple[DensityBinThreshold, ...] = DEFAULT_DENSITY_BINS

    def __post_init__(self) -> None:
        if self.k_neighbors < 3:
            raise ValueError("k_neighbors must be at least three")
        if not 0.0 < self.margin_quantile < 0.5:
            raise ValueError("margin_quantile must lie in (0, 0.5)")
        if (
            not self.density_bins
            or self.density_bins[-1].maximum_point_count is not None
        ):
            raise ValueError("density_bins must end with an open upper bin")
        finite = [
            row.maximum_point_count
            for row in self.density_bins
            if row.maximum_point_count is not None
        ]
        if finite != sorted(finite) or len(finite) != len(set(finite)):
            raise ValueError("finite density-bin limits must be unique and increasing")


@dataclass(frozen=True)
class LocalOrderEvidence:
    information_boundary: str
    point_count: int
    k_neighbors: int
    normal_coherence: float
    local_order_margin: float
    minimum_normal_coherence: float
    minimum_local_order_margin: float
    model_adequate: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalOrderCaseResult:
    profile: str
    point_count: int
    noise: float
    geometry: ShiftGeometry
    repeat: int
    seed: int
    evidence: LocalOrderEvidence
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
class DensitySummary:
    point_count: int
    case_count: int
    base_safe_accept_count: int
    guarded_safe_accept_count: int
    safe_accept_retention: float | None
    base_false_safe_count: int
    guarded_false_safe_count: int
    density_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalOrderGuardResult:
    artifact_schema: str
    role: str
    information_boundary: str
    calibration_source: str
    seed: int
    reference_count: int
    repeats: int
    profiles: tuple[SamplingProfile, ...]
    geometries: tuple[ShiftGeometry, ...]
    sampling_gate_config: SamplingSufficiencyConfig
    guard_config: LocalOrderGuardConfig
    cases: tuple[LocalOrderCaseResult, ...]
    density_summaries: tuple[DensitySummary, ...]
    case_count: int
    base_safe_accept_count: int
    guarded_safe_accept_count: int
    safe_accept_retention: float
    base_false_safe_count: int
    guarded_false_safe_count: int
    removed_false_safe_count: int
    phase6_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "calibration_source": self.calibration_source,
            "seed": self.seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "profiles": [
                asdict(profile) | {"name": profile.name}
                for profile in self.profiles
            ],
            "geometries": [geometry.value for geometry in self.geometries],
            "sampling_gate_config": asdict(self.sampling_gate_config),
            "guard_config": {
                "k_neighbors": self.guard_config.k_neighbors,
                "margin_quantile": self.guard_config.margin_quantile,
                "density_bins": [
                    asdict(row) for row in self.guard_config.density_bins
                ],
            },
            "cases": [case.to_dict() for case in self.cases],
            "density_summaries": [row.to_dict() for row in self.density_summaries],
            "case_count": self.case_count,
            "base_safe_accept_count": self.base_safe_accept_count,
            "guarded_safe_accept_count": self.guarded_safe_accept_count,
            "safe_accept_retention": self.safe_accept_retention,
            "base_false_safe_count": self.base_false_safe_count,
            "guarded_false_safe_count": self.guarded_false_safe_count,
            "removed_false_safe_count": self.removed_false_safe_count,
            "phase6_supported": self.phase6_supported,
            "deployment_supported": self.deployment_supported,
        }


def _threshold_for_count(
    point_count: int,
    config: LocalOrderGuardConfig,
) -> DensityBinThreshold:
    return next(
        row
        for row in config.density_bins
        if row.maximum_point_count is None or point_count <= row.maximum_point_count
    )


def estimate_local_order_guard(
    points: FloatArray,
    inferred_labels: IntArray,
    config: LocalOrderGuardConfig | None = None,
) -> LocalOrderEvidence:
    """Estimate density-normalized local layer-order adequacy from observations."""

    selected = LocalOrderGuardConfig() if config is None else config
    point_array = np.asarray(points, dtype=np.float64)
    labels = np.asarray(inferred_labels, dtype=np.int64)
    if labels.shape != (point_array.shape[0],) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("inferred_labels must contain two aligned layers")
    geometry = local_neighborhood_geometry(
        point_array,
        k_neighbors=selected.k_neighbors,
    )
    normals = geometry.eigenvectors[:, :, 0]
    orientation_tensor = np.mean(
        normals[:, :, None] * normals[:, None, :],
        axis=0,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(orientation_tensor)
    normal = eigenvectors[:, -1]
    tangent = eigenvectors[:, :2]
    centered = point_array - np.mean(point_array, axis=0)
    tangent_coordinates = centered @ tangent
    normal_coordinates = centered @ normal
    ratios: list[float] = []
    for layer in (0, 1):
        indices = np.flatnonzero(labels == layer)
        opposite = np.flatnonzero(labels != layer)
        within_spacing = cKDTree(tangent_coordinates[indices]).query(
            tangent_coordinates[indices],
            k=2,
            workers=1,
        )[0][:, 1]
        opposite_indices = cKDTree(tangent_coordinates[opposite]).query(
            tangent_coordinates[indices],
            k=1,
            workers=1,
        )[1]
        normal_gap = np.abs(
            normal_coordinates[indices]
            - normal_coordinates[opposite[opposite_indices]]
        )
        ratios.extend(
            (normal_gap / np.maximum(within_spacing, np.finfo(float).eps)).tolist()
        )
    margin = float(
        np.quantile(np.asarray(ratios), selected.margin_quantile)
        / math.sqrt(point_array.shape[0])
    )
    threshold = _threshold_for_count(point_array.shape[0], selected)
    coherence = float(eigenvalues[-1])
    return LocalOrderEvidence(
        information_boundary="observed_coordinates_and_inferred_layers_only",
        point_count=point_array.shape[0],
        k_neighbors=geometry.k_neighbors,
        normal_coherence=coherence,
        local_order_margin=margin,
        minimum_normal_coherence=threshold.minimum_normal_coherence,
        minimum_local_order_margin=threshold.minimum_local_order_margin,
        model_adequate=bool(
            coherence >= threshold.minimum_normal_coherence
            and margin >= threshold.minimum_local_order_margin
        ),
    )


def route_with_local_order_guard(
    construction: TwoLayerConstruction,
    inferred_endpoints: SurfaceEndpointMetrics,
    evidence: LocalOrderEvidence,
) -> tuple[SamplingGateDecision, SamplingGateDecision]:
    base = route_two_layer_output(construction, inferred_endpoints)
    guarded = base
    if base is SamplingGateDecision.ACCEPT and not evidence.model_adequate:
        guarded = SamplingGateDecision.UNSUPPORTED
    return base, guarded


def _density_summary(
    cases: Sequence[LocalOrderCaseResult],
    point_count: int,
) -> DensitySummary:
    rows = [case for case in cases if case.point_count == point_count]
    base_safe = sum(case.base_safe_accept for case in rows)
    guarded_safe = sum(case.guarded_safe_accept for case in rows)
    retention = None if not base_safe else guarded_safe / base_safe
    guarded_false = sum(case.guarded_false_safe for case in rows)
    return DensitySummary(
        point_count=point_count,
        case_count=len(rows),
        base_safe_accept_count=base_safe,
        guarded_safe_accept_count=guarded_safe,
        safe_accept_retention=retention,
        base_false_safe_count=sum(case.base_false_safe for case in rows),
        guarded_false_safe_count=guarded_false,
        density_gate_passed=bool(
            guarded_false == 0
            and (base_safe < 8 or (retention is not None and retention >= 0.75))
        ),
    )


def evaluate_local_order_guard(
    *,
    profiles: Sequence[SamplingProfile] = DEFAULT_PROFILES,
    geometries: Sequence[ShiftGeometry | str] = DEFAULT_GEOMETRIES,
    reference_count: int = 2048,
    repeats: int = 8,
    seed: int = 20400804,
    surface_sample_count: int = 256,
    sampling_gate_config: SamplingSufficiencyConfig | None = None,
    guard_config: LocalOrderGuardConfig | None = None,
) -> LocalOrderGuardResult:
    """Run the frozen Phase-6 held-out transfer validation."""

    selected_profiles = tuple(profiles)
    selected_geometries = tuple(ShiftGeometry(value) for value in geometries)
    if repeats < 1 or not selected_profiles or not selected_geometries:
        raise ValueError("profiles/geometries must be non-empty and repeats positive")
    selected_sampling = sampling_gate_config or SamplingSufficiencyConfig(
        minimum_separation_snr=3.0
    )
    selected_guard = guard_config or LocalOrderGuardConfig(
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
    results: list[LocalOrderCaseResult] = []
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
                evidence = estimate_local_order_guard(
                    case.points,
                    construction.inference.layer_ids,
                    selected_guard,
                )
                base, guarded = route_with_local_order_guard(
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
                    LocalOrderCaseResult(
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
    point_counts = sorted({profile.point_count for profile in selected_profiles})
    density_summaries = tuple(
        _density_summary(results, point_count) for point_count in point_counts
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
        and retention >= 0.85
        and all(row.density_gate_passed for row in density_summaries)
    )
    return LocalOrderGuardResult(
        artifact_schema="pftf_alpha_local_order_guard_phase6/v1",
        role="density_normalized_local_layer_order_fail_closed_guard",
        information_boundary=(
            "guard uses observed coordinates and inferred layers only; profiles, "
            "geometries, true labels, and references are evaluation-only"
        ),
        calibration_source="phase5_seed_20300804",
        seed=seed,
        reference_count=reference_count,
        repeats=repeats,
        profiles=selected_profiles,
        geometries=selected_geometries,
        sampling_gate_config=selected_sampling,
        guard_config=selected_guard,
        cases=tuple(results),
        density_summaries=density_summaries,
        case_count=len(results),
        base_safe_accept_count=base_safe,
        guarded_safe_accept_count=guarded_safe,
        safe_accept_retention=retention,
        base_false_safe_count=base_false,
        guarded_false_safe_count=guarded_false,
        removed_false_safe_count=base_false - guarded_false,
        phase6_supported=supported,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20400804)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_local_order_guard(
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
