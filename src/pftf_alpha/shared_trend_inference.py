"""Shared-trend layer inference and frozen Phase-7 held-out evaluation."""

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
from .sampling_gate import (
    ParallelLayerInference,
    SamplingGateDecision,
    SamplingSufficiencyConfig,
    SamplingSufficiencyEvidence,
)
from .surface import SurfaceEndpointMetrics, evaluate_surface
from .two_layer_connectivity import (
    TwoLayerConstruction,
    construct_two_layer_surface,
    construct_two_layer_surface_from_inference,
    route_two_layer_output,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SharedTrendConfig:
    k_neighbors: int = 12
    maximum_iterations: int = 32
    minimum_cluster_fraction: float = 0.20
    minimum_separation_snr: float = 3.0
    cross_knn_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.k_neighbors < 3:
            raise ValueError("k_neighbors must be at least three")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        for name, value in (
            ("minimum_cluster_fraction", self.minimum_cluster_fraction),
            ("minimum_separation_snr", self.minimum_separation_snr),
            ("cross_knn_threshold", self.cross_knn_threshold),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_cluster_fraction >= 0.5:
            raise ValueError("minimum_cluster_fraction must be below 0.5")
        if self.cross_knn_threshold >= 1.0:
            raise ValueError("cross_knn_threshold must be below one")


@dataclass(frozen=True)
class SharedTrendDiagnostics:
    information_boundary: str
    polynomial_terms: tuple[str, ...]
    iterations: int
    converged: bool
    residual_layer_centers: tuple[float, float]
    shared_trend_rmse: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SharedTrendInference:
    inference: ParallelLayerInference
    diagnostics: SharedTrendDiagnostics


@dataclass(frozen=True)
class SharedTrendCaseResult:
    profile: str
    point_count: int
    noise: float
    geometry: ShiftGeometry
    repeat: int
    seed: int
    diagnostics: SharedTrendDiagnostics
    base_decision: SamplingGateDecision
    candidate_decision: SamplingGateDecision
    base_true_safe_output: bool
    candidate_true_safe_output: bool
    base_safe_accept: bool
    candidate_safe_accept: bool
    base_false_safe: bool
    candidate_false_safe: bool
    repaired_base_false_safe: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["geometry"] = self.geometry.value
        payload["base_decision"] = self.base_decision.value
        payload["candidate_decision"] = self.candidate_decision.value
        return payload


@dataclass(frozen=True)
class SharedTrendDensitySummary:
    point_count: int
    case_count: int
    base_safe_accept_count: int
    candidate_retained_base_safe_count: int
    base_safe_retention: float | None
    base_false_safe_count: int
    repaired_base_false_safe_count: int
    candidate_false_safe_count: int
    density_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SharedTrendResult:
    artifact_schema: str
    role: str
    information_boundary: str
    calibration_sources: tuple[str, ...]
    seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    profiles: tuple[SamplingProfile, ...]
    geometries: tuple[ShiftGeometry, ...]
    base_gate_config: SamplingSufficiencyConfig
    shared_trend_config: SharedTrendConfig
    cases: tuple[SharedTrendCaseResult, ...]
    density_summaries: tuple[SharedTrendDensitySummary, ...]
    case_count: int
    base_safe_accept_count: int
    candidate_retained_base_safe_count: int
    base_safe_retention: float
    base_false_safe_count: int
    repaired_base_false_safe_count: int
    repair_fraction: float
    candidate_safe_accept_count: int
    candidate_false_safe_count: int
    phase7_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "calibration_sources": list(self.calibration_sources),
            "seed": self.seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "profiles": [
                asdict(profile) | {"name": profile.name}
                for profile in self.profiles
            ],
            "geometries": [geometry.value for geometry in self.geometries],
            "base_gate_config": asdict(self.base_gate_config),
            "shared_trend_config": asdict(self.shared_trend_config),
            "cases": [case.to_dict() for case in self.cases],
            "density_summaries": [row.to_dict() for row in self.density_summaries],
            "case_count": self.case_count,
            "base_safe_accept_count": self.base_safe_accept_count,
            "candidate_retained_base_safe_count": (
                self.candidate_retained_base_safe_count
            ),
            "base_safe_retention": self.base_safe_retention,
            "base_false_safe_count": self.base_false_safe_count,
            "repaired_base_false_safe_count": self.repaired_base_false_safe_count,
            "repair_fraction": self.repair_fraction,
            "candidate_safe_accept_count": self.candidate_safe_accept_count,
            "candidate_false_safe_count": self.candidate_false_safe_count,
            "phase7_supported": self.phase7_supported,
            "deployment_supported": self.deployment_supported,
        }


def _two_means(values: FloatArray) -> tuple[IntArray, tuple[float, float]]:
    centers = np.quantile(values, [0.25, 0.75]).astype(np.float64)
    labels = np.zeros(values.size, dtype=np.int64)
    for _ in range(64):
        updated = np.argmin(
            np.abs(values[:, None] - centers[None, :]),
            axis=1,
        ).astype(np.int64)
        if np.any(np.bincount(updated, minlength=2) == 0):
            break
        new_centers = np.asarray(
            [np.mean(values[updated == layer]) for layer in (0, 1)]
        )
        labels = updated
        if np.allclose(centers, new_centers, rtol=0.0, atol=1e-14):
            centers = new_centers
            break
        centers = new_centers
    order = np.argsort(centers)
    remap = np.empty(2, dtype=np.int64)
    remap[order] = np.arange(2, dtype=np.int64)
    return remap[labels], tuple(float(value) for value in centers[order])


def infer_shared_trend_layers(
    points: FloatArray,
    config: SharedTrendConfig | None = None,
) -> SharedTrendInference:
    """Infer two layers after removing a shared quadratic tangent-plane trend."""

    selected = SharedTrendConfig() if config is None else config
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if point_array.shape[0] <= selected.k_neighbors:
        raise ValueError("point count must exceed k_neighbors")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must be finite")

    geometry = local_neighborhood_geometry(
        point_array,
        k_neighbors=selected.k_neighbors,
    )
    normals = geometry.eigenvectors[:, :, 0]
    tensor = np.mean(normals[:, :, None] * normals[:, None, :], axis=0)
    frame = np.linalg.eigh(tensor)[1]
    coordinates = (point_array - np.mean(point_array, axis=0)) @ frame
    u, v, height = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    trend = np.column_stack(
        (np.ones(point_array.shape[0]), u, v, u * u, u * v, v * v)
    )
    labels, _ = _two_means(height)
    converged = False
    iterations = 0
    for iteration in range(selected.maximum_iterations):
        iterations = iteration + 1
        design = np.column_stack((trend, labels))
        coefficients = np.linalg.lstsq(design, height, rcond=None)[0]
        common = trend @ coefficients[:-1]
        residuals = height - common
        updated, centers = _two_means(residuals)
        if np.array_equal(labels, updated):
            labels = updated
            converged = True
            break
        labels = updated

    design = np.column_stack((trend, labels))
    coefficients = np.linalg.lstsq(design, height, rcond=None)[0]
    common = trend @ coefficients[:-1]
    residuals = height - common
    labels, centers = _two_means(residuals)
    cluster_counts = np.bincount(labels, minlength=2)
    cluster_fraction = float(np.min(cluster_counts) / point_array.shape[0])
    layer_gap = abs(centers[1] - centers[0])
    centered_residuals = residuals - np.asarray(centers)[labels]
    pooled_spread = float(np.sqrt(np.mean(centered_residuals**2)))
    separation_snr = layer_gap / max(pooled_spread, np.finfo(float).eps)
    selected_k = min(selected.k_neighbors, point_array.shape[0] - 1)
    neighbors = cKDTree(point_array).query(
        point_array,
        k=selected_k + 1,
        workers=1,
    )[1][:, 1:]
    cross_fraction = float(np.mean(labels[neighbors] != labels[:, None]))
    identifiable = bool(
        cluster_fraction >= selected.minimum_cluster_fraction
        and separation_snr >= selected.minimum_separation_snr
    )
    sufficient = bool(
        identifiable and cross_fraction <= selected.cross_knn_threshold
    )
    evidence = SamplingSufficiencyEvidence(
        information_boundary="observed_coordinates_shared_quadratic_trend_only",
        point_count=point_array.shape[0],
        k_neighbors=selected_k,
        cluster_sizes=(int(cluster_counts[0]), int(cluster_counts[1])),
        minimum_cluster_fraction=cluster_fraction,
        estimated_layer_gap=layer_gap,
        pooled_normal_spread=pooled_spread,
        separation_snr=separation_snr,
        estimated_cross_knn_fraction=cross_fraction,
        cross_knn_threshold=selected.cross_knn_threshold,
        two_layer_identifiable=identifiable,
        sampling_sufficient=sufficient,
    )
    diagnostics = SharedTrendDiagnostics(
        information_boundary="observed_coordinates_only",
        polynomial_terms=("1", "u", "v", "u2", "uv", "v2"),
        iterations=iterations,
        converged=converged,
        residual_layer_centers=centers,
        shared_trend_rmse=pooled_spread,
    )
    return SharedTrendInference(
        inference=ParallelLayerInference(layer_ids=labels, evidence=evidence),
        diagnostics=diagnostics,
    )


def construct_shared_trend_surface(
    points: FloatArray,
    config: SharedTrendConfig | None = None,
) -> tuple[TwoLayerConstruction, SharedTrendDiagnostics]:
    inferred = infer_shared_trend_layers(points, config)
    construction = construct_two_layer_surface_from_inference(
        points,
        inferred.inference,
    )
    return construction, inferred.diagnostics


def _true_safe(metrics: SurfaceEndpointMetrics) -> bool:
    return bool(
        metrics.component_error == 0
        and int(metrics.labeled_false_bridge_edges or 0) == 0
        and int(metrics.labeled_false_bridge_faces or 0) == 0
    )


def _density_summary(
    cases: Sequence[SharedTrendCaseResult],
    point_count: int,
) -> SharedTrendDensitySummary:
    rows = [case for case in cases if case.point_count == point_count]
    base_safe = sum(case.base_safe_accept for case in rows)
    retained = sum(
        case.base_safe_accept and case.candidate_safe_accept for case in rows
    )
    retention = None if not base_safe else retained / base_safe
    base_false = sum(case.base_false_safe for case in rows)
    repaired = sum(case.repaired_base_false_safe for case in rows)
    candidate_false = sum(case.candidate_false_safe for case in rows)
    return SharedTrendDensitySummary(
        point_count=point_count,
        case_count=len(rows),
        base_safe_accept_count=base_safe,
        candidate_retained_base_safe_count=retained,
        base_safe_retention=retention,
        base_false_safe_count=base_false,
        repaired_base_false_safe_count=repaired,
        candidate_false_safe_count=candidate_false,
        density_gate_passed=bool(
            candidate_false == 0
            and (base_safe < 8 or (retention is not None and retention >= 0.85))
        ),
    )


def evaluate_shared_trend_inference(
    *,
    profiles: Sequence[SamplingProfile] = DEFAULT_PROFILES,
    geometries: Sequence[ShiftGeometry | str] = DEFAULT_GEOMETRIES,
    reference_count: int = 2048,
    repeats: int = 8,
    seed: int = 20500804,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
) -> SharedTrendResult:
    """Run the frozen Phase-7 shared-trend reconstruction held-out."""

    selected_profiles = tuple(profiles)
    selected_geometries = tuple(ShiftGeometry(value) for value in geometries)
    if repeats < 1 or not selected_profiles or not selected_geometries:
        raise ValueError("profiles/geometries must be non-empty and repeats positive")
    selected_base = base_gate_config or SamplingSufficiencyConfig(
        minimum_separation_snr=3.0
    )
    selected_trend = shared_trend_config or SharedTrendConfig(
        k_neighbors=selected_base.k_neighbors,
        minimum_cluster_fraction=selected_base.minimum_cluster_fraction,
        minimum_separation_snr=selected_base.minimum_separation_snr,
        cross_knn_threshold=selected_base.cross_knn_threshold,
    )
    reconstruction_config = ReacquisitionConfig(
        base_point_count=max(profile.point_count for profile in selected_profiles),
        evaluation_reference_count=reference_count,
        candidate_pool_count=reference_count,
        added_point_counts=(1,),
        repeats=1,
        seed=seed,
        surface_sample_count=surface_sample_count,
        k_neighbors=selected_base.k_neighbors,
    )
    results: list[SharedTrendCaseResult] = []
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
                base = construct_two_layer_surface(case.points, selected_base)
                candidate, diagnostics = construct_shared_trend_surface(
                    case.points,
                    selected_trend,
                )
                base_inferred = evaluate_surface(
                    base.mesh,
                    case.reference_points,
                    expected_components=2,
                    expected_betti=(2, 0, 0),
                    vertex_component_labels=base.inference.layer_ids,
                    characteristic_length=case.characteristic_length,
                    sample_count=surface_sample_count,
                    threshold_fraction=(
                        reconstruction_config.fscore_threshold_fraction
                    ),
                    seed=case_seed + 41,
                )
                candidate_inferred = evaluate_surface(
                    candidate.mesh,
                    case.reference_points,
                    expected_components=2,
                    expected_betti=(2, 0, 0),
                    vertex_component_labels=candidate.inference.layer_ids,
                    characteristic_length=case.characteristic_length,
                    sample_count=surface_sample_count,
                    threshold_fraction=(
                        reconstruction_config.fscore_threshold_fraction
                    ),
                    seed=case_seed + 41,
                )
                base_truth = evaluate_surface(
                    base.mesh,
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
                candidate_truth = evaluate_surface(
                    candidate.mesh,
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
                base_decision = route_two_layer_output(base, base_inferred)
                candidate_decision = route_two_layer_output(
                    candidate,
                    candidate_inferred,
                )
                base_safe = _true_safe(base_truth)
                candidate_safe = _true_safe(candidate_truth)
                base_accept = base_decision is SamplingGateDecision.ACCEPT
                candidate_accept = (
                    candidate_decision is SamplingGateDecision.ACCEPT
                )
                base_false = bool(base_accept and not base_safe)
                candidate_safe_accept = bool(candidate_accept and candidate_safe)
                results.append(
                    SharedTrendCaseResult(
                        profile=profile.name,
                        point_count=profile.point_count,
                        noise=profile.noise,
                        geometry=geometry,
                        repeat=repeat,
                        seed=case_seed,
                        diagnostics=diagnostics,
                        base_decision=base_decision,
                        candidate_decision=candidate_decision,
                        base_true_safe_output=base_safe,
                        candidate_true_safe_output=candidate_safe,
                        base_safe_accept=bool(base_accept and base_safe),
                        candidate_safe_accept=candidate_safe_accept,
                        base_false_safe=base_false,
                        candidate_false_safe=bool(
                            candidate_accept and not candidate_safe
                        ),
                        repaired_base_false_safe=bool(
                            base_false and candidate_safe_accept
                        ),
                    )
                )

    point_counts = sorted({profile.point_count for profile in selected_profiles})
    summaries = tuple(
        _density_summary(results, point_count) for point_count in point_counts
    )
    base_safe = sum(case.base_safe_accept for case in results)
    retained = sum(
        case.base_safe_accept and case.candidate_safe_accept for case in results
    )
    base_false = sum(case.base_false_safe for case in results)
    repaired = sum(case.repaired_base_false_safe for case in results)
    candidate_false = sum(case.candidate_false_safe for case in results)
    retention = 0.0 if not base_safe else retained / base_safe
    repair_fraction = 0.0 if not base_false else repaired / base_false
    supported = bool(
        selected_profiles == DEFAULT_PROFILES
        and selected_geometries == DEFAULT_GEOMETRIES
        and repeats >= 8
        and len(results) == 360
        and base_false > 0
        and candidate_false == 0
        and retention >= 0.90
        and repair_fraction >= 0.90
        and all(row.density_gate_passed for row in summaries)
    )
    return SharedTrendResult(
        artifact_schema="pftf_alpha_shared_trend_inference_phase7/v1",
        role="shared_quadratic_trend_two_layer_reconstruction",
        information_boundary=(
            "both routes use observed coordinates only; profiles, geometries, "
            "true labels, and references are evaluation-only"
        ),
        calibration_sources=("phase5_seed_20300804", "phase6_seed_20400804"),
        seed=seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        profiles=selected_profiles,
        geometries=selected_geometries,
        base_gate_config=selected_base,
        shared_trend_config=selected_trend,
        cases=tuple(results),
        density_summaries=summaries,
        case_count=len(results),
        base_safe_accept_count=base_safe,
        candidate_retained_base_safe_count=retained,
        base_safe_retention=retention,
        base_false_safe_count=base_false,
        repaired_base_false_safe_count=repaired,
        repair_fraction=repair_fraction,
        candidate_safe_accept_count=sum(
            case.candidate_safe_accept for case in results
        ),
        candidate_false_safe_count=candidate_false,
        phase7_supported=supported,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20500804)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_shared_trend_inference(
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
