"""Local tangent-plane leave-one-out consensus and frozen Phase-10 audit."""

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

from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .sensor_stress import (
    DEFAULT_POINT_COUNTS,
    DEFAULT_STRESSES,
    SensorStress,
    evaluate_sensor_stress,
    make_sensor_stress_case,
)
from .shared_trend_inference import (
    SharedTrendConfig,
    construct_shared_trend_surface,
)
from .surface import SurfaceMesh, mesh_statistics

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class LocalSurfaceConsensusConfig:
    """Frozen observed-only thresholds for the Phase-10 guard."""

    k_neighbors: int = 12
    mad_consistency_factor: float = 1.4826
    minimum_scale_fraction: float = 0.04
    maximum_standardized_residual: float = 5.0
    harmful_distance_fraction: float = 0.025

    def __post_init__(self) -> None:
        if self.k_neighbors < 3:
            raise ValueError("k_neighbors must be at least three")
        for name, value in (
            ("mad_consistency_factor", self.mad_consistency_factor),
            ("minimum_scale_fraction", self.minimum_scale_fraction),
            ("maximum_standardized_residual", self.maximum_standardized_residual),
            ("harmful_distance_fraction", self.harmful_distance_fraction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class LocalConsensusScores:
    """Per-point leave-one-out residuals retained for unit-level auditing."""

    residuals: FloatArray
    local_scales: FloatArray
    standardized_residuals: FloatArray
    neighbor_counts: IntArray


@dataclass(frozen=True)
class LocalSurfaceConsensusEvidence:
    information_boundary: str
    point_count: int
    minimum_neighbor_count: int
    median_standardized_residual: float
    percentile95_standardized_residual: float
    maximum_standardized_residual: float
    allowed_maximum_standardized_residual: float
    maximum_raw_residual: float
    flagged_point_count: int
    flagged_point_fraction: float
    surface_consistent: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GeometryTopologyHarmEndpoint:
    """Evaluation-only split between provenance and realized output harm."""

    information_boundary: str
    harmful_distance_threshold: float
    source_outlier_vertex_count: int
    used_source_outlier_vertex_count: int
    provenance_violation_face_count: int
    provenance_violation_edge_count: int
    harmful_outlier_vertex_count: int
    harmful_outlier_face_count: int
    clean_cross_layer_face_count: int
    component_error: int
    betti_error: int
    provenance_violation_present: bool
    geometry_topology_harm_present: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalSurfaceConsensusCaseResult:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    evidence: LocalSurfaceConsensusEvidence
    endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision
    guarded_decision: SamplingGateDecision
    unguarded_safe_accept: bool
    guarded_safe_accept: bool
    unguarded_harmful_outlier_false_safe: bool
    guarded_harmful_outlier_false_safe: bool
    unguarded_provenance_violation_accept: bool
    guarded_provenance_violation_accept: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress"] = self.stress.value
        payload["unguarded_decision"] = self.unguarded_decision.value
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class LocalSurfaceConsensusSummary:
    stress: SensorStress
    case_count: int
    unguarded_safe_accept_count: int
    guarded_safe_accept_count: int
    safe_accept_retention: float | None
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int
    gate_relevant: bool
    group_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress"] = self.stress.value
        return payload


@dataclass(frozen=True)
class LocalSurfaceConsensusResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    consensus_config: LocalSurfaceConsensusConfig
    cases: tuple[LocalSurfaceConsensusCaseResult, ...]
    stress_summaries: tuple[LocalSurfaceConsensusSummary, ...]
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int
    clean_local_bump_unguarded_safe_accept_count: int
    clean_local_bump_guarded_safe_accept_count: int
    clean_local_bump_safe_accept_retention: float
    phase10_supported: bool
    trimmed_reconstruction_supported: bool
    real_scan_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_predecessor": self.frozen_predecessor,
            "seed": self.seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "consensus_config": asdict(self.consensus_config),
            "cases": [case.to_dict() for case in self.cases],
            "stress_summaries": [row.to_dict() for row in self.stress_summaries],
            "case_count": self.case_count,
            "unguarded_harmful_outlier_false_safe_count": (
                self.unguarded_harmful_outlier_false_safe_count
            ),
            "guarded_harmful_outlier_false_safe_count": (
                self.guarded_harmful_outlier_false_safe_count
            ),
            "unguarded_provenance_violation_accept_count": (
                self.unguarded_provenance_violation_accept_count
            ),
            "guarded_provenance_violation_accept_count": (
                self.guarded_provenance_violation_accept_count
            ),
            "clean_local_bump_unguarded_safe_accept_count": (
                self.clean_local_bump_unguarded_safe_accept_count
            ),
            "clean_local_bump_guarded_safe_accept_count": (
                self.clean_local_bump_guarded_safe_accept_count
            ),
            "clean_local_bump_safe_accept_retention": (
                self.clean_local_bump_safe_accept_retention
            ),
            "phase10_supported": self.phase10_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "real_scan_supported": self.real_scan_supported,
            "deployment_supported": self.deployment_supported,
        }


def local_tangent_plane_scores(
    points: FloatArray,
    inferred_labels: IntArray,
    config: LocalSurfaceConsensusConfig | None = None,
) -> LocalConsensusScores:
    """Fit a leave-one-out tangent plane in each inferred layer."""

    selected = LocalSurfaceConsensusConfig() if config is None else config
    point_array = np.asarray(points, dtype=np.float64)
    labels = np.asarray(inferred_labels, dtype=np.int64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if labels.shape != (point_array.shape[0],) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("inferred_labels must contain two aligned layers")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must be finite")

    residuals = np.zeros(point_array.shape[0], dtype=np.float64)
    local_scales = np.zeros(point_array.shape[0], dtype=np.float64)
    neighbor_counts = np.zeros(point_array.shape[0], dtype=np.int64)
    epsilon = np.finfo(float).eps
    for layer in (0, 1):
        indices = np.flatnonzero(labels == layer)
        if indices.size < 4:
            raise ValueError("each inferred layer requires at least four points")
        layer_points = point_array[indices]
        selected_k = min(selected.k_neighbors, indices.size - 1)
        local_neighbors = cKDTree(layer_points).query(
            layer_points,
            k=selected_k + 1,
            workers=1,
        )[1][:, 1:]
        for local_index, neighbor_rows in enumerate(local_neighbors):
            neighbors = layer_points[neighbor_rows]
            center = np.mean(neighbors, axis=0)
            centered = neighbors - center
            normal = np.linalg.eigh(centered.T @ centered)[1][:, 0]
            signed = centered @ normal
            signed_center = float(np.median(signed))
            plane_origin = center + signed_center * normal
            robust_scale = selected.mad_consistency_factor * float(
                np.median(np.abs(signed - signed_center))
            )
            tangent_vectors = centered - signed[:, None] * normal[None, :]
            tangent_radius = float(
                np.median(np.linalg.norm(tangent_vectors, axis=1))
            )
            scale = max(
                robust_scale,
                selected.minimum_scale_fraction * tangent_radius,
                epsilon,
            )
            global_index = int(indices[local_index])
            residuals[global_index] = abs(
                float((point_array[global_index] - plane_origin) @ normal)
            )
            local_scales[global_index] = scale
            neighbor_counts[global_index] = selected_k

    return LocalConsensusScores(
        residuals=residuals,
        local_scales=local_scales,
        standardized_residuals=residuals / local_scales,
        neighbor_counts=neighbor_counts,
    )


def estimate_local_surface_consensus(
    points: FloatArray,
    inferred_labels: IntArray,
    config: LocalSurfaceConsensusConfig | None = None,
) -> LocalSurfaceConsensusEvidence:
    """Summarize observed-only leave-one-out surface consistency."""

    selected = LocalSurfaceConsensusConfig() if config is None else config
    scores = local_tangent_plane_scores(points, inferred_labels, selected)
    standardized = scores.standardized_residuals
    flagged = standardized > selected.maximum_standardized_residual
    return LocalSurfaceConsensusEvidence(
        information_boundary="observed_coordinates_and_inferred_layers_only",
        point_count=standardized.size,
        minimum_neighbor_count=int(np.min(scores.neighbor_counts)),
        median_standardized_residual=float(np.median(standardized)),
        percentile95_standardized_residual=float(
            np.percentile(standardized, 95.0)
        ),
        maximum_standardized_residual=float(np.max(standardized)),
        allowed_maximum_standardized_residual=(
            selected.maximum_standardized_residual
        ),
        maximum_raw_residual=float(np.max(scores.residuals)),
        flagged_point_count=int(np.sum(flagged)),
        flagged_point_fraction=float(np.mean(flagged)),
        surface_consistent=bool(not np.any(flagged)),
    )


def route_with_local_surface_consensus(
    decision: SamplingGateDecision,
    evidence: LocalSurfaceConsensusEvidence,
) -> SamplingGateDecision:
    if decision is SamplingGateDecision.ACCEPT and not evidence.surface_consistent:
        return SamplingGateDecision.UNSUPPORTED
    return decision


def evaluate_geometry_topology_harm(
    mesh: SurfaceMesh,
    reference_points: FloatArray,
    source_labels: IntArray,
    *,
    characteristic_length: float,
    config: LocalSurfaceConsensusConfig | None = None,
) -> GeometryTopologyHarmEndpoint:
    """Separate any source-2 use from output-relevant geometric harm."""

    selected = LocalSurfaceConsensusConfig() if config is None else config
    labels = np.asarray(source_labels, dtype=np.int64)
    reference = np.asarray(reference_points, dtype=np.float64)
    if labels.shape != (mesh.vertices.shape[0],):
        raise ValueError("source_labels must align with mesh vertices")
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise ValueError("source_labels must use only 0, 1, and 2")
    if reference.ndim != 2 or reference.shape[1] != 3 or not reference.size:
        raise ValueError("reference_points must have shape (n, 3)")
    if not math.isfinite(characteristic_length) or characteristic_length <= 0.0:
        raise ValueError("characteristic_length must be finite and positive")

    threshold = selected.harmful_distance_fraction * characteristic_length
    faces = mesh.faces
    face_labels = labels[faces]
    provenance_faces = np.any(face_labels == 2, axis=1)
    clean_cross_layer_faces = np.logical_and(
        np.any(face_labels == 0, axis=1),
        np.any(face_labels == 1, axis=1),
    )

    edge_rows: set[tuple[int, int]] = set()
    for first, second, third in faces.tolist():
        edge_rows.update(
            (
                tuple(sorted((first, second))),
                tuple(sorted((second, third))),
                tuple(sorted((third, first))),
            )
        )
    provenance_edges = sum(
        (labels[left] == 2) != (labels[right] == 2)
        for left, right in edge_rows
    )

    outlier_indices = np.flatnonzero(labels == 2)
    used_vertices = np.unique(faces) if faces.size else np.empty(0, dtype=np.int64)
    used_outliers = np.intersect1d(outlier_indices, used_vertices)
    harmful_vertex_mask = np.zeros(mesh.vertices.shape[0], dtype=bool)
    if outlier_indices.size:
        distances = cKDTree(reference).query(
            mesh.vertices[outlier_indices],
            workers=1,
        )[0]
        harmful_vertex_mask[outlier_indices] = distances > threshold
    harmful_used = np.flatnonzero(
        harmful_vertex_mask & np.isin(np.arange(mesh.vertices.shape[0]), used_vertices)
    )
    harmful_faces = (
        np.any(harmful_vertex_mask[faces], axis=1)
        if faces.size
        else np.empty(0, dtype=bool)
    )

    statistics = mesh_statistics(mesh)
    component_error = abs(statistics.connected_components - 2)
    betti_error = sum(
        abs(actual - expected)
        for actual, expected in zip(
            (statistics.betti_0, statistics.betti_1, statistics.betti_2),
            (2, 0, 0),
            strict=True,
        )
    )
    harmful_face_count = int(np.sum(harmful_faces))
    cross_layer_face_count = int(np.sum(clean_cross_layer_faces))
    harm_present = bool(
        harmful_face_count > 0
        or cross_layer_face_count > 0
        or component_error > 0
        or betti_error > 0
    )
    return GeometryTopologyHarmEndpoint(
        information_boundary=(
            "evaluation_only_source_labels_and_clean_reference; never used by route"
        ),
        harmful_distance_threshold=threshold,
        source_outlier_vertex_count=int(outlier_indices.size),
        used_source_outlier_vertex_count=int(used_outliers.size),
        provenance_violation_face_count=int(np.sum(provenance_faces)),
        provenance_violation_edge_count=int(provenance_edges),
        harmful_outlier_vertex_count=int(harmful_used.size),
        harmful_outlier_face_count=harmful_face_count,
        clean_cross_layer_face_count=cross_layer_face_count,
        component_error=component_error,
        betti_error=betti_error,
        provenance_violation_present=bool(np.any(provenance_faces)),
        geometry_topology_harm_present=harm_present,
    )


def _summary(
    rows: Sequence[LocalSurfaceConsensusCaseResult],
    stress: SensorStress,
) -> LocalSurfaceConsensusSummary:
    selected = [case for case in rows if case.stress is stress]
    unguarded_safe = sum(case.unguarded_safe_accept for case in selected)
    guarded_safe = sum(case.guarded_safe_accept for case in selected)
    retention = None if not unguarded_safe else guarded_safe / unguarded_safe
    guarded_harmful = sum(
        case.guarded_harmful_outlier_false_safe for case in selected
    )
    gate_relevant = stress.is_outlier_stress or stress in (
        SensorStress.CONTROL,
        SensorStress.LOCAL_BUMP,
    )
    if stress.is_outlier_stress:
        passed = guarded_harmful == 0
    elif stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP):
        passed = retention is not None and retention >= 0.90
    else:
        passed = True
    return LocalSurfaceConsensusSummary(
        stress=stress,
        case_count=len(selected),
        unguarded_safe_accept_count=unguarded_safe,
        guarded_safe_accept_count=guarded_safe,
        safe_accept_retention=retention,
        unguarded_harmful_outlier_false_safe_count=sum(
            case.unguarded_harmful_outlier_false_safe for case in selected
        ),
        guarded_harmful_outlier_false_safe_count=guarded_harmful,
        unguarded_provenance_violation_accept_count=sum(
            case.unguarded_provenance_violation_accept for case in selected
        ),
        guarded_provenance_violation_accept_count=sum(
            case.guarded_provenance_violation_accept for case in selected
        ),
        gate_relevant=gate_relevant,
        group_gate_passed=bool(passed),
    )


def evaluate_local_surface_consensus(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    seed: int = 20800804,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
    consensus_config: LocalSurfaceConsensusConfig | None = None,
) -> LocalSurfaceConsensusResult:
    """Run the once-frozen Phase-10 local-consensus held-out panel."""

    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_consensus = (
        LocalSurfaceConsensusConfig()
        if consensus_config is None
        else consensus_config
    )
    if repeats < 1 or not selected_counts or not selected_stresses:
        raise ValueError("counts/stresses must be non-empty and repeats positive")
    base_result = evaluate_sensor_stress(
        point_counts=selected_counts,
        stresses=selected_stresses,
        reference_count=reference_count,
        repeats=repeats,
        seed=seed,
        surface_sample_count=surface_sample_count,
        base_gate_config=base_gate_config,
        shared_trend_config=shared_trend_config,
    )
    results: list[LocalSurfaceConsensusCaseResult] = []
    for case_row in base_result.cases:
        case = make_sensor_stress_case(
            case_row.stress,
            case_row.point_count,
            reference_count=max(reference_count, case_row.point_count),
            seed=case_row.seed,
        )
        construction, _ = construct_shared_trend_surface(
            case.points,
            shared_trend_config,
        )
        evidence = estimate_local_surface_consensus(
            case.points,
            construction.inference.layer_ids,
            selected_consensus,
        )
        guarded_decision = route_with_local_surface_consensus(
            case_row.candidate_decision,
            evidence,
        )
        endpoint = evaluate_geometry_topology_harm(
            construction.mesh,
            case.reference_points,
            case.point_component_labels,
            characteristic_length=case.characteristic_length,
            config=selected_consensus,
        )
        unguarded_accept = (
            case_row.candidate_decision is SamplingGateDecision.ACCEPT
        )
        guarded_accept = guarded_decision is SamplingGateDecision.ACCEPT
        harmful_outlier = bool(
            case_row.stress.is_outlier_stress
            and endpoint.geometry_topology_harm_present
        )
        results.append(
            LocalSurfaceConsensusCaseResult(
                stress=case_row.stress,
                point_count=case_row.point_count,
                repeat=case_row.repeat,
                seed=case_row.seed,
                evidence=evidence,
                endpoint=endpoint,
                unguarded_decision=case_row.candidate_decision,
                guarded_decision=guarded_decision,
                unguarded_safe_accept=bool(
                    unguarded_accept and not endpoint.geometry_topology_harm_present
                ),
                guarded_safe_accept=bool(
                    guarded_accept and not endpoint.geometry_topology_harm_present
                ),
                unguarded_harmful_outlier_false_safe=bool(
                    unguarded_accept and harmful_outlier
                ),
                guarded_harmful_outlier_false_safe=bool(
                    guarded_accept and harmful_outlier
                ),
                unguarded_provenance_violation_accept=bool(
                    unguarded_accept and endpoint.provenance_violation_present
                ),
                guarded_provenance_violation_accept=bool(
                    guarded_accept and endpoint.provenance_violation_present
                ),
            )
        )

    summaries = tuple(
        _summary(results, stress) for stress in selected_stresses
    )
    clean_local = [
        case
        for case in results
        if case.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_clean = sum(case.unguarded_safe_accept for case in clean_local)
    guarded_clean = sum(
        case.unguarded_safe_accept and case.guarded_safe_accept
        for case in clean_local
    )
    retention = 0.0 if not unguarded_clean else guarded_clean / unguarded_clean
    unguarded_harmful = sum(
        case.unguarded_harmful_outlier_false_safe for case in results
    )
    guarded_harmful = sum(
        case.guarded_harmful_outlier_false_safe for case in results
    )
    supported = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and repeats >= 8
        and len(results) == 216
        and unguarded_harmful > 0
        and guarded_harmful == 0
        and retention >= 0.90
        and all(row.group_gate_passed for row in summaries if row.gate_relevant)
    )
    return LocalSurfaceConsensusResult(
        artifact_schema="pftf_alpha_local_surface_consensus_phase10/v1",
        role="local_tangent_plane_leave_one_out_fail_closed_guard",
        information_boundary=(
            "route uses observed coordinates and inferred layers only; stress, "
            "source labels, and clean references are evaluation-only"
        ),
        frozen_predecessor="phase9_seed_20700804",
        seed=seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        consensus_config=selected_consensus,
        cases=tuple(results),
        stress_summaries=summaries,
        case_count=len(results),
        unguarded_harmful_outlier_false_safe_count=unguarded_harmful,
        guarded_harmful_outlier_false_safe_count=guarded_harmful,
        unguarded_provenance_violation_accept_count=sum(
            case.unguarded_provenance_violation_accept for case in results
        ),
        guarded_provenance_violation_accept_count=sum(
            case.guarded_provenance_violation_accept for case in results
        ),
        clean_local_bump_unguarded_safe_accept_count=unguarded_clean,
        clean_local_bump_guarded_safe_accept_count=guarded_clean,
        clean_local_bump_safe_accept_retention=retention,
        phase10_supported=supported,
        trimmed_reconstruction_supported=False,
        real_scan_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20800804)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_local_surface_consensus(
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
