"""Untouched Phase-50 benchmark for bounded two-layer reconstruction efficacy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .adaptive import pca_anisotropic_filtration
from .filtration import AlphaFiltration
from .sampling_gate import (
    SamplingGateDecision,
    SamplingSufficiencyConfig,
)
from .sensor_stress import SensorStress, make_sensor_stress_case
from .shared_trend_inference import (
    SharedTrendConfig,
    SharedTrendDiagnostics,
    construct_shared_trend_surface,
)
from .surface import SurfaceEndpointMetrics, SurfaceMesh, evaluate_surface
from .synthetic import SyntheticCase
from .two_layer_confirmatory_protocol import (
    B5_MAX_NORMAL_PENALTY,
    B5_SCALE_MULTIPLIER,
    FSCORE_THRESHOLD_FRACTION,
    HELD_OUT_SEED,
    K_NEIGHBORS,
    M1_SCALE_MULTIPLIER,
    M1_WEIGHT_SCALE,
    MINIMUM_CASEWISE_FSCORE_WIN_RATE,
    MINIMUM_MEAN_FSCORE_MARGIN,
    MINIMUM_OVERALL_SAFE_ACCEPTANCE,
    MINIMUM_REPAIRED_BASE_FALSE_SAFE,
    MINIMUM_SUBGROUP_SAFE_ACCEPTANCE,
    POINT_COUNTS,
    REFERENCE_COUNT,
    REPEATS,
    STRESSES,
    SURFACE_SAMPLE_COUNT,
    preregister_two_layer_confirmatory,
)
from .two_layer_connectivity import (
    construct_two_layer_surface,
    route_two_layer_output,
)
from .weighted_alpha import PointSubmersionError, weighted_alpha_filtration

FloatArray = NDArray[np.float64]

RESULT_SCHEMA = "pftf_alpha_two_layer_confirmatory_phase50/v1"
PROTOCOL_COMMIT = "940219376d7fe3c50233fbe44cdef6c33a4890a7"
PROTOCOL_SHA256 = (
    "7615721e347647def8589cbf9204723ba000c529487a32dbfd7dd2d1a6839c76"
)
POSE_SEED_OFFSET = 7_000_001
EVALUATION_SEED_OFFSET = 50_000


@dataclass(frozen=True)
class ConfirmatoryMethodMetrics:
    fscore: float
    geometry_loss: float
    component_error: int
    betti_error: int
    labeled_false_bridge_edges: int
    labeled_false_bridge_faces: int
    topology_error: int
    nonmanifold_edges: int
    connected_components: int
    betti_0: int
    betti_1: int
    betti_2: int
    faces: int

    @classmethod
    def from_endpoints(
        cls, endpoints: SurfaceEndpointMetrics
    ) -> ConfirmatoryMethodMetrics:
        betti_error = int(endpoints.betti_error or 0)
        bridge_edges = int(endpoints.labeled_false_bridge_edges or 0)
        bridge_faces = int(endpoints.labeled_false_bridge_faces or 0)
        topology_error = (
            endpoints.component_error
            + betti_error
            + bridge_edges
            + bridge_faces
        )
        return cls(
            fscore=endpoints.fscore,
            geometry_loss=(
                endpoints.normalized_chamfer_squared
                + endpoints.normalized_hausdorff
            ),
            component_error=endpoints.component_error,
            betti_error=betti_error,
            labeled_false_bridge_edges=bridge_edges,
            labeled_false_bridge_faces=bridge_faces,
            topology_error=topology_error,
            nonmanifold_edges=endpoints.nonmanifold_edges,
            connected_components=endpoints.connected_components,
            betti_0=endpoints.betti_0,
            betti_1=endpoints.betti_1,
            betti_2=endpoints.betti_2,
            faces=endpoints.faces,
        )


@dataclass(frozen=True)
class TwoLayerConfirmatoryCaseResult:
    point_count: int
    stress: SensorStress
    repeat: int
    seed: int
    rotation_matrix: tuple[tuple[float, float, float], ...]
    rotation_determinant: float
    rotation_orthogonality_error: float
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
    base: ConfirmatoryMethodMetrics
    candidate: ConfirmatoryMethodMetrics
    b5: ConfirmatoryMethodMetrics
    m1: ConfirmatoryMethodMetrics | None
    m1_point_submerged: bool
    candidate_fscore_wins_b5: bool
    candidate_fscore_wins_m1: bool | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress"] = self.stress.value
        payload["base_decision"] = self.base_decision.value
        payload["candidate_decision"] = self.candidate_decision.value
        return payload


@dataclass(frozen=True)
class TwoLayerConfirmatorySubgroup:
    point_count: int
    stress: SensorStress
    case_count: int
    candidate_safe_accept_count: int
    candidate_false_safe_count: int
    safe_acceptance_coverage: float
    candidate_mean_fscore: float
    b5_mean_fscore: float
    m1_mean_fscore: float | None
    candidate_b5_mean_fscore_margin: float
    candidate_m1_mean_fscore_margin: float | None
    subgroup_safety_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress"] = self.stress.value
        return payload


@dataclass(frozen=True)
class TwoLayerConfirmatoryResult:
    artifact_schema: str
    role: str
    protocol_commit: str
    protocol_sha256: str
    protocol_identity_passed: bool
    information_boundary: str
    seed: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    repeats: int
    reference_count: int
    surface_sample_count: int
    k_neighbors: int
    b5_scale_multiplier: float
    b5_max_normal_penalty: float
    m1_weight_scale: float
    m1_scale_multiplier: float
    cases: tuple[TwoLayerConfirmatoryCaseResult, ...]
    subgroups: tuple[TwoLayerConfirmatorySubgroup, ...]
    case_count: int
    expected_case_count: int
    m1_available_case_count: int
    base_safe_accept_count: int
    base_false_safe_count: int
    candidate_safe_accept_count: int
    candidate_false_safe_count: int
    repaired_base_false_safe_count: int
    candidate_safe_acceptance_coverage: float
    candidate_mean_fscore: float
    b5_mean_fscore: float
    m1_mean_fscore: float | None
    candidate_b5_mean_fscore_margin: float
    candidate_m1_mean_fscore_margin: float | None
    candidate_mean_geometry_loss: float
    b5_mean_geometry_loss: float
    m1_mean_geometry_loss: float | None
    candidate_b5_casewise_fscore_win_rate: float
    candidate_m1_casewise_fscore_win_rate: float | None
    candidate_topology_error_sum: int
    b5_topology_error_sum: int
    m1_topology_error_sum: int | None
    candidate_nonmanifold_edges_sum: int
    safety_gate_passed: bool
    efficacy_gate_passed: bool
    topology_gate_passed: bool
    ablation_gate_passed: bool
    phase50_supported: bool
    promotion_supported: bool
    pftf_superiority_supported: bool
    real_scan_supported: bool
    deployment_supported: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "protocol_commit": self.protocol_commit,
            "protocol_sha256": self.protocol_sha256,
            "protocol_identity_passed": self.protocol_identity_passed,
            "information_boundary": self.information_boundary,
            "seed": self.seed,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "repeats": self.repeats,
            "reference_count": self.reference_count,
            "surface_sample_count": self.surface_sample_count,
            "k_neighbors": self.k_neighbors,
            "b5_scale_multiplier": self.b5_scale_multiplier,
            "b5_max_normal_penalty": self.b5_max_normal_penalty,
            "m1_weight_scale": self.m1_weight_scale,
            "m1_scale_multiplier": self.m1_scale_multiplier,
            "cases": [case.to_dict() for case in self.cases],
            "subgroups": [row.to_dict() for row in self.subgroups],
            "case_count": self.case_count,
            "expected_case_count": self.expected_case_count,
            "m1_available_case_count": self.m1_available_case_count,
            "base_safe_accept_count": self.base_safe_accept_count,
            "base_false_safe_count": self.base_false_safe_count,
            "candidate_safe_accept_count": self.candidate_safe_accept_count,
            "candidate_false_safe_count": self.candidate_false_safe_count,
            "repaired_base_false_safe_count": (
                self.repaired_base_false_safe_count
            ),
            "candidate_safe_acceptance_coverage": (
                self.candidate_safe_acceptance_coverage
            ),
            "candidate_mean_fscore": self.candidate_mean_fscore,
            "b5_mean_fscore": self.b5_mean_fscore,
            "m1_mean_fscore": self.m1_mean_fscore,
            "candidate_b5_mean_fscore_margin": (
                self.candidate_b5_mean_fscore_margin
            ),
            "candidate_m1_mean_fscore_margin": (
                self.candidate_m1_mean_fscore_margin
            ),
            "candidate_mean_geometry_loss": self.candidate_mean_geometry_loss,
            "b5_mean_geometry_loss": self.b5_mean_geometry_loss,
            "m1_mean_geometry_loss": self.m1_mean_geometry_loss,
            "candidate_b5_casewise_fscore_win_rate": (
                self.candidate_b5_casewise_fscore_win_rate
            ),
            "candidate_m1_casewise_fscore_win_rate": (
                self.candidate_m1_casewise_fscore_win_rate
            ),
            "candidate_topology_error_sum": self.candidate_topology_error_sum,
            "b5_topology_error_sum": self.b5_topology_error_sum,
            "m1_topology_error_sum": self.m1_topology_error_sum,
            "candidate_nonmanifold_edges_sum": (
                self.candidate_nonmanifold_edges_sum
            ),
            "safety_gate_passed": self.safety_gate_passed,
            "efficacy_gate_passed": self.efficacy_gate_passed,
            "topology_gate_passed": self.topology_gate_passed,
            "ablation_gate_passed": self.ablation_gate_passed,
            "phase50_supported": self.phase50_supported,
            "promotion_supported": self.promotion_supported,
            "pftf_superiority_supported": self.pftf_superiority_supported,
            "real_scan_supported": self.real_scan_supported,
            "deployment_supported": self.deployment_supported,
            "claim_boundary": self.claim_boundary,
        }


def proper_rotation(seed: int) -> FloatArray:
    """Return one deterministic, proper, orthogonal 3D rotation."""

    rng = np.random.default_rng(seed + POSE_SEED_OFFSET)
    orthogonal, triangular = np.linalg.qr(rng.normal(size=(3, 3)))
    diagonal = np.sign(np.diag(triangular))
    diagonal[diagonal == 0.0] = 1.0
    rotation = orthogonal @ np.diag(diagonal)
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return np.ascontiguousarray(rotation, dtype=np.float64)


def _rotate_case(case: SyntheticCase, rotation: FloatArray) -> SyntheticCase:
    return SyntheticCase(
        family=case.family,
        split=case.split,
        points=case.points @ rotation.T,
        reference_points=case.reference_points @ rotation.T,
        expected_components=case.expected_components,
        characteristic_length=case.characteristic_length,
        variation=case.variation,
        seed=case.seed,
        expected_surface_betti=case.expected_surface_betti,
        point_component_labels=case.point_component_labels,
    )


def _evaluate(
    mesh: SurfaceMesh,
    case: SyntheticCase,
    *,
    labels: NDArray[np.int64],
    expected_components: int,
    sample_count: int,
    seed: int,
) -> SurfaceEndpointMetrics:
    return evaluate_surface(
        mesh,
        case.reference_points,
        expected_components=expected_components,
        expected_betti=(expected_components, 0, 0),
        vertex_component_labels=labels,
        characteristic_length=case.characteristic_length,
        sample_count=sample_count,
        threshold_fraction=FSCORE_THRESHOLD_FRACTION,
        seed=seed,
    )


def _true_safe(endpoints: SurfaceEndpointMetrics) -> bool:
    return bool(
        endpoints.component_error == 0
        and int(endpoints.labeled_false_bridge_edges or 0) == 0
        and int(endpoints.labeled_false_bridge_faces or 0) == 0
    )


def _protocol_identity(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    expected = preregister_two_layer_confirmatory().to_dict()
    return digest, bool(digest == PROTOCOL_SHA256 and payload == expected)


def _mean(rows: Sequence[float]) -> float:
    return float(np.mean(np.asarray(rows, dtype=np.float64)))


def _subgroup(
    cases: Sequence[TwoLayerConfirmatoryCaseResult],
    point_count: int,
    stress: SensorStress,
) -> TwoLayerConfirmatorySubgroup:
    rows = [
        case
        for case in cases
        if case.point_count == point_count and case.stress is stress
    ]
    m1_rows = [case.m1 for case in rows if case.m1 is not None]
    candidate_mean = _mean([case.candidate.fscore for case in rows])
    b5_mean = _mean([case.b5.fscore for case in rows])
    m1_mean = (
        _mean([metrics.fscore for metrics in m1_rows])
        if len(m1_rows) == len(rows)
        else None
    )
    safe_accepts = sum(case.candidate_safe_accept for case in rows)
    false_safe = sum(case.candidate_false_safe for case in rows)
    coverage = safe_accepts / len(rows)
    return TwoLayerConfirmatorySubgroup(
        point_count=point_count,
        stress=stress,
        case_count=len(rows),
        candidate_safe_accept_count=safe_accepts,
        candidate_false_safe_count=false_safe,
        safe_acceptance_coverage=coverage,
        candidate_mean_fscore=candidate_mean,
        b5_mean_fscore=b5_mean,
        m1_mean_fscore=m1_mean,
        candidate_b5_mean_fscore_margin=candidate_mean - b5_mean,
        candidate_m1_mean_fscore_margin=(
            None if m1_mean is None else candidate_mean - m1_mean
        ),
        subgroup_safety_gate_passed=bool(
            false_safe == 0 and coverage >= MINIMUM_SUBGROUP_SAFE_ACCEPTANCE
        ),
    )


def evaluate_two_layer_confirmatory(
    *,
    point_counts: Sequence[int] = POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = STRESSES,
    repeats: int = REPEATS,
    reference_count: int = REFERENCE_COUNT,
    surface_sample_count: int = SURFACE_SAMPLE_COUNT,
    seed: int = HELD_OUT_SEED,
    protocol_path: str | Path = (
        "benchmark-out/two_layer_confirmatory_protocol_phase50.json"
    ),
) -> TwoLayerConfirmatoryResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    if not selected_counts or not selected_stresses or repeats < 1:
        raise ValueError("point_counts/stresses must be non-empty and repeats positive")
    if any(count < 32 for count in selected_counts):
        raise ValueError("point counts must be at least 32")
    if reference_count < max(selected_counts):
        raise ValueError("reference_count must cover the largest point count")
    if surface_sample_count < 1:
        raise ValueError("surface_sample_count must be positive")

    protocol_digest, protocol_file_matches = _protocol_identity(Path(protocol_path))
    base_config = SamplingSufficiencyConfig(minimum_separation_snr=3.0)
    candidate_config = SharedTrendConfig(
        k_neighbors=base_config.k_neighbors,
        minimum_cluster_fraction=base_config.minimum_cluster_fraction,
        minimum_separation_snr=base_config.minimum_separation_snr,
        cross_knn_threshold=base_config.cross_knn_threshold,
    )
    cases: list[TwoLayerConfirmatoryCaseResult] = []
    for count_index, point_count in enumerate(selected_counts):
        for stress_index, stress in enumerate(selected_stresses):
            for repeat in range(repeats):
                case_seed = (
                    seed
                    + count_index * 1_000_003
                    + stress_index * 100_003
                    + repeat * 10_007
                )
                raw_case = make_sensor_stress_case(
                    stress,
                    point_count,
                    reference_count=reference_count,
                    seed=case_seed,
                )
                rotation = proper_rotation(case_seed)
                case = _rotate_case(raw_case, rotation)
                base = construct_two_layer_surface(case.points, base_config)
                candidate, diagnostics = construct_shared_trend_surface(
                    case.points,
                    candidate_config,
                )
                evaluation_seed = case_seed + EVALUATION_SEED_OFFSET
                base_inferred = _evaluate(
                    base.mesh,
                    case,
                    labels=base.inference.layer_ids,
                    expected_components=2,
                    sample_count=surface_sample_count,
                    seed=evaluation_seed,
                )
                candidate_inferred = _evaluate(
                    candidate.mesh,
                    case,
                    labels=candidate.inference.layer_ids,
                    expected_components=2,
                    sample_count=surface_sample_count,
                    seed=evaluation_seed,
                )
                true_labels = np.asarray(case.point_component_labels, dtype=np.int64)
                base_truth = _evaluate(
                    base.mesh,
                    case,
                    labels=true_labels,
                    expected_components=2,
                    sample_count=surface_sample_count,
                    seed=evaluation_seed,
                )
                candidate_truth = _evaluate(
                    candidate.mesh,
                    case,
                    labels=true_labels,
                    expected_components=2,
                    sample_count=surface_sample_count,
                    seed=evaluation_seed,
                )

                filtration = AlphaFiltration.from_points(case.points)
                b5_adaptive = pca_anisotropic_filtration(
                    filtration,
                    k_neighbors=K_NEIGHBORS,
                    max_normal_penalty=B5_MAX_NORMAL_PENALTY,
                )
                b5_truth = _evaluate(
                    b5_adaptive.surface_at(B5_SCALE_MULTIPLIER),
                    case,
                    labels=true_labels,
                    expected_components=2,
                    sample_count=surface_sample_count,
                    seed=evaluation_seed,
                )
                try:
                    m1_adaptive = weighted_alpha_filtration(
                        case.points,
                        k_neighbors=K_NEIGHBORS,
                        weight_scale=M1_WEIGHT_SCALE,
                    )
                    m1_truth = _evaluate(
                        m1_adaptive.surface_at(M1_SCALE_MULTIPLIER),
                        case,
                        labels=true_labels,
                        expected_components=2,
                        sample_count=surface_sample_count,
                        seed=evaluation_seed,
                    )
                except PointSubmersionError:
                    m1_truth = None

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
                base_metrics = ConfirmatoryMethodMetrics.from_endpoints(base_truth)
                candidate_metrics = ConfirmatoryMethodMetrics.from_endpoints(
                    candidate_truth
                )
                b5_metrics = ConfirmatoryMethodMetrics.from_endpoints(b5_truth)
                m1_metrics = (
                    None
                    if m1_truth is None
                    else ConfirmatoryMethodMetrics.from_endpoints(m1_truth)
                )
                cases.append(
                    TwoLayerConfirmatoryCaseResult(
                        point_count=point_count,
                        stress=stress,
                        repeat=repeat,
                        seed=case_seed,
                        rotation_matrix=tuple(
                            tuple(float(value) for value in row) for row in rotation
                        ),
                        rotation_determinant=float(np.linalg.det(rotation)),
                        rotation_orthogonality_error=float(
                            np.max(np.abs(rotation @ rotation.T - np.eye(3)))
                        ),
                        diagnostics=diagnostics,
                        base_decision=base_decision,
                        candidate_decision=candidate_decision,
                        base_true_safe_output=base_safe,
                        candidate_true_safe_output=candidate_safe,
                        base_safe_accept=bool(base_accept and base_safe),
                        candidate_safe_accept=bool(
                            candidate_accept and candidate_safe
                        ),
                        base_false_safe=bool(base_accept and not base_safe),
                        candidate_false_safe=bool(
                            candidate_accept and not candidate_safe
                        ),
                        repaired_base_false_safe=bool(
                            base_accept
                            and not base_safe
                            and candidate_accept
                            and candidate_safe
                        ),
                        base=base_metrics,
                        candidate=candidate_metrics,
                        b5=b5_metrics,
                        m1=m1_metrics,
                        m1_point_submerged=m1_metrics is None,
                        candidate_fscore_wins_b5=(
                            candidate_metrics.fscore > b5_metrics.fscore
                        ),
                        candidate_fscore_wins_m1=(
                            None
                            if m1_metrics is None
                            else candidate_metrics.fscore > m1_metrics.fscore
                        ),
                    )
                )

    subgroups = tuple(
        _subgroup(cases, point_count, stress)
        for point_count in selected_counts
        for stress in selected_stresses
    )
    case_count = len(cases)
    expected_case_count = len(POINT_COUNTS) * len(STRESSES) * REPEATS
    m1_rows = [case.m1 for case in cases if case.m1 is not None]
    m1_available = len(m1_rows)
    base_safe_accepts = sum(case.base_safe_accept for case in cases)
    base_false_safe = sum(case.base_false_safe for case in cases)
    candidate_safe_accepts = sum(case.candidate_safe_accept for case in cases)
    candidate_false_safe = sum(case.candidate_false_safe for case in cases)
    repaired = sum(case.repaired_base_false_safe for case in cases)
    coverage = candidate_safe_accepts / case_count

    candidate_mean_fscore = _mean([case.candidate.fscore for case in cases])
    b5_mean_fscore = _mean([case.b5.fscore for case in cases])
    m1_mean_fscore = (
        _mean([metrics.fscore for metrics in m1_rows])
        if m1_available == case_count
        else None
    )
    candidate_geometry = _mean(
        [case.candidate.geometry_loss for case in cases]
    )
    b5_geometry = _mean([case.b5.geometry_loss for case in cases])
    m1_geometry = (
        _mean([metrics.geometry_loss for metrics in m1_rows])
        if m1_available == case_count
        else None
    )
    b5_win_rate = sum(case.candidate_fscore_wins_b5 for case in cases) / case_count
    m1_win_rate = (
        sum(bool(case.candidate_fscore_wins_m1) for case in cases) / case_count
        if m1_available == case_count
        else None
    )
    candidate_topology = sum(case.candidate.topology_error for case in cases)
    b5_topology = sum(case.b5.topology_error for case in cases)
    m1_topology = (
        sum(metrics.topology_error for metrics in m1_rows)
        if m1_available == case_count
        else None
    )
    candidate_nonmanifold = sum(
        case.candidate.nonmanifold_edges for case in cases
    )

    configuration_matches = bool(
        protocol_file_matches
        and protocol_digest == PROTOCOL_SHA256
        and seed == HELD_OUT_SEED
        and selected_counts == POINT_COUNTS
        and tuple(stress.value for stress in selected_stresses) == STRESSES
        and repeats == REPEATS
        and reference_count == REFERENCE_COUNT
        and surface_sample_count == SURFACE_SAMPLE_COUNT
        and case_count == expected_case_count
    )
    safety_gate = bool(
        candidate_false_safe == 0
        and coverage >= MINIMUM_OVERALL_SAFE_ACCEPTANCE
        and all(row.subgroup_safety_gate_passed for row in subgroups)
    )
    efficacy_gate = bool(
        m1_mean_fscore is not None
        and m1_geometry is not None
        and m1_win_rate is not None
        and candidate_mean_fscore - b5_mean_fscore
        >= MINIMUM_MEAN_FSCORE_MARGIN
        and candidate_mean_fscore - m1_mean_fscore
        >= MINIMUM_MEAN_FSCORE_MARGIN
        and candidate_geometry < b5_geometry
        and candidate_geometry < m1_geometry
        and b5_win_rate >= MINIMUM_CASEWISE_FSCORE_WIN_RATE
        and m1_win_rate >= MINIMUM_CASEWISE_FSCORE_WIN_RATE
    )
    topology_gate = bool(
        m1_topology is not None
        and candidate_topology == 0
        and candidate_nonmanifold == 0
        and b5_topology > 0
        and m1_topology > 0
    )
    ablation_gate = bool(
        candidate_safe_accepts >= base_safe_accepts
        and repaired >= MINIMUM_REPAIRED_BASE_FALSE_SAFE
    )
    phase50_supported = bool(
        configuration_matches
        and safety_gate
        and efficacy_gate
        and topology_gate
        and ablation_gate
    )
    return TwoLayerConfirmatoryResult(
        artifact_schema=RESULT_SCHEMA,
        role="untouched_bounded_positive_two_layer_confirmatory_test",
        protocol_commit=PROTOCOL_COMMIT,
        protocol_sha256=protocol_digest,
        protocol_identity_passed=configuration_matches,
        information_boundary=(
            "candidate routes use observed coordinates only; stress identity, "
            "true labels, dense reference, and comparator endpoints are "
            "evaluation-only"
        ),
        seed=seed,
        point_counts=selected_counts,
        stresses=selected_stresses,
        repeats=repeats,
        reference_count=reference_count,
        surface_sample_count=surface_sample_count,
        k_neighbors=K_NEIGHBORS,
        b5_scale_multiplier=B5_SCALE_MULTIPLIER,
        b5_max_normal_penalty=B5_MAX_NORMAL_PENALTY,
        m1_weight_scale=M1_WEIGHT_SCALE,
        m1_scale_multiplier=M1_SCALE_MULTIPLIER,
        cases=tuple(cases),
        subgroups=subgroups,
        case_count=case_count,
        expected_case_count=expected_case_count,
        m1_available_case_count=m1_available,
        base_safe_accept_count=base_safe_accepts,
        base_false_safe_count=base_false_safe,
        candidate_safe_accept_count=candidate_safe_accepts,
        candidate_false_safe_count=candidate_false_safe,
        repaired_base_false_safe_count=repaired,
        candidate_safe_acceptance_coverage=coverage,
        candidate_mean_fscore=candidate_mean_fscore,
        b5_mean_fscore=b5_mean_fscore,
        m1_mean_fscore=m1_mean_fscore,
        candidate_b5_mean_fscore_margin=(
            candidate_mean_fscore - b5_mean_fscore
        ),
        candidate_m1_mean_fscore_margin=(
            None
            if m1_mean_fscore is None
            else candidate_mean_fscore - m1_mean_fscore
        ),
        candidate_mean_geometry_loss=candidate_geometry,
        b5_mean_geometry_loss=b5_geometry,
        m1_mean_geometry_loss=m1_geometry,
        candidate_b5_casewise_fscore_win_rate=b5_win_rate,
        candidate_m1_casewise_fscore_win_rate=m1_win_rate,
        candidate_topology_error_sum=candidate_topology,
        b5_topology_error_sum=b5_topology,
        m1_topology_error_sum=m1_topology,
        candidate_nonmanifold_edges_sum=candidate_nonmanifold,
        safety_gate_passed=safety_gate,
        efficacy_gate_passed=efficacy_gate,
        topology_gate_passed=topology_gate,
        ablation_gate_passed=ablation_gate,
        phase50_supported=phase50_supported,
        promotion_supported=False,
        pftf_superiority_supported=False,
        real_scan_supported=False,
        deployment_supported=False,
        claim_boundary=(
            "positive support is limited to sampling-sufficient, globally "
            "separable, non-outlier synthetic two-layer surfaces; no outlier, "
            "arbitrary-surface, PFTF/local-SPD, real-scan, exactness, or deployment "
            "claim follows"
        ),
    )


def write_result(result: TwoLayerConfirmatoryResult, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    output.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/two_layer_confirmatory_phase50.json"),
    )
    parser.add_argument("--protocol", type=Path, default=Path(
        "benchmark-out/two_layer_confirmatory_protocol_phase50.json"
    ))
    parser.add_argument("--point-counts", type=int, nargs="+", default=POINT_COUNTS)
    parser.add_argument("--stresses", nargs="+", default=STRESSES)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--reference", type=int, default=REFERENCE_COUNT)
    parser.add_argument("--surface-samples", type=int, default=SURFACE_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=HELD_OUT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_two_layer_confirmatory(
        point_counts=args.point_counts,
        stresses=args.stresses,
        repeats=args.repeats,
        reference_count=args.reference,
        surface_sample_count=args.surface_samples,
        seed=args.seed,
        protocol_path=args.protocol,
    )
    digest = write_result(result, args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(
        "phase50_supported="
        f"{str(result.phase50_supported).lower()} "
        f"safe={result.candidate_safe_accept_count}/{result.case_count} "
        f"false_safe={result.candidate_false_safe_count} "
        f"fscore={result.candidate_mean_fscore:.6f}/"
        f"{result.b5_mean_fscore:.6f}/{result.m1_mean_fscore}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
