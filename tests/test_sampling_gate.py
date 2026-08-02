import numpy as np

from pftf_alpha.sampling_gate import (
    SamplingGateDecision,
    SamplingSufficiencyConfig,
    estimate_sampling_sufficiency,
    evaluate_sampling_sufficiency_gate,
    route_sampling_gate,
)


def _parallel_sheets(count: int, gap: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 2
    xy = rng.uniform(-1.0, 1.0, size=(count, 2))
    z = np.where(labels == 0, -0.5 * gap, 0.5 * gap)
    return np.column_stack((xy, z)) + rng.normal(scale=0.002, size=(count, 3))


def test_sampling_estimator_distinguishes_sparse_and_resolved_layers() -> None:
    config = SamplingSufficiencyConfig(k_neighbors=8)
    sparse = estimate_sampling_sufficiency(_parallel_sheets(32, 0.1, 3), config)
    resolved = estimate_sampling_sufficiency(_parallel_sheets(256, 0.5, 3), config)
    assert sparse.two_layer_identifiable
    assert not sparse.sampling_sufficient
    assert resolved.sampling_sufficient
    assert resolved.estimated_cross_knn_fraction < sparse.estimated_cross_knn_fraction


def test_router_separates_rescan_and_algorithm_failure() -> None:
    config = SamplingSufficiencyConfig(k_neighbors=8)
    sparse = estimate_sampling_sufficiency(_parallel_sheets(32, 0.1, 7), config)
    resolved = estimate_sampling_sufficiency(_parallel_sheets(256, 0.5, 7), config)
    assert route_sampling_gate(
        sparse,
        flagged_boundary_faces=3,
        flagged_boundary_edges=2,
    ) is SamplingGateDecision.RESCAN_REQUIRED
    assert route_sampling_gate(
        resolved,
        flagged_boundary_faces=3,
        flagged_boundary_edges=2,
    ) is SamplingGateDecision.ALGORITHM_FAILURE
    assert route_sampling_gate(
        resolved,
        flagged_boundary_faces=0,
        flagged_boundary_edges=0,
    ) is SamplingGateDecision.ACCEPT


def test_phase1_smoke_records_no_silent_false_safe() -> None:
    result = evaluate_sampling_sufficiency_gate(
        point_count=48,
        reference_count=128,
        gaps=(0.18, 1.20),
        repeats=1,
        seed=31,
        surface_sample_count=64,
        gate_config=SamplingSufficiencyConfig(k_neighbors=8),
    )
    assert result.artifact_schema.endswith("/v1")
    assert len(result.cases) == 2
    assert result.false_safe_count == 0
    assert result.deployment_supported is False
