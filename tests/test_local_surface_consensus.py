import numpy as np

from pftf_alpha.local_surface_consensus import (
    LocalSurfaceConsensusConfig,
    estimate_local_surface_consensus,
    evaluate_geometry_topology_harm,
    evaluate_local_surface_consensus,
    local_tangent_plane_scores,
)
from pftf_alpha.sensor_stress import SensorStress, make_sensor_stress_case
from pftf_alpha.shared_trend_inference import infer_shared_trend_layers
from pftf_alpha.surface import SurfaceMesh


def _parallel_grid_with_outlier() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.linspace(-1.0, 1.0, 6)
    xy = np.asarray([(x, y) for x in coordinates for y in coordinates])
    lower = np.column_stack((xy, np.full(xy.shape[0], -0.4)))
    upper = np.column_stack((xy, np.full(xy.shape[0], 0.4)))
    outlier = np.asarray([[0.0, 0.0, 0.25]])
    points = np.vstack((lower, upper, outlier))
    labels = np.concatenate(
        (
            np.zeros(lower.shape[0] + 1, dtype=np.int64),
            np.ones(upper.shape[0], dtype=np.int64),
        )
    )
    points = np.vstack((lower, outlier, upper))
    return points, labels


def test_leave_one_out_score_flags_isolated_off_surface_point() -> None:
    points, labels = _parallel_grid_with_outlier()
    config = LocalSurfaceConsensusConfig(k_neighbors=8)
    scores = local_tangent_plane_scores(points, labels, config)
    outlier_index = 36
    assert scores.standardized_residuals[outlier_index] > (
        config.maximum_standardized_residual
    )
    evidence = estimate_local_surface_consensus(points, labels, config)
    assert evidence.flagged_point_count >= 1
    assert evidence.surface_consistent is False


def test_local_bump_is_not_rejected_by_global_shape_mismatch() -> None:
    case = make_sensor_stress_case(
        SensorStress.LOCAL_BUMP,
        160,
        reference_count=256,
        seed=17,
    )
    inference = infer_shared_trend_layers(case.points)
    evidence = estimate_local_surface_consensus(
        case.points,
        inference.inference.layer_ids,
    )
    assert evidence.surface_consistent


def test_harm_endpoint_separates_near_surface_provenance() -> None:
    base_vertices = np.asarray(
        [
            [0.0, 0.0, -0.4],
            [1.0, 0.0, -0.4],
            [0.0, 1.0, -0.4],
            [0.0, 0.0, 0.4],
            [1.0, 0.0, 0.4],
            [0.0, 1.0, 0.4],
        ]
    )
    faces = np.asarray([[0, 1, 6], [3, 4, 5]], dtype=np.int64)
    labels = np.asarray([0, 0, 0, 1, 1, 1, 2], dtype=np.int64)
    reference = base_vertices.copy()
    characteristic_length = float(np.linalg.norm(np.ptp(reference, axis=0)))

    near_mesh = SurfaceMesh(
        vertices=np.vstack((base_vertices, [[0.0, 1.0, -0.39]])),
        faces=faces,
    )
    near = evaluate_geometry_topology_harm(
        near_mesh,
        reference,
        labels,
        characteristic_length=characteristic_length,
    )
    assert near.provenance_violation_present
    assert near.harmful_outlier_vertex_count == 0
    assert near.geometry_topology_harm_present is False

    far_mesh = SurfaceMesh(
        vertices=np.vstack((base_vertices, [[0.0, 1.0, 0.0]])),
        faces=faces,
    )
    far = evaluate_geometry_topology_harm(
        far_mesh,
        reference,
        labels,
        characteristic_length=characteristic_length,
    )
    assert far.provenance_violation_present
    assert far.harmful_outlier_vertex_count == 1
    assert far.geometry_topology_harm_present


def test_phase10_smoke_cannot_promote_a_reduced_panel() -> None:
    result = evaluate_local_surface_consensus(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        seed=37,
        surface_sample_count=64,
    )
    assert result.case_count == 1
    assert result.phase10_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.real_scan_supported is False
    assert result.deployment_supported is False
