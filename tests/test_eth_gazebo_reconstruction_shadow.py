import numpy as np

from pftf_alpha.eth_gazebo_reconstruction_shadow import (
    ReconstructionEndpoint,
    ReconstructionShadowCase,
    _split_source_points,
    _summarize,
    _transform_and_crop,
)


def _endpoint(
    *,
    geometry: float,
    fscore: float,
    recall: float,
    components: int = 2,
    betti_1: int = 3,
    betti_2: int = 1,
    nonmanifold_fraction: float = 0.02,
) -> ReconstructionEndpoint:
    return ReconstructionEndpoint(
        normalized_chamfer_squared=0.01,
        normalized_hausdorff=geometry - 0.01,
        geometry_loss=geometry,
        precision=fscore,
        recall=recall,
        fscore=fscore,
        used_vertices=10,
        edges=20,
        faces=12,
        connected_components=components,
        betti_0=components,
        betti_1=betti_1,
        betti_2=betti_2,
        euler_characteristic=0,
        boundary_edges=0,
        nonmanifold_edges=1,
        nonmanifold_edge_fraction=nonmanifold_fraction,
        watertight=False,
    )


def _case(source: int, baseline: ReconstructionEndpoint, guard: ReconstructionEndpoint):
    return ReconstructionShadowCase(
        source_index=source,
        pair_count=4,
        accepted_pair_count=3,
        rejected_pair_count=1,
        observed_source_point_count=100,
        heldout_reference_point_count=25,
        baseline_fused_point_count=200,
        guard_fused_point_count=175,
        characteristic_length_meters=10.0,
        baseline=baseline,
        guard=guard,
        geometry_loss_margin=baseline.geometry_loss - guard.geometry_loss,
        fscore_margin=guard.fscore - baseline.fscore,
        recall_margin=guard.recall - baseline.recall,
        component_count_reduction=(
            baseline.connected_components - guard.connected_components
        ),
        betti_1_reduction=baseline.betti_1 - guard.betti_1,
        betti_2_reduction=baseline.betti_2 - guard.betti_2,
        nonmanifold_fraction_reduction=(
            baseline.nonmanifold_edge_fraction
            - guard.nonmanifold_edge_fraction
        ),
    )


def test_phase40_source_split_is_disjoint_and_exhaustive() -> None:
    points = np.arange(45, dtype=float).reshape(15, 3)

    observed, heldout = _split_source_points(points)

    assert observed.shape == (12, 3)
    assert heldout.shape == (3, 3)
    assert np.array_equal(heldout, points[[0, 5, 10]])
    assert {tuple(row) for row in observed}.isdisjoint(
        {tuple(row) for row in heldout}
    )


def test_phase40_transform_and_crop_uses_target_to_source_matrix() -> None:
    points = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    matrix = np.eye(4)
    matrix[0, 3] = 1.0

    result = _transform_and_crop(
        points,
        matrix,
        np.array((0.5, -1.0, -1.0)),
        np.array((1.5, 1.0, 1.0)),
    )

    assert np.array_equal(result, np.array(((1.0, 0.0, 0.0),)))


def test_phase40_summary_applies_frozen_geometry_gates() -> None:
    baseline = _endpoint(geometry=0.20, fscore=0.40, recall=0.50)
    guard = _endpoint(geometry=0.18, fscore=0.42, recall=0.495)

    summary = _summarize((_case(2, baseline, guard), _case(3, baseline, guard)))

    assert summary.case_count == 2
    assert summary.rejected_pair_count == 2
    assert summary.geometry_win_count == 2
    assert summary.geometry_loss_improved is True
    assert summary.fscore_nonregressed is True
    assert summary.recall_nonregressed is True
    assert summary.geometry_shadow_supported is True


def test_phase40_topology_remains_descriptive_not_a_geometry_gate() -> None:
    baseline = _endpoint(geometry=0.20, fscore=0.40, recall=0.50, betti_1=2)
    guard = _endpoint(geometry=0.18, fscore=0.42, recall=0.50, betti_1=20)

    summary = _summarize((_case(2, baseline, guard),))

    assert summary.mean_guard_betti_1 == 20.0
    assert summary.geometry_shadow_supported is True
