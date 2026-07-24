import numpy as np
import pytest

from pftf_alpha.surface import (
    SurfaceMesh,
    convex_hull_surface,
    evaluate_surface,
    mesh_statistics,
    sample_triangle_mesh,
    surface_distance_metrics,
)


def _tetrahedron() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_tetrahedron_hull_is_watertight_with_euler_two() -> None:
    mesh = convex_hull_surface(_tetrahedron())
    statistics = mesh_statistics(mesh)

    assert statistics.used_vertices == 4
    assert statistics.edges == 6
    assert statistics.faces == 4
    assert statistics.connected_components == 1
    assert statistics.euler_characteristic == 2
    assert (statistics.betti_0, statistics.betti_1, statistics.betti_2) == (1, 0, 1)
    assert statistics.euler_characteristic == (
        statistics.betti_0 - statistics.betti_1 + statistics.betti_2
    )
    assert statistics.boundary_edges == 0
    assert statistics.nonmanifold_edges == 0
    assert statistics.watertight


def test_triangle_sampling_and_self_distance_are_deterministic() -> None:
    mesh = SurfaceMesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )
    first = sample_triangle_mesh(mesh, 128, seed=10)
    second = sample_triangle_mesh(mesh, 128, seed=10)
    np.testing.assert_array_equal(first, second)

    distances = surface_distance_metrics(first, first, threshold=1.0e-9)
    assert distances.chamfer_squared == pytest.approx(0.0)
    assert distances.hausdorff == pytest.approx(0.0)
    assert distances.fscore == pytest.approx(1.0)


def test_open_annulus_has_one_first_homology_generator() -> None:
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-0.4, -0.4, 0.0],
            [0.4, -0.4, 0.0],
            [0.4, 0.4, 0.0],
            [-0.4, 0.4, 0.0],
        ]
    )
    faces = []
    for index in range(4):
        next_index = (index + 1) % 4
        faces.append([index, next_index, 4 + next_index])
        faces.append([index, 4 + next_index, 4 + index])
    statistics = mesh_statistics(
        SurfaceMesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))
    )

    assert (statistics.betti_0, statistics.betti_1, statistics.betti_2) == (1, 1, 0)
    assert statistics.euler_characteristic == 0
    assert statistics.boundary_edges == 8
    assert not statistics.watertight


def test_surface_rejects_duplicate_unoriented_faces() -> None:
    with pytest.raises(ValueError, match="unique as unoriented simplices"):
        SurfaceMesh(vertices=_tetrahedron(), faces=np.array([[0, 1, 2], [2, 1, 0]]))


def test_labeled_false_bridge_counts_cross_component_edges_and_faces() -> None:
    mesh = SurfaceMesh(
        vertices=_tetrahedron(),
        faces=np.array([[0, 1, 2], [1, 2, 3]]),
    )
    endpoints = evaluate_surface(
        mesh,
        _tetrahedron(),
        expected_components=2,
        vertex_component_labels=np.array([0, 0, 1, 1]),
        characteristic_length=2.0,
        sample_count=32,
        threshold_fraction=0.01,
        seed=2,
    )

    assert endpoints.false_bridges == 1
    assert endpoints.labeled_false_bridge_edges == 3
    assert endpoints.labeled_false_bridge_faces == 2
    assert endpoints.labeled_false_bridge_present


def test_labeled_false_bridge_rejects_invalid_label_shape() -> None:
    mesh = SurfaceMesh(
        vertices=_tetrahedron(),
        faces=np.array([[0, 1, 2]]),
    )
    with pytest.raises(ValueError, match="surface vertex count"):
        evaluate_surface(
            mesh,
            _tetrahedron(),
            expected_components=2,
            vertex_component_labels=np.array([0, 0, 1]),
            characteristic_length=2.0,
            sample_count=16,
            threshold_fraction=0.01,
            seed=3,
        )


def test_empty_surface_receives_finite_failure_penalty() -> None:
    mesh = SurfaceMesh(
        vertices=_tetrahedron(),
        faces=np.empty((0, 3), dtype=np.int64),
    )
    endpoints = evaluate_surface(
        mesh,
        _tetrahedron(),
        expected_components=1,
        expected_betti=(1, 0, 1),
        characteristic_length=2.0,
        sample_count=32,
        threshold_fraction=0.01,
        seed=1,
    )

    assert endpoints.normalized_chamfer_squared == pytest.approx(4.0)
    assert endpoints.normalized_hausdorff == pytest.approx(2.0)
    assert endpoints.fscore == 0.0
    assert endpoints.component_error == 1
    assert (endpoints.betti_0, endpoints.betti_1, endpoints.betti_2) == (0, 0, 0)
    assert (
        endpoints.expected_betti_0,
        endpoints.expected_betti_1,
        endpoints.expected_betti_2,
    ) == (1, 0, 1)
    assert endpoints.betti_error == 2
    assert endpoints.labeled_false_bridge_edges is None
    assert endpoints.labeled_false_bridge_faces is None
    assert endpoints.labeled_false_bridge_present is None
