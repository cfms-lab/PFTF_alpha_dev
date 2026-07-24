import numpy as np

from pftf_alpha.adaptive import (
    density_scaled_filtration,
    local_neighborhood_geometry,
    pca_anisotropic_filtration,
)
from pftf_alpha.filtration import AlphaFiltration
from pftf_alpha.surface import mesh_statistics


def _random_points(seed: int = 10) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(24, 3))


def test_density_scaled_scores_are_scale_and_rotation_invariant() -> None:
    points = _random_points()
    rotation, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))
    transformed = 3.5 * points @ rotation + np.array([4.0, -2.0, 1.0])

    original = density_scaled_filtration(
        AlphaFiltration.from_points(points), k_neighbors=6
    )
    moved = density_scaled_filtration(
        AlphaFiltration.from_points(transformed), k_neighbors=6
    )

    np.testing.assert_allclose(
        np.sort(original.scores),
        np.sort(moved.scores),
        rtol=1.0e-9,
        atol=1.0e-10,
    )


def test_pca_anisotropic_scores_are_scale_and_rotation_invariant() -> None:
    points = _random_points(seed=20)
    rotation, _ = np.linalg.qr(np.random.default_rng(21).normal(size=(3, 3)))
    transformed = 2.25 * points @ rotation - np.array([1.0, 3.0, -2.0])

    original = pca_anisotropic_filtration(
        AlphaFiltration.from_points(points),
        k_neighbors=7,
        max_normal_penalty=4.0,
    )
    moved = pca_anisotropic_filtration(
        AlphaFiltration.from_points(transformed),
        k_neighbors=7,
        max_normal_penalty=4.0,
    )

    np.testing.assert_allclose(
        np.sort(original.scores),
        np.sort(moved.scores),
        rtol=1.0e-8,
        atol=1.0e-9,
    )


def test_planar_neighborhood_has_high_pca_planarity() -> None:
    grid_x, grid_y = np.meshgrid(
        np.linspace(-1.0, 1.0, 5),
        np.linspace(-1.0, 1.0, 5),
    )
    points = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)))
    geometry = local_neighborhood_geometry(points, k_neighbors=8)

    assert geometry.planarity[12] > 0.8
    assert np.all(geometry.scales > 0.0)


def test_adaptive_surface_is_boundary_of_selected_top_cell_closure() -> None:
    adaptive = density_scaled_filtration(
        AlphaFiltration.from_points(_random_points(seed=30)),
        k_neighbors=6,
    )
    threshold = float(np.median(adaptive.scores))
    mesh = adaptive.surface_at(threshold)
    statistics = mesh_statistics(mesh)

    assert 0 < adaptive.selected_cell_count(threshold) < len(adaptive.scores)
    assert statistics.faces > 0
    assert statistics.nonmanifold_edges >= 0
