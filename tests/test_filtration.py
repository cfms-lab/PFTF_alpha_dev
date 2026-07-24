import numpy as np
import pytest

from pftf_alpha import AlphaConvention, AlphaFiltration, BoundaryMode


def test_single_triangle_filtration_and_boundary_modes() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    filtration = AlphaFiltration.from_points(points)

    np.testing.assert_allclose(filtration.critical_values(), [0.25, 0.5])

    early = filtration.simplices_at(0.5, convention=AlphaConvention.RADIUS)
    assert early[0].shape == (3, 1)
    assert early[1].shape == (2, 2)
    assert early[2].shape == (0, 3)
    assert filtration.boundary_facets_at(0.25, mode=BoundaryMode.REGULARIZED).shape == (
        0,
        2,
    )
    assert filtration.boundary_facets_at(0.25, mode=BoundaryMode.GENERAL).shape == (
        2,
        2,
    )

    complete = filtration.simplices_at(0.5)
    assert complete[1].shape == (3, 2)
    assert complete[2].shape == (1, 3)
    assert filtration.boundary_facets_at(0.5).shape == (3, 2)

    statistics = filtration.statistics(0.5)
    assert statistics.simplex_counts == (3, 3, 1)
    assert statistics.connected_components == 1
    assert statistics.euler_characteristic == 1
    assert statistics.boundary_facets == 3


def test_square_regularized_boundary_excludes_internal_diagonal() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    filtration = AlphaFiltration.from_points(points)

    boundary = filtration.boundary_facets_at(0.5)
    boundary_edges = {tuple(edge) for edge in boundary.tolist()}
    assert boundary_edges == {(0, 1), (0, 3), (1, 2), (2, 3)}
    assert filtration.statistics(0.5).top_simplices == 2


def test_duplicate_and_rank_deficient_points_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        AlphaFiltration.from_points(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="span"):
        AlphaFiltration.from_points(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


@pytest.mark.parametrize("dimension, point_count", [(2, 12), (3, 16)])
def test_random_filtration_is_closed_and_scale_covariant(
    dimension: int, point_count: int
) -> None:
    random = np.random.default_rng(20260724 + dimension)
    points = random.normal(size=(point_count, dimension))
    filtration = AlphaFiltration.from_points(points)
    records = {record.vertices: record for record in filtration.records}

    for record in filtration.records:
        if record.dimension == 0:
            continue
        for face in (
            record.vertices[:index] + record.vertices[index + 1 :]
            for index in range(len(record.vertices))
        ):
            assert records[face].alpha_squared <= (record.alpha_squared + 1.0e-12)

    scaled = AlphaFiltration.from_points(3.0 * points + 7.0)
    np.testing.assert_allclose(
        scaled.critical_values(),
        9.0 * filtration.critical_values(),
        rtol=1.0e-9,
        atol=1.0e-10,
    )
