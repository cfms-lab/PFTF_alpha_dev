import numpy as np
import pytest

from pftf_alpha.affine_spd_alpha import global_affine_spd_alpha
from pftf_alpha.integrable_spatial_alpha import (
    AffineCoordinateMap3D,
    NonIntegrableJacobianError,
    QuadraticShearMap3D,
    audit_jacobian_integrability,
    coordinate_map_spatial_alpha,
    numerical_jacobians,
    require_integrable_jacobian_field,
)


def _canonical(cells) -> set[tuple[int, ...]]:
    return {tuple(sorted(int(vertex) for vertex in cell)) for cell in cells}


def test_quadratic_shear_roundtrip_jacobian_and_metric_variation() -> None:
    points = np.random.default_rng(41).normal(size=(32, 3))
    coordinate_map = QuadraticShearMap3D(strength=0.2)
    construction = coordinate_map_spatial_alpha(points, coordinate_map)
    numerical = numerical_jacobians(
        coordinate_map,
        points,
        finite_difference_step=1.0e-6,
    )

    np.testing.assert_allclose(
        coordinate_map.inverse(coordinate_map.forward(points)), points
    )
    np.testing.assert_allclose(construction.jacobians, numerical, atol=1.0e-8)
    assert construction.minimum_jacobian_determinant == pytest.approx(1.0)
    assert construction.minimum_metric_eigenvalue > 0.0
    assert construction.maximum_relative_metric_variation > 0.0


def test_affine_coordinate_map_matches_phase46() -> None:
    points = np.random.default_rng(43).normal(size=(36, 3))
    factor = np.asarray(
        [[1.2, 0.0, 0.0], [0.1, 0.9, 0.0], [0.05, -0.1, 1.1]]
    )
    coordinate_map = AffineCoordinateMap3D(factor=factor, offset=np.ones(3))
    spatial = coordinate_map_spatial_alpha(points, coordinate_map)
    affine = global_affine_spd_alpha(points, factor @ factor.T)

    assert _canonical(spatial.filtration.top_simplices) == _canonical(
        affine.filtration.top_simplices
    )
    assert {
        record.vertices: record.alpha_squared for record in spatial.filtration.records
    } == pytest.approx(
        {
            record.vertices: record.alpha_squared
            for record in affine.filtration.records
        }
    )


def test_integrability_audit_accepts_shear_and_rejects_curl() -> None:
    points = np.random.default_rng(47).normal(size=(24, 3))
    shear = QuadraticShearMap3D(strength=0.2)

    def nonintegrable(point_array: np.ndarray) -> np.ndarray:
        result = np.repeat(np.eye(3)[None, :, :], len(point_array), axis=0)
        result[:, 0, 1] = 0.35 * point_array[:, 1]
        return result

    accepted = audit_jacobian_integrability(
        shear.jacobians,
        points,
        finite_difference_step=1.0e-6,
        absolute_tolerance=1.0e-8,
        minimum_jacobian_determinant=0.5,
    )
    rejected = audit_jacobian_integrability(
        nonintegrable,
        points,
        finite_difference_step=1.0e-6,
        absolute_tolerance=1.0e-8,
        minimum_jacobian_determinant=0.5,
    )

    assert accepted.compatible
    assert accepted.maximum_mixed_partial_residual <= 1.0e-8
    assert not rejected.compatible
    assert rejected.maximum_mixed_partial_residual == pytest.approx(0.35)
    with pytest.raises(NonIntegrableJacobianError, match="not locally compatible"):
        require_integrable_jacobian_field(
            nonintegrable,
            points,
            finite_difference_step=1.0e-6,
            absolute_tolerance=1.0e-8,
            minimum_jacobian_determinant=0.5,
        )


def test_affine_map_rejects_nonpositive_determinant() -> None:
    with pytest.raises(ValueError, match="positive determinant"):
        AffineCoordinateMap3D(
            factor=np.diag([-1.0, 1.0, 1.0]),
            offset=np.zeros(3),
        )
