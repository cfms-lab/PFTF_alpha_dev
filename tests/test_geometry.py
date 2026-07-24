import numpy as np
import pytest

from pftf_alpha import intrinsic_circumsphere


def test_right_triangle_circumcircle() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sphere = intrinsic_circumsphere(points)

    np.testing.assert_allclose(sphere.center, [0.5, 0.5])
    assert sphere.radius_squared == pytest.approx(0.5)


def test_right_tetrahedron_circumsphere() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    sphere = intrinsic_circumsphere(points)

    np.testing.assert_allclose(sphere.center, [0.5, 0.5, 0.5])
    assert sphere.radius_squared == pytest.approx(0.75)


def test_degenerate_simplex_is_rejected() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        intrinsic_circumsphere(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
