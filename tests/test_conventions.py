import math

import pytest

from pftf_alpha import (
    AlphaConvention,
    alpha_to_squared_radius,
    squared_radius_to_alpha,
)


def test_alpha_conventions_are_explicit_and_invertible() -> None:
    assert alpha_to_squared_radius(2.0, AlphaConvention.RADIUS) == 4.0
    assert alpha_to_squared_radius(4.0) == 4.0
    assert squared_radius_to_alpha(4.0, AlphaConvention.RADIUS) == 2.0
    assert squared_radius_to_alpha(4.0) == 4.0


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_alpha_conventions_reject_invalid_values(value: float) -> None:
    with pytest.raises(ValueError):
        alpha_to_squared_radius(value)
