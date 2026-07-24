"""Explicit alpha-parameter conventions.

The project uses squared physical radius internally, matching CGAL's 3D alpha
shape convention.  Conversions happen only at public boundaries so a radius is
never silently compared with a squared radius.
"""

from __future__ import annotations

import math
from enum import StrEnum


class AlphaConvention(StrEnum):
    """Supported external meanings of an alpha value."""

    SQUARED_RADIUS = "squared_radius"
    RADIUS = "radius"


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return result


def alpha_to_squared_radius(
    alpha: float,
    convention: AlphaConvention | str = AlphaConvention.SQUARED_RADIUS,
) -> float:
    """Convert an external alpha value to the internal squared-radius form."""

    value = _finite_nonnegative(alpha, name="alpha")
    selected = AlphaConvention(convention)
    if selected is AlphaConvention.RADIUS:
        return value * value
    return value


def squared_radius_to_alpha(
    radius_squared: float,
    convention: AlphaConvention | str = AlphaConvention.SQUARED_RADIUS,
) -> float:
    """Convert an internal squared radius to an explicitly selected convention."""

    value = _finite_nonnegative(radius_squared, name="radius_squared")
    selected = AlphaConvention(convention)
    if selected is AlphaConvention.RADIUS:
        return math.sqrt(value)
    return value
