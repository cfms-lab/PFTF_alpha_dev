import numpy as np
import pytest

from pftf_alpha.confidence_alpha_field import observed_point_confidence
from pftf_alpha.confidence_power_alpha import (
    confidence_power_alpha_filtration,
    confidence_power_weights,
)
from pftf_alpha.confidence_power_alpha_protocol import (
    M1_DENSITY_WEIGHT_SCALE,
    make_confidence_power_alpha_panel,
)
from pftf_alpha.synthetic import PanelSplit
from pftf_alpha.weighted_alpha import weighted_alpha_filtration


def _canonical(cells) -> set[tuple[int, int, int, int]]:
    return {tuple(sorted(int(value) for value in cell)) for cell in cells}


def _case():
    return make_confidence_power_alpha_panel(PanelSplit.CALIBRATION)[0]


def test_zero_penalty_is_exact_m1_density_construction() -> None:
    case = _case()
    confidence = observed_point_confidence(
        case.anchor_points, case.target_points
    ).point_confidence
    candidate = confidence_power_alpha_filtration(
        case.points,
        confidence,
        k_neighbors=12,
        density_weight_scale=M1_DENSITY_WEIGHT_SCALE,
        confidence_penalty_scale=0.0,
    )
    m1 = weighted_alpha_filtration(
        case.points,
        k_neighbors=12,
        weight_scale=M1_DENSITY_WEIGHT_SCALE,
    )

    assert _canonical(candidate.top_simplices) == _canonical(m1.top_simplices)
    np.testing.assert_allclose(np.sort(candidate.scores), np.sort(m1.scores))


def test_full_confidence_removes_penalty_effect() -> None:
    case = _case()
    candidate = confidence_power_alpha_filtration(
        case.points,
        np.ones(case.points.shape[0]),
        k_neighbors=12,
        density_weight_scale=M1_DENSITY_WEIGHT_SCALE,
        confidence_penalty_scale=0.5,
    )
    m1 = weighted_alpha_filtration(
        case.points,
        k_neighbors=12,
        weight_scale=M1_DENSITY_WEIGHT_SCALE,
    )

    assert _canonical(candidate.top_simplices) == _canonical(m1.top_simplices)
    np.testing.assert_allclose(np.sort(candidate.scores), np.sort(m1.scores))


def test_lower_confidence_continuously_lowers_power_weight() -> None:
    points = np.random.default_rng(7).normal(size=(24, 3))
    high, _ = confidence_power_weights(
        points,
        np.ones(24),
        k_neighbors=6,
        density_weight_scale=0.375,
        confidence_penalty_scale=0.25,
    )
    confidence = np.ones(24)
    confidence[3] = 0.0
    low, _ = confidence_power_weights(
        points,
        confidence,
        k_neighbors=6,
        density_weight_scale=0.375,
        confidence_penalty_scale=0.25,
    )

    assert low[3] < high[3]
    np.testing.assert_array_equal(low[np.arange(24) != 3], high[np.arange(24) != 3])


def test_confidence_power_rejects_mismatched_confidence() -> None:
    case = _case()
    with pytest.raises(ValueError, match="point_confidence"):
        confidence_power_weights(
            case.points,
            np.ones(4),
            k_neighbors=12,
            density_weight_scale=0.375,
            confidence_penalty_scale=0.25,
        )
