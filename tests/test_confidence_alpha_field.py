import numpy as np
import pytest

from pftf_alpha.adaptive import density_scaled_filtration
from pftf_alpha.confidence_alpha_field import (
    binary_confidence_subset,
    confidence_weighted_filtration,
    observed_point_confidence,
)
from pftf_alpha.confidence_alpha_panel import (
    MisregistrationProfile,
    ReferenceSurfaceFamily,
    make_confidence_alpha_case,
)
from pftf_alpha.filtration import AlphaFiltration
from pftf_alpha.synthetic import PanelSplit


def _case(profile: MisregistrationProfile = MisregistrationProfile.MILD):
    return make_confidence_alpha_case(
        ReferenceSurfaceFamily.TORUS,
        profile,
        split=PanelSplit.CALIBRATION,
        seed=43_001,
    )


def test_observed_confidence_is_deterministic_and_bounded() -> None:
    case = _case()
    result = observed_point_confidence(case.anchor_points, case.target_points)
    repeated = observed_point_confidence(case.anchor_points, case.target_points)

    assert result.point_confidence.shape == (144,)
    np.testing.assert_array_equal(result.point_confidence[:72], 1.0)
    assert np.all((result.target_confidence >= 0.0) & (result.target_confidence <= 1.0))
    np.testing.assert_array_equal(result.target_confidence, repeated.target_confidence)


def test_continuous_zero_strength_is_b4_and_penalty_is_monotone() -> None:
    case = _case()
    filtration = AlphaFiltration.from_points(case.points)
    confidence = observed_point_confidence(
        case.anchor_points, case.target_points
    ).point_confidence
    b4 = density_scaled_filtration(filtration, k_neighbors=12)
    zero = confidence_weighted_filtration(
        filtration, confidence, k_neighbors=12, penalty_strength=0.0
    )
    positive = confidence_weighted_filtration(
        filtration, confidence, k_neighbors=12, penalty_strength=2.0
    )

    np.testing.assert_array_equal(zero.scores, b4.scores)
    assert np.all(positive.scores >= b4.scores)
    assert np.any(positive.scores > b4.scores)
    assert np.all((positive.cell_confidence >= 0.0) & (positive.cell_confidence <= 1.0))


def test_all_confident_points_leave_scores_unchanged() -> None:
    case = _case()
    filtration = AlphaFiltration.from_points(case.points)
    b4 = density_scaled_filtration(filtration, k_neighbors=12)
    weighted = confidence_weighted_filtration(
        filtration,
        np.ones(case.points.shape[0]),
        k_neighbors=12,
        penalty_strength=4.0,
    )

    np.testing.assert_array_equal(weighted.scores, b4.scores)


def test_binary_comparator_returns_retention_mask() -> None:
    case = _case()
    confidence = observed_point_confidence(
        case.anchor_points, case.target_points
    ).target_confidence
    points, retained = binary_confidence_subset(
        case.anchor_points,
        case.target_points,
        confidence,
        threshold=0.5,
    )

    assert retained.shape == (72,)
    assert points.shape[0] == 72 + np.count_nonzero(retained)
    np.testing.assert_array_equal(points[:72], case.anchor_points)


def test_confidence_rejects_reference_sized_input() -> None:
    case = _case(MisregistrationProfile.COHERENT)
    with pytest.raises(ValueError, match="point_confidence"):
        confidence_weighted_filtration(
            AlphaFiltration.from_points(case.points),
            np.ones(case.reference_points.shape[0]),
            penalty_strength=1.0,
        )
