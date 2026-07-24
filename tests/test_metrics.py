import numpy as np
import pytest

from pftf_alpha import (
    LocalMetricField,
    hard_alpha_gate,
    metric_circumradius_squared,
    soft_alpha_gate,
)


def test_factor_parameterization_produces_spd_metrics() -> None:
    factors = np.array(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 3.0]],
        ]
    )
    field = LocalMetricField.from_factors(
        factors, confidence=[0.9, 0.8], epsilon=1.0e-6
    )

    assert np.all(np.linalg.eigvalsh(field.matrices) > 0.0)
    assert field.squared_distance(0, [1.0, 0.0]) == pytest.approx(4.000001)


def test_low_confidence_simplex_fails_closed() -> None:
    field = LocalMetricField(
        matrices=np.array([np.eye(2), 2.0 * np.eye(2), 3.0 * np.eye(2)]),
        confidence=np.array([0.9, 0.2, 0.8]),
    )
    decision = field.metric_for_simplex(
        [0, 1, 2],
        confidence_threshold=0.5,
        fallback_metric=4.0 * np.eye(2),
    )

    assert decision.used_fallback
    assert decision.confidence == pytest.approx(0.2)
    assert decision.reason == "confidence_below_threshold"
    np.testing.assert_allclose(decision.metric, 4.0 * np.eye(2))


def test_confident_simplex_uses_weighted_spd_average() -> None:
    field = LocalMetricField(
        matrices=np.array([np.eye(2), 3.0 * np.eye(2)]),
        confidence=np.array([1.0, 0.5]),
    )
    decision = field.metric_for_simplex([0, 1], confidence_threshold=0.5)

    assert not decision.used_fallback
    np.testing.assert_allclose(
        decision.metric, (1.0 * np.eye(2) + 0.5 * 3.0 * np.eye(2)) / 1.5
    )


def test_metric_circumradius_matches_coordinate_scaling() -> None:
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    assert metric_circumradius_squared(triangle, np.eye(2)) == pytest.approx(0.5)
    assert metric_circumradius_squared(triangle, 4.0 * np.eye(2)) == pytest.approx(2.0)


def test_soft_gate_and_hard_limit_direction() -> None:
    assert soft_alpha_gate(0.5, 0.5, 0.1) == pytest.approx(0.5)
    assert soft_alpha_gate(1.0, 0.5, 0.01) > 1.0 - 1.0e-12
    assert soft_alpha_gate(0.5, 1.0, 0.01) < 1.0e-12
    assert hard_alpha_gate(0.5, 0.5)
    assert not hard_alpha_gate(0.49, 0.5)


def test_asymmetric_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        LocalMetricField(
            matrices=np.array([[[1.0, 1.0], [0.0, 1.0]]]),
            confidence=np.array([1.0]),
        )
