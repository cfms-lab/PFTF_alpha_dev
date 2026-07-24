import numpy as np
import pytest

from pftf_alpha.adaptive import (
    density_scaled_filtration,
    pftf_confidence_fallback_filtration,
    pftf_local_metric_filtration,
)
from pftf_alpha.filtration import AlphaFiltration
from pftf_alpha.pftf import directed_scale_contrast, pftf_relation_field


def _random_points(seed: int = 801) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(32, 3))


def _p1(points: np.ndarray):
    return pftf_local_metric_filtration(
        AlphaFiltration.from_points(points),
        k_neighbors=7,
        relation_gain=2.0,
        max_condition_number=9.0,
        density_contrast_scale=0.5,
        receiver_imbalance_weight=0.5,
    )


def test_directed_scale_contrast_reverses_on_edge_reversal() -> None:
    forward = directed_scale_contrast(
        1.0,
        [2.0],
        contrast_scale=0.5,
    )[0]
    reverse = directed_scale_contrast(
        2.0,
        [1.0],
        contrast_scale=0.5,
    )[0]

    assert forward == pytest.approx(-reverse)
    assert forward > 0.0


def test_relation_field_is_spd_bounded_and_trace_free() -> None:
    relation = pftf_relation_field(
        _random_points(),
        k_neighbors=7,
        relation_gain=2.0,
        max_condition_number=9.0,
        density_contrast_scale=0.5,
        receiver_imbalance_weight=0.5,
    )

    normalized = relation.metric_field.matrices * relation.scales[:, None, None] ** 2
    eigenvalues = np.linalg.eigvalsh(normalized)
    condition = eigenvalues[:, -1] / eigenvalues[:, 0]

    assert np.all(eigenvalues > 0.0)
    assert np.max(condition) <= 9.0 + 1.0e-10
    assert np.all(relation.metric_field.confidence >= 0.0)
    assert np.all(relation.metric_field.confidence <= 1.0)
    np.testing.assert_allclose(
        np.trace(relation.relation_tensors, axis1=1, axis2=2),
        0.0,
        atol=1.0e-12,
    )
    assert np.max(relation.relation_strength) > 0.0


def test_p1_scores_are_scale_and_rotation_invariant() -> None:
    points = _random_points(seed=811)
    rotation, _ = np.linalg.qr(np.random.default_rng(812).normal(size=(3, 3)))
    transformed = 4.25 * points @ rotation + np.array([5.0, -2.0, 3.0])

    original = _p1(points)
    moved = _p1(transformed)

    np.testing.assert_allclose(
        np.sort(original.scores),
        np.sort(moved.scores),
        rtol=1.0e-8,
        atol=1.0e-9,
    )
    assert original.diagnostics["metric_condition_max"] <= 9.0 + 1.0e-10
    assert original.diagnostics["fallback_fraction"] == 0.0


def test_p1_relation_scores_are_not_the_b4_density_scores() -> None:
    filtration = AlphaFiltration.from_points(_random_points(seed=821))
    density = density_scaled_filtration(filtration, k_neighbors=7)
    relation = pftf_local_metric_filtration(
        filtration,
        k_neighbors=7,
        relation_gain=2.0,
        max_condition_number=9.0,
        density_contrast_scale=0.5,
        receiver_imbalance_weight=0.5,
    )

    assert not np.allclose(density.scores, relation.scores)
    assert relation.method == "P1_pftf_local_spd"


def test_p2_low_confidence_cells_pass_p1_and_trusted_b4_guards() -> None:
    filtration = AlphaFiltration.from_points(_random_points(seed=831))
    p1 = pftf_local_metric_filtration(
        filtration,
        k_neighbors=7,
        relation_gain=2.0,
        max_condition_number=9.0,
        density_contrast_scale=0.5,
        receiver_imbalance_weight=0.5,
    )
    trusted = density_scaled_filtration(filtration, k_neighbors=7)
    assert p1.cell_confidence is not None
    confidence_threshold = float(np.median(p1.cell_confidence))
    p2 = pftf_confidence_fallback_filtration(
        filtration,
        k_neighbors=7,
        relation_gain=2.0,
        max_condition_number=9.0,
        density_contrast_scale=0.5,
        receiver_imbalance_weight=0.5,
        confidence_threshold=confidence_threshold,
    )

    assert p2.fallback_mask is not None
    assert p2.guard_scores is not None
    np.testing.assert_allclose(p2.guard_scores, trusted.scores)
    low_confidence = p2.fallback_mask
    assert np.any(low_confidence)
    assert np.any(~low_confidence)
    assert np.all(p2.scores[low_confidence] >= p1.scores[low_confidence])
    assert np.all(p2.scores[low_confidence] >= trusted.scores[low_confidence])
    np.testing.assert_allclose(
        p2.scores[~low_confidence],
        p1.scores[~low_confidence],
    )
    assert p2.diagnostics["fallback_guard_violation_count"] == 0.0

    threshold = float(np.median(p2.scores))
    diagnostics = p2.diagnostics_at(threshold)
    selected = p2.scores <= threshold
    expected_fallback = np.count_nonzero(low_confidence & selected)
    assert diagnostics["selected_cell_count"] == np.count_nonzero(selected)
    assert diagnostics["selected_fallback_cell_count"] == expected_fallback
    assert diagnostics["selected_fallback_fraction"] == pytest.approx(
        expected_fallback / np.count_nonzero(selected)
    )

    assert diagnostics["selected_guard_violation_count"] == 0.0
    assert diagnostics["selected_guard_violation_fraction"] == 0.0
    assert diagnostics["downward_closure_complete"] == 1.0
    assert diagnostics["face_incidence_over_two_count"] == 0.0
    assert diagnostics["boundary_face_count"] == p2.surface_at(threshold).faces.shape[0]


def test_p2_scores_are_scale_and_rotation_invariant() -> None:
    points = _random_points(seed=841)
    rotation, _ = np.linalg.qr(np.random.default_rng(842).normal(size=(3, 3)))
    transformed = 3.75 * points @ rotation - np.array([2.0, 4.0, -1.0])

    def build(point_array: np.ndarray):
        return pftf_confidence_fallback_filtration(
            AlphaFiltration.from_points(point_array),
            k_neighbors=7,
            relation_gain=2.0,
            max_condition_number=9.0,
            density_contrast_scale=0.5,
            receiver_imbalance_weight=0.5,
            confidence_threshold=0.5,
        )

    original = build(points)
    moved = build(transformed)

    np.testing.assert_allclose(
        np.sort(original.scores),
        np.sort(moved.scores),
        rtol=1.0e-8,
        atol=1.0e-9,
    )
    assert original.method == "P2_pftf_confidence_b4_guard"
