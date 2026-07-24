import numpy as np

from pftf_alpha.adaptive import (
    AdaptiveCellFiltration,
    boundary_bridge_localization,
    boundary_region_cut_intervention,
    boundary_risk_region_analysis,
    bridge_penalized_filtration,
    density_scaled_filtration,
    geometric_bridge_risk,
    iterative_boundary_owner_intervention,
    local_neighborhood_geometry,
    pca_anisotropic_filtration,
    pftf_confidence_fallback_filtration,
)
from pftf_alpha.filtration import AlphaFiltration
from pftf_alpha.surface import mesh_statistics
from pftf_alpha.synthetic import PanelSplit, SyntheticFamily, make_synthetic_case


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


def test_bridge_risk_routes_parallel_sheets_by_normal_coherence() -> None:
    case = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.HELD_OUT,
        point_count=48,
        reference_count=96,
        seed=20_270_731,
    )
    filtration = AlphaFiltration.from_points(case.points)

    probe = geometric_bridge_risk(filtration, k_neighbors=12)
    repeated = geometric_bridge_risk(filtration, k_neighbors=12)

    assert probe.route == "parallel_normal"
    assert probe.normal_coherence > probe.normal_coherence_threshold
    assert probe.risk.shape == (filtration.top_simplices.shape[0],)
    assert np.all(np.isfinite(probe.risk))
    np.testing.assert_array_equal(probe.risk, repeated.risk)


def test_bridge_risk_routes_disconnected_parts_by_normalized_length() -> None:
    case = make_synthetic_case(
        SyntheticFamily.DISCONNECTED_PARTS,
        split=PanelSplit.HELD_OUT,
        point_count=48,
        reference_count=96,
        seed=20_300_745,
    )
    filtration = AlphaFiltration.from_points(case.points)

    probe = geometric_bridge_risk(filtration, k_neighbors=12)

    assert probe.route == "long_edge"
    assert probe.normal_coherence < probe.normal_coherence_threshold
    assert np.any(probe.risk > 1.0)


def test_bridge_risk_is_scale_and_rotation_invariant() -> None:
    points = _random_points(seed=40)
    rotation, _ = np.linalg.qr(np.random.default_rng(41).normal(size=(3, 3)))
    transformed = 2.75 * points @ rotation + np.array([2.0, -5.0, 1.5])

    original = geometric_bridge_risk(AlphaFiltration.from_points(points), k_neighbors=7)
    moved = geometric_bridge_risk(
        AlphaFiltration.from_points(transformed), k_neighbors=7
    )

    assert original.route == moved.route
    np.testing.assert_allclose(
        original.normal_coherence, moved.normal_coherence, rtol=1.0e-10, atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.sort(original.risk),
        np.sort(moved.risk),
        rtol=1.0e-8,
        atol=1.0e-9,
    )


def test_bridge_penalty_zero_is_exact_p2_and_positive_is_monotone() -> None:
    case = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.HELD_OUT,
        point_count=48,
        reference_count=96,
        seed=20_270_731,
    )
    filtration = AlphaFiltration.from_points(case.points)
    base = pftf_confidence_fallback_filtration(
        filtration,
        k_neighbors=12,
        relation_gain=2.0,
        max_condition_number=9.0,
        density_contrast_scale=0.5,
        receiver_imbalance_weight=0.5,
        confidence_threshold=0.3,
    )
    risk = geometric_bridge_risk(filtration, k_neighbors=12)

    zero = bridge_penalized_filtration(base, risk, strength=0.0)
    positive = bridge_penalized_filtration(base, risk, strength=0.2)
    flagged = risk.risk > 1.0

    np.testing.assert_array_equal(zero.scores, base.scores)
    np.testing.assert_array_equal(
        zero.surface_at(1.2).faces, base.surface_at(1.2).faces
    )
    assert np.all(positive.scores >= base.scores)
    np.testing.assert_array_equal(positive.scores[~flagged], base.scores[~flagged])
    assert np.any(positive.scores[flagged] > base.scores[flagged])
    assert positive.diagnostics["bridge_penalty_strength"] == 0.2
    assert positive.diagnostics["bridge_penalty_changed_fraction"] > 0.0


def test_boundary_bridge_localization_tracks_boundary_and_dual_cut_structure() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    base = AdaptiveCellFiltration(
        points=points,
        top_simplices=np.array([[0, 1, 2, 3], [0, 1, 2, 4]]),
        scores=np.array([0.5, 0.5]),
        method="test",
        diagnostics={},
    )

    localization = boundary_bridge_localization(
        base,
        scale_multiplier=1.0,
        k_neighbors=3,
    )

    np.testing.assert_array_equal(
        localization.boundary_faces,
        base.surface_at(1.0).faces,
    )
    assert localization.boundary_faces.shape == (6, 3)
    assert localization.boundary_edges.shape == (9, 2)
    assert localization.selected_cell_count == 2
    assert localization.selected_dual_component_count == 1
    assert localization.selected_dual_edge_count == 1
    assert localization.selected_dual_bridge_edge_count == 1
    assert localization.selected_dual_articulation_cell_count == 0
    np.testing.assert_array_equal(localization.owner_dual_degree, 1)
    np.testing.assert_array_equal(localization.owner_boundary_face_count, 3)
    np.testing.assert_array_equal(localization.owner_articulation_mask, False)
    np.testing.assert_array_equal(localization.owner_dual_bridge_fraction, 1.0)
    assert np.all(np.isfinite(localization.boundary_face_risk))
    assert np.all(np.isfinite(localization.boundary_edge_risk))

    empty = boundary_bridge_localization(
        base,
        scale_multiplier=0.0,
        k_neighbors=3,
    )
    assert empty.selected_cell_count == 0
    assert empty.selected_dual_component_count == 0
    assert empty.boundary_faces.shape == (0, 3)
    assert empty.boundary_edges.shape == (0, 2)


def test_boundary_owner_intervention_recomputes_and_preserves_zero_rounds() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    base = AdaptiveCellFiltration(
        points=points,
        top_simplices=np.array([[0, 1, 2, 3], [0, 1, 2, 4]]),
        scores=np.array([0.5, 0.5]),
        method="test",
        diagnostics={},
    )

    zero = iterative_boundary_owner_intervention(
        base,
        scale_multiplier=1.0,
        max_rounds=0,
        risk_threshold=0.0,
        k_neighbors=3,
    )
    positive = iterative_boundary_owner_intervention(
        base,
        scale_multiplier=1.0,
        max_rounds=1,
        risk_threshold=0.0,
        k_neighbors=3,
    )

    assert zero.filtration is base
    assert zero.executed_rounds == 0
    assert zero.boundary_recomputation_count == 1
    assert zero.removed_cell_indices.size == 0
    assert positive.executed_rounds == 1
    assert positive.removed_cells_per_round == (2,)
    np.testing.assert_array_equal(positive.removed_cell_indices, [0, 1])
    assert positive.initial_selected_cell_count == 2
    assert positive.final_selected_cell_count == 0
    assert positive.boundary_recomputation_count == 2
    assert np.all(positive.filtration.scores > 1.0)
    np.testing.assert_array_equal(base.scores, [0.5, 0.5])


def test_boundary_risk_regions_and_safe_cut_are_label_free_candidates() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    base = AdaptiveCellFiltration(
        points=points,
        top_simplices=np.array([[0, 1, 2, 3], [0, 1, 2, 4]]),
        scores=np.array([0.5, 0.5]),
        method="test",
        diagnostics={},
    )

    analysis = boundary_risk_region_analysis(
        base,
        scale_multiplier=1.0,
        risk_threshold=0.0,
        k_neighbors=3,
    )
    baseline = boundary_region_cut_intervention(
        base,
        scale_multiplier=1.0,
        strategy="baseline",
        risk_threshold=0.0,
        k_neighbors=3,
    )
    largest = boundary_region_cut_intervention(
        base,
        scale_multiplier=1.0,
        strategy="largest_risk_region",
        risk_threshold=0.0,
        k_neighbors=3,
    )
    safe_cut = boundary_region_cut_intervention(
        base,
        scale_multiplier=1.0,
        strategy="safe_backbone_cut",
        risk_threshold=0.0,
        k_neighbors=3,
    )

    assert analysis.region_face_counts.tolist() == [6]
    assert analysis.region_owner_counts.tolist() == [2]
    assert analysis.safe_boundary_component_count == 5
    assert np.count_nonzero(analysis.flagged_edge_cut_mask) == 9
    assert baseline.filtration is base
    assert baseline.stopping_reason == "baseline"
    for candidate in (largest, safe_cut):
        np.testing.assert_array_equal(candidate.removed_cell_indices, [0, 1])
        assert candidate.initial_selected_cell_count == 2
        assert candidate.final_selected_cell_count == 0
        assert candidate.candidate_face_count == 6
        assert candidate.stopping_reason == "candidate_applied"
        assert np.all(candidate.filtration.scores > 1.0)
    np.testing.assert_array_equal(base.scores, [0.5, 0.5])
