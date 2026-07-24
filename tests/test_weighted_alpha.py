import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

from pftf_alpha.weighted_alpha import (
    PointSubmersionError,
    evaluate_m1_ablation,
    main,
    regular_triangulation,
    weighted_alpha_filtration,
)


def _canonical(cells) -> set[tuple[int, int, int, int]]:
    return {tuple(sorted(int(v) for v in cell)) for cell in np.asarray(cells)}


def test_zero_weights_reproduce_delaunay() -> None:
    rng = np.random.default_rng(11)
    points = rng.normal(size=(40, 3))
    regular = regular_triangulation(points, np.zeros(len(points)))
    delaunay = Delaunay(points).simplices
    assert _canonical(regular) == _canonical(delaunay)


def test_nonzero_weights_change_connectivity_but_stay_valid() -> None:
    rng = np.random.default_rng(5)
    points = rng.normal(size=(40, 3))
    weights = rng.uniform(0.0, 0.2, size=len(points))
    regular = regular_triangulation(points, weights)
    base = regular_triangulation(points, np.zeros(len(points)))
    assert _canonical(regular) != _canonical(base)
    assert all(len(set(cell.tolist())) == 4 for cell in regular)


def test_weight_scale_zero_matches_b4_connectivity() -> None:
    from pftf_alpha.adaptive import density_scaled_filtration
    from pftf_alpha.filtration import AlphaFiltration

    rng = np.random.default_rng(2)
    points = rng.normal(size=(36, 3))
    m1 = weighted_alpha_filtration(points, k_neighbors=8, weight_scale=0.0)
    b4 = density_scaled_filtration(
        AlphaFiltration.from_points(points), k_neighbors=8
    )
    assert _canonical(m1.top_simplices) == _canonical(b4.top_simplices)
    np.testing.assert_allclose(np.sort(m1.scores), np.sort(b4.scores))


def test_large_weight_scale_raises_point_submersion() -> None:
    rng = np.random.default_rng(1)
    points = rng.normal(size=(40, 3))
    # A very large scale drives some points below the lower hull.
    with pytest.raises(PointSubmersionError):
        weighted_alpha_filtration(points, k_neighbors=8, weight_scale=50.0)


def test_weighted_filtration_rejects_bad_scale() -> None:
    points = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.raises(ValueError):
        weighted_alpha_filtration(points, k_neighbors=6, weight_scale=-1.0)


def test_ablation_reports_baselines_and_never_promotes_flag_in_artifact() -> None:
    result = evaluate_m1_ablation(
        point_count=40,
        reference_count=256,
        candidate_budget=4,
        weight_scales=(0.0, 0.25),
        seed=17,
    )
    payload = result.to_dict()
    assert payload["artifact_schema"] == "pftf_alpha_m1_weighted_alpha_ablation/v1"
    assert payload["promotion_supported"] is False
    assert "m1_dominates_b4" in payload
    assert "b4_dominant_weight_scale" in payload
    # scale 0 must equal the B4 baseline aggregate (same connectivity + scoring).
    scale0 = next(row for row in result.m1_by_scale if row.weight_scale == 0.0)
    assert (
        scale0.labeled_false_bridge_edges_sum
        == result.b4.labeled_false_bridge_edges_sum
    )
    assert scale0.betti_error_sum == result.b4.betti_error_sum


def test_cli_writes_artifact(tmp_path: Path) -> None:
    output = tmp_path / "m1.json"
    exit_code = main(
        ["--point-count", "40", "--reference-count", "256",
         "--candidate-budget", "4", "--seed", "9", "--output", str(output)]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_schema"] == "pftf_alpha_m1_weighted_alpha_ablation/v1"
