import numpy as np
import pytest

from pftf_alpha.affine_spd_alpha import (
    IncompatibleLocalMetricError,
    audit_global_metric_compatibility,
    global_affine_spd_alpha,
    global_affine_spd_alpha_from_field,
)
from pftf_alpha.filtration import AlphaFiltration
from pftf_alpha.metrics import LocalMetricField


def _canonical(cells) -> set[tuple[int, ...]]:
    return {tuple(sorted(int(vertex) for vertex in cell)) for cell in cells}


def _records(filtration: AlphaFiltration) -> dict[tuple[int, ...], float]:
    return {record.vertices: record.alpha_squared for record in filtration.records}


def test_identity_global_metric_reproduces_euclidean_filtration() -> None:
    points = np.random.default_rng(17).normal(size=(32, 3))
    expected = AlphaFiltration.from_points(points)
    construction = global_affine_spd_alpha(points, np.eye(3))

    assert _canonical(construction.filtration.top_simplices) == _canonical(
        expected.top_simplices
    )
    assert _records(construction.filtration) == pytest.approx(_records(expected))
    np.testing.assert_array_equal(construction.filtration.points, points)


def test_global_metric_matches_explicit_transformed_coordinates() -> None:
    points = np.random.default_rng(23).normal(size=(36, 3))
    metric = np.asarray(
        [[2.0, 0.3, 0.1], [0.3, 1.2, -0.05], [0.1, -0.05, 0.7]]
    )
    construction = global_affine_spd_alpha(points, metric)
    expected = AlphaFiltration.from_points(points @ np.linalg.cholesky(metric))

    assert _canonical(construction.filtration.top_simplices) == _canonical(
        expected.top_simplices
    )
    assert _records(construction.filtration) == pytest.approx(_records(expected))


def test_constant_field_is_accepted() -> None:
    points = np.random.default_rng(29).normal(size=(28, 3))
    metric = np.diag([2.0, 1.0, 0.5])
    field = LocalMetricField(
        matrices=np.repeat(metric[None, :, :], len(points), axis=0),
        confidence=np.ones(len(points)),
    )

    audit = audit_global_metric_compatibility(field)
    construction = global_affine_spd_alpha_from_field(points, field)

    assert audit.compatible
    assert audit.maximum_relative_deviation == 0.0
    assert construction.metric == pytest.approx(metric)


def test_varying_field_fails_closed() -> None:
    points = np.random.default_rng(31).normal(size=(24, 3))
    matrices = np.repeat(np.eye(3)[None, :, :], len(points), axis=0)
    matrices[1:, 0, 0] = 2.0
    field = LocalMetricField(matrices=matrices, confidence=np.ones(len(points)))

    audit = audit_global_metric_compatibility(field)

    assert not audit.compatible
    with pytest.raises(IncompatibleLocalMetricError, match="not globally"):
        global_affine_spd_alpha_from_field(points, field)


def test_field_shape_must_match_points() -> None:
    points = np.random.default_rng(37).normal(size=(24, 3))
    field = LocalMetricField(
        matrices=np.repeat(np.eye(3)[None, :, :], 23, axis=0),
        confidence=np.ones(23),
    )

    with pytest.raises(ValueError, match="field shape"):
        global_affine_spd_alpha_from_field(points, field)
