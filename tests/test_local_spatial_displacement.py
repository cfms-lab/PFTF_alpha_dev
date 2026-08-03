from __future__ import annotations

import numpy as np
import pytest

from pftf_alpha.local_spatial_displacement import (
    LocalSpatialDisplacementConfig,
    estimate_local_spatial_displacement_evidence,
)


def _grid() -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, 5)
    xx, yy = np.meshgrid(axis, axis)
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))


def test_local_evidence_is_zero_for_a_shared_translation() -> None:
    primary = _grid()
    repeat = primary + np.asarray((0.1, -0.2, 0.3))

    evidence = estimate_local_spatial_displacement_evidence(primary, repeat)

    assert evidence.maximum_local_residual == pytest.approx(0.0, abs=1.0e-12)
    assert evidence.maximum_local_score_excess == pytest.approx(0.0, abs=1.0e-12)
    assert evidence.neighbor_count == 8


def test_isolated_displacement_has_stronger_local_residual_than_smooth_field() -> None:
    primary = _grid()
    smooth_repeat = primary.copy()
    smooth_repeat[:, 2] += 0.03 * primary[:, 0]
    isolated_repeat = smooth_repeat.copy()
    isolated_repeat[12, 2] += 0.8

    smooth = estimate_local_spatial_displacement_evidence(primary, smooth_repeat)
    isolated = estimate_local_spatial_displacement_evidence(
        primary,
        isolated_repeat,
    )

    assert isolated.maximum_local_residual > smooth.maximum_local_residual
    assert isolated.maximum_local_score_excess > smooth.maximum_local_score_excess
    assert isolated.peak_neighbor_score_support_fraction < 0.5


def test_local_evidence_is_invariant_to_joint_pair_permutation() -> None:
    primary = _grid()
    repeat = primary.copy()
    repeat[7, 2] += 0.6
    permutation = np.random.default_rng(7).permutation(primary.shape[0])

    original = estimate_local_spatial_displacement_evidence(primary, repeat)
    permuted = estimate_local_spatial_displacement_evidence(
        primary[permutation],
        repeat[permutation],
    )

    assert permuted.maximum_local_residual == pytest.approx(
        original.maximum_local_residual
    )
    assert permuted.support_local_residual == pytest.approx(
        original.support_local_residual
    )
    assert permuted.maximum_local_score_excess == pytest.approx(
        original.maximum_local_score_excess
    )


def test_local_config_rejects_too_few_neighbors() -> None:
    with pytest.raises(ValueError, match="at least two"):
        LocalSpatialDisplacementConfig(neighbor_count=1)
