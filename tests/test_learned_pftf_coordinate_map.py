import numpy as np

from pftf_alpha.learned_pftf_coordinate_map import (
    geometry_summary_features,
    make_map_recovery_cases,
    pftf_summary_features,
)
from pftf_alpha.synthetic import PanelSplit


def test_phase48_recovery_case_uses_exact_inverse_corruption() -> None:
    case = make_map_recovery_cases(
        split=PanelSplit.TRAIN,
        seeds=(48_001,),
        strengths=(0.20,),
        families=("torus",),
        point_count=32,
        reference_count=64,
    )[0]

    assert case.latent_points.shape == (32, 3)
    assert np.allclose(np.mean(case.latent_points, axis=0), 0.0, atol=1.0e-15)
    assert np.isclose(
        np.sqrt(np.mean(np.sum(case.latent_points**2, axis=1))),
        1.0,
    )
    recovered = case.observed_points.copy()
    recovered[:, 1] += case.true_strength * recovered[:, 0] ** 2
    assert np.allclose(recovered, case.latent_points, atol=1.0e-15)


def test_phase48_observed_feature_vectors_are_finite_and_frozen_size() -> None:
    case = make_map_recovery_cases(
        split=PanelSplit.TRAIN,
        seeds=(48_001,),
        strengths=(0.20,),
        families=("sharp_crease",),
        point_count=32,
        reference_count=64,
    )[0]

    pftf = pftf_summary_features(case.observed_points, k_neighbors=8)
    geometry = geometry_summary_features(case.observed_points)

    assert pftf.shape == (7,)
    assert geometry.shape == (4,)
    assert np.all(np.isfinite(pftf))
    assert np.all(np.isfinite(geometry))
