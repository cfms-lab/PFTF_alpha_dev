import numpy as np
import pytest

from pftf_alpha.local_insertion_influence import LocalInsertionInfluenceConfig
from pftf_alpha.multiscale_surface_consensus import MultiscaleQuadraticConfig
from pftf_alpha.observed_identifiability import (
    FEATURE_NAMES,
    ObservedIdentifiabilitySignature,
    estimate_observed_identifiability_signature,
    evaluate_observed_identifiability,
    fit_robust_feature_scaling,
)
from pftf_alpha.sensor_stress import SensorStress, make_sensor_stress_case
from pftf_alpha.shared_trend_inference import infer_shared_trend_layers


def test_observed_signature_is_finite_and_complete() -> None:
    case = make_sensor_stress_case(
        SensorStress.LOCAL_BUMP,
        96,
        reference_count=128,
        seed=17,
    )
    inference = infer_shared_trend_layers(case.points)
    signature = estimate_observed_identifiability_signature(
        case.points,
        inference.inference.layer_ids,
        influence_config=LocalInsertionInfluenceConfig(neighbor_counts=(8, 12)),
        multiscale_config=MultiscaleQuadraticConfig(neighbor_counts=(8, 12)),
    )
    assert signature.values().shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(signature.values()))


def test_robust_scaling_has_positive_fallback_for_constant_features() -> None:
    first = ObservedIdentifiabilitySignature(*([1.0] * len(FEATURE_NAMES)))
    second = ObservedIdentifiabilitySignature(*([1.0] * len(FEATURE_NAMES)))
    scaling = fit_robust_feature_scaling((first, second))
    assert scaling.feature_names == FEATURE_NAMES
    assert np.all(np.asarray(scaling.scales) > 0.0)
    assert np.allclose(scaling.scales, 0.05)


def test_phase14_reduced_panel_cannot_claim_identifiability() -> None:
    result = evaluate_observed_identifiability(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_seed=79,
        held_out_seed=83,
        surface_sample_count=64,
        influence_config=LocalInsertionInfluenceConfig(neighbor_counts=(8, 12)),
        multiscale_config=MultiscaleQuadraticConfig(neighbor_counts=(8, 12)),
    )
    assert result.full_protocol is False
    assert result.calibration.harmful_case_count == 0
    assert result.calibration.audited_case_count == 0
    assert result.feature_identifiable is False
    assert result.guard_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.real_scan_supported is False
    assert result.deployment_supported is False


def test_phase14_rejects_seed_reuse() -> None:
    with pytest.raises(ValueError, match="must differ"):
        evaluate_observed_identifiability(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            calibration_seed=89,
            held_out_seed=89,
            surface_sample_count=64,
        )
