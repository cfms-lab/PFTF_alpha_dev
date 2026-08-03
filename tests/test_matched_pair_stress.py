import numpy as np
import pytest

from pftf_alpha.conservative_influence_calibration import InfluenceFeatureCohort
from pftf_alpha.matched_pair_stress import (
    DEFAULT_STRESS_SPECS,
    MatchedPairStressProfile,
    calibrate_profile_aware_rectangle,
    evaluate_matched_pair_stress,
    perturb_matched_pairs,
)
from pftf_alpha.sensor_stress import SensorStress


def test_exact_and_combined_pair_stress_are_deterministic() -> None:
    rng = np.random.default_rng(17)
    primary = rng.normal(size=(100, 3))
    repeat = primary + rng.normal(scale=0.01, size=primary.shape)
    exact = perturb_matched_pairs(
        primary,
        repeat,
        DEFAULT_STRESS_SPECS[0],
        seed=23,
    )
    assert exact.retained_pair_count == 100
    assert exact.missing_pair_count == 0
    assert exact.mismatched_pair_count == 0
    assert np.array_equal(exact.primary_points, primary)
    assert np.array_equal(exact.repeat_points, repeat)

    first = perturb_matched_pairs(
        primary,
        repeat,
        DEFAULT_STRESS_SPECS[-1],
        seed=29,
    )
    second = perturb_matched_pairs(
        primary,
        repeat,
        DEFAULT_STRESS_SPECS[-1],
        seed=29,
    )
    assert first.retained_pair_count == 90
    assert first.missing_pair_count == 10
    assert 0 <= first.mismatched_pair_count <= 2
    assert first.rotation_degrees == 0.5
    assert np.isclose(np.linalg.norm(first.rotation_axis), 1.0)
    assert np.array_equal(first.primary_points, second.primary_points)
    assert np.array_equal(first.repeat_points, second.repeat_points)
    assert first.presented_pair_map_sha256 == second.presented_pair_map_sha256


def test_profile_calibration_rejects_harm_in_every_group() -> None:
    groups = (
        InfluenceFeatureCohort(
            harmful=((5.0, 5.0),),
            focus_safe=((1.0, 1.0), (2.0, 2.0)),
            all_safe=((1.0, 1.0), (2.0, 2.0)),
        ),
        InfluenceFeatureCohort(
            harmful=((3.0, 3.0),),
            focus_safe=((1.0, 1.0), (2.5, 2.5)),
            all_safe=((1.0, 1.0), (2.5, 2.5)),
        ),
    )
    rectangle = calibrate_profile_aware_rectangle(groups)
    assert rectangle is not None
    assert rectangle.retained_focus_safe_count == 4
    for group in groups:
        for peak, support in group.harmful:
            assert not (
                peak <= rectangle.peak_threshold
                and support <= rectangle.support_threshold
            )


def test_profile_calibration_requires_harm_in_every_group() -> None:
    populated = InfluenceFeatureCohort(
        harmful=((2.0, 2.0),),
        focus_safe=((1.0, 1.0),),
        all_safe=((1.0, 1.0),),
    )
    empty = InfluenceFeatureCohort(
        harmful=(),
        focus_safe=((1.0, 1.0),),
        all_safe=((1.0, 1.0),),
    )
    assert calibrate_profile_aware_rectangle((populated, empty)) is None


def test_phase18_reduced_panel_cannot_open_final_held_out() -> None:
    result = evaluate_matched_pair_stress(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_a_seed=173,
        calibration_b_seed=179,
        final_held_out_seed=181,
        surface_sample_count=64,
    )
    assert result.calibration_a.case_count == len(DEFAULT_STRESS_SPECS)
    assert result.calibration_b.case_count == len(DEFAULT_STRESS_SPECS)
    assert result.calibration_a.full_protocol is False
    assert result.calibration_b.full_protocol is False
    assert all(
        "pairing_correctness_unknown_to_route" in row.evidence.information_boundary
        for row in result.calibration_a.cases
    )
    assert result.final_held_out is None
    assert result.phase18_supported is False
    assert result.correspondence_stress_synthetic_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase18_rejects_reused_seeds() -> None:
    with pytest.raises(ValueError, match="must differ"):
        evaluate_matched_pair_stress(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            calibration_a_seed=191,
            calibration_b_seed=191,
            final_held_out_seed=193,
            surface_sample_count=64,
        )


def test_stress_profile_names_are_unique() -> None:
    assert {spec.profile for spec in DEFAULT_STRESS_SPECS} == set(
        MatchedPairStressProfile
    )
