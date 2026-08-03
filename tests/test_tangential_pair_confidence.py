import numpy as np
import pytest

from pftf_alpha.matched_pair_stress import (
    DEFAULT_STRESS_SPECS,
    MatchedPairStressProfile,
    perturb_matched_pairs,
)
from pftf_alpha.sensor_stress import SensorStress
from pftf_alpha.tangential_pair_confidence import (
    FROZEN_PHASE18_RECTANGLE,
    PairScoreCohort,
    calibrate_pair_confidence_cutoff,
    evaluate_tangential_pair_confidence,
    tangential_pair_confidence_scores,
)


def test_tangential_score_separates_normal_excursion_from_pair_swap() -> None:
    axis = np.linspace(-1.0, 1.0, 5)
    xx, yy = np.meshgrid(axis, axis)
    primary = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    repeat = primary.copy()
    repeat[0, 2] = 0.75
    repeat[[1, -1]] = repeat[[-1, 1]]

    result = tangential_pair_confidence_scores(primary, repeat)

    assert result.scores[0] < 0.1
    assert result.scores[1] > 1.0
    assert result.scores[-1] > 1.0
    assert result.alignment_inlier_count == 20


def test_pair_cutoff_is_immediately_below_smallest_mismatch() -> None:
    groups = (
        PairScoreCohort(
            profile=MatchedPairStressProfile.MISMATCH_02,
            scores=np.asarray((0.1, 0.4, 1.2)),
            correct_mask=np.asarray((True, True, False)),
            mismatch_required=True,
        ),
        PairScoreCohort(
            profile=MatchedPairStressProfile.COMBINED,
            scores=np.asarray((0.2, 0.5, 1.8)),
            correct_mask=np.asarray((True, True, False)),
            mismatch_required=True,
        ),
    )
    result = calibrate_pair_confidence_cutoff(groups)
    assert result is not None
    assert result.cutoff == np.nextafter(1.2, -np.inf)
    assert result.retained_correct_pair_count == 4
    assert result.retained_mismatch_pair_count == 0


def test_pair_cutoff_requires_mismatch_in_required_group() -> None:
    group = PairScoreCohort(
        profile=MatchedPairStressProfile.MISMATCH_02,
        scores=np.asarray((0.1, 0.2)),
        correct_mask=np.asarray((True, True)),
        mismatch_required=True,
    )
    assert calibrate_pair_confidence_cutoff((group,)) is None


def test_perturbation_exposes_evaluation_only_pair_source_map() -> None:
    rng = np.random.default_rng(41)
    primary = rng.normal(size=(100, 3))
    repeat = primary + rng.normal(scale=0.01, size=primary.shape)
    perturbed = perturb_matched_pairs(
        primary,
        repeat,
        DEFAULT_STRESS_SPECS[3],
        seed=43,
    )
    assert perturbed.primary_ids.shape == (100,)
    assert perturbed.repeat_source_ids.shape == (100,)
    assert np.sum(perturbed.primary_ids != perturbed.repeat_source_ids) == 2
    assert perturbed.mismatched_pair_count == 2


def test_phase19_reduced_panel_cannot_open_final() -> None:
    result = evaluate_tangential_pair_confidence(
        point_counts=(64,),
        stresses=(SensorStress.CONTROL,),
        reference_count=128,
        repeats=1,
        calibration_a_seed=233,
        calibration_b_seed=239,
        final_held_out_seed=241,
        surface_sample_count=64,
    )
    assert result.pair_cutoff_calibration is not None
    assert result.calibration_a is not None
    assert result.calibration_b is not None
    assert result.calibration_a.case_count == len(DEFAULT_STRESS_SPECS)
    assert result.calibration_b.case_count == len(DEFAULT_STRESS_SPECS)
    assert result.calibration_a.full_protocol is False
    assert result.calibration_b.full_protocol is False
    assert result.final_held_out is None
    assert result.phase19_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_supported is False
    assert result.trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase19_rejects_phase18_unopened_final_seed() -> None:
    with pytest.raises(ValueError, match="must not be reused"):
        evaluate_tangential_pair_confidence(
            point_counts=(64,),
            stresses=(SensorStress.CONTROL,),
            reference_count=128,
            repeats=1,
            calibration_a_seed=22900804,
            calibration_b_seed=251,
            final_held_out_seed=257,
            surface_sample_count=64,
        )


def test_phase19_keeps_frozen_phase18_rectangle() -> None:
    assert FROZEN_PHASE18_RECTANGLE.peak_threshold == 10.922625244331805
    assert np.isinf(FROZEN_PHASE18_RECTANGLE.support_threshold)
