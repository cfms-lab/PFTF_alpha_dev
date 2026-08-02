from pftf_alpha.guard_domain_shift import (
    SamplingProfile,
    ShiftGeometry,
    evaluate_guard_domain_shift,
    make_shift_case,
)


def test_shift_case_is_balanced_and_observed_only() -> None:
    profile = SamplingProfile(point_count=96, noise=0.01)
    case = make_shift_case(
        ShiftGeometry.SADDLE_024,
        profile,
        reference_count=128,
        seed=7,
    )
    assert case.points.shape == (96, 3)
    assert case.reference_points.shape == (128, 3)
    assert sum(case.point_component_labels == 0) == 48
    assert sum(case.point_component_labels == 1) == 48


def test_domain_shift_smoke_keeps_threshold_frozen() -> None:
    profile = SamplingProfile(point_count=64, noise=0.01)
    result = evaluate_guard_domain_shift(
        profiles=(profile,),
        geometries=(
            ShiftGeometry.PARABOLOID_024,
            ShiftGeometry.SADDLE_024,
        ),
        reference_count=128,
        repeats=1,
        seed=19,
        surface_sample_count=64,
    )
    assert result.case_count == 2
    assert result.guard_config.minimum_normal_coherence == 0.82
    assert result.phase5_supported is False
    assert result.deployment_supported is False
