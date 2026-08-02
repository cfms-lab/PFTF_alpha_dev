from pftf_alpha.curvature_guard import (
    CurvatureGuardConfig,
    estimate_curvature_guard,
    evaluate_curvature_guard,
)
from pftf_alpha.two_layer_boundary import BoundaryAxis, make_boundary_case


def test_normal_coherence_separates_calibration_anchor_and_failure() -> None:
    anchor = make_boundary_case(
        BoundaryAxis.CURVATURE,
        0.24,
        point_count=160,
        reference_count=256,
        seed=7,
    )
    failure = make_boundary_case(
        BoundaryAxis.CURVATURE,
        0.48,
        point_count=160,
        reference_count=256,
        seed=7,
    )
    config = CurvatureGuardConfig(minimum_normal_coherence=0.82)
    assert estimate_curvature_guard(anchor.points, config).model_adequate
    assert not estimate_curvature_guard(failure.points, config).model_adequate


def test_guard_smoke_preserves_information_boundary() -> None:
    result = evaluate_curvature_guard(
        point_count=64,
        reference_count=128,
        repeats=1,
        seed=19,
        levels={BoundaryAxis.CURVATURE: (0.12, 0.48)},
        surface_sample_count=64,
    )
    assert result.case_count == 2
    assert all(
        case.evidence.information_boundary == "observed_point_coordinates_only"
        for case in result.cases
    )
    assert result.phase4b_supported is False
    assert result.deployment_supported is False
