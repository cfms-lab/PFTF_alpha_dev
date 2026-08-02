import numpy as np

from pftf_alpha.two_layer_boundary import (
    BoundaryAxis,
    evaluate_two_layer_boundary,
    make_boundary_case,
)


def test_boundary_case_encodes_contact_minimum_gap() -> None:
    case = make_boundary_case(
        BoundaryAxis.CONTACT_SEVERITY,
        0.76,
        point_count=160,
        reference_count=256,
        seed=7,
        noise=0.0,
    )
    lower = case.points[case.point_component_labels == 0]
    upper = case.points[case.point_component_labels == 1]
    assert np.min(-2.0 * lower[:, 2]) >= 0.04
    assert np.min(2.0 * upper[:, 2]) >= 0.04
    assert np.isclose(case.variation["sheet_gap"], 0.04)


def test_boundary_smoke_reports_every_requested_level() -> None:
    levels = {
        BoundaryAxis.CURVATURE: (0.0, 0.12),
        BoundaryAxis.CONTACT_SEVERITY: (0.0, 0.76),
    }
    result = evaluate_two_layer_boundary(
        point_count=64,
        reference_count=128,
        repeats=1,
        seed=19,
        levels=levels,
        surface_sample_count=64,
    )
    assert result.case_count == 4
    assert len(result.level_summaries) == 4
    assert len(result.axis_summaries) == 2
    assert all(0.0 <= row.acceptance_rate <= 1.0 for row in result.level_summaries)
    assert result.phase4_diagnostic_supported is False
    assert result.deployment_supported is False
