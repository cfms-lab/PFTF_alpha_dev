import numpy as np

from pftf_alpha.guard_domain_shift import (
    SamplingProfile,
    ShiftGeometry,
    make_shift_case,
)
from pftf_alpha.shared_trend_inference import (
    evaluate_shared_trend_inference,
    infer_shared_trend_layers,
)


def _label_error(labels: np.ndarray, truth: np.ndarray) -> float:
    direct = float(np.mean(labels != truth))
    return min(direct, 1.0 - direct)


def test_shared_trend_repairs_both_phase6_sparse_failures() -> None:
    rows = (
        (ShiftGeometry.PARABOLOID_024, 0.005, 20430825),
        (ShiftGeometry.PARABOLOID_036, 0.025, 22520827),
    )
    for geometry, noise, seed in rows:
        case = make_shift_case(
            geometry,
            SamplingProfile(point_count=96, noise=noise),
            reference_count=128,
            seed=seed,
        )
        result = infer_shared_trend_layers(case.points)
        assert result.diagnostics.converged
        assert _label_error(
            result.inference.layer_ids,
            case.point_component_labels,
        ) == 0.0


def test_phase7_smoke_cannot_promote_a_reduced_panel() -> None:
    result = evaluate_shared_trend_inference(
        profiles=(SamplingProfile(point_count=64, noise=0.01),),
        geometries=(ShiftGeometry.PARABOLOID_024,),
        reference_count=128,
        repeats=1,
        seed=23,
        surface_sample_count=64,
    )
    assert result.case_count == 1
    assert result.phase7_supported is False
    assert result.deployment_supported is False
