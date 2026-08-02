from pftf_alpha.guard_domain_shift import (
    SamplingProfile,
    ShiftGeometry,
    make_shift_case,
)
from pftf_alpha.local_order_guard import (
    estimate_local_order_guard,
    evaluate_local_order_guard,
)
from pftf_alpha.sampling_gate import SamplingSufficiencyConfig
from pftf_alpha.two_layer_connectivity import construct_two_layer_surface


def test_local_order_margin_is_density_normalized() -> None:
    margins = []
    for point_count in (96, 256):
        case = make_shift_case(
            ShiftGeometry.PARABOLOID_024,
            SamplingProfile(point_count=point_count, noise=0.01),
            reference_count=256,
            seed=7,
        )
        construction = construct_two_layer_surface(
            case.points,
            SamplingSufficiencyConfig(minimum_separation_snr=3.0),
        )
        evidence = estimate_local_order_guard(
            case.points,
            construction.inference.layer_ids,
        )
        margins.append(evidence.local_order_margin)
    assert abs(margins[0] - margins[1]) < 0.10


def test_phase6_smoke_keeps_calibrated_bins_frozen() -> None:
    result = evaluate_local_order_guard(
        profiles=(SamplingProfile(point_count=64, noise=0.01),),
        geometries=(ShiftGeometry.PARABOLOID_024,),
        reference_count=128,
        repeats=1,
        seed=19,
        surface_sample_count=64,
    )
    assert result.case_count == 1
    assert result.guard_config.density_bins[0].maximum_point_count == 96
    assert result.phase6_supported is False
    assert result.deployment_supported is False
