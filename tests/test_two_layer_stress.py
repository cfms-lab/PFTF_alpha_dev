import numpy as np

from pftf_alpha.sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from pftf_alpha.surface import mesh_statistics
from pftf_alpha.two_layer_connectivity import construct_two_layer_surface
from pftf_alpha.two_layer_stress import (
    TwoLayerStressFamily,
    evaluate_two_layer_stress,
    make_stress_case,
)


def test_rotated_parallel_construction_is_rotation_invariant() -> None:
    case = make_stress_case(
        TwoLayerStressFamily.ROTATED_PARALLEL,
        point_count=96,
        reference_count=256,
        seed=7,
    )
    construction = construct_two_layer_surface(
        case.points,
        SamplingSufficiencyConfig(minimum_separation_snr=3.0),
    )
    statistics = mesh_statistics(construction.mesh)
    assert statistics.connected_components == 2
    assert statistics.betti_0 == 2
    assert np.all(np.bincount(construction.inference.layer_ids) > 0)


def test_phase3_smoke_reports_negative_false_accepts() -> None:
    result = evaluate_two_layer_stress(
        point_count=64,
        reference_count=128,
        repeats=1,
        seed=19,
        families=(
            TwoLayerStressFamily.ROTATED_PARALLEL,
            TwoLayerStressFamily.NEAR_CONTACT,
            TwoLayerStressFamily.CROSSING,
        ),
        surface_sample_count=64,
        gate_config=SamplingSufficiencyConfig(
            k_neighbors=8,
            minimum_separation_snr=3.0,
        ),
    )
    negative = [case for case in result.cases if not case.declared_in_scope]
    assert result.negative_accept_count == sum(
        case.decision is SamplingGateDecision.ACCEPT for case in negative
    )
    assert result.negative_accept_count == sum(
        case.out_of_scope_false_accept for case in negative
    )
    assert result.deployment_supported is False
