from pftf_alpha.sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from pftf_alpha.surface import evaluate_surface, mesh_statistics
from pftf_alpha.synthetic import PanelSplit, SyntheticFamily, make_synthetic_case
from pftf_alpha.two_layer_connectivity import (
    construct_two_layer_surface,
    evaluate_two_layer_connectivity,
)


def test_construction_has_two_independent_disk_components() -> None:
    case = make_synthetic_case(
        SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.HELD_OUT,
        point_count=96,
        reference_count=256,
        seed=7,
        variation_overrides={"sheet_gap": 0.8, "noise": 0.01},
    )
    construction = construct_two_layer_surface(
        case.points,
        SamplingSufficiencyConfig(minimum_separation_snr=3.0),
    )
    statistics = mesh_statistics(construction.mesh)
    assert statistics.connected_components == 2
    assert statistics.betti_0 == 2
    assert statistics.betti_1 == 0
    assert statistics.betti_2 == 0

    endpoints = evaluate_surface(
        construction.mesh,
        case.reference_points,
        expected_components=2,
        expected_betti=(2, 0, 0),
        vertex_component_labels=construction.inference.layer_ids,
        characteristic_length=case.characteristic_length,
        sample_count=64,
        threshold_fraction=0.025,
        seed=11,
    )
    assert endpoints.component_error == 0
    assert endpoints.labeled_false_bridge_edges == 0
    assert endpoints.labeled_false_bridge_faces == 0


def test_phase2_smoke_never_accepts_under_resolved_case() -> None:
    result = evaluate_two_layer_connectivity(
        point_count=48,
        reference_count=128,
        gaps=(0.18, 1.20),
        repeats=1,
        seed=19,
        surface_sample_count=64,
        gate_config=SamplingSufficiencyConfig(
            k_neighbors=8,
            minimum_separation_snr=3.0,
        ),
    )
    low_gap = next(case for case in result.cases if case.sheet_gap == 0.18)
    assert low_gap.decision is not SamplingGateDecision.ACCEPT
    assert result.false_safe_count == 0
    assert result.deployment_supported is False
