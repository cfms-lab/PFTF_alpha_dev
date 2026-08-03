from pftf_alpha.eth_gazebo_local_support_shadow import (
    LocalSupportValidationCase,
    _summarize,
)
from pftf_alpha.eth_gazebo_reconstruction_shadow import ReconstructionEndpoint


def _endpoint(geometry: float, fscore: float, recall: float) -> ReconstructionEndpoint:
    return ReconstructionEndpoint(
        normalized_chamfer_squared=0.01,
        normalized_hausdorff=geometry - 0.01,
        geometry_loss=geometry,
        precision=fscore,
        recall=recall,
        fscore=fscore,
        used_vertices=10,
        edges=20,
        faces=12,
        connected_components=2,
        betti_0=2,
        betti_1=3,
        betti_2=1,
        euler_characteristic=0,
        boundary_edges=0,
        nonmanifold_edges=1,
        nonmanifold_edge_fraction=0.05,
        watertight=False,
    )


def _case(source: int, *, supported_cells: int = 4) -> LocalSupportValidationCase:
    anchor = _endpoint(0.20, 0.70, 0.72)
    scan = _endpoint(0.30, 0.10, 0.15)
    local = _endpoint(0.18, 0.68, 0.715)
    return LocalSupportValidationCase(
        source_index=source,
        pair_count=6,
        accepted_pair_count=6,
        rejected_pair_count=0,
        anchor_cell_count=100,
        target_only_cell_count=50,
        corroborated_target_only_cell_count=supported_cells,
        rejected_target_only_cell_count=50 - supported_cells,
        mean_target_support=1.5,
        mean_target_dispersion_meters=0.2,
        anchor_baseline=anchor,
        scan_fused_baseline=scan,
        local_support=local,
        geometry_margin_vs_anchor=0.02,
        geometry_margin_vs_scan=0.12,
        fscore_margin_vs_anchor=-0.02,
        recall_margin_vs_anchor=-0.005,
    )


def test_phase41_summary_applies_frozen_aggregate_gate() -> None:
    summary = _summarize((_case(1), _case(17)))

    assert summary.case_count == 2
    assert summary.geometry_win_vs_anchor_count == 2
    assert summary.geometry_win_vs_scan_count == 2
    assert summary.every_case_exercised_local_support is True
    assert summary.mean_geometry_beats_both_baselines is True
    assert summary.mean_fscore_within_anchor_tolerance is True
    assert summary.mean_recall_within_anchor_tolerance is True
    assert summary.local_support_shadow_supported is True


def test_phase41_summary_fails_closed_when_a_case_has_no_supported_cell() -> None:
    summary = _summarize((_case(1), _case(17, supported_cells=0)))

    assert summary.every_case_exercised_local_support is False
    assert summary.local_support_shadow_supported is False
