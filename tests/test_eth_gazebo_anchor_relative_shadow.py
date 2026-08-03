from pftf_alpha.eth_gazebo_anchor_relative_shadow import (
    AnchorRelativeValidationCase,
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


def _case(source: int, *, added: int = 3) -> AnchorRelativeValidationCase:
    anchor = _endpoint(0.20, 0.70, 0.72)
    phase41 = _endpoint(0.22, 0.68, 0.71)
    relative = _endpoint(0.18, 0.69, 0.715)
    return AnchorRelativeValidationCase(
        source_index=source,
        pair_count=3,
        phase41_candidate_cell_count=10,
        anchor_relative_cell_count=added,
        rejected_by_anchor_relative_count=10 - added,
        anchor_baseline=anchor,
        phase41_baseline=phase41,
        anchor_relative=relative,
        geometry_margin_vs_anchor=0.02,
        geometry_margin_vs_phase41=0.04,
        fscore_margin_vs_anchor=-0.01,
        recall_margin_vs_anchor=-0.005,
    )


def test_phase42_summary_applies_frozen_gate() -> None:
    summary = _summarize((_case(25), _case(26), _case(27)))

    assert summary.case_count == 3
    assert summary.every_case_exercised_anchor_relative is True
    assert summary.mean_geometry_beats_both_baselines is True
    assert summary.mean_fscore_within_anchor_tolerance is True
    assert summary.mean_recall_within_anchor_tolerance is True
    assert summary.anchor_relative_shadow_supported is True


def test_phase42_summary_requires_every_case_to_exercise_route() -> None:
    summary = _summarize((_case(25), _case(26), _case(27, added=0)))

    assert summary.every_case_exercised_anchor_relative is False
    assert summary.anchor_relative_shadow_supported is False
