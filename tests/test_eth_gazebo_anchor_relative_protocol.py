from pftf_alpha.eth_gazebo_anchor_relative_protocol import (
    MINIMUM_DIRECT_PAIR_COUNT,
    SELECTED_MAXIMUM_ANCHOR_PLANE_RESIDUAL_METERS,
    SELECTED_MAXIMUM_NEAREST_ANCHOR_DISTANCE_METERS,
    SELECTED_MINIMUM_NORMAL_ALIGNMENT,
    VALIDATION_SOURCE_INDICES,
    _validation_sources,
)
from pftf_alpha.eth_gazebo_local_support_protocol import (
    VALIDATION_SOURCE_INDICES as PHASE41_SOURCES,
)
from pftf_alpha.eth_gazebo_reconstruction_protocol import (
    VALIDATION_SOURCE_INDICES as PHASE40_SOURCES,
)


def test_phase42_reserves_only_late_sources_with_three_pairs() -> None:
    decisions = []
    for source in VALIDATION_SOURCE_INDICES:
        decisions.extend(
            {
                "source_index": source,
                "target_index": source + offset + 2,
                "guarded_accept": True,
            }
            for offset in range(MINIMUM_DIRECT_PAIR_COUNT)
        )
    decisions.extend(
        {
            "source_index": source,
            "target_index": source + 2,
            "guarded_accept": True,
        }
        for source in (0, *PHASE40_SOURCES, *PHASE41_SOURCES)
    )

    rows = _validation_sources(decisions)

    assert tuple(row.source_index for row in rows) == VALIDATION_SOURCE_INDICES
    assert all(row.rejected_pair_count == 0 for row in rows)
    assert SELECTED_MAXIMUM_NEAREST_ANCHOR_DISTANCE_METERS == 1.5
    assert SELECTED_MAXIMUM_ANCHOR_PLANE_RESIDUAL_METERS == 0.5
    assert SELECTED_MINIMUM_NORMAL_ALIGNMENT == 0.75
