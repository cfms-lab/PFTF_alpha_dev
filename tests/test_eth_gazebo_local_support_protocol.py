from pftf_alpha.eth_gazebo_local_support_protocol import (
    MINIMUM_DIRECT_PAIR_COUNT,
    SELECTED_MAXIMUM_DISPERSION_METERS,
    SELECTED_MINIMUM_SUPPORT,
    VALIDATION_SOURCE_INDICES,
    _validation_sources,
)
from pftf_alpha.eth_gazebo_reconstruction_protocol import (
    VALIDATION_SOURCE_INDICES as PHASE40_VALIDATION_SOURCE_INDICES,
)


def test_phase41_validation_sources_exclude_phase40_and_development() -> None:
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
        for source in (0, *PHASE40_VALIDATION_SOURCE_INDICES)
    )

    rows = _validation_sources(decisions)

    assert tuple(row.source_index for row in rows) == VALIDATION_SOURCE_INDICES
    assert all(row.rejected_pair_count == 0 for row in rows)
    assert SELECTED_MINIMUM_SUPPORT == 2
    assert SELECTED_MAXIMUM_DISPERSION_METERS == 0.15
