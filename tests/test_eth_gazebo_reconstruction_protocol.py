from pathlib import Path

import pftf_alpha.eth_gazebo_reconstruction_protocol as protocol


def _decision(source: int, target: int, accept: bool) -> dict[str, object]:
    return {
        "source_index": source,
        "target_index": target,
        "guarded_accept": accept,
    }


def test_phase40_source_selection_is_label_free_and_excludes_development() -> None:
    rows: list[dict[str, object]] = []
    for source in (0, *protocol.VALIDATION_SOURCE_INDICES):
        rows.extend(
            (
                _decision(source, source + 2, True),
                _decision(source, source + 3, False),
            )
        )

    plans = protocol._source_plans(rows)

    assert tuple(plan.source_index for plan in plans) == (
        protocol.VALIDATION_SOURCE_INDICES
    )
    assert protocol.DEVELOPMENT_SOURCE_INDEX not in {
        plan.source_index for plan in plans
    }
    assert all(plan.accepted_count == plan.rejected_count == 1 for plan in plans)


def test_phase40_protocol_never_loads_registration_audit(
    monkeypatch,
) -> None:
    verification = type("Verification", (), {"sha256": "archive-hash"})()
    decisions = [
        _decision(source, source + offset, accept)
        for source in (0, *protocol.VALIDATION_SOURCE_INDICES)
        for offset, accept in ((2, True), (3, False))
    ]
    payloads = {
        protocol.PREDICTION_SCHEMA: {
            "artifact_schema": protocol.PREDICTION_SCHEMA,
            "validation_label_member_opened": False,
        },
        protocol.DECISION_SCHEMA: {
            "artifact_schema": protocol.DECISION_SCHEMA,
            "validation_label_values_accessed": False,
            "decisions": decisions,
        },
    }

    monkeypatch.setattr(
        protocol,
        "verify_gazebo_archive_directory",
        lambda path: verification,
    )
    monkeypatch.setattr(
        protocol,
        "_load_locked_json",
        lambda path, *, expected_sha256, expected_schema: payloads[expected_schema],
    )

    result = protocol.preregister_gazebo_reconstruction(
        Path(protocol.ARCHIVE_NAME),
        Path("predictions.json"),
        Path("decisions.json"),
    )

    assert result.registration_label_values_accessed is False
    assert result.validation_reference_values_accessed is False
    assert result.development_source_index == 0
    assert tuple(row.source_index for row in result.validation_sources) == (
        protocol.VALIDATION_SOURCE_INDICES
    )
    assert "pose_scanner_leica.csv" in result.label_boundary


def test_phase40_alpha_is_frozen_from_registration_voxel() -> None:
    assert protocol.ALPHA_METERS == 2.0 * protocol.SOURCE_VOXEL_METERS
    assert protocol.REFERENCE_VOXEL_METERS < protocol.SOURCE_VOXEL_METERS
    assert protocol.FUSION_VOXEL_METERS > protocol.SOURCE_VOXEL_METERS
    assert protocol.HELDOUT_MODULUS == 5
    assert protocol.HELDOUT_REMAINDER == 0
