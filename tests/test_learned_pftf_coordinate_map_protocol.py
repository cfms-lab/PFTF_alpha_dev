import json
from pathlib import Path

from pftf_alpha.learned_pftf_coordinate_map_protocol import (
    preregister_learned_pftf_coordinate_map,
    write_protocol,
)


def test_phase48_protocol_freezes_splits_comparators_and_claim_boundary() -> None:
    protocol = preregister_learned_pftf_coordinate_map()

    assert len(protocol.train_seeds) * len(protocol.train_strengths) * 3 == 60
    assert (
        len(protocol.calibration_seeds)
        * len(protocol.calibration_strengths)
        * 3
        == 30
    )
    assert len(protocol.held_out_seeds) * len(protocol.held_out_strengths) * 3 == 45
    assert set(protocol.train_seeds).isdisjoint(protocol.held_out_seeds)
    assert "relation_xy_mean" in protocol.pftf_features
    assert "non-PFTF" in protocol.frozen_comparators[2]
    assert "strictly beat" in protocol.pftf_value_gate
    assert "coordinate-aligned" in protocol.claim_boundary
    assert "real-scan transfer" in protocol.claim_boundary


def test_phase48_protocol_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_protocol(first)
    second_digest = write_protocol(second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["artifact_schema"].endswith("phase48/v1")
    assert payload["held_out_rule"].startswith("evaluate the frozen models once")
