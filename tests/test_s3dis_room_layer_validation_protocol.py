from __future__ import annotations

import hashlib
from pathlib import Path

from pftf_alpha.s3dis_room_layer_validation_protocol import (
    preregister_s3dis_room_layer_validation,
    write_protocol,
)


def test_phase51c_protocol_freezes_building_disjoint_final_gates() -> None:
    protocol = preregister_s3dis_room_layer_validation()

    assert protocol.held_out_area == "Area_5"
    assert protocol.minimum_eligible_cases == 20
    assert protocol.minimum_safe_acceptance_coverage == 0.90
    assert protocol.minimum_b5_fscore_margin == 0.20
    assert protocol.minimum_m1_fscore_margin == 0.30
    assert protocol.maximum_topology_error_ratio == 0.25
    assert "pftf_superiority_supported" in protocol.always_false_flags
    assert "long-gap" in protocol.claim_boundary


def test_phase51c_protocol_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_protocol(first)
    second_digest = write_protocol(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_digest == second_digest
    assert first_digest == hashlib.sha256(first.read_bytes()).hexdigest()
