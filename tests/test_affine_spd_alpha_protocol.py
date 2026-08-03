import json
from pathlib import Path

from pftf_alpha.affine_spd_alpha_protocol import (
    preregister_affine_spd_alpha,
    write_protocol,
)


def test_phase46_protocol_freezes_compatibility_and_claim_boundary() -> None:
    protocol = preregister_affine_spd_alpha()

    assert protocol.audit_seed == 46_001
    assert protocol.point_count == 48
    assert "M_i = L L^T" in protocol.compatibility_condition
    assert "y = x L" in protocol.construction
    assert "M_prime = A^-1 M A^-T" in protocol.coordinate_covariance_rule
    assert "spatially rotating" in protocol.frozen_controls[-1]
    assert "performance advantage" in protocol.claim_boundary


def test_phase46_protocol_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_protocol(first)
    second_digest = write_protocol(second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["artifact_schema"].endswith("phase46/v1")
