import json
from pathlib import Path

from pftf_alpha.integrable_spatial_alpha_protocol import (
    preregister_integrable_spatial_alpha,
    write_protocol,
)


def test_phase47_protocol_freezes_map_integrability_and_claim_boundary() -> None:
    protocol = preregister_integrable_spatial_alpha()

    assert protocol.audit_seed == 47_001
    assert protocol.point_count == 56
    assert "y + s*x^2" in protocol.coordinate_map
    assert "J_Phi(x) J_Phi(x)^T" in protocol.induced_metric
    assert "mixed-partial" in protocol.local_integrability_condition
    assert "change Delaunay connectivity" in protocol.frozen_controls[5]
    assert "arbitrary point-local metrics" in protocol.claim_boundary


def test_phase47_protocol_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_protocol(first)
    second_digest = write_protocol(second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["artifact_schema"].endswith("phase47/v1")
