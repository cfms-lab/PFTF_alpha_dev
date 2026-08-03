import json
from pathlib import Path

from pftf_alpha.integrable_spatial_alpha_audit import (
    evaluate_integrable_spatial_alpha_audit,
    write_audit,
)


def test_phase47_audit_passes_bounded_analytic_controls() -> None:
    result = evaluate_integrable_spatial_alpha_audit()

    assert result.passed_control_count == 8
    assert result.total_control_count == 8
    assert result.connectivity_symmetric_difference_count > 0
    assert result.analytic_integrable_spatial_spd_complex_supported
    assert not result.arbitrary_point_local_spd_complex_supported
    assert not result.pftf_conditioned_spatial_alpha_supported
    assert not result.exact_integrable_spatial_predicates_supported
    assert not result.spatial_alpha_reconstruction_advantage_supported
    assert not result.spatial_alpha_topology_correctness_supported
    assert not result.spatial_alpha_real_scan_transfer_supported
    assert not result.spatial_alpha_deployment_supported


def test_phase47_audit_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_audit(first)
    second_digest = write_audit(second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["analytic_integrable_spatial_spd_complex_supported"] is True
    assert payload["arbitrary_point_local_spd_complex_supported"] is False
