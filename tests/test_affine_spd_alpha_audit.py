import json
from pathlib import Path

from pftf_alpha.affine_spd_alpha_audit import (
    evaluate_affine_spd_alpha_audit,
    write_audit,
)


def test_phase46_audit_passes_global_controls_and_rejects_local_field() -> None:
    result = evaluate_affine_spd_alpha_audit()

    assert result.passed_control_count == 5
    assert result.total_control_count == 5
    assert result.global_affine_spd_complex_supported
    assert not result.spatially_varying_spd_complex_supported
    assert not result.point_local_alpha_field_supported
    rejected = next(
        control
        for control in result.controls
        if control.name == "rotating_local_field_fails_closed"
    )
    assert rejected.expected_rejection
    assert rejected.observed_rejection
    assert rejected.comparison is None
    assert rejected.metric_maximum_relative_deviation > 0.0


def test_phase46_audit_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_audit(first)
    second_digest = write_audit(second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["global_affine_spd_complex_supported"] is True
    assert payload["spatially_varying_spd_complex_supported"] is False
