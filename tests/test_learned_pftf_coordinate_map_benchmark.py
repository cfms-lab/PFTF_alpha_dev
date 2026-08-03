import json
from pathlib import Path

from pftf_alpha.learned_pftf_coordinate_map import (
    evaluate_learned_pftf_coordinate_map,
    write_result,
)


def test_phase48_benchmark_obeys_frozen_boundaries() -> None:
    result = evaluate_learned_pftf_coordinate_map()

    assert result.train_case_count == 60
    assert result.calibration_case_count == 30
    assert result.held_out_case_count == 45
    assert result.construction_gate_passed
    assert result.bounded_prediction_count == result.total_prediction_count
    assert result.maximum_inverse_roundtrip_error <= 1.0e-12
    assert result.minimum_jacobian_determinant == 1.0
    assert result.maximum_jacobian_determinant == 1.0
    assert not result.arbitrary_point_local_spd_complex_supported
    assert not result.general_nonlinear_map_learner_supported
    assert not result.global_alpha_selection_supported
    assert not result.reconstruction_advantage_supported
    assert not result.real_scan_transfer_supported
    assert not result.deployment_supported


def test_phase48_benchmark_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_result(first)
    second_digest = write_result(second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["artifact_schema"].endswith("phase48/v1")
    assert payload["held_out_case_count"] == 45
