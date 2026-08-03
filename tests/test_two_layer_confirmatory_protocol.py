import json
from pathlib import Path

from pftf_alpha.reacquisition import ReacquisitionConfig
from pftf_alpha.two_layer_confirmatory_protocol import (
    M1_CALIBRATION_SHA256,
    preregister_two_layer_confirmatory,
    write_protocol,
)


def test_phase50_protocol_freezes_untouched_positive_scope() -> None:
    protocol = preregister_two_layer_confirmatory()

    assert protocol.expected_case_count == 144
    assert protocol.point_counts == (160, 256)
    assert len(protocol.stresses) == 6
    assert not any("outlier" in stress for stress in protocol.stresses)
    assert protocol.held_out_seed == 35_000_804
    assert protocol.b5_scale_multiplier == ReacquisitionConfig().b5_scale_multiplier
    assert protocol.m1_weight_scale == 0.375
    assert protocol.m1_scale_multiplier == 2.5009326930224836
    assert len(M1_CALIBRATION_SHA256) == 64
    assert protocol.minimum_overall_safe_acceptance == 0.95
    assert protocol.minimum_subgroup_safe_acceptance == 0.90
    assert protocol.minimum_mean_fscore_margin == 0.10
    assert protocol.minimum_casewise_fscore_win_rate == 0.75
    assert "no Phase-50" in protocol.held_out_prohibition
    assert "PFTF or local-SPD superiority" in protocol.excluded_scope


def test_phase50_protocol_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_protocol(first)
    second_digest = write_protocol(second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["artifact_schema"].endswith("phase50/v1")
    assert payload["role"].startswith("untouched_bounded_positive")
    assert payload["expected_case_count"] == 144
