import json
from pathlib import Path

from pftf_alpha.pftf_shear_identifiability_protocol import (
    preregister_pftf_shear_identifiability,
    write_protocol,
)


def test_phase49_protocol_freezes_diagnostic_and_information_boundaries() -> None:
    protocol = preregister_pftf_shear_identifiability()

    assert len(protocol.train_seeds) * len(protocol.families) == 12
    assert len(protocol.calibration_seeds) * len(protocol.families) == 6
    assert "prohibited" in protocol.held_out_prohibition
    assert "lexicographically" in protocol.training_feature_selection
    assert protocol.minimum_calibration_median_within_block_r2 == 0.75
    assert protocol.minimum_calibration_sign_consistency == 5.0 / 6.0
    assert "standalone gate" in protocol.next_panel_gate
    assert "not a deployable estimator" in protocol.reference_boundary


def test_phase49_protocol_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_protocol(first)
    second_digest = write_protocol(second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["artifact_schema"].endswith("phase49/v1")
    assert payload["role"].startswith("train_calibration_only")
