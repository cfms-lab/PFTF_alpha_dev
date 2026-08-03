import json

import pytest

from pftf_alpha.confidence_power_alpha_benchmark import (
    EXPECTED_PROTOCOL_SHA256,
    _prepare_confidence_power,
    calibrate_confidence_power_penalty,
    verify_protocol,
)
from pftf_alpha.confidence_power_alpha_protocol import (
    make_confidence_power_alpha_panel,
)
from pftf_alpha.synthetic import PanelSplit


def test_phase45_protocol_hash_is_locked() -> None:
    verify_protocol("benchmark-out/confidence_power_alpha_protocol_phase45.json")
    assert len(EXPECTED_PROTOCOL_SHA256) == 64


def test_phase45_protocol_tampering_fails_closed(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"artifact_schema": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_protocol(path)


def test_phase45_small_calibration_selects_valid_penalty() -> None:
    case = make_confidence_power_alpha_panel(PanelSplit.CALIBRATION)[0]

    rows, selected = calibrate_confidence_power_penalty(
        (case,),
        penalty_scales=(0.125, 0.25),
        sample_count=48,
    )

    assert len(rows) == 2
    assert all(row.valid for row in rows)
    assert selected in (0.125, 0.25)
    assert all(row.mean_objective is not None for row in rows)


def test_phase45_large_penalty_fails_closed_to_m1() -> None:
    case = make_confidence_power_alpha_panel(PanelSplit.CALIBRATION)[0]

    prepared = _prepare_confidence_power(case, 50.0, allow_fallback=True)

    assert prepared.fallback_to_m1 is True
    assert prepared.connectivity_changed_from_m1 is False
    assert prepared.connectivity_jaccard_distance == 0.0
