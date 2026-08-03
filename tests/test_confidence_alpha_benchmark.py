import json

import pytest

from pftf_alpha.confidence_alpha_benchmark import (
    EXPECTED_PROTOCOL_SHA256,
    METHOD_ORDER,
    calibrate_confidence_alpha_methods,
    verify_protocol,
)
from pftf_alpha.confidence_alpha_panel import (
    MisregistrationProfile,
    ReferenceSurfaceFamily,
    make_confidence_alpha_case,
)
from pftf_alpha.synthetic import PanelSplit


def test_phase43_protocol_hash_is_locked() -> None:
    path = "benchmark-out/confidence_alpha_panel_protocol_phase43.json"

    verify_protocol(path)
    assert len(EXPECTED_PROTOCOL_SHA256) == 64


def test_phase43_protocol_tampering_fails_closed(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"artifact_schema": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_protocol(path)


def test_phase43_small_calibration_selects_every_method() -> None:
    case = make_confidence_alpha_case(
        ReferenceSurfaceFamily.SPHERE,
        MisregistrationProfile.MILD,
        split=PanelSplit.CALIBRATION,
        seed=43_001,
        points_per_view=28,
        reference_point_count=96,
    )

    selected = calibrate_confidence_alpha_methods(
        (case,),
        scale_quantiles=(0.26,),
        continuous_strengths=(1.0,),
        binary_thresholds=(0.5,),
        sample_count=48,
    )

    assert tuple(row.method_id for row in selected) == METHOD_ORDER
    assert all(row.scale_quantile == 0.26 for row in selected)
    assert all(row.score_threshold > 0.0 for row in selected)
    assert all(row.calibration_mean_objective >= 0.0 for row in selected)
