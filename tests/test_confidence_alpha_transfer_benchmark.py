import json

import numpy as np
import pytest

from pftf_alpha.adaptive import AdaptiveCellFiltration
from pftf_alpha.confidence_alpha_panel import ReferenceSurfaceFamily
from pftf_alpha.confidence_alpha_transfer_benchmark import (
    EXPECTED_PROTOCOL_SHA256,
    METHOD_ORDER,
    calibrate_transfer_methods,
    complete_critical_gap_threshold,
    verify_protocol,
)
from pftf_alpha.confidence_alpha_transfer_panel import (
    TransferStressProfile,
    make_confidence_alpha_transfer_case,
)
from pftf_alpha.synthetic import PanelSplit


def _adaptive(scores: np.ndarray) -> AdaptiveCellFiltration:
    cell_count = scores.shape[0]
    points = np.arange((cell_count + 3) * 3, dtype=float).reshape(-1, 3)
    cells = np.asarray(
        [[index, index + 1, index + 2, index + 3] for index in range(cell_count)]
    )
    return AdaptiveCellFiltration(
        points=points,
        top_simplices=cells,
        scores=scores,
        method="test",
        diagnostics={},
    )


def test_phase44_complete_scan_selects_largest_eligible_log_gap() -> None:
    scores = np.asarray([1.0, 1.1, 1.2, 3.0, 3.1, 3.2, 10.0, 11.0, 12.0, 13.0])

    selected = complete_critical_gap_threshold(_adaptive(scores))

    assert selected.selected_cell_count == 6
    assert selected.selected_cell_fraction == 0.6
    assert selected.lower_critical_score == 3.2
    assert selected.upper_critical_score == 10.0
    assert selected.threshold == pytest.approx(np.sqrt(32.0))


def test_phase44_critical_gap_selection_is_scale_invariant() -> None:
    scores = np.asarray([1.0, 1.1, 1.2, 3.0, 3.1, 3.2, 10.0, 11.0])

    original = complete_critical_gap_threshold(scores)
    scaled = complete_critical_gap_threshold(7.5 * scores)

    assert scaled.selected_cell_count == original.selected_cell_count
    assert scaled.selected_cell_fraction == original.selected_cell_fraction
    assert scaled.log_score_gap == pytest.approx(original.log_score_gap)
    assert scaled.threshold == pytest.approx(7.5 * original.threshold)


def test_phase44_protocol_hash_is_locked() -> None:
    verify_protocol("benchmark-out/confidence_alpha_transfer_protocol_phase44.json")
    assert len(EXPECTED_PROTOCOL_SHA256) == 64


def test_phase44_protocol_tampering_fails_closed(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"artifact_schema": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_protocol(path)


def test_phase44_small_calibration_selects_every_method() -> None:
    case = make_confidence_alpha_transfer_case(
        ReferenceSurfaceFamily.SPHERE,
        TransferStressProfile.DENSITY_SHIFT,
        split=PanelSplit.CALIBRATION,
        seed=44_001,
        reference_point_count=128,
    )

    selected = calibrate_transfer_methods(
        (case,),
        continuous_strengths=(1.0,),
        binary_thresholds=(0.5,),
        sample_count=48,
    )

    assert tuple(row.method_id for row in selected) == METHOD_ORDER
    assert all(row.calibration_mean_objective >= 0.0 for row in selected)
    assert all(
        0.5 <= row.calibration_mean_selected_cell_fraction <= 0.98
        for row in selected
    )
