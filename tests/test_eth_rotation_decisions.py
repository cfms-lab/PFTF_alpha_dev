import json
from pathlib import Path

import pytest

import pftf_alpha.eth_rotation_decisions as decisions
from pftf_alpha.fresh_external_protocol import EXPECTED_PAIR_COUNT


def test_phase38_decision_input_is_hash_locked(tmp_path: Path) -> None:
    path = tmp_path / "changed.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        decisions.materialize_eth_rotation_decisions(path)


def test_phase38_decisions_are_materialized_before_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "predictions.json"
    rows = [
        {
            "source_index": index,
            "target_index": index + 2,
            "target_to_source_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        for index in range(EXPECTED_PAIR_COUNT)
    ]
    payload = {
        "artifact_schema": "pftf_alpha_eth_open3d_fgr_predictions_phase38/v1",
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "complete_prediction_set_materialized": True,
        "ground_truth_label_member_opened": False,
        "predictions": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        decisions,
        "_sha256",
        lambda value: decisions.EXPECTED_PREDICTION_SHA256,
    )

    result = decisions.materialize_eth_rotation_decisions(path)

    assert len(result.decisions) == EXPECTED_PAIR_COUNT
    assert result.accepted_count == EXPECTED_PAIR_COUNT
    assert result.rejected_count == 0
    assert result.complete_decision_set_materialized is True
    assert result.label_values_accessed is False
