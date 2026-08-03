import json
from pathlib import Path

import pytest

import pftf_alpha.eth_gazebo_rotation_decisions as decisions
from pftf_alpha.eth_gazebo_validation_protocol import EXPECTED_PAIR_COUNT


def test_phase39_gazebo_decision_input_is_hash_locked(tmp_path: Path) -> None:
    path = tmp_path / "changed.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        decisions.materialize_gazebo_decisions(path)


def test_phase39_gazebo_tied_rotations_share_midrank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "predictions.json"
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    payload = {
        "artifact_schema": "pftf_alpha_eth_gazebo_predictions_phase39/v1",
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "complete_prediction_set_materialized": True,
        "validation_label_member_opened": False,
        "predictions": [
            {
                "source_index": index,
                "target_index": index + 2,
                "target_to_source_matrix": matrix,
            }
            for index in range(EXPECTED_PAIR_COUNT)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        decisions,
        "_sha256",
        lambda value: decisions.EXPECTED_PREDICTION_SHA256,
    )

    result = decisions.materialize_gazebo_decisions(path)

    assert result.accepted_count == EXPECTED_PAIR_COUNT
    assert result.rejected_count == 0
    assert result.validation_label_values_accessed is False
