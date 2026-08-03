import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import pftf_alpha.scene_relative_rotation_guard as rotation_guard
from pftf_alpha.scene_relative_rotation_guard import (
    empirical_midrank_percentiles,
    evaluate_scene_relative_rotation_guard,
    prediction_rotation_radians,
)


def _rotation_matrix(angle: float) -> list[list[float]]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        [cosine, -sine, 0.0, 0.0],
        [sine, cosine, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _observations() -> list[dict[str, object]]:
    labels = (True,) * 8 + (False, False)
    return [
        {
            "source_index": index,
            "target_index": index + 2,
            "prediction_matrix": _rotation_matrix(index * 0.05),
            "official_correct": label,
        }
        for index, label in enumerate(labels)
    ]


def _write_artifacts(root: Path) -> tuple[Path, Path]:
    phase32 = root / "phase32.json"
    phase32.write_text(
        json.dumps(
            {
                "artifact_schema": (
                    "pftf_alpha_threedmatch_registration_guard_phase32/v1"
                ),
                "dataset_name": "7-scenes-redkitchen",
                "real_registration_labels_supported": True,
                "ground_truth_overlap_pair_count": 8,
                "observations": _observations(),
            }
        ),
        encoding="utf-8",
    )
    phase33 = root / "phase33.json"
    phase33.write_text(
        json.dumps(
            {
                "artifact_schema": (
                    "pftf_alpha_threedmatch_transfer_audit_phase33/v1"
                ),
                "scene": {
                    "scene_name": "sun3d-hotel_umd-maryland_hotel3",
                },
                "real_registration_labels_supported": True,
                "ground_truth_overlap_pair_count": 8,
                "observations": _observations(),
            }
        ),
        encoding="utf-8",
    )
    return phase32, phase33


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prediction_rotation_radians_recovers_principal_angle() -> None:
    matrix = _rotation_matrix(0.75)

    assert prediction_rotation_radians(matrix) == pytest.approx(0.75)


def test_empirical_midrank_percentiles_share_tied_rank() -> None:
    percentiles = empirical_midrank_percentiles((0.0, 1.0, 1.0, 3.0))

    np.testing.assert_allclose(percentiles, (0.125, 0.5, 0.5, 0.875))


def test_phase34_materializes_both_scenes_before_joining_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase32, phase33 = _write_artifacts(tmp_path)
    monkeypatch.setattr(
        rotation_guard,
        "EXPECTED_PHASE32_SHA256",
        _sha256(phase32),
    )
    monkeypatch.setattr(
        rotation_guard,
        "EXPECTED_PHASE33_SHA256",
        _sha256(phase33),
    )
    events: list[str] = []
    original_blind = rotation_guard._blind_scene_observations
    original_join = rotation_guard._join_scene_labels

    def observed_blind(*args: object, **kwargs: object):
        events.append("blind")
        return original_blind(*args, **kwargs)

    def observed_join(*args: object, **kwargs: object):
        events.append("labels")
        return original_join(*args, **kwargs)

    monkeypatch.setattr(
        rotation_guard,
        "_blind_scene_observations",
        observed_blind,
    )
    monkeypatch.setattr(rotation_guard, "_join_scene_labels", observed_join)

    result = evaluate_scene_relative_rotation_guard(phase32, phase33)

    assert events == ["blind", "blind", "labels", "labels"]
    assert result.phase34_design_supported is True
    assert all(
        summary.scene_design_gate_passed for summary in result.scene_summaries
    )
    assert result.held_out_validation_artifacts_accessed is False
    assert result.held_out_validation_supported is False
    assert result.cross_scene_real_registration_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_trimmed_reconstruction_supported is False
    assert result.deployment_supported is False


def test_phase34_rejects_changed_input_artifact(tmp_path: Path) -> None:
    phase32, phase33 = _write_artifacts(tmp_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_scene_relative_rotation_guard(phase32, phase33)
