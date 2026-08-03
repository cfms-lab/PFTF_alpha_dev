import json
import math
from pathlib import Path

import numpy as np
import pytest

import pftf_alpha.independent_pipeline_rotation_audit as audit
from pftf_alpha.independent_pipeline_rotation_audit import (
    evaluate_independent_pipeline_rotation_audit,
)
from pftf_alpha.open3d_fgr_pipeline import (
    OPEN3D_VERSION,
    PHASE37_SCENE_INPUTS,
    nonconsecutive_fragment_pairs,
    phase37_parameters,
)
from pftf_alpha.threedmatch_redkitchen import (
    RegistrationInfoEntry,
    RegistrationLogEntry,
    ThreeDMatchArchiveVerification,
)


def _matrix(angle: float) -> list[list[float]]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        [cosine, -sine, 0.0, 0.0],
        [sine, cosine, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _prediction_scene(scene_index: int) -> dict[str, object]:
    scene = PHASE37_SCENE_INPUTS[scene_index]
    pairs = nonconsecutive_fragment_pairs(scene.fragment_count)
    predictions = [
        {
            "source_index": source,
            "target_index": target,
            "fragment_count": scene.fragment_count,
            "pair_random_seed": 370803 + index,
            "benchmark_target_to_source_matrix": _matrix(index / len(pairs)),
            "fitness": 0.1,
            "inlier_rmse": 0.01,
            "correspondence_count": 12,
        }
        for index, (source, target) in enumerate(pairs)
    ]
    return {
        "scene": scene.to_dict(),
        "source_root": f"scene-{scene_index}",
        "fragment_verification": {},
        "pair_universe": ("all_source_lt_target_pairs_with_target_minus_source_gt_1"),
        "expected_pair_count": len(pairs),
        "preprocessed_fragments": [],
        "predictions": predictions,
    }


def _write_predictions(root: Path) -> Path:
    path = root / "predictions.json"
    path.write_text(
        json.dumps(
            {
                "artifact_schema": ("pftf_alpha_open3d_fgr_predictions_phase37/v2"),
                "role": "test",
                "pipeline_name": ("open3d_0.19.0_fpfh_fast_global_registration"),
                "label_boundary": "test",
                "pair_selection_rule": "test",
                "matrix_convention": "test",
                "generation_correction_history": "test correction",
                "open3d_version": OPEN3D_VERSION,
                "python_version": "3.12.test",
                "platform": "test",
                "open3d_build_config": {},
                "official_parameter_source": "test",
                "parameters": phase37_parameters(),
                "scenes": [_prediction_scene(0), _prediction_scene(1)],
                "external_method_generation_reproduced": True,
                "ground_truth_artifacts_accessed_by_generator": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _labels_for_scene(
    scene_index: int,
    *,
    fail: bool,
) -> tuple[tuple[RegistrationLogEntry, ...], tuple[RegistrationInfoEntry, ...]]:
    scene = PHASE37_SCENE_INPUTS[scene_index]
    pairs = nonconsecutive_fragment_pairs(scene.fragment_count)
    selected = pairs[-8:] if fail else pairs[:8]
    logs = []
    infos = []
    for pair in selected:
        prediction_index = pairs.index(pair)
        logs.append(
            RegistrationLogEntry(
                source_index=pair[0],
                target_index=pair[1],
                fragment_count=scene.fragment_count,
                source_to_target_matrix=_matrix(prediction_index / len(pairs)),
            )
        )
        infos.append(
            RegistrationInfoEntry(
                source_index=pair[0],
                target_index=pair[1],
                fragment_count=scene.fragment_count,
                information_matrix=np.eye(6),
            )
        )
    return tuple(logs), tuple(infos)


def _patch_external_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_scene_index: int | None = None,
    events: list[str] | None = None,
) -> None:
    monkeypatch.setattr(audit, "_verify_phase36", lambda path: {})

    def verify(root: Path, scene: object) -> ThreeDMatchArchiveVerification:
        if events is not None:
            events.append("verify-label")
        return ThreeDMatchArchiveVerification(
            role="evaluation",
            archive_path=str(root),
            byte_count=1,
            md5="0" * 32,
            sha256="0" * 64,
            file_count=3,
            verified=True,
        )

    def labels(root: Path, scene: object) -> tuple[object, object]:
        if events is not None:
            events.append("decode-label")
        index = int(str(root).rsplit("-", 1)[-1])
        return _labels_for_scene(index, fail=index == failed_scene_index)

    monkeypatch.setattr(audit, "_verify_evaluation_archive", verify)
    monkeypatch.setattr(audit, "_labels", labels)


def test_nonconsecutive_pair_universe_is_complete() -> None:
    assert len(nonconsecutive_fragment_pairs(60)) == 1711
    assert len(nonconsecutive_fragment_pairs(37)) == 630
    assert nonconsecutive_fragment_pairs(4) == ((0, 2), (0, 3), (1, 3))


def test_phase37_materializes_both_guard_sets_before_label_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = _write_predictions(tmp_path)
    phase36 = tmp_path / "phase36.json"
    phase36.write_text("{}", encoding="utf-8")
    events: list[str] = []
    original_blind = audit._blind_scene

    def blind(*args: object, **kwargs: object) -> object:
        events.append("blind")
        return original_blind(*args, **kwargs)

    monkeypatch.setattr(audit, "_blind_scene", blind)
    _patch_external_inputs(monkeypatch, events=events)

    result = evaluate_independent_pipeline_rotation_audit(
        predictions,
        phase36,
        tmp_path / "scene-0",
        tmp_path / "scene-1",
    )

    assert events[:2] == ["blind", "blind"]
    assert all(event != "blind" for event in events[2:])
    assert result.phase37_fixed_parameter_audit_supported is True
    assert result.independent_end_to_end_pipeline_transfer_supported is True
    assert result.fresh_label_blind_validation_supported is False
    assert result.independent_algorithm_implementation_supported is False


def test_phase37_any_failed_scene_fails_the_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = _write_predictions(tmp_path)
    phase36 = tmp_path / "phase36.json"
    phase36.write_text("{}", encoding="utf-8")
    _patch_external_inputs(monkeypatch, failed_scene_index=1)

    result = evaluate_independent_pipeline_rotation_audit(
        predictions,
        phase36,
        tmp_path / "scene-0",
        tmp_path / "scene-1",
    )

    assert result.scene_summaries[0].scene_transfer_gate_passed is True
    assert result.scene_summaries[1].scene_transfer_gate_passed is False
    assert result.phase37_fixed_parameter_audit_supported is False
    assert result.independent_end_to_end_pipeline_transfer_supported is False


def test_phase37_rejects_changed_generation_protocol(
    tmp_path: Path,
) -> None:
    predictions = _write_predictions(tmp_path)
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    payload["parameters"]["voxel_size_meters"] = 0.051
    predictions.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="parameters"):
        audit._verify_prediction_artifact(predictions)
