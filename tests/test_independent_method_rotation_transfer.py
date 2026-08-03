import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

import pftf_alpha.independent_method_rotation_transfer as transfer
from pftf_alpha.independent_method_rotation_transfer import (
    PHASE36_SCENES,
    SourceFileSpec,
    SyntheticMethodSceneSpec,
    evaluate_independent_method_rotation_transfer,
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


def _registration_log(indices: Sequence[int]) -> str:
    lines = []
    for index in indices:
        lines.append(f"{index}\t{index + 2}\t20")
        lines.extend(
            "\t".join(f"{value:.12f}" for value in row)
            for row in _matrix(0.05 * index)
        )
    return "\n".join(lines) + "\n"


def _registration_info(indices: Sequence[int]) -> str:
    lines = []
    for index in indices:
        lines.append(f"{index}\t{index + 2}\t20")
        lines.extend(
            "\t".join("1.0" if row == column else "0.0" for column in range(6))
            for row in range(6)
        )
    return "\n".join(lines) + "\n"


def _file_spec(path: Path) -> SourceFileSpec:
    payload = path.read_bytes()
    return SourceFileSpec(
        file_name=path.name,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _write_scene(
    root: Path,
    template: SyntheticMethodSceneSpec,
    *,
    fail: bool,
) -> SyntheticMethodSceneSpec:
    scene_root = root / template.evaluation_name
    scene_root.mkdir(parents=True)
    ground_truth_indices = (*range(8), 9) if fail else tuple(range(8))
    paths = {
        "fpfh": scene_root / "fpfh.log",
        "spin": scene_root / "spin.log",
        "ground_truth_log": scene_root / "gt.log",
        "ground_truth_info": scene_root / "gt.info",
    }
    paths["fpfh"].write_text(_registration_log(range(10)), encoding="ascii")
    paths["spin"].write_text(_registration_log(range(10)), encoding="ascii")
    paths["ground_truth_log"].write_text(
        _registration_log(ground_truth_indices),
        encoding="ascii",
    )
    paths["ground_truth_info"].write_text(
        _registration_info(ground_truth_indices),
        encoding="ascii",
    )
    return SyntheticMethodSceneSpec(
        scene_name=template.scene_name,
        evaluation_name=template.evaluation_name,
        fpfh_log=_file_spec(paths["fpfh"]),
        spin_log=_file_spec(paths["spin"]),
        ground_truth_log=_file_spec(paths["ground_truth_log"]),
        ground_truth_info=_file_spec(paths["ground_truth_info"]),
    )


def _write_phase35(root: Path) -> Path:
    path = root / "phase35.json"
    path.write_text(
        json.dumps(
            {
                "artifact_schema": (
                    "pftf_alpha_scene_relative_rotation_validation_phase35/v1"
                ),
                "feature_name": (
                    "scene_relative_prediction_rotation_midrank_percentile"
                ),
                "rotation_percentile_cutoff": 0.90,
                "minimum_correct_retention": 0.90,
                "minimum_incorrect_rejection": 0.10,
                "phase35_validation_supported": True,
                "held_out_validation_supported": True,
                "cross_scene_real_registration_supported": True,
                "real_registration_labels_supported": True,
                "real_correspondence_supported": False,
                "real_trimmed_reconstruction_supported": False,
                "deployment_supported": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_panel(
    root: Path,
    *,
    failed_scene_index: int | None = None,
) -> tuple[Path, tuple[SyntheticMethodSceneSpec, ...]]:
    scenes = tuple(
        _write_scene(root, template, fail=index == failed_scene_index)
        for index, template in enumerate(PHASE36_SCENES)
    )
    return _write_phase35(root), scenes


def test_phase36_materializes_all_predictions_before_any_label_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase35, scenes = _write_panel(tmp_path)
    monkeypatch.setattr(
        transfer,
        "EXPECTED_PHASE35_SHA256",
        hashlib.sha256(phase35.read_bytes()).hexdigest(),
    )
    events: list[str] = []
    original_reader = transfer._read_ascii

    def observed_reader(path: Path) -> str:
        events.append(path.name)
        return original_reader(path)

    monkeypatch.setattr(transfer, "_read_ascii", observed_reader)

    result = evaluate_independent_method_rotation_transfer(
        tmp_path,
        phase35,
        scenes=scenes,
    )

    assert events[:8] == ["fpfh.log", "spin.log"] * 4
    assert all(name in {"gt.log", "gt.info"} for name in events[8:])
    assert result.phase36_panel_supported is True
    assert result.independent_method_transfer_supported is True
    assert result.independent_end_to_end_pipeline_transfer_supported is False
    assert result.cross_benchmark_transfer_supported is True
    assert result.external_method_generation_reproduced is False
    assert all(summary.block_transfer_gate_passed for summary in result.block_summaries)


def test_phase36_any_failed_method_scene_block_fails_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase35, scenes = _write_panel(tmp_path, failed_scene_index=2)
    monkeypatch.setattr(
        transfer,
        "EXPECTED_PHASE35_SHA256",
        hashlib.sha256(phase35.read_bytes()).hexdigest(),
    )

    result = evaluate_independent_method_rotation_transfer(
        tmp_path,
        phase35,
        scenes=scenes,
    )

    assert any(
        not summary.block_transfer_gate_passed
        for summary in result.block_summaries
    )
    assert result.phase36_panel_supported is False
    assert result.independent_method_transfer_supported is False
    assert result.cross_benchmark_transfer_supported is False


def test_phase36_rejects_changed_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase35, scenes = _write_panel(tmp_path)
    monkeypatch.setattr(
        transfer,
        "EXPECTED_PHASE35_SHA256",
        hashlib.sha256(phase35.read_bytes()).hexdigest(),
    )
    target = tmp_path / scenes[0].evaluation_name / "spin.log"
    target.write_text(target.read_text(encoding="ascii") + "changed", encoding="ascii")

    with pytest.raises(ValueError, match="byte count mismatch"):
        evaluate_independent_method_rotation_transfer(
            tmp_path,
            phase35,
            scenes=scenes,
        )


def test_phase36_rejects_changed_predecessor_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase35, scenes = _write_panel(tmp_path)
    payload = json.loads(phase35.read_text(encoding="utf-8"))
    payload["minimum_incorrect_rejection"] = 0.11
    phase35.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        transfer,
        "EXPECTED_PHASE35_SHA256",
        hashlib.sha256(phase35.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="minimum_incorrect_rejection"):
        evaluate_independent_method_rotation_transfer(
            tmp_path,
            phase35,
            scenes=scenes,
        )
