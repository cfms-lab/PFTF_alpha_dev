import hashlib
import json
import math
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

import pftf_alpha.scene_relative_rotation_validation as validation
from pftf_alpha.scene_relative_rotation_guard import UNTOUCHED_VALIDATION_SCENES
from pftf_alpha.scene_relative_rotation_validation import (
    ThreeDMatchEvaluationSpec,
    evaluate_scene_relative_rotation_validation,
    verify_evaluation_archive,
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


def _write_archive(
    root: Path,
    scene_name: str,
    *,
    fail: bool,
) -> tuple[Path, ThreeDMatchEvaluationSpec]:
    evaluation_name = f"{scene_name}-evaluation"
    path = root / f"{evaluation_name}.zip"
    ground_truth_indices = (*range(8), 9) if fail else tuple(range(8))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{evaluation_name}/", "")
        archive.writestr(
            f"{evaluation_name}/gt.info",
            _registration_info(ground_truth_indices),
        )
        archive.writestr(
            f"{evaluation_name}/gt.log",
            _registration_log(ground_truth_indices),
        )
        archive.writestr(
            f"{evaluation_name}/3dmatch.log",
            _registration_log(range(10)),
        )
    payload = path.read_bytes()
    return path, ThreeDMatchEvaluationSpec(
        scene_name=scene_name,
        evaluation_name=evaluation_name,
        archive_name=path.name,
        url=f"https://example.invalid/{path.name}",
        md5=hashlib.md5(payload).hexdigest(),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _write_phase34(root: Path) -> Path:
    path = root / "phase34.json"
    path.write_text(
        json.dumps(
            {
                "artifact_schema": (
                    "pftf_alpha_scene_relative_rotation_guard_phase34/v1"
                ),
                "feature_name": (
                    "scene_relative_prediction_rotation_midrank_percentile"
                ),
                "rotation_percentile_cutoff": 0.90,
                "minimum_correct_retention": 0.90,
                "minimum_incorrect_rejection": 0.10,
                "phase34_design_supported": True,
                "held_out_validation_artifacts_accessed": False,
                "held_out_validation_supported": False,
                "cross_scene_real_registration_supported": False,
                "untouched_validation_scenes": list(
                    UNTOUCHED_VALIDATION_SCENES
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_panel(
    root: Path,
    *,
    failed_scene_index: int | None = None,
) -> tuple[Path, tuple[ThreeDMatchEvaluationSpec, ...]]:
    specs = tuple(
        _write_archive(
            root,
            scene_name,
            fail=index == failed_scene_index,
        )[1]
        for index, scene_name in enumerate(UNTOUCHED_VALIDATION_SCENES)
    )
    return _write_phase34(root), specs


def test_phase35_verifies_exact_evaluation_archive(tmp_path: Path) -> None:
    path, spec = _write_archive(
        tmp_path,
        UNTOUCHED_VALIDATION_SCENES[0],
        fail=False,
    )

    verified = verify_evaluation_archive(path, spec)

    assert verified.verified is True
    assert verified.file_count == 3
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify_evaluation_archive(path, spec)


def test_phase35_materializes_all_scenes_before_reading_any_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase34, specs = _write_panel(tmp_path)
    monkeypatch.setattr(
        validation,
        "EXPECTED_PHASE34_SHA256",
        hashlib.sha256(phase34.read_bytes()).hexdigest(),
    )
    events: list[str] = []
    original_reader = validation._read_archive_member

    def observed_reader(*args: object, **kwargs: object) -> str:
        member_name = str(args[2])
        events.append(member_name)
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(validation, "_read_archive_member", observed_reader)

    result = evaluate_scene_relative_rotation_validation(
        tmp_path,
        phase34,
        scenes=specs,
    )

    assert events[:6] == ["3dmatch.log"] * 6
    assert all(name in {"gt.log", "gt.info"} for name in events[6:])
    assert result.phase35_validation_supported is True
    assert result.cross_scene_real_registration_supported is True
    assert result.real_registration_labels_supported is True
    assert all(
        summary.scene_validation_gate_passed
        for summary in result.scene_summaries
    )


def test_phase35_any_failed_scene_fails_the_whole_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase34, specs = _write_panel(tmp_path, failed_scene_index=3)
    monkeypatch.setattr(
        validation,
        "EXPECTED_PHASE34_SHA256",
        hashlib.sha256(phase34.read_bytes()).hexdigest(),
    )

    result = evaluate_scene_relative_rotation_validation(
        tmp_path,
        phase34,
        scenes=specs,
    )

    assert result.scene_summaries[3].scene_validation_gate_passed is False
    assert result.phase35_validation_supported is False
    assert result.held_out_validation_supported is False
    assert result.cross_scene_real_registration_supported is False


def test_phase35_rejects_changed_frozen_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase34, specs = _write_panel(tmp_path)
    payload = json.loads(phase34.read_text(encoding="utf-8"))
    payload["rotation_percentile_cutoff"] = 0.91
    phase34.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        validation,
        "EXPECTED_PHASE34_SHA256",
        hashlib.sha256(phase34.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="rotation_percentile_cutoff"):
        evaluate_scene_relative_rotation_validation(
            tmp_path,
            phase34,
            scenes=specs,
        )
