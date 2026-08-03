import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import pftf_alpha.threedmatch_redkitchen as redkitchen
import pftf_alpha.threedmatch_registration_guard as registration_guard
from pftf_alpha.matched_guard_signature import SIGNATURE_FEATURE_NAMES
from pftf_alpha.threedmatch_redkitchen import (
    RegistrationInfoEntry,
    RegistrationLogEntry,
    official_transformation_error,
    read_binary_ply_xyz,
)


def _ply_bytes(points: np.ndarray) -> bytes:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    return header + np.asarray(points, dtype="<f4").tobytes()


def _matrix(translation: float) -> np.ndarray:
    result = np.eye(4)
    result[0, 3] = translation
    return result


def _log_record(source: int, target: int, count: int, matrix: np.ndarray) -> str:
    rows = [f"{source} {target} {count}"]
    rows.extend(" ".join(f"{value:.12g}" for value in row) for row in matrix)
    return "\n".join(rows)


def _info_record(source: int, target: int, count: int) -> str:
    rows = [f"{source} {target} {count}"]
    matrix = np.eye(6) * 5000.0
    rows.extend(" ".join(f"{value:.12g}" for value in row) for row in matrix)
    return "\n".join(rows)


def _write_archives(root: Path, points: np.ndarray) -> tuple[Path, Path]:
    fragments_path = root / redkitchen.FRAGMENT_ARCHIVE_NAME
    with zipfile.ZipFile(fragments_path, "w") as archive:
        archive.writestr(f"{redkitchen.SCENE_NAME}/", b"")
        clouds = (
            points,
            points,
            points + (1.0, 0.0, 0.0),
            points,
        )
        for index, cloud in enumerate(clouds):
            archive.writestr(
                f"{redkitchen.SCENE_NAME}/cloud_bin_{index}.ply",
                _ply_bytes(cloud),
            )
    evaluation_path = root / redkitchen.EVALUATION_ARCHIVE_NAME
    prediction_log = "\n".join(
        (
            _log_record(0, 2, 4, _matrix(1.0)),
            _log_record(0, 3, 4, _matrix(0.0)),
        )
    )
    ground_truth_log = "\n".join(
        (
            _log_record(0, 1, 4, _matrix(0.0)),
            _log_record(0, 2, 4, _matrix(1.0)),
        )
    )
    ground_truth_info = "\n".join(
        (
            _info_record(0, 1, 4),
            _info_record(0, 2, 4),
        )
    )
    with zipfile.ZipFile(evaluation_path, "w") as archive:
        prefix = redkitchen.EVALUATION_NAME
        archive.writestr(f"{prefix}/", b"")
        archive.writestr(f"{prefix}/3dmatch.log", prediction_log)
        archive.writestr(f"{prefix}/gt.log", ground_truth_log)
        archive.writestr(f"{prefix}/gt.info", ground_truth_info)
    return fragments_path, evaluation_path


def _patch_archive_constants(
    monkeypatch: pytest.MonkeyPatch,
    fragments: Path,
    evaluation: Path,
) -> None:
    monkeypatch.setattr(redkitchen, "FRAGMENT_COUNT", 4)
    monkeypatch.setattr(registration_guard, "FRAGMENT_COUNT", 4)
    for prefix, path in (("FRAGMENT", fragments), ("EVALUATION", evaluation)):
        payload = path.read_bytes()
        monkeypatch.setattr(
            redkitchen,
            f"{prefix}_ARCHIVE_MD5",
            hashlib.md5(payload).hexdigest(),
        )
        monkeypatch.setattr(
            redkitchen,
            f"{prefix}_ARCHIVE_SHA256",
            hashlib.sha256(payload).hexdigest(),
        )


def _phase28_artifact(path: Path) -> None:
    dimension = len(SIGNATURE_FEATURE_NAMES)
    model = {
        "feature_names": list(SIGNATURE_FEATURE_NAMES),
        "feature_center": [0.0] * dimension,
        "feature_scale": [1.0] * dimension,
        "intercept": 0.0,
        "coefficients": [0.0] * dimension,
        "rejection_cutoff": 1.0,
        "ridge_penalty": 1.0,
        "calibration_valid": True,
        "training_case_count": 2,
        "training_harmful_case_count": 1,
        "training_safe_case_count": 1,
        "rejected_training_harmful_case_count": 1,
        "retained_training_harmful_case_count": 0,
        "rejected_training_safe_case_count": 0,
        "retained_training_safe_case_count": 1,
    }
    path.write_text(
        json.dumps(
            {
                "artifact_schema": (
                    "pftf_alpha_local_spatial_residual_guard_phase28/v1"
                ),
                "predecessor_model": model,
            }
        ),
        encoding="utf-8",
    )


def test_binary_ply_reader_preserves_xyz(tmp_path: Path) -> None:
    points = np.asarray(((0.0, 1.0, 2.0), (3.0, 4.0, 5.0)))
    path = tmp_path / "fragment.ply"
    path.write_bytes(_ply_bytes(points))

    loaded = read_binary_ply_xyz(path)

    assert np.array_equal(loaded, points)


def test_farthest_anchors_do_not_repeat_indices_for_duplicate_points() -> None:
    points = np.zeros((4, 3), dtype=np.float64)

    anchors = registration_guard._farthest_anchor_indices(points, 4)

    assert anchors.tolist() == [0, 1, 2, 3]


def test_official_transformation_error_matches_translation_quadratic() -> None:
    ground_truth = RegistrationLogEntry(0, 2, 4, _matrix(1.0))
    information = RegistrationInfoEntry(0, 2, 4, np.eye(6) * 5000.0)
    exact = RegistrationLogEntry(0, 2, 4, _matrix(1.0))
    shifted = RegistrationLogEntry(0, 2, 4, _matrix(1.3))

    assert official_transformation_error(
        ground_truth,
        exact,
        information,
    ) == pytest.approx(0.0)
    assert official_transformation_error(
        ground_truth,
        shifted,
        information,
    ) == pytest.approx(0.09)


def test_phase32_joins_labels_after_blind_guard_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(19)
    points = rng.uniform(0.0, 0.2, size=(120, 3))
    fragments, evaluation = _write_archives(tmp_path, points)
    _patch_archive_constants(monkeypatch, fragments, evaluation)
    artifact = tmp_path / "phase28.json"
    _phase28_artifact(artifact)

    result = registration_guard.evaluate_threedmatch_registration_guard(
        tmp_path,
        artifact,
        distance_thresholds=(0.01,),
        maximum_points_per_fragment=100,
        patch_size=9,
        patch_count=1,
    )

    assert result.raw_prediction_count == 2
    assert result.eligible_prediction_count == 2
    assert result.ground_truth_overlap_pair_count == 1
    assert sum(item.official_correct for item in result.observations) == 1
    assert result.real_registration_labels_supported is True
    assert result.phase32_supported is False
    assert result.tail_sensitive_real_registration_supported is False
    assert result.real_correspondence_supported is False
    assert result.real_trimmed_reconstruction_supported is False
    assert result.deployment_supported is False
