import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pftf_alpha.threedmatch_registration_guard as registration_guard
import pftf_alpha.threedmatch_transfer_audit as transfer_audit
from pftf_alpha.local_spatial_residual_guard import (
    EXPECTED_LOCAL_REJECTION_CUTOFF,
)
from pftf_alpha.matched_guard_signature import SIGNATURE_FEATURE_NAMES
from pftf_alpha.tail_sensitive_local_guard import (
    EXPECTED_TAIL_REJECTION_CUTOFF,
)
from pftf_alpha.threedmatch_scene import (
    ThreeDMatchSceneSpec,
    verify_threedmatch_scene_archive,
)
from pftf_alpha.threedmatch_transfer_audit import (
    EXPECTED_GLOBAL_REJECTION_CUTOFF,
    evaluate_threedmatch_transfer_audit,
    load_frozen_transfer_protocol,
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
    scene_name = "synthetic-scene"
    evaluation_name = f"{scene_name}-evaluation"
    fragments_path = root / f"{scene_name}.zip"
    with zipfile.ZipFile(fragments_path, "w") as archive:
        archive.writestr(f"{scene_name}/", b"")
        clouds = (
            points,
            points,
            points + (1.0, 0.0, 0.0),
            points,
        )
        for index, cloud in enumerate(clouds):
            archive.writestr(
                f"{scene_name}/cloud_bin_{index}.ply",
                _ply_bytes(cloud),
            )
    evaluation_path = root / f"{evaluation_name}.zip"
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
        archive.writestr(f"{evaluation_name}/", b"")
        archive.writestr(f"{evaluation_name}/3dmatch.log", prediction_log)
        archive.writestr(f"{evaluation_name}/gt.log", ground_truth_log)
        archive.writestr(f"{evaluation_name}/gt.info", ground_truth_info)
    return fragments_path, evaluation_path


def _hash(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def _scene_spec(fragments: Path, evaluation: Path) -> ThreeDMatchSceneSpec:
    return ThreeDMatchSceneSpec(
        scene_name="synthetic-scene",
        evaluation_name="synthetic-scene-evaluation",
        fragment_archive_name=fragments.name,
        evaluation_archive_name=evaluation.name,
        fragment_url="https://example.invalid/fragments.zip",
        evaluation_url="https://example.invalid/evaluation.zip",
        fragment_archive_md5=_hash(fragments, "md5"),
        fragment_archive_sha256=_hash(fragments, "sha256"),
        evaluation_archive_md5=_hash(evaluation, "md5"),
        evaluation_archive_sha256=_hash(evaluation, "sha256"),
        fragment_count=4,
        dataset_source="synthetic test fixture",
        dataset_license_boundary="test fixture only",
    )


def _phase28_artifact(path: Path) -> str:
    dimension = len(SIGNATURE_FEATURE_NAMES)
    model = {
        "feature_names": list(SIGNATURE_FEATURE_NAMES),
        "feature_center": [0.0] * dimension,
        "feature_scale": [1.0] * dimension,
        "intercept": 0.0,
        "coefficients": [0.0] * dimension,
        "rejection_cutoff": EXPECTED_GLOBAL_REJECTION_CUTOFF,
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
    return _hash(path, "sha256")


def _reference_artifact(path: Path, phase28_sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_schema": (
                    "pftf_alpha_threedmatch_registration_guard_phase32/v1"
                ),
                "dataset_name": "7-scenes-redkitchen",
                "distance_thresholds": [0.02, 0.05],
                "maximum_points_per_fragment": 10_000,
                "patch_size": 96,
                "patch_count": 4,
                "official_error_threshold_squared": 0.04,
                "global_signature_rejection_cutoff": (
                    EXPECTED_GLOBAL_REJECTION_CUTOFF
                ),
                "local_percentile95_rejection_cutoff": (
                    EXPECTED_LOCAL_REJECTION_CUTOFF
                ),
                "isolated_tail_ratio_rejection_cutoff": (
                    EXPECTED_TAIL_REJECTION_CUTOFF
                ),
                "phase28_artifact_sha256": phase28_sha256,
                "phase32_supported": False,
                "tail_sensitive_real_registration_supported": False,
            }
        ),
        encoding="utf-8",
    )


def test_scene_archive_verification_checks_exact_identity(tmp_path: Path) -> None:
    points = np.zeros((120, 3), dtype=np.float64)
    fragments, evaluation = _write_archives(tmp_path, points)
    scene = _scene_spec(fragments, evaluation)

    verified = verify_threedmatch_scene_archive(
        fragments,
        scene,
        role="fragments",
    )

    assert verified.verified is True
    assert verified.file_count == 4
    changed = replace(scene, fragment_archive_md5="0" * 32)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify_threedmatch_scene_archive(
            fragments,
            changed,
            role="fragments",
        )


def test_reference_protocol_rejects_changed_threshold(tmp_path: Path) -> None:
    phase28 = tmp_path / "phase28.json"
    phase28_sha256 = _phase28_artifact(phase28)
    reference = tmp_path / "phase32.json"
    _reference_artifact(reference, phase28_sha256)
    payload = json.loads(reference.read_text(encoding="utf-8"))
    payload["distance_thresholds"] = [0.03, 0.05]
    reference.write_text(json.dumps(payload), encoding="utf-8")
    expected_hash = _hash(reference, "sha256")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            transfer_audit,
            "EXPECTED_REFERENCE_SHA256",
            expected_hash,
        )
        with pytest.raises(ValueError, match="distance thresholds"):
            load_frozen_transfer_protocol(reference)


def test_reference_protocol_rejects_changed_artifact_hash(tmp_path: Path) -> None:
    reference = tmp_path / "phase32.json"
    reference.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_transfer_protocol(reference)


def test_phase33_reuses_frozen_route_and_joins_labels_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(23)
    points = rng.uniform(0.0, 0.2, size=(120, 3))
    fragments, evaluation = _write_archives(tmp_path, points)
    scene = _scene_spec(fragments, evaluation)
    phase28 = tmp_path / "phase28.json"
    phase28_sha256 = _phase28_artifact(phase28)
    reference = tmp_path / "phase32.json"
    _reference_artifact(reference, phase28_sha256)
    monkeypatch.setattr(
        transfer_audit,
        "EXPECTED_REFERENCE_SHA256",
        _hash(reference, "sha256"),
    )
    blind_count = 0
    original_blind = registration_guard._blind_observation
    original_log_reader = registration_guard.read_registration_log

    def observed_blind(*args: object, **kwargs: object):
        nonlocal blind_count
        result = original_blind(*args, **kwargs)
        blind_count += 1
        return result

    def ordered_log_reader(path: str | Path):
        if Path(path).name == "gt.log":
            assert blind_count == 2
        return original_log_reader(path)

    monkeypatch.setattr(registration_guard, "_blind_observation", observed_blind)
    monkeypatch.setattr(
        registration_guard,
        "read_registration_log",
        ordered_log_reader,
    )

    result = evaluate_threedmatch_transfer_audit(
        tmp_path,
        phase28,
        reference,
        scene=scene,
    )

    assert result.raw_prediction_count == 2
    assert result.eligible_prediction_count == 2
    assert result.ground_truth_overlap_pair_count == 1
    assert sum(item.official_correct for item in result.observations) == 1
    assert result.phase33_audit_completed is True
    assert result.cross_scene_guard_supported is False
    assert result.real_registration_labels_supported is True
    assert result.real_correspondence_supported is False
    assert result.real_trimmed_reconstruction_supported is False
    assert result.deployment_supported is False
