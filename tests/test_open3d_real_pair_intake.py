import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import pftf_alpha.open3d_demo_icp as demo_icp
from pftf_alpha.matched_guard_signature import SIGNATURE_FEATURE_NAMES
from pftf_alpha.open3d_real_pair_intake import (
    _reciprocal_matches,
    evaluate_open3d_real_pair_intake,
)


def _pcd_bytes(points: np.ndarray) -> bytes:
    rows = np.zeros((points.shape[0], 8), dtype="<f4")
    rows[:, :3] = points
    header = (
        "# .PCD v0.7\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb normal_x normal_y normal_z curvature\n"
        "SIZE 4 4 4 4 4 4 4 4\n"
        "TYPE F F F F F F F F\n"
        "COUNT 1 1 1 1 1 1 1 1\n"
        f"WIDTH {points.shape[0]}\n"
        "HEIGHT 1\n"
        f"POINTS {points.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")
    return header + rows.tobytes()


def _log_text(translation: float) -> str:
    logged_translation = -translation
    rows = []
    for source, target in ((0, 1), (1, 2)):
        rows.extend(
            (
                f"{source} {target} 57",
                f"1 0 0 {logged_translation}",
                "0 1 0 0",
                "0 0 1 0",
                "0 0 0 1",
            )
        )
    return "\n".join(rows) + "\n"


def _write_demo_archive(path: Path, clouds: tuple[np.ndarray, ...]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for index, points in enumerate(clouds):
            archive.writestr(f"cloud_bin_{index}.pcd", _pcd_bytes(points))
        archive.writestr("init.log", _log_text(1.0))


def _patch_archive_hashes(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    payload = path.read_bytes()
    monkeypatch.setattr(
        demo_icp,
        "EXPECTED_ARCHIVE_MD5",
        hashlib.md5(payload).hexdigest(),
    )
    monkeypatch.setattr(
        demo_icp,
        "EXPECTED_ARCHIVE_SHA256",
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


def test_binary_pcd_and_transform_log_parsers(tmp_path: Path) -> None:
    points = np.asarray(((0.0, 1.0, 2.0), (3.0, 4.0, 5.0)))
    pcd_path = tmp_path / "cloud.pcd"
    pcd_path.write_bytes(_pcd_bytes(points))
    log_path = tmp_path / "init.log"
    log_path.write_text(_log_text(1.0), encoding="ascii")

    loaded = demo_icp.read_binary_pcd_xyz(pcd_path)
    entries = demo_icp.read_transformation_log(log_path)

    assert np.array_equal(loaded, points)
    assert [(entry.source_index, entry.target_index) for entry in entries] == [
        (0, 1),
        (1, 2),
    ]
    transformed = demo_icp.transform_points(
        points,
        entries[0].source_to_target_matrix,
    )
    assert np.allclose(transformed[:, 0], points[:, 0] + 1.0)


def test_reciprocal_matching_removes_nonreciprocal_candidates() -> None:
    source = np.asarray(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (1.0, 0.0, 0.0)))
    target = np.asarray(((0.01, 0.0, 0.0), (1.01, 0.0, 0.0)))

    source_indices, target_indices, distances = _reciprocal_matches(
        source,
        target,
        0.02,
    )

    assert source_indices.tolist() == [0, 2]
    assert target_indices.tolist() == [0, 1]
    assert distances == pytest.approx((0.01, 0.01))


def test_phase31_real_pair_intake_preserves_claim_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(17)
    cloud0 = rng.uniform(0.0, 0.2, size=(120, 3))
    cloud1 = cloud0 + (1.0, 0.0, 0.0)
    cloud2 = cloud1 + (1.0, 0.0, 0.0)
    archive_path = tmp_path / demo_icp.ARCHIVE_NAME
    _write_demo_archive(archive_path, (cloud0, cloud1, cloud2))
    _patch_archive_hashes(monkeypatch, archive_path)
    artifact_path = tmp_path / "phase28.json"
    _phase28_artifact(artifact_path)

    result = evaluate_open3d_real_pair_intake(
        tmp_path,
        artifact_path,
        distance_thresholds=(0.01,),
        patch_size=9,
        patch_count=2,
    )

    assert result.real_paired_scan_intake_supported is True
    assert result.patch_observation_count == 4
    assert result.observational_guard_stack_pass_count == 4
    assert all(
        pair.direction_diagnostic.inverse_direction_better
        for pair in result.scan_pairs
    )
    assert result.real_correspondence_supported is False
    assert result.real_paired_scan_guard_supported is False
    assert result.real_trimmed_reconstruction_supported is False
    assert result.deployment_supported is False
