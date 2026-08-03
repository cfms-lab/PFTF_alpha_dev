"""Phase-39 ETH pipeline calibration on the opened Mountain Plain scene."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .eth_open3d_fgr_pipeline import _load_open3d, _load_xyz
from .fresh_external_protocol import (
    ARCHIVE_NAME,
    EXPECTED_PAIR_COUNT,
    LABEL_MEMBER,
    MAX_RELATIVE_ROTATION_ERROR_DEGREES,
    MAX_RELATIVE_TRANSLATION_ERROR_METERS,
    SCAN_COUNT,
    SCAN_MEMBERS,
    verify_archive_directory,
)
from .fresh_external_rotation_audit import (
    _parse_pose_labels,
    _relative_target_to_source,
    _rigid_errors,
)
from .open3d_fgr_pipeline import (
    BASE_RANDOM_SEED,
    FGR_DECREASE_MU,
    FGR_DISTANCE_MULTIPLIER,
    FGR_DIVISION_FACTOR,
    FGR_ITERATION_COUNT,
    FGR_MAXIMUM_TUPLE_COUNT,
    FGR_TUPLE_SCALE,
    FGR_TUPLE_TEST,
    FGR_USE_ABSOLUTE_SCALE,
    FPFH_MAX_NEIGHBORS,
    FPFH_RADIUS_MULTIPLIER,
    NORMAL_MAX_NEIGHBORS,
    NORMAL_RADIUS_MULTIPLIER,
    nonconsecutive_fragment_pairs,
)

CALIBRATION_VOXEL_SIZES_METERS = (0.10, 0.20, 0.30, 0.50)
CALIBRATION_REFINEMENT_MODES = (False, True)
ICP_DISTANCE_MULTIPLIER = 1.5
ICP_RELATIVE_FITNESS = 1.0e-6
ICP_RELATIVE_RMSE = 1.0e-6
ICP_MAX_ITERATIONS = 50


@dataclass(frozen=True)
class ETHCalibrationPrediction:
    source_index: int
    target_index: int
    pair_random_seed: int
    target_to_source_matrix: tuple[tuple[float, ...], ...]
    relative_rotation_error_degrees: float
    relative_translation_error_meters: float
    frozen_correct: bool
    fitness: float
    inlier_rmse: float
    correspondence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "target_to_source_matrix": [
                list(row) for row in self.target_to_source_matrix
            ],
        }


@dataclass(frozen=True)
class ETHCalibrationCandidate:
    candidate_id: str
    voxel_size_meters: float
    use_point_to_plane_icp: bool
    parameters: dict[str, object]
    prediction_count: int
    correct_count: int
    incorrect_count: int
    rotation_threshold_pass_count: int
    translation_threshold_pass_count: int
    median_rotation_error_degrees: float
    median_translation_error_meters: float
    elapsed_seconds: float
    predictions: tuple[ETHCalibrationPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "predictions": [row.to_dict() for row in self.predictions],
        }


@dataclass(frozen=True)
class ETHPipelineCalibrationArtifact:
    artifact_schema: str
    role: str
    scene_name: str
    archive_path: str
    archive_sha256: str
    candidate_grid: tuple[dict[str, object], ...]
    selection_metric: str
    tie_breaker: str
    p90_guard_used_for_selection: bool
    correctness_rule: str
    expected_pair_count: int
    candidates: tuple[ETHCalibrationCandidate, ...]
    selected_candidate_id: str | None
    selected_parameters: dict[str, object] | None
    calibration_viable: bool
    calibration_label_values_accessed: bool
    fresh_validation_label_values_accessed: bool
    open3d_version: str
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "candidate_grid": list(self.candidate_grid),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def candidate_grid() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "candidate_id": (
                f"fgr{'_icp' if use_icp else ''}_v{round(voxel * 100):03d}"
            ),
            "voxel_size_meters": voxel,
            "use_point_to_plane_icp": use_icp,
        }
        for voxel in CALIBRATION_VOXEL_SIZES_METERS
        for use_icp in CALIBRATION_REFINEMENT_MODES
    )


def candidate_parameters(
    voxel_size_meters: float,
    *,
    use_icp: bool,
) -> dict[str, object]:
    return {
        "voxel_size_meters": voxel_size_meters,
        "normal_radius_meters": voxel_size_meters * NORMAL_RADIUS_MULTIPLIER,
        "normal_max_neighbors": NORMAL_MAX_NEIGHBORS,
        "fpfh_radius_meters": voxel_size_meters * FPFH_RADIUS_MULTIPLIER,
        "fpfh_max_neighbors": FPFH_MAX_NEIGHBORS,
        "fgr_maximum_correspondence_distance_meters": (
            voxel_size_meters * FGR_DISTANCE_MULTIPLIER
        ),
        "fgr_division_factor": FGR_DIVISION_FACTOR,
        "fgr_use_absolute_scale": FGR_USE_ABSOLUTE_SCALE,
        "fgr_decrease_mu": FGR_DECREASE_MU,
        "fgr_iteration_count": FGR_ITERATION_COUNT,
        "fgr_tuple_scale": FGR_TUPLE_SCALE,
        "fgr_maximum_tuple_count": FGR_MAXIMUM_TUPLE_COUNT,
        "fgr_tuple_test": FGR_TUPLE_TEST,
        "base_random_seed": BASE_RANDOM_SEED,
        "use_point_to_plane_icp": use_icp,
        "icp_maximum_correspondence_distance_meters": (
            voxel_size_meters * ICP_DISTANCE_MULTIPLIER if use_icp else None
        ),
        "icp_relative_fitness": ICP_RELATIVE_FITNESS if use_icp else None,
        "icp_relative_rmse": ICP_RELATIVE_RMSE if use_icp else None,
        "icp_max_iterations": ICP_MAX_ITERATIONS if use_icp else None,
    }


def _preprocess(o3d: Any, points: np.ndarray, voxel: float) -> tuple[Any, Any]:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    down = cloud.voxel_down_sample(voxel)
    if len(down.points) < 3:
        raise ValueError("calibration downsampling left too few points")
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * NORMAL_RADIUS_MULTIPLIER,
            max_nn=NORMAL_MAX_NEIGHBORS,
        )
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * FPFH_RADIUS_MULTIPLIER,
            max_nn=FPFH_MAX_NEIGHBORS,
        ),
    )
    return down, feature


def _pair_seed(source: int, target: int) -> int:
    return BASE_RANDOM_SEED + 1_009 * source + target


def _prediction(
    source: int,
    target: int,
    seed: int,
    matrix: np.ndarray,
    result: Any,
    ground_truth: np.ndarray,
) -> ETHCalibrationPrediction:
    rotation_error, translation_error = _rigid_errors(matrix, ground_truth)
    correct = (
        rotation_error < MAX_RELATIVE_ROTATION_ERROR_DEGREES
        and translation_error < MAX_RELATIVE_TRANSLATION_ERROR_METERS
    )
    return ETHCalibrationPrediction(
        source_index=source,
        target_index=target,
        pair_random_seed=seed,
        target_to_source_matrix=tuple(
            tuple(float(value) for value in row) for row in matrix
        ),
        relative_rotation_error_degrees=rotation_error,
        relative_translation_error_meters=translation_error,
        frozen_correct=correct,
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        correspondence_count=len(result.correspondence_set),
    )


def _candidate_summary(
    candidate_id: str,
    voxel: float,
    use_icp: bool,
    predictions: list[ETHCalibrationPrediction],
    elapsed_seconds: float,
) -> ETHCalibrationCandidate:
    correct = sum(row.frozen_correct for row in predictions)
    rotation_errors = np.asarray(
        [row.relative_rotation_error_degrees for row in predictions]
    )
    translation_errors = np.asarray(
        [row.relative_translation_error_meters for row in predictions]
    )
    return ETHCalibrationCandidate(
        candidate_id=candidate_id,
        voxel_size_meters=voxel,
        use_point_to_plane_icp=use_icp,
        parameters=candidate_parameters(voxel, use_icp=use_icp),
        prediction_count=len(predictions),
        correct_count=correct,
        incorrect_count=len(predictions) - correct,
        rotation_threshold_pass_count=int(
            np.sum(rotation_errors < MAX_RELATIVE_ROTATION_ERROR_DEGREES)
        ),
        translation_threshold_pass_count=int(
            np.sum(translation_errors < MAX_RELATIVE_TRANSLATION_ERROR_METERS)
        ),
        median_rotation_error_degrees=float(np.median(rotation_errors)),
        median_translation_error_meters=float(np.median(translation_errors)),
        elapsed_seconds=elapsed_seconds,
        predictions=tuple(predictions),
    )


def select_candidate(
    candidates: Sequence[ETHCalibrationCandidate],
) -> ETHCalibrationCandidate | None:
    if not candidates:
        raise ValueError("calibration requires candidates")
    selected = min(
        candidates,
        key=lambda item: (
            -item.correct_count,
            item.voxel_size_meters,
            item.use_point_to_plane_icp,
        ),
    )
    return selected if selected.correct_count > 0 else None


def calibrate_eth_pipeline(
    archive_path: str | Path,
) -> ETHPipelineCalibrationArtifact:
    archive = Path(archive_path)
    verification = verify_archive_directory(archive)
    o3d = _load_open3d()
    points: list[np.ndarray] = []
    with zipfile.ZipFile(archive) as source:
        for member in SCAN_MEMBERS:
            with source.open(member) as stream:
                points.append(_load_xyz(stream))
        with source.open(LABEL_MEMBER) as stream:
            poses = _parse_pose_labels(stream)
    pairs = nonconsecutive_fragment_pairs(SCAN_COUNT)
    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("Phase-39 pair count mismatch")
    candidate_results: list[ETHCalibrationCandidate] = []
    for voxel in CALIBRATION_VOXEL_SIZES_METERS:
        prepared = [_preprocess(o3d, cloud, voxel) for cloud in points]
        fgr_rows: list[ETHCalibrationPrediction] = []
        icp_rows: list[ETHCalibrationPrediction] = []
        started = time.perf_counter()
        for pair_index, (source, target) in enumerate(pairs, start=1):
            seed = _pair_seed(source, target)
            o3d.utility.random.seed(seed)
            fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                prepared[target][0],
                prepared[source][0],
                prepared[target][1],
                prepared[source][1],
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    division_factor=FGR_DIVISION_FACTOR,
                    use_absolute_scale=FGR_USE_ABSOLUTE_SCALE,
                    decrease_mu=FGR_DECREASE_MU,
                    maximum_correspondence_distance=(
                        voxel * FGR_DISTANCE_MULTIPLIER
                    ),
                    iteration_number=FGR_ITERATION_COUNT,
                    tuple_scale=FGR_TUPLE_SCALE,
                    maximum_tuple_count=FGR_MAXIMUM_TUPLE_COUNT,
                    tuple_test=FGR_TUPLE_TEST,
                ),
            )
            ground_truth = _relative_target_to_source(
                poses[source],
                poses[target],
            )
            fgr_matrix = np.asarray(fgr.transformation, dtype=np.float64)
            fgr_rows.append(
                _prediction(
                    source,
                    target,
                    seed,
                    fgr_matrix,
                    fgr,
                    ground_truth,
                )
            )
            icp = o3d.pipelines.registration.registration_icp(
                prepared[target][0],
                prepared[source][0],
                voxel * ICP_DISTANCE_MULTIPLIER,
                fgr_matrix,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=ICP_RELATIVE_FITNESS,
                    relative_rmse=ICP_RELATIVE_RMSE,
                    max_iteration=ICP_MAX_ITERATIONS,
                ),
            )
            icp_rows.append(
                _prediction(
                    source,
                    target,
                    seed,
                    np.asarray(icp.transformation, dtype=np.float64),
                    icp,
                    ground_truth,
                )
            )
            if pair_index % 50 == 0 or pair_index == len(pairs):
                print(
                    f"[ETH calibration voxel={voxel:.2f}] "
                    f"registered {pair_index}/{len(pairs)}",
                    flush=True,
                )
        elapsed = time.perf_counter() - started
        candidate_results.append(
            _candidate_summary(
                f"fgr_v{round(voxel * 100):03d}",
                voxel,
                False,
                fgr_rows,
                elapsed,
            )
        )
        candidate_results.append(
            _candidate_summary(
                f"fgr_icp_v{round(voxel * 100):03d}",
                voxel,
                True,
                icp_rows,
                elapsed,
            )
        )
    selected = select_candidate(candidate_results)
    return ETHPipelineCalibrationArtifact(
        artifact_schema="pftf_alpha_eth_pipeline_calibration_phase39/v1",
        role="opened_scene_predictor_calibration_not_fresh_validation",
        scene_name="ETH Mountain Plain",
        archive_path=str(archive),
        archive_sha256=verification.sha256,
        candidate_grid=candidate_grid(),
        selection_metric=(
            "maximum count satisfying strict RRE<15deg and RTE<0.30m"
        ),
        tie_breaker="smaller voxel, then FGR without ICP",
        p90_guard_used_for_selection=False,
        correctness_rule="strict RRE < 15 degrees and strict RTE < 0.30 meters",
        expected_pair_count=EXPECTED_PAIR_COUNT,
        candidates=tuple(candidate_results),
        selected_candidate_id=(selected.candidate_id if selected else None),
        selected_parameters=(selected.parameters if selected else None),
        calibration_viable=selected is not None,
        calibration_label_values_accessed=True,
        fresh_validation_label_values_accessed=False,
        open3d_version=o3d.__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("benchmark-data/eth_mountain_plain") / ARCHIVE_NAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_pipeline_calibration_phase39.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = calibrate_eth_pipeline(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
