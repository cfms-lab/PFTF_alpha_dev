"""Phase-38 label-blind Open3D predictions for ETH Mountain Plain."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .fresh_external_protocol import (
    ARCHIVE_NAME,
    EXPECTED_PAIR_COUNT,
    SCAN_COUNT,
    SCAN_MEMBERS,
    preregister_phase38,
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
    OFFICIAL_OPEN3D_TUTORIAL_URL,
    OPEN3D_VERSION,
    VOXEL_SIZE_METERS,
    nonconsecutive_fragment_pairs,
    phase37_parameters,
)
from .scene_relative_rotation_guard import prediction_rotation_radians

EXPECTED_PROTOCOL_SHA256 = (
    "1d183f5b6c8dd7eaeb35a6950ac3fdb16e3306f21e187489e5d0272279973649"
)
PREREGISTRATION_COMMIT = "352829779fe60b7b35bb14d5b1e7368452338126"


@dataclass(frozen=True)
class ETHPreprocessedScan:
    index: int
    archive_member: str
    raw_point_count: int
    downsampled_point_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ETHFGRPrediction:
    source_index: int
    target_index: int
    scan_count: int
    pair_random_seed: int
    target_to_source_matrix: tuple[tuple[float, ...], ...]
    prediction_rotation_radians: float
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
class ETHFGRPredictionArtifact:
    artifact_schema: str
    role: str
    preregistration_commit: str
    protocol_artifact_path: str
    protocol_artifact_sha256: str
    archive_path: str
    archive_sha256: str
    pipeline_name: str
    open3d_version: str
    python_version: str
    platform: str
    open3d_build_config: dict[str, object]
    official_parameter_source: str
    parameters: dict[str, object]
    pair_selection_rule: str
    matrix_convention: str
    opened_archive_members: tuple[str, ...]
    preprocessed_scans: tuple[ETHPreprocessedScan, ...]
    expected_pair_count: int
    predictions: tuple[ETHFGRPrediction, ...]
    label_boundary: str
    complete_prediction_set_materialized: bool
    ground_truth_label_member_opened: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "opened_archive_members": list(self.opened_archive_members),
            "preprocessed_scans": [
                scan.to_dict() for scan in self.preprocessed_scans
            ],
            "predictions": [row.to_dict() for row in self.predictions],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase-38 protocol artifact must be a JSON object")
    return payload


def _verify_protocol(path: Path) -> Mapping[str, object]:
    if _sha256(path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Phase-38 preregistration artifact SHA-256 mismatch")
    payload = _load_json_object(path)
    expected = {
        "artifact_schema": "pftf_alpha_fresh_external_protocol_phase38/v1",
        "role": "pre_label_fixed_protocol",
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "pipeline_parameters": phase37_parameters(),
        "label_values_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase-38 preregistration mismatch: {key}")
    return payload


def _load_xyz(stream: BinaryIO) -> np.ndarray:
    """Load only x/y/z from one frozen Hokuyo CSV member."""

    header = stream.readline().decode("ascii").strip()
    if header != "Time_in_sec,x,y,z,Intensities,2DscanId,PointId":
        raise ValueError(f"unexpected ETH Hokuyo header: {header}")
    with io.TextIOWrapper(stream, encoding="ascii", newline="") as text:
        points = np.loadtxt(
            text,
            delimiter=",",
            dtype=np.float64,
            usecols=(1, 2, 3),
            ndmin=2,
        )
    if points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("ETH Hokuyo member contains too few XYZ points")
    if not np.isfinite(points).all():
        raise ValueError("ETH Hokuyo member contains non-finite XYZ points")
    return points


def _load_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Phase 38 requires the isolated open3d==0.19.0 environment"
        ) from error
    if o3d.__version__ != OPEN3D_VERSION:
        raise RuntimeError(
            f"Phase 38 requires Open3D {OPEN3D_VERSION}, got {o3d.__version__}"
        )
    return o3d


def _preprocess(o3d: Any, points: np.ndarray) -> tuple[Any, Any]:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    down = cloud.voxel_down_sample(VOXEL_SIZE_METERS)
    if len(down.points) < 3:
        raise ValueError("Open3D downsampling left too few ETH points")
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE_METERS * NORMAL_RADIUS_MULTIPLIER,
            max_nn=NORMAL_MAX_NEIGHBORS,
        )
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE_METERS * FPFH_RADIUS_MULTIPLIER,
            max_nn=FPFH_MAX_NEIGHBORS,
        ),
    )
    return down, feature


def _pair_seed(source: int, target: int) -> int:
    return BASE_RANDOM_SEED + 1_009 * source + target


def generate_eth_fgr_predictions(
    archive_path: str | Path,
    protocol_path: str | Path,
) -> ETHFGRPredictionArtifact:
    """Generate the complete pair universe without opening the label member."""

    archive = Path(archive_path)
    protocol = Path(protocol_path)
    _verify_protocol(protocol)
    preregistration = preregister_phase38(archive)
    o3d = _load_open3d()
    opened_members: list[str] = []
    prepared: list[tuple[Any, Any]] = []
    summaries: list[ETHPreprocessedScan] = []
    with zipfile.ZipFile(archive) as source:
        for index, member in enumerate(SCAN_MEMBERS):
            opened_members.append(member)
            with source.open(member) as stream:
                points = _load_xyz(stream)
            down, feature = _preprocess(o3d, points)
            prepared.append((down, feature))
            summaries.append(
                ETHPreprocessedScan(
                    index=index,
                    archive_member=member,
                    raw_point_count=int(points.shape[0]),
                    downsampled_point_count=len(down.points),
                )
            )
            print(f"[ETH Mountain Plain] preprocessed {index + 1}/{SCAN_COUNT}")
    if tuple(opened_members) != SCAN_MEMBERS:
        raise RuntimeError("Phase-38 generator opened an unexpected member set")
    pairs = nonconsecutive_fragment_pairs(SCAN_COUNT)
    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("Phase-38 pair count differs from preregistration")
    predictions: list[ETHFGRPrediction] = []
    for pair_index, (source, target) in enumerate(pairs, start=1):
        seed = _pair_seed(source, target)
        o3d.utility.random.seed(seed)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            prepared[target][0],
            prepared[source][0],
            prepared[target][1],
            prepared[source][1],
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                division_factor=FGR_DIVISION_FACTOR,
                use_absolute_scale=FGR_USE_ABSOLUTE_SCALE,
                decrease_mu=FGR_DECREASE_MU,
                maximum_correspondence_distance=(
                    VOXEL_SIZE_METERS * FGR_DISTANCE_MULTIPLIER
                ),
                iteration_number=FGR_ITERATION_COUNT,
                tuple_scale=FGR_TUPLE_SCALE,
                maximum_tuple_count=FGR_MAXIMUM_TUPLE_COUNT,
                tuple_test=FGR_TUPLE_TEST,
            ),
        )
        matrix = np.asarray(result.transformation, dtype=np.float64)
        rotation = prediction_rotation_radians(matrix)
        predictions.append(
            ETHFGRPrediction(
                source_index=source,
                target_index=target,
                scan_count=SCAN_COUNT,
                pair_random_seed=seed,
                target_to_source_matrix=tuple(
                    tuple(float(value) for value in row) for row in matrix
                ),
                prediction_rotation_radians=rotation,
                fitness=float(result.fitness),
                inlier_rmse=float(result.inlier_rmse),
                correspondence_count=len(result.correspondence_set),
            )
        )
        if pair_index % 50 == 0 or pair_index == len(pairs):
            print(
                f"[ETH Mountain Plain] registered {pair_index}/{len(pairs)}",
                flush=True,
            )
    build_config = {
        str(key): value
        for key, value in dict(getattr(o3d, "_build_config", {})).items()
        if isinstance(value, (bool, float, int, str))
    }
    return ETHFGRPredictionArtifact(
        artifact_schema="pftf_alpha_eth_open3d_fgr_predictions_phase38/v1",
        role="fresh_label_blind_fixed_parameter_predictions",
        preregistration_commit=PREREGISTRATION_COMMIT,
        protocol_artifact_path=str(protocol),
        protocol_artifact_sha256=EXPECTED_PROTOCOL_SHA256,
        archive_path=str(archive),
        archive_sha256=preregistration.archive_verification.sha256,
        pipeline_name="open3d_0.19.0_fpfh_fast_global_registration",
        open3d_version=o3d.__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        open3d_build_config=build_config,
        official_parameter_source=OFFICIAL_OPEN3D_TUTORIAL_URL,
        parameters=phase37_parameters(),
        pair_selection_rule=(
            "all source<target pairs with target-source>1; no label, pose, "
            "overlap, fitness, or result filtering"
        ),
        matrix_convention="target-index local coordinates to source-index",
        opened_archive_members=tuple(opened_members),
        preprocessed_scans=tuple(summaries),
        expected_pair_count=EXPECTED_PAIR_COUNT,
        predictions=tuple(predictions),
        label_boundary=(
            "generator verifies the preregistration and outer archive, then "
            "opens exactly the 31 frozen Hokuyo members; it contains no code "
            "path that opens the Leica pose member"
        ),
        complete_prediction_set_materialized=(
            len(predictions) == EXPECTED_PAIR_COUNT
        ),
        ground_truth_label_member_opened=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("benchmark-data/eth_mountain_plain") / ARCHIVE_NAME,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmark-out/fresh_external_protocol_phase38.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_open3d_fgr_predictions_phase38.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = generate_eth_fgr_predictions(args.archive, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
