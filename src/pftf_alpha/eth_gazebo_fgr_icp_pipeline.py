"""Generate Phase-39 label-blind ETH Gazebo Summer predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .eth_gazebo_validation_protocol import (
    ARCHIVE_NAME,
    EXPECTED_PAIR_COUNT,
    SCAN_COUNT,
    SCAN_MEMBERS,
    SELECTED_CANDIDATE_ID,
    SELECTED_PARAMETERS,
    verify_gazebo_archive_directory,
)
from .eth_open3d_fgr_pipeline import _load_open3d, _load_xyz
from .eth_pipeline_calibration import _pair_seed, _preprocess
from .open3d_fgr_pipeline import (
    FGR_DECREASE_MU,
    FGR_DIVISION_FACTOR,
    FGR_ITERATION_COUNT,
    FGR_MAXIMUM_TUPLE_COUNT,
    FGR_TUPLE_SCALE,
    FGR_TUPLE_TEST,
    FGR_USE_ABSOLUTE_SCALE,
    OFFICIAL_OPEN3D_TUTORIAL_URL,
    nonconsecutive_fragment_pairs,
)
from .scene_relative_rotation_guard import prediction_rotation_radians

EXPECTED_PROTOCOL_SHA256 = (
    "1711e23cdb29f0f305950c3eb3015309d8dea4f686c4b79a4d1a80c0af335059"
)
PREREGISTRATION_COMMIT = "d4174a11e55e0e12ad3e3ca1ddbf46e53d47d6dc"


@dataclass(frozen=True)
class GazeboPreprocessedScan:
    index: int
    archive_member: str
    raw_point_count: int
    downsampled_point_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GazeboPrediction:
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
class GazeboPredictionArtifact:
    artifact_schema: str
    role: str
    preregistration_commit: str
    protocol_artifact_path: str
    protocol_artifact_sha256: str
    archive_path: str
    archive_sha256: str
    calibration_artifact_sha256: str
    selected_candidate_id: str
    selected_parameters: dict[str, object]
    pipeline_name: str
    open3d_version: str
    python_version: str
    platform: str
    open3d_build_config: dict[str, object]
    official_parameter_source: str
    pair_selection_rule: str
    matrix_convention: str
    opened_archive_members: tuple[str, ...]
    preprocessed_scans: tuple[GazeboPreprocessedScan, ...]
    expected_pair_count: int
    predictions: tuple[GazeboPrediction, ...]
    label_boundary: str
    complete_prediction_set_materialized: bool
    validation_label_member_opened: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "opened_archive_members": list(self.opened_archive_members),
            "preprocessed_scans": [scan.to_dict() for scan in self.preprocessed_scans],
            "predictions": [row.to_dict() for row in self.predictions],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_protocol(path: Path) -> Mapping[str, object]:
    if _sha256(path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Gazebo validation protocol SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gazebo validation protocol must be a JSON object")
    expected = {
        "artifact_schema": (
            "pftf_alpha_eth_gazebo_validation_protocol_phase39/v1"
        ),
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_parameters": SELECTED_PARAMETERS,
        "validation_label_values_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Gazebo validation protocol mismatch: {key}")
    return payload


def generate_gazebo_predictions(
    archive_path: str | Path,
    protocol_path: str | Path,
) -> GazeboPredictionArtifact:
    archive = Path(archive_path)
    protocol = Path(protocol_path)
    protocol_payload = _verify_protocol(protocol)
    verification = verify_gazebo_archive_directory(archive)
    o3d = _load_open3d()
    voxel = float(SELECTED_PARAMETERS["voxel_size_meters"])
    opened_members: list[str] = []
    prepared: list[tuple[object, object]] = []
    summaries: list[GazeboPreprocessedScan] = []
    with zipfile.ZipFile(archive) as source:
        for index, member in enumerate(SCAN_MEMBERS):
            opened_members.append(member)
            with source.open(member) as stream:
                points = _load_xyz(stream)
            down, feature = _preprocess(o3d, points, voxel)
            prepared.append((down, feature))
            summaries.append(
                GazeboPreprocessedScan(
                    index=index,
                    archive_member=member,
                    raw_point_count=int(points.shape[0]),
                    downsampled_point_count=len(down.points),
                )
            )
            print(
                f"[ETH Gazebo Summer] preprocessed {index + 1}/{SCAN_COUNT}",
                flush=True,
            )
    if tuple(opened_members) != SCAN_MEMBERS:
        raise RuntimeError("Gazebo generator opened an unexpected member set")
    pairs = nonconsecutive_fragment_pairs(SCAN_COUNT)
    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("Gazebo validation pair count mismatch")
    predictions: list[GazeboPrediction] = []
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
                maximum_correspondence_distance=float(
                    SELECTED_PARAMETERS[
                        "fgr_maximum_correspondence_distance_meters"
                    ]
                ),
                iteration_number=FGR_ITERATION_COUNT,
                tuple_scale=FGR_TUPLE_SCALE,
                maximum_tuple_count=FGR_MAXIMUM_TUPLE_COUNT,
                tuple_test=FGR_TUPLE_TEST,
            ),
        )
        icp = o3d.pipelines.registration.registration_icp(
            prepared[target][0],
            prepared[source][0],
            float(
                SELECTED_PARAMETERS[
                    "icp_maximum_correspondence_distance_meters"
                ]
            ),
            np.asarray(fgr.transformation, dtype=np.float64),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=float(SELECTED_PARAMETERS["icp_relative_fitness"]),
                relative_rmse=float(SELECTED_PARAMETERS["icp_relative_rmse"]),
                max_iteration=int(SELECTED_PARAMETERS["icp_max_iterations"]),
            ),
        )
        matrix = np.asarray(icp.transformation, dtype=np.float64)
        predictions.append(
            GazeboPrediction(
                source_index=source,
                target_index=target,
                scan_count=SCAN_COUNT,
                pair_random_seed=seed,
                target_to_source_matrix=tuple(
                    tuple(float(value) for value in row) for row in matrix
                ),
                prediction_rotation_radians=prediction_rotation_radians(matrix),
                fitness=float(icp.fitness),
                inlier_rmse=float(icp.inlier_rmse),
                correspondence_count=len(icp.correspondence_set),
            )
        )
        if pair_index % 50 == 0 or pair_index == len(pairs):
            print(
                f"[ETH Gazebo Summer] registered {pair_index}/{len(pairs)}",
                flush=True,
            )
    build_config = {
        str(key): value
        for key, value in dict(getattr(o3d, "_build_config", {})).items()
        if isinstance(value, (bool, float, int, str))
    }
    return GazeboPredictionArtifact(
        artifact_schema="pftf_alpha_eth_gazebo_predictions_phase39/v1",
        role="fresh_label_blind_calibrated_pipeline_predictions",
        preregistration_commit=PREREGISTRATION_COMMIT,
        protocol_artifact_path=str(protocol),
        protocol_artifact_sha256=EXPECTED_PROTOCOL_SHA256,
        archive_path=str(archive),
        archive_sha256=verification.sha256,
        calibration_artifact_sha256=str(
            protocol_payload["calibration_artifact_sha256"]
        ),
        selected_candidate_id=SELECTED_CANDIDATE_ID,
        selected_parameters=SELECTED_PARAMETERS,
        pipeline_name="open3d_0.19.0_fpfh_fgr_point_to_plane_icp",
        open3d_version=o3d.__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        open3d_build_config=build_config,
        official_parameter_source=OFFICIAL_OPEN3D_TUTORIAL_URL,
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
            "generator verifies the hash-locked protocol and opens exactly "
            "the 32 frozen Hokuyo members; no code path opens a Gazebo pose "
            "member"
        ),
        complete_prediction_set_materialized=(
            len(predictions) == EXPECTED_PAIR_COUNT
        ),
        validation_label_member_opened=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("benchmark-data/eth_gazebo_summer") / ARCHIVE_NAME,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_validation_protocol_phase39.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_predictions_phase39.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = generate_gazebo_predictions(args.archive, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
