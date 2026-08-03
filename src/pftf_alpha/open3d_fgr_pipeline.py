"""Phase-37 independent Open3D FPFH+FGR prediction generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .scene_relative_rotation_guard import prediction_rotation_radians
from .threedmatch_redkitchen import (
    EVALUATION_ARCHIVE_NAME as REDKITCHEN_EVALUATION_ARCHIVE_NAME,
)
from .threedmatch_redkitchen import (
    EVALUATION_NAME as REDKITCHEN_EVALUATION_NAME,
)
from .threedmatch_redkitchen import (
    FRAGMENT_ARCHIVE_MD5 as REDKITCHEN_FRAGMENT_ARCHIVE_MD5,
)
from .threedmatch_redkitchen import (
    FRAGMENT_ARCHIVE_NAME as REDKITCHEN_FRAGMENT_ARCHIVE_NAME,
)
from .threedmatch_redkitchen import (
    FRAGMENT_ARCHIVE_SHA256 as REDKITCHEN_FRAGMENT_ARCHIVE_SHA256,
)
from .threedmatch_redkitchen import FRAGMENT_COUNT as REDKITCHEN_FRAGMENT_COUNT
from .threedmatch_redkitchen import SCENE_NAME as REDKITCHEN_SCENE_NAME
from .threedmatch_redkitchen import verify_redkitchen_archive
from .threedmatch_scene import (
    MARYLAND_HOTEL3_SPEC,
    verify_threedmatch_scene_archive,
)

OPEN3D_VERSION = "0.19.0"
VOXEL_SIZE_METERS = 0.05
NORMAL_RADIUS_MULTIPLIER = 2.0
NORMAL_MAX_NEIGHBORS = 30
FPFH_RADIUS_MULTIPLIER = 5.0
FPFH_MAX_NEIGHBORS = 100
FGR_DISTANCE_MULTIPLIER = 0.5
FGR_DIVISION_FACTOR = 1.4
FGR_USE_ABSOLUTE_SCALE = False
FGR_DECREASE_MU = False
FGR_ITERATION_COUNT = 64
FGR_TUPLE_SCALE = 0.95
FGR_MAXIMUM_TUPLE_COUNT = 1000
FGR_TUPLE_TEST = True
BASE_RANDOM_SEED = 370803
OFFICIAL_OPEN3D_TUTORIAL_URL = (
    "https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html"
)


@dataclass(frozen=True)
class Phase37SceneInput:
    scene_name: str
    evaluation_name: str
    fragment_archive_name: str
    evaluation_archive_name: str
    fragment_archive_md5: str
    fragment_archive_sha256: str
    fragment_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REDKITCHEN_PHASE37_INPUT = Phase37SceneInput(
    scene_name=REDKITCHEN_SCENE_NAME,
    evaluation_name=REDKITCHEN_EVALUATION_NAME,
    fragment_archive_name=REDKITCHEN_FRAGMENT_ARCHIVE_NAME,
    evaluation_archive_name=REDKITCHEN_EVALUATION_ARCHIVE_NAME,
    fragment_archive_md5=REDKITCHEN_FRAGMENT_ARCHIVE_MD5,
    fragment_archive_sha256=REDKITCHEN_FRAGMENT_ARCHIVE_SHA256,
    fragment_count=REDKITCHEN_FRAGMENT_COUNT,
)
MARYLAND_PHASE37_INPUT = Phase37SceneInput(
    scene_name=MARYLAND_HOTEL3_SPEC.scene_name,
    evaluation_name=MARYLAND_HOTEL3_SPEC.evaluation_name,
    fragment_archive_name=MARYLAND_HOTEL3_SPEC.fragment_archive_name,
    evaluation_archive_name=MARYLAND_HOTEL3_SPEC.evaluation_archive_name,
    fragment_archive_md5=MARYLAND_HOTEL3_SPEC.fragment_archive_md5,
    fragment_archive_sha256=MARYLAND_HOTEL3_SPEC.fragment_archive_sha256,
    fragment_count=MARYLAND_HOTEL3_SPEC.fragment_count,
)
PHASE37_SCENE_INPUTS = (
    REDKITCHEN_PHASE37_INPUT,
    MARYLAND_PHASE37_INPUT,
)


@dataclass(frozen=True)
class ExtractedFragmentVerification:
    scene_name: str
    archive_path: str
    archive_sha256: str
    fragment_count: int
    extracted_manifest_sha256: str
    verified_against_archive_members: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessedFragment:
    index: int
    raw_point_count: int
    downsampled_point_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FGRPrediction:
    source_index: int
    target_index: int
    fragment_count: int
    pair_random_seed: int
    benchmark_target_to_source_matrix: tuple[tuple[float, ...], ...]
    fitness: float
    inlier_rmse: float
    correspondence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "benchmark_target_to_source_matrix": [
                list(row) for row in self.benchmark_target_to_source_matrix
            ],
        }


@dataclass(frozen=True)
class FGRScenePredictions:
    scene: Phase37SceneInput
    source_root: str
    fragment_verification: ExtractedFragmentVerification
    pair_universe: str
    expected_pair_count: int
    preprocessed_fragments: tuple[PreprocessedFragment, ...]
    predictions: tuple[FGRPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scene": self.scene.to_dict(),
            "source_root": self.source_root,
            "fragment_verification": self.fragment_verification.to_dict(),
            "pair_universe": self.pair_universe,
            "expected_pair_count": self.expected_pair_count,
            "preprocessed_fragments": [
                item.to_dict() for item in self.preprocessed_fragments
            ],
            "predictions": [item.to_dict() for item in self.predictions],
        }


@dataclass(frozen=True)
class Open3DFGRPredictionArtifact:
    artifact_schema: str
    role: str
    pipeline_name: str
    label_boundary: str
    pair_selection_rule: str
    matrix_convention: str
    generation_correction_history: str
    open3d_version: str
    python_version: str
    platform: str
    open3d_build_config: dict[str, object]
    official_parameter_source: str
    parameters: dict[str, object]
    scenes: tuple[FGRScenePredictions, ...]
    external_method_generation_reproduced: bool
    ground_truth_artifacts_accessed_by_generator: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "pipeline_name": self.pipeline_name,
            "label_boundary": self.label_boundary,
            "pair_selection_rule": self.pair_selection_rule,
            "matrix_convention": self.matrix_convention,
            "generation_correction_history": self.generation_correction_history,
            "open3d_version": self.open3d_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "open3d_build_config": self.open3d_build_config,
            "official_parameter_source": self.official_parameter_source,
            "parameters": self.parameters,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "external_method_generation_reproduced": (
                self.external_method_generation_reproduced
            ),
            "ground_truth_artifacts_accessed_by_generator": (
                self.ground_truth_artifacts_accessed_by_generator
            ),
        }


def nonconsecutive_fragment_pairs(
    fragment_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return every source<target pair separated by at least one fragment."""

    if fragment_count < 3:
        raise ValueError("fragment_count must be at least three")
    return tuple(
        (source, target)
        for source in range(fragment_count)
        for target in range(source + 2, fragment_count)
    )


def phase37_parameters() -> dict[str, object]:
    return {
        "voxel_size_meters": VOXEL_SIZE_METERS,
        "normal_radius_meters": (VOXEL_SIZE_METERS * NORMAL_RADIUS_MULTIPLIER),
        "normal_max_neighbors": NORMAL_MAX_NEIGHBORS,
        "fpfh_radius_meters": VOXEL_SIZE_METERS * FPFH_RADIUS_MULTIPLIER,
        "fpfh_max_neighbors": FPFH_MAX_NEIGHBORS,
        "fgr_maximum_correspondence_distance_meters": (
            VOXEL_SIZE_METERS * FGR_DISTANCE_MULTIPLIER
        ),
        "fgr_division_factor": FGR_DIVISION_FACTOR,
        "fgr_use_absolute_scale": FGR_USE_ABSOLUTE_SCALE,
        "fgr_decrease_mu": FGR_DECREASE_MU,
        "fgr_iteration_count": FGR_ITERATION_COUNT,
        "fgr_tuple_scale": FGR_TUPLE_SCALE,
        "fgr_maximum_tuple_count": FGR_MAXIMUM_TUPLE_COUNT,
        "fgr_tuple_test": FGR_TUPLE_TEST,
        "base_random_seed": BASE_RANDOM_SEED,
    }


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _verify_extracted_fragments(
    root: Path,
    scene: Phase37SceneInput,
) -> ExtractedFragmentVerification:
    archive_path = root / scene.fragment_archive_name
    if scene == REDKITCHEN_PHASE37_INPUT:
        archive = verify_redkitchen_archive(archive_path, role="fragments")
    elif scene == MARYLAND_PHASE37_INPUT:
        archive = verify_threedmatch_scene_archive(
            archive_path,
            MARYLAND_HOTEL3_SPEC,
            role="fragments",
        )
    else:
        raise ValueError("unsupported Phase-37 scene")
    manifest = hashlib.sha256()
    with zipfile.ZipFile(archive_path) as source:
        for index in range(scene.fragment_count):
            relative = f"{scene.scene_name}/cloud_bin_{index}.ply"
            extracted = root / Path(relative)
            if not extracted.is_file():
                raise FileNotFoundError(extracted)
            with extracted.open("rb") as stream:
                extracted_sha = _sha256_stream(stream)
            with source.open(relative) as stream:
                archived_sha = _sha256_stream(stream)
            if extracted_sha != archived_sha:
                raise ValueError(f"extracted fragment differs from archive: {relative}")
            manifest.update(relative.encode("utf-8"))
            manifest.update(b"\0")
            manifest.update(extracted_sha.encode("ascii"))
            manifest.update(b"\n")
    return ExtractedFragmentVerification(
        scene_name=scene.scene_name,
        archive_path=str(archive_path),
        archive_sha256=archive.sha256,
        fragment_count=scene.fragment_count,
        extracted_manifest_sha256=manifest.hexdigest(),
        verified_against_archive_members=True,
    )


def _load_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is optional; run Phase 37 with open3d==0.19.0 installed"
        ) from error
    if o3d.__version__ != OPEN3D_VERSION:
        raise RuntimeError(
            f"Phase 37 requires Open3D {OPEN3D_VERSION}, got {o3d.__version__}"
        )
    return o3d


def _preprocess_fragment(o3d: Any, path: Path) -> tuple[Any, Any, int, int]:
    cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=True)
    raw_count = len(cloud.points)
    if raw_count == 0:
        raise ValueError(f"Open3D read no points from {path}")
    down = cloud.voxel_down_sample(VOXEL_SIZE_METERS)
    down_count = len(down.points)
    if down_count < 3:
        raise ValueError(f"Open3D downsampling left too few points in {path}")
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
    return down, feature, raw_count, down_count


def _pair_seed(scene_index: int, source: int, target: int) -> int:
    return BASE_RANDOM_SEED + 1_000_003 * scene_index + 1_009 * source + target


def _generate_scene(
    o3d: Any,
    root: Path,
    scene: Phase37SceneInput,
    scene_index: int,
) -> FGRScenePredictions:
    verification = _verify_extracted_fragments(root, scene)
    prepared: list[tuple[Any, Any]] = []
    fragment_summaries: list[PreprocessedFragment] = []
    for index in range(scene.fragment_count):
        path = root / scene.scene_name / f"cloud_bin_{index}.ply"
        down, feature, raw_count, down_count = _preprocess_fragment(o3d, path)
        prepared.append((down, feature))
        fragment_summaries.append(
            PreprocessedFragment(
                index=index,
                raw_point_count=raw_count,
                downsampled_point_count=down_count,
            )
        )
        print(
            f"[{scene.scene_name}] preprocessed {index + 1}/{scene.fragment_count}",
            flush=True,
        )
    pairs = nonconsecutive_fragment_pairs(scene.fragment_count)
    predictions: list[FGRPrediction] = []
    for pair_index, (source, target) in enumerate(pairs, start=1):
        seed = _pair_seed(scene_index, source, target)
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
        prediction_rotation_radians(matrix)
        predictions.append(
            FGRPrediction(
                source_index=source,
                target_index=target,
                fragment_count=scene.fragment_count,
                pair_random_seed=seed,
                benchmark_target_to_source_matrix=tuple(
                    tuple(float(value) for value in row) for row in matrix
                ),
                fitness=float(result.fitness),
                inlier_rmse=float(result.inlier_rmse),
                correspondence_count=len(result.correspondence_set),
            )
        )
        if pair_index % 50 == 0 or pair_index == len(pairs):
            print(
                f"[{scene.scene_name}] registered {pair_index}/{len(pairs)}",
                flush=True,
            )
    return FGRScenePredictions(
        scene=scene,
        source_root=str(root),
        fragment_verification=verification,
        pair_universe="all_source_lt_target_pairs_with_target_minus_source_gt_1",
        expected_pair_count=len(pairs),
        preprocessed_fragments=tuple(fragment_summaries),
        predictions=tuple(predictions),
    )


def generate_open3d_fgr_predictions(
    redkitchen_root: str | Path,
    maryland_root: str | Path,
) -> Open3DFGRPredictionArtifact:
    """Generate both complete prediction sets without opening evaluation data."""

    o3d = _load_open3d()
    roots = (Path(redkitchen_root), Path(maryland_root))
    scenes = tuple(
        _generate_scene(o3d, root, scene, index)
        for index, (root, scene) in enumerate(
            zip(roots, PHASE37_SCENE_INPUTS, strict=True)
        )
    )
    build_config = {
        str(key): value
        for key, value in dict(getattr(o3d, "_build_config", {})).items()
        if isinstance(value, (bool, float, int, str))
    }
    return Open3DFGRPredictionArtifact(
        artifact_schema="pftf_alpha_open3d_fgr_predictions_phase37/v2",
        role="label_free_fixed_parameter_independent_pipeline_predictions",
        pipeline_name="open3d_0.19.0_fpfh_fast_global_registration",
        label_boundary=(
            "generator accepts only verified fragment archives and extracted "
            "fragment PLY files; it has no evaluation archive argument and "
            "does not access 3dmatch.log, gt.log, or gt.info"
        ),
        pair_selection_rule=(
            "all source<target fragment pairs with target-source>1; no "
            "overlap, ground-truth, fitness, or label filtering"
        ),
        matrix_convention=(
            "for header (source_index,target_index), Open3D moving source is "
            "the target-index fragment and fixed target is the source-index "
            "fragment; the stored matrix therefore aligns fragment2 to "
            "fragment1 exactly as the official 3DMatch log convention"
        ),
        generation_correction_history=(
            "a preliminary uncommitted v1 execution passed fragment1 as the "
            "Open3D moving source; its zero-correct audit exposed a direction "
            "mismatch. Official 3DMatch register2Fragments.m and "
            "getGtInfoLog.m independently confirm fragment2-to-fragment1, so "
            "v1 was invalidated and v2 corrects only call direction without "
            "changing pairs, features, FGR parameters, guard, or gates"
        ),
        open3d_version=o3d.__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        open3d_build_config=build_config,
        official_parameter_source=OFFICIAL_OPEN3D_TUTORIAL_URL,
        parameters=phase37_parameters(),
        scenes=scenes,
        external_method_generation_reproduced=True,
        ground_truth_artifacts_accessed_by_generator=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--redkitchen-root",
        type=Path,
        default=Path("benchmark-data/3dmatch_redkitchen"),
    )
    parser.add_argument(
        "--maryland-root",
        type=Path,
        default=Path("benchmark-data/3dmatch_maryland_hotel3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/open3d_fgr_predictions_phase37.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = generate_open3d_fgr_predictions(
        args.redkitchen_root,
        args.maryland_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
