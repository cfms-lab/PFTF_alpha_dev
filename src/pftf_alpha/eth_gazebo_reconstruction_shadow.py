"""Run the preregistered Phase-40 ETH Gazebo alpha reconstruction shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .eth_gazebo_reconstruction_protocol import (
    ALPHA_METERS,
    DECISION_SCHEMA,
    DISTANCE_THRESHOLD_FRACTION,
    EVALUATION_SEED,
    EXPECTED_DECISION_SHA256,
    EXPECTED_PREDICTION_SHA256,
    FUSION_VOXEL_METERS,
    HELDOUT_MODULUS,
    HELDOUT_REMAINDER,
    PREDICTION_SCHEMA,
    PROTOCOL_SCHEMA,
    RECALL_NONREGRESSION_TOLERANCE,
    REFERENCE_VOXEL_METERS,
    ROI_LOWER_QUANTILE,
    ROI_MARGIN_METERS,
    ROI_UPPER_QUANTILE,
    SOURCE_VOXEL_METERS,
    SURFACE_SAMPLE_COUNT,
    VALIDATION_SOURCE_INDICES,
)
from .eth_gazebo_validation_protocol import (
    ARCHIVE_NAME,
    ARCHIVE_SHA256,
    SCAN_MEMBERS,
    verify_gazebo_archive_directory,
)
from .eth_open3d_fgr_pipeline import _load_open3d, _load_xyz
from .surface import (
    SurfaceDistanceMetrics,
    SurfaceMesh,
    mesh_statistics,
    sample_triangle_mesh,
    surface_distance_metrics,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
EXPECTED_PROTOCOL_SHA256 = (
    "cd829c3e1c1d9585ccef5c6fa98311e6a62507d9f57fb55e960405dd53ba635b"
)


@dataclass(frozen=True)
class ReconstructionEndpoint:
    normalized_chamfer_squared: float
    normalized_hausdorff: float
    geometry_loss: float
    precision: float
    recall: float
    fscore: float
    used_vertices: int
    edges: int
    faces: int
    connected_components: int
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    boundary_edges: int
    nonmanifold_edges: int
    nonmanifold_edge_fraction: float
    watertight: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionShadowCase:
    source_index: int
    pair_count: int
    accepted_pair_count: int
    rejected_pair_count: int
    observed_source_point_count: int
    heldout_reference_point_count: int
    baseline_fused_point_count: int
    guard_fused_point_count: int
    characteristic_length_meters: float
    baseline: ReconstructionEndpoint
    guard: ReconstructionEndpoint
    geometry_loss_margin: float
    fscore_margin: float
    recall_margin: float
    component_count_reduction: int
    betti_1_reduction: int
    betti_2_reduction: int
    nonmanifold_fraction_reduction: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["baseline"] = self.baseline.to_dict()
        payload["guard"] = self.guard.to_dict()
        return payload


@dataclass(frozen=True)
class ReconstructionShadowSummary:
    case_count: int
    baseline_pair_count: int
    guarded_pair_count: int
    rejected_pair_count: int
    mean_baseline_geometry_loss: float
    mean_guard_geometry_loss: float
    mean_geometry_loss_margin: float
    geometry_win_count: int
    mean_baseline_fscore: float
    mean_guard_fscore: float
    mean_fscore_margin: float
    mean_baseline_recall: float
    mean_guard_recall: float
    mean_recall_margin: float
    mean_baseline_components: float
    mean_guard_components: float
    mean_baseline_betti_1: float
    mean_guard_betti_1: float
    mean_baseline_betti_2: float
    mean_guard_betti_2: float
    mean_baseline_nonmanifold_edge_fraction: float
    mean_guard_nonmanifold_edge_fraction: float
    mean_nonmanifold_fraction_reduction: float
    all_meshes_materialized: bool
    geometry_loss_improved: bool
    fscore_nonregressed: bool
    recall_nonregressed: bool
    geometry_shadow_supported: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GazeboReconstructionShadow:
    artifact_schema: str
    role: str
    protocol_artifact_path: str
    protocol_artifact_sha256: str
    prediction_artifact_path: str
    prediction_artifact_sha256: str
    decision_artifact_path: str
    decision_artifact_sha256: str
    archive_path: str
    archive_sha256: str
    opened_archive_members: tuple[str, ...]
    development_source_index_excluded: int
    validation_source_indices: tuple[int, ...]
    alpha_meters: float
    cases: tuple[ReconstructionShadowCase, ...]
    summary: ReconstructionShadowSummary
    validation_reference_values_accessed_by_evaluator: bool
    registration_label_values_accessed: bool
    real_alpha_reconstruction_shadow_executed: bool
    geometry_shadow_supported: bool
    topology_endpoint_comparison_executed: bool
    topology_correctness_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["opened_archive_members"] = list(self.opened_archive_members)
        payload["validation_source_indices"] = list(
            self.validation_source_indices
        )
        payload["cases"] = [case.to_dict() for case in self.cases]
        payload["summary"] = self.summary.to_dict()
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_locked_json(
    path: Path,
    *,
    expected_sha256: str,
    expected_schema: str,
) -> Mapping[str, object]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"Gazebo artifact SHA-256 mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gazebo artifact must be a JSON object")
    if payload.get("artifact_schema") != expected_schema:
        raise ValueError("Gazebo artifact schema mismatch")
    return payload


def _verify_protocol(path: Path) -> Mapping[str, object]:
    payload = _load_locked_json(
        path,
        expected_sha256=EXPECTED_PROTOCOL_SHA256,
        expected_schema=PROTOCOL_SCHEMA,
    )
    expected = {
        "archive_sha256": ARCHIVE_SHA256,
        "prediction_artifact_sha256": EXPECTED_PREDICTION_SHA256,
        "decision_artifact_sha256": EXPECTED_DECISION_SHA256,
        "development_source_index": 0,
        "source_voxel_meters": SOURCE_VOXEL_METERS,
        "reference_voxel_meters": REFERENCE_VOXEL_METERS,
        "fusion_voxel_meters": FUSION_VOXEL_METERS,
        "alpha_meters": ALPHA_METERS,
        "surface_sample_count": SURFACE_SAMPLE_COUNT,
        "distance_threshold_fraction": DISTANCE_THRESHOLD_FRACTION,
        "registration_label_values_accessed": False,
        "validation_reference_values_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase-40 protocol mismatch: {key}")
    raw_sources = payload.get("validation_sources")
    if not isinstance(raw_sources, list):
        raise ValueError("Phase-40 protocol validation sources are missing")
    sources = tuple(int(row["source_index"]) for row in raw_sources)
    if sources != VALIDATION_SOURCE_INDICES:
        raise ValueError("Phase-40 protocol validation source set mismatch")
    return payload


def _split_source_points(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    indices = np.arange(points.shape[0])
    heldout = indices % HELDOUT_MODULUS == HELDOUT_REMAINDER
    return (
        np.ascontiguousarray(points[~heldout]),
        np.ascontiguousarray(points[heldout]),
    )


def _voxel_downsample(o3d: object, points: FloatArray, voxel: float) -> FloatArray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    downsampled = cloud.voxel_down_sample(voxel)
    return np.ascontiguousarray(
        np.asarray(downsampled.points, dtype=np.float64)
    )


def _transform_and_crop(
    points: FloatArray,
    matrix: Sequence[Sequence[float]],
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    transformed = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T[:, :3]
    retained = np.all((transformed >= lower) & (transformed <= upper), axis=1)
    return np.ascontiguousarray(transformed[retained])


def _canonical_surface(mesh: object) -> SurfaceMesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
    if faces.shape[0] > 0:
        distinct = np.asarray(
            [len(set(face.tolist())) == 3 for face in faces], dtype=bool
        )
        faces = np.unique(np.sort(faces[distinct], axis=1), axis=0)
    return SurfaceMesh(vertices=vertices, faces=faces)


def _alpha_surface(o3d: object, points: FloatArray) -> SurfaceMesh:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        cloud,
        ALPHA_METERS,
    )
    return _canonical_surface(mesh)


def _endpoint(
    mesh: SurfaceMesh,
    reference: FloatArray,
    *,
    characteristic_length: float,
    seed: int,
) -> ReconstructionEndpoint:
    statistics = mesh_statistics(mesh)
    sampled = sample_triangle_mesh(mesh, SURFACE_SAMPLE_COUNT, seed=seed)
    if sampled.shape[0] == 0:
        distances = SurfaceDistanceMetrics(
            chamfer_squared=4.0 * characteristic_length**2,
            hausdorff=2.0 * characteristic_length,
            precision=0.0,
            recall=0.0,
            fscore=0.0,
        )
    else:
        distances = surface_distance_metrics(
            sampled,
            reference,
            threshold=DISTANCE_THRESHOLD_FRACTION * characteristic_length,
        )
    normalized_chamfer = distances.chamfer_squared / characteristic_length**2
    normalized_hausdorff = distances.hausdorff / characteristic_length
    return ReconstructionEndpoint(
        normalized_chamfer_squared=float(normalized_chamfer),
        normalized_hausdorff=float(normalized_hausdorff),
        geometry_loss=float(normalized_chamfer + normalized_hausdorff),
        precision=distances.precision,
        recall=distances.recall,
        fscore=distances.fscore,
        used_vertices=statistics.used_vertices,
        edges=statistics.edges,
        faces=statistics.faces,
        connected_components=statistics.connected_components,
        betti_0=statistics.betti_0,
        betti_1=statistics.betti_1,
        betti_2=statistics.betti_2,
        euler_characteristic=statistics.euler_characteristic,
        boundary_edges=statistics.boundary_edges,
        nonmanifold_edges=statistics.nonmanifold_edges,
        nonmanifold_edge_fraction=(
            statistics.nonmanifold_edges / max(statistics.edges, 1)
        ),
        watertight=statistics.watertight,
    )


def _summarize(
    cases: Sequence[ReconstructionShadowCase],
) -> ReconstructionShadowSummary:
    if not cases:
        raise ValueError("Phase-40 reconstruction shadow requires cases")

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    baseline_geometry = mean([case.baseline.geometry_loss for case in cases])
    guard_geometry = mean([case.guard.geometry_loss for case in cases])
    baseline_fscore = mean([case.baseline.fscore for case in cases])
    guard_fscore = mean([case.guard.fscore for case in cases])
    baseline_recall = mean([case.baseline.recall for case in cases])
    guard_recall = mean([case.guard.recall for case in cases])
    geometry_improved = guard_geometry < baseline_geometry
    fscore_nonregressed = guard_fscore >= baseline_fscore
    recall_nonregressed = (
        guard_recall >= baseline_recall - RECALL_NONREGRESSION_TOLERANCE
    )
    all_meshes = all(
        case.baseline.faces > 0 and case.guard.faces > 0 for case in cases
    )
    return ReconstructionShadowSummary(
        case_count=len(cases),
        baseline_pair_count=sum(case.pair_count for case in cases),
        guarded_pair_count=sum(case.accepted_pair_count for case in cases),
        rejected_pair_count=sum(case.rejected_pair_count for case in cases),
        mean_baseline_geometry_loss=baseline_geometry,
        mean_guard_geometry_loss=guard_geometry,
        mean_geometry_loss_margin=baseline_geometry - guard_geometry,
        geometry_win_count=sum(case.geometry_loss_margin > 0.0 for case in cases),
        mean_baseline_fscore=baseline_fscore,
        mean_guard_fscore=guard_fscore,
        mean_fscore_margin=guard_fscore - baseline_fscore,
        mean_baseline_recall=baseline_recall,
        mean_guard_recall=guard_recall,
        mean_recall_margin=guard_recall - baseline_recall,
        mean_baseline_components=mean(
            [float(case.baseline.connected_components) for case in cases]
        ),
        mean_guard_components=mean(
            [float(case.guard.connected_components) for case in cases]
        ),
        mean_baseline_betti_1=mean(
            [float(case.baseline.betti_1) for case in cases]
        ),
        mean_guard_betti_1=mean([float(case.guard.betti_1) for case in cases]),
        mean_baseline_betti_2=mean(
            [float(case.baseline.betti_2) for case in cases]
        ),
        mean_guard_betti_2=mean([float(case.guard.betti_2) for case in cases]),
        mean_baseline_nonmanifold_edge_fraction=mean(
            [case.baseline.nonmanifold_edge_fraction for case in cases]
        ),
        mean_guard_nonmanifold_edge_fraction=mean(
            [case.guard.nonmanifold_edge_fraction for case in cases]
        ),
        mean_nonmanifold_fraction_reduction=mean(
            [case.nonmanifold_fraction_reduction for case in cases]
        ),
        all_meshes_materialized=all_meshes,
        geometry_loss_improved=geometry_improved,
        fscore_nonregressed=fscore_nonregressed,
        recall_nonregressed=recall_nonregressed,
        geometry_shadow_supported=(
            all_meshes
            and geometry_improved
            and fscore_nonregressed
            and recall_nonregressed
        ),
    )


def evaluate_gazebo_reconstruction_shadow(
    protocol_path: str | Path,
    prediction_path: str | Path,
    decision_path: str | Path,
    archive_path: str | Path,
) -> GazeboReconstructionShadow:
    protocol_file = Path(protocol_path)
    prediction_file = Path(prediction_path)
    decision_file = Path(decision_path)
    archive_file = Path(archive_path)
    _verify_protocol(protocol_file)
    predictions = _load_locked_json(
        prediction_file,
        expected_sha256=EXPECTED_PREDICTION_SHA256,
        expected_schema=PREDICTION_SCHEMA,
    )
    decisions = _load_locked_json(
        decision_file,
        expected_sha256=EXPECTED_DECISION_SHA256,
        expected_schema=DECISION_SCHEMA,
    )
    if predictions.get("validation_label_member_opened") is not False:
        raise ValueError("prediction artifact crossed the registration label boundary")
    if decisions.get("validation_label_values_accessed") is not False:
        raise ValueError("decision artifact crossed the registration label boundary")
    raw_predictions = predictions.get("predictions")
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_predictions, list) or not isinstance(raw_decisions, list):
        raise ValueError("Gazebo prediction or decision rows are missing")
    if len(raw_predictions) != len(raw_decisions):
        raise ValueError("Gazebo prediction and decision counts differ")
    prediction_by_source: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    accept_by_pair: dict[tuple[int, int], bool] = {}
    for prediction, decision in zip(
        raw_predictions,
        raw_decisions,
        strict=True,
    ):
        pair = (
            int(prediction["source_index"]),
            int(prediction["target_index"]),
        )
        decision_pair = (
            int(decision["source_index"]),
            int(decision["target_index"]),
        )
        if pair != decision_pair:
            raise ValueError("Gazebo prediction and decision pair order differs")
        prediction_by_source[pair[0]].append(prediction)
        accept_by_pair[pair] = bool(decision["guarded_accept"])

    verification = verify_gazebo_archive_directory(archive_file)
    o3d = _load_open3d()
    raw_scans: list[FloatArray] = []
    opened_members: list[str] = []
    with zipfile.ZipFile(archive_file) as source:
        for member in SCAN_MEMBERS:
            opened_members.append(member)
            with source.open(member) as stream:
                raw_scans.append(_load_xyz(stream))
    if tuple(opened_members) != SCAN_MEMBERS:
        raise RuntimeError("Phase-40 evaluator opened an unexpected member set")
    target_scans = tuple(
        _voxel_downsample(o3d, points, SOURCE_VOXEL_METERS)
        for points in raw_scans
    )

    cases: list[ReconstructionShadowCase] = []
    for source_index in VALIDATION_SOURCE_INDICES:
        source_observed_raw, source_reference_raw = _split_source_points(
            raw_scans[source_index]
        )
        source_observed = _voxel_downsample(
            o3d,
            source_observed_raw,
            SOURCE_VOXEL_METERS,
        )
        reference = _voxel_downsample(
            o3d,
            source_reference_raw,
            REFERENCE_VOXEL_METERS,
        )
        lower = (
            np.quantile(source_observed, ROI_LOWER_QUANTILE, axis=0)
            - ROI_MARGIN_METERS
        )
        upper = (
            np.quantile(source_observed, ROI_UPPER_QUANTILE, axis=0)
            + ROI_MARGIN_METERS
        )
        source_inside = source_observed[
            np.all((source_observed >= lower) & (source_observed <= upper), axis=1)
        ]
        reference = np.ascontiguousarray(
            reference[np.all((reference >= lower) & (reference <= upper), axis=1)]
        )
        baseline_chunks = [source_inside]
        guard_chunks = [source_inside]
        source_predictions = prediction_by_source[source_index]
        accepted_count = 0
        for prediction in source_predictions:
            target_index = int(prediction["target_index"])
            transformed = _transform_and_crop(
                target_scans[target_index],
                prediction["target_to_source_matrix"],
                lower,
                upper,
            )
            baseline_chunks.append(transformed)
            if accept_by_pair[(source_index, target_index)]:
                guard_chunks.append(transformed)
                accepted_count += 1
        baseline_points = _voxel_downsample(
            o3d,
            np.vstack(baseline_chunks),
            FUSION_VOXEL_METERS,
        )
        guard_points = _voxel_downsample(
            o3d,
            np.vstack(guard_chunks),
            FUSION_VOXEL_METERS,
        )
        baseline_mesh = _alpha_surface(o3d, baseline_points)
        guard_mesh = _alpha_surface(o3d, guard_points)
        characteristic_length = float(np.linalg.norm(upper - lower))
        if not math.isfinite(characteristic_length) or characteristic_length <= 0.0:
            raise ValueError("Phase-40 source ROI has degenerate extent")
        baseline_endpoint = _endpoint(
            baseline_mesh,
            reference,
            characteristic_length=characteristic_length,
            seed=EVALUATION_SEED + 2 * source_index,
        )
        guard_endpoint = _endpoint(
            guard_mesh,
            reference,
            characteristic_length=characteristic_length,
            seed=EVALUATION_SEED + 2 * source_index,
        )
        rejected_count = len(source_predictions) - accepted_count
        if accepted_count == 0 or rejected_count == 0:
            raise ValueError(
                "Phase-40 validation source no longer meets selection rule"
            )
        cases.append(
            ReconstructionShadowCase(
                source_index=source_index,
                pair_count=len(source_predictions),
                accepted_pair_count=accepted_count,
                rejected_pair_count=rejected_count,
                observed_source_point_count=int(source_inside.shape[0]),
                heldout_reference_point_count=int(reference.shape[0]),
                baseline_fused_point_count=int(baseline_points.shape[0]),
                guard_fused_point_count=int(guard_points.shape[0]),
                characteristic_length_meters=characteristic_length,
                baseline=baseline_endpoint,
                guard=guard_endpoint,
                geometry_loss_margin=(
                    baseline_endpoint.geometry_loss - guard_endpoint.geometry_loss
                ),
                fscore_margin=guard_endpoint.fscore - baseline_endpoint.fscore,
                recall_margin=guard_endpoint.recall - baseline_endpoint.recall,
                component_count_reduction=(
                    baseline_endpoint.connected_components
                    - guard_endpoint.connected_components
                ),
                betti_1_reduction=(
                    baseline_endpoint.betti_1 - guard_endpoint.betti_1
                ),
                betti_2_reduction=(
                    baseline_endpoint.betti_2 - guard_endpoint.betti_2
                ),
                nonmanifold_fraction_reduction=(
                    baseline_endpoint.nonmanifold_edge_fraction
                    - guard_endpoint.nonmanifold_edge_fraction
                ),
            )
        )
        print(
            f"[ETH Gazebo Phase 40] source {source_index}: "
            f"{len(source_predictions)} pairs, {rejected_count} rejected",
            flush=True,
        )
    frozen_cases = tuple(cases)
    summary = _summarize(frozen_cases)
    return GazeboReconstructionShadow(
        artifact_schema="pftf_alpha_eth_gazebo_reconstruction_shadow_phase40/v1",
        role="post_protocol_real_alpha_reconstruction_shadow",
        protocol_artifact_path=str(protocol_file),
        protocol_artifact_sha256=EXPECTED_PROTOCOL_SHA256,
        prediction_artifact_path=str(prediction_file),
        prediction_artifact_sha256=EXPECTED_PREDICTION_SHA256,
        decision_artifact_path=str(decision_file),
        decision_artifact_sha256=EXPECTED_DECISION_SHA256,
        archive_path=str(archive_file),
        archive_sha256=verification.sha256,
        opened_archive_members=tuple(opened_members),
        development_source_index_excluded=0,
        validation_source_indices=VALIDATION_SOURCE_INDICES,
        alpha_meters=ALPHA_METERS,
        cases=frozen_cases,
        summary=summary,
        validation_reference_values_accessed_by_evaluator=True,
        registration_label_values_accessed=False,
        real_alpha_reconstruction_shadow_executed=summary.all_meshes_materialized,
        geometry_shadow_supported=summary.geometry_shadow_supported,
        topology_endpoint_comparison_executed=summary.all_meshes_materialized,
        topology_correctness_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
        claim_boundary=(
            "The frozen p90 decision is used only to route scan provenance into a "
            "fixed-alpha real-data shadow. Geometry is source-view heldout "
            "consistency, not full-scene or physical surface truth. Topology has no "
            "ground-truth target and is descriptive. Registration correctness labels, "
            "correspondence identity, deployed trimming, and deployment are not used "
            "or supported."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "benchmark-out/eth_gazebo_reconstruction_protocol_phase40.json"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_predictions_phase39.json"),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_rotation_decisions_phase39.json"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("benchmark-data/eth_gazebo_summer") / ARCHIVE_NAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/eth_gazebo_reconstruction_shadow_phase40.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_gazebo_reconstruction_shadow(
        args.protocol,
        args.predictions,
        args.decisions,
        args.archive,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
