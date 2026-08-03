"""Evaluate the frozen Phase-50 methods on S3DIS calibration room layers."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .adaptive import pca_anisotropic_filtration
from .filtration import AlphaFiltration
from .s3dis_two_layer_calibration import load_xyz
from .s3dis_two_layer_calibration_benchmark import (
    DEFAULT_OBSERVED_PER_LAYER,
    DEFAULT_REFERENCE_PER_LAYER,
    DEFAULT_SURFACE_SAMPLE_COUNT,
    RealTwoLayerCase,
    _evaluate,
    _split_layer,
    _true_safe,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .shared_trend_inference import SharedTrendConfig, construct_shared_trend_surface
from .two_layer_confirmatory import ConfirmatoryMethodMetrics
from .two_layer_confirmatory_protocol import (
    B5_MAX_NORMAL_PENALTY,
    B5_SCALE_MULTIPLIER,
    K_NEIGHBORS,
    M1_SCALE_MULTIPLIER,
    M1_WEIGHT_SCALE,
)
from .two_layer_connectivity import (
    construct_two_layer_surface,
    route_two_layer_output,
)
from .weighted_alpha import PointSubmersionError, weighted_alpha_filtration

RESULT_SCHEMA = "pftf_alpha_s3dis_room_layer_calibration_benchmark_phase51b/v1"
GENERAL_POSITION_JOGGLE = 1.0e-4


@dataclass(frozen=True)
class RoomLayerBenchmarkCase:
    name: str
    area: str
    room: str
    normal_angle_degrees: float
    bbox_overlap_fraction: float
    gap_to_median_spacing: float
    gap_to_residual_snr: float
    candidate_decision: SamplingGateDecision
    base_decision: SamplingGateDecision
    candidate_true_safe: bool
    candidate_safe_accept: bool
    candidate_false_safe: bool
    base_true_safe: bool
    base_safe_accept: bool
    base_false_safe: bool
    candidate: ConfirmatoryMethodMetrics
    base: ConfirmatoryMethodMetrics
    b5: ConfirmatoryMethodMetrics | None
    b5_construction_failed: bool
    m1: ConfirmatoryMethodMetrics | None
    candidate_fscore_wins_b5: bool | None
    candidate_fscore_wins_m1: bool | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_decision"] = self.candidate_decision.value
        payload["base_decision"] = self.base_decision.value
        return payload


def _load_paths(root: Path, paths: list[str]) -> np.ndarray:
    return np.unique(
        np.vstack([load_xyz(root / relative) for relative in paths]),
        axis=0,
    )


def build_room_layer_case(
    calibration_root: str | Path,
    pair: dict[str, object],
    *,
    observed_per_layer: int = DEFAULT_OBSERVED_PER_LAYER,
    reference_per_layer: int = DEFAULT_REFERENCE_PER_LAYER,
) -> RealTwoLayerCase:
    root = Path(calibration_root)
    floor_points = _load_paths(root, list(pair["floor_paths"]))
    ceiling_points = _load_paths(root, list(pair["ceiling_paths"]))
    floor = pair["floor"]
    if not isinstance(floor, dict):
        raise ValueError("room-layer pair has no floor summary")
    center = np.asarray(floor["centroid"], dtype=np.float64)
    axes = np.column_stack(
        (
            np.asarray(floor["tangent_u"], dtype=np.float64),
            np.asarray(floor["tangent_v"], dtype=np.float64),
        )
    )
    floor_uv = (floor_points - center) @ axes
    ceiling_uv = (ceiling_points - center) @ axes
    common_lower = np.maximum(
        np.min(floor_uv, axis=0),
        np.min(ceiling_uv, axis=0),
    )
    common_upper = np.minimum(
        np.max(floor_uv, axis=0),
        np.max(ceiling_uv, axis=0),
    )
    common_extent = common_upper - common_lower
    if np.any(common_extent <= 0.0):
        raise ValueError("floor and ceiling have no common projected footprint")
    floor_inside = np.all(
        (floor_uv >= common_lower) & (floor_uv <= common_upper), axis=1
    )
    ceiling_inside = np.all(
        (ceiling_uv >= common_lower) & (ceiling_uv <= common_upper), axis=1
    )
    cropped_floor = floor_points[floor_inside]
    cropped_ceiling = ceiling_points[ceiling_inside]
    footprint_center = 0.5 * (common_lower + common_upper)
    origin = center + axes @ footprint_center
    diagonal = float(np.linalg.norm(common_extent))
    normalized_floor = (cropped_floor - origin) / diagonal
    normalized_ceiling = (cropped_ceiling - origin) / diagonal
    floor_observed, floor_reference = _split_layer(
        normalized_floor,
        observed_per_layer,
        reference_per_layer,
        salt=51_200_001,
    )
    ceiling_observed, ceiling_reference = _split_layer(
        normalized_ceiling,
        observed_per_layer,
        reference_per_layer,
        salt=51_200_003,
    )
    points = np.vstack((floor_observed, ceiling_observed))
    seed_bytes = hashlib.sha256(
        f"{pair['area']}/{pair['room']}".encode()
    ).digest()[:8]
    joggle_seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
    joggle = np.random.default_rng(joggle_seed).normal(
        scale=GENERAL_POSITION_JOGGLE,
        size=points.shape,
    )
    points = points + joggle
    reference = np.vstack((floor_reference, ceiling_reference))
    labels = np.concatenate(
        (
            np.zeros(observed_per_layer, dtype=np.int64),
            np.ones(observed_per_layer, dtype=np.int64),
        )
    )
    return RealTwoLayerCase(
        name=f"{pair['area']}/{pair['room']}",
        points=np.ascontiguousarray(points),
        reference_points=np.ascontiguousarray(reference),
        true_labels=labels,
        characteristic_length=1.0,
    )


def _eligible(
    pair: dict[str, object],
    *,
    maximum_angle_degrees: float,
    minimum_bbox_overlap: float,
    minimum_common_points_per_layer: int,
    minimum_gap_to_residual_snr: float,
    maximum_plane_residual_to_spacing: float,
) -> bool:
    floor = pair["floor"]
    ceiling = pair["ceiling"]
    floor_ratio = float(floor["residual_p95"]) / float(floor["median_spacing"])
    ceiling_ratio = float(ceiling["residual_p95"]) / float(
        ceiling["median_spacing"]
    )
    return bool(
        float(pair["normal_angle_degrees"]) <= maximum_angle_degrees
        and float(pair["bbox_overlap_fraction"]) >= minimum_bbox_overlap
        and int(pair["floor_points_in_common_footprint"])
        >= minimum_common_points_per_layer
        and int(pair["ceiling_points_in_common_footprint"])
        >= minimum_common_points_per_layer
        and float(pair["gap_to_residual_snr"]) >= minimum_gap_to_residual_snr
        and floor_ratio <= maximum_plane_residual_to_spacing
        and ceiling_ratio <= maximum_plane_residual_to_spacing
    )


def evaluate_room_layer_calibration_benchmark(
    calibration_artifact: str | Path,
    *,
    maximum_angle_degrees: float = 5.0,
    minimum_bbox_overlap: float = 0.75,
    minimum_common_points_per_layer: int = 592,
    minimum_gap_to_residual_snr: float = 10.0,
    maximum_plane_residual_to_spacing: float = 10.0,
    observed_per_layer: int = DEFAULT_OBSERVED_PER_LAYER,
    reference_per_layer: int = DEFAULT_REFERENCE_PER_LAYER,
    surface_sample_count: int = DEFAULT_SURFACE_SAMPLE_COUNT,
) -> dict[str, object]:
    artifact_path = Path(calibration_artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    root = str(payload["calibration_root"])
    selected_pairs = [
        pair
        for pair in payload["pairs"]
        if _eligible(
            pair,
            maximum_angle_degrees=maximum_angle_degrees,
            minimum_bbox_overlap=minimum_bbox_overlap,
            minimum_common_points_per_layer=minimum_common_points_per_layer,
            minimum_gap_to_residual_snr=minimum_gap_to_residual_snr,
            maximum_plane_residual_to_spacing=maximum_plane_residual_to_spacing,
        )
    ]
    base_config = SamplingSufficiencyConfig(minimum_separation_snr=3.0)
    candidate_config = SharedTrendConfig(
        k_neighbors=base_config.k_neighbors,
        minimum_cluster_fraction=base_config.minimum_cluster_fraction,
        minimum_separation_snr=base_config.minimum_separation_snr,
        cross_knn_threshold=base_config.cross_knn_threshold,
    )
    rows: list[RoomLayerBenchmarkCase] = []
    skipped: list[dict[str, str]] = []
    for index, pair in enumerate(selected_pairs):
        try:
            case = build_room_layer_case(
                root,
                pair,
                observed_per_layer=observed_per_layer,
                reference_per_layer=reference_per_layer,
            )
        except ValueError as error:
            skipped.append(
                {"name": f"{pair['area']}/{pair['room']}", "reason": str(error)}
            )
            continue
        seed = 51_300_000 + index
        base = construct_two_layer_surface(case.points, base_config)
        candidate, _ = construct_shared_trend_surface(case.points, candidate_config)
        candidate_inferred = _evaluate(
            candidate.mesh,
            case,
            candidate.inference.layer_ids,
            sample_count=surface_sample_count,
            seed=seed,
        )
        candidate_truth = _evaluate(
            candidate.mesh,
            case,
            case.true_labels,
            sample_count=surface_sample_count,
            seed=seed,
        )
        base_inferred = _evaluate(
            base.mesh,
            case,
            base.inference.layer_ids,
            sample_count=surface_sample_count,
            seed=seed,
        )
        base_truth = _evaluate(
            base.mesh,
            case,
            case.true_labels,
            sample_count=surface_sample_count,
            seed=seed,
        )
        try:
            filtration = AlphaFiltration.from_points(case.points)
            b5 = pca_anisotropic_filtration(
                filtration,
                k_neighbors=K_NEIGHBORS,
                max_normal_penalty=B5_MAX_NORMAL_PENALTY,
            )
            b5_truth = _evaluate(
                b5.surface_at(B5_SCALE_MULTIPLIER),
                case,
                case.true_labels,
                sample_count=surface_sample_count,
                seed=seed,
            )
        except ValueError:
            b5_truth = None
        try:
            m1 = weighted_alpha_filtration(
                case.points,
                k_neighbors=K_NEIGHBORS,
                weight_scale=M1_WEIGHT_SCALE,
            )
            m1_truth = _evaluate(
                m1.surface_at(M1_SCALE_MULTIPLIER),
                case,
                case.true_labels,
                sample_count=surface_sample_count,
                seed=seed,
            )
        except PointSubmersionError:
            m1_truth = None
        candidate_decision = route_two_layer_output(candidate, candidate_inferred)
        base_decision = route_two_layer_output(base, base_inferred)
        candidate_safe = _true_safe(candidate_truth)
        base_safe = _true_safe(base_truth)
        candidate_accept = candidate_decision is SamplingGateDecision.ACCEPT
        base_accept = base_decision is SamplingGateDecision.ACCEPT
        candidate_metrics = ConfirmatoryMethodMetrics.from_endpoints(candidate_truth)
        b5_metrics = (
            None
            if b5_truth is None
            else ConfirmatoryMethodMetrics.from_endpoints(b5_truth)
        )
        m1_metrics = (
            None
            if m1_truth is None
            else ConfirmatoryMethodMetrics.from_endpoints(m1_truth)
        )
        rows.append(
            RoomLayerBenchmarkCase(
                name=case.name,
                area=str(pair["area"]),
                room=str(pair["room"]),
                normal_angle_degrees=float(pair["normal_angle_degrees"]),
                bbox_overlap_fraction=float(pair["bbox_overlap_fraction"]),
                gap_to_median_spacing=float(pair["gap_to_median_spacing"]),
                gap_to_residual_snr=float(pair["gap_to_residual_snr"]),
                candidate_decision=candidate_decision,
                base_decision=base_decision,
                candidate_true_safe=candidate_safe,
                candidate_safe_accept=bool(candidate_accept and candidate_safe),
                candidate_false_safe=bool(candidate_accept and not candidate_safe),
                base_true_safe=base_safe,
                base_safe_accept=bool(base_accept and base_safe),
                base_false_safe=bool(base_accept and not base_safe),
                candidate=candidate_metrics,
                base=ConfirmatoryMethodMetrics.from_endpoints(base_truth),
                b5=b5_metrics,
                b5_construction_failed=b5_metrics is None,
                m1=m1_metrics,
                candidate_fscore_wins_b5=(
                    None
                    if b5_metrics is None
                    else candidate_metrics.fscore > b5_metrics.fscore
                ),
                candidate_fscore_wins_m1=(
                    None
                    if m1_metrics is None
                    else candidate_metrics.fscore > m1_metrics.fscore
                ),
            )
        )
    candidate_fscores = [row.candidate.fscore for row in rows]
    available_b5_rows = [row for row in rows if row.b5 is not None]
    available_m1 = [row.m1 for row in rows if row.m1 is not None]
    candidate_mean = float(np.mean(candidate_fscores)) if rows else None
    b5_mean = (
        float(np.mean([row.b5.fscore for row in available_b5_rows]))
        if available_b5_rows
        else None
    )
    m1_mean = (
        float(np.mean([metrics.fscore for metrics in available_m1]))
        if rows and len(available_m1) == len(rows)
        else None
    )
    candidate_geometry = (
        float(np.mean([row.candidate.geometry_loss for row in rows]))
        if rows
        else None
    )
    b5_geometry = (
        float(np.mean([row.b5.geometry_loss for row in available_b5_rows]))
        if available_b5_rows
        else None
    )
    m1_geometry = (
        float(np.mean([metrics.geometry_loss for metrics in available_m1]))
        if rows and len(available_m1) == len(rows)
        else None
    )
    return {
        "artifact_schema": RESULT_SCHEMA,
        "role": "calibration_only_real_floor_ceiling_method_development",
        "calibration_artifact": str(artifact_path),
        "calibration_artifact_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
        "eligibility": {
            "maximum_angle_degrees": maximum_angle_degrees,
            "minimum_bbox_overlap": minimum_bbox_overlap,
            "minimum_common_points_per_layer": minimum_common_points_per_layer,
            "minimum_gap_to_residual_snr": minimum_gap_to_residual_snr,
            "maximum_plane_residual_to_spacing": (
                maximum_plane_residual_to_spacing
            ),
        },
        "observation": {
            "observed_per_layer": observed_per_layer,
            "reference_per_layer": reference_per_layer,
            "surface_sample_count": surface_sample_count,
            "common_footprint_crop": True,
            "coordinate_hash_split": True,
            "general_position_joggle": GENERAL_POSITION_JOGGLE,
            "general_position_joggle_scope": (
                "same deterministic sub-precision observed-XYZ perturbation for "
                "candidate, base, B5, and M1; references remain unchanged"
            ),
        },
        "eligible_pair_count": len(selected_pairs),
        "evaluated_case_count": len(rows),
        "skipped": skipped,
        "cases": [row.to_dict() for row in rows],
        "candidate_safe_accept_count": sum(row.candidate_safe_accept for row in rows),
        "candidate_false_safe_count": sum(row.candidate_false_safe for row in rows),
        "base_safe_accept_count": sum(row.base_safe_accept for row in rows),
        "base_false_safe_count": sum(row.base_false_safe for row in rows),
        "candidate_safe_acceptance_coverage": (
            sum(row.candidate_safe_accept for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "base_safe_acceptance_coverage": (
            sum(row.base_safe_accept for row in rows) / len(rows) if rows else 0.0
        ),
        "candidate_mean_fscore": candidate_mean,
        "b5_available_case_count": len(available_b5_rows),
        "b5_construction_failure_count": len(rows) - len(available_b5_rows),
        "b5_mean_fscore": b5_mean,
        "m1_available_case_count": len(available_m1),
        "m1_mean_fscore": m1_mean,
        "candidate_b5_mean_fscore_margin": (
            None
            if not available_b5_rows
            else float(
                np.mean(
                    [
                        row.candidate.fscore - row.b5.fscore
                        for row in available_b5_rows
                    ]
                )
            )
        ),
        "candidate_m1_mean_fscore_margin": (
            None
            if len(available_m1) != len(rows) or not rows
            else float(
                np.mean(
                    [
                        row.candidate.fscore - row.m1.fscore
                        for row in rows
                        if row.m1 is not None
                    ]
                )
            )
        ),
        "candidate_mean_geometry_loss": candidate_geometry,
        "b5_mean_geometry_loss": b5_geometry,
        "m1_mean_geometry_loss": m1_geometry,
        "candidate_b5_casewise_win_count": sum(
            row.candidate_fscore_wins_b5 is True for row in rows
        ),
        "candidate_b5_casewise_win_rate": (
            sum(row.candidate_fscore_wins_b5 is True for row in rows)
            / len(available_b5_rows)
            if available_b5_rows
            else None
        ),
        "candidate_m1_casewise_win_count": sum(
            row.candidate_fscore_wins_m1 is True for row in rows
        ),
        "candidate_m1_casewise_win_rate": (
            sum(row.candidate_fscore_wins_m1 is True for row in rows)
            / len(available_m1)
            if available_m1
            else None
        ),
        "candidate_topology_error_sum": sum(
            row.candidate.topology_error for row in rows
        ),
        "b5_topology_error_sum": sum(
            row.b5.topology_error for row in available_b5_rows
        ),
        "m1_topology_error_sum": (
            sum(metrics.topology_error for metrics in available_m1)
            if len(available_m1) == len(rows)
            else None
        ),
        "reserved_content_opened": False,
        "eligibility_frozen": False,
        "real_scan_supported": False,
        "held_out_validation_supported": False,
        "claim_boundary": (
            "calibration-only floor--ceiling results may freeze a later external "
            "test but do not support Area-5 or close-layer transfer"
        ),
    }


def write_result(payload: dict[str, object], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark-out/s3dis_room_layer_calibration_benchmark_phase51b.json"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = evaluate_room_layer_calibration_benchmark(args.calibration_artifact)
    digest = write_result(payload, args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(
        f"eligible={payload['eligible_pair_count']} "
        f"evaluated={payload['evaluated_case_count']} "
        f"safe_accept={payload['candidate_safe_accept_count']} "
        f"false_safe={payload['candidate_false_safe_count']}"
    )


if __name__ == "__main__":
    main()
