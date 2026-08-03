"""Develop Phase-51 wall--board eligibility on calibration areas only."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .adaptive import pca_anisotropic_filtration
from .filtration import AlphaFiltration
from .s3dis_two_layer_calibration import load_xyz
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .shared_trend_inference import SharedTrendConfig, construct_shared_trend_surface
from .surface import SurfaceEndpointMetrics, evaluate_surface
from .two_layer_confirmatory import ConfirmatoryMethodMetrics
from .two_layer_confirmatory_protocol import (
    B5_MAX_NORMAL_PENALTY,
    B5_SCALE_MULTIPLIER,
    FSCORE_THRESHOLD_FRACTION,
    K_NEIGHBORS,
    M1_SCALE_MULTIPLIER,
    M1_WEIGHT_SCALE,
)
from .two_layer_connectivity import (
    construct_two_layer_surface,
    route_two_layer_output,
)
from .weighted_alpha import PointSubmersionError, weighted_alpha_filtration

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
RESULT_SCHEMA = "pftf_alpha_s3dis_two_layer_calibration_benchmark_phase51/v1"
DEFAULT_OBSERVED_PER_LAYER = 80
DEFAULT_REFERENCE_PER_LAYER = 512
DEFAULT_SURFACE_SAMPLE_COUNT = 512


@dataclass(frozen=True)
class RealTwoLayerCase:
    name: str
    points: FloatArray
    reference_points: FloatArray
    true_labels: IntArray
    characteristic_length: float


@dataclass(frozen=True)
class CalibrationBenchmarkCase:
    name: str
    area: str
    room: str
    support: float
    gap_to_spacing: float
    normal_angle_degrees: float
    candidate_decision: SamplingGateDecision
    candidate_true_safe: bool
    candidate_safe_accept: bool
    candidate_false_safe: bool
    candidate: ConfirmatoryMethodMetrics
    base: ConfirmatoryMethodMetrics
    b5: ConfirmatoryMethodMetrics
    m1: ConfirmatoryMethodMetrics | None
    candidate_fscore_wins_b5: bool
    candidate_fscore_wins_m1: bool | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_decision"] = self.candidate_decision.value
        return payload


def _coordinate_order(points: FloatArray, salt: int) -> IntArray:
    bits = np.ascontiguousarray(points, dtype=np.float64).view(np.uint64)
    bits = bits.reshape(points.shape[0], 3)
    scores = (
        bits[:, 0] * np.uint64(0x9E3779B185EBCA87)
        ^ bits[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ bits[:, 2] * np.uint64(0x165667B19E3779F9)
        ^ np.uint64(salt)
    )
    return np.lexsort((np.arange(points.shape[0], dtype=np.int64), scores))


def _split_layer(
    points: FloatArray,
    observed_count: int,
    reference_count: int,
    *,
    salt: int,
) -> tuple[FloatArray, FloatArray]:
    required = 2 * observed_count
    if points.shape[0] < required:
        raise ValueError(
            f"layer has {points.shape[0]} points; need at least {required}"
        )
    order = _coordinate_order(points, salt)
    observed = points[order[:observed_count]]
    remaining = order[observed_count:]
    selected_reference = remaining[: min(reference_count, remaining.size)]
    return np.ascontiguousarray(observed), np.ascontiguousarray(
        points[selected_reference]
    )


def build_case_from_pair(
    calibration_root: str | Path,
    pair: dict[str, object],
    *,
    observed_per_layer: int = DEFAULT_OBSERVED_PER_LAYER,
    reference_per_layer: int = DEFAULT_REFERENCE_PER_LAYER,
    crop_margin_fraction: float = 0.10,
) -> RealTwoLayerCase:
    root = Path(calibration_root)
    board_path = root / str(pair["board_path"])
    wall_path = root / str(pair["selected_wall_path"])
    board_points = load_xyz(board_path)
    wall_points = load_xyz(wall_path)
    board_summary = pair["board"]
    if not isinstance(board_summary, dict):
        raise ValueError("calibration pair has no board summary")
    center = np.asarray(board_summary["centroid"], dtype=np.float64)
    tangent_u = np.asarray(board_summary["tangent_u"], dtype=np.float64)
    tangent_v = np.asarray(board_summary["tangent_v"], dtype=np.float64)
    board_uv = np.column_stack(
        ((board_points - center) @ tangent_u, (board_points - center) @ tangent_v)
    )
    wall_uv = np.column_stack(
        ((wall_points - center) @ tangent_u, (wall_points - center) @ tangent_v)
    )
    lower = np.min(board_uv, axis=0)
    upper = np.max(board_uv, axis=0)
    margin = crop_margin_fraction * np.maximum(upper - lower, np.finfo(float).eps)
    selected_wall = np.all(
        (wall_uv >= lower - margin) & (wall_uv <= upper + margin), axis=1
    )
    unique_board = np.unique(board_points, axis=0)
    cropped_wall = np.unique(wall_points[selected_wall], axis=0)
    combined = np.vstack((unique_board, cropped_wall))
    _, inverse, counts = np.unique(
        combined,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    board_count = unique_board.shape[0]
    unique_board = unique_board[counts[inverse[:board_count]] == 1]
    cropped_wall = cropped_wall[counts[inverse[board_count:]] == 1]
    diagonal = float(np.linalg.norm(upper - lower))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("board footprint is degenerate")
    normalized_board = (unique_board - center) / diagonal
    normalized_wall = (cropped_wall - center) / diagonal
    board_observed, board_reference = _split_layer(
        normalized_board,
        observed_per_layer,
        reference_per_layer,
        salt=51_000_001,
    )
    wall_observed, wall_reference = _split_layer(
        normalized_wall,
        observed_per_layer,
        reference_per_layer,
        salt=51_000_003,
    )
    points = np.vstack((board_observed, wall_observed))
    reference = np.vstack((board_reference, wall_reference))
    labels = np.concatenate(
        (
            np.zeros(observed_per_layer, dtype=np.int64),
            np.ones(observed_per_layer, dtype=np.int64),
        )
    )
    return RealTwoLayerCase(
        name=f"{pair['area']}/{pair['room']}/{Path(str(pair['board_path'])).name}",
        points=np.ascontiguousarray(points),
        reference_points=np.ascontiguousarray(reference),
        true_labels=labels,
        characteristic_length=1.0,
    )


def _evaluate(
    mesh,
    case: RealTwoLayerCase,
    labels: IntArray,
    *,
    sample_count: int,
    seed: int,
) -> SurfaceEndpointMetrics:
    return evaluate_surface(
        mesh,
        case.reference_points,
        expected_components=2,
        expected_betti=(2, 0, 0),
        vertex_component_labels=labels,
        characteristic_length=case.characteristic_length,
        sample_count=sample_count,
        threshold_fraction=FSCORE_THRESHOLD_FRACTION,
        seed=seed,
    )


def _true_safe(endpoints: SurfaceEndpointMetrics) -> bool:
    return bool(
        endpoints.component_error == 0
        and int(endpoints.labeled_false_bridge_edges or 0) == 0
        and int(endpoints.labeled_false_bridge_faces or 0) == 0
    )


def _eligible(
    pair: dict[str, object],
    *,
    minimum_support: float,
    minimum_wall_points: int,
    minimum_gap_to_spacing: float,
    minimum_gap_to_residual_snr: float,
    maximum_angle_degrees: float,
    maximum_plane_residual_to_spacing: float,
) -> bool:
    selected = pair.get("selected_pair")
    board = pair.get("board")
    wall = pair.get("selected_wall")
    if (
        not isinstance(selected, dict)
        or not isinstance(board, dict)
        or not isinstance(wall, dict)
    ):
        return False
    board_ratio = float(board["residual_p95"]) / float(board["median_spacing"])
    wall_ratio = float(wall["residual_p95"]) / float(wall["median_spacing"])
    residual_scale = np.hypot(
        float(board["residual_rms"]),
        float(wall["residual_rms"]),
    )
    gap_to_residual_snr = float(selected["plane_gap"]) / max(
        residual_scale, np.finfo(float).eps
    )
    return bool(
        float(selected["board_footprint_grid_support"]) >= minimum_support
        and int(selected["wall_points_in_board_footprint"]) >= minimum_wall_points
        and float(selected["gap_to_median_spacing"]) >= minimum_gap_to_spacing
        and gap_to_residual_snr >= minimum_gap_to_residual_snr
        and float(selected["normal_angle_degrees"]) <= maximum_angle_degrees
        and board_ratio <= maximum_plane_residual_to_spacing
        and wall_ratio <= maximum_plane_residual_to_spacing
    )


def evaluate_calibration_benchmark(
    calibration_artifact: str | Path,
    *,
    minimum_support: float = 0.20,
    minimum_wall_points: int = 320,
    minimum_gap_to_spacing: float = 0.50,
    minimum_gap_to_residual_snr: float = 0.0,
    maximum_angle_degrees: float = 15.0,
    maximum_plane_residual_to_spacing: float = 2.0,
    observed_per_layer: int = DEFAULT_OBSERVED_PER_LAYER,
    reference_per_layer: int = DEFAULT_REFERENCE_PER_LAYER,
    surface_sample_count: int = DEFAULT_SURFACE_SAMPLE_COUNT,
) -> dict[str, object]:
    artifact_path = Path(calibration_artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    calibration_root = str(payload["calibration_root"])
    selected_pairs = [
        pair
        for pair in payload["pairs"]
        if _eligible(
            pair,
            minimum_support=minimum_support,
            minimum_wall_points=minimum_wall_points,
            minimum_gap_to_spacing=minimum_gap_to_spacing,
            minimum_gap_to_residual_snr=minimum_gap_to_residual_snr,
            maximum_angle_degrees=maximum_angle_degrees,
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
    rows: list[CalibrationBenchmarkCase] = []
    skipped: list[dict[str, str]] = []
    for index, pair in enumerate(selected_pairs):
        try:
            case = build_case_from_pair(
                calibration_root,
                pair,
                observed_per_layer=observed_per_layer,
                reference_per_layer=reference_per_layer,
            )
        except ValueError as error:
            skipped.append(
                {"board_path": str(pair["board_path"]), "reason": str(error)}
            )
            continue
        evaluation_seed = 51_100_000 + index
        base = construct_two_layer_surface(case.points, base_config)
        candidate, _ = construct_shared_trend_surface(case.points, candidate_config)
        candidate_inferred = _evaluate(
            candidate.mesh,
            case,
            candidate.inference.layer_ids,
            sample_count=surface_sample_count,
            seed=evaluation_seed,
        )
        candidate_truth = _evaluate(
            candidate.mesh,
            case,
            case.true_labels,
            sample_count=surface_sample_count,
            seed=evaluation_seed,
        )
        base_truth = _evaluate(
            base.mesh,
            case,
            case.true_labels,
            sample_count=surface_sample_count,
            seed=evaluation_seed,
        )
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
            seed=evaluation_seed,
        )
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
                seed=evaluation_seed,
            )
        except PointSubmersionError:
            m1_truth = None
        decision = route_two_layer_output(candidate, candidate_inferred)
        candidate_metrics = ConfirmatoryMethodMetrics.from_endpoints(candidate_truth)
        b5_metrics = ConfirmatoryMethodMetrics.from_endpoints(b5_truth)
        m1_metrics = (
            None
            if m1_truth is None
            else ConfirmatoryMethodMetrics.from_endpoints(m1_truth)
        )
        safe = _true_safe(candidate_truth)
        accept = decision is SamplingGateDecision.ACCEPT
        selected = pair["selected_pair"]
        rows.append(
            CalibrationBenchmarkCase(
                name=case.name,
                area=str(pair["area"]),
                room=str(pair["room"]),
                support=float(selected["board_footprint_grid_support"]),
                gap_to_spacing=float(selected["gap_to_median_spacing"]),
                normal_angle_degrees=float(selected["normal_angle_degrees"]),
                candidate_decision=decision,
                candidate_true_safe=safe,
                candidate_safe_accept=bool(accept and safe),
                candidate_false_safe=bool(accept and not safe),
                candidate=candidate_metrics,
                base=ConfirmatoryMethodMetrics.from_endpoints(base_truth),
                b5=b5_metrics,
                m1=m1_metrics,
                candidate_fscore_wins_b5=candidate_metrics.fscore > b5_metrics.fscore,
                candidate_fscore_wins_m1=(
                    None
                    if m1_metrics is None
                    else candidate_metrics.fscore > m1_metrics.fscore
                ),
            )
        )
    available_m1 = [row.m1 for row in rows if row.m1 is not None]
    candidate_mean = (
        float(np.mean([row.candidate.fscore for row in rows])) if rows else None
    )
    b5_mean = float(np.mean([row.b5.fscore for row in rows])) if rows else None
    m1_mean = (
        float(np.mean([metrics.fscore for metrics in available_m1]))
        if len(available_m1) == len(rows) and rows
        else None
    )
    return {
        "artifact_schema": RESULT_SCHEMA,
        "role": "calibration_only_real_wall_board_method_development",
        "calibration_artifact": str(artifact_path),
        "calibration_artifact_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
        "eligibility": {
            "minimum_support": minimum_support,
            "minimum_wall_points": minimum_wall_points,
            "minimum_gap_to_spacing": minimum_gap_to_spacing,
            "minimum_gap_to_residual_snr": minimum_gap_to_residual_snr,
            "maximum_angle_degrees": maximum_angle_degrees,
            "maximum_plane_residual_to_spacing": maximum_plane_residual_to_spacing,
        },
        "observation": {
            "observed_per_layer": observed_per_layer,
            "reference_per_layer": reference_per_layer,
            "surface_sample_count": surface_sample_count,
            "crop_margin_fraction": 0.10,
            "coordinate_hash_split": True,
        },
        "eligible_pair_count": len(selected_pairs),
        "evaluated_case_count": len(rows),
        "skipped": skipped,
        "cases": [row.to_dict() for row in rows],
        "candidate_safe_accept_count": sum(row.candidate_safe_accept for row in rows),
        "candidate_false_safe_count": sum(row.candidate_false_safe for row in rows),
        "candidate_mean_fscore": candidate_mean,
        "b5_mean_fscore": b5_mean,
        "m1_mean_fscore": m1_mean,
        "candidate_b5_mean_fscore_margin": (
            None
            if candidate_mean is None or b5_mean is None
            else candidate_mean - b5_mean
        ),
        "candidate_m1_mean_fscore_margin": (
            None
            if candidate_mean is None or m1_mean is None
            else candidate_mean - m1_mean
        ),
        "candidate_b5_casewise_win_count": sum(
            row.candidate_fscore_wins_b5 for row in rows
        ),
        "candidate_m1_casewise_win_count": sum(
            row.candidate_fscore_wins_m1 is True for row in rows
        ),
        "reserved_content_opened": False,
        "pair_eligibility_frozen": False,
        "real_scan_supported": False,
        "held_out_validation_supported": False,
        "claim_boundary": (
            "calibration-only development may select a final physical eligibility "
            "rule but cannot support a real held-out efficacy claim"
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
    parser.add_argument("--minimum-support", type=float, default=0.20)
    parser.add_argument("--minimum-wall-points", type=int, default=320)
    parser.add_argument("--minimum-gap-to-spacing", type=float, default=0.50)
    parser.add_argument("--minimum-gap-to-residual-snr", type=float, default=0.0)
    parser.add_argument("--maximum-angle-degrees", type=float, default=15.0)
    parser.add_argument(
        "--maximum-plane-residual-to-spacing", type=float, default=2.0
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/s3dis_two_layer_calibration_benchmark_phase51.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = evaluate_calibration_benchmark(
        args.calibration_artifact,
        minimum_support=args.minimum_support,
        minimum_wall_points=args.minimum_wall_points,
        minimum_gap_to_spacing=args.minimum_gap_to_spacing,
        minimum_gap_to_residual_snr=args.minimum_gap_to_residual_snr,
        maximum_angle_degrees=args.maximum_angle_degrees,
        maximum_plane_residual_to_spacing=args.maximum_plane_residual_to_spacing,
    )
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
