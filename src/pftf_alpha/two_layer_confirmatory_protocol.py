"""Phase-50 preregistration for bounded two-layer reconstruction efficacy."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .sensor_stress import SensorStress

PROTOCOL_SCHEMA = "pftf_alpha_two_layer_confirmatory_protocol_phase50/v1"
HELD_OUT_SEED = 35_000_804
POINT_COUNTS = (160, 256)
STRESSES = (
    SensorStress.CONTROL.value,
    SensorStress.UPPER_OCCLUSION.value,
    SensorStress.IMBALANCED_75_25.value,
    SensorStress.ANISOTROPIC_NOISE.value,
    SensorStress.SINUSOIDAL.value,
    SensorStress.LOCAL_BUMP.value,
)
REPEATS = 12
REFERENCE_COUNT = 4096
SURFACE_SAMPLE_COUNT = 1024
K_NEIGHBORS = 12
FSCORE_THRESHOLD_FRACTION = 0.025
B5_SCALE_MULTIPLIER = 2.80293354289327
B5_MAX_NORMAL_PENALTY = 4.0
M1_WEIGHT_SCALE = 0.375
M1_SCALE_MULTIPLIER = 2.5009326930224836
M1_CALIBRATION_SHA256 = (
    "78831ffaf2a43409fbc17ef4e79447041eb8c946a9bb48e465626fa64e799c66"
)
MINIMUM_OVERALL_SAFE_ACCEPTANCE = 0.95
MINIMUM_SUBGROUP_SAFE_ACCEPTANCE = 0.90
MINIMUM_MEAN_FSCORE_MARGIN = 0.10
MINIMUM_CASEWISE_FSCORE_WIN_RATE = 0.75
MINIMUM_REPAIRED_BASE_FALSE_SAFE = 1


@dataclass(frozen=True)
class TwoLayerConfirmatoryProtocol:
    artifact_schema: str
    role: str
    source_candidate: str
    held_out_seed: int
    point_counts: tuple[int, ...]
    stresses: tuple[str, ...]
    repeats: int
    expected_case_count: int
    reference_count: int
    surface_sample_count: int
    k_neighbors: int
    fscore_threshold_fraction: float
    case_seed_rule: str
    pose_rule: str
    information_boundary: str
    held_out_prohibition: str
    candidate_method: str
    candidate_gate: str
    base_ablation: str
    b5_method: str
    b5_scale_multiplier: float
    b5_max_normal_penalty: float
    m1_method: str
    m1_weight_scale: float
    m1_scale_multiplier: float
    m1_calibration_artifact: str
    m1_calibration_sha256: str
    geometry_loss: str
    topology_error: str
    minimum_overall_safe_acceptance: float
    minimum_subgroup_safe_acceptance: float
    minimum_mean_fscore_margin: float
    minimum_casewise_fscore_win_rate: float
    minimum_repaired_base_false_safe: int
    safety_gate: str
    efficacy_gate: str
    topology_gate: str
    ablation_gate: str
    phase_gate: str
    excluded_scope: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in ("point_counts", "stresses", "excluded_scope"):
            payload[name] = list(payload[name])
        return payload


def preregister_two_layer_confirmatory() -> TwoLayerConfirmatoryProtocol:
    expected_case_count = len(POINT_COUNTS) * len(STRESSES) * REPEATS
    return TwoLayerConfirmatoryProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="untouched_bounded_positive_two_layer_confirmatory_test",
        source_candidate="frozen Phase-7 shared-trend residual layer inference",
        held_out_seed=HELD_OUT_SEED,
        point_counts=POINT_COUNTS,
        stresses=STRESSES,
        repeats=REPEATS,
        expected_case_count=expected_case_count,
        reference_count=REFERENCE_COUNT,
        surface_sample_count=SURFACE_SAMPLE_COUNT,
        k_neighbors=K_NEIGHBORS,
        fscore_threshold_fraction=FSCORE_THRESHOLD_FRACTION,
        case_seed_rule=(
            "held_out_seed + point_count_index*1000003 + "
            "stress_index*100003 + repeat*10007"
        ),
        pose_rule=(
            "derive one 3x3 Gaussian matrix from case_seed+7000001, use QR, "
            "make the diagonal sign deterministic, correct determinant to +1, "
            "and apply the same proper rotation to observed and reference points"
        ),
        information_boundary=(
            "candidate routing uses observed coordinates only; stress identity, "
            "true layer labels, clean reference points, and baseline endpoints are "
            "evaluation-only"
        ),
        held_out_prohibition=(
            "no Phase-50 point, label, reference, endpoint, or aggregate may alter "
            "the candidate, baseline parameters, thresholds, case set, or gates"
        ),
        candidate_method=(
            "unchanged shared quadratic trend fit, residual two-means, observed-only "
            "sampling gate, and one 2D Delaunay surface per inferred layer"
        ),
        candidate_gate=(
            "unchanged Phase-7 SharedTrendConfig with k=12, minimum cluster "
            "fraction 0.20, separation SNR 3.0, and cross-kNN fraction 0.05"
        ),
        base_ablation=(
            "unchanged global-normal parallel-layer inference and per-layer 2D "
            "Delaunay construction with the same observed-only gate"
        ),
        b5_method=(
            "PCA-anisotropic Delaunay filtration with frozen Phase-0 local-scale "
            "multiplier and maximum normal penalty"
        ),
        b5_scale_multiplier=B5_SCALE_MULTIPLIER,
        b5_max_normal_penalty=B5_MAX_NORMAL_PENALTY,
        m1_method=(
            "regular weighted-Delaunay power-alpha filtration at the independently "
            "calibrated B4-dominating design point"
        ),
        m1_weight_scale=M1_WEIGHT_SCALE,
        m1_scale_multiplier=M1_SCALE_MULTIPLIER,
        m1_calibration_artifact="benchmark-out/m1_weighted_alpha_ablation.json",
        m1_calibration_sha256=M1_CALIBRATION_SHA256,
        geometry_loss="normalized_chamfer_squared + normalized_hausdorff",
        topology_error=(
            "component_error + betti_error + labeled_false_bridge_edges + "
            "labeled_false_bridge_faces"
        ),
        minimum_overall_safe_acceptance=MINIMUM_OVERALL_SAFE_ACCEPTANCE,
        minimum_subgroup_safe_acceptance=MINIMUM_SUBGROUP_SAFE_ACCEPTANCE,
        minimum_mean_fscore_margin=MINIMUM_MEAN_FSCORE_MARGIN,
        minimum_casewise_fscore_win_rate=MINIMUM_CASEWISE_FSCORE_WIN_RATE,
        minimum_repaired_base_false_safe=MINIMUM_REPAIRED_BASE_FALSE_SAFE,
        safety_gate=(
            "candidate false-safe count is zero; overall safe-acceptance coverage "
            "is at least 0.95; every point-count x stress subgroup is at least 0.90"
        ),
        efficacy_gate=(
            "over all 144 cases, candidate mean F-score exceeds both B5 and M1 by "
            "at least 0.10, candidate mean geometry loss is strictly lower than "
            "both, and candidate wins F-score casewise against each on at least "
            "75 percent of cases"
        ),
        topology_gate=(
            "candidate aggregate topology error and nonmanifold-edge count are "
            "zero, while each of B5 and M1 has strictly positive aggregate "
            "topology error"
        ),
        ablation_gate=(
            "candidate safe accepts are no fewer than the global-normal base and "
            "at least one base false-safe is repaired to a candidate safe accept"
        ),
        phase_gate=(
            "all configuration identity checks plus safety, efficacy, topology, "
            "and ablation gates must pass without retuning"
        ),
        excluded_scope=(
            "spatial outliers",
            "point counts below 160",
            "intersecting or non-separable surfaces",
            "arbitrary surface families",
            "real scans",
            "PFTF or local-SPD superiority",
            "exact predicates",
            "deployment",
        ),
        claim_boundary=(
            "a positive result supports only a sampling-sufficient, globally "
            "separable, non-outlier synthetic two-layer reconstruction method; it "
            "does not support outlier robustness, general alpha selection, PFTF "
            "conditioning, real transfer, exactness, or deployment"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_two_layer_confirmatory().to_dict(),
        indent=2,
        sort_keys=True,
    ) + "\n"
    output.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark-out/two_layer_confirmatory_protocol_phase50.json"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_protocol(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
