"""Phase-51C final protocol for untouched S3DIS Area-5 room layers."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL_SCHEMA = "pftf_alpha_s3dis_room_layer_validation_protocol_phase51c/v1"
CALIBRATION_COMMIT = "30e3ddf038a539d1d4830efe472cba39651d2d6d"
CALIBRATION_ARTIFACT = (
    "benchmark-out/s3dis_room_layer_calibration_benchmark_phase51b.json"
)
CALIBRATION_SHA256 = (
    "3dccb4c48cb97b73c432eca31f1d3d7f833d7df70895ab540dfd752b0f71798f"
)
HELD_OUT_AREA = "Area_5"
MAXIMUM_ANGLE_DEGREES = 5.0
MINIMUM_BBOX_OVERLAP = 0.75
MINIMUM_COMMON_POINTS_PER_LAYER = 592
MINIMUM_GAP_TO_RESIDUAL_SNR = 10.0
MAXIMUM_PLANE_RESIDUAL_TO_SPACING = 10.0
OBSERVED_PER_LAYER = 80
REFERENCE_PER_LAYER = 512
SURFACE_SAMPLE_COUNT = 512
GENERAL_POSITION_JOGGLE = 1.0e-4
MINIMUM_ELIGIBLE_CASES = 20
MINIMUM_SAFE_ACCEPTANCE_COVERAGE = 0.90
MINIMUM_B5_AVAILABILITY = 0.90
MINIMUM_M1_AVAILABILITY = 0.95
MINIMUM_B5_FSCORE_MARGIN = 0.20
MINIMUM_M1_FSCORE_MARGIN = 0.30
MINIMUM_CASEWISE_WIN_RATE = 0.85
MAXIMUM_TOPOLOGY_ERROR_RATIO = 0.25
MAXIMUM_BASE_COVERAGE_DEFICIT = 0.02
MAXIMUM_BASE_FSCORE_DEFICIT = 0.01


@dataclass(frozen=True)
class S3DISRoomLayerValidationProtocol:
    artifact_schema: str
    role: str
    calibration_commit: str
    calibration_artifact: str
    calibration_sha256: str
    held_out_area: str
    held_out_status_at_preregistration: str
    case_enumeration_rule: str
    maximum_angle_degrees: float
    minimum_bbox_overlap: float
    minimum_common_points_per_layer: int
    minimum_gap_to_residual_snr: float
    maximum_plane_residual_to_spacing: float
    observed_per_layer: int
    reference_per_layer: int
    surface_sample_count: int
    coordinate_split_rule: str
    common_footprint_rule: str
    normalization_rule: str
    general_position_joggle: float
    general_position_joggle_rule: str
    information_boundary: str
    candidate_method: str
    base_ablation: str
    b5_method: str
    m1_method: str
    minimum_eligible_cases: int
    minimum_safe_acceptance_coverage: float
    minimum_b5_availability: float
    minimum_m1_availability: float
    minimum_b5_fscore_margin: float
    minimum_m1_fscore_margin: float
    minimum_casewise_win_rate: float
    maximum_topology_error_ratio: float
    maximum_base_coverage_deficit: float
    maximum_base_fscore_deficit: float
    panel_gate: str
    safety_gate: str
    construction_gate: str
    geometry_gate: str
    topology_gate: str
    ablation_gate: str
    phase_gate: str
    success_flags: tuple[str, ...]
    always_false_flags: tuple[str, ...]
    excluded_scope: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "success_flags",
            "always_false_flags",
            "excluded_scope",
        ):
            payload[name] = list(payload[name])
        return payload


def preregister_s3dis_room_layer_validation() -> S3DISRoomLayerValidationProtocol:
    return S3DISRoomLayerValidationProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="untouched_building_disjoint_real_floor_ceiling_validation",
        calibration_commit=CALIBRATION_COMMIT,
        calibration_artifact=CALIBRATION_ARTIFACT,
        calibration_sha256=CALIBRATION_SHA256,
        held_out_area=HELD_OUT_AREA,
        held_out_status_at_preregistration=(
            "Area-5 ZIP member names and sizes have been counted, but no Area-5 "
            "point-cloud content has been extracted, opened, parsed, summarized, "
            "visualized, or evaluated"
        ),
        case_enumeration_rule=(
            "sort every Area-5 Annotations directory lexicographically; merge all "
            "floor*.txt members and all ceiling*.txt members within one room; "
            "apply the frozen eligibility rules automatically; do not manually "
            "include, exclude, or relabel a room"
        ),
        maximum_angle_degrees=MAXIMUM_ANGLE_DEGREES,
        minimum_bbox_overlap=MINIMUM_BBOX_OVERLAP,
        minimum_common_points_per_layer=MINIMUM_COMMON_POINTS_PER_LAYER,
        minimum_gap_to_residual_snr=MINIMUM_GAP_TO_RESIDUAL_SNR,
        maximum_plane_residual_to_spacing=MAXIMUM_PLANE_RESIDUAL_TO_SPACING,
        observed_per_layer=OBSERVED_PER_LAYER,
        reference_per_layer=REFERENCE_PER_LAYER,
        surface_sample_count=SURFACE_SAMPLE_COUNT,
        coordinate_split_rule=(
            "use the frozen float64-bit coordinate order with salts 51200001 for "
            "floor and 51200003 for ceiling; take the first 80 unique points as "
            "observed and up to the next 512 as disjoint reference"
        ),
        common_footprint_rule=(
            "fit the floor tangent frame, intersect floor and ceiling projected "
            "axis-aligned bounding boxes, and retain points inside the exact "
            "intersection without endpoint-dependent cropping"
        ),
        normalization_rule=(
            "center at the common projected-footprint midpoint on the floor plane "
            "and divide XYZ by the common-footprint diagonal"
        ),
        general_position_joggle=GENERAL_POSITION_JOGGLE,
        general_position_joggle_rule=(
            "derive one RNG seed from SHA-256(area/room), add deterministic "
            "Gaussian scale 1e-4 to observed normalized XYZ for candidate, base, "
            "B5, and M1 equally, and leave references unchanged"
        ),
        information_boundary=(
            "semantic class and instance files may construct and score the frozen "
            "corpus, but candidate routing receives only combined observed XYZ; "
            "true labels, references, baselines, and endpoints are evaluation-only"
        ),
        candidate_method=(
            "unchanged Phase-50 shared quadratic trend, residual two-means, "
            "observed-only sampling gate, and one 2D Delaunay surface per layer"
        ),
        base_ablation=(
            "unchanged global-normal parallel-layer inference with the same gate "
            "and per-layer construction"
        ),
        b5_method=(
            "frozen Phase-50 PCA-anisotropic alpha with multiplier "
            "2.80293354289327 and maximum normal penalty 4; construction errors "
            "are recorded as unavailable, never used to delete a case"
        ),
        m1_method=(
            "frozen Phase-50 weighted power-alpha with weight scale 0.375 and "
            "multiplier 2.5009326930224836; point submersion is unavailable"
        ),
        minimum_eligible_cases=MINIMUM_ELIGIBLE_CASES,
        minimum_safe_acceptance_coverage=MINIMUM_SAFE_ACCEPTANCE_COVERAGE,
        minimum_b5_availability=MINIMUM_B5_AVAILABILITY,
        minimum_m1_availability=MINIMUM_M1_AVAILABILITY,
        minimum_b5_fscore_margin=MINIMUM_B5_FSCORE_MARGIN,
        minimum_m1_fscore_margin=MINIMUM_M1_FSCORE_MARGIN,
        minimum_casewise_win_rate=MINIMUM_CASEWISE_WIN_RATE,
        maximum_topology_error_ratio=MAXIMUM_TOPOLOGY_ERROR_RATIO,
        maximum_base_coverage_deficit=MAXIMUM_BASE_COVERAGE_DEFICIT,
        maximum_base_fscore_deficit=MAXIMUM_BASE_FSCORE_DEFICIT,
        panel_gate="at least 20 Area-5 rooms satisfy every frozen eligibility rule",
        safety_gate=(
            "candidate false-safe accept count is zero and candidate safe-accept "
            "coverage is at least 0.90"
        ),
        construction_gate=(
            "B5 is available on at least 0.90 and M1 on at least 0.95 of every "
            "eligible case; unavailable cases remain in panel and safety counts"
        ),
        geometry_gate=(
            "on comparator-available paired cases, candidate mean F-score margin "
            "is at least +0.20 versus B5 and +0.30 versus M1, candidate casewise "
            "win rate is at least 0.85 against each, and candidate mean geometry "
            "loss is strictly lower than each comparator"
        ),
        topology_gate=(
            "on comparator-available paired cases, candidate topology-error sum "
            "is at most 0.25 of the comparator topology-error sum for both B5 and "
            "M1; a zero comparator denominator fails the corresponding gate"
        ),
        ablation_gate=(
            "candidate safe-accept coverage is no more than 0.02 below the "
            "global-normal base and candidate mean F-score is no more than 0.01 "
            "below the base; this is non-inferiority, not shared-trend superiority"
        ),
        phase_gate=(
            "protocol identity plus panel, safety, construction, geometry, "
            "topology, and ablation gates all pass on the once-opened panel"
        ),
        success_flags=(
            "phase51c_supported",
            "real_long_gap_two_layer_supported",
            "real_scan_supported",
            "held_out_validation_supported",
        ),
        always_false_flags=(
            "pftf_superiority_supported",
            "local_spd_superiority_supported",
            "shared_trend_superiority_supported",
            "close_layer_transfer_supported",
            "deployment_supported",
        ),
        excluded_scope=(
            "wall--board close layers",
            "semantic scene discovery",
            "annotation-free corpus extraction",
            "arbitrary surface families",
            "PFTF or local-SPD superiority",
            "shared-trend superiority",
            "deployment",
        ),
        claim_boundary=(
            "a positive result supports only real, building-disjoint, long-gap, "
            "approximately parallel floor--ceiling reconstruction with "
            "annotation-defined corpus extraction and observed-XYZ-only routing"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_s3dis_room_layer_validation().to_dict(),
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
            "benchmark-out/s3dis_room_layer_validation_protocol_phase51c.json"
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
