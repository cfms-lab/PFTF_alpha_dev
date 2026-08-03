"""Phase-51B preregistration for S3DIS floor--ceiling calibration intake."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .s3dis_two_layer_intake_protocol import (
    ARCHIVE_NAME,
    CALIBRATION_AREAS,
    DATASET_NAME,
    DATASET_URL,
    RESERVED_HELD_OUT_AREA,
)

PROTOCOL_SCHEMA = "pftf_alpha_s3dis_room_layer_protocol_phase51b/v1"
TARGET_CLASSES = ("floor", "ceiling")
PHASE51A_CALIBRATION_SHA256 = (
    "069004f9c98f772c266bba1820207b86a45ca5be922cbbe1a6175f84fb627a0e"
)
PHASE51A_BENCHMARK_SHA256 = (
    "d3edfa629681dc70294b6bf4068b27ef6d0e49dc93bdd38bf5b7266c713e082f"
)


@dataclass(frozen=True)
class S3DISRoomLayerProtocol:
    artifact_schema: str
    role: str
    dataset_name: str
    dataset_url: str
    archive_name: str
    predecessor_calibration_artifact: str
    predecessor_calibration_sha256: str
    predecessor_benchmark_artifact: str
    predecessor_benchmark_sha256: str
    pivot_reason: str
    calibration_areas: tuple[str, ...]
    reserved_held_out_area: str
    target_classes: tuple[str, ...]
    calibration_content_allowed: str
    reserved_content_prohibition: str
    pair_definition_development: str
    candidate_method: str
    comparator_methods: tuple[str, ...]
    runtime_information_boundary: str
    evaluation_only_information: tuple[str, ...]
    regime_boundary: str
    final_preregistration_requirement: str
    current_support_flags: dict[str, bool]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "calibration_areas",
            "target_classes",
            "comparator_methods",
            "evaluation_only_information",
        ):
            payload[name] = list(payload[name])
        return payload


def preregister_s3dis_room_layer() -> S3DISRoomLayerProtocol:
    return S3DISRoomLayerProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="external_real_floor_ceiling_calibration_with_reserved_holdout",
        dataset_name=DATASET_NAME,
        dataset_url=DATASET_URL,
        archive_name=ARCHIVE_NAME,
        predecessor_calibration_artifact=(
            "benchmark-out/s3dis_two_layer_calibration_phase51.json"
        ),
        predecessor_calibration_sha256=PHASE51A_CALIBRATION_SHA256,
        predecessor_benchmark_artifact=(
            "benchmark-out/s3dis_two_layer_calibration_benchmark_phase51.json"
        ),
        predecessor_benchmark_sha256=PHASE51A_BENCHMARK_SHA256,
        pivot_reason=(
            "Phase-51A wall--board calibration had positive geometry margins "
            "but zero topology-safe accepts because opaque-board occlusion left "
            "non-overlapping wall support; floor and ceiling provide genuinely "
            "observed overlapping footprints"
        ),
        calibration_areas=CALIBRATION_AREAS,
        reserved_held_out_area=RESERVED_HELD_OUT_AREA,
        target_classes=TARGET_CLASSES,
        calibration_content_allowed=(
            "floor and ceiling annotation contents in Areas 1, 2, 3, 4, and 6 "
            "may be opened to develop room pairing, common-footprint crop, "
            "plane-quality, overlap, sampling, and deterministic split rules"
        ),
        reserved_content_prohibition=(
            "before a separate final floor--ceiling protocol is committed, do "
            "not extract, open, parse, summarize, visualize, or compute any "
            "statistic from an Area-5 point-cloud member"
        ),
        pair_definition_development=(
            "merge same-class annotation fragments within one calibration room, "
            "fit one robust plane per class, project to a shared tangent frame, "
            "and develop deterministic common-footprint and eligibility rules"
        ),
        candidate_method=(
            "unchanged Phase-50 shared quadratic trend fit, residual two-means, "
            "observed-only sampling gate, and one 2D Delaunay surface per layer"
        ),
        comparator_methods=(
            "frozen B5 PCA-anisotropic alpha",
            "frozen M1 weighted power-alpha",
            "global-normal two-layer ablation",
        ),
        runtime_information_boundary=(
            "candidate reconstruction and routing receive only combined observed "
            "XYZ coordinates; class, room, area, RGB, instance, and reference "
            "information cannot influence output"
        ),
        evaluation_only_information=(
            "floor versus ceiling source label",
            "held-out reference points",
            "false cross-layer edges and faces",
            "baseline endpoints",
        ),
        regime_boundary=(
            "floor--ceiling is an easier long-gap real two-layer regime; a "
            "positive result cannot be generalized to close wall--board layers, "
            "arbitrary surfaces, or scene-wide reconstruction"
        ),
        final_preregistration_requirement=(
            "after calibration-only evaluation, commit exact plane, overlap, "
            "gap, point-count, crop, coordinate-hash split, metrics, gates, and "
            "Area-5 case enumeration before opening Area-5 contents"
        ),
        current_support_flags={
            "floor_ceiling_calibration_supported": False,
            "real_scan_supported": False,
            "held_out_validation_supported": False,
            "pftf_superiority_supported": False,
            "deployment_supported": False,
        },
        claim_boundary=(
            "this protocol permits leakage-controlled calibration on real "
            "floor--ceiling pairs but is not itself a real held-out result"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_s3dis_room_layer().to_dict(),
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
        default=Path("benchmark-out/s3dis_room_layer_protocol_phase51b.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_protocol(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
