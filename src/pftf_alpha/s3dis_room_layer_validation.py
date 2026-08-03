"""Once-opened S3DIS Area-5 floor--ceiling validation and frozen gate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .s3dis_room_layer_calibration import (
    DEFAULT_MAX_FIT_POINTS,
    DEFAULT_MAX_SPACING_POINTS,
    RoomLayerPairCalibration,
    _room_pair,
)
from .s3dis_room_layer_calibration_benchmark import (
    evaluate_room_layer_calibration_benchmark,
)
from .s3dis_room_layer_calibration_benchmark import write_result as write_core_result
from .s3dis_room_layer_intake import _area, _parts, _room_layer
from .s3dis_room_layer_validation_protocol import (
    CALIBRATION_ARTIFACT,
    CALIBRATION_SHA256,
    GENERAL_POSITION_JOGGLE,
    HELD_OUT_AREA,
    MAXIMUM_ANGLE_DEGREES,
    MAXIMUM_BASE_COVERAGE_DEFICIT,
    MAXIMUM_BASE_FSCORE_DEFICIT,
    MAXIMUM_PLANE_RESIDUAL_TO_SPACING,
    MAXIMUM_TOPOLOGY_ERROR_RATIO,
    MINIMUM_B5_AVAILABILITY,
    MINIMUM_B5_FSCORE_MARGIN,
    MINIMUM_BBOX_OVERLAP,
    MINIMUM_CASEWISE_WIN_RATE,
    MINIMUM_COMMON_POINTS_PER_LAYER,
    MINIMUM_ELIGIBLE_CASES,
    MINIMUM_GAP_TO_RESIDUAL_SNR,
    MINIMUM_M1_AVAILABILITY,
    MINIMUM_M1_FSCORE_MARGIN,
    MINIMUM_SAFE_ACCEPTANCE_COVERAGE,
    OBSERVED_PER_LAYER,
    REFERENCE_PER_LAYER,
    SURFACE_SAMPLE_COUNT,
    preregister_s3dis_room_layer_validation,
)
from .s3dis_two_layer_intake_protocol import ARCHIVE_NAME

RESULT_SCHEMA = "pftf_alpha_s3dis_room_layer_validation_phase51c/v1"
GEOMETRY_SCHEMA = "pftf_alpha_s3dis_room_layer_area5_geometry_phase51c/v1"
PROTOCOL_COMMIT = "330b340595c79a83271f6309144f8840a99cd6a5"
PROTOCOL_SHA256 = (
    "fbbb56cad2dff5127bf51677e8b1f520b8870353de23dc927cce9db314485807"
)


@dataclass(frozen=True)
class HeldOutExtraction:
    archive_path: str
    extraction_root: str
    area: str
    extracted_member_count: int
    extracted_uncompressed_bytes: int
    floor_member_count: int
    ceiling_member_count: int
    held_out_artifacts_accessed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenGateAudit:
    eligible_case_count: int
    candidate_safe_acceptance_coverage: float
    candidate_false_safe_count: int
    b5_availability: float
    m1_availability: float
    candidate_b5_fscore_margin: float | None
    candidate_m1_fscore_margin: float | None
    candidate_b5_casewise_win_rate: float | None
    candidate_m1_casewise_win_rate: float | None
    candidate_b5_paired_geometry_loss: float | None
    b5_geometry_loss: float | None
    candidate_m1_paired_geometry_loss: float | None
    m1_geometry_loss: float | None
    candidate_b5_paired_topology_error: int | None
    b5_topology_error: int | None
    candidate_m1_paired_topology_error: int | None
    m1_topology_error: int | None
    candidate_b5_topology_error_ratio: float | None
    candidate_m1_topology_error_ratio: float | None
    base_safe_acceptance_coverage: float
    candidate_mean_fscore: float | None
    base_mean_fscore: float | None
    panel_gate_passed: bool
    safety_gate_passed: bool
    construction_gate_passed: bool
    geometry_gate_passed: bool
    topology_gate_passed: bool
    ablation_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_identity(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    expected = preregister_s3dis_room_layer_validation().to_dict()
    return digest, bool(digest == PROTOCOL_SHA256 and payload == expected)


def extract_heldout_room_layers(
    archive_path: str | Path,
    extraction_root: str | Path,
) -> HeldOutExtraction:
    archive = Path(archive_path)
    if archive.name != ARCHIVE_NAME:
        raise ValueError(f"expected archive name {ARCHIVE_NAME!r}")
    root = Path(extraction_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected: list[zipfile.ZipInfo] = []
    floor_count = 0
    ceiling_count = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            parts = _parts(info.filename)
            target = _room_layer(parts)
            if (
                _area(parts) == HELD_OUT_AREA
                and target is not None
                and not info.is_dir()
            ):
                selected.append(info)
                if target == "floor":
                    floor_count += 1
                else:
                    ceiling_count += 1
        extracted_bytes = 0
        for info in sorted(selected, key=lambda item: item.filename):
            parts = _parts(info.filename)
            if _area(parts) != HELD_OUT_AREA:
                raise AssertionError("non-held-out member reached Area-5 extraction")
            destination = root.joinpath(*parts).resolve()
            if root not in destination.parents:
                raise ValueError(f"unsafe extraction destination: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != info.file_size:
                    raise FileExistsError(
                        f"existing extraction has wrong size: {destination}"
                    )
            else:
                with bundle.open(info) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            extracted_bytes += info.file_size
    if not selected or floor_count == 0 or ceiling_count == 0:
        raise ValueError("Area 5 has no complete floor--ceiling target set")
    return HeldOutExtraction(
        archive_path=str(archive.resolve()),
        extraction_root=str(root),
        area=HELD_OUT_AREA,
        extracted_member_count=len(selected),
        extracted_uncompressed_bytes=extracted_bytes,
        floor_member_count=floor_count,
        ceiling_member_count=ceiling_count,
        held_out_artifacts_accessed=True,
    )


def build_heldout_geometry(root: str | Path) -> dict[str, object]:
    heldout_root = Path(root).resolve()
    annotation_paths = sorted(
        path for path in heldout_root.rglob("Annotations") if path.is_dir()
    )
    pairs: list[RoomLayerPairCalibration] = []
    missing_floor = 0
    missing_ceiling = 0
    for annotation_path in annotation_paths:
        floor_paths = sorted(annotation_path.glob("floor*.txt"))
        ceiling_paths = sorted(annotation_path.glob("ceiling*.txt"))
        if not floor_paths:
            missing_floor += 1
        if not ceiling_paths:
            missing_ceiling += 1
        if not floor_paths or not ceiling_paths:
            continue
        pairs.append(
            _room_pair(
                heldout_root,
                annotation_path,
                floor_paths,
                ceiling_paths,
                max_fit_points=DEFAULT_MAX_FIT_POINTS,
                max_spacing_points=DEFAULT_MAX_SPACING_POINTS,
                allowed_areas=(HELD_OUT_AREA,),
            )
        )
    return {
        "artifact_schema": GEOMETRY_SCHEMA,
        "role": "once_opened_area5_floor_ceiling_geometry",
        "calibration_root": str(heldout_root),
        "held_out_area": HELD_OUT_AREA,
        "held_out_artifacts_accessed": True,
        "annotation_directory_count": len(annotation_paths),
        "missing_floor_count": missing_floor,
        "missing_ceiling_count": missing_ceiling,
        "paired_room_count": len(pairs),
        "pairs": [pair.to_dict() for pair in pairs],
    }


def write_json(payload: dict[str, object], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def audit_frozen_gates(core: dict[str, object]) -> FrozenGateAudit:
    cases = list(core["cases"])
    count = len(cases)
    b5_cases = [case for case in cases if case["b5"] is not None]
    m1_cases = [case for case in cases if case["m1"] is not None]
    b5_availability = len(b5_cases) / count if count else 0.0
    m1_availability = len(m1_cases) / count if count else 0.0
    candidate_coverage = float(core["candidate_safe_acceptance_coverage"])
    base_coverage = float(core["base_safe_acceptance_coverage"])
    false_safe = int(core["candidate_false_safe_count"])
    candidate_mean = _mean([float(case["candidate"]["fscore"]) for case in cases])
    base_mean = _mean([float(case["base"]["fscore"]) for case in cases])
    b5_margin = _mean(
        [
            float(case["candidate"]["fscore"]) - float(case["b5"]["fscore"])
            for case in b5_cases
        ]
    )
    m1_margin = _mean(
        [
            float(case["candidate"]["fscore"]) - float(case["m1"]["fscore"])
            for case in m1_cases
        ]
    )
    b5_win_rate = _mean(
        [float(case["candidate"]["fscore"] > case["b5"]["fscore"]) for case in b5_cases]
    )
    m1_win_rate = _mean(
        [float(case["candidate"]["fscore"] > case["m1"]["fscore"]) for case in m1_cases]
    )
    candidate_b5_geometry = _mean(
        [float(case["candidate"]["geometry_loss"]) for case in b5_cases]
    )
    b5_geometry = _mean([float(case["b5"]["geometry_loss"]) for case in b5_cases])
    candidate_m1_geometry = _mean(
        [float(case["candidate"]["geometry_loss"]) for case in m1_cases]
    )
    m1_geometry = _mean([float(case["m1"]["geometry_loss"]) for case in m1_cases])
    candidate_b5_topology = (
        sum(int(case["candidate"]["topology_error"]) for case in b5_cases)
        if b5_cases
        else None
    )
    b5_topology = (
        sum(int(case["b5"]["topology_error"]) for case in b5_cases)
        if b5_cases
        else None
    )
    candidate_m1_topology = (
        sum(int(case["candidate"]["topology_error"]) for case in m1_cases)
        if m1_cases
        else None
    )
    m1_topology = (
        sum(int(case["m1"]["topology_error"]) for case in m1_cases)
        if m1_cases
        else None
    )
    b5_ratio = (
        candidate_b5_topology / b5_topology
        if candidate_b5_topology is not None and b5_topology
        else None
    )
    m1_ratio = (
        candidate_m1_topology / m1_topology
        if candidate_m1_topology is not None and m1_topology
        else None
    )
    panel_gate = count >= MINIMUM_ELIGIBLE_CASES
    safety_gate = bool(
        false_safe == 0
        and candidate_coverage >= MINIMUM_SAFE_ACCEPTANCE_COVERAGE
    )
    construction_gate = bool(
        b5_availability >= MINIMUM_B5_AVAILABILITY
        and m1_availability >= MINIMUM_M1_AVAILABILITY
    )
    geometry_gate = bool(
        b5_margin is not None
        and m1_margin is not None
        and b5_win_rate is not None
        and m1_win_rate is not None
        and candidate_b5_geometry is not None
        and b5_geometry is not None
        and candidate_m1_geometry is not None
        and m1_geometry is not None
        and b5_margin >= MINIMUM_B5_FSCORE_MARGIN
        and m1_margin >= MINIMUM_M1_FSCORE_MARGIN
        and b5_win_rate >= MINIMUM_CASEWISE_WIN_RATE
        and m1_win_rate >= MINIMUM_CASEWISE_WIN_RATE
        and candidate_b5_geometry < b5_geometry
        and candidate_m1_geometry < m1_geometry
    )
    topology_gate = bool(
        b5_ratio is not None
        and m1_ratio is not None
        and b5_ratio <= MAXIMUM_TOPOLOGY_ERROR_RATIO
        and m1_ratio <= MAXIMUM_TOPOLOGY_ERROR_RATIO
    )
    ablation_gate = bool(
        candidate_mean is not None
        and base_mean is not None
        and candidate_coverage
        >= base_coverage - MAXIMUM_BASE_COVERAGE_DEFICIT
        and candidate_mean >= base_mean - MAXIMUM_BASE_FSCORE_DEFICIT
    )
    return FrozenGateAudit(
        eligible_case_count=count,
        candidate_safe_acceptance_coverage=candidate_coverage,
        candidate_false_safe_count=false_safe,
        b5_availability=b5_availability,
        m1_availability=m1_availability,
        candidate_b5_fscore_margin=b5_margin,
        candidate_m1_fscore_margin=m1_margin,
        candidate_b5_casewise_win_rate=b5_win_rate,
        candidate_m1_casewise_win_rate=m1_win_rate,
        candidate_b5_paired_geometry_loss=candidate_b5_geometry,
        b5_geometry_loss=b5_geometry,
        candidate_m1_paired_geometry_loss=candidate_m1_geometry,
        m1_geometry_loss=m1_geometry,
        candidate_b5_paired_topology_error=candidate_b5_topology,
        b5_topology_error=b5_topology,
        candidate_m1_paired_topology_error=candidate_m1_topology,
        m1_topology_error=m1_topology,
        candidate_b5_topology_error_ratio=b5_ratio,
        candidate_m1_topology_error_ratio=m1_ratio,
        base_safe_acceptance_coverage=base_coverage,
        candidate_mean_fscore=candidate_mean,
        base_mean_fscore=base_mean,
        panel_gate_passed=panel_gate,
        safety_gate_passed=safety_gate,
        construction_gate_passed=construction_gate,
        geometry_gate_passed=geometry_gate,
        topology_gate_passed=topology_gate,
        ablation_gate_passed=ablation_gate,
    )


def evaluate_s3dis_room_layer_validation(
    archive_path: str | Path,
    extraction_root: str | Path,
    *,
    protocol_path: str | Path,
    calibration_artifact: str | Path = CALIBRATION_ARTIFACT,
    geometry_output: str | Path,
    core_output: str | Path,
) -> dict[str, object]:
    protocol_digest, protocol_identity = _protocol_identity(Path(protocol_path))
    calibration_path = Path(calibration_artifact)
    calibration_digest = _sha256(calibration_path)
    extraction = extract_heldout_room_layers(archive_path, extraction_root)
    geometry = build_heldout_geometry(extraction_root)
    geometry_digest = write_json(geometry, geometry_output)
    core = evaluate_room_layer_calibration_benchmark(
        geometry_output,
        maximum_angle_degrees=MAXIMUM_ANGLE_DEGREES,
        minimum_bbox_overlap=MINIMUM_BBOX_OVERLAP,
        minimum_common_points_per_layer=MINIMUM_COMMON_POINTS_PER_LAYER,
        minimum_gap_to_residual_snr=MINIMUM_GAP_TO_RESIDUAL_SNR,
        maximum_plane_residual_to_spacing=MAXIMUM_PLANE_RESIDUAL_TO_SPACING,
        observed_per_layer=OBSERVED_PER_LAYER,
        reference_per_layer=REFERENCE_PER_LAYER,
        surface_sample_count=SURFACE_SAMPLE_COUNT,
    )
    core_digest = write_core_result(core, core_output)
    gates = audit_frozen_gates(core)
    identity_gate = bool(
        protocol_identity
        and protocol_digest == PROTOCOL_SHA256
        and calibration_digest == CALIBRATION_SHA256
        and float(core["observation"]["general_position_joggle"])
        == GENERAL_POSITION_JOGGLE
    )
    phase_supported = bool(
        identity_gate
        and gates.panel_gate_passed
        and gates.safety_gate_passed
        and gates.construction_gate_passed
        and gates.geometry_gate_passed
        and gates.topology_gate_passed
        and gates.ablation_gate_passed
    )
    return {
        "artifact_schema": RESULT_SCHEMA,
        "role": "once_opened_building_disjoint_real_floor_ceiling_validation",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": protocol_digest,
        "protocol_identity_passed": protocol_identity,
        "calibration_artifact": str(calibration_path),
        "calibration_sha256": calibration_digest,
        "configuration_identity_gate_passed": identity_gate,
        "extraction": extraction.to_dict(),
        "geometry_artifact": str(geometry_output),
        "geometry_artifact_sha256": geometry_digest,
        "core_artifact": str(core_output),
        "core_artifact_sha256": core_digest,
        "core": core,
        "gates": gates.to_dict(),
        "phase51c_supported": phase_supported,
        "real_long_gap_two_layer_supported": phase_supported,
        "real_scan_supported": phase_supported,
        "held_out_validation_supported": phase_supported,
        "pftf_superiority_supported": False,
        "local_spd_superiority_supported": False,
        "shared_trend_superiority_supported": False,
        "close_layer_transfer_supported": False,
        "deployment_supported": False,
        "claim_boundary": (
            "support is limited to real, building-disjoint, long-gap, "
            "approximately parallel S3DIS floor--ceiling reconstruction with "
            "annotation-defined extraction and observed-XYZ-only routing"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "benchmark-out/s3dis_room_layer_validation_protocol_phase51c.json"
        ),
    )
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=Path(CALIBRATION_ARTIFACT),
    )
    parser.add_argument(
        "--geometry-output",
        type=Path,
        default=Path("benchmark-out/s3dis_room_layer_area5_geometry_phase51c.json"),
    )
    parser.add_argument(
        "--core-output",
        type=Path,
        default=Path("benchmark-out/s3dis_room_layer_validation_core_phase51c.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/s3dis_room_layer_validation_phase51c.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_s3dis_room_layer_validation(
        args.archive,
        args.extract_root,
        protocol_path=args.protocol,
        calibration_artifact=args.calibration_artifact,
        geometry_output=args.geometry_output,
        core_output=args.core_output,
    )
    digest = write_json(result, args.output)
    gates = result["gates"]
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(
        f"eligible={gates['eligible_case_count']} "
        f"safe_coverage={gates['candidate_safe_acceptance_coverage']:.6f} "
        f"false_safe={gates['candidate_false_safe_count']} "
        f"phase51c_supported={str(result['phase51c_supported']).lower()}"
    )


if __name__ == "__main__":
    main()
