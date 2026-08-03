"""Phase-51 preregistration for real S3DIS wall--board data intake.

This protocol freezes the external corpus, building-disjoint split, and the
information boundary before any reserved Area-5 point coordinates are read.
It is an intake protocol, not a held-out reconstruction result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL_SCHEMA = "pftf_alpha_s3dis_two_layer_intake_protocol_phase51/v1"
DATASET_NAME = "Stanford Large-Scale Indoor Spaces 3D Dataset v1.2 aligned"
DATASET_URL = (
    "https://cvg-data.inf.ethz.ch/s3dis/"
    "Stanford3dDataset_v1.2_Aligned_Version.zip"
)
README_URL = "https://cvg-data.inf.ethz.ch/s3dis/ReadMe.txt"
ORIGINAL_RESOURCE_PAGE = "https://cvgl.stanford.edu/resources.html"
ARCHIVE_NAME = "Stanford3dDataset_v1.2_Aligned_Version.zip"
CALIBRATION_AREAS = ("Area_1", "Area_2", "Area_3", "Area_4", "Area_6")
RESERVED_HELD_OUT_AREA = "Area_5"
TARGET_CLASSES = ("board", "wall")


@dataclass(frozen=True)
class S3DISTwoLayerIntakeProtocol:
    artifact_schema: str
    role: str
    dataset_name: str
    dataset_url: str
    dataset_readme_url: str
    original_resource_page: str
    archive_name: str
    dataset_terms_boundary: str
    calibration_areas: tuple[str, ...]
    reserved_held_out_area: str
    split_rationale: str
    target_classes: tuple[str, ...]
    archive_metadata_allowed_before_final_preregistration: tuple[str, ...]
    calibration_content_allowed: str
    reserved_content_prohibition: str
    calibration_objective: str
    pair_definition_development: str
    future_pair_eligibility_requirements: tuple[str, ...]
    future_observation_reference_split: str
    candidate_method: str
    comparator_methods: tuple[str, ...]
    runtime_information_boundary: str
    evaluation_only_information: tuple[str, ...]
    final_preregistration_requirement: str
    current_support_flags: dict[str, bool]
    excluded_scope: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "calibration_areas",
            "target_classes",
            "archive_metadata_allowed_before_final_preregistration",
            "future_pair_eligibility_requirements",
            "comparator_methods",
            "evaluation_only_information",
            "excluded_scope",
        ):
            payload[name] = list(payload[name])
        return payload


def preregister_s3dis_two_layer_intake() -> S3DISTwoLayerIntakeProtocol:
    """Return the frozen acquisition and split contract for Phase 51."""

    return S3DISTwoLayerIntakeProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="external_real_two_layer_calibration_intake_with_reserved_holdout",
        dataset_name=DATASET_NAME,
        dataset_url=DATASET_URL,
        dataset_readme_url=README_URL,
        original_resource_page=ORIGINAL_RESOURCE_PAGE,
        archive_name=ARCHIVE_NAME,
        dataset_terms_boundary=(
            "the distributed README requests citation but does not state a license; "
            "retain the archive locally for research validation, cite the dataset, "
            "do not redistribute it, and do not claim broader usage rights"
        ),
        calibration_areas=CALIBRATION_AREAS,
        reserved_held_out_area=RESERVED_HELD_OUT_AREA,
        split_rationale=(
            "the dataset README maps Areas 1, 3, and 6 to Building 1, Areas 2 "
            "and 4 to Building 2, and Area 5 alone to Building 3; reserving "
            "Area 5 therefore gives a building-disjoint external test"
        ),
        target_classes=TARGET_CLASSES,
        archive_metadata_allowed_before_final_preregistration=(
            "member path",
            "compressed size",
            "uncompressed size",
            "archive byte size",
            "archive SHA-256",
        ),
        calibration_content_allowed=(
            "XYZRGB contents in Areas 1, 2, 3, 4, and 6 may be opened only to "
            "develop deterministic wall--board pairing, crop, sampling, and "
            "eligibility rules"
        ),
        reserved_content_prohibition=(
            "before a separate final-evaluation protocol is committed, do not "
            "extract, open, parse, summarize, visualize, or compute any statistic "
            "from an Area-5 point-cloud member; central-directory names and sizes "
            "are the only allowed Area-5 information"
        ),
        calibration_objective=(
            "freeze a data-independent automatic definition of a close, locally "
            "overlapping, approximately parallel wall--board two-surface patch"
        ),
        pair_definition_development=(
            "use board and wall instance annotations only to construct the corpus; "
            "develop minimum point count, plane-fit, parallelism, footprint "
            "overlap, gap-to-spacing, and deterministic crop thresholds solely "
            "from calibration areas"
        ),
        future_pair_eligibility_requirements=(
            "both annotated instances meet a frozen minimum point count",
            "both robust plane fits meet a frozen residual bound",
            "plane normals meet a frozen absolute-angle bound",
            "projected board footprint overlaps a frozen wall crop",
            "positive separation and gap-to-spacing ratio lie in frozen ranges",
            "deterministic XYZ-only observation subsample is sampling-sufficient",
        ),
        future_observation_reference_split=(
            "within each eligible annotated instance, assign points by a frozen "
            "coordinate-hash rule to disjoint observed and reference subsets; "
            "the candidate receives only the combined observed XYZ coordinates"
        ),
        candidate_method=(
            "unchanged Phase-50 shared quadratic trend fit, residual two-means, "
            "observed-only sampling gate, and one 2D Delaunay surface per "
            "inferred layer"
        ),
        comparator_methods=(
            "frozen B5 PCA-anisotropic alpha",
            "frozen M1 weighted power-alpha",
            "global-normal two-layer ablation",
        ),
        runtime_information_boundary=(
            "after corpus extraction, reconstruction and routing receive XYZ "
            "coordinates only; RGB, area, room, class names, instance IDs, and "
            "reference points may not influence candidate output"
        ),
        evaluation_only_information=(
            "board versus wall source label",
            "instance identity",
            "held-out reference points",
            "false cross-layer edges and faces",
            "baseline endpoints",
        ),
        final_preregistration_requirement=(
            "after calibration-only intake, commit pair thresholds, Area-5 case "
            "enumeration rule, observation/reference hash split, metrics, gates, "
            "and failure handling before opening any Area-5 member content"
        ),
        current_support_flags={
            "external_archive_intake_supported": False,
            "real_scan_supported": False,
            "held_out_validation_supported": False,
            "pftf_superiority_supported": False,
            "deployment_supported": False,
        },
        excluded_scope=(
            "semantic scene discovery",
            "RGB-conditioned routing",
            "annotation-free corpus extraction",
            "general indoor reconstruction",
            "PFTF or local-SPD superiority",
            "commercial dataset rights",
            "deployment",
        ),
        claim_boundary=(
            "this protocol supports only a leakage-controlled route to a real "
            "building-disjoint wall--board validation; it is not itself evidence "
            "of real-scan efficacy"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_s3dis_two_layer_intake().to_dict(),
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
        default=Path("benchmark-out/s3dis_two_layer_intake_protocol_phase51.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_protocol(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
