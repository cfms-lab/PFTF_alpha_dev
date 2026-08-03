"""Phase-48 preregistration for a learned PFTF-conditioned coordinate map."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL_SCHEMA = "pftf_alpha_learned_pftf_coordinate_map_protocol_phase48/v1"
POINT_COUNT = 64
REFERENCE_COUNT = 256
K_NEIGHBORS = 8
MAP_STRENGTH_BOUNDS = (0.0, 0.40)
TRAIN_SEEDS = (48_001, 48_002, 48_003, 48_004)
CALIBRATION_SEEDS = (48_101, 48_102)
HELD_OUT_SEEDS = (48_201, 48_202, 48_203)
TRAIN_STRENGTHS = (0.00, 0.08, 0.16, 0.24, 0.32)
CALIBRATION_STRENGTHS = (0.04, 0.12, 0.20, 0.28, 0.36)
HELD_OUT_STRENGTHS = (0.02, 0.10, 0.18, 0.26, 0.34)
RIDGE_GRID = (0.0, 1.0e-4, 1.0e-2, 1.0)
FAMILIES = ("torus", "disconnected_parts", "sharp_crease")
PFTF_FEATURES = (
    "relation_xy_mean",
    "relation_xy_std",
    "relation_strength_median",
    "relation_strength_q90",
    "confidence_mean",
    "reciprocity_mean",
    "log_scale_std",
)
GEOMETRY_BASELINE_FEATURES = (
    "normalized_covariance_xy",
    "x_skewness",
    "y_skewness",
    "axis_variance_ratio",
)


@dataclass(frozen=True)
class LearnedPFTFCoordinateMapProtocol:
    artifact_schema: str
    role: str
    ambient_dimension: int
    point_count: int
    reference_count: int
    families: tuple[str, ...]
    train_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    held_out_seeds: tuple[int, ...]
    train_strengths: tuple[float, ...]
    calibration_strengths: tuple[float, ...]
    held_out_strengths: tuple[float, ...]
    map_strength_bounds: tuple[float, float]
    corruption_map: str
    correction_map: str
    inverse_certificate: str
    normalization: str
    k_neighbors: int
    pftf_features: tuple[str, ...]
    geometry_baseline_features: tuple[str, ...]
    learner: str
    ridge_grid: tuple[float, ...]
    training_rule: str
    calibration_rule: str
    held_out_rule: str
    frozen_comparators: tuple[str, ...]
    primary_endpoints: tuple[str, ...]
    pftf_value_gate: str
    construction_gate: str
    reference_boundary: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "families",
            "train_seeds",
            "calibration_seeds",
            "held_out_seeds",
            "train_strengths",
            "calibration_strengths",
            "held_out_strengths",
            "map_strength_bounds",
            "pftf_features",
            "geometry_baseline_features",
            "ridge_grid",
            "frozen_comparators",
            "primary_endpoints",
        ):
            payload[name] = list(payload[name])
        return payload


def preregister_learned_pftf_coordinate_map() -> LearnedPFTFCoordinateMapProtocol:
    return LearnedPFTFCoordinateMapProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="pre_implementation_synthetic_pftf_conditioned_map_recovery",
        ambient_dimension=3,
        point_count=POINT_COUNT,
        reference_count=REFERENCE_COUNT,
        families=FAMILIES,
        train_seeds=TRAIN_SEEDS,
        calibration_seeds=CALIBRATION_SEEDS,
        held_out_seeds=HELD_OUT_SEEDS,
        train_strengths=TRAIN_STRENGTHS,
        calibration_strengths=CALIBRATION_STRENGTHS,
        held_out_strengths=HELD_OUT_STRENGTHS,
        map_strength_bounds=MAP_STRENGTH_BOUNDS,
        corruption_map="Phi_-s(x,y,z) = (x, y - s*x^2, z)",
        correction_map="Phi_s(x,y,z) = (x, y + s*x^2, z)",
        inverse_certificate=(
            "every predicted scalar s defines the Phase-47 analytic quadratic "
            "shear with explicit global inverse Phi_-s and determinant one"
        ),
        normalization=(
            "each latent synthetic cloud is centered and divided by its RMS "
            "radius before the frozen corruption is applied"
        ),
        k_neighbors=K_NEIGHBORS,
        pftf_features=PFTF_FEATURES,
        geometry_baseline_features=GEOMETRY_BASELINE_FEATURES,
        learner=(
            "training-standardized linear ridge regression with an unpenalized "
            "intercept; predictions are clipped to the declared [0,0.40] family"
        ),
        ridge_grid=RIDGE_GRID,
        training_rule=(
            "fit feature centering/scaling and ridge coefficients only on the "
            "60 TRAIN cases; zero-variance feature scales are replaced by one"
        ),
        calibration_rule=(
            "choose ridge penalty only by minimum coefficient MAE on the 30 "
            "CALIBRATION cases, breaking exact ties toward the larger penalty; "
            "then freeze preprocessing, coefficients, penalty, and clipping"
        ),
        held_out_rule=(
            "evaluate the frozen models once on 45 HELD_OUT cases; held-out "
            "labels, latent points, and alpha complexes cannot change any model"
        ),
        frozen_comparators=(
            "identity map with coefficient zero",
            "constant coefficient equal to the TRAIN mean",
            "ridge model using only four non-PFTF global covariance/moment features",
            "oracle true coefficient as an evaluation ceiling only",
        ),
        primary_endpoints=(
            "held-out correction-coefficient MAE",
            "held-out corrected-coordinate RMS error to latent coordinates",
            "held-out Delaunay top-cell Jaccard against the latent alpha complex",
        ),
        pftf_value_gate=(
            "the frozen PFTF model must strictly beat both the TRAIN-mean and "
            "non-PFTF geometry models on all three mean primary endpoints"
        ),
        construction_gate=(
            "every predicted map must remain inside the strength bounds, have "
            "determinant one, and pass inverse roundtrip error <= 1e-12"
        ),
        reference_boundary=(
            "PFTF and comparator features use corrupted observed coordinates "
            "only; latent points and true coefficients are training labels or "
            "evaluation targets, never selection inputs on held-out cases"
        ),
        claim_boundary=(
            "a positive result supports only recovery of one coordinate-aligned "
            "scalar quadratic-shear family on the declared synthetic panel; it "
            "does not establish an arbitrary local-SPD field, a general neural "
            "or nonlinear map learner, alpha selection, reconstruction or "
            "topology advantage, real-scan transfer, exact predicates, or deployment"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_learned_pftf_coordinate_map().to_dict(),
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
            "benchmark-out/learned_pftf_coordinate_map_protocol_phase48.json"
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
