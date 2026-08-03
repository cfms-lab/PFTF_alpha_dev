"""Phase-49 preregistration for PFTF shear-signal identifiability."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .learned_pftf_coordinate_map_protocol import (
    CALIBRATION_SEEDS,
    CALIBRATION_STRENGTHS,
    FAMILIES,
    GEOMETRY_BASELINE_FEATURES,
    PFTF_FEATURES,
    TRAIN_SEEDS,
    TRAIN_STRENGTHS,
)

PROTOCOL_SCHEMA = "pftf_alpha_pftf_shear_identifiability_protocol_phase49/v1"
MINIMUM_CALIBRATION_MEDIAN_WITHIN_BLOCK_R2 = 0.75
MINIMUM_CALIBRATION_SIGN_CONSISTENCY = 5.0 / 6.0
MINIMUM_CALIBRATION_FAMILY_DIRECTION_FRACTION = 1.0
MINIMUM_STANDARDIZED_SPAN_EFFECT = 0.25
MAXIMUM_STANDALONE_MAE_FRACTION_OF_TRAIN_MEAN = 0.75


@dataclass(frozen=True)
class PFTFShearIdentifiabilityProtocol:
    artifact_schema: str
    role: str
    source_panel: str
    train_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    train_strengths: tuple[float, ...]
    calibration_strengths: tuple[float, ...]
    families: tuple[str, ...]
    pftf_features: tuple[str, ...]
    geometry_baseline_features: tuple[str, ...]
    held_out_prohibition: str
    block_definition: str
    training_feature_selection: str
    frozen_direction_rule: str
    within_block_endpoints: tuple[str, ...]
    confounding_endpoints: tuple[str, ...]
    standalone_endpoint: str
    minimum_calibration_median_within_block_r2: float
    minimum_calibration_sign_consistency: float
    minimum_calibration_family_direction_fraction: float
    minimum_standardized_span_effect: float
    maximum_standalone_mae_fraction_of_train_mean: float
    stable_signal_gate: str
    pftf_specific_gate: str
    standalone_gate: str
    next_panel_gate: str
    reference_boundary: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "train_seeds",
            "calibration_seeds",
            "train_strengths",
            "calibration_strengths",
            "families",
            "pftf_features",
            "geometry_baseline_features",
            "within_block_endpoints",
            "confounding_endpoints",
        ):
            payload[name] = list(payload[name])
        return payload


def preregister_pftf_shear_identifiability() -> PFTFShearIdentifiabilityProtocol:
    return PFTFShearIdentifiabilityProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="train_calibration_only_post_phase48_identifiability_audit",
        source_panel="Phase-48 quadratic-shear synthetic recovery panel",
        train_seeds=TRAIN_SEEDS,
        calibration_seeds=CALIBRATION_SEEDS,
        train_strengths=TRAIN_STRENGTHS,
        calibration_strengths=CALIBRATION_STRENGTHS,
        families=FAMILIES,
        pftf_features=PFTF_FEATURES,
        geometry_baseline_features=GEOMETRY_BASELINE_FEATURES,
        held_out_prohibition=(
            "Phase-48 held-out seeds, points, labels, predictions, and endpoint "
            "values are prohibited from feature selection and every Phase-49 gate"
        ),
        block_definition=(
            "one block is one surface family and one seed observed at all five "
            "declared strengths; TRAIN has 12 blocks and CALIBRATION has 6"
        ),
        training_feature_selection=(
            "separately for PFTF and non-PFTF geometry, select one feature on "
            "TRAIN by lexicographically maximizing median within-block linear "
            "R2, slope-sign consistency, then standardized span effect; ties use "
            "the earlier frozen feature order"
        ),
        frozen_direction_rule=(
            "the sign of the median TRAIN block slope is frozen before applying "
            "the selected feature to CALIBRATION"
        ),
        within_block_endpoints=(
            "median per-block linear R2",
            "fraction of block slopes matching the frozen TRAIN direction",
            "fraction of family-median slopes matching the frozen direction",
            "median slope times strength span divided by TRAIN feature "
            "standard deviation",
        ),
        confounding_endpoints=(
            "pooled linear R2 without family or seed labels",
            "block variance fraction from a categorical block-intercept model",
            "partial strength R2 after block intercepts",
            "block-to-strength explained-sum-of-squares ratio",
        ),
        standalone_endpoint=(
            "CALIBRATION coefficient MAE of one clipped univariate affine model "
            "fit only on TRAIN, compared with the TRAIN-mean coefficient"
        ),
        minimum_calibration_median_within_block_r2=(
            MINIMUM_CALIBRATION_MEDIAN_WITHIN_BLOCK_R2
        ),
        minimum_calibration_sign_consistency=(
            MINIMUM_CALIBRATION_SIGN_CONSISTENCY
        ),
        minimum_calibration_family_direction_fraction=(
            MINIMUM_CALIBRATION_FAMILY_DIRECTION_FRACTION
        ),
        minimum_standardized_span_effect=MINIMUM_STANDARDIZED_SPAN_EFFECT,
        maximum_standalone_mae_fraction_of_train_mean=(
            MAXIMUM_STANDALONE_MAE_FRACTION_OF_TRAIN_MEAN
        ),
        stable_signal_gate=(
            "the selected PFTF feature must meet all four frozen CALIBRATION "
            "within-block thresholds and retain the TRAIN slope direction"
        ),
        pftf_specific_gate=(
            "the stable PFTF feature must strictly exceed the independently "
            "TRAIN-selected geometry feature in CALIBRATION median within-block R2"
        ),
        standalone_gate=(
            "the TRAIN-only PFTF affine decoder must have CALIBRATION MAE below "
            "both the geometry decoder and 0.75 times the TRAIN-mean baseline MAE"
        ),
        next_panel_gate=(
            "new representation development requires the stable and PFTF-specific "
            "gates; a new held-out panel additionally requires the standalone gate"
        ),
        reference_boundary=(
            "true strengths organize repeated diagnostic blocks and score the "
            "audit; no dense reference surface or alpha/reconstruction endpoint "
            "is consumed, and the block analysis is not a deployable estimator"
        ),
        claim_boundary=(
            "a positive within-block result shows only an observed response to a "
            "known coordinate-aligned synthetic shear under repeated measurements; "
            "it does not establish standalone identifiability, PFTF reconstruction "
            "value, alpha selection, real transfer, exactness, or deployment"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_pftf_shear_identifiability().to_dict(),
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
            "benchmark-out/pftf_shear_identifiability_protocol_phase49.json"
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
