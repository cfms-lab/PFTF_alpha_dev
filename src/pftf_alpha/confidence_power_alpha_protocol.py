"""Phase-45 preregistration for confidence-aware regular alpha construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .confidence_alpha_panel import ReferenceSurfaceFamily
from .confidence_alpha_transfer_panel import (
    TransferStressProfile,
    make_confidence_alpha_transfer_case,
)
from .synthetic import PanelSplit

PROTOCOL_SCHEMA = "pftf_alpha_confidence_power_alpha_protocol_phase45/v1"
CALIBRATION_SEEDS = (45_001,)
HELD_OUT_SEEDS = (45_101, 45_102, 45_103)
M1_DENSITY_WEIGHT_SCALE = 0.375
CONFIDENCE_PENALTY_SCALES = (0.125, 0.25, 0.375, 0.5)
FROZEN_BINARY_CONFIDENCE_THRESHOLD = 0.25
FROZEN_CONTINUOUS_PENALTY_STRENGTH = 1.0
MINIMUM_SELECTED_CELL_FRACTION = 0.50
MAXIMUM_SELECTED_CELL_FRACTION = 0.98
SURFACE_SAMPLE_COUNT = 768
MINIMUM_CASEWISE_JOINT_WIN_FRACTION = 2.0 / 3.0
MAXIMUM_FALLBACK_FRACTION = 0.10
MINIMUM_CONNECTIVITY_CHANGE_FRACTION = 0.50


@dataclass(frozen=True)
class ConfidencePowerAlphaProtocol:
    artifact_schema: str
    role: str
    families: tuple[str, ...]
    profiles: tuple[str, ...]
    calibration_seeds: tuple[int, ...]
    held_out_seeds: tuple[int, ...]
    calibration_case_count: int
    held_out_case_count: int
    m1_density_weight_scale: float
    confidence_penalty_scales: tuple[float, ...]
    confidence_power_weight_formula: str
    point_submersion_policy: str
    frozen_binary_confidence_threshold: float
    frozen_continuous_penalty_strength: float
    critical_score_selection: str
    calibration_objective: str
    frozen_comparators: tuple[str, ...]
    validation_gate: str
    reference_boundary: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "families",
            "profiles",
            "calibration_seeds",
            "held_out_seeds",
            "confidence_penalty_scales",
            "frozen_comparators",
        ):
            payload[key] = list(payload[key])
        return payload


def make_confidence_power_alpha_panel(split: PanelSplit | str):
    selected_split = PanelSplit(split)
    if selected_split not in (PanelSplit.CALIBRATION, PanelSplit.HELD_OUT):
        raise ValueError("Phase-45 panel permits calibration or held_out only")
    seeds = (
        CALIBRATION_SEEDS
        if selected_split is PanelSplit.CALIBRATION
        else HELD_OUT_SEEDS
    )
    return tuple(
        make_confidence_alpha_transfer_case(
            family,
            profile,
            split=selected_split,
            seed=seed,
        )
        for family in ReferenceSurfaceFamily
        for profile in TransferStressProfile
        for seed in seeds
    )


def preregister_confidence_power_alpha() -> ConfidencePowerAlphaProtocol:
    block_count = len(ReferenceSurfaceFamily) * len(TransferStressProfile)
    return ConfidencePowerAlphaProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="pre_method_confidence_power_alpha_protocol",
        families=tuple(family.value for family in ReferenceSurfaceFamily),
        profiles=tuple(profile.value for profile in TransferStressProfile),
        calibration_seeds=CALIBRATION_SEEDS,
        held_out_seeds=HELD_OUT_SEEDS,
        calibration_case_count=block_count * len(CALIBRATION_SEEDS),
        held_out_case_count=block_count * len(HELD_OUT_SEEDS),
        m1_density_weight_scale=M1_DENSITY_WEIGHT_SCALE,
        confidence_penalty_scales=CONFIDENCE_PENALTY_SCALES,
        confidence_power_weight_formula=(
            "w_i = spacing_i^2 * (0.375^2 - penalty_scale^2 * "
            "(1 - observed_confidence_i))"
        ),
        point_submersion_policy=(
            "reject any calibration penalty that submerges a point in any case; "
            "on held-out, fail closed per case to M1 density weights at scale "
            "0.375 and count the fallback"
        ),
        frozen_binary_confidence_threshold=FROZEN_BINARY_CONFIDENCE_THRESHOLD,
        frozen_continuous_penalty_strength=FROZEN_CONTINUOUS_PENALTY_STRENGTH,
        critical_score_selection=(
            "reuse unchanged Phase-44 complete adjacent log-gap selection over "
            "all finite unique top-cell scores with lower selected-cell fraction "
            "in [0.50, 0.98]"
        ),
        calibration_objective=(
            "among non-submerging penalty scales, minimize calibration mean "
            "normalized Chamfer-squared + normalized Hausdorff + 0.05 * Betti "
            "L1 error; ties prefer the lower penalty scale"
        ),
        frozen_comparators=(
            "anchor_density_B4",
            "fused_density_B4",
            "fused_pca_B5",
            "M1_density_power_alpha_scale_0.375",
            "binary_confidence_deletion_threshold_0.25",
            "fixed_cell_continuous_confidence_strength_1.0",
        ),
        validation_gate=(
            "candidate must have lower held-out mean geometry and objective than "
            "anchor B4, fused B4, M1, binary deletion, and fixed-cell continuous; "
            "no larger mean Betti error than all five; repeat stability no larger "
            "than M1 and fixed-cell continuous; lower objective than B5; joint "
            "casewise objective wins over M1 and fixed-cell continuous in at least "
            "2/3 of cases; connectivity must differ from M1 in at least 1/2 of "
            "cases; fallback fraction must not exceed 0.10"
        ),
        reference_boundary=(
            "confidence and power weights use observed coordinates only; reference "
            "points, component labels, family/profile identity, and perturbation "
            "values are evaluation-only"
        ),
        claim_boundary=(
            "a positive result supports only a floating-Qhull confidence-aware "
            "regular alpha construction; it is not exact, a PFTF-trained alpha, "
            "a local-SPD metric complex, real-scan evidence, or deployment evidence"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_confidence_power_alpha().to_dict(),
        indent=2,
        sort_keys=True,
    ) + "\n"
    output.write_text(text, encoding="utf-8")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/confidence_power_alpha_protocol_phase45.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_protocol(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
