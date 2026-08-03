"""Run the Phase-49 TRAIN/CALIBRATION-only shear identifiability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .learned_pftf_coordinate_map import (
    MapRecoveryCase,
    geometry_summary_features,
    make_map_recovery_cases,
    pftf_summary_features,
)
from .learned_pftf_coordinate_map_protocol import (
    CALIBRATION_SEEDS,
    CALIBRATION_STRENGTHS,
    FAMILIES,
    GEOMETRY_BASELINE_FEATURES,
    MAP_STRENGTH_BOUNDS,
    PFTF_FEATURES,
    TRAIN_SEEDS,
    TRAIN_STRENGTHS,
)
from .pftf_shear_identifiability_protocol import (
    MAXIMUM_STANDALONE_MAE_FRACTION_OF_TRAIN_MEAN,
    MINIMUM_CALIBRATION_FAMILY_DIRECTION_FRACTION,
    MINIMUM_CALIBRATION_MEDIAN_WITHIN_BLOCK_R2,
    MINIMUM_CALIBRATION_SIGN_CONSISTENCY,
    MINIMUM_STANDARDIZED_SPAN_EFFECT,
    preregister_pftf_shear_identifiability,
)
from .synthetic import PanelSplit

FloatArray = NDArray[np.float64]
RESULT_SCHEMA = "pftf_alpha_pftf_shear_identifiability_phase49/v1"


@dataclass(frozen=True)
class FeatureTrainingScore:
    feature_name: str
    median_within_block_r2: float
    block_slope_sign_consistency: float
    median_slope: float
    standardized_span_effect: float


@dataclass(frozen=True)
class SelectedFeatureAudit:
    feature_name: str
    frozen_train_direction: int
    train_median_within_block_r2: float
    train_slope_sign_consistency: float
    train_median_slope: float
    train_standardized_span_effect: float
    calibration_median_within_block_r2: float
    calibration_slope_sign_consistency: float
    calibration_median_slope: float
    calibration_family_direction_fraction: float
    calibration_standardized_span_effect: float
    calibration_pooled_r2: float
    calibration_block_variance_fraction: float
    calibration_partial_strength_r2: float
    calibration_block_to_strength_ss_ratio: float
    standalone_calibration_mae: float


@dataclass(frozen=True)
class PFTFShearIdentifiabilityResult:
    artifact_schema: str
    protocol: dict[str, object]
    train_case_count: int
    calibration_case_count: int
    prohibited_held_out_case_count: int
    pftf_training_scores: tuple[FeatureTrainingScore, ...]
    geometry_training_scores: tuple[FeatureTrainingScore, ...]
    selected_pftf: SelectedFeatureAudit
    selected_geometry: SelectedFeatureAudit
    train_mean_strength: float
    train_mean_calibration_mae: float
    stable_within_block_pftf_signal_supported: bool
    pftf_specific_within_block_signal_supported: bool
    standalone_pftf_identifiability_supported: bool
    new_representation_development_justified: bool
    new_held_out_panel_justified: bool
    phase49_identifiability_supported: bool
    pftf_reconstruction_value_supported: bool
    global_alpha_selection_supported: bool
    real_scan_transfer_supported: bool
    exact_predicates_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _blocks(cases: tuple[MapRecoveryCase, ...]) -> tuple[tuple[int, ...], ...]:
    keys: list[tuple[str, int]] = []
    for case in cases:
        key = (case.family, case.seed)
        if key not in keys:
            keys.append(key)
    return tuple(
        tuple(
            index
            for index, case in enumerate(cases)
            if (case.family, case.seed) == key
        )
        for key in keys
    )


def _slope_and_r2(strengths: FloatArray, values: FloatArray) -> tuple[float, float]:
    design = np.column_stack((np.ones(len(strengths)), strengths))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    prediction = design @ coefficients
    residual_sum = float(np.sum((values - prediction) ** 2))
    total_sum = float(np.sum((values - np.mean(values)) ** 2))
    r2 = (
        0.0
        if total_sum <= np.finfo(np.float64).eps
        else 1.0 - residual_sum / total_sum
    )
    return float(coefficients[1]), float(np.clip(r2, 0.0, 1.0))


def _group_matrix(
    cases: tuple[MapRecoveryCase, ...],
    *,
    feature_group: str,
) -> FloatArray:
    extractor = {
        "pftf": pftf_summary_features,
        "geometry": geometry_summary_features,
    }.get(feature_group)
    if extractor is None:
        raise ValueError("feature_group must be 'pftf' or 'geometry'")
    return np.vstack([extractor(case.observed_points) for case in cases])


def _training_scores(
    cases: tuple[MapRecoveryCase, ...],
    features: FloatArray,
    feature_names: tuple[str, ...],
) -> tuple[FeatureTrainingScore, ...]:
    strengths = np.asarray([case.true_strength for case in cases], dtype=np.float64)
    blocks = _blocks(cases)
    scores: list[FeatureTrainingScore] = []
    for feature_index, feature_name in enumerate(feature_names):
        slopes: list[float] = []
        r2_values: list[float] = []
        for block in blocks:
            indices = np.asarray(block, dtype=np.int64)
            slope, r2 = _slope_and_r2(
                strengths[indices], features[indices, feature_index]
            )
            slopes.append(slope)
            r2_values.append(r2)
        median_slope = float(np.median(slopes))
        direction = 1 if median_slope >= 0.0 else -1
        sign_consistency = float(np.mean(np.sign(slopes) == direction))
        feature_scale = float(np.std(features[:, feature_index]))
        if feature_scale <= np.finfo(np.float64).eps:
            span_effect = 0.0
        else:
            span_effect = (
                abs(median_slope)
                * float(np.max(strengths) - np.min(strengths))
                / feature_scale
            )
        scores.append(
            FeatureTrainingScore(
                feature_name=feature_name,
                median_within_block_r2=float(np.median(r2_values)),
                block_slope_sign_consistency=sign_consistency,
                median_slope=median_slope,
                standardized_span_effect=float(span_effect),
            )
        )
    return tuple(scores)


def _selected_index(scores: tuple[FeatureTrainingScore, ...]) -> int:
    return max(
        range(len(scores)),
        key=lambda index: (
            scores[index].median_within_block_r2,
            scores[index].block_slope_sign_consistency,
            scores[index].standardized_span_effect,
            -index,
        ),
    )


def _variance_decomposition(
    cases: tuple[MapRecoveryCase, ...],
    values: FloatArray,
) -> tuple[float, float, float, float]:
    strengths = np.asarray([case.true_strength for case in cases], dtype=np.float64)
    total_sum = float(np.sum((values - np.mean(values)) ** 2))
    _, pooled_r2 = _slope_and_r2(strengths, values)
    blocks = _blocks(cases)
    block_design = np.zeros((len(cases), len(blocks)), dtype=np.float64)
    for block_index, block in enumerate(blocks):
        block_design[np.asarray(block, dtype=np.int64), block_index] = 1.0
    block_prediction = block_design @ np.linalg.lstsq(
        block_design, values, rcond=None
    )[0]
    block_sse = float(np.sum((values - block_prediction) ** 2))
    full_design = np.column_stack((block_design, strengths))
    full_prediction = full_design @ np.linalg.lstsq(
        full_design, values, rcond=None
    )[0]
    full_sse = float(np.sum((values - full_prediction) ** 2))
    block_ss = max(0.0, total_sum - block_sse)
    strength_ss = max(0.0, block_sse - full_sse)
    block_fraction = 0.0 if total_sum <= 0.0 else block_ss / total_sum
    partial_strength_r2 = 0.0 if block_sse <= 0.0 else strength_ss / block_sse
    ratio = math.inf if strength_ss <= 0.0 else block_ss / strength_ss
    return (
        pooled_r2,
        float(block_fraction),
        float(partial_strength_r2),
        float(ratio),
    )


def _standalone_mae(
    train_values: FloatArray,
    train_targets: FloatArray,
    calibration_values: FloatArray,
    calibration_targets: FloatArray,
) -> float:
    design = np.column_stack((np.ones(len(train_values)), train_values))
    coefficients = np.linalg.lstsq(design, train_targets, rcond=None)[0]
    calibration_design = np.column_stack(
        (np.ones(len(calibration_values)), calibration_values)
    )
    predictions = np.clip(
        calibration_design @ coefficients,
        MAP_STRENGTH_BOUNDS[0],
        MAP_STRENGTH_BOUNDS[1],
    )
    return float(np.mean(np.abs(predictions - calibration_targets)))


def _selected_audit(
    train_cases: tuple[MapRecoveryCase, ...],
    calibration_cases: tuple[MapRecoveryCase, ...],
    train_features: FloatArray,
    calibration_features: FloatArray,
    scores: tuple[FeatureTrainingScore, ...],
) -> SelectedFeatureAudit:
    selected_index = _selected_index(scores)
    selected_score = scores[selected_index]
    direction = 1 if selected_score.median_slope >= 0.0 else -1
    train_values = train_features[:, selected_index]
    calibration_values = calibration_features[:, selected_index]
    train_targets = np.asarray(
        [case.true_strength for case in train_cases], dtype=np.float64
    )
    calibration_targets = np.asarray(
        [case.true_strength for case in calibration_cases], dtype=np.float64
    )

    calibration_slopes: list[float] = []
    calibration_r2: list[float] = []
    block_families: list[str] = []
    for block in _blocks(calibration_cases):
        indices = np.asarray(block, dtype=np.int64)
        slope, r2 = _slope_and_r2(
            calibration_targets[indices], calibration_values[indices]
        )
        calibration_slopes.append(slope)
        calibration_r2.append(r2)
        block_families.append(calibration_cases[block[0]].family)
    family_directions: list[bool] = []
    for family in FAMILIES:
        family_slopes = [
            slope
            for slope, block_family in zip(
                calibration_slopes, block_families, strict=True
            )
            if block_family == family
        ]
        family_directions.append(np.sign(np.median(family_slopes)) == direction)
    train_scale = float(np.std(train_values))
    calibration_median_slope = float(np.median(calibration_slopes))
    span = float(np.max(calibration_targets) - np.min(calibration_targets))
    standardized_effect = (
        0.0
        if train_scale <= np.finfo(np.float64).eps
        else abs(calibration_median_slope) * span / train_scale
    )
    pooled_r2, block_fraction, partial_r2, ss_ratio = _variance_decomposition(
        calibration_cases, calibration_values
    )
    return SelectedFeatureAudit(
        feature_name=selected_score.feature_name,
        frozen_train_direction=direction,
        train_median_within_block_r2=selected_score.median_within_block_r2,
        train_slope_sign_consistency=selected_score.block_slope_sign_consistency,
        train_median_slope=selected_score.median_slope,
        train_standardized_span_effect=selected_score.standardized_span_effect,
        calibration_median_within_block_r2=float(np.median(calibration_r2)),
        calibration_slope_sign_consistency=float(
            np.mean(np.sign(calibration_slopes) == direction)
        ),
        calibration_median_slope=calibration_median_slope,
        calibration_family_direction_fraction=float(np.mean(family_directions)),
        calibration_standardized_span_effect=float(standardized_effect),
        calibration_pooled_r2=pooled_r2,
        calibration_block_variance_fraction=block_fraction,
        calibration_partial_strength_r2=partial_r2,
        calibration_block_to_strength_ss_ratio=ss_ratio,
        standalone_calibration_mae=_standalone_mae(
            train_values,
            train_targets,
            calibration_values,
            calibration_targets,
        ),
    )


@lru_cache(maxsize=1)
def evaluate_pftf_shear_identifiability() -> PFTFShearIdentifiabilityResult:
    protocol = preregister_pftf_shear_identifiability()
    train_cases = make_map_recovery_cases(
        split=PanelSplit.TRAIN,
        seeds=TRAIN_SEEDS,
        strengths=TRAIN_STRENGTHS,
    )
    calibration_cases = make_map_recovery_cases(
        split=PanelSplit.CALIBRATION,
        seeds=CALIBRATION_SEEDS,
        strengths=CALIBRATION_STRENGTHS,
    )
    train_pftf = _group_matrix(train_cases, feature_group="pftf")
    calibration_pftf = _group_matrix(calibration_cases, feature_group="pftf")
    train_geometry = _group_matrix(train_cases, feature_group="geometry")
    calibration_geometry = _group_matrix(
        calibration_cases, feature_group="geometry"
    )
    pftf_scores = _training_scores(train_cases, train_pftf, PFTF_FEATURES)
    geometry_scores = _training_scores(
        train_cases, train_geometry, GEOMETRY_BASELINE_FEATURES
    )
    pftf_audit = _selected_audit(
        train_cases,
        calibration_cases,
        train_pftf,
        calibration_pftf,
        pftf_scores,
    )
    geometry_audit = _selected_audit(
        train_cases,
        calibration_cases,
        train_geometry,
        calibration_geometry,
        geometry_scores,
    )
    train_targets = np.asarray(
        [case.true_strength for case in train_cases], dtype=np.float64
    )
    calibration_targets = np.asarray(
        [case.true_strength for case in calibration_cases], dtype=np.float64
    )
    train_mean = float(np.mean(train_targets))
    train_mean_mae = float(np.mean(np.abs(calibration_targets - train_mean)))

    stable_signal = (
        pftf_audit.calibration_median_within_block_r2
        >= MINIMUM_CALIBRATION_MEDIAN_WITHIN_BLOCK_R2
        and pftf_audit.calibration_slope_sign_consistency
        >= MINIMUM_CALIBRATION_SIGN_CONSISTENCY
        and pftf_audit.calibration_family_direction_fraction
        >= MINIMUM_CALIBRATION_FAMILY_DIRECTION_FRACTION
        and pftf_audit.calibration_standardized_span_effect
        >= MINIMUM_STANDARDIZED_SPAN_EFFECT
        and np.sign(pftf_audit.calibration_median_slope)
        == pftf_audit.frozen_train_direction
    )
    pftf_specific = (
        stable_signal
        and pftf_audit.calibration_median_within_block_r2
        > geometry_audit.calibration_median_within_block_r2
    )
    standalone = (
        pftf_audit.standalone_calibration_mae
        < geometry_audit.standalone_calibration_mae
        and pftf_audit.standalone_calibration_mae
        < MAXIMUM_STANDALONE_MAE_FRACTION_OF_TRAIN_MEAN * train_mean_mae
    )
    new_representation = pftf_specific
    new_held_out = pftf_specific and standalone
    return PFTFShearIdentifiabilityResult(
        artifact_schema=RESULT_SCHEMA,
        protocol=protocol.to_dict(),
        train_case_count=len(train_cases),
        calibration_case_count=len(calibration_cases),
        prohibited_held_out_case_count=0,
        pftf_training_scores=pftf_scores,
        geometry_training_scores=geometry_scores,
        selected_pftf=pftf_audit,
        selected_geometry=geometry_audit,
        train_mean_strength=train_mean,
        train_mean_calibration_mae=train_mean_mae,
        stable_within_block_pftf_signal_supported=stable_signal,
        pftf_specific_within_block_signal_supported=pftf_specific,
        standalone_pftf_identifiability_supported=standalone,
        new_representation_development_justified=new_representation,
        new_held_out_panel_justified=new_held_out,
        phase49_identifiability_supported=new_held_out,
        pftf_reconstruction_value_supported=False,
        global_alpha_selection_supported=False,
        real_scan_transfer_supported=False,
        exact_predicates_supported=False,
        deployment_supported=False,
    )


def write_result(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        evaluate_pftf_shear_identifiability().to_dict(),
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
        default=Path("benchmark-out/pftf_shear_identifiability_phase49.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_pftf_shear_identifiability()
    digest = write_result(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(f"selected_pftf={result.selected_pftf.feature_name}")
    print(f"selected_geometry={result.selected_geometry.feature_name}")
    print(
        "pftf_calibration_median_within_block_r2="
        f"{result.selected_pftf.calibration_median_within_block_r2:.12g}"
    )
    print(
        "geometry_calibration_median_within_block_r2="
        f"{result.selected_geometry.calibration_median_within_block_r2:.12g}"
    )
    print(
        "pftf_standalone_calibration_mae="
        f"{result.selected_pftf.standalone_calibration_mae:.12g}"
    )
    print(
        "stable_within_block_pftf_signal_supported="
        f"{str(result.stable_within_block_pftf_signal_supported).lower()}"
    )
    print(
        "pftf_specific_within_block_signal_supported="
        f"{str(result.pftf_specific_within_block_signal_supported).lower()}"
    )
    print(
        "standalone_pftf_identifiability_supported="
        f"{str(result.standalone_pftf_identifiability_supported).lower()}"
    )
    print(
        "new_held_out_panel_justified="
        f"{str(result.new_held_out_panel_justified).lower()}"
    )


if __name__ == "__main__":
    main()
