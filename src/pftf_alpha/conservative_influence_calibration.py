"""Dual-cohort conservative calibration for frozen Phase-13 influence audit."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .local_insertion_influence import (
    InfluenceRectangle,
    LocalInsertionInfluenceConfig,
    LocalInsertionInfluencePanelResult,
    LocalInsertionInfluenceRawCase,
    evaluate_local_insertion_influence_raw_panel,
    local_insertion_influence_features,
    materialize_local_insertion_influence_panel,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .sensor_stress import DEFAULT_POINT_COUNTS, DEFAULT_STRESSES, SensorStress
from .shared_trend_inference import SharedTrendConfig

FloatArray = NDArray[np.float64]

CALIBRATION_A_SEED = 21300804
CALIBRATION_B_SEED = 21400804
FINAL_HELD_OUT_SEED = 21500804


@dataclass(frozen=True)
class InfluenceFeatureCohort:
    """Evaluation-labeled feature groups used only during calibration."""

    harmful: tuple[tuple[float, float], ...]
    focus_safe: tuple[tuple[float, float], ...]
    all_safe: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ConservativeInfluenceCalibrationResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    calibration_a_seed: int
    calibration_b_seed: int
    final_held_out_seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    influence_config: LocalInsertionInfluenceConfig
    rectangle_selection_rule: str
    selected_rectangle: InfluenceRectangle | None
    calibration_a: LocalInsertionInfluencePanelResult
    calibration_b: LocalInsertionInfluencePanelResult
    final_held_out: LocalInsertionInfluencePanelResult | None
    phase13_supported: bool
    trimmed_reconstruction_supported: bool
    real_scan_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_predecessor": self.frozen_predecessor,
            "calibration_a_seed": self.calibration_a_seed,
            "calibration_b_seed": self.calibration_b_seed,
            "final_held_out_seed": self.final_held_out_seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "influence_config": asdict(self.influence_config),
            "rectangle_selection_rule": self.rectangle_selection_rule,
            "selected_rectangle": (
                None
                if self.selected_rectangle is None
                else self.selected_rectangle.to_dict()
            ),
            "calibration_a": self.calibration_a.to_dict(),
            "calibration_b": self.calibration_b.to_dict(),
            "final_held_out": (
                None
                if self.final_held_out is None
                else self.final_held_out.to_dict()
            ),
            "phase13_supported": self.phase13_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "real_scan_supported": self.real_scan_supported,
            "deployment_supported": self.deployment_supported,
        }


def _feature_array(
    values: Sequence[tuple[float, float]],
    *,
    name: str,
) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if not result.size:
        return np.empty((0, 2), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must contain (peak, support) pairs")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _accepted_count(
    features: FloatArray,
    peak_threshold: float,
    support_threshold: float,
) -> int:
    return int(
        np.sum(
            np.logical_and(
                features[:, 0] <= peak_threshold,
                features[:, 1] <= support_threshold,
            )
        )
    )


def calibrate_dual_cohort_rectangle(
    cohort_a: InfluenceFeatureCohort,
    cohort_b: InfluenceFeatureCohort,
) -> InfluenceRectangle | None:
    """Select a zero-harm rectangle using worst-cohort retention first."""

    cohorts = (cohort_a, cohort_b)
    harmful = tuple(
        _feature_array(cohort.harmful, name=f"cohort_{index}_harmful")
        for index, cohort in enumerate(cohorts)
    )
    focus_safe = tuple(
        _feature_array(cohort.focus_safe, name=f"cohort_{index}_focus_safe")
        for index, cohort in enumerate(cohorts)
    )
    all_safe = tuple(
        _feature_array(cohort.all_safe, name=f"cohort_{index}_all_safe")
        for index, cohort in enumerate(cohorts)
    )
    if any(not values.shape[0] for values in harmful):
        return None

    combined_harmful = np.vstack(harmful)
    peak_candidates = sorted(
        {math.inf}
        | {
            float(np.nextafter(value, -np.inf))
            for value in combined_harmful[:, 0].tolist()
        }
    )
    support_candidates = sorted(
        {math.inf}
        | {
            float(np.nextafter(value, -np.inf))
            for value in combined_harmful[:, 1].tolist()
        }
    )

    best: InfluenceRectangle | None = None
    best_key: tuple[float, int, float, int, float, float] | None = None
    for peak_threshold in peak_candidates:
        for support_threshold in support_candidates:
            if any(
                _accepted_count(values, peak_threshold, support_threshold) > 0
                for values in harmful
            ):
                continue
            focus_counts = tuple(
                _accepted_count(values, peak_threshold, support_threshold)
                for values in focus_safe
            )
            all_counts = tuple(
                _accepted_count(values, peak_threshold, support_threshold)
                for values in all_safe
            )
            focus_retentions = tuple(
                0.0 if not values.shape[0] else count / values.shape[0]
                for count, values in zip(focus_counts, focus_safe, strict=True)
            )
            all_retentions = tuple(
                0.0 if not values.shape[0] else count / values.shape[0]
                for count, values in zip(all_counts, all_safe, strict=True)
            )
            key = (
                min(focus_retentions),
                sum(focus_counts),
                min(all_retentions),
                sum(all_counts),
                peak_threshold,
                support_threshold,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = InfluenceRectangle(
                    peak_threshold=peak_threshold,
                    support_threshold=support_threshold,
                    retained_focus_safe_count=sum(focus_counts),
                    retained_all_safe_count=sum(all_counts),
                )
    return best


def _feature_cohort(
    rows: tuple[LocalInsertionInfluenceRawCase, ...],
) -> InfluenceFeatureCohort:
    harmful = tuple(
        local_insertion_influence_features(row)
        for row in rows
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
        and row.stress.is_outlier_stress
        and row.endpoint.geometry_topology_harm_present
    )
    safe_rows = tuple(
        row
        for row in rows
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
        and not row.endpoint.geometry_topology_harm_present
    )
    focus_safe = tuple(
        local_insertion_influence_features(row)
        for row in safe_rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    )
    return InfluenceFeatureCohort(
        harmful=harmful,
        focus_safe=focus_safe,
        all_safe=tuple(
            local_insertion_influence_features(row) for row in safe_rows
        ),
    )


def evaluate_conservative_influence_calibration(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    calibration_a_seed: int = CALIBRATION_A_SEED,
    calibration_b_seed: int = CALIBRATION_B_SEED,
    final_held_out_seed: int = FINAL_HELD_OUT_SEED,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
    influence_config: LocalInsertionInfluenceConfig | None = None,
) -> ConservativeInfluenceCalibrationResult:
    """Calibrate on two cohorts and conditionally open the final held-out panel."""

    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_influence = (
        LocalInsertionInfluenceConfig()
        if influence_config is None
        else influence_config
    )
    seeds = (calibration_a_seed, calibration_b_seed, final_held_out_seed)
    if repeats < 1 or not selected_counts or not selected_stresses:
        raise ValueError("counts/stresses must be non-empty and repeats positive")
    if len(set(seeds)) != 3:
        raise ValueError("all calibration and held-out seeds must differ")
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and repeats == 8
        and reference_count == 2048
        and surface_sample_count == 256
        and seeds
        == (CALIBRATION_A_SEED, CALIBRATION_B_SEED, FINAL_HELD_OUT_SEED)
    )

    common = {
        "point_counts": selected_counts,
        "stresses": selected_stresses,
        "reference_count": reference_count,
        "repeats": repeats,
        "surface_sample_count": surface_sample_count,
        "base_gate_config": base_gate_config,
        "shared_trend_config": shared_trend_config,
        "influence_config": selected_influence,
    }
    calibration_a_raw = evaluate_local_insertion_influence_raw_panel(
        seed=calibration_a_seed,
        **common,
    )
    calibration_b_raw = evaluate_local_insertion_influence_raw_panel(
        seed=calibration_b_seed,
        **common,
    )
    rectangle = calibrate_dual_cohort_rectangle(
        _feature_cohort(calibration_a_raw),
        _feature_cohort(calibration_b_raw),
    )
    calibration_a = materialize_local_insertion_influence_panel(
        calibration_a_raw,
        panel_role="calibration_a",
        seed=calibration_a_seed,
        rectangle=rectangle,
        full_protocol=full_protocol,
    )
    calibration_b = materialize_local_insertion_influence_panel(
        calibration_b_raw,
        panel_role="calibration_b",
        seed=calibration_b_seed,
        rectangle=rectangle,
        full_protocol=full_protocol,
    )
    calibration_passed = bool(
        calibration_a.panel_gate_passed and calibration_b.panel_gate_passed
    )

    final_held_out: LocalInsertionInfluencePanelResult | None = None
    if calibration_passed:
        final_raw = evaluate_local_insertion_influence_raw_panel(
            seed=final_held_out_seed,
            **common,
        )
        final_held_out = materialize_local_insertion_influence_panel(
            final_raw,
            panel_role="final_held_out",
            seed=final_held_out_seed,
            rectangle=rectangle,
            full_protocol=full_protocol,
        )

    supported = bool(
        calibration_passed
        and final_held_out is not None
        and final_held_out.panel_gate_passed
    )
    return ConservativeInfluenceCalibrationResult(
        artifact_schema="pftf_alpha_conservative_influence_phase13/v1",
        role="dual_cohort_worst_retention_influence_calibration",
        information_boundary=(
            "route uses observed coordinates and inferred layers only; stress, "
            "source labels, and clean references are evaluation-only"
        ),
        frozen_predecessor="phase12_seeds_21100804_21200804_negative",
        calibration_a_seed=calibration_a_seed,
        calibration_b_seed=calibration_b_seed,
        final_held_out_seed=final_held_out_seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        influence_config=selected_influence,
        rectangle_selection_rule=(
            "zero harm in both calibration cohorts; maximize worst focus "
            "retention, total focus count, worst all-safe retention, total "
            "all-safe count, then peak and support thresholds"
        ),
        selected_rectangle=rectangle,
        calibration_a=calibration_a,
        calibration_b=calibration_b,
        final_held_out=final_held_out,
        phase13_supported=supported,
        trimmed_reconstruction_supported=False,
        real_scan_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--calibration-a-seed", type=int, default=CALIBRATION_A_SEED)
    parser.add_argument("--calibration-b-seed", type=int, default=CALIBRATION_B_SEED)
    parser.add_argument(
        "--final-held-out-seed",
        type=int,
        default=FINAL_HELD_OUT_SEED,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_conservative_influence_calibration(
        reference_count=args.reference,
        repeats=args.repeats,
        calibration_a_seed=args.calibration_a_seed,
        calibration_b_seed=args.calibration_b_seed,
        final_held_out_seed=args.final_held_out_seed,
        surface_sample_count=args.surface_samples,
    )
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
