"""Split-cohort matched-guard score and cutoff audit for Phase 24."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .matched_guard_signature import (
    GUARD_PROFILE_SPECS,
    TAIL_RATIO_FLOOR,
    MatchedGuardModel,
    MatchedGuardPanel,
    MatchedGuardSignature,
    _materialize_panel,
    fit_matched_guard_model,
    matched_guard_signature,
    score_matched_guard_signature,
)
from .matched_pair_consistency import MatchedPairConfig
from .matched_pair_stress import (
    MatchedPairStressConfig,
    MatchedPairStressSpec,
    _raw_panel,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .sensor_stress import DEFAULT_POINT_COUNTS, DEFAULT_STRESSES, SensorStress
from .shared_trend_inference import SharedTrendConfig

SCORE_FIT_SEED = 24900804
CUTOFF_CALIBRATION_SEED = 25000804
VALIDATION_A_SEED = 25100804
VALIDATION_B_SEED = 25200804
FINAL_HELD_OUT_SEED = 25300804
FORBIDDEN_PRIOR_SEEDS = frozenset(
    index * 100000 + 804 for index in range(233, 249)
)
CONSERVATIVE_GAP_FRACTION = 0.25


@dataclass(frozen=True)
class SplitCohortCutoffCalibration:
    seed: int
    candidate_case_count: int
    harmful_case_count: int
    focus_safe_case_count: int
    minimum_harmful_score: float
    maximum_focus_safe_score: float
    separation_gap: float
    conservative_gap_fraction: float
    rejection_cutoff: float
    rejected_harmful_case_count: int
    retained_harmful_case_count: int
    rejected_focus_safe_case_count: int
    retained_focus_safe_case_count: int
    calibration_valid: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SplitCohortGuardCalibrationResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    score_fit_seed: int
    cutoff_calibration_seed: int
    validation_a_seed: int
    validation_b_seed: int
    final_held_out_seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    matched_pair_config: MatchedPairConfig
    stress_config: MatchedPairStressConfig
    profile_specs: tuple[MatchedPairStressSpec, ...]
    tail_ratio_floor: float
    conservative_gap_fraction: float
    model: MatchedGuardModel
    cutoff_calibration: SplitCohortCutoffCalibration
    score_fit: MatchedGuardPanel
    cutoff_calibration_panel: MatchedGuardPanel
    prevalidation_gate_passed: bool
    validation_a: MatchedGuardPanel | None
    validation_b: MatchedGuardPanel | None
    final_held_out: MatchedGuardPanel | None
    phase24_supported: bool
    split_cohort_guard_calibration_synthetic_supported: bool
    real_correspondence_supported: bool
    real_paired_scan_supported: bool
    trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_predecessor": self.frozen_predecessor,
            "score_fit_seed": self.score_fit_seed,
            "cutoff_calibration_seed": self.cutoff_calibration_seed,
            "validation_a_seed": self.validation_a_seed,
            "validation_b_seed": self.validation_b_seed,
            "final_held_out_seed": self.final_held_out_seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "matched_pair_config": asdict(self.matched_pair_config),
            "stress_config": asdict(self.stress_config),
            "profile_specs": [
                {**asdict(spec), "profile": spec.profile.value}
                for spec in self.profile_specs
            ],
            "tail_ratio_floor": self.tail_ratio_floor,
            "conservative_gap_fraction": self.conservative_gap_fraction,
            "model": self.model.to_dict(),
            "cutoff_calibration": self.cutoff_calibration.to_dict(),
            "score_fit": self.score_fit.to_dict(),
            "cutoff_calibration_panel": self.cutoff_calibration_panel.to_dict(),
            "prevalidation_gate_passed": self.prevalidation_gate_passed,
            "validation_a": (
                None if self.validation_a is None else self.validation_a.to_dict()
            ),
            "validation_b": (
                None if self.validation_b is None else self.validation_b.to_dict()
            ),
            "final_held_out": (
                None
                if self.final_held_out is None
                else self.final_held_out.to_dict()
            ),
            "phase24_supported": self.phase24_supported,
            "split_cohort_guard_calibration_synthetic_supported": (
                self.split_cohort_guard_calibration_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def calibrate_split_cohort_cutoff(
    score_fit_model: MatchedGuardModel,
    *,
    score_fit_signatures: Sequence[MatchedGuardSignature],
    score_fit_harmful_labels: Sequence[bool],
    calibration_signatures: Sequence[MatchedGuardSignature],
    calibration_harmful_labels: Sequence[bool],
    calibration_focus_safe_labels: Sequence[bool],
    calibration_seed: int,
    gap_fraction: float = CONSERVATIVE_GAP_FRACTION,
) -> tuple[MatchedGuardModel, SplitCohortCutoffCalibration]:
    if not 0.0 < gap_fraction < 1.0:
        raise ValueError("gap_fraction must lie strictly between zero and one")
    fit_rows = tuple(score_fit_signatures)
    fit_harmful = np.asarray(tuple(score_fit_harmful_labels), dtype=np.bool_)
    calibration_rows = tuple(calibration_signatures)
    calibration_harmful = np.asarray(
        tuple(calibration_harmful_labels),
        dtype=np.bool_,
    )
    calibration_focus = np.asarray(
        tuple(calibration_focus_safe_labels),
        dtype=np.bool_,
    )
    if fit_harmful.shape != (len(fit_rows),):
        raise ValueError("score-fit signatures and labels must align")
    if calibration_harmful.shape != (len(calibration_rows),):
        raise ValueError("calibration signatures and harmful labels must align")
    if calibration_focus.shape != (len(calibration_rows),):
        raise ValueError("calibration signatures and focus labels must align")
    if np.any(calibration_harmful & calibration_focus):
        raise ValueError("harmful and focus-safe calibration labels must be disjoint")

    fit_scores = np.asarray(
        [score_matched_guard_signature(score_fit_model, row) for row in fit_rows],
        dtype=np.float64,
    )
    calibration_scores = np.asarray(
        [
            score_matched_guard_signature(score_fit_model, row)
            for row in calibration_rows
        ],
        dtype=np.float64,
    )
    harmful_count = int(np.sum(calibration_harmful))
    focus_count = int(np.sum(calibration_focus))
    if score_fit_model.calibration_valid and harmful_count and focus_count:
        minimum_harmful = float(
            np.min(calibration_scores[calibration_harmful])
        )
        maximum_focus = float(np.max(calibration_scores[calibration_focus]))
        separation_gap = minimum_harmful - maximum_focus
        valid = bool(
            math.isfinite(minimum_harmful)
            and math.isfinite(maximum_focus)
            and separation_gap > 0.0
        )
    else:
        minimum_harmful = np.finfo(float).max
        maximum_focus = -np.finfo(float).max
        separation_gap = 0.0
        valid = False
    cutoff = (
        maximum_focus + gap_fraction * separation_gap
        if valid
        else -np.finfo(float).max
    )
    rejected_calibration = (
        calibration_scores >= cutoff
        if valid
        else np.ones(len(calibration_rows), dtype=np.bool_)
    )
    rejected_fit = (
        fit_scores >= cutoff
        if valid
        else np.ones(len(fit_rows), dtype=np.bool_)
    )
    fit_harmful_count = int(np.sum(fit_harmful))
    fit_safe = ~fit_harmful
    rejected_fit_harmful = int(np.sum(rejected_fit & fit_harmful))
    rejected_fit_safe = int(np.sum(rejected_fit & fit_safe))
    model = replace(
        score_fit_model,
        rejection_cutoff=float(cutoff),
        calibration_valid=valid,
        training_case_count=len(fit_rows),
        training_harmful_case_count=fit_harmful_count,
        training_safe_case_count=len(fit_rows) - fit_harmful_count,
        rejected_training_harmful_case_count=rejected_fit_harmful,
        retained_training_harmful_case_count=(
            fit_harmful_count - rejected_fit_harmful
        ),
        rejected_training_safe_case_count=rejected_fit_safe,
        retained_training_safe_case_count=(
            len(fit_rows) - fit_harmful_count - rejected_fit_safe
        ),
    )
    rejected_harmful = int(
        np.sum(rejected_calibration & calibration_harmful)
    )
    rejected_focus = int(np.sum(rejected_calibration & calibration_focus))
    summary = SplitCohortCutoffCalibration(
        seed=calibration_seed,
        candidate_case_count=len(calibration_rows),
        harmful_case_count=harmful_count,
        focus_safe_case_count=focus_count,
        minimum_harmful_score=float(minimum_harmful),
        maximum_focus_safe_score=float(maximum_focus),
        separation_gap=float(separation_gap),
        conservative_gap_fraction=float(gap_fraction),
        rejection_cutoff=float(cutoff),
        rejected_harmful_case_count=rejected_harmful,
        retained_harmful_case_count=harmful_count - rejected_harmful,
        rejected_focus_safe_case_count=rejected_focus,
        retained_focus_safe_case_count=focus_count - rejected_focus,
        calibration_valid=valid,
    )
    return model, summary


def evaluate_split_cohort_guard_calibration(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    score_fit_seed: int = SCORE_FIT_SEED,
    cutoff_calibration_seed: int = CUTOFF_CALIBRATION_SEED,
    validation_a_seed: int = VALIDATION_A_SEED,
    validation_b_seed: int = VALIDATION_B_SEED,
    final_held_out_seed: int = FINAL_HELD_OUT_SEED,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
    matched_pair_config: MatchedPairConfig | None = None,
    stress_config: MatchedPairStressConfig | None = None,
    profile_specs: Sequence[MatchedPairStressSpec] = GUARD_PROFILE_SPECS,
) -> SplitCohortGuardCalibrationResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_matched = (
        MatchedPairConfig() if matched_pair_config is None else matched_pair_config
    )
    selected_stress = (
        MatchedPairStressConfig() if stress_config is None else stress_config
    )
    selected_profiles = tuple(profile_specs)
    seeds = (
        score_fit_seed,
        cutoff_calibration_seed,
        validation_a_seed,
        validation_b_seed,
        final_held_out_seed,
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "score-fit, calibration, validation, and final seeds must differ"
        )
    if any(seed in FORBIDDEN_PRIOR_SEEDS for seed in seeds):
        raise ValueError("Phase-20--23 opened or reserved seeds must not be reused")
    if reference_count < 1 or repeats < 1 or surface_sample_count < 1:
        raise ValueError("panel sizes must be positive")
    if not selected_counts or min(selected_counts) < 4:
        raise ValueError("point_counts must contain values of at least four")
    if not selected_stresses or not selected_profiles:
        raise ValueError("stresses and profile_specs must not be empty")
    if len({spec.profile for spec in selected_profiles}) != len(selected_profiles):
        raise ValueError("stress profiles must be unique")
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and selected_profiles == GUARD_PROFILE_SPECS
        and repeats >= 8
        and reference_count >= 2048
        and surface_sample_count >= 256
        and seeds
        == (
            SCORE_FIT_SEED,
            CUTOFF_CALIBRATION_SEED,
            VALIDATION_A_SEED,
            VALIDATION_B_SEED,
            FINAL_HELD_OUT_SEED,
        )
    )
    common = {
        "point_counts": selected_counts,
        "stresses": selected_stresses,
        "reference_count": reference_count,
        "repeats": repeats,
        "surface_sample_count": surface_sample_count,
        "base_gate_config": base_gate_config,
        "shared_trend_config": shared_trend_config,
        "matched_pair_config": selected_matched,
        "stress_config": selected_stress,
        "profile_specs": selected_profiles,
    }
    score_fit_raw = _raw_panel(seed=score_fit_seed, **common)
    score_fit_candidates = tuple(
        row
        for row in score_fit_raw
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
    )
    score_fit_signatures = tuple(
        matched_guard_signature(row) for row in score_fit_candidates
    )
    score_fit_harmful = tuple(
        row.endpoint.geometry_topology_harm_present
        for row in score_fit_candidates
    )
    score_fit_model = fit_matched_guard_model(
        score_fit_signatures,
        score_fit_harmful,
    )

    calibration_raw = _raw_panel(seed=cutoff_calibration_seed, **common)
    calibration_candidates = tuple(
        row
        for row in calibration_raw
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
    )
    calibration_signatures = tuple(
        matched_guard_signature(row) for row in calibration_candidates
    )
    calibration_harmful = tuple(
        row.endpoint.geometry_topology_harm_present
        for row in calibration_candidates
    )
    calibration_focus_safe = tuple(
        row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
        and not row.endpoint.geometry_topology_harm_present
        for row in calibration_candidates
    )
    model, cutoff_calibration = calibrate_split_cohort_cutoff(
        score_fit_model,
        score_fit_signatures=score_fit_signatures,
        score_fit_harmful_labels=score_fit_harmful,
        calibration_signatures=calibration_signatures,
        calibration_harmful_labels=calibration_harmful,
        calibration_focus_safe_labels=calibration_focus_safe,
        calibration_seed=cutoff_calibration_seed,
    )
    panel_common = {
        "model": model,
        "profile_specs": selected_profiles,
        "full_protocol": full_protocol,
    }
    score_fit = _materialize_panel(
        score_fit_raw,
        panel_role="score_fit",
        seed=score_fit_seed,
        **panel_common,
    )
    cutoff_panel = _materialize_panel(
        calibration_raw,
        panel_role="cutoff_calibration",
        seed=cutoff_calibration_seed,
        **panel_common,
    )
    prevalidation_passed = bool(
        cutoff_calibration.calibration_valid
        and score_fit.panel_gate_passed
        and cutoff_panel.panel_gate_passed
    )
    validation_a: MatchedGuardPanel | None = None
    validation_b: MatchedGuardPanel | None = None
    final_panel: MatchedGuardPanel | None = None
    if prevalidation_passed:
        validation_a = _materialize_panel(
            _raw_panel(seed=validation_a_seed, **common),
            panel_role="validation_a",
            seed=validation_a_seed,
            **panel_common,
        )
        if validation_a.panel_gate_passed:
            validation_b = _materialize_panel(
                _raw_panel(seed=validation_b_seed, **common),
                panel_role="validation_b",
                seed=validation_b_seed,
                **panel_common,
            )
        if validation_b is not None and validation_b.panel_gate_passed:
            final_panel = _materialize_panel(
                _raw_panel(seed=final_held_out_seed, **common),
                panel_role="final_held_out",
                seed=final_held_out_seed,
                **panel_common,
            )
    supported = bool(final_panel is not None and final_panel.panel_gate_passed)
    return SplitCohortGuardCalibrationResult(
        artifact_schema="pftf_alpha_split_cohort_guard_calibration_phase24/v1",
        role="synthetic_supervised_split_cohort_guard_calibration_audit",
        information_boundary=(
            "route uses exact presented pairs, observed displacement-tail "
            "signature, upstream candidate decision, score coefficients fitted "
            "only on the score-fit cohort, and a cutoff frozen only on the "
            "calibration cohort; endpoints are fit/calibration/evaluation-only"
        ),
        frozen_predecessor="phase23_single_cohort_cutoff_margin_transfer_failure",
        score_fit_seed=score_fit_seed,
        cutoff_calibration_seed=cutoff_calibration_seed,
        validation_a_seed=validation_a_seed,
        validation_b_seed=validation_b_seed,
        final_held_out_seed=final_held_out_seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        matched_pair_config=selected_matched,
        stress_config=selected_stress,
        profile_specs=selected_profiles,
        tail_ratio_floor=TAIL_RATIO_FLOOR,
        conservative_gap_fraction=CONSERVATIVE_GAP_FRACTION,
        model=model,
        cutoff_calibration=cutoff_calibration,
        score_fit=score_fit,
        cutoff_calibration_panel=cutoff_panel,
        prevalidation_gate_passed=prevalidation_passed,
        validation_a=validation_a,
        validation_b=validation_b,
        final_held_out=final_panel,
        phase24_supported=supported,
        split_cohort_guard_calibration_synthetic_supported=supported,
        real_correspondence_supported=False,
        real_paired_scan_supported=False,
        trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_split_cohort_guard_calibration(
        reference_count=args.reference,
        repeats=args.repeats,
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
