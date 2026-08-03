"""Strict focus-envelope cutoff transfer audit for Phase 27."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .frozen_partition_reconstruction import (
    FrozenPartitionPanel,
    _materialize_panel,
    _panel_case_seeds,
    _routed_endpoint,
)
from .matched_guard_signature import (
    GUARD_PROFILE_SPECS,
    TAIL_RATIO_FLOOR,
    MatchedGuardModel,
    MatchedGuardSignature,
    fit_matched_guard_model,
    matched_guard_signature,
    score_matched_guard_signature,
)
from .matched_pair_consistency import MatchedPairConfig
from .matched_pair_stress import (
    MatchedPairStressConfig,
    MatchedPairStressRawCase,
    MatchedPairStressSpec,
    _raw_panel,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .sensor_stress import DEFAULT_POINT_COUNTS, DEFAULT_STRESSES, SensorStress
from .shared_trend_inference import SharedTrendConfig
from .split_cohort_guard_calibration import (
    CONSERVATIVE_GAP_FRACTION,
    SplitCohortCutoffCalibration,
    calibrate_split_cohort_cutoff,
)

DESIGN_SCORE_FIT_SEED = 24900804
DESIGN_CUTOFF_CALIBRATION_SEED = 25000804
CUTOFF_DESIGN_A_SEED = 27500804
CUTOFF_DESIGN_B_SEED = 27600804
DIAGNOSIS_ONLY_FINAL_SEED = 27700804
VALIDATION_A_SEED = 27800804
VALIDATION_B_SEED = 27900804
FINAL_HELD_OUT_SEED = 28000804
PRIOR_STANDARD_BASE_SEEDS = tuple(
    index * 100000 + 804 for index in range(203, 257)
)
PRIOR_PHASE26_BASE_SEEDS = (
    CUTOFF_DESIGN_A_SEED,
    CUTOFF_DESIGN_B_SEED,
    DIAGNOSIS_ONLY_FINAL_SEED,
)
EXPECTED_PREDECESSOR_HARMFUL_COUNT = 164
EXPECTED_PREDECESSOR_FOCUS_COUNT = 126
EXPECTED_PREDECESSOR_CUTOFF = 0.19784302031484602
EXPECTED_FOCUS_COUNT = 255
EXPECTED_HARMFUL_COUNT = 330
EXPECTED_MAXIMUM_FOCUS_SCORE = 0.18181536333942855
EXPECTED_MINIMUM_HARMFUL_SCORE = 0.28460336155814553
EXPECTED_SEPARATION_GAP = 0.10278799821871698
EXPECTED_FOCUS_ENVELOPE_CUTOFF = 0.18181536333942858
REPRODUCTION_TOLERANCE = 1.0e-15


@dataclass(frozen=True)
class Phase27CaseSeedAudit:
    case_seed_formula: str
    prior_base_seed_ranges: tuple[str, ...]
    panel_case_count: int
    validation_a_prior_overlap_count: int
    validation_b_prior_overlap_count: int
    final_prior_overlap_count: int
    validation_a_b_overlap_count: int
    validation_a_final_overlap_count: int
    validation_b_final_overlap_count: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FocusEnvelopeCalibration:
    design_a_seed: int
    design_b_seed: int
    focus_safe_case_count: int
    routed_harmful_case_count: int
    maximum_focus_safe_score: float | None
    minimum_routed_harmful_score: float | None
    separation_gap: float | None
    rejection_cutoff: float
    binary64_increment: float | None
    calibration_valid: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FocusEnvelopeCutoffResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    design_score_fit_seed: int
    design_cutoff_calibration_seed: int
    cutoff_design_a_seed: int
    cutoff_design_b_seed: int
    diagnosis_only_final_seed: int
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
    predecessor_conservative_gap_fraction: float
    case_seed_disjointness: Phase27CaseSeedAudit
    predecessor_model: MatchedGuardModel
    predecessor_cutoff_calibration: SplitCohortCutoffCalibration
    focus_envelope_calibration: FocusEnvelopeCalibration
    model: MatchedGuardModel
    design_score_fit: FrozenPartitionPanel
    design_cutoff_calibration: FrozenPartitionPanel
    cutoff_design_a: FrozenPartitionPanel
    cutoff_design_b: FrozenPartitionPanel
    design_reproduced: bool
    design_gate_passed: bool
    validation_a: FrozenPartitionPanel | None
    validation_b: FrozenPartitionPanel | None
    final_held_out: FrozenPartitionPanel | None
    phase27_supported: bool
    focus_envelope_cutoff_synthetic_supported: bool
    real_correspondence_supported: bool
    real_paired_scan_supported: bool
    real_trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_predecessor": self.frozen_predecessor,
            "design_score_fit_seed": self.design_score_fit_seed,
            "design_cutoff_calibration_seed": self.design_cutoff_calibration_seed,
            "cutoff_design_a_seed": self.cutoff_design_a_seed,
            "cutoff_design_b_seed": self.cutoff_design_b_seed,
            "diagnosis_only_final_seed": self.diagnosis_only_final_seed,
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
            "predecessor_conservative_gap_fraction": (
                self.predecessor_conservative_gap_fraction
            ),
            "case_seed_disjointness": self.case_seed_disjointness.to_dict(),
            "predecessor_model": self.predecessor_model.to_dict(),
            "predecessor_cutoff_calibration": (
                self.predecessor_cutoff_calibration.to_dict()
            ),
            "focus_envelope_calibration": (
                self.focus_envelope_calibration.to_dict()
            ),
            "model": self.model.to_dict(),
            "design_score_fit": self.design_score_fit.to_dict(),
            "design_cutoff_calibration": (
                self.design_cutoff_calibration.to_dict()
            ),
            "cutoff_design_a": self.cutoff_design_a.to_dict(),
            "cutoff_design_b": self.cutoff_design_b.to_dict(),
            "design_reproduced": self.design_reproduced,
            "design_gate_passed": self.design_gate_passed,
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
            "phase27_supported": self.phase27_supported,
            "focus_envelope_cutoff_synthetic_supported": (
                self.focus_envelope_cutoff_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "real_trimmed_reconstruction_supported": (
                self.real_trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def audit_phase27_case_seed_disjointness(
    validation_a_seed: int,
    validation_b_seed: int,
    final_held_out_seed: int,
    *,
    point_counts: tuple[int, ...] = DEFAULT_POINT_COUNTS,
    stresses: tuple[SensorStress, ...] = DEFAULT_STRESSES,
    repeats: int = 8,
) -> Phase27CaseSeedAudit:
    prior_bases = PRIOR_STANDARD_BASE_SEEDS + PRIOR_PHASE26_BASE_SEEDS
    prior = frozenset().union(
        *(
            _panel_case_seeds(
                seed,
                point_counts=DEFAULT_POINT_COUNTS,
                stresses=DEFAULT_STRESSES,
                repeats=8,
            )
            for seed in prior_bases
        )
    )
    panels = tuple(
        _panel_case_seeds(
            seed,
            point_counts=point_counts,
            stresses=stresses,
            repeats=repeats,
        )
        for seed in (validation_a_seed, validation_b_seed, final_held_out_seed)
    )
    prior_overlaps = tuple(len(panel & prior) for panel in panels)
    mutual_overlaps = (
        len(panels[0] & panels[1]),
        len(panels[0] & panels[2]),
        len(panels[1] & panels[2]),
    )
    return Phase27CaseSeedAudit(
        case_seed_formula=(
            "base + count_index*1000003 + stress_index*100003 + repeat*10007"
        ),
        prior_base_seed_ranges=(
            "20300804--25600804",
            "27500804--27700804",
        ),
        panel_case_count=len(panels[0]),
        validation_a_prior_overlap_count=prior_overlaps[0],
        validation_b_prior_overlap_count=prior_overlaps[1],
        final_prior_overlap_count=prior_overlaps[2],
        validation_a_b_overlap_count=mutual_overlaps[0],
        validation_a_final_overlap_count=mutual_overlaps[1],
        validation_b_final_overlap_count=mutual_overlaps[2],
        passed=not any(prior_overlaps + mutual_overlaps),
    )


def _candidate_rows(
    rows: Sequence[MatchedPairStressRawCase],
) -> tuple[MatchedPairStressRawCase, ...]:
    return tuple(
        row
        for row in rows
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
    )


def calibrate_focus_envelope_cutoff(
    predecessor_model: MatchedGuardModel,
    *,
    score_fit_signatures: Sequence[MatchedGuardSignature],
    score_fit_harmful_labels: Sequence[bool],
    design_a_rows: Sequence[MatchedPairStressRawCase],
    design_b_rows: Sequence[MatchedPairStressRawCase],
) -> tuple[MatchedGuardModel, FocusEnvelopeCalibration]:
    design_rows = _candidate_rows((*design_a_rows, *design_b_rows))
    focus_scores = tuple(
        score_matched_guard_signature(
            predecessor_model,
            matched_guard_signature(row),
        )
        for row in design_rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
        and not row.endpoint.geometry_topology_harm_present
    )
    harmful_scores = tuple(
        score_matched_guard_signature(
            predecessor_model,
            matched_guard_signature(row),
        )
        for row in design_rows
        if row.stress.is_outlier_stress
        and _routed_endpoint(row).geometry_topology_harm_present
    )
    maximum_focus = max(focus_scores) if focus_scores else None
    minimum_harmful = min(harmful_scores) if harmful_scores else None
    valid = bool(
        predecessor_model.calibration_valid
        and maximum_focus is not None
        and minimum_harmful is not None
        and minimum_harmful > maximum_focus
    )
    cutoff = (
        math.nextafter(maximum_focus, math.inf)
        if valid and maximum_focus is not None
        else -sys.float_info.max
    )
    training_scores = tuple(
        score_matched_guard_signature(predecessor_model, signature)
        for signature in score_fit_signatures
    )
    harmful_labels = tuple(bool(value) for value in score_fit_harmful_labels)
    if len(training_scores) != len(harmful_labels):
        raise ValueError("score-fit signatures and labels must align")
    rejected_harmful = sum(
        harmful and score >= cutoff
        for score, harmful in zip(training_scores, harmful_labels, strict=True)
    )
    rejected_safe = sum(
        not harmful and score >= cutoff
        for score, harmful in zip(training_scores, harmful_labels, strict=True)
    )
    harmful_count = sum(harmful_labels)
    safe_count = len(harmful_labels) - harmful_count
    model = replace(
        predecessor_model,
        rejection_cutoff=cutoff,
        calibration_valid=valid,
        rejected_training_harmful_case_count=rejected_harmful,
        retained_training_harmful_case_count=harmful_count - rejected_harmful,
        rejected_training_safe_case_count=rejected_safe,
        retained_training_safe_case_count=safe_count - rejected_safe,
    )
    return model, FocusEnvelopeCalibration(
        design_a_seed=CUTOFF_DESIGN_A_SEED,
        design_b_seed=CUTOFF_DESIGN_B_SEED,
        focus_safe_case_count=len(focus_scores),
        routed_harmful_case_count=len(harmful_scores),
        maximum_focus_safe_score=maximum_focus,
        minimum_routed_harmful_score=minimum_harmful,
        separation_gap=(
            None
            if maximum_focus is None or minimum_harmful is None
            else minimum_harmful - maximum_focus
        ),
        rejection_cutoff=cutoff,
        binary64_increment=(
            None if maximum_focus is None else cutoff - maximum_focus
        ),
        calibration_valid=valid,
    )


def evaluate_focus_envelope_cutoff(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    validation_a_seed: int = VALIDATION_A_SEED,
    validation_b_seed: int = VALIDATION_B_SEED,
    final_held_out_seed: int = FINAL_HELD_OUT_SEED,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
    matched_pair_config: MatchedPairConfig | None = None,
    stress_config: MatchedPairStressConfig | None = None,
    profile_specs: Sequence[MatchedPairStressSpec] = GUARD_PROFILE_SPECS,
) -> FocusEnvelopeCutoffResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_matched = (
        MatchedPairConfig() if matched_pair_config is None else matched_pair_config
    )
    selected_stress = (
        MatchedPairStressConfig() if stress_config is None else stress_config
    )
    selected_profiles = tuple(profile_specs)
    fresh_seeds = (validation_a_seed, validation_b_seed, final_held_out_seed)
    if len(set(fresh_seeds)) != len(fresh_seeds):
        raise ValueError("validation and final base seeds must differ")
    if reference_count < 1 or repeats < 1 or surface_sample_count < 1:
        raise ValueError("panel sizes must be positive")
    if not selected_counts or min(selected_counts) < 4:
        raise ValueError("point_counts must contain values of at least four")
    if not selected_stresses or not selected_profiles:
        raise ValueError("stresses and profile_specs must not be empty")
    if len({spec.profile for spec in selected_profiles}) != len(selected_profiles):
        raise ValueError("stress profiles must be unique")
    seed_audit = audit_phase27_case_seed_disjointness(
        *fresh_seeds,
        point_counts=selected_counts,
        stresses=selected_stresses,
        repeats=repeats,
    )
    if not seed_audit.passed:
        raise ValueError(
            "validation/final case seeds must be mutually disjoint and disjoint "
            "from prior standard and Phase-26 panels"
        )
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and selected_profiles == GUARD_PROFILE_SPECS
        and repeats >= 8
        and reference_count >= 2048
        and surface_sample_count >= 256
        and fresh_seeds
        == (VALIDATION_A_SEED, VALIDATION_B_SEED, FINAL_HELD_OUT_SEED)
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
    design_fit_raw = _raw_panel(seed=DESIGN_SCORE_FIT_SEED, **common)
    design_fit_candidates = _candidate_rows(design_fit_raw)
    design_fit_signatures = tuple(
        matched_guard_signature(row) for row in design_fit_candidates
    )
    design_fit_harmful = tuple(
        row.endpoint.geometry_topology_harm_present
        for row in design_fit_candidates
    )
    score_fit_model = fit_matched_guard_model(
        design_fit_signatures,
        design_fit_harmful,
    )

    design_calibration_raw = _raw_panel(
        seed=DESIGN_CUTOFF_CALIBRATION_SEED,
        **common,
    )
    design_calibration_candidates = _candidate_rows(design_calibration_raw)
    predecessor_model, predecessor_calibration = calibrate_split_cohort_cutoff(
        score_fit_model,
        score_fit_signatures=design_fit_signatures,
        score_fit_harmful_labels=design_fit_harmful,
        calibration_signatures=tuple(
            matched_guard_signature(row) for row in design_calibration_candidates
        ),
        calibration_harmful_labels=tuple(
            _routed_endpoint(row).geometry_topology_harm_present
            for row in design_calibration_candidates
        ),
        calibration_focus_safe_labels=tuple(
            row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
            and not row.endpoint.geometry_topology_harm_present
            for row in design_calibration_candidates
        ),
        calibration_seed=DESIGN_CUTOFF_CALIBRATION_SEED,
    )
    cutoff_design_a_raw = _raw_panel(seed=CUTOFF_DESIGN_A_SEED, **common)
    cutoff_design_b_raw = _raw_panel(seed=CUTOFF_DESIGN_B_SEED, **common)
    model, focus_calibration = calibrate_focus_envelope_cutoff(
        predecessor_model,
        score_fit_signatures=design_fit_signatures,
        score_fit_harmful_labels=design_fit_harmful,
        design_a_rows=cutoff_design_a_raw,
        design_b_rows=cutoff_design_b_raw,
    )
    panel_common = {
        "model": model,
        "profile_specs": selected_profiles,
        "full_protocol": full_protocol,
    }
    design_score_fit = _materialize_panel(
        design_fit_raw,
        panel_role="design_score_fit",
        seed=DESIGN_SCORE_FIT_SEED,
        **panel_common,
    )
    design_cutoff_calibration = _materialize_panel(
        design_calibration_raw,
        panel_role="design_cutoff_calibration",
        seed=DESIGN_CUTOFF_CALIBRATION_SEED,
        **panel_common,
    )
    cutoff_design_a = _materialize_panel(
        cutoff_design_a_raw,
        panel_role="cutoff_design_a",
        seed=CUTOFF_DESIGN_A_SEED,
        **panel_common,
    )
    cutoff_design_b = _materialize_panel(
        cutoff_design_b_raw,
        panel_role="cutoff_design_b",
        seed=CUTOFF_DESIGN_B_SEED,
        **panel_common,
    )
    reproduced_maximum_focus = (
        -sys.float_info.max
        if focus_calibration.maximum_focus_safe_score is None
        else focus_calibration.maximum_focus_safe_score
    )
    reproduced_minimum_harmful = (
        -sys.float_info.max
        if focus_calibration.minimum_routed_harmful_score is None
        else focus_calibration.minimum_routed_harmful_score
    )
    reproduced_gap = (
        -sys.float_info.max
        if focus_calibration.separation_gap is None
        else focus_calibration.separation_gap
    )
    design_reproduced = bool(
        predecessor_calibration.calibration_valid
        and predecessor_calibration.harmful_case_count
        == EXPECTED_PREDECESSOR_HARMFUL_COUNT
        and predecessor_calibration.focus_safe_case_count
        == EXPECTED_PREDECESSOR_FOCUS_COUNT
        and math.isclose(
            predecessor_calibration.rejection_cutoff,
            EXPECTED_PREDECESSOR_CUTOFF,
            rel_tol=0.0,
            abs_tol=REPRODUCTION_TOLERANCE,
        )
        and focus_calibration.calibration_valid
        and focus_calibration.focus_safe_case_count == EXPECTED_FOCUS_COUNT
        and focus_calibration.routed_harmful_case_count == EXPECTED_HARMFUL_COUNT
        and math.isclose(
            reproduced_maximum_focus,
            EXPECTED_MAXIMUM_FOCUS_SCORE,
            rel_tol=0.0,
            abs_tol=REPRODUCTION_TOLERANCE,
        )
        and math.isclose(
            reproduced_minimum_harmful,
            EXPECTED_MINIMUM_HARMFUL_SCORE,
            rel_tol=0.0,
            abs_tol=REPRODUCTION_TOLERANCE,
        )
        and math.isclose(
            reproduced_gap,
            EXPECTED_SEPARATION_GAP,
            rel_tol=0.0,
            abs_tol=REPRODUCTION_TOLERANCE,
        )
        and math.isclose(
            model.rejection_cutoff,
            EXPECTED_FOCUS_ENVELOPE_CUTOFF,
            rel_tol=0.0,
            abs_tol=REPRODUCTION_TOLERANCE,
        )
    )
    design_gate_passed = bool(
        design_reproduced
        and seed_audit.passed
        and design_score_fit.panel_gate_passed
        and design_cutoff_calibration.panel_gate_passed
        and cutoff_design_a.panel_gate_passed
        and cutoff_design_b.panel_gate_passed
    )

    validation_a: FrozenPartitionPanel | None = None
    validation_b: FrozenPartitionPanel | None = None
    final_panel: FrozenPartitionPanel | None = None
    if design_gate_passed:
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
    return FocusEnvelopeCutoffResult(
        artifact_schema="pftf_alpha_focus_envelope_cutoff_phase27/v1",
        role="synthetic_focus_envelope_cutoff_transfer_audit",
        information_boundary=(
            "route uses full primary coordinates and frozen observed partition, "
            "presented retained-pair IDs and repeat coordinates, upstream "
            "decision, and frozen score; truth-supervised focus and harm labels "
            "are design/evaluation-only"
        ),
        frozen_predecessor=(
            "phase26_positive_score_separation_but_final_cutoff_margin_failure"
        ),
        design_score_fit_seed=DESIGN_SCORE_FIT_SEED,
        design_cutoff_calibration_seed=DESIGN_CUTOFF_CALIBRATION_SEED,
        cutoff_design_a_seed=CUTOFF_DESIGN_A_SEED,
        cutoff_design_b_seed=CUTOFF_DESIGN_B_SEED,
        diagnosis_only_final_seed=DIAGNOSIS_ONLY_FINAL_SEED,
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
        predecessor_conservative_gap_fraction=CONSERVATIVE_GAP_FRACTION,
        case_seed_disjointness=seed_audit,
        predecessor_model=predecessor_model,
        predecessor_cutoff_calibration=predecessor_calibration,
        focus_envelope_calibration=focus_calibration,
        model=model,
        design_score_fit=design_score_fit,
        design_cutoff_calibration=design_cutoff_calibration,
        cutoff_design_a=cutoff_design_a,
        cutoff_design_b=cutoff_design_b,
        design_reproduced=design_reproduced,
        design_gate_passed=design_gate_passed,
        validation_a=validation_a,
        validation_b=validation_b,
        final_held_out=final_panel,
        phase27_supported=supported,
        focus_envelope_cutoff_synthetic_supported=supported,
        real_correspondence_supported=False,
        real_paired_scan_supported=False,
        real_trimmed_reconstruction_supported=False,
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
    result = evaluate_focus_envelope_cutoff(
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
