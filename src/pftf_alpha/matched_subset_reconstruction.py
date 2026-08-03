"""Matched-subset reconstruction plus frozen observed guard for Phase 25."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .local_surface_consensus import GeometryTopologyHarmEndpoint
from .matched_guard_signature import (
    GUARD_PROFILE_SPECS,
    TAIL_RATIO_FLOOR,
    MatchedGuardModel,
    MatchedGuardSignature,
    fit_matched_guard_model,
    matched_guard_signature,
    score_matched_guard_signature,
)
from .matched_pair_consistency import MatchedPairConfig, MatchedPairEvidence
from .matched_pair_stress import (
    MatchedPairStressConfig,
    MatchedPairStressProfile,
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
VALIDATION_A_SEED = 25400804
VALIDATION_B_SEED = 25500804
FINAL_HELD_OUT_SEED = 25600804
FORBIDDEN_FRESH_SEEDS = frozenset(
    index * 100000 + 804 for index in range(233, 254)
)
EXPECTED_DESIGN_HARMFUL_COUNT = 164
EXPECTED_DESIGN_FOCUS_COUNT = 126
EXPECTED_DESIGN_CUTOFF = 0.19784302031484602
DESIGN_REPRODUCTION_TOLERANCE = 1.0e-15


@dataclass(frozen=True)
class MatchedSubsetCaseResult:
    profile: MatchedPairStressProfile
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    perturbation_seed: int
    repeat_transient_outlier_count: int
    repeat_transient_outlier_index_sha256: str
    retained_pair_count: int
    missing_pair_count: int
    rotation_degrees: float
    rotation_axis: tuple[float, float, float]
    presented_pair_map_sha256: str
    matched_subset_applied: bool
    signature: MatchedGuardSignature
    model_score: float
    evidence: MatchedPairEvidence
    original_endpoint: GeometryTopologyHarmEndpoint
    routed_endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision
    guarded_decision: SamplingGateDecision
    unguarded_safe_accept: bool
    guarded_safe_accept: bool
    unguarded_harmful_outlier_false_safe: bool
    guarded_harmful_outlier_false_safe: bool
    unguarded_provenance_violation_accept: bool
    guarded_provenance_violation_accept: bool
    introduced_routed_endpoint_harm_accept: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        payload["stress"] = self.stress.value
        payload["signature"] = self.signature.to_dict()
        payload["unguarded_decision"] = self.unguarded_decision.value
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class MatchedSubsetProfileSummary:
    profile: MatchedPairStressProfile
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    all_stress_unguarded_safe_accept_count: int
    all_stress_guarded_safe_accept_count: int
    introduced_routed_endpoint_harm_accept_count: int
    profile_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        return payload


@dataclass(frozen=True)
class MatchedSubsetPanel:
    panel_role: str
    seed: int
    cases: tuple[MatchedSubsetCaseResult, ...]
    profile_summaries: tuple[MatchedSubsetProfileSummary, ...]
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    all_stress_unguarded_safe_accept_count: int
    all_stress_guarded_safe_accept_count: int
    matched_subset_case_count: int
    introduced_routed_endpoint_harm_accept_count: int
    full_protocol: bool
    panel_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_role": self.panel_role,
            "seed": self.seed,
            "cases": [case.to_dict() for case in self.cases],
            "profile_summaries": [
                summary.to_dict() for summary in self.profile_summaries
            ],
            "case_count": self.case_count,
            "unguarded_harmful_outlier_false_safe_count": (
                self.unguarded_harmful_outlier_false_safe_count
            ),
            "guarded_harmful_outlier_false_safe_count": (
                self.guarded_harmful_outlier_false_safe_count
            ),
            "focus_unguarded_safe_accept_count": (
                self.focus_unguarded_safe_accept_count
            ),
            "focus_guarded_safe_accept_count": (
                self.focus_guarded_safe_accept_count
            ),
            "focus_safe_accept_retention": self.focus_safe_accept_retention,
            "all_stress_unguarded_safe_accept_count": (
                self.all_stress_unguarded_safe_accept_count
            ),
            "all_stress_guarded_safe_accept_count": (
                self.all_stress_guarded_safe_accept_count
            ),
            "matched_subset_case_count": self.matched_subset_case_count,
            "introduced_routed_endpoint_harm_accept_count": (
                self.introduced_routed_endpoint_harm_accept_count
            ),
            "full_protocol": self.full_protocol,
            "panel_gate_passed": self.panel_gate_passed,
        }


@dataclass(frozen=True)
class MatchedSubsetReconstructionResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    design_score_fit_seed: int
    design_cutoff_calibration_seed: int
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
    design_score_fit: MatchedSubsetPanel
    design_cutoff_calibration: MatchedSubsetPanel
    design_reproduced: bool
    design_gate_passed: bool
    validation_a: MatchedSubsetPanel | None
    validation_b: MatchedSubsetPanel | None
    final_held_out: MatchedSubsetPanel | None
    phase25_supported: bool
    matched_subset_reconstruction_synthetic_supported: bool
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
            "design_cutoff_calibration_seed": (
                self.design_cutoff_calibration_seed
            ),
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
            "design_score_fit": self.design_score_fit.to_dict(),
            "design_cutoff_calibration": (
                self.design_cutoff_calibration.to_dict()
            ),
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
            "phase25_supported": self.phase25_supported,
            "matched_subset_reconstruction_synthetic_supported": (
                self.matched_subset_reconstruction_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "real_trimmed_reconstruction_supported": (
                self.real_trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def _routed_endpoint(
    raw: MatchedPairStressRawCase,
) -> GeometryTopologyHarmEndpoint:
    return raw.matched_subset_endpoint if raw.missing_pair_count else raw.endpoint


def _materialize_case(
    raw: MatchedPairStressRawCase,
    model: MatchedGuardModel,
) -> MatchedSubsetCaseResult:
    signature = matched_guard_signature(raw)
    score = score_matched_guard_signature(model, signature)
    unguarded_accept = raw.unguarded_decision is SamplingGateDecision.ACCEPT
    consistent = bool(model.calibration_valid and score < model.rejection_cutoff)
    guarded_decision = (
        SamplingGateDecision.UNSUPPORTED
        if unguarded_accept and not consistent
        else raw.unguarded_decision
    )
    guarded_accept = guarded_decision is SamplingGateDecision.ACCEPT
    routed_endpoint = _routed_endpoint(raw)
    original_harmful_outlier = bool(
        raw.stress.is_outlier_stress
        and raw.endpoint.geometry_topology_harm_present
    )
    routed_harmful_outlier = bool(
        raw.stress.is_outlier_stress
        and routed_endpoint.geometry_topology_harm_present
    )
    return MatchedSubsetCaseResult(
        profile=raw.profile,
        stress=raw.stress,
        point_count=raw.point_count,
        repeat=raw.repeat,
        seed=raw.seed,
        replicate_seed=raw.replicate_seed,
        perturbation_seed=raw.perturbation_seed,
        repeat_transient_outlier_count=raw.repeat_transient_outlier_count,
        repeat_transient_outlier_index_sha256=(
            raw.repeat_transient_outlier_index_sha256
        ),
        retained_pair_count=raw.retained_pair_count,
        missing_pair_count=raw.missing_pair_count,
        rotation_degrees=raw.rotation_degrees,
        rotation_axis=raw.rotation_axis,
        presented_pair_map_sha256=raw.presented_pair_map_sha256,
        matched_subset_applied=bool(raw.missing_pair_count),
        signature=signature,
        model_score=score,
        evidence=raw.evidence,
        original_endpoint=raw.endpoint,
        routed_endpoint=routed_endpoint,
        unguarded_decision=raw.unguarded_decision,
        guarded_decision=guarded_decision,
        unguarded_safe_accept=bool(
            unguarded_accept and not raw.endpoint.geometry_topology_harm_present
        ),
        guarded_safe_accept=bool(
            guarded_accept and not routed_endpoint.geometry_topology_harm_present
        ),
        unguarded_harmful_outlier_false_safe=bool(
            unguarded_accept and original_harmful_outlier
        ),
        guarded_harmful_outlier_false_safe=bool(
            guarded_accept and routed_harmful_outlier
        ),
        unguarded_provenance_violation_accept=bool(
            unguarded_accept and raw.endpoint.provenance_violation_present
        ),
        guarded_provenance_violation_accept=bool(
            guarded_accept and routed_endpoint.provenance_violation_present
        ),
        introduced_routed_endpoint_harm_accept=bool(
            guarded_accept
            and not raw.endpoint.geometry_topology_harm_present
            and routed_endpoint.geometry_topology_harm_present
        ),
    )


def _profile_summary(
    rows: Sequence[MatchedSubsetCaseResult],
    profile: MatchedPairStressProfile,
    *,
    full_protocol: bool,
    model_valid: bool,
) -> MatchedSubsetProfileSummary:
    selected = [row for row in rows if row.profile is profile]
    focus = [
        row
        for row in selected
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    retention = 0.0 if not unguarded_focus else guarded_focus / unguarded_focus
    unguarded_harm = sum(
        row.unguarded_harmful_outlier_false_safe for row in selected
    )
    guarded_harm = sum(
        row.guarded_harmful_outlier_false_safe for row in selected
    )
    all_safe = [row for row in selected if row.unguarded_safe_accept]
    introduced_harm = sum(
        row.introduced_routed_endpoint_harm_accept for row in selected
    )
    return MatchedSubsetProfileSummary(
        profile=profile,
        case_count=len(selected),
        unguarded_harmful_outlier_false_safe_count=unguarded_harm,
        guarded_harmful_outlier_false_safe_count=guarded_harm,
        unguarded_provenance_violation_accept_count=sum(
            row.unguarded_provenance_violation_accept for row in selected
        ),
        guarded_provenance_violation_accept_count=sum(
            row.guarded_provenance_violation_accept for row in selected
        ),
        focus_unguarded_safe_accept_count=unguarded_focus,
        focus_guarded_safe_accept_count=guarded_focus,
        focus_safe_accept_retention=retention,
        all_stress_unguarded_safe_accept_count=len(all_safe),
        all_stress_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in all_safe
        ),
        introduced_routed_endpoint_harm_accept_count=introduced_harm,
        profile_gate_passed=bool(
            full_protocol
            and model_valid
            and unguarded_harm > 0
            and guarded_harm == 0
            and retention >= 0.90
            and introduced_harm == 0
        ),
    )


def _materialize_panel(
    raw_rows: Sequence[MatchedPairStressRawCase],
    *,
    panel_role: str,
    seed: int,
    model: MatchedGuardModel,
    profile_specs: tuple[MatchedPairStressSpec, ...],
    full_protocol: bool,
) -> MatchedSubsetPanel:
    rows = tuple(_materialize_case(raw, model) for raw in raw_rows)
    summaries = tuple(
        _profile_summary(
            rows,
            spec.profile,
            full_protocol=full_protocol,
            model_valid=model.calibration_valid,
        )
        for spec in profile_specs
    )
    focus = [
        row
        for row in rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    all_safe = [row for row in rows if row.unguarded_safe_accept]
    return MatchedSubsetPanel(
        panel_role=panel_role,
        seed=seed,
        cases=rows,
        profile_summaries=summaries,
        case_count=len(rows),
        unguarded_harmful_outlier_false_safe_count=sum(
            row.unguarded_harmful_outlier_false_safe for row in rows
        ),
        guarded_harmful_outlier_false_safe_count=sum(
            row.guarded_harmful_outlier_false_safe for row in rows
        ),
        focus_unguarded_safe_accept_count=unguarded_focus,
        focus_guarded_safe_accept_count=guarded_focus,
        focus_safe_accept_retention=(
            0.0 if not unguarded_focus else guarded_focus / unguarded_focus
        ),
        all_stress_unguarded_safe_accept_count=len(all_safe),
        all_stress_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in all_safe
        ),
        matched_subset_case_count=sum(row.matched_subset_applied for row in rows),
        introduced_routed_endpoint_harm_accept_count=sum(
            row.introduced_routed_endpoint_harm_accept for row in rows
        ),
        full_protocol=full_protocol,
        panel_gate_passed=bool(
            full_protocol
            and model.calibration_valid
            and len(summaries) == len(profile_specs)
            and all(summary.profile_gate_passed for summary in summaries)
        ),
    )


def evaluate_matched_subset_reconstruction(
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
) -> MatchedSubsetReconstructionResult:
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
        raise ValueError("validation and final seeds must differ")
    if any(seed in FORBIDDEN_FRESH_SEEDS for seed in fresh_seeds):
        raise ValueError("Phase-20--24 opened or reserved seeds must not be reused")
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
    design_fit_candidates = tuple(
        row
        for row in design_fit_raw
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
    )
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
    design_calibration_candidates = tuple(
        row
        for row in design_calibration_raw
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
    )
    calibration_signatures = tuple(
        matched_guard_signature(row) for row in design_calibration_candidates
    )
    calibration_harmful = tuple(
        _routed_endpoint(row).geometry_topology_harm_present
        for row in design_calibration_candidates
    )
    calibration_focus_safe = tuple(
        row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
        and not row.endpoint.geometry_topology_harm_present
        for row in design_calibration_candidates
    )
    model, cutoff_calibration = calibrate_split_cohort_cutoff(
        score_fit_model,
        score_fit_signatures=design_fit_signatures,
        score_fit_harmful_labels=design_fit_harmful,
        calibration_signatures=calibration_signatures,
        calibration_harmful_labels=calibration_harmful,
        calibration_focus_safe_labels=calibration_focus_safe,
        calibration_seed=DESIGN_CUTOFF_CALIBRATION_SEED,
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
    design_reproduced = bool(
        cutoff_calibration.calibration_valid
        and cutoff_calibration.harmful_case_count
        == EXPECTED_DESIGN_HARMFUL_COUNT
        and cutoff_calibration.focus_safe_case_count == EXPECTED_DESIGN_FOCUS_COUNT
        and math.isclose(
            cutoff_calibration.rejection_cutoff,
            EXPECTED_DESIGN_CUTOFF,
            rel_tol=0.0,
            abs_tol=DESIGN_REPRODUCTION_TOLERANCE,
        )
    )
    design_gate_passed = bool(
        design_reproduced
        and design_score_fit.panel_gate_passed
        and design_cutoff_calibration.panel_gate_passed
    )

    validation_a: MatchedSubsetPanel | None = None
    validation_b: MatchedSubsetPanel | None = None
    final_panel: MatchedSubsetPanel | None = None
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
    return MatchedSubsetReconstructionResult(
        artifact_schema="pftf_alpha_matched_subset_reconstruction_phase25/v1",
        role="synthetic_matched_subset_reconstruction_guard_audit",
        information_boundary=(
            "route uses the full primary acquisition, presented retained-pair "
            "map, presented repeat coordinates, upstream candidate decision, "
            "and frozen observed guard; source labels, clean references, and "
            "harm endpoints are design/evaluation-only"
        ),
        frozen_predecessor=(
            "phase24_missing_harmful_pair_absent_from_retained_guard_evidence"
        ),
        design_score_fit_seed=DESIGN_SCORE_FIT_SEED,
        design_cutoff_calibration_seed=DESIGN_CUTOFF_CALIBRATION_SEED,
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
        design_score_fit=design_score_fit,
        design_cutoff_calibration=design_cutoff_calibration,
        design_reproduced=design_reproduced,
        design_gate_passed=design_gate_passed,
        validation_a=validation_a,
        validation_b=validation_b,
        final_held_out=final_panel,
        phase25_supported=supported,
        matched_subset_reconstruction_synthetic_supported=supported,
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
    result = evaluate_matched_subset_reconstruction(
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
