"""Targeted incremental local-residual challenge for Phase 29."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .frozen_partition_reconstruction import _materialize_panel, _panel_case_seeds
from .local_spatial_residual_guard import (
    EXPECTED_LIMITING_LOCAL_VALUE,
    EXPECTED_LOCAL_REJECTION_CUTOFF,
    LocalSpatialPanel,
    _materialize_local_panel,
    evaluate_local_spatial_residual_guard,
)
from .matched_guard_signature import GUARD_PROFILE_SPECS
from .matched_pair_stress import _raw_panel
from .sensor_stress import DEFAULT_POINT_COUNTS, DEFAULT_STRESSES, SensorStress

TARGET_POINT_COUNTS = (96,)
TARGET_STRESSES = (
    SensorStress.CONTROL,
    SensorStress.LOCAL_BUMP,
    SensorStress.OUTLIERS_01,
)
TARGET_REPEATS = 64
VALIDATION_A_SEED = 30200804
VALIDATION_B_SEED = 30300804
FINAL_HELD_OUT_SEED = 30400804
PRIOR_BASE_SEEDS = tuple(index * 100000 + 804 for index in range(203, 260)) + tuple(
    index * 100000 + 804 for index in range(275, 284)
)


@dataclass(frozen=True)
class Phase29CaseSeedAudit:
    case_seed_formula: str
    prior_base_seed_ranges: tuple[str, ...]
    targeted_point_counts: tuple[int, ...]
    targeted_stresses: tuple[SensorStress, ...]
    targeted_repeats: int
    panel_case_count: int
    validation_a_prior_overlap_count: int
    validation_b_prior_overlap_count: int
    final_prior_overlap_count: int
    validation_a_b_overlap_count: int
    validation_a_final_overlap_count: int
    validation_b_final_overlap_count: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["targeted_stresses"] = [
            stress.value for stress in self.targeted_stresses
        ]
        return payload


@dataclass(frozen=True)
class IncrementalEvidenceSummary:
    opened_panel_count: int
    required_panel_count: int
    predecessor_residual_harmful_count: int
    combined_residual_harmful_count: int
    locally_rescued_harmful_count: int
    original_focus_safe_accept_count: int
    predecessor_focus_safe_accept_count: int
    combined_focus_safe_accept_count: int
    introduced_routed_endpoint_harm_accept_count: int
    informative_predecessor_residual_observed: bool
    all_opened_safety_gates_passed: bool
    incremental_evidence_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TargetedLocalResidualChallengeResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    target_point_counts: tuple[int, ...]
    target_stresses: tuple[SensorStress, ...]
    target_repeats: int
    validation_a_seed: int
    validation_b_seed: int
    final_held_out_seed: int
    reference_count: int
    surface_sample_count: int
    case_seed_disjointness: Phase29CaseSeedAudit
    phase28_design_reproduced: bool
    phase28_design_gate_passed: bool
    frozen_phase28_score_cutoff: float
    frozen_phase28_local_limiting_value: float
    frozen_phase28_local_cutoff: float
    design_gate_passed: bool
    challenge_execution_requested: bool
    validation_a: LocalSpatialPanel | None
    validation_b: LocalSpatialPanel | None
    final_held_out: LocalSpatialPanel | None
    incremental_evidence: IncrementalEvidenceSummary
    phase29_supported: bool
    incremental_local_rescue_synthetic_supported: bool
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
            "target_point_counts": list(self.target_point_counts),
            "target_stresses": [stress.value for stress in self.target_stresses],
            "target_repeats": self.target_repeats,
            "validation_a_seed": self.validation_a_seed,
            "validation_b_seed": self.validation_b_seed,
            "final_held_out_seed": self.final_held_out_seed,
            "reference_count": self.reference_count,
            "surface_sample_count": self.surface_sample_count,
            "case_seed_disjointness": self.case_seed_disjointness.to_dict(),
            "phase28_design_reproduced": self.phase28_design_reproduced,
            "phase28_design_gate_passed": self.phase28_design_gate_passed,
            "frozen_phase28_score_cutoff": self.frozen_phase28_score_cutoff,
            "frozen_phase28_local_limiting_value": (
                self.frozen_phase28_local_limiting_value
            ),
            "frozen_phase28_local_cutoff": self.frozen_phase28_local_cutoff,
            "design_gate_passed": self.design_gate_passed,
            "challenge_execution_requested": self.challenge_execution_requested,
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
            "incremental_evidence": self.incremental_evidence.to_dict(),
            "phase29_supported": self.phase29_supported,
            "incremental_local_rescue_synthetic_supported": (
                self.incremental_local_rescue_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "real_trimmed_reconstruction_supported": (
                self.real_trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def audit_phase29_case_seed_disjointness(
    validation_a_seed: int,
    validation_b_seed: int,
    final_held_out_seed: int,
) -> Phase29CaseSeedAudit:
    prior = frozenset().union(
        *(
            _panel_case_seeds(
                seed,
                point_counts=DEFAULT_POINT_COUNTS,
                stresses=DEFAULT_STRESSES,
                repeats=8,
            )
            for seed in PRIOR_BASE_SEEDS
        )
    )
    panels = tuple(
        _panel_case_seeds(
            seed,
            point_counts=TARGET_POINT_COUNTS,
            stresses=TARGET_STRESSES,
            repeats=TARGET_REPEATS,
        )
        for seed in (validation_a_seed, validation_b_seed, final_held_out_seed)
    )
    prior_overlaps = tuple(len(panel & prior) for panel in panels)
    mutual_overlaps = (
        len(panels[0] & panels[1]),
        len(panels[0] & panels[2]),
        len(panels[1] & panels[2]),
    )
    return Phase29CaseSeedAudit(
        case_seed_formula=(
            "base + count_index*1000003 + stress_index*100003 + repeat*10007"
        ),
        prior_base_seed_ranges=(
            "20300804--25900804",
            "27500804--28300804",
        ),
        targeted_point_counts=TARGET_POINT_COUNTS,
        targeted_stresses=TARGET_STRESSES,
        targeted_repeats=TARGET_REPEATS,
        panel_case_count=len(panels[0]),
        validation_a_prior_overlap_count=prior_overlaps[0],
        validation_b_prior_overlap_count=prior_overlaps[1],
        final_prior_overlap_count=prior_overlaps[2],
        validation_a_b_overlap_count=mutual_overlaps[0],
        validation_a_final_overlap_count=mutual_overlaps[1],
        validation_b_final_overlap_count=mutual_overlaps[2],
        passed=not any(prior_overlaps + mutual_overlaps),
    )


def summarize_incremental_evidence(
    panels: Sequence[LocalSpatialPanel],
    *,
    required_panel_count: int = 3,
) -> IncrementalEvidenceSummary:
    selected = tuple(panels)
    predecessor_harm = sum(
        panel.predecessor_guarded_harmful_outlier_false_safe_count
        for panel in selected
    )
    combined_harm = sum(
        panel.guarded_harmful_outlier_false_safe_count for panel in selected
    )
    safety_passed = bool(
        len(selected) == required_panel_count
        and all(panel.panel_gate_passed for panel in selected)
    )
    informative = predecessor_harm > 0
    return IncrementalEvidenceSummary(
        opened_panel_count=len(selected),
        required_panel_count=required_panel_count,
        predecessor_residual_harmful_count=predecessor_harm,
        combined_residual_harmful_count=combined_harm,
        locally_rescued_harmful_count=max(predecessor_harm - combined_harm, 0),
        original_focus_safe_accept_count=sum(
            panel.focus_unguarded_safe_accept_count for panel in selected
        ),
        predecessor_focus_safe_accept_count=sum(
            panel.focus_predecessor_guarded_safe_accept_count for panel in selected
        ),
        combined_focus_safe_accept_count=sum(
            panel.focus_guarded_safe_accept_count for panel in selected
        ),
        introduced_routed_endpoint_harm_accept_count=sum(
            panel.introduced_routed_endpoint_harm_accept_count
            for panel in selected
        ),
        informative_predecessor_residual_observed=informative,
        all_opened_safety_gates_passed=safety_passed,
        incremental_evidence_gate_passed=bool(
            safety_passed and informative and combined_harm == 0
        ),
    )


def evaluate_targeted_local_residual_challenge(
    *,
    validation_a_seed: int = VALIDATION_A_SEED,
    validation_b_seed: int = VALIDATION_B_SEED,
    final_held_out_seed: int = FINAL_HELD_OUT_SEED,
    reference_count: int = 2048,
    surface_sample_count: int = 256,
    open_challenge: bool = True,
) -> TargetedLocalResidualChallengeResult:
    seeds = (validation_a_seed, validation_b_seed, final_held_out_seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("validation and final base seeds must differ")
    seed_audit = audit_phase29_case_seed_disjointness(*seeds)
    if not seed_audit.passed:
        raise ValueError(
            "challenge case seeds must be mutually disjoint and disjoint from "
            "all prior full panels"
        )
    full_protocol = bool(
        seeds == (VALIDATION_A_SEED, VALIDATION_B_SEED, FINAL_HELD_OUT_SEED)
        and reference_count >= 2048
        and surface_sample_count >= 256
    )
    predecessor = evaluate_local_spatial_residual_guard(open_fresh=False)
    design_reproduced = bool(
        full_protocol
        and predecessor.design_reproduced
        and predecessor.design_gate_passed
        and predecessor.validation_a is None
        and predecessor.validation_b is None
        and predecessor.final_held_out is None
        and predecessor.local_calibration.limiting_local_value
        == EXPECTED_LIMITING_LOCAL_VALUE
        and predecessor.local_calibration.rejection_cutoff
        == EXPECTED_LOCAL_REJECTION_CUTOFF
    )
    design_gate_passed = bool(design_reproduced and seed_audit.passed)
    panels: list[LocalSpatialPanel] = []
    common = {
        "point_counts": TARGET_POINT_COUNTS,
        "stresses": TARGET_STRESSES,
        "reference_count": reference_count,
        "repeats": TARGET_REPEATS,
        "surface_sample_count": surface_sample_count,
        "base_gate_config": None,
        "shared_trend_config": None,
        "matched_pair_config": predecessor.matched_pair_config,
        "stress_config": predecessor.stress_config,
        "profile_specs": GUARD_PROFILE_SPECS,
    }

    def challenge_panel(seed: int, role: str) -> LocalSpatialPanel:
        predecessor_panel = _materialize_panel(
            _raw_panel(seed=seed, **common),
            panel_role=f"{role}_predecessor",
            seed=seed,
            model=predecessor.predecessor_model,
            profile_specs=GUARD_PROFILE_SPECS,
            full_protocol=False,
        )
        return _materialize_local_panel(
            predecessor_panel,
            predecessor.local_calibration,
            panel_role=role,
            full_protocol=full_protocol,
        )

    if open_challenge and design_gate_passed:
        validation_a = challenge_panel(validation_a_seed, "validation_a")
        panels.append(validation_a)
        if validation_a.panel_gate_passed:
            validation_b = challenge_panel(validation_b_seed, "validation_b")
            panels.append(validation_b)
            if validation_b.panel_gate_passed:
                final_panel = challenge_panel(final_held_out_seed, "final_held_out")
                panels.append(final_panel)
    validation_a = panels[0] if len(panels) >= 1 else None
    validation_b = panels[1] if len(panels) >= 2 else None
    final_panel = panels[2] if len(panels) >= 3 else None
    incremental = summarize_incremental_evidence(panels)
    supported = incremental.incremental_evidence_gate_passed
    return TargetedLocalResidualChallengeResult(
        artifact_schema="pftf_alpha_targeted_local_residual_challenge_phase29/v1",
        role="synthetic_targeted_incremental_local_rescue_audit",
        information_boundary=(
            "route is exactly the frozen Phase-28 score and primary-coordinate "
            "8-NN local residual guard; challenge targets are declared before "
            "opening and endpoint truth is evaluation-only"
        ),
        frozen_predecessor="phase28_combined_route_with_unresolved_fresh_margin",
        target_point_counts=TARGET_POINT_COUNTS,
        target_stresses=TARGET_STRESSES,
        target_repeats=TARGET_REPEATS,
        validation_a_seed=validation_a_seed,
        validation_b_seed=validation_b_seed,
        final_held_out_seed=final_held_out_seed,
        reference_count=reference_count,
        surface_sample_count=surface_sample_count,
        case_seed_disjointness=seed_audit,
        phase28_design_reproduced=design_reproduced,
        phase28_design_gate_passed=predecessor.design_gate_passed,
        frozen_phase28_score_cutoff=predecessor.predecessor_model.rejection_cutoff,
        frozen_phase28_local_limiting_value=EXPECTED_LIMITING_LOCAL_VALUE,
        frozen_phase28_local_cutoff=EXPECTED_LOCAL_REJECTION_CUTOFF,
        design_gate_passed=design_gate_passed,
        challenge_execution_requested=open_challenge,
        validation_a=validation_a,
        validation_b=validation_b,
        final_held_out=final_panel,
        incremental_evidence=incremental,
        phase29_supported=supported,
        incremental_local_rescue_synthetic_supported=supported,
        real_correspondence_supported=False,
        real_paired_scan_supported=False,
        real_trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--design-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_targeted_local_residual_challenge(
        open_challenge=not args.design_only,
    )
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result.phase29_supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
