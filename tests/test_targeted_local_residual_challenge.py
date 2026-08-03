from __future__ import annotations

from dataclasses import replace

from pftf_alpha.local_spatial_residual_guard import LocalSpatialPanel
from pftf_alpha.targeted_local_residual_challenge import (
    FINAL_HELD_OUT_SEED,
    TARGET_REPEATS,
    VALIDATION_A_SEED,
    VALIDATION_B_SEED,
    audit_phase29_case_seed_disjointness,
    summarize_incremental_evidence,
)


def _panel(*, predecessor_harm: int, combined_harm: int) -> LocalSpatialPanel:
    return LocalSpatialPanel(
        panel_role="test",
        seed=1,
        cases=(),
        profile_summaries=(),
        case_count=0,
        unguarded_harmful_outlier_false_safe_count=3,
        predecessor_guarded_harmful_outlier_false_safe_count=predecessor_harm,
        guarded_harmful_outlier_false_safe_count=combined_harm,
        focus_unguarded_safe_accept_count=10,
        focus_predecessor_guarded_safe_accept_count=10,
        focus_guarded_safe_accept_count=9,
        focus_safe_accept_retention=0.9,
        all_stress_unguarded_safe_accept_count=20,
        all_stress_predecessor_guarded_safe_accept_count=19,
        all_stress_guarded_safe_accept_count=18,
        introduced_routed_endpoint_harm_accept_count=0,
        full_protocol=True,
        panel_gate_passed=True,
    )


def test_default_targeted_case_seeds_are_disjoint() -> None:
    audit = audit_phase29_case_seed_disjointness(
        VALIDATION_A_SEED,
        VALIDATION_B_SEED,
        FINAL_HELD_OUT_SEED,
    )

    assert audit.passed is True
    assert audit.targeted_repeats == TARGET_REPEATS
    assert audit.panel_case_count == 192
    assert audit.validation_a_prior_overlap_count == 0
    assert audit.validation_a_b_overlap_count == 0


def test_sequential_base_284_is_not_fresh_for_the_targeted_protocol() -> None:
    audit = audit_phase29_case_seed_disjointness(
        28400804,
        VALIDATION_B_SEED,
        FINAL_HELD_OUT_SEED,
    )

    assert audit.passed is False
    assert audit.validation_a_prior_overlap_count > 0


def test_incremental_gate_requires_an_observed_predecessor_residual() -> None:
    panels = tuple(_panel(predecessor_harm=0, combined_harm=0) for _ in range(3))

    summary = summarize_incremental_evidence(panels)

    assert summary.all_opened_safety_gates_passed is True
    assert summary.informative_predecessor_residual_observed is False
    assert summary.incremental_evidence_gate_passed is False


def test_incremental_gate_passes_only_when_local_guard_rescues_residual() -> None:
    base = _panel(predecessor_harm=0, combined_harm=0)
    panels = (
        replace(base, predecessor_guarded_harmful_outlier_false_safe_count=1),
        base,
        base,
    )

    summary = summarize_incremental_evidence(panels)

    assert summary.predecessor_residual_harmful_count == 1
    assert summary.combined_residual_harmful_count == 0
    assert summary.locally_rescued_harmful_count == 1
    assert summary.incremental_evidence_gate_passed is True


def test_incremental_gate_fails_when_local_residual_survives() -> None:
    base = _panel(predecessor_harm=0, combined_harm=0)
    panels = (
        replace(
            base,
            predecessor_guarded_harmful_outlier_false_safe_count=1,
            guarded_harmful_outlier_false_safe_count=1,
            panel_gate_passed=False,
        ),
        base,
        base,
    )

    summary = summarize_incremental_evidence(panels)

    assert summary.informative_predecessor_residual_observed is True
    assert summary.combined_residual_harmful_count == 1
    assert summary.locally_rescued_harmful_count == 0
    assert summary.incremental_evidence_gate_passed is False
