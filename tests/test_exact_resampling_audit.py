import sys

from pftf_alpha.baselines import BaselineID, BenchmarkConfig, run_case_benchmarks
from pftf_alpha.exact_backend import evaluate_exact_construction_panel
from pftf_alpha.exact_filtration import evaluate_exact_filtration_panel
from pftf_alpha.exact_index_audit import evaluate_exact_critical_index_audit
from pftf_alpha.exact_resampling_audit import (
    evaluate_exact_resampling_threshold_audit,
)
from pftf_alpha.exact_shadow import evaluate_exact_connectivity_shadow
from pftf_alpha.exact_value_shadow import evaluate_exact_value_shadow
from pftf_alpha.synthetic import PanelSplit, make_minimal_panel


def _case():
    return make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=16,
        reference_count=32,
        seed=20260729,
    )[0]


def _audit_chain(case, *, backend: bool = True):
    config = BenchmarkConfig(
        surface_sample_count=24,
        resample_repeats=2,
        b3_candidate_budget=6,
        seed=20260730,
    )
    methods = (
        BaselineID.B2_CRITICAL_ORACLE,
        BaselineID.B3_PERSISTENCE_STABILITY,
    )
    primary = run_case_benchmarks(case, config=config, methods=methods)
    backend_command = (
        (sys.executable, "-m", "pftf_alpha.exact_python_backend") if backend else None
    )
    construction = evaluate_exact_construction_panel(
        ((case.family.value, case.points),),
        evaluation_split=PanelSplit.HELD_OUT.value,
        backend_command=backend_command,
        timeout_seconds=10.0,
    )
    filtration_audit = evaluate_exact_filtration_panel(
        (case,),
        construction_result=construction,
    )
    connectivity_shadow = evaluate_exact_connectivity_shadow(
        (case,),
        (primary,),
        construction_result=construction,
        config=config,
        methods=methods,
    )
    value_shadow = evaluate_exact_value_shadow(
        (case,),
        (primary,),
        construction_result=construction,
        filtration_audit=filtration_audit,
        connectivity_shadow=connectivity_shadow,
        config=config,
        methods=methods,
    )
    critical_index_audit = evaluate_exact_critical_index_audit(
        (case,),
        construction_result=construction,
        filtration_audit=filtration_audit,
        connectivity_shadow=connectivity_shadow,
        value_shadow=value_shadow,
        config=config,
        methods=methods,
    )
    return (
        config,
        primary,
        construction,
        filtration_audit,
        connectivity_shadow,
        value_shadow,
        critical_index_audit,
    )


def test_exact_resampling_threshold_audit_reproduces_b3_stability() -> None:
    case = _case()
    (
        config,
        primary,
        construction,
        filtration_audit,
        connectivity_shadow,
        value_shadow,
        critical_index_audit,
    ) = _audit_chain(case)
    primary_before = primary.to_dict()

    audit = evaluate_exact_resampling_threshold_audit(
        (case,),
        construction_result=construction,
        filtration_audit=filtration_audit,
        connectivity_shadow=connectivity_shadow,
        value_shadow=value_shadow,
        critical_index_audit=critical_index_audit,
        config=config,
    )
    payload = audit.to_dict()

    assert payload["audited_case_count"] == 1
    assert payload["threshold_effect_reproduction_failure_case_count"] == 0
    assert payload["all_reported_stability_reproduced"]
    assert not payload["exact_resampled_connectivity_constructed"]
    assert not payload["primary_benchmark_results_changed"]
    assert payload["selection_effect"] == "none"
    assert not payload["promotion_supported"]
    assert payload["blocking_reasons"] == [
        "exact_resampling_threshold_audit_not_deployed"
    ]
    case_payload = payload["cases"][0]
    assert case_payload["selected_index_matches"]
    assert case_payload["full_surface_samples_match"]
    assert (
        case_payload["floating_full_samples_sha256"]
        == case_payload["exact_full_samples_sha256"]
    )
    assert case_payload["resample_repeat_count"] == config.resample_repeats
    assert case_payload["floating_recomputation_matches_report"]
    assert case_payload["exact_recomputation_matches_report"]
    assert case_payload["threshold_effect_reproduced"]
    if not case_payload["reported_stability_matches"]:
        assert case_payload["resampled_boundary_difference_repeat_count"] > 0
    assert primary.to_dict() == primary_before


def test_exact_resampling_threshold_audit_fails_closed_without_backend() -> None:
    case = _case()
    (
        config,
        _,
        construction,
        filtration_audit,
        connectivity_shadow,
        value_shadow,
        critical_index_audit,
    ) = _audit_chain(case, backend=False)

    audit = evaluate_exact_resampling_threshold_audit(
        (case,),
        construction_result=construction,
        filtration_audit=filtration_audit,
        connectivity_shadow=connectivity_shadow,
        value_shadow=value_shadow,
        critical_index_audit=critical_index_audit,
        config=config,
    )
    payload = audit.to_dict()

    assert payload["audited_case_count"] == 0
    assert payload["blocking_reasons"] == [
        "no_exact_construction_backend",
        "exact_resampling_threshold_audit_not_deployed",
    ]
    assert payload["cases"][0]["status"] == "not_audited"
    assert payload["cases"][0]["rejection_reasons"] == ["backend_result_missing"]
