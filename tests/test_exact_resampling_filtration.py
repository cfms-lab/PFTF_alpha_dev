import sys

from pftf_alpha.baselines import BaselineID, BenchmarkConfig, run_case_benchmarks
from pftf_alpha.exact_b3_shadow import evaluate_exact_b3_selection_shadow
from pftf_alpha.exact_backend import evaluate_exact_construction_panel
from pftf_alpha.exact_filtration import evaluate_exact_filtration_panel
from pftf_alpha.exact_index_audit import evaluate_exact_critical_index_audit
from pftf_alpha.exact_resampling_audit import (
    evaluate_exact_resampling_threshold_audit,
)
from pftf_alpha.exact_resampling_filtration import (
    evaluate_exact_resampling_filtration_audit,
)
from pftf_alpha.exact_shadow import evaluate_exact_connectivity_shadow
from pftf_alpha.exact_value_shadow import evaluate_exact_value_shadow
from pftf_alpha.synthetic import PanelSplit, make_minimal_panel


def _prerequisites():
    case = make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=16,
        reference_count=32,
        seed=20260731,
    )[0]
    config = BenchmarkConfig(
        surface_sample_count=24,
        resample_repeats=2,
        b3_candidate_budget=6,
        seed=20260801,
    )
    methods = (
        BaselineID.B2_CRITICAL_ORACLE,
        BaselineID.B3_PERSISTENCE_STABILITY,
    )
    primary = run_case_benchmarks(case, config=config, methods=methods)
    backend_command = (sys.executable, "-m", "pftf_alpha.exact_python_backend")
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
    threshold_audit = evaluate_exact_resampling_threshold_audit(
        (case,),
        construction_result=construction,
        filtration_audit=filtration_audit,
        connectivity_shadow=connectivity_shadow,
        value_shadow=value_shadow,
        critical_index_audit=critical_index_audit,
        config=config,
    )
    return (
        case,
        config,
        primary,
        construction,
        threshold_audit,
        value_shadow,
        backend_command,
    )


def test_exact_resampling_filtration_audits_every_repeat() -> None:
    (
        case,
        config,
        primary,
        construction,
        threshold_audit,
        _,
        command,
    ) = _prerequisites()
    primary_before = primary.to_dict()

    audit = evaluate_exact_resampling_filtration_audit(
        (case,),
        construction_result=construction,
        threshold_audit=threshold_audit,
        config=config,
        backend_command=command,
        backend_timeout_seconds=10.0,
    )
    payload = audit.to_dict()

    assert payload["audited_case_count"] == 1
    assert payload["requested_repeat_count"] == config.resample_repeats
    assert payload["audited_repeat_count"] == config.resample_repeats
    assert payload["rejected_repeat_count"] == 0
    assert payload["all_resamples_audited"]
    assert payload["selected_complex_difference_repeat_count"] == (
        payload["selected_boundary_difference_repeat_count"]
    )
    exact_resamples = audit.exact_resampled_filtrations(case.family.value)
    assert exact_resamples is not None
    assert len(exact_resamples) == config.resample_repeats
    assert "_exact_filtrations_by_case" not in payload
    assert not payload["primary_benchmark_results_changed"]
    assert payload["selection_effect"] == "none"
    assert not payload["promotion_supported"]
    assert payload["blocking_reasons"] == [
        "exact_resampling_filtration_audit_not_deployed"
    ]
    case_payload = payload["cases"][0]
    assert case_payload["full_surface_samples_match"]
    assert case_payload["floating_recomputation_matches_threshold_audit"]
    assert case_payload["audited_repeat_count"] == config.resample_repeats
    assert all(
        repeat["backend_accepted"]
        and repeat["exact_filtration_audited"]
        and repeat["exact_filtration_sha256"] is not None
        and not repeat["rejection_reasons"]
        for repeat in case_payload["repeats"]
    )
    assert primary.to_dict() == primary_before


def test_exact_resampling_filtration_fails_closed_without_backend() -> None:
    case, config, _, construction, threshold_audit, _, _ = _prerequisites()

    audit = evaluate_exact_resampling_filtration_audit(
        (case,),
        construction_result=construction,
        threshold_audit=threshold_audit,
        config=config,
        backend_command=None,
    )
    payload = audit.to_dict()

    assert payload["audited_case_count"] == 0
    assert payload["audited_repeat_count"] == 0
    assert not payload["all_resamples_audited"]
    assert payload["blocking_reasons"] == [
        "no_exact_resampling_backend",
        "exact_resampling_filtration_audit_not_deployed",
    ]
    assert payload["cases"][0]["rejection_reasons"] == [
        "no_exact_resampling_backend"
    ]


def test_exact_b3_selection_shadow_evaluates_every_budgeted_candidate() -> None:
    (
        case,
        config,
        primary,
        construction,
        threshold_audit,
        value_shadow,
        command,
    ) = _prerequisites()
    primary_before = primary.to_dict()
    resampling_audit = evaluate_exact_resampling_filtration_audit(
        (case,),
        construction_result=construction,
        threshold_audit=threshold_audit,
        config=config,
        backend_command=command,
        backend_timeout_seconds=10.0,
    )

    shadow = evaluate_exact_b3_selection_shadow(
        (case,),
        construction_result=construction,
        exact_value_shadow=value_shadow,
        exact_resampling_audit=resampling_audit,
        config=config,
    )
    payload = shadow.to_dict()

    assert payload["shadow_case_count"] == 1
    assert payload["all_prerequisite_cases_evaluated"]
    assert not payload["primary_benchmark_results_changed"]
    assert payload["selection_effect"] == "none"
    assert not payload["promotion_supported"]
    assert payload["blocking_reasons"] == [
        "exact_b3_selection_shadow_not_deployed"
    ]
    case_payload = payload["cases"][0]
    assert case_payload["reference_report_reproduced"]
    assert 0 < case_payload["candidate_count"] <= config.b3_candidate_budget
    assert len(case_payload["candidates"]) == case_payload["candidate_count"]
    assert all(
        candidate["nonstability_terms_match"]
        for candidate in case_payload["candidates"]
    )
    assert case_payload["exact_selection_shadow_result"]["method"] == "B3"
    assert primary.to_dict() == primary_before


def test_exact_b3_selection_shadow_fails_closed_without_schema24_context() -> None:
    (
        case,
        config,
        _,
        construction,
        threshold_audit,
        value_shadow,
        _,
    ) = _prerequisites()
    resampling_audit = evaluate_exact_resampling_filtration_audit(
        (case,),
        construction_result=construction,
        threshold_audit=threshold_audit,
        config=config,
        backend_command=None,
    )

    shadow = evaluate_exact_b3_selection_shadow(
        (case,),
        construction_result=construction,
        exact_value_shadow=value_shadow,
        exact_resampling_audit=resampling_audit,
        config=config,
    )
    payload = shadow.to_dict()

    assert payload["shadow_case_count"] == 0
    assert not payload["all_prerequisite_cases_evaluated"]
    assert payload["blocking_reasons"] == [
        "no_exact_resampling_backend",
        "one_or_more_schema24_exact_resampling_cases_missing",
        "exact_b3_selection_shadow_not_deployed",
    ]
    assert payload["cases"][0]["rejection_reasons"] == [
        "schema24_exact_resampling_context_missing"
    ]
