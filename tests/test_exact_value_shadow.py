import sys
from dataclasses import replace

from pftf_alpha.baselines import BaselineID, BenchmarkConfig, run_case_benchmarks
from pftf_alpha.exact_backend import evaluate_exact_construction_panel
from pftf_alpha.exact_filtration import (
    ExactFiltrationPanelAudit,
    evaluate_exact_filtration_panel,
)
from pftf_alpha.exact_shadow import evaluate_exact_connectivity_shadow
from pftf_alpha.exact_value_shadow import evaluate_exact_value_shadow
from pftf_alpha.synthetic import PanelSplit, make_minimal_panel


def _case():
    return make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=16,
        reference_count=32,
        seed=20260727,
    )[0]


def _construction(case, *, backend: bool = True):
    command = (
        (sys.executable, "-m", "pftf_alpha.exact_python_backend") if backend else None
    )
    return evaluate_exact_construction_panel(
        ((case.family.value, case.points),),
        evaluation_split=PanelSplit.HELD_OUT.value,
        backend_command=command,
        timeout_seconds=10.0,
    )


def _prerequisites(case, *, backend: bool = True):
    config = BenchmarkConfig(surface_sample_count=24, seed=20260728)
    methods = (BaselineID.B1_FIXED_ALPHA,)
    primary = run_case_benchmarks(case, config=config, methods=methods)
    construction = _construction(case, backend=backend)
    audit = evaluate_exact_filtration_panel(
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
    return config, methods, primary, construction, audit, connectivity_shadow


def test_exact_value_shadow_isolated_from_primary_and_floating_connectivity() -> None:
    case = _case()
    config, methods, primary, construction, audit, connectivity_shadow = _prerequisites(
        case
    )
    primary_before = primary.to_dict()
    assert audit.float_value_difference_count > 0

    result = evaluate_exact_value_shadow(
        (case,),
        (primary,),
        construction_result=construction,
        filtration_audit=audit,
        connectivity_shadow=connectivity_shadow,
        config=config,
        methods=methods,
    )
    payload = result.to_dict()

    assert payload["shadow_case_count"] == 1
    assert payload["primary_output_difference_case_count"] == 0
    assert payload["value_only_output_difference_case_count"] == 0
    assert payload["selected_alpha_difference_case_count"] == 0
    assert payload["objective_difference_case_count"] == 0
    assert payload["endpoint_difference_case_count"] == 0
    assert payload["candidate_bookkeeping_difference_case_count"] == 0
    assert payload["all_prerequisite_cases_evaluated"]
    assert not payload["primary_benchmark_results_changed"]
    assert payload["selection_effect"] == "none"
    assert not payload["promotion_supported"]
    assert payload["blocking_reasons"] == ["exact_rounded_value_shadow_not_deployed"]
    comparison = payload["cases"][0]
    assert comparison["exact_audit_verified"]
    assert comparison["connectivity_matches_primary"]
    assert comparison["changed_methods_vs_primary"] == []
    assert comparison["changed_methods_vs_floating_connectivity_shadow"] == []
    assert comparison["all_nonruntime_outputs_match_primary"]
    assert comparison["all_nonruntime_outputs_match_floating_connectivity_shadow"]
    assert primary.to_dict() == primary_before


def test_exact_value_shadow_fails_closed_without_backend() -> None:
    case = _case()
    config, methods, primary, construction, audit, connectivity_shadow = _prerequisites(
        case, backend=False
    )
    primary_before = primary.to_dict()

    result = evaluate_exact_value_shadow(
        (case,),
        (primary,),
        construction_result=construction,
        filtration_audit=audit,
        connectivity_shadow=connectivity_shadow,
        config=config,
        methods=methods,
    )
    payload = result.to_dict()

    assert payload["shadow_case_count"] == 0
    assert payload["blocking_reasons"] == [
        "no_exact_construction_backend",
        "exact_rounded_value_shadow_not_deployed",
    ]
    assert not payload["cases"][0]["shadow_ran"]
    assert payload["cases"][0]["shadow_report"] is None
    assert primary.to_dict() == primary_before


def test_exact_value_shadow_rejects_audit_digest_mismatch() -> None:
    case = _case()
    config, methods, primary, construction, audit, connectivity_shadow = _prerequisites(
        case
    )
    altered_case = replace(audit.cases[0], exact_filtration_sha256="0" * 64)
    altered_audit = ExactFiltrationPanelAudit(
        evaluation_split=audit.evaluation_split,
        backend_requested=audit.backend_requested,
        requested_case_count=audit.requested_case_count,
        accepted_backend_case_count=audit.accepted_backend_case_count,
        cases=(altered_case,),
    )

    result = evaluate_exact_value_shadow(
        (case,),
        (primary,),
        construction_result=construction,
        filtration_audit=altered_audit,
        connectivity_shadow=connectivity_shadow,
        config=config,
        methods=methods,
    )

    assert result.shadow_case_count == 0
    assert result.cases[0].rejection_reasons == ("exact_filtration_digest_mismatch",)
