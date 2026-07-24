import sys

from pftf_alpha.baselines import BaselineID, BenchmarkConfig, run_case_benchmarks
from pftf_alpha.exact_backend import evaluate_exact_construction_panel
from pftf_alpha.exact_filtration import evaluate_exact_filtration_panel
from pftf_alpha.exact_index_audit import evaluate_exact_critical_index_audit
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
    return (
        config,
        methods,
        primary,
        construction,
        filtration_audit,
        connectivity_shadow,
        value_shadow,
    )


def test_exact_critical_index_audit_preserves_selection_identity() -> None:
    case = _case()
    (
        config,
        methods,
        primary,
        construction,
        filtration_audit,
        connectivity_shadow,
        value_shadow,
    ) = _audit_chain(case)
    primary_before = primary.to_dict()

    audit = evaluate_exact_critical_index_audit(
        (case,),
        construction_result=construction,
        filtration_audit=filtration_audit,
        connectivity_shadow=connectivity_shadow,
        value_shadow=value_shadow,
        config=config,
        methods=methods,
    )
    payload = audit.to_dict()

    assert payload["audited_case_count"] == 1
    assert payload["critical_count_mismatch_case_count"] == 0
    assert payload["birth_group_mismatch_case_count"] == 0
    assert payload["selected_index_mismatch_method_count"] == 0
    assert payload["selected_complex_mismatch_method_count"] == 0
    assert payload["selected_boundary_mismatch_method_count"] == 0
    assert payload["b3_signature_mismatch_case_count"] == 0
    assert payload["b3_candidate_index_mismatch_case_count"] == 0
    assert payload["all_selection_identities_match"]
    assert not payload["primary_benchmark_results_changed"]
    assert payload["selection_effect"] == "none"
    assert not payload["promotion_supported"]
    assert payload["blocking_reasons"] == ["exact_critical_index_audit_not_deployed"]
    case_payload = payload["cases"][0]
    assert case_payload["critical_counts_match"]
    assert case_payload["rounded_exact_critical_count_matches_rational"]
    assert case_payload["critical_birth_group_sequence_matches"]
    assert case_payload["b3_signature_sequence_matches"]
    assert case_payload["b3_candidate_index_sequence_matches"]
    assert len(case_payload["method_identities"]) == 2
    assert all(
        identity["selected_index_matches"]
        and identity["selected_complex_matches"]
        and identity["selected_boundary_matches"]
        and identity["endpoints_match"]
        for identity in case_payload["method_identities"]
    )
    assert primary.to_dict() == primary_before


def test_exact_critical_index_audit_fails_closed_without_backend() -> None:
    case = _case()
    (
        config,
        methods,
        _,
        construction,
        filtration_audit,
        connectivity_shadow,
        value_shadow,
    ) = _audit_chain(case, backend=False)

    audit = evaluate_exact_critical_index_audit(
        (case,),
        construction_result=construction,
        filtration_audit=filtration_audit,
        connectivity_shadow=connectivity_shadow,
        value_shadow=value_shadow,
        config=config,
        methods=methods,
    )
    payload = audit.to_dict()

    assert payload["audited_case_count"] == 0
    assert payload["blocking_reasons"] == [
        "no_exact_construction_backend",
        "exact_critical_index_audit_not_deployed",
    ]
    assert payload["cases"][0]["status"] == "not_audited"
    assert payload["cases"][0]["rejection_reasons"] == ["backend_result_missing"]
