import json
from pathlib import Path

import pytest

from pftf_alpha.benchmark import main
from pftf_alpha.exact_python_backend import BACKEND_KERNEL, BACKEND_NAME


def test_cli_records_missing_exact_backend_as_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "exact_backend_missing.json"

    exit_code = main(
        [
            "--split",
            "held_out",
            "--methods",
            "P2",
            "--p2-scale-multiplier",
            "1.2",
            "--p2-confidence-threshold",
            "0.3",
            "--evaluate-exact-construction",
            "--evaluate-exact-connectivity-shadow",
            "--point-count",
            "24",
            "--reference-count",
            "48",
            "--surface-samples",
            "24",
            "--adaptive-knn",
            "6",
            "--seed",
            "908",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 25
    contract = payload["exact_construction_backend_contract"]
    assert contract["role"] == "optional_backend_handoff_validation_no_selection"
    assert (
        contract["construction_effect"]
        == "validated_connectivity_available_for_shadow_only"
    )
    assert contract["selection_effect"] == "none"
    shadow_contract = payload["exact_connectivity_shadow_contract"]
    assert shadow_contract["role"] == "evaluation_only_shadow_no_selection"
    assert shadow_contract["filtration_construction"].endswith("from_top_simplices")
    assert shadow_contract["selection_effect"] == "none"

    handoff = payload["exact_construction_backend"]
    assert handoff["enabled"]
    assert handoff["source_split"] == "held_out"
    assert not handoff["backend_executable_explicit"]
    result = handoff["result"]
    assert not result["backend_requested"]
    assert not result["backend_handoff_validated"]
    assert not result["exact_construction_applied_to_benchmark"]
    assert not result["changes_benchmark_selection"]
    assert not result["promotion_supported"]
    assert result["blocking_reasons"] == ["no_exact_construction_backend"]

    shadow = payload["exact_connectivity_shadow"]
    assert shadow["enabled"]
    assert shadow["source_split"] == "held_out"
    shadow_result = shadow["result"]
    assert not shadow_result["backend_requested"]
    assert shadow_result["shadow_case_count"] == 0
    assert shadow_result["output_difference_case_count"] == 0
    assert not shadow_result["primary_benchmark_results_changed"]
    assert shadow_result["selection_effect"] == "none"
    assert not shadow_result["promotion_supported"]
    assert shadow_result["blocking_reasons"] == [
        "no_exact_construction_backend",
        "exact_connectivity_shadow_not_deployed",
    ]
    assert len(shadow_result["cases"]) == 6
    assert all(
        not case["shadow_ran"] and case["shadow_report"] is None
        for case in shadow_result["cases"]
    )


def test_cli_runs_builtin_exact_backend_in_shadow(tmp_path: Path) -> None:
    output = tmp_path / "builtin_exact_backend.json"

    exit_code = main(
        [
            "--split",
            "held_out",
            "--methods",
            "B1",
            "--evaluate-exact-predicates",
            "--evaluate-exact-construction",
            "--exact-python-backend",
            "--evaluate-exact-connectivity-shadow",
            "--evaluate-exact-filtration-values",
            "--evaluate-exact-value-shadow",
            "--point-count",
            "16",
            "--reference-count",
            "32",
            "--surface-samples",
            "24",
            "--seed",
            "909",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 25
    contract = payload["exact_python_backend_contract"]
    assert contract["role"] == "small_panel_exact_construction_backend_no_selection"
    assert contract["candidate_connectivity_source"] == "none"
    assert not contract["scipy_qhull_connectivity_use"]
    assert contract["max_point_count"] == 64
    assert contract["selection_effect"] == "none"

    predicate_result = payload["exact_predicate_audit"]["result"]
    assert predicate_result["exact_construction_backend_integrated"]
    assert predicate_result["blocking_reasons"] == []
    assert not predicate_result["promotion_supported"]

    handoff = payload["exact_construction_backend"]
    assert handoff["enabled"]
    assert handoff["backend_mode"] == "builtin_python_exact"
    assert handoff["builtin_python_backend_requested"]
    assert not handoff["backend_executable_explicit"]
    handoff_result = handoff["result"]
    assert handoff_result["backend_requested"]
    assert handoff_result["accepted_case_count"] == 6
    assert handoff_result["backend_handoff_validated"]
    assert not handoff_result["changes_benchmark_selection"]
    assert handoff_result["blocking_reasons"] == [
        "exact_connectivity_not_applied_to_benchmark_selection"
    ]
    assert all(
        case["backend_name"] == BACKEND_NAME
        and case["backend_kernel"] == BACKEND_KERNEL
        and case["accepted"]
        for case in handoff_result["cases"]
    )

    filtration_contract = payload["exact_filtration_value_audit_contract"]
    assert (
        filtration_contract["role"]
        == "exact_rational_filtration_value_audit_no_selection"
    )
    assert filtration_contract["simplex_dimensions"] == [0, 1, 2, 3]
    assert not filtration_contract["exact_values_applied_to_primary"]
    assert filtration_contract["selection_effect"] == "none"

    filtration_result = payload["exact_filtration_value_audit"]["result"]
    assert filtration_result["accepted_backend_case_count"] == 6
    assert filtration_result["audited_case_count"] == 6
    assert filtration_result["all_accepted_cases_audited"]
    assert not filtration_result["exact_filtration_values_applied_to_primary"]
    assert not filtration_result["primary_benchmark_results_changed"]
    assert filtration_result["selection_effect"] == "none"
    assert filtration_result["total_simplex_count"] == sum(
        case["simplex_count"] for case in filtration_result["cases"]
    )
    assert filtration_result["float_value_difference_count"] > 0
    assert filtration_result["gabriel_disagreement_case_count"] == 0
    assert filtration_result["exact_tie_split_case_count"] == 0
    assert (
        filtration_result["correctly_rounded_critical_count_mismatch_case_count"] == 0
    )
    assert filtration_result["floating_critical_count_mismatch_case_count"] == 0
    assert filtration_result["order_violation_case_count"] == 0
    assert filtration_result["blocking_reasons"] == [
        "exact_filtration_values_audit_only_not_deployed"
    ]
    assert all(
        case["status"] == "audited" and case["exact_filtration_sha256"] is not None
        for case in filtration_result["cases"]
    )

    shadow_result = payload["exact_connectivity_shadow"]["result"]
    assert shadow_result["accepted_backend_case_count"] == 6
    assert shadow_result["shadow_case_count"] == 6
    assert shadow_result["output_difference_case_count"] == 0
    assert shadow_result["all_accepted_cases_evaluated"]
    assert not shadow_result["primary_benchmark_results_changed"]
    assert shadow_result["blocking_reasons"] == [
        "exact_connectivity_shadow_not_deployed"
    ]
    assert all(
        case["connectivity_matches_primary"]
        and case["shadow_ran"]
        and case["all_nonruntime_outputs_match"]
        for case in shadow_result["cases"]
    )

    value_contract = payload["exact_value_shadow_contract"]
    assert (
        value_contract["role"]
        == "evaluation_only_exact_rounded_value_shadow_no_selection"
    )
    assert value_contract["filtration_value_source"] == (
        "correctly_rounded_exact_rationals"
    )
    assert value_contract["threshold_and_objective_arithmetic"] == "floating_point"
    assert value_contract["selection_effect"] == "none"
    assert "not an end-to-end exact alpha complex" in value_contract["claim_boundary"]

    value_result = payload["exact_value_shadow"]["result"]
    assert value_result["accepted_backend_case_count"] == 6
    assert value_result["audited_case_count"] == 6
    assert value_result["floating_connectivity_shadow_case_count"] == 6
    assert value_result["shadow_case_count"] == 6
    assert value_result["primary_output_difference_case_count"] == 0
    assert value_result["value_only_output_difference_case_count"] == 0
    assert value_result["selected_alpha_difference_case_count"] == 0
    assert value_result["objective_difference_case_count"] == 0
    assert value_result["endpoint_difference_case_count"] == 0
    assert value_result["candidate_bookkeeping_difference_case_count"] == 0
    assert value_result["all_prerequisite_cases_evaluated"]
    assert not value_result["primary_benchmark_results_changed"]
    assert value_result["selection_effect"] == "none"
    assert not value_result["promotion_supported"]
    assert value_result["blocking_reasons"] == [
        "exact_rounded_value_shadow_not_deployed"
    ]
    assert all(
        case["exact_audit_verified"]
        and case["shadow_ran"]
        and case["all_nonruntime_outputs_match_primary"]
        and case["all_nonruntime_outputs_match_floating_connectivity_shadow"]
        for case in value_result["cases"]
    )


def test_builtin_exact_backend_requires_construction_audit() -> None:
    with pytest.raises(SystemExit):
        main(["--exact-python-backend", "--methods", "B1"])


def test_builtin_exact_backend_conflicts_with_external_backend() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-construction",
                "--exact-python-backend",
                "--exact-backend",
                "fake-backend",
                "--methods",
                "B1",
            ]
        )


def test_exact_filtration_audit_requires_construction_audit() -> None:
    with pytest.raises(SystemExit):
        main(["--evaluate-exact-filtration-values", "--methods", "B1"])


def test_exact_connectivity_shadow_requires_construction_audit() -> None:
    with pytest.raises(SystemExit):
        main(["--evaluate-exact-connectivity-shadow", "--methods", "B1"])


def test_exact_connectivity_shadow_requires_connectivity_dependent_method() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-construction",
                "--evaluate-exact-connectivity-shadow",
                "--methods",
                "B0",
            ]
        )


def test_exact_value_shadow_requires_full_audit_chain() -> None:
    with pytest.raises(SystemExit):
        main(["--evaluate-exact-value-shadow", "--methods", "B1"])
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-construction",
                "--evaluate-exact-value-shadow",
                "--methods",
                "B1",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-construction",
                "--evaluate-exact-filtration-values",
                "--evaluate-exact-value-shadow",
                "--methods",
                "B1",
            ]
        )


def test_cli_audits_exact_critical_index_identity(tmp_path: Path) -> None:
    output = tmp_path / "exact_critical_index.json"

    exit_code = main(
        [
            "--split",
            "held_out",
            "--methods",
            "B2",
            "B3",
            "--evaluate-exact-construction",
            "--exact-python-backend",
            "--evaluate-exact-connectivity-shadow",
            "--evaluate-exact-filtration-values",
            "--evaluate-exact-value-shadow",
            "--evaluate-exact-critical-index-audit",
            "--evaluate-exact-resampling-threshold-audit",
            "--evaluate-exact-resampling-filtration-audit",
            "--evaluate-exact-b3-selection-shadow",
            "--point-count",
            "16",
            "--reference-count",
            "24",
            "--surface-samples",
            "12",
            "--b3-candidate-budget",
            "4",
            "--seed",
            "910",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 25
    contract = payload["exact_critical_index_audit_contract"]
    assert contract["role"] == ("evaluation_only_exact_critical_index_identity_audit")
    assert contract["methods"] == ["B2", "B3"]
    assert "selected_critical_rank" in contract["critical_identity"]
    assert contract["selection_effect"] == "none"

    audit = payload["exact_critical_index_audit"]
    assert audit["enabled"]
    result = audit["result"]
    assert result["requested_methods"] == ["B2", "B3"]
    assert result["accepted_backend_case_count"] == 6
    assert result["audited_case_count"] == 6
    assert result["critical_count_mismatch_case_count"] == 0
    assert result["birth_group_mismatch_case_count"] == 0
    assert 0 <= result["selected_index_mismatch_method_count"] <= 12
    assert 0 <= result["selected_complex_mismatch_method_count"] <= 12
    assert 0 <= result["selected_boundary_mismatch_method_count"] <= 12
    assert result["b3_signature_mismatch_case_count"] == 0
    assert result["b3_candidate_index_mismatch_case_count"] == 0
    assert result["all_selection_identities_match"] == (
        result["selected_index_mismatch_method_count"] == 0
        and result["selected_complex_mismatch_method_count"] == 0
        and result["selected_boundary_mismatch_method_count"] == 0
    )
    assert not result["primary_benchmark_results_changed"]
    assert result["selection_effect"] == "none"
    assert not result["promotion_supported"]
    assert result["blocking_reasons"] == ["exact_critical_index_audit_not_deployed"]
    assert all(
        case["critical_birth_group_sequence_matches"]
        and len(case["method_identities"]) == 2
        for case in result["cases"]
    )

    resampling_contract = payload["exact_resampling_threshold_audit_contract"]
    assert (
        resampling_contract["role"]
        == "evaluation_only_exact_selected_threshold_resampling_audit"
    )
    assert resampling_contract["method"] == "B3"
    assert not resampling_contract["exact_resampled_connectivity_constructed"]
    assert resampling_contract["selection_effect"] == "none"

    resampling_audit = payload["exact_resampling_threshold_audit"]
    assert resampling_audit["enabled"]
    resampling_result = resampling_audit["result"]
    assert resampling_result["accepted_backend_case_count"] == 6
    assert 0 <= resampling_result["audited_case_count"] <= 6
    assert resampling_result["threshold_effect_reproduction_failure_case_count"] == 0
    assert not resampling_result["exact_resampled_connectivity_constructed"]
    assert not resampling_result["primary_benchmark_results_changed"]
    assert resampling_result["selection_effect"] == "none"
    assert not resampling_result["promotion_supported"]
    assert all(
        case["threshold_effect_reproduced"]
        for case in resampling_result["cases"]
        if case["status"] == "audited"
    )

    filtration_contract = payload["exact_resampling_filtration_audit_contract"]
    assert filtration_contract["role"] == (
        "evaluation_only_exact_resampling_connectivity_and_filtration_audit"
    )
    assert filtration_contract["method"] == "B3"
    assert filtration_contract["exact_resampled_connectivity_constructed"]
    assert filtration_contract["exact_resampled_filtration_values_constructed"]
    assert filtration_contract["selection_effect"] == "none"

    filtration_audit = payload["exact_resampling_filtration_audit"]
    assert filtration_audit["enabled"]
    filtration_result = filtration_audit["result"]
    assert filtration_result["requested_case_count"] == 6
    assert filtration_result["requested_repeat_count"] == 12
    assert filtration_result["audited_repeat_count"] == (
        2 * filtration_result["audited_case_count"]
    )
    assert filtration_result["rejected_repeat_count"] == (
        12 - filtration_result["audited_repeat_count"]
    )
    assert filtration_result["selected_complex_difference_repeat_count"] == (
        filtration_result["selected_boundary_difference_repeat_count"]
    )
    assert not filtration_result["primary_benchmark_results_changed"]
    assert filtration_result["selection_effect"] == "none"
    assert not filtration_result["promotion_supported"]
    assert all(
        case["floating_recomputation_matches_threshold_audit"]
        for case in filtration_result["cases"]
        if case["status"] == "audited"
    )

    b3_shadow_contract = payload["exact_b3_selection_shadow_contract"]
    assert b3_shadow_contract["role"] == (
        "evaluation_only_exact_resampling_B3_selection_shadow"
    )
    assert b3_shadow_contract["method"] == "B3"
    assert b3_shadow_contract["candidate_source"] == (
        "same_budgeted_exact_full_critical_values"
    )
    assert b3_shadow_contract["selection_effect"] == "none"

    b3_shadow = payload["exact_b3_selection_shadow"]
    assert b3_shadow["enabled"]
    b3_shadow_result = b3_shadow["result"]
    assert b3_shadow_result["requested_case_count"] == 6
    assert b3_shadow_result["schema24_audited_case_count"] == (
        filtration_result["audited_case_count"]
    )
    assert b3_shadow_result["shadow_case_count"] == (
        filtration_result["audited_case_count"]
    )
    assert b3_shadow_result["all_prerequisite_cases_evaluated"] == (
        b3_shadow_result["shadow_case_count"]
        == b3_shadow_result["requested_case_count"]
    )
    assert not b3_shadow_result["primary_benchmark_results_changed"]
    assert b3_shadow_result["selection_effect"] == "none"
    assert not b3_shadow_result["promotion_supported"]
    assert b3_shadow_result["blocking_reasons"][-1] == (
        "exact_b3_selection_shadow_not_deployed"
    )
    if filtration_result["audited_case_count"] < 6:
        assert "one_or_more_schema24_exact_resampling_cases_missing" in (
            b3_shadow_result["blocking_reasons"]
        )
    assert all(
        case["reference_report_reproduced"]
        and case["candidate_count"] == len(case["candidates"])
        and all(
            candidate["nonstability_terms_match"]
            for candidate in case["candidates"]
        )
        for case in b3_shadow_result["cases"]
        if case["status"] == "audited"
    )


def test_exact_critical_index_audit_requires_value_shadow_and_b2_or_b3() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-critical-index-audit",
                "--methods",
                "B2",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-resampling-filtration-audit",
                "--methods",
                "B3",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-b3-selection-shadow",
                "--methods",
                "B3",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-construction",
                "--evaluate-exact-connectivity-shadow",
                "--evaluate-exact-filtration-values",
                "--evaluate-exact-value-shadow",
                "--evaluate-exact-critical-index-audit",
                "--methods",
                "B1",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-resampling-threshold-audit",
                "--methods",
                "B3",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--evaluate-exact-construction",
                "--evaluate-exact-connectivity-shadow",
                "--evaluate-exact-filtration-values",
                "--evaluate-exact-value-shadow",
                "--evaluate-exact-critical-index-audit",
                "--evaluate-exact-resampling-threshold-audit",
                "--methods",
                "B2",
            ]
        )
