import json
from pathlib import Path

from pftf_alpha.benchmark import main


def test_cli_calibrates_and_freezes_p2(tmp_path: Path) -> None:
    output = tmp_path / "p2.json"

    exit_code = main(
        [
            "--split",
            "held_out",
            "--methods",
            "P2",
            "--calibrate-adaptive",
            "--adaptive-calibration-budget",
            "4",
            "--point-count",
            "24",
            "--reference-count",
            "48",
            "--surface-samples",
            "24",
            "--adaptive-knn",
            "6",
            "--seed",
            "901",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 25
    topology_contract = payload["topology_endpoint_contract"]
    assert topology_contract["homology_coefficients"] == "GF(2)"
    assert topology_contract["betti_target_role"] == "evaluation_only"
    assert topology_contract["selection_topology_term"] == "component_error"
    assert topology_contract["labeled_false_bridge_role"] == "evaluation_only"
    bridge_contract = payload["bridge_risk_probe_contract"]
    assert bridge_contract["role"] == "evaluation_only"
    assert bridge_contract["risk_inputs"] == "observed_points_only"
    assert not bridge_contract["risk_uses_reference_or_labels"]
    assert bridge_contract["labels_role"] == "AUC_recall_FPR_evaluation_only"
    assert bridge_contract["selection_effect"] == "none"
    penalty_contract = payload["bridge_penalty_ablation_contract"]
    assert penalty_contract["role"] == "calibration_only_evaluation_no_selection"
    assert penalty_contract["risk_inputs"] == "observed_points_only"
    assert penalty_contract["selection_effect"] == "none"
    assert not payload["bridge_penalty_ablation"]["enabled"]
    boundary_contract = payload["boundary_bridge_localization_contract"]
    assert boundary_contract["role"] == "evaluation_only_no_selection"
    assert boundary_contract["risk_inputs"] == (
        "observed_points_and_frozen_P2_boundary_only"
    )
    assert not boundary_contract["reference_geometry_use"]
    assert boundary_contract["selection_effect"] == "none"
    assert not payload["boundary_bridge_localization"]["enabled"]
    intervention_contract = payload["boundary_owner_intervention_contract"]
    assert intervention_contract["role"] == ("calibration_only_evaluation_no_selection")
    assert intervention_contract["boundary_recomputation"] == "after_every_round"
    assert intervention_contract["selection_effect"] == "none"
    assert intervention_contract["held_out_tuning"] == "prohibited"
    assert not payload["boundary_owner_intervention"]["enabled"]
    region_cut_contract = payload["boundary_region_cut_ablation_contract"]
    assert region_cut_contract["role"] == ("calibration_only_evaluation_no_selection")
    assert region_cut_contract["region_adjacency"] == (
        "flagged_faces_share_flagged_boundary_edge"
    )
    assert region_cut_contract["selection_effect"] == "none"
    assert region_cut_contract["held_out_tuning"] == "prohibited"
    assert not payload["boundary_region_cut_ablation"]["enabled"]
    assert payload["config"]["p2_scale_multiplier"] is not None
    assert [
        result["method"] for result in payload["adaptive_calibration"]["results"]
    ] == ["P2"]
    confidence_calibration = payload["p2_confidence_calibration"]
    assert confidence_calibration["enabled"]
    assert not confidence_calibration["explicit_threshold"]
    assert not confidence_calibration["result"]["uses_reference_for_selection"]
    assert (
        payload["config"]["p2_confidence_threshold"]
        == confidence_calibration["result"]["threshold"]
    )

    results = [result for case in payload["cases"] for result in case["results"]]
    assert len(results) == 6
    assert all(not result["uses_reference_for_selection"] for result in results)
    assert all(
        result["selection_mode"] == "frozen_local_scale_multiplier"
        for result in results
    )
    assert all(result["candidate_count"] == 1 for result in results)
    assert all(len(case["expected_surface_betti"]) == 3 for case in payload["cases"])
    assert all(
        len(case["point_component_sizes"]) == case["expected_components"]
        and sum(case["point_component_sizes"]) == case["point_count"]
        for case in payload["cases"]
    )
    probes = [case["bridge_risk_probe"] for case in payload["cases"]]
    assert all(probe is not None for probe in probes)
    for case, probe in zip(payload["cases"], probes, strict=True):
        assert probe["selection_role"] == "evaluation_only"
        assert probe["risk_inputs"] == "observed_points_only"
        assert not probe["uses_reference_or_labels_for_risk"]
        assert probe["uses_component_labels_for_evaluation"]
        assert probe["cell_count"] == (
            probe["labeled_mixed_cell_count"]
            + probe["labeled_same_component_cell_count"]
        )
        assert (probe["labeled_auc"] is not None) == (case["expected_components"] > 1)
        assert probe["normal_coherence_threshold"] == 0.9
        assert probe["normal_edge_threshold"] == 0.02
        assert probe["length_edge_threshold"] == 1.8
    assert all(
        result["endpoints"]["labeled_false_bridge_edges"] is not None
        for result in results
    )
    assert all(
        result["endpoints"]["labeled_false_bridge_faces"] is not None
        for result in results
    )
    assert all(result["method_diagnostics"] is not None for result in results)
    assert all(result["endpoints"]["betti_error"] is not None for result in results)
    assert all(
        result["endpoints"]["betti_0"]
        - result["endpoints"]["betti_1"]
        + result["endpoints"]["betti_2"]
        == result["endpoints"]["euler_characteristic"]
        for result in results
    )
    assert all(
        result["endpoints"]["labeled_false_bridge_present"]
        == (result["endpoints"]["labeled_false_bridge_edges"] > 0)
        for result in results
    )
    assert all(
        result["method_diagnostics"]["fallback_guard_violation_count"] == 0
        for result in results
    )
    assert all(
        0.0 <= result["method_diagnostics"]["selected_fallback_fraction"] <= 1.0
        for result in results
    )
    assert all(
        result["method_diagnostics"]["selected_guard_violation_count"] == 0
        for result in results
    )
    assert all(
        result["method_diagnostics"]["downward_closure_complete"] == 1
        for result in results
    )
    assert all(
        result["method_diagnostics"]["face_incidence_over_two_count"] == 0
        for result in results
    )


def test_cli_respects_explicit_p2_confidence_threshold(tmp_path: Path) -> None:
    output = tmp_path / "p2_explicit.json"

    exit_code = main(
        [
            "--split",
            "held_out",
            "--methods",
            "P2",
            "--calibrate-adaptive",
            "--p2-confidence-threshold",
            "0.3",
            "--adaptive-calibration-budget",
            "2",
            "--point-count",
            "16",
            "--reference-count",
            "24",
            "--surface-samples",
            "16",
            "--adaptive-knn",
            "5",
            "--seed",
            "902",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    confidence_calibration = payload["p2_confidence_calibration"]
    assert not confidence_calibration["enabled"]
    assert confidence_calibration["explicit_threshold"]
    assert confidence_calibration["result"] is None
    assert payload["config"]["p2_confidence_threshold"] == 0.3
    results = [result for case in payload["cases"] for result in case["results"]]
    assert all(
        result["method_diagnostics"]["confidence_threshold"] == 0.3
        for result in results
    )


def test_cli_evaluates_bridge_penalty_without_deploying_it(tmp_path: Path) -> None:
    output = tmp_path / "bridge_penalty.json"

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
            "--evaluate-bridge-penalty",
            "--bridge-penalty-strengths",
            "0",
            "0.2",
            "0.8",
            "--point-count",
            "16",
            "--reference-count",
            "24",
            "--surface-samples",
            "16",
            "--adaptive-knn",
            "5",
            "--seed",
            "903",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 25
    ablation = payload["bridge_penalty_ablation"]
    assert ablation["enabled"]
    assert ablation["source_split"] == "calibration"
    assert ablation["requested_strengths"] == [0.0, 0.2, 0.8]
    result = ablation["result"]
    assert result["role"] == "evaluation_only_no_selection"
    assert not result["changes_benchmark_selection"]
    assert result["candidate_count"] == 3
    assert [point["strength"] for point in result["curve"]] == [0.0, 0.2, 0.8]
    assert payload["config"]["p2_scale_multiplier"] == 1.2
    benchmark_results = [row for case in payload["cases"] for row in case["results"]]
    assert all(
        row["selection_mode"] == "frozen_local_scale_multiplier"
        and row["selection_parameter_value"] == 1.2
        for row in benchmark_results
    )


def test_cli_evaluates_boundary_bridges_without_changing_p2(tmp_path: Path) -> None:
    output = tmp_path / "boundary_bridges.json"

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
            "--evaluate-boundary-bridges",
            "--point-count",
            "24",
            "--reference-count",
            "48",
            "--surface-samples",
            "24",
            "--adaptive-knn",
            "6",
            "--seed",
            "904",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 25
    localization = payload["boundary_bridge_localization"]
    assert localization["enabled"]
    assert localization["source_split"] == "held_out"
    result = localization["result"]
    assert result["role"] == "evaluation_only_no_selection"
    assert result["evaluation_split"] == "held_out"
    assert not result["uses_reference_geometry"]
    assert result["uses_component_labels_for_evaluation"]
    assert not result["changes_benchmark_selection"]
    assert result["risk_threshold"] == 1.0
    assert result["case_count"] == 6
    assert result["pooled_boundary_face_count"] > 0
    assert result["pooled_boundary_edge_count"] > 0
    assert len(result["cases"]) == 6
    benchmark_results = [row for case in payload["cases"] for row in case["results"]]
    assert all(
        row["selection_mode"] == "frozen_local_scale_multiplier"
        and row["selection_parameter_value"] == 1.2
        for row in benchmark_results
    )


def test_cli_evaluates_boundary_intervention_without_changing_p2(
    tmp_path: Path,
) -> None:
    output = tmp_path / "boundary_intervention.json"

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
            "--evaluate-boundary-intervention",
            "--boundary-intervention-rounds",
            "0",
            "1",
            "2",
            "--point-count",
            "24",
            "--reference-count",
            "48",
            "--surface-samples",
            "24",
            "--adaptive-knn",
            "6",
            "--seed",
            "905",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 25
    ablation = payload["boundary_owner_intervention"]
    assert ablation["enabled"]
    assert ablation["source_split"] == "calibration"
    assert ablation["requested_rounds"] == [0, 1, 2]
    result = ablation["result"]
    assert result["role"] == "calibration_only_evaluation_no_selection"
    assert not result["changes_benchmark_selection"]
    assert result["recomputes_boundary_each_round"]
    assert result["candidate_count"] == 3
    assert [point["rounds"] for point in result["curve"]] == [0, 1, 2]
    assert result["curve"][0]["removed_cell_count"] == 0
    assert payload["config"]["p2_scale_multiplier"] == 1.2
    benchmark_results = [row for case in payload["cases"] for row in case["results"]]
    assert all(
        row["selection_mode"] == "frozen_local_scale_multiplier"
        and row["selection_parameter_value"] == 1.2
        for row in benchmark_results
    )


def test_cli_evaluates_boundary_region_cuts_without_changing_p2(
    tmp_path: Path,
) -> None:
    output = tmp_path / "boundary_region_cuts.json"

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
            "--evaluate-boundary-region-cuts",
            "--point-count",
            "24",
            "--reference-count",
            "48",
            "--surface-samples",
            "24",
            "--adaptive-knn",
            "6",
            "--seed",
            "906",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 25
    ablation = payload["boundary_region_cut_ablation"]
    assert ablation["enabled"]
    assert ablation["source_split"] == "calibration"
    assert ablation["requested_strategies"] == [
        "baseline",
        "largest_risk_region",
        "safe_backbone_cut",
    ]
    result = ablation["result"]
    assert result["role"] == "calibration_only_evaluation_no_selection"
    assert not result["changes_benchmark_selection"]
    assert result["requested_strategies"] == ablation["requested_strategies"]
    assert [point["strategy"] for point in result["curve"]] == (
        ablation["requested_strategies"]
    )
    baseline, _, safe_cut = result["curve"]
    assert baseline["removed_cell_count"] == 0
    assert safe_cut["safe_backbone_cut_edge_count"] == 0
    assert safe_cut["removed_cell_count"] == 0
    assert payload["config"]["p2_scale_multiplier"] == 1.2
    benchmark_results = [row for case in payload["cases"] for row in case["results"]]
    assert all(
        row["selection_mode"] == "frozen_local_scale_multiplier"
        and row["selection_parameter_value"] == 1.2
        for row in benchmark_results
    )
