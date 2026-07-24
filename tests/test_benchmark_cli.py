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
    assert payload["schema_version"] == 9
    topology_contract = payload["topology_endpoint_contract"]
    assert topology_contract["homology_coefficients"] == "GF(2)"
    assert topology_contract["betti_target_role"] == "evaluation_only"
    assert topology_contract["selection_topology_term"] == "component_error"
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
