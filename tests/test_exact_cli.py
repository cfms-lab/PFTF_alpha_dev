import json
from pathlib import Path

from pftf_alpha.benchmark import main


def test_cli_audits_predicates_without_claiming_exact_construction(
    tmp_path: Path,
) -> None:
    output = tmp_path / "exact_predicates.json"

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
            "--evaluate-exact-predicates",
            "--point-count",
            "24",
            "--reference-count",
            "48",
            "--surface-samples",
            "24",
            "--adaptive-knn",
            "6",
            "--seed",
            "907",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 25
    contract = payload["exact_predicate_audit_contract"]
    assert contract["role"] == "readiness_audit_no_selection"
    assert contract["construction_effect"] == "none"
    assert contract["selection_effect"] == "none"
    assert "not exact construction" in contract["claim_boundary"]

    audit = payload["exact_predicate_audit"]
    assert audit["enabled"]
    assert audit["source_split"] == "held_out"
    result = audit["result"]
    assert result["role"] == "readiness_audit_no_selection"
    assert result["coordinate_model"] == "binary64_values_as_exact_rationals"
    assert result["triangulation_source"] == "SciPy_Qhull_floating_point"
    assert not result["exact_construction_backend_integrated"]
    assert not result["changes_benchmark_selection"]
    assert not result["promotion_supported"]
    assert "no_exact_construction_backend" in result["blocking_reasons"]
    assert result["totals"]["case_count"] == 6
    assert len(result["cases"]) == 6

    benchmark_results = [row for case in payload["cases"] for row in case["results"]]
    assert all(
        row["selection_mode"] == "frozen_local_scale_multiplier"
        and row["selection_parameter_value"] == 1.2
        for row in benchmark_results
    )
