import json
from pathlib import Path

from pftf_alpha.benchmark import main


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
    assert payload["schema_version"] == 17
    contract = payload["exact_construction_backend_contract"]
    assert contract["role"] == "optional_backend_handoff_validation_no_selection"
    assert contract["construction_effect"] == "validated_connectivity_not_applied"
    assert contract["selection_effect"] == "none"

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
