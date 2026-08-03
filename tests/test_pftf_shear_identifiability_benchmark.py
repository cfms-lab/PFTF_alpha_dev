import json
from pathlib import Path

from pftf_alpha.pftf_shear_identifiability import (
    evaluate_pftf_shear_identifiability,
    write_result,
)


def test_phase49_result_flags_are_internally_consistent() -> None:
    result = evaluate_pftf_shear_identifiability()

    assert result.new_representation_development_justified == (
        result.pftf_specific_within_block_signal_supported
    )
    assert result.new_held_out_panel_justified == (
        result.pftf_specific_within_block_signal_supported
        and result.standalone_pftf_identifiability_supported
    )
    assert result.phase49_identifiability_supported == (
        result.new_held_out_panel_justified
    )


def test_phase49_result_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_result(first)
    second_digest = write_result(second)

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["artifact_schema"].endswith("phase49/v1")
    assert payload["prohibited_held_out_case_count"] == 0
