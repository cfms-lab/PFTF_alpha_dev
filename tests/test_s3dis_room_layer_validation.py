from __future__ import annotations

import zipfile
from pathlib import Path

from pftf_alpha.s3dis_room_layer_validation import (
    audit_frozen_gates,
    extract_heldout_room_layers,
)
from pftf_alpha.s3dis_two_layer_intake_protocol import ARCHIVE_NAME


def _method(fscore: float, geometry: float, topology: int) -> dict[str, object]:
    return {
        "fscore": fscore,
        "geometry_loss": geometry,
        "topology_error": topology,
    }


def test_frozen_gate_audit_accepts_preregistered_positive_pattern() -> None:
    cases = [
        {
            "candidate": _method(0.80, 0.10, 1),
            "base": _method(0.795, 0.10, 1),
            "b5": _method(0.50, 0.20, 10),
            "m1": _method(0.40, 0.25, 20),
        }
        for _ in range(20)
    ]
    core = {
        "cases": cases,
        "candidate_safe_acceptance_coverage": 0.95,
        "base_safe_acceptance_coverage": 0.95,
        "candidate_false_safe_count": 0,
    }

    gates = audit_frozen_gates(core)

    assert gates.panel_gate_passed
    assert gates.safety_gate_passed
    assert gates.construction_gate_passed
    assert gates.geometry_gate_passed
    assert gates.topology_gate_passed
    assert gates.ablation_gate_passed


def test_heldout_extractor_opens_only_area5_room_layers(tmp_path: Path) -> None:
    archive = tmp_path / ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w") as bundle:
        for area in ("Area_4", "Area_5"):
            prefix = f"dataset/{area}/office_1/Annotations"
            bundle.writestr(f"{prefix}/floor_1.txt", "0 0 0\n")
            bundle.writestr(f"{prefix}/ceiling_1.txt", "0 0 3\n")
            bundle.writestr(f"{prefix}/wall_1.txt", "0 1 0\n")
    extraction = tmp_path / "heldout"

    result = extract_heldout_room_layers(archive, extraction)

    assert result.extracted_member_count == 2
    assert result.floor_member_count == 1
    assert result.ceiling_member_count == 1
    assert result.held_out_artifacts_accessed
    assert list(extraction.rglob("Area_5"))
    assert not list(extraction.rglob("Area_4"))
    assert not list(extraction.rglob("wall*.txt"))
