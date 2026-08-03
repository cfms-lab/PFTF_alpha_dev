from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pftf_alpha.s3dis_two_layer_intake_protocol import (
    preregister_s3dis_two_layer_intake,
    write_protocol,
)


def test_phase51_intake_protocol_reserves_building_disjoint_area5() -> None:
    protocol = preregister_s3dis_two_layer_intake()

    assert protocol.calibration_areas == (
        "Area_1",
        "Area_2",
        "Area_3",
        "Area_4",
        "Area_6",
    )
    assert protocol.reserved_held_out_area == "Area_5"
    assert protocol.target_classes == ("board", "wall")
    assert "do not extract" in protocol.reserved_content_prohibition
    assert "XYZ coordinates only" in protocol.runtime_information_boundary
    assert not protocol.current_support_flags["real_scan_supported"]
    assert not protocol.current_support_flags["held_out_validation_supported"]


def test_phase51_intake_protocol_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_protocol(first)
    second_digest = write_protocol(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_digest == second_digest
    assert first_digest == hashlib.sha256(first.read_bytes()).hexdigest()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["artifact_schema"].endswith("phase51/v1")
    assert payload["reserved_held_out_area"] == "Area_5"
