from __future__ import annotations

import hashlib
from pathlib import Path

from pftf_alpha.s3dis_room_layer_protocol import (
    preregister_s3dis_room_layer,
    write_protocol,
)


def test_phase51b_protocol_keeps_area5_reserved() -> None:
    protocol = preregister_s3dis_room_layer()

    assert protocol.target_classes == ("floor", "ceiling")
    assert protocol.reserved_held_out_area == "Area_5"
    assert "long-gap" in protocol.regime_boundary
    assert "do not extract" in protocol.reserved_content_prohibition
    assert not protocol.current_support_flags["real_scan_supported"]


def test_phase51b_protocol_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_digest = write_protocol(first)
    second_digest = write_protocol(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_digest == second_digest
    assert first_digest == hashlib.sha256(first.read_bytes()).hexdigest()
