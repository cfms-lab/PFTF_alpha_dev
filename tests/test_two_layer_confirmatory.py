import json
from pathlib import Path

import numpy as np

from pftf_alpha.two_layer_confirmatory import (
    evaluate_two_layer_confirmatory,
    main,
    proper_rotation,
)


def test_phase50_rotation_is_deterministic_proper_and_orthogonal() -> None:
    first = proper_rotation(12345)
    second = proper_rotation(12345)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first @ first.T, np.eye(3), atol=1.0e-14)
    np.testing.assert_allclose(np.linalg.det(first), 1.0, atol=1.0e-14)


def test_phase50_reduced_panel_runs_without_claim_promotion() -> None:
    result = evaluate_two_layer_confirmatory(
        point_counts=(64,),
        stresses=("control",),
        repeats=1,
        reference_count=96,
        surface_sample_count=32,
        seed=1234,
    )

    assert result.case_count == 1
    assert result.m1_available_case_count in (0, 1)
    assert not result.protocol_identity_passed
    assert not result.phase50_supported
    assert not result.promotion_supported
    assert not result.pftf_superiority_supported
    assert abs(result.cases[0].rotation_determinant - 1.0) < 1.0e-14


def test_phase50_cli_writes_reduced_artifact(tmp_path: Path) -> None:
    output = tmp_path / "phase50.json"
    exit_code = main(
        [
            "--output",
            str(output),
            "--point-counts",
            "64",
            "--stresses",
            "control",
            "--repeats",
            "1",
            "--reference",
            "96",
            "--surface-samples",
            "32",
            "--seed",
            "1234",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_schema"].endswith("phase50/v1")
    assert payload["case_count"] == 1
    assert payload["phase50_supported"] is False
