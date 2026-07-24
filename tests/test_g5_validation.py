import json
from pathlib import Path

from pftf_alpha.baselines import BaselineID
from pftf_alpha.g5_validation import (
    ADAPTIVE_METHODS,
    default_g5_profiles,
    evaluate_g5_preflight,
    main,
)


def test_default_profiles_declare_density_noise_and_geometry_shifts() -> None:
    profiles = default_g5_profiles(point_count=24, reference_count=48)

    assert [profile.name for profile in profiles] == [
        "base",
        "sparse",
        "noisy",
        "hard_geometry",
    ]
    assert profiles[1].point_count == 16
    assert profiles[2].noise_scale == 2.0
    assert profiles[3].geometry_scale == 0.75


def test_g5_preflight_freezes_calibration_and_never_promotes_smoke() -> None:
    result = evaluate_g5_preflight(
        point_count=16,
        reference_count=24,
        surface_sample_count=8,
        candidate_budget=2,
        adaptive_k_neighbors=4,
        repeat_count=1,
        seed=311,
    )
    payload = result.to_dict()

    assert payload["artifact_schema"] == "pftf_alpha_g5_preflight/v1"
    assert payload["selection_contract"]["held_out_tuning"] == "prohibited"
    assert payload["calibration"]["source_split"] == "calibration"
    assert [row["method"] for row in payload["calibration"]["multipliers"]] == [
        method.value for method in ADAPTIVE_METHODS
    ]
    assert all(
        payload["calibration"]["config"][name] is not None
        for name in (
            "b4_scale_multiplier",
            "b5_scale_multiplier",
            "p1_scale_multiplier",
            "p2_scale_multiplier",
        )
    )
    assert len(payload["cases"]) == 24
    assert all(case["split"] == "held_out" for case in payload["cases"])
    assert all(
        not method["uses_reference_for_selection"]
        for case in payload["cases"]
        for method in case["results"]
    )
    assert len(payload["summaries"]) == 16
    assert len(payload["comparisons"]) == 8
    assert len(payload["profile_shifts"]) == 12
    p2_summaries = [
        summary
        for summary in payload["summaries"]
        if summary["method"] == BaselineID.P2_CONFIDENCE_FALLBACK.value
    ]
    assert all(
        summary["fallback_guard_violation_count"] == 0
        for summary in p2_summaries
    )
    assert not payload["promotion_supported"]
    assert "g4_exact_or_validated_fail_closed_fallback_not_deployed" in payload[
        "promotion_blockers"
    ]


def test_g5_cli_writes_declared_artifact(tmp_path: Path) -> None:
    output = tmp_path / "g5.json"

    exit_code = main(
        [
            "--point-count",
            "16",
            "--reference-count",
            "24",
            "--surface-samples",
            "8",
            "--adaptive-calibration-budget",
            "2",
            "--adaptive-knn",
            "4",
            "--repeats",
            "1",
            "--seed",
            "312",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_schema"] == "pftf_alpha_g5_preflight/v1"
    assert payload["evaluation_role"] == "synthetic_frozen_held_out_preflight"
