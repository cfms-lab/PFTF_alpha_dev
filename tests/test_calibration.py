import math
from dataclasses import replace

import numpy as np
import pytest

from pftf_alpha.baselines import BaselineID, BenchmarkConfig, run_case_benchmarks
from pftf_alpha.calibration import (
    calibrate_adaptive_multiplier,
    calibrate_p2_confidence_threshold,
)
from pftf_alpha.synthetic import PanelSplit, make_minimal_panel


@pytest.mark.parametrize(
    "method,field_name",
    [
        (BaselineID.B4_DENSITY_SCALED, "b4_scale_multiplier"),
        (BaselineID.B5_PCA_ANISOTROPIC, "b5_scale_multiplier"),
        (BaselineID.P1_PFTF_LOCAL_SPD, "p1_scale_multiplier"),
        (BaselineID.P2_CONFIDENCE_FALLBACK, "p2_scale_multiplier"),
    ],
)
def test_adaptive_calibration_freezes_one_panel_wide_multiplier(
    method: BaselineID,
    field_name: str,
) -> None:
    calibration_cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=731,
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        seed=732,
    )

    calibration = calibrate_adaptive_multiplier(
        calibration_cases,
        method,
        config=config,
        candidate_budget=4,
    )

    assert calibration.method is method
    assert calibration.calibration_case_count == len(calibration_cases)
    assert calibration.candidate_count >= 4
    assert calibration.candidate_min <= calibration.multiplier
    assert calibration.multiplier <= calibration.candidate_max
    assert math.isfinite(calibration.selected_mean_objective)

    frozen_config = replace(config, **{field_name: calibration.multiplier})
    held_out_case = make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=24,
        reference_count=48,
        seed=731,
    )[0]
    result = run_case_benchmarks(
        held_out_case,
        config=frozen_config,
        methods=[method],
    ).results[0]

    assert not result.uses_reference_for_selection
    assert result.selection_mode == "frozen_local_scale_multiplier"
    assert result.selection_parameter_value == calibration.multiplier


def test_adaptive_calibration_rejects_nonadaptive_method() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=740,
    )

    with pytest.raises(ValueError, match="only B4, B5, P1, or P2"):
        calibrate_adaptive_multiplier(
            cases,
            BaselineID.B3_PERSISTENCE_STABILITY,
            config=BenchmarkConfig(surface_sample_count=24),
            candidate_budget=4,
        )


def test_p2_confidence_calibration_is_reference_free_and_deterministic() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=24,
        reference_count=48,
        seed=750,
    )
    config = BenchmarkConfig(
        surface_sample_count=24,
        adaptive_k_neighbors=6,
        seed=751,
    )

    first = calibrate_p2_confidence_threshold(
        cases,
        config=config,
        target_fallback_fraction=0.25,
    )
    altered_references = tuple(
        replace(case, reference_points=np.zeros_like(case.reference_points))
        for case in cases
    )
    second = calibrate_p2_confidence_threshold(
        altered_references,
        config=config,
        target_fallback_fraction=0.25,
    )

    assert first == second
    assert not first.uses_reference_for_selection
    assert first.calibration_case_count == len(cases)
    assert 0.0 <= first.threshold <= 1.0
    assert first.achieved_fallback_fraction == pytest.approx(
        first.fallback_count / first.cell_count
    )
    assert first.per_case_fallback_min <= first.per_case_fallback_median
    assert first.per_case_fallback_median <= first.per_case_fallback_max


def test_p2_confidence_calibration_rejects_boundary_target() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=16,
        reference_count=24,
        seed=760,
    )
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        calibrate_p2_confidence_threshold(
            cases,
            config=BenchmarkConfig(),
            target_fallback_fraction=0.0,
        )
