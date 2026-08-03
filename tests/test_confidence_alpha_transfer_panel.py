import numpy as np
import pytest

from pftf_alpha.confidence_alpha_panel import ReferenceSurfaceFamily
from pftf_alpha.confidence_alpha_transfer_panel import (
    CALIBRATION_SEEDS,
    HELD_OUT_SEEDS,
    ConfidenceAlphaTransferCase,
    TransferStressProfile,
    make_confidence_alpha_transfer_case,
    make_confidence_alpha_transfer_panel,
    preregister_confidence_alpha_transfer_panel,
)
from pftf_alpha.synthetic import PanelSplit


def test_phase44_protocol_freezes_disjoint_transfer_panel() -> None:
    protocol = preregister_confidence_alpha_transfer_panel()

    assert set(CALIBRATION_SEEDS).isdisjoint(HELD_OUT_SEEDS)
    assert protocol.calibration_case_count == 9
    assert protocol.held_out_case_count == 27
    assert "every finite unique top-cell score" in protocol.critical_score_selection
    assert "at least 2/3" in protocol.validation_gate


@pytest.mark.parametrize("family", list(ReferenceSurfaceFamily))
@pytest.mark.parametrize("profile", list(TransferStressProfile))
def test_phase44_cases_are_deterministic_and_profiled(
    family: ReferenceSurfaceFamily,
    profile: TransferStressProfile,
) -> None:
    case = make_confidence_alpha_transfer_case(
        family,
        profile,
        split=PanelSplit.CALIBRATION,
        seed=CALIBRATION_SEEDS[0],
    )
    repeated = make_confidence_alpha_transfer_case(
        family,
        profile,
        split=PanelSplit.CALIBRATION,
        seed=CALIBRATION_SEEDS[0],
    )

    assert isinstance(case, ConfidenceAlphaTransferCase)
    np.testing.assert_array_equal(case.points, repeated.points)
    np.testing.assert_array_equal(case.reference_points, repeated.reference_points)
    assert case.reference_points.shape == (1536, 3)
    assert case.expected_surface_betti[0] == case.expected_components
    if profile is TransferStressProfile.DENSITY_SHIFT:
        assert case.anchor_points.shape[0] == 48
        assert case.target_points.shape[0] == 96
    elif profile is TransferStressProfile.TARGET_OCCLUSION:
        assert case.occlusion_retained_fraction == 0.5
    else:
        assert case.local_warp_fraction == 0.08


def test_phase44_heldout_has_three_repeats_per_family_profile() -> None:
    panel = make_confidence_alpha_transfer_panel(PanelSplit.HELD_OUT)

    assert len(panel) == 27
    blocks = {(case.family, case.profile) for case in panel}
    assert len(blocks) == 9
    for block in blocks:
        assert {
            case.seed for case in panel if (case.family, case.profile) == block
        } == set(HELD_OUT_SEEDS)


def test_phase44_rejects_training_split() -> None:
    with pytest.raises(ValueError, match="calibration or held_out"):
        make_confidence_alpha_transfer_panel(PanelSplit.TRAIN)
