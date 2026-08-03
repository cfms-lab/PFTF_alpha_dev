import numpy as np
import pytest

from pftf_alpha.confidence_alpha_panel import (
    CALIBRATION_SEEDS,
    HELD_OUT_SEEDS,
    ConfidenceAlphaCase,
    MisregistrationProfile,
    ReferenceSurfaceFamily,
    make_confidence_alpha_case,
    make_confidence_alpha_panel,
    preregister_confidence_alpha_panel,
)
from pftf_alpha.synthetic import PanelSplit


def test_phase43_panel_is_disjoint_and_preregistered() -> None:
    protocol = preregister_confidence_alpha_panel()

    assert set(CALIBRATION_SEEDS).isdisjoint(HELD_OUT_SEEDS)
    assert protocol.calibration_case_count == 6
    assert protocol.held_out_case_count == 18
    assert protocol.families == (
        "sphere",
        "torus",
        "disconnected_spheres",
    )
    assert "evaluation-only" in protocol.reference_boundary
    assert "one global alpha" in protocol.claim_boundary


@pytest.mark.parametrize("family", list(ReferenceSurfaceFamily))
@pytest.mark.parametrize("profile", list(MisregistrationProfile))
def test_phase43_case_is_deterministic_and_keeps_views_separate(
    family: ReferenceSurfaceFamily,
    profile: MisregistrationProfile,
) -> None:
    case = make_confidence_alpha_case(
        family,
        profile,
        split=PanelSplit.CALIBRATION,
        seed=CALIBRATION_SEEDS[0],
    )
    repeated = make_confidence_alpha_case(
        family,
        profile,
        split=PanelSplit.CALIBRATION,
        seed=CALIBRATION_SEEDS[0],
    )

    assert isinstance(case, ConfidenceAlphaCase)
    assert case.anchor_points.shape == (72, 3)
    assert case.target_points.shape == (72, 3)
    assert case.reference_points.shape == (1536, 3)
    np.testing.assert_array_equal(case.points, repeated.points)
    np.testing.assert_array_equal(case.reference_points, repeated.reference_points)
    np.testing.assert_array_equal(case.point_view_labels[:72], 0)
    np.testing.assert_array_equal(case.point_view_labels[72:], 1)
    assert case.expected_surface_betti[0] == case.expected_components


def test_phase43_heldout_panel_has_three_repeats_per_block() -> None:
    panel = make_confidence_alpha_panel(PanelSplit.HELD_OUT)

    assert len(panel) == 18
    blocks = {(case.family, case.profile) for case in panel}
    assert len(blocks) == 6
    for block in blocks:
        seeds = {
            case.seed for case in panel if (case.family, case.profile) == block
        }
        assert seeds == set(HELD_OUT_SEEDS)


def test_phase43_rejects_training_split() -> None:
    with pytest.raises(ValueError, match="calibration or held_out"):
        make_confidence_alpha_case(
            ReferenceSurfaceFamily.SPHERE,
            MisregistrationProfile.MILD,
            split=PanelSplit.TRAIN,
            seed=1,
        )
