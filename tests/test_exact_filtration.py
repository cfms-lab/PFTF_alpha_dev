import sys
from fractions import Fraction

import numpy as np
import pytest

from pftf_alpha.exact_backend import evaluate_exact_construction_panel
from pftf_alpha.exact_filtration import (
    audit_exact_filtration_case,
    evaluate_exact_filtration_panel,
    exact_rounded_filtration,
    exact_simplex_filtration,
)
from pftf_alpha.synthetic import PanelSplit, make_minimal_panel


def _tetrahedron_points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )


def _exact_construction(cases):
    return evaluate_exact_construction_panel(
        ((case.family.value, case.points) for case in cases),
        evaluation_split=PanelSplit.HELD_OUT.value,
        backend_command=(
            sys.executable,
            "-m",
            "pftf_alpha.exact_python_backend",
        ),
        timeout_seconds=10.0,
    )


def test_exact_simplex_filtration_matches_known_tetrahedron_values() -> None:
    result = exact_simplex_filtration(
        _tetrahedron_points(),
        ((0, 1, 2, 3),),
    )
    records = {record.vertices: record for record in result.records}

    assert len(records) == 15
    assert records[(0,)].alpha_squared == 0
    assert records[(0, 1)].alpha_squared == 1
    assert records[(1, 2)].alpha_squared == 2
    assert records[(0, 1, 2)].alpha_squared == 2
    assert records[(1, 2, 3)].alpha_squared == 3
    assert not records[(1, 2, 3)].is_gabriel
    assert records[(0, 1, 2, 3)].alpha_squared == 3
    assert len(result.sha256) == 64


def test_exact_filtration_does_not_call_floating_circumsphere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_floating_sphere(*args, **kwargs):
        raise AssertionError(f"floating sphere must not be called: {args} {kwargs}")

    monkeypatch.setattr(
        "pftf_alpha.filtration.intrinsic_circumsphere",
        forbidden_floating_sphere,
    )

    result = exact_simplex_filtration(
        _tetrahedron_points(),
        ((0, 1, 2, 3),),
    )

    assert len(result.records) == 15


def test_exact_filtration_rejects_nonempty_top_circumsphere() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.25, 0.25, 0.25],
        ]
    )

    with pytest.raises(ArithmeticError, match="exact empty sphere"):
        exact_simplex_filtration(
            points,
            ((0, 1, 2, 3), (0, 1, 2, 4)),
        )


def test_exact_intrinsic_fraction_is_preserved_before_rounding() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )

    result = exact_simplex_filtration(points, ((0, 1, 2, 3),))
    records = {record.vertices: record for record in result.records}

    assert records[(0, 1, 2, 3)].alpha_squared == Fraction(3, 2)


def test_case_audit_records_exact_float_comparison() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=16,
        reference_count=24,
        seed=20260724,
    )[:1]
    construction = _exact_construction(cases)
    assert construction.cases[0].accepted

    audit = audit_exact_filtration_case(
        cases[0].family.value,
        cases[0].points,
        construction.cases[0],
    )

    assert audit.audited
    assert audit.simplex_count > audit.top_simplex_count
    assert (
        audit.float_value_exact_match_count + audit.float_value_difference_count
        == audit.simplex_count
    )
    assert audit.float_gabriel_disagreement_count == 0
    assert audit.adjacent_exact_order_violation_count == 0
    assert audit.exact_filtration_sha256 is not None


def test_panel_audit_is_nonselecting_and_fail_closed_without_backend() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=16,
        reference_count=24,
        seed=20260725,
    )[:1]
    construction = evaluate_exact_construction_panel(
        ((case.family.value, case.points) for case in cases),
        evaluation_split=PanelSplit.HELD_OUT.value,
        backend_command=None,
    )

    audit = evaluate_exact_filtration_panel(
        cases,
        construction_result=construction,
    )
    payload = audit.to_dict()

    assert payload["audited_case_count"] == 0
    assert not payload["exact_filtration_values_applied_to_primary"]
    assert not payload["primary_benchmark_results_changed"]
    assert payload["selection_effect"] == "none"
    assert not payload["promotion_supported"]
    assert payload["blocking_reasons"] == [
        "no_exact_construction_backend",
        "exact_filtration_values_audit_only_not_deployed",
    ]


def test_panel_audit_covers_every_accepted_backend_case() -> None:
    cases = make_minimal_panel(
        split=PanelSplit.HELD_OUT,
        point_count=16,
        reference_count=24,
        seed=20260726,
    )[:2]
    construction = _exact_construction(cases)
    assert construction.accepted_case_count == 2

    audit = evaluate_exact_filtration_panel(
        cases,
        construction_result=construction,
    )
    payload = audit.to_dict()

    assert payload["audited_case_count"] == 2
    assert payload["all_accepted_cases_audited"]
    assert payload["gabriel_disagreement_case_count"] == 0
    assert payload["order_violation_case_count"] == 0
    assert payload["blocking_reasons"] == [
        "exact_filtration_values_audit_only_not_deployed"
    ]


def test_exact_rounded_filtration_uses_exact_records_without_floating_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_floating_sphere(*args, **kwargs):
        raise AssertionError(f"floating sphere must not be called: {args} {kwargs}")

    monkeypatch.setattr(
        "pftf_alpha.filtration.intrinsic_circumsphere",
        forbidden_floating_sphere,
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    exact = exact_simplex_filtration(points, ((0, 1, 2, 3),))

    rounded = exact_rounded_filtration(points, ((0, 1, 2, 3),))

    assert rounded.exact_filtration_sha256 == exact.sha256
    assert rounded.simplex_count == len(exact.records)
    assert [
        (record.vertices, record.alpha_squared, record.is_gabriel)
        for record in rounded.filtration.records
    ] == [
        (record.vertices, float(record.alpha_squared), record.is_gabriel)
        for record in exact.records
    ]
