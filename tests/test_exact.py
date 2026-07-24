import numpy as np

from pftf_alpha.exact import (
    audit_delaunay_predicates,
    audit_exact_predicate_panel,
)


def test_exact_predicates_accept_a_locally_delaunay_bipyramid() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])

    audit = audit_delaunay_predicates("bipyramid", points, cells)

    assert audit.interior_facet_count == 1
    assert audit.audited_interior_facet_count == 1
    assert audit.exact_orientation_zero_count == 0
    assert audit.exact_cospherical_interior_facet_count == 0
    assert audit.exact_local_delaunay_violation_count == 0
    assert audit.predicate_consistent
    assert audit.unique_delaunay_combinatorics_supported


def test_exact_predicates_detect_a_local_delaunay_violation() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.125, 0.125, -0.125],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])

    audit = audit_delaunay_predicates("local_violation", points, cells)

    assert audit.interior_facet_side_violation_count == 0
    assert audit.exact_local_delaunay_violation_count == 1
    assert not audit.predicate_consistent
    assert not audit.unique_delaunay_combinatorics_supported


def test_exact_predicates_report_cospherical_ambiguity() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])

    audit = audit_delaunay_predicates("cospherical", points, cells)

    assert audit.exact_cospherical_interior_facet_count == 1
    assert audit.exact_local_delaunay_violation_count == 0
    assert audit.predicate_consistent
    assert not audit.unique_delaunay_combinatorics_supported


def test_panel_audit_blocks_promotion_without_exact_construction_backend() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )

    result = audit_exact_predicate_panel(
        [("bipyramid", points)],
        evaluation_split="held_out",
    )
    payload = result.to_dict()

    assert payload["role"] == "readiness_audit_no_selection"
    assert payload["coordinate_model"] == "binary64_values_as_exact_rationals"
    assert not payload["exact_construction_backend_integrated"]
    assert not payload["changes_benchmark_selection"]
    assert not payload["promotion_supported"]
    assert payload["blocking_reasons"] == ["no_exact_construction_backend"]
    assert payload["totals"]["case_count"] == 1

    integrated_payload = result.to_dict(exact_construction_backend_integrated=True)
    assert integrated_payload["exact_construction_backend_integrated"]
    assert integrated_payload["blocking_reasons"] == []
    assert not integrated_payload["promotion_supported"]
