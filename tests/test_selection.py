import numpy as np
import pytest

from pftf_alpha import (
    AlphaFiltration,
    ObjectiveTerms,
    ObjectiveWeights,
    scan_critical_alphas,
    select_best_alpha,
)


def test_critical_scan_selects_declared_objective_minimum() -> None:
    filtration = AlphaFiltration.from_points(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )

    def evaluator(_filtration, alpha_squared, statistics):
        return ObjectiveTerms(
            geometry=abs(alpha_squared - 0.5),
            topology=abs(statistics.connected_components - 1),
            complexity=0.01 * statistics.top_simplices,
        )

    evaluations = scan_critical_alphas(
        filtration,
        evaluator,
        weights=ObjectiveWeights(
            geometry=1.0,
            topology=1.0,
            stability=0.0,
            complexity=1.0,
        ),
    )
    selected = select_best_alpha(evaluations)

    assert [row.alpha_squared for row in evaluations] == [0.25, 0.5]
    assert selected.alpha_squared == pytest.approx(0.5)
    assert selected.statistics.top_simplices == 1


def test_selection_rejects_negative_objective_terms() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ObjectiveTerms(geometry=-1.0)
