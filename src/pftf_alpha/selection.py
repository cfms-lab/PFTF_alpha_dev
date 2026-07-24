"""Critical-alpha scanning and auditable multi-objective selection."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .filtration import AlphaFiltration, ComplexStatistics


@dataclass(frozen=True)
class ObjectiveTerms:
    """Unweighted, non-negative loss terms for one alpha candidate."""

    geometry: float = 0.0
    topology: float = 0.0
    stability: float = 0.0
    complexity: float = 0.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"objective term {name} must be finite and non-negative"
                )


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights in the documented geometry-topology-stability objective."""

    geometry: float = 1.0
    topology: float = 1.0
    stability: float = 1.0
    complexity: float = 1.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"objective weight {name} must be finite and non-negative"
                )

    def apply(self, terms: ObjectiveTerms) -> float:
        return (
            self.geometry * terms.geometry
            + self.topology * terms.topology
            + self.stability * terms.stability
            + self.complexity * terms.complexity
        )


@dataclass(frozen=True)
class AlphaEvaluation:
    """One row in a critical-alpha scan."""

    alpha_squared: float
    terms: ObjectiveTerms
    total: float
    statistics: ComplexStatistics


AlphaEvaluator = Callable[
    [AlphaFiltration, float, ComplexStatistics],
    ObjectiveTerms | Mapping[str, float],
]


def _as_terms(result: ObjectiveTerms | Mapping[str, float]) -> ObjectiveTerms:
    if isinstance(result, ObjectiveTerms):
        return result
    return ObjectiveTerms(**dict(result))


def scan_critical_alphas(
    filtration: AlphaFiltration,
    evaluator: AlphaEvaluator,
    *,
    weights: ObjectiveWeights | None = None,
    candidates: Iterable[float] | None = None,
    include_zero: bool = False,
) -> tuple[AlphaEvaluation, ...]:
    """Evaluate every finite critical value using a caller-declared objective.

    The evaluator receives the filtration, the candidate in squared-radius
    convention, and cheap complex statistics.  Geometry, topology, and
    stability remain dataset-specific and are never guessed by this module.
    """

    if candidates is None:
        candidate_values = filtration.critical_values(
            include_zero=include_zero
        ).tolist()
    else:
        candidate_values = sorted({float(value) for value in candidates})
        if not include_zero:
            candidate_values = [value for value in candidate_values if value > 0.0]
    if not candidate_values:
        raise ValueError("no alpha candidates are available")
    if any(not math.isfinite(value) or value < 0.0 for value in candidate_values):
        raise ValueError("alpha candidates must be finite squared radii")
    selected_weights = ObjectiveWeights() if weights is None else weights

    evaluations: list[AlphaEvaluation] = []
    for alpha_squared in candidate_values:
        statistics = filtration.statistics(alpha_squared)
        terms = _as_terms(evaluator(filtration, alpha_squared, statistics))
        evaluations.append(
            AlphaEvaluation(
                alpha_squared=alpha_squared,
                terms=terms,
                total=selected_weights.apply(terms),
                statistics=statistics,
            )
        )
    return tuple(evaluations)


def select_best_alpha(
    evaluations: Iterable[AlphaEvaluation],
) -> AlphaEvaluation:
    """Select deterministically: total loss, complexity, then smaller alpha."""

    rows = tuple(evaluations)
    if not rows:
        raise ValueError("at least one alpha evaluation is required")
    return min(
        rows,
        key=lambda row: (
            row.total,
            row.terms.complexity,
            row.alpha_squared,
        ),
    )
