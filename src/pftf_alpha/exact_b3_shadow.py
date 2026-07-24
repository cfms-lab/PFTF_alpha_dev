"""Evaluation-only B3 selection shadow with exact resampled filtrations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .baselines import (
    BaselineID,
    BaselineResult,
    BenchmarkConfig,
    _b3_with_resampled_filtrations,
    _B3SelectionTrace,
    _resampled_filtrations,
)
from .exact_backend import ExactConstructionPanelResult
from .exact_filtration import exact_rounded_filtration
from .exact_index_audit import _boundary_sha256, _complex_sha256, _result
from .exact_resampling_filtration import ExactResamplingFiltrationPanelAudit
from .exact_shadow import (
    FLOAT_ABS_TOLERANCE,
    FLOAT_REL_TOLERANCE,
    _equivalent,
)
from .exact_value_shadow import ExactValueShadowPanelResult
from .synthetic import SyntheticCase


@dataclass(frozen=True)
class ExactB3CandidateShadow:
    """One budgeted exact-full B3 candidate under two resampling sources."""

    local_position: int
    critical_index: int
    alpha_squared: float
    reference_stability: float
    exact_resampling_stability: float
    stability_absolute_difference: float
    stability_matches: bool
    reference_objective_total: float
    exact_resampling_objective_total: float
    objective_absolute_difference: float
    objective_matches: bool
    nonstability_terms_match: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExactB3SelectionShadowCase:
    """Case-level B3 reselection on exact resampled filtrations."""

    case_id: str
    status: str
    schema24_case_audited: bool
    exact_value_shadow_available: bool
    reference_report_reproduced: bool | None
    candidate_count: int
    candidates: tuple[ExactB3CandidateShadow, ...]
    reference_selected_local_position: int | None
    exact_selected_local_position: int | None
    reference_selected_critical_index: int | None
    exact_selected_critical_index: int | None
    selected_index_matches: bool | None
    reference_alpha_squared: float | None
    exact_alpha_squared: float | None
    alpha_absolute_difference: float | None
    selected_alpha_matches: bool | None
    reference_selected_complex_sha256: str | None
    exact_selected_complex_sha256: str | None
    selected_complex_matches: bool | None
    reference_selected_boundary_sha256: str | None
    exact_selected_boundary_sha256: str | None
    selected_boundary_matches: bool | None
    selected_objective_matches: bool | None
    selected_endpoints_match: bool | None
    exact_selection_shadow_result: BaselineResult | None
    rejection_reasons: tuple[str, ...]

    @property
    def shadow_ran(self) -> bool:
        return self.status == "audited"

    @property
    def candidate_stability_difference_count(self) -> int:
        return sum(not candidate.stability_matches for candidate in self.candidates)

    @property
    def candidate_objective_difference_count(self) -> int:
        return sum(not candidate.objective_matches for candidate in self.candidates)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "schema24_case_audited": self.schema24_case_audited,
            "exact_value_shadow_available": self.exact_value_shadow_available,
            "reference_report_reproduced": self.reference_report_reproduced,
            "candidate_count": self.candidate_count,
            "candidate_stability_difference_count": (
                self.candidate_stability_difference_count
            ),
            "candidate_objective_difference_count": (
                self.candidate_objective_difference_count
            ),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reference_selected_local_position": (
                self.reference_selected_local_position
            ),
            "exact_selected_local_position": self.exact_selected_local_position,
            "reference_selected_critical_index": (
                self.reference_selected_critical_index
            ),
            "exact_selected_critical_index": self.exact_selected_critical_index,
            "selected_index_matches": self.selected_index_matches,
            "reference_alpha_squared": self.reference_alpha_squared,
            "exact_alpha_squared": self.exact_alpha_squared,
            "alpha_absolute_difference": self.alpha_absolute_difference,
            "selected_alpha_matches": self.selected_alpha_matches,
            "reference_selected_complex_sha256": (
                self.reference_selected_complex_sha256
            ),
            "exact_selected_complex_sha256": self.exact_selected_complex_sha256,
            "selected_complex_matches": self.selected_complex_matches,
            "reference_selected_boundary_sha256": (
                self.reference_selected_boundary_sha256
            ),
            "exact_selected_boundary_sha256": self.exact_selected_boundary_sha256,
            "selected_boundary_matches": self.selected_boundary_matches,
            "selected_objective_matches": self.selected_objective_matches,
            "selected_endpoints_match": self.selected_endpoints_match,
            "exact_selection_shadow_result": (
                None
                if self.exact_selection_shadow_result is None
                else self.exact_selection_shadow_result.to_dict()
            ),
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class ExactB3SelectionShadowPanel:
    """Panel summary for exact-resampling B3 candidate selection."""

    evaluation_split: str
    backend_requested: bool
    requested_case_count: int
    schema24_audited_case_count: int
    cases: tuple[ExactB3SelectionShadowCase, ...]

    @property
    def shadow_case_count(self) -> int:
        return sum(case.shadow_ran for case in self.cases)

    @property
    def candidate_stability_difference_case_count(self) -> int:
        return sum(
            case.shadow_ran and case.candidate_stability_difference_count > 0
            for case in self.cases
        )

    @property
    def candidate_stability_difference_count(self) -> int:
        return sum(case.candidate_stability_difference_count for case in self.cases)

    @property
    def selected_index_difference_case_count(self) -> int:
        return sum(
            case.shadow_ran and case.selected_index_matches is False
            for case in self.cases
        )

    @property
    def selected_complex_difference_case_count(self) -> int:
        return sum(
            case.shadow_ran and case.selected_complex_matches is False
            for case in self.cases
        )

    @property
    def selected_boundary_difference_case_count(self) -> int:
        return sum(
            case.shadow_ran and case.selected_boundary_matches is False
            for case in self.cases
        )

    @property
    def selected_objective_difference_case_count(self) -> int:
        return sum(
            case.shadow_ran and case.selected_objective_matches is False
            for case in self.cases
        )

    @property
    def selected_endpoint_difference_case_count(self) -> int:
        return sum(
            case.shadow_ran and case.selected_endpoints_match is False
            for case in self.cases
        )

    @property
    def all_prerequisite_cases_evaluated(self) -> bool:
        return (
            self.requested_case_count > 0
            and self.schema24_audited_case_count == self.requested_case_count
            and self.shadow_case_count == self.requested_case_count
        )

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.backend_requested:
            reasons.append("no_exact_resampling_backend")
        if self.schema24_audited_case_count != self.requested_case_count:
            reasons.append("one_or_more_schema24_exact_resampling_cases_missing")
        if self.shadow_case_count != self.schema24_audited_case_count:
            reasons.append("one_or_more_exact_b3_selection_shadows_not_evaluated")
        reasons.append("exact_b3_selection_shadow_not_deployed")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "evaluation_only_exact_resampling_B3_selection_shadow",
            "evaluation_split": self.evaluation_split,
            "full_filtration": "correctly_rounded_exact_rationals",
            "resampled_filtrations": "host_validated_exact_connectivity_and_values",
            "candidate_source": "budgeted_exact_full_B3_critical_values",
            "reference": "exact_full_with_floating_resampled_filtrations",
            "comparison_float_relative_tolerance": FLOAT_REL_TOLERANCE,
            "comparison_float_absolute_tolerance": FLOAT_ABS_TOLERANCE,
            "backend_requested": self.backend_requested,
            "requested_case_count": self.requested_case_count,
            "schema24_audited_case_count": self.schema24_audited_case_count,
            "shadow_case_count": self.shadow_case_count,
            "candidate_stability_difference_case_count": (
                self.candidate_stability_difference_case_count
            ),
            "candidate_stability_difference_count": (
                self.candidate_stability_difference_count
            ),
            "selected_index_difference_case_count": (
                self.selected_index_difference_case_count
            ),
            "selected_complex_difference_case_count": (
                self.selected_complex_difference_case_count
            ),
            "selected_boundary_difference_case_count": (
                self.selected_boundary_difference_case_count
            ),
            "selected_objective_difference_case_count": (
                self.selected_objective_difference_case_count
            ),
            "selected_endpoint_difference_case_count": (
                self.selected_endpoint_difference_case_count
            ),
            "all_prerequisite_cases_evaluated": (
                self.all_prerequisite_cases_evaluated
            ),
            "primary_benchmark_results_changed": False,
            "selection_effect": "none",
            "promotion_supported": False,
            "blocking_reasons": list(self.blocking_reasons),
            "cases": [case.to_dict() for case in self.cases],
        }


def _matches(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=FLOAT_REL_TOLERANCE,
        abs_tol=FLOAT_ABS_TOLERANCE,
    )


def _nonruntime_payload(result: BaselineResult) -> dict[str, object]:
    payload = result.to_dict()
    payload.pop("runtime_seconds")
    return payload


def _selected_local_position(trace: _B3SelectionTrace) -> int:
    return next(
        index
        for index, evaluation in enumerate(trace.evaluations)
        if evaluation.alpha_squared == trace.result.alpha_squared
    )


def _not_evaluated_case(
    case: SyntheticCase,
    *,
    schema24_audited: bool,
    exact_value_shadow_available: bool,
    reason: str,
) -> ExactB3SelectionShadowCase:
    return ExactB3SelectionShadowCase(
        case_id=case.family.value,
        status="not_evaluated",
        schema24_case_audited=schema24_audited,
        exact_value_shadow_available=exact_value_shadow_available,
        reference_report_reproduced=None,
        candidate_count=0,
        candidates=(),
        reference_selected_local_position=None,
        exact_selected_local_position=None,
        reference_selected_critical_index=None,
        exact_selected_critical_index=None,
        selected_index_matches=None,
        reference_alpha_squared=None,
        exact_alpha_squared=None,
        alpha_absolute_difference=None,
        selected_alpha_matches=None,
        reference_selected_complex_sha256=None,
        exact_selected_complex_sha256=None,
        selected_complex_matches=None,
        reference_selected_boundary_sha256=None,
        exact_selected_boundary_sha256=None,
        selected_boundary_matches=None,
        selected_objective_matches=None,
        selected_endpoints_match=None,
        exact_selection_shadow_result=None,
        rejection_reasons=(reason,),
    )


def evaluate_exact_b3_selection_shadow(
    cases: Iterable[SyntheticCase],
    *,
    construction_result: ExactConstructionPanelResult,
    exact_value_shadow: ExactValueShadowPanelResult,
    exact_resampling_audit: ExactResamplingFiltrationPanelAudit,
    config: BenchmarkConfig,
) -> ExactB3SelectionShadowPanel:
    """Re-evaluate every budgeted B3 candidate with exact resamples."""

    materialized_cases = tuple(cases)
    if not materialized_cases:
        raise ValueError("cases must be non-empty")
    if not (
        construction_result.evaluation_split
        == exact_value_shadow.evaluation_split
        == exact_resampling_audit.evaluation_split
    ):
        raise ValueError("exact B3 shadow prerequisites must use the same split")

    construction_by_id = {
        construction.case_id: construction
        for construction in construction_result.cases
    }
    value_by_id = {case.case_id: case for case in exact_value_shadow.cases}
    resampling_by_id = {case.case_id: case for case in exact_resampling_audit.cases}
    for name, materialized, indexed in (
        ("construction result", construction_result.cases, construction_by_id),
        ("exact value shadow", exact_value_shadow.cases, value_by_id),
        ("exact resampling audit", exact_resampling_audit.cases, resampling_by_id),
    ):
        if len(materialized) != len(indexed):
            raise ValueError(f"{name} contains duplicate case identifiers")

    case_shadows = []
    for case in materialized_cases:
        case_id = case.family.value
        construction = construction_by_id.get(case_id)
        value_case = value_by_id.get(case_id)
        resampling_case = resampling_by_id.get(case_id)
        exact_resamples = exact_resampling_audit.exact_resampled_filtrations(case_id)
        schema24_audited = bool(resampling_case and resampling_case.audited)
        value_available = bool(
            value_case
            and value_case.shadow_ran
            and value_case.shadow_report is not None
        )
        if (
            construction is None
            or not construction.accepted
            or construction.validated_top_simplices is None
        ):
            case_shadows.append(
                _not_evaluated_case(
                    case,
                    schema24_audited=schema24_audited,
                    exact_value_shadow_available=value_available,
                    reason="full_exact_construction_missing",
                )
            )
            continue
        if not schema24_audited or exact_resamples is None:
            case_shadows.append(
                _not_evaluated_case(
                    case,
                    schema24_audited=schema24_audited,
                    exact_value_shadow_available=value_available,
                    reason="schema24_exact_resampling_context_missing",
                )
            )
            continue
        if (
            not value_available
            or value_case is None
            or value_case.shadow_report is None
        ):
            case_shadows.append(
                _not_evaluated_case(
                    case,
                    schema24_audited=True,
                    exact_value_shadow_available=False,
                    reason="exact_value_B3_reference_missing",
                )
            )
            continue

        try:
            reference_result = _result(
                value_case.shadow_report,
                BaselineID.B3_PERSISTENCE_STABILITY,
            )
            exact_full = exact_rounded_filtration(
                case.points,
                construction.validated_top_simplices,
            ).filtration
            reference_trace = _b3_with_resampled_filtrations(
                exact_full,
                case,
                config,
                _resampled_filtrations(case, config),
            )
            exact_trace = _b3_with_resampled_filtrations(
                exact_full,
                case,
                config,
                exact_resamples,
            )
            if reference_trace.candidate_indices != exact_trace.candidate_indices:
                raise ValueError("B3 candidate index sequence differs")
            if len(reference_trace.evaluations) != len(exact_trace.evaluations):
                raise ValueError("B3 candidate count differs")
            reference_reproduced = _equivalent(
                _nonruntime_payload(reference_trace.result),
                _nonruntime_payload(reference_result),
            )
            if not reference_reproduced:
                raise ValueError("exact-value B3 reference was not reproduced")

            candidate_shadows = []
            for local_position, (
                critical_index,
                reference_evaluation,
                exact_evaluation,
            ) in enumerate(
                zip(
                    reference_trace.candidate_indices,
                    reference_trace.evaluations,
                    exact_trace.evaluations,
                    strict=True,
                )
            ):
                if reference_evaluation.alpha_squared != exact_evaluation.alpha_squared:
                    raise ValueError("B3 candidate alpha differs")
                reference_terms = reference_evaluation.terms
                exact_terms = exact_evaluation.terms
                nonstability_terms_match = (
                    _matches(reference_terms.geometry, exact_terms.geometry)
                    and _matches(reference_terms.topology, exact_terms.topology)
                    and _matches(reference_terms.complexity, exact_terms.complexity)
                    and reference_evaluation.statistics == exact_evaluation.statistics
                )
                if not nonstability_terms_match:
                    raise ValueError("non-stability B3 candidate terms changed")
                candidate_shadows.append(
                    ExactB3CandidateShadow(
                        local_position=local_position,
                        critical_index=critical_index,
                        alpha_squared=reference_evaluation.alpha_squared,
                        reference_stability=reference_terms.stability,
                        exact_resampling_stability=exact_terms.stability,
                        stability_absolute_difference=abs(
                            reference_terms.stability - exact_terms.stability
                        ),
                        stability_matches=_matches(
                            reference_terms.stability,
                            exact_terms.stability,
                        ),
                        reference_objective_total=reference_evaluation.total,
                        exact_resampling_objective_total=exact_evaluation.total,
                        objective_absolute_difference=abs(
                            reference_evaluation.total - exact_evaluation.total
                        ),
                        objective_matches=_matches(
                            reference_evaluation.total,
                            exact_evaluation.total,
                        ),
                        nonstability_terms_match=True,
                    )
                )

            reference_local = _selected_local_position(reference_trace)
            exact_local = _selected_local_position(exact_trace)
            reference_critical = reference_trace.candidate_indices[reference_local]
            exact_critical = exact_trace.candidate_indices[exact_local]
            reference_alpha = reference_trace.result.alpha_squared
            exact_alpha = exact_trace.result.alpha_squared
            if reference_alpha is None or exact_alpha is None:
                raise ValueError("B3 selected alpha missing")
            reference_complex = _complex_sha256(exact_full, reference_alpha)
            exact_complex = _complex_sha256(exact_full, exact_alpha)
            reference_boundary = _boundary_sha256(exact_full, reference_alpha)
            exact_boundary = _boundary_sha256(exact_full, exact_alpha)
            reference_objective = reference_trace.result.objective_terms
            exact_objective = exact_trace.result.objective_terms
            if reference_objective is None or exact_objective is None:
                raise ValueError("B3 selected objective missing")
        except (ArithmeticError, StopIteration, ValueError):
            case_shadows.append(
                _not_evaluated_case(
                    case,
                    schema24_audited=True,
                    exact_value_shadow_available=True,
                    reason="exact_B3_selection_shadow_failed",
                )
            )
            continue

        case_shadows.append(
            ExactB3SelectionShadowCase(
                case_id=case_id,
                status="audited",
                schema24_case_audited=True,
                exact_value_shadow_available=True,
                reference_report_reproduced=True,
                candidate_count=len(candidate_shadows),
                candidates=tuple(candidate_shadows),
                reference_selected_local_position=reference_local,
                exact_selected_local_position=exact_local,
                reference_selected_critical_index=reference_critical,
                exact_selected_critical_index=exact_critical,
                selected_index_matches=reference_critical == exact_critical,
                reference_alpha_squared=reference_alpha,
                exact_alpha_squared=exact_alpha,
                alpha_absolute_difference=abs(reference_alpha - exact_alpha),
                selected_alpha_matches=_matches(reference_alpha, exact_alpha),
                reference_selected_complex_sha256=reference_complex,
                exact_selected_complex_sha256=exact_complex,
                selected_complex_matches=reference_complex == exact_complex,
                reference_selected_boundary_sha256=reference_boundary,
                exact_selected_boundary_sha256=exact_boundary,
                selected_boundary_matches=reference_boundary == exact_boundary,
                selected_objective_matches=_equivalent(
                    asdict(reference_objective),
                    asdict(exact_objective),
                ),
                selected_endpoints_match=_equivalent(
                    reference_trace.result.endpoints.to_dict(),
                    exact_trace.result.endpoints.to_dict(),
                ),
                exact_selection_shadow_result=exact_trace.result,
                rejection_reasons=(),
            )
        )

    return ExactB3SelectionShadowPanel(
        evaluation_split=construction_result.evaluation_split,
        backend_requested=exact_resampling_audit.backend_requested,
        requested_case_count=len(materialized_cases),
        schema24_audited_case_count=exact_resampling_audit.audited_case_count,
        cases=tuple(case_shadows),
    )
