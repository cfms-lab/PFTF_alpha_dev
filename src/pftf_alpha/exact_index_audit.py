"""Exact critical-index and selected-complex identity audit for B2/B3."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np

from .baselines import (
    BaselineID,
    BaselineResult,
    BenchmarkConfig,
    CaseBenchmark,
    _b3_critical_trace,
)
from .exact_backend import ExactConstructionCaseResult, ExactConstructionPanelResult
from .exact_filtration import (
    ExactFiltrationCaseAudit,
    ExactFiltrationPanelAudit,
    ExactSimplexFiltrationRecord,
    exact_rounded_filtration,
)
from .exact_shadow import (
    FLOAT_ABS_TOLERANCE,
    FLOAT_REL_TOLERANCE,
    ExactConnectivityShadowCaseResult,
    ExactConnectivityShadowPanelResult,
    _equivalent,
)
from .exact_value_shadow import (
    ExactValueShadowCaseResult,
    ExactValueShadowPanelResult,
)
from .filtration import AlphaFiltration, Simplex
from .synthetic import SyntheticCase


@dataclass(frozen=True)
class ExactCriticalMethodIdentity:
    """Selected critical rank and complex identity for one B2/B3 method."""

    method: str
    floating_selected_index: int
    exact_selected_index: int
    selected_index_matches: bool
    floating_alpha_squared: float
    exact_alpha_squared: float
    alpha_absolute_difference: float
    alpha_relative_difference: float
    floating_selected_complex_sha256: str
    exact_selected_complex_sha256: str
    selected_complex_matches: bool
    floating_selected_boundary_sha256: str
    exact_selected_boundary_sha256: str
    selected_boundary_matches: bool
    objective_matches: bool
    endpoints_match: bool

    @property
    def objective_changed_with_same_complex(self) -> bool:
        return self.selected_complex_matches and not self.objective_matches

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["objective_changed_with_same_complex"] = (
            self.objective_changed_with_same_complex
        )
        return payload


@dataclass(frozen=True)
class ExactCriticalIndexCaseAudit:
    """One case-level rank, birth-group, and selected-complex audit."""

    case_id: str
    status: str
    backend_accepted: bool
    backend_name: str | None
    backend_version: str | None
    backend_kernel: str | None
    point_count: int
    top_simplex_count: int
    floating_critical_count: int
    exact_rounded_critical_count: int
    exact_rational_critical_count: int
    critical_counts_match: bool | None
    rounded_exact_critical_count_matches_rational: bool | None
    floating_birth_group_sequence_sha256: str | None
    exact_birth_group_sequence_sha256: str | None
    critical_birth_group_sequence_matches: bool | None
    method_identities: tuple[ExactCriticalMethodIdentity, ...]
    b3_signature_sequence_matches: bool | None
    floating_b3_candidate_indices: tuple[int, ...]
    exact_b3_candidate_indices: tuple[int, ...]
    b3_candidate_index_sequence_matches: bool | None
    b3_persistence_value_difference_count: int
    b3_max_absolute_persistence_difference: float | None
    floating_b3_selected_persistence: float | None
    exact_b3_selected_persistence: float | None
    b3_selected_persistence_matches: bool | None
    b3_topology_term_matches: bool | None
    b3_stability_term_matches: bool | None
    rejection_reasons: tuple[str, ...]

    @property
    def audited(self) -> bool:
        return self.status == "audited"

    @property
    def b3_objective_changed_with_same_complex(self) -> bool:
        return any(
            identity.method == BaselineID.B3_PERSISTENCE_STABILITY.value
            and identity.objective_changed_with_same_complex
            for identity in self.method_identities
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["method_identities"] = [
            identity.to_dict() for identity in self.method_identities
        ]
        payload["floating_b3_candidate_indices"] = list(
            self.floating_b3_candidate_indices
        )
        payload["exact_b3_candidate_indices"] = list(self.exact_b3_candidate_indices)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["b3_objective_changed_with_same_complex"] = (
            self.b3_objective_changed_with_same_complex
        )
        return payload


@dataclass(frozen=True)
class ExactCriticalIndexPanelAudit:
    """Panel audit that cannot alter B2/B3 selection."""

    evaluation_split: str
    backend_requested: bool
    requested_case_count: int
    accepted_backend_case_count: int
    requested_methods: tuple[str, ...]
    cases: tuple[ExactCriticalIndexCaseAudit, ...]

    @property
    def audited_case_count(self) -> int:
        return sum(case.audited for case in self.cases)

    @property
    def critical_count_mismatch_case_count(self) -> int:
        return sum(
            case.audited and not bool(case.critical_counts_match) for case in self.cases
        )

    @property
    def birth_group_mismatch_case_count(self) -> int:
        return sum(
            case.audited and not bool(case.critical_birth_group_sequence_matches)
            for case in self.cases
        )

    @property
    def selected_index_mismatch_method_count(self) -> int:
        return sum(
            not identity.selected_index_matches
            for case in self.cases
            if case.audited
            for identity in case.method_identities
        )

    @property
    def selected_complex_mismatch_method_count(self) -> int:
        return sum(
            not identity.selected_complex_matches
            for case in self.cases
            if case.audited
            for identity in case.method_identities
        )

    @property
    def selected_boundary_mismatch_method_count(self) -> int:
        return sum(
            not identity.selected_boundary_matches
            for case in self.cases
            if case.audited
            for identity in case.method_identities
        )

    @property
    def b3_signature_mismatch_case_count(self) -> int:
        return sum(
            case.audited and case.b3_signature_sequence_matches is False
            for case in self.cases
        )

    @property
    def b3_candidate_index_mismatch_case_count(self) -> int:
        return sum(
            case.audited and case.b3_candidate_index_sequence_matches is False
            for case in self.cases
        )

    @property
    def b3_persistence_difference_case_count(self) -> int:
        return sum(
            case.audited and case.b3_persistence_value_difference_count > 0
            for case in self.cases
        )

    @property
    def b3_selected_persistence_difference_case_count(self) -> int:
        return sum(
            case.audited and case.b3_selected_persistence_matches is False
            for case in self.cases
        )

    @property
    def b3_objective_difference_with_same_complex_case_count(self) -> int:
        return sum(
            case.audited and case.b3_objective_changed_with_same_complex
            for case in self.cases
        )

    @property
    def b3_topology_term_difference_case_count(self) -> int:
        return sum(
            case.audited and case.b3_topology_term_matches is False
            for case in self.cases
        )

    @property
    def b3_stability_term_difference_case_count(self) -> int:
        return sum(
            case.audited and case.b3_stability_term_matches is False
            for case in self.cases
        )

    @property
    def all_selection_identities_match(self) -> bool:
        return (
            self.audited_case_count > 0
            and self.audited_case_count == self.accepted_backend_case_count
            and self.critical_count_mismatch_case_count == 0
            and self.birth_group_mismatch_case_count == 0
            and self.selected_index_mismatch_method_count == 0
            and self.selected_complex_mismatch_method_count == 0
            and self.selected_boundary_mismatch_method_count == 0
            and self.b3_signature_mismatch_case_count == 0
            and self.b3_candidate_index_mismatch_case_count == 0
        )

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.backend_requested:
            reasons.append("no_exact_construction_backend")
        elif self.accepted_backend_case_count != self.requested_case_count:
            reasons.append("one_or_more_backend_results_rejected")
        if self.audited_case_count != self.accepted_backend_case_count:
            reasons.append("one_or_more_critical_index_audits_not_completed")
        reasons.append("exact_critical_index_audit_not_deployed")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "evaluation_only_exact_critical_index_identity_audit",
            "evaluation_split": self.evaluation_split,
            "requested_methods": list(self.requested_methods),
            "comparison_float_relative_tolerance": FLOAT_REL_TOLERANCE,
            "comparison_float_absolute_tolerance": FLOAT_ABS_TOLERANCE,
            "backend_requested": self.backend_requested,
            "requested_case_count": self.requested_case_count,
            "accepted_backend_case_count": self.accepted_backend_case_count,
            "audited_case_count": self.audited_case_count,
            "critical_count_mismatch_case_count": (
                self.critical_count_mismatch_case_count
            ),
            "birth_group_mismatch_case_count": self.birth_group_mismatch_case_count,
            "selected_index_mismatch_method_count": (
                self.selected_index_mismatch_method_count
            ),
            "selected_complex_mismatch_method_count": (
                self.selected_complex_mismatch_method_count
            ),
            "selected_boundary_mismatch_method_count": (
                self.selected_boundary_mismatch_method_count
            ),
            "b3_signature_mismatch_case_count": self.b3_signature_mismatch_case_count,
            "b3_candidate_index_mismatch_case_count": (
                self.b3_candidate_index_mismatch_case_count
            ),
            "b3_persistence_difference_case_count": (
                self.b3_persistence_difference_case_count
            ),
            "b3_selected_persistence_difference_case_count": (
                self.b3_selected_persistence_difference_case_count
            ),
            "b3_objective_difference_with_same_complex_case_count": (
                self.b3_objective_difference_with_same_complex_case_count
            ),
            "b3_topology_term_difference_case_count": (
                self.b3_topology_term_difference_case_count
            ),
            "b3_stability_term_difference_case_count": (
                self.b3_stability_term_difference_case_count
            ),
            "all_selection_identities_match": self.all_selection_identities_match,
            "primary_benchmark_results_changed": False,
            "selection_effect": "none",
            "promotion_supported": False,
            "blocking_reasons": list(self.blocking_reasons),
            "cases": [case.to_dict() for case in self.cases],
        }


def _sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _floating_birth_groups(
    filtration: AlphaFiltration,
) -> tuple[tuple[Simplex, ...], ...]:
    top_records = tuple(
        record
        for record in filtration.records
        if record.dimension == filtration.ambient_dimension
    )
    values = sorted({record.alpha_squared for record in top_records})
    return tuple(
        tuple(
            sorted(
                record.vertices
                for record in top_records
                if record.alpha_squared == value
            )
        )
        for value in values
    )


def _exact_birth_groups(
    records: tuple[ExactSimplexFiltrationRecord, ...],
) -> tuple[tuple[Simplex, ...], ...]:
    top_records = tuple(record for record in records if len(record.vertices) == 4)
    values: list[Fraction] = sorted({record.alpha_squared for record in top_records})
    return tuple(
        tuple(
            sorted(
                record.vertices
                for record in top_records
                if record.alpha_squared == value
            )
        )
        for value in values
    )


def _group_sha256(groups: tuple[tuple[Simplex, ...], ...]) -> str:
    return _sha256([[list(simplex) for simplex in group] for group in groups])


def _complex_sha256(filtration: AlphaFiltration, alpha_squared: float) -> str:
    simplices = filtration.simplices_at(alpha_squared)
    payload = [
        {
            "dimension": dimension,
            "simplices": [list(simplex) for simplex in values.tolist()],
        }
        for dimension, values in sorted(simplices.items())
    ]
    return _sha256(payload)


def _boundary_sha256(filtration: AlphaFiltration, alpha_squared: float) -> str:
    facets = filtration.boundary_facets_at(alpha_squared)
    canonical = sorted(
        tuple(sorted(int(vertex) for vertex in facet)) for facet in facets
    )
    return _sha256([list(facet) for facet in canonical])


def _result(report: CaseBenchmark, method: BaselineID) -> BaselineResult:
    matches = tuple(result for result in report.results if result.method is method)
    if len(matches) != 1:
        raise ValueError(f"report must contain exactly one {method.value} result")
    return matches[0]


def _critical_index(candidates: np.ndarray, alpha_squared: float) -> int:
    matches = np.flatnonzero(candidates == alpha_squared)
    if matches.size != 1:
        raise ValueError("selected alpha is not a unique critical candidate")
    return int(matches[0])


def _method_identity(
    method: BaselineID,
    floating_filtration: AlphaFiltration,
    exact_filtration: AlphaFiltration,
    floating_report: CaseBenchmark,
    exact_report: CaseBenchmark,
) -> ExactCriticalMethodIdentity:
    floating_result = _result(floating_report, method)
    exact_result = _result(exact_report, method)
    if floating_result.alpha_squared is None or exact_result.alpha_squared is None:
        raise ValueError(f"{method.value} must select a finite critical alpha")
    floating_candidates = floating_filtration.critical_values(
        dimensions=[floating_filtration.ambient_dimension]
    )
    exact_candidates = exact_filtration.critical_values(
        dimensions=[exact_filtration.ambient_dimension]
    )
    floating_index = _critical_index(floating_candidates, floating_result.alpha_squared)
    exact_index = _critical_index(exact_candidates, exact_result.alpha_squared)
    floating_complex_sha256 = _complex_sha256(
        floating_filtration,
        floating_result.alpha_squared,
    )
    exact_complex_sha256 = _complex_sha256(
        exact_filtration,
        exact_result.alpha_squared,
    )
    floating_boundary_sha256 = _boundary_sha256(
        floating_filtration,
        floating_result.alpha_squared,
    )
    exact_boundary_sha256 = _boundary_sha256(
        exact_filtration,
        exact_result.alpha_squared,
    )
    absolute_difference = abs(
        floating_result.alpha_squared - exact_result.alpha_squared
    )
    relative_difference = absolute_difference / max(
        abs(floating_result.alpha_squared),
        abs(exact_result.alpha_squared),
        np.finfo(np.float64).tiny,
    )
    objective_matches = _equivalent(
        (
            floating_result.objective_total,
            None
            if floating_result.objective_terms is None
            else asdict(floating_result.objective_terms),
        ),
        (
            exact_result.objective_total,
            None
            if exact_result.objective_terms is None
            else asdict(exact_result.objective_terms),
        ),
    )
    endpoints_match = _equivalent(
        floating_result.endpoints.to_dict(),
        exact_result.endpoints.to_dict(),
    )
    return ExactCriticalMethodIdentity(
        method=method.value,
        floating_selected_index=floating_index,
        exact_selected_index=exact_index,
        selected_index_matches=floating_index == exact_index,
        floating_alpha_squared=floating_result.alpha_squared,
        exact_alpha_squared=exact_result.alpha_squared,
        alpha_absolute_difference=absolute_difference,
        alpha_relative_difference=relative_difference,
        floating_selected_complex_sha256=floating_complex_sha256,
        exact_selected_complex_sha256=exact_complex_sha256,
        selected_complex_matches=floating_complex_sha256 == exact_complex_sha256,
        floating_selected_boundary_sha256=floating_boundary_sha256,
        exact_selected_boundary_sha256=exact_boundary_sha256,
        selected_boundary_matches=floating_boundary_sha256 == exact_boundary_sha256,
        objective_matches=objective_matches,
        endpoints_match=endpoints_match,
    )


def _not_audited_case(
    case: SyntheticCase,
    *,
    construction: ExactConstructionCaseResult | None,
    reason: str,
) -> ExactCriticalIndexCaseAudit:
    rejection_reasons = (
        construction.rejection_reasons
        if construction is not None and construction.rejection_reasons
        else (reason,)
    )
    return ExactCriticalIndexCaseAudit(
        case_id=case.family.value,
        status="not_audited",
        backend_accepted=bool(construction and construction.accepted),
        backend_name=None if construction is None else construction.backend_name,
        backend_version=None if construction is None else construction.backend_version,
        backend_kernel=None if construction is None else construction.backend_kernel,
        point_count=len(case.points),
        top_simplex_count=0 if construction is None else construction.top_simplex_count,
        floating_critical_count=0,
        exact_rounded_critical_count=0,
        exact_rational_critical_count=0,
        critical_counts_match=None,
        rounded_exact_critical_count_matches_rational=None,
        floating_birth_group_sequence_sha256=None,
        exact_birth_group_sequence_sha256=None,
        critical_birth_group_sequence_matches=None,
        method_identities=(),
        b3_signature_sequence_matches=None,
        floating_b3_candidate_indices=(),
        exact_b3_candidate_indices=(),
        b3_candidate_index_sequence_matches=None,
        b3_persistence_value_difference_count=0,
        b3_max_absolute_persistence_difference=None,
        floating_b3_selected_persistence=None,
        exact_b3_selected_persistence=None,
        b3_selected_persistence_matches=None,
        b3_topology_term_matches=None,
        b3_stability_term_matches=None,
        rejection_reasons=rejection_reasons,
    )


def evaluate_exact_critical_index_audit(
    cases: Iterable[SyntheticCase],
    *,
    construction_result: ExactConstructionPanelResult,
    filtration_audit: ExactFiltrationPanelAudit,
    connectivity_shadow: ExactConnectivityShadowPanelResult,
    value_shadow: ExactValueShadowPanelResult,
    config: BenchmarkConfig,
    methods: Iterable[BaselineID | str],
) -> ExactCriticalIndexPanelAudit:
    """Audit B2/B3 critical identity without changing any benchmark result."""

    materialized_cases = tuple(cases)
    selected_methods = tuple(
        BaselineID(method)
        for method in methods
        if BaselineID(method)
        in (BaselineID.B2_CRITICAL_ORACLE, BaselineID.B3_PERSISTENCE_STABILITY)
    )
    if not materialized_cases:
        raise ValueError("cases must be non-empty")
    if not selected_methods:
        raise ValueError("exact critical index audit requires B2 or B3")
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("methods must not contain duplicates")
    if not (
        construction_result.evaluation_split
        == filtration_audit.evaluation_split
        == connectivity_shadow.evaluation_split
        == value_shadow.evaluation_split
    ):
        raise ValueError("exact critical index prerequisites must use the same split")

    construction_by_id = {case.case_id: case for case in construction_result.cases}
    audit_by_id = {case.case_id: case for case in filtration_audit.cases}
    connectivity_by_id = {case.case_id: case for case in connectivity_shadow.cases}
    value_by_id = {case.case_id: case for case in value_shadow.cases}
    for name, materialized, indexed in (
        ("construction result", construction_result.cases, construction_by_id),
        ("filtration audit", filtration_audit.cases, audit_by_id),
        ("connectivity shadow", connectivity_shadow.cases, connectivity_by_id),
        ("value shadow", value_shadow.cases, value_by_id),
    ):
        if len(indexed) != len(materialized):
            raise ValueError(f"{name} contains duplicate case identifiers")

    case_audits = []
    for case in materialized_cases:
        case_id = case.family.value
        construction = construction_by_id.get(case_id)
        audit: ExactFiltrationCaseAudit | None = audit_by_id.get(case_id)
        floating_case: ExactConnectivityShadowCaseResult | None = (
            connectivity_by_id.get(case_id)
        )
        exact_case: ExactValueShadowCaseResult | None = value_by_id.get(case_id)
        if construction is None:
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=None,
                    reason="backend_result_missing",
                )
            )
            continue
        if not construction.accepted or construction.validated_top_simplices is None:
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="backend_connectivity_not_accepted",
                )
            )
            continue
        if audit is None or not audit.audited or audit.exact_filtration_sha256 is None:
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="exact_filtration_audit_missing",
                )
            )
            continue
        if (
            floating_case is None
            or not floating_case.shadow_ran
            or floating_case.shadow_report is None
        ):
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="floating_connectivity_shadow_missing",
                )
            )
            continue
        if (
            exact_case is None
            or not exact_case.shadow_ran
            or exact_case.shadow_report is None
        ):
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="exact_value_shadow_missing",
                )
            )
            continue
        if exact_case.exact_filtration_sha256 != audit.exact_filtration_sha256:
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="exact_value_shadow_digest_mismatch",
                )
            )
            continue

        try:
            floating_filtration = AlphaFiltration.from_top_simplices(
                case.points,
                construction.validated_top_simplices,
            )
            rounded = exact_rounded_filtration(
                case.points,
                construction.validated_top_simplices,
            )
            if rounded.exact_filtration_sha256 != audit.exact_filtration_sha256:
                raise ValueError("exact filtration digest mismatch")
            exact_filtration = rounded.filtration
            floating_candidates = floating_filtration.critical_values(
                dimensions=[floating_filtration.ambient_dimension]
            )
            exact_candidates = exact_filtration.critical_values(
                dimensions=[exact_filtration.ambient_dimension]
            )
            floating_groups = _floating_birth_groups(floating_filtration)
            exact_groups = _exact_birth_groups(rounded.exact_records)
            method_identities = tuple(
                _method_identity(
                    method,
                    floating_filtration,
                    exact_filtration,
                    floating_case.shadow_report,
                    exact_case.shadow_report,
                )
                for method in selected_methods
            )

            if BaselineID.B3_PERSISTENCE_STABILITY in selected_methods:
                floating_trace = _b3_critical_trace(
                    floating_filtration,
                    config.b3_candidate_budget,
                )
                exact_trace = _b3_critical_trace(
                    exact_filtration,
                    config.b3_candidate_budget,
                )
                persistence_differences = np.abs(
                    floating_trace.persistence - exact_trace.persistence
                )
                b3_identity = next(
                    identity
                    for identity in method_identities
                    if identity.method == BaselineID.B3_PERSISTENCE_STABILITY.value
                )
                floating_selected_persistence = float(
                    floating_trace.persistence[b3_identity.floating_selected_index]
                )
                exact_selected_persistence = float(
                    exact_trace.persistence[b3_identity.exact_selected_index]
                )
                floating_b3_result = _result(
                    floating_case.shadow_report,
                    BaselineID.B3_PERSISTENCE_STABILITY,
                )
                exact_b3_result = _result(
                    exact_case.shadow_report,
                    BaselineID.B3_PERSISTENCE_STABILITY,
                )
                assert floating_b3_result.objective_terms is not None
                assert exact_b3_result.objective_terms is not None
                b3_signature_sequence_matches: bool | None = (
                    floating_trace.signatures == exact_trace.signatures
                )
                floating_b3_candidate_indices = tuple(
                    int(index) for index in floating_trace.selected_indices
                )
                exact_b3_candidate_indices = tuple(
                    int(index) for index in exact_trace.selected_indices
                )
                b3_candidate_index_sequence_matches: bool | None = (
                    floating_b3_candidate_indices == exact_b3_candidate_indices
                )
                b3_persistence_value_difference_count = int(
                    np.count_nonzero(persistence_differences)
                )
                b3_max_absolute_persistence_difference: float | None = float(
                    np.max(persistence_differences)
                )
                b3_selected_persistence_matches: bool | None = math.isclose(
                    floating_selected_persistence,
                    exact_selected_persistence,
                    rel_tol=FLOAT_REL_TOLERANCE,
                    abs_tol=FLOAT_ABS_TOLERANCE,
                )
                b3_topology_term_matches: bool | None = _equivalent(
                    floating_b3_result.objective_terms.topology,
                    exact_b3_result.objective_terms.topology,
                )
                b3_stability_term_matches: bool | None = _equivalent(
                    floating_b3_result.objective_terms.stability,
                    exact_b3_result.objective_terms.stability,
                )
            else:
                b3_signature_sequence_matches = None
                floating_b3_candidate_indices = ()
                exact_b3_candidate_indices = ()
                b3_candidate_index_sequence_matches = None
                b3_persistence_value_difference_count = 0
                b3_max_absolute_persistence_difference = None
                floating_selected_persistence = None
                exact_selected_persistence = None
                b3_selected_persistence_matches = None
                b3_topology_term_matches = None
                b3_stability_term_matches = None
        except (ArithmeticError, ValueError):
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="critical_index_audit_failed",
                )
            )
            continue

        floating_group_sha256 = _group_sha256(floating_groups)
        exact_group_sha256 = _group_sha256(exact_groups)
        case_audits.append(
            ExactCriticalIndexCaseAudit(
                case_id=case_id,
                status="audited",
                backend_accepted=True,
                backend_name=construction.backend_name,
                backend_version=construction.backend_version,
                backend_kernel=construction.backend_kernel,
                point_count=len(case.points),
                top_simplex_count=construction.top_simplex_count,
                floating_critical_count=len(floating_candidates),
                exact_rounded_critical_count=len(exact_candidates),
                exact_rational_critical_count=len(exact_groups),
                critical_counts_match=len(floating_candidates) == len(exact_candidates),
                rounded_exact_critical_count_matches_rational=(
                    len(exact_candidates) == len(exact_groups)
                ),
                floating_birth_group_sequence_sha256=floating_group_sha256,
                exact_birth_group_sequence_sha256=exact_group_sha256,
                critical_birth_group_sequence_matches=(
                    floating_group_sha256 == exact_group_sha256
                ),
                method_identities=method_identities,
                b3_signature_sequence_matches=b3_signature_sequence_matches,
                floating_b3_candidate_indices=floating_b3_candidate_indices,
                exact_b3_candidate_indices=exact_b3_candidate_indices,
                b3_candidate_index_sequence_matches=(
                    b3_candidate_index_sequence_matches
                ),
                b3_persistence_value_difference_count=(
                    b3_persistence_value_difference_count
                ),
                b3_max_absolute_persistence_difference=(
                    b3_max_absolute_persistence_difference
                ),
                floating_b3_selected_persistence=floating_selected_persistence,
                exact_b3_selected_persistence=exact_selected_persistence,
                b3_selected_persistence_matches=b3_selected_persistence_matches,
                b3_topology_term_matches=b3_topology_term_matches,
                b3_stability_term_matches=b3_stability_term_matches,
                rejection_reasons=(),
            )
        )

    return ExactCriticalIndexPanelAudit(
        evaluation_split=construction_result.evaluation_split,
        backend_requested=construction_result.backend_requested,
        requested_case_count=len(materialized_cases),
        accepted_backend_case_count=construction_result.accepted_case_count,
        requested_methods=tuple(method.value for method in selected_methods),
        cases=tuple(case_audits),
    )
