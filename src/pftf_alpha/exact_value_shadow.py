"""Evaluation-only use of correctly rounded exact filtration values."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .baselines import BaselineID, BenchmarkConfig, CaseBenchmark, run_case_benchmarks
from .exact_backend import ExactConstructionCaseResult, ExactConstructionPanelResult
from .exact_filtration import (
    ExactFiltrationCaseAudit,
    ExactFiltrationPanelAudit,
    exact_rounded_filtration,
)
from .exact_shadow import (
    FLOAT_ABS_TOLERANCE,
    FLOAT_REL_TOLERANCE,
    ExactConnectivityShadowCaseResult,
    ExactConnectivityShadowPanelResult,
    _equivalent,
    _report_differences,
)
from .synthetic import SyntheticCase


@dataclass(frozen=True)
class ExactValueShadowCaseResult:
    """One primary/floating-shadow/exact-value-shadow comparison."""

    case_id: str
    status: str
    backend_accepted: bool
    backend_name: str | None
    backend_version: str | None
    backend_kernel: str | None
    connectivity_matches_primary: bool | None
    exact_audit_verified: bool
    exact_filtration_sha256: str | None
    simplex_count: int
    shadow_ran: bool
    compared_method_count: int
    connectivity_dependent_method_count: int
    changed_methods_vs_primary: tuple[str, ...]
    changed_methods_vs_floating_connectivity_shadow: tuple[str, ...]
    selected_alpha_changed_methods_vs_floating_connectivity_shadow: tuple[str, ...]
    objective_changed_methods_vs_floating_connectivity_shadow: tuple[str, ...]
    endpoint_changed_methods_vs_floating_connectivity_shadow: tuple[str, ...]
    candidate_bookkeeping_changed_methods_vs_floating_connectivity_shadow: tuple[
        str, ...
    ]
    bridge_risk_probe_changed_vs_primary: bool | None
    bridge_risk_probe_changed_vs_floating_connectivity_shadow: bool | None
    all_nonruntime_outputs_match_primary: bool | None
    all_nonruntime_outputs_match_floating_connectivity_shadow: bool | None
    rejection_reasons: tuple[str, ...]
    shadow_report: CaseBenchmark | None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "backend_accepted": self.backend_accepted,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "backend_kernel": self.backend_kernel,
            "connectivity_matches_primary": self.connectivity_matches_primary,
            "exact_audit_verified": self.exact_audit_verified,
            "exact_filtration_sha256": self.exact_filtration_sha256,
            "simplex_count": self.simplex_count,
            "shadow_ran": self.shadow_ran,
            "compared_method_count": self.compared_method_count,
            "connectivity_dependent_method_count": (
                self.connectivity_dependent_method_count
            ),
            "changed_methods_vs_primary": list(self.changed_methods_vs_primary),
            "changed_methods_vs_floating_connectivity_shadow": list(
                self.changed_methods_vs_floating_connectivity_shadow
            ),
            "selected_alpha_changed_methods_vs_floating_connectivity_shadow": list(
                self.selected_alpha_changed_methods_vs_floating_connectivity_shadow
            ),
            "objective_changed_methods_vs_floating_connectivity_shadow": list(
                self.objective_changed_methods_vs_floating_connectivity_shadow
            ),
            "endpoint_changed_methods_vs_floating_connectivity_shadow": list(
                self.endpoint_changed_methods_vs_floating_connectivity_shadow
            ),
            (
                "candidate_bookkeeping_changed_methods_vs_floating_connectivity_shadow"
            ): list(
                self.candidate_bookkeeping_changed_methods_vs_floating_connectivity_shadow
            ),
            "bridge_risk_probe_changed_vs_primary": (
                self.bridge_risk_probe_changed_vs_primary
            ),
            "bridge_risk_probe_changed_vs_floating_connectivity_shadow": (
                self.bridge_risk_probe_changed_vs_floating_connectivity_shadow
            ),
            "all_nonruntime_outputs_match_primary": (
                self.all_nonruntime_outputs_match_primary
            ),
            "all_nonruntime_outputs_match_floating_connectivity_shadow": (
                self.all_nonruntime_outputs_match_floating_connectivity_shadow
            ),
            "rejection_reasons": list(self.rejection_reasons),
            "shadow_report": (
                None if self.shadow_report is None else self.shadow_report.to_dict()
            ),
        }


@dataclass(frozen=True)
class ExactValueShadowPanelResult:
    """Panel result isolating exact-value effects on validated connectivity."""

    evaluation_split: str
    backend_requested: bool
    requested_case_count: int
    accepted_backend_case_count: int
    audited_case_count: int
    floating_connectivity_shadow_case_count: int
    cases: tuple[ExactValueShadowCaseResult, ...]

    @property
    def shadow_case_count(self) -> int:
        return sum(case.shadow_ran for case in self.cases)

    @property
    def primary_output_difference_case_count(self) -> int:
        return sum(
            case.shadow_ran and not bool(case.all_nonruntime_outputs_match_primary)
            for case in self.cases
        )

    @property
    def value_only_output_difference_case_count(self) -> int:
        return sum(
            case.shadow_ran
            and not bool(case.all_nonruntime_outputs_match_floating_connectivity_shadow)
            for case in self.cases
        )

    @property
    def selected_alpha_difference_case_count(self) -> int:
        return sum(
            bool(case.selected_alpha_changed_methods_vs_floating_connectivity_shadow)
            for case in self.cases
            if case.shadow_ran
        )

    @property
    def objective_difference_case_count(self) -> int:
        return sum(
            bool(case.objective_changed_methods_vs_floating_connectivity_shadow)
            for case in self.cases
            if case.shadow_ran
        )

    @property
    def endpoint_difference_case_count(self) -> int:
        return sum(
            bool(case.endpoint_changed_methods_vs_floating_connectivity_shadow)
            for case in self.cases
            if case.shadow_ran
        )

    @property
    def candidate_bookkeeping_difference_case_count(self) -> int:
        return sum(
            bool(
                case.candidate_bookkeeping_changed_methods_vs_floating_connectivity_shadow
            )
            for case in self.cases
            if case.shadow_ran
        )

    @property
    def all_prerequisite_cases_evaluated(self) -> bool:
        return (
            self.accepted_backend_case_count > 0
            and self.audited_case_count == self.accepted_backend_case_count
            and self.floating_connectivity_shadow_case_count
            == self.accepted_backend_case_count
            and self.shadow_case_count == self.accepted_backend_case_count
        )

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.backend_requested:
            reasons.append("no_exact_construction_backend")
        elif self.accepted_backend_case_count != self.requested_case_count:
            reasons.append("one_or_more_backend_results_rejected")
        if self.audited_case_count != self.accepted_backend_case_count:
            reasons.append("one_or_more_exact_filtration_audits_missing")
        if (
            self.floating_connectivity_shadow_case_count
            != self.accepted_backend_case_count
        ):
            reasons.append("one_or_more_floating_connectivity_shadows_missing")
        if self.shadow_case_count != self.accepted_backend_case_count:
            reasons.append("one_or_more_exact_value_shadows_not_evaluated")
        reasons.append("exact_rounded_value_shadow_not_deployed")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "evaluation_only_exact_rounded_value_shadow_no_selection",
            "evaluation_split": self.evaluation_split,
            "connectivity_source": "host_validated_exact_backend_tetrahedra",
            "filtration_value_source": "correctly_rounded_exact_rationals",
            "runtime_value_type": "binary64",
            "threshold_and_objective_arithmetic": "floating_point",
            "value_effect_reference": "same_connectivity_floating_value_shadow",
            "comparison_float_relative_tolerance": FLOAT_REL_TOLERANCE,
            "comparison_float_absolute_tolerance": FLOAT_ABS_TOLERANCE,
            "backend_requested": self.backend_requested,
            "requested_case_count": self.requested_case_count,
            "accepted_backend_case_count": self.accepted_backend_case_count,
            "audited_case_count": self.audited_case_count,
            "floating_connectivity_shadow_case_count": (
                self.floating_connectivity_shadow_case_count
            ),
            "shadow_case_count": self.shadow_case_count,
            "primary_output_difference_case_count": (
                self.primary_output_difference_case_count
            ),
            "value_only_output_difference_case_count": (
                self.value_only_output_difference_case_count
            ),
            "selected_alpha_difference_case_count": (
                self.selected_alpha_difference_case_count
            ),
            "objective_difference_case_count": self.objective_difference_case_count,
            "endpoint_difference_case_count": self.endpoint_difference_case_count,
            "candidate_bookkeeping_difference_case_count": (
                self.candidate_bookkeeping_difference_case_count
            ),
            "all_prerequisite_cases_evaluated": (self.all_prerequisite_cases_evaluated),
            "primary_benchmark_results_changed": False,
            "selection_effect": "none",
            "promotion_supported": False,
            "blocking_reasons": list(self.blocking_reasons),
            "cases": [case.to_dict() for case in self.cases],
        }


def _not_run_case(
    case_id: str,
    *,
    construction: ExactConstructionCaseResult | None,
    audit: ExactFiltrationCaseAudit | None,
    connectivity_shadow: ExactConnectivityShadowCaseResult | None,
    reason: str,
) -> ExactValueShadowCaseResult:
    rejection_reasons = (
        construction.rejection_reasons
        if construction is not None and construction.rejection_reasons
        else (reason,)
    )
    return ExactValueShadowCaseResult(
        case_id=case_id,
        status="not_run",
        backend_accepted=bool(construction and construction.accepted),
        backend_name=None if construction is None else construction.backend_name,
        backend_version=None if construction is None else construction.backend_version,
        backend_kernel=None if construction is None else construction.backend_kernel,
        connectivity_matches_primary=(
            None
            if connectivity_shadow is None
            else connectivity_shadow.connectivity_matches_primary
        ),
        exact_audit_verified=bool(audit and audit.audited),
        exact_filtration_sha256=(
            None if audit is None else audit.exact_filtration_sha256
        ),
        simplex_count=0 if audit is None else audit.simplex_count,
        shadow_ran=False,
        compared_method_count=0,
        connectivity_dependent_method_count=0,
        changed_methods_vs_primary=(),
        changed_methods_vs_floating_connectivity_shadow=(),
        selected_alpha_changed_methods_vs_floating_connectivity_shadow=(),
        objective_changed_methods_vs_floating_connectivity_shadow=(),
        endpoint_changed_methods_vs_floating_connectivity_shadow=(),
        candidate_bookkeeping_changed_methods_vs_floating_connectivity_shadow=(),
        bridge_risk_probe_changed_vs_primary=None,
        bridge_risk_probe_changed_vs_floating_connectivity_shadow=None,
        all_nonruntime_outputs_match_primary=None,
        all_nonruntime_outputs_match_floating_connectivity_shadow=None,
        rejection_reasons=rejection_reasons,
        shadow_report=None,
    )


def _method_field_differences(
    left: CaseBenchmark,
    right: CaseBenchmark,
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    left_results = {result.method: result.to_dict() for result in left.results}
    right_results = {result.method: result.to_dict() for result in right.results}
    if left_results.keys() != right_results.keys():
        raise ValueError("reports must contain the same methods")
    return tuple(
        method.value
        for method in left_results
        if any(
            not _equivalent(left_results[method][field], right_results[method][field])
            for field in fields
        )
    )


def evaluate_exact_value_shadow(
    cases: Iterable[SyntheticCase],
    primary_reports: Iterable[CaseBenchmark],
    *,
    construction_result: ExactConstructionPanelResult,
    filtration_audit: ExactFiltrationPanelAudit,
    connectivity_shadow: ExactConnectivityShadowPanelResult,
    config: BenchmarkConfig,
    methods: Iterable[BaselineID | str],
) -> ExactValueShadowPanelResult:
    """Rerun methods on exact-rounded values without changing primary reports."""

    materialized_cases = tuple(cases)
    materialized_reports = tuple(primary_reports)
    selected_methods = tuple(BaselineID(method) for method in methods)
    if not materialized_cases:
        raise ValueError("cases must be non-empty")
    if len(materialized_cases) != len(materialized_reports):
        raise ValueError("cases and primary_reports must have the same length")
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("methods must not contain duplicates")
    if not any(method is not BaselineID.B0_CONVEX_HULL for method in selected_methods):
        raise ValueError("exact value shadow requires a non-B0 method")
    if not (
        construction_result.evaluation_split
        == filtration_audit.evaluation_split
        == connectivity_shadow.evaluation_split
    ):
        raise ValueError("exact shadow prerequisites must use the same split")

    construction_by_id = {case.case_id: case for case in construction_result.cases}
    audit_by_id = {case.case_id: case for case in filtration_audit.cases}
    connectivity_by_id = {case.case_id: case for case in connectivity_shadow.cases}
    primary_by_id = {report.family: report for report in materialized_reports}
    for name, materialized, indexed in (
        ("construction result", construction_result.cases, construction_by_id),
        ("filtration audit", filtration_audit.cases, audit_by_id),
        ("connectivity shadow", connectivity_shadow.cases, connectivity_by_id),
        ("primary reports", materialized_reports, primary_by_id),
    ):
        if len(indexed) != len(materialized):
            raise ValueError(f"{name} contains duplicate case identifiers")

    comparisons = []
    for case in materialized_cases:
        case_id = case.family.value
        primary = primary_by_id.get(case_id)
        if primary is None:
            raise ValueError(f"primary report missing for case {case_id}")
        construction = construction_by_id.get(case_id)
        audit = audit_by_id.get(case_id)
        floating_shadow = connectivity_by_id.get(case_id)
        if construction is None:
            comparisons.append(
                _not_run_case(
                    case_id,
                    construction=None,
                    audit=audit,
                    connectivity_shadow=floating_shadow,
                    reason="backend_result_missing",
                )
            )
            continue
        if not construction.accepted or construction.validated_top_simplices is None:
            comparisons.append(
                _not_run_case(
                    case_id,
                    construction=construction,
                    audit=audit,
                    connectivity_shadow=floating_shadow,
                    reason="backend_connectivity_not_accepted",
                )
            )
            continue
        if audit is None or not audit.audited or audit.exact_filtration_sha256 is None:
            comparisons.append(
                _not_run_case(
                    case_id,
                    construction=construction,
                    audit=audit,
                    connectivity_shadow=floating_shadow,
                    reason="exact_filtration_audit_missing",
                )
            )
            continue
        if (
            floating_shadow is None
            or not floating_shadow.shadow_ran
            or floating_shadow.shadow_report is None
        ):
            comparisons.append(
                _not_run_case(
                    case_id,
                    construction=construction,
                    audit=audit,
                    connectivity_shadow=floating_shadow,
                    reason="floating_connectivity_shadow_missing",
                )
            )
            continue

        try:
            rounded = exact_rounded_filtration(
                case.points,
                construction.validated_top_simplices,
            )
        except (ArithmeticError, ValueError):
            comparisons.append(
                _not_run_case(
                    case_id,
                    construction=construction,
                    audit=audit,
                    connectivity_shadow=floating_shadow,
                    reason="exact_rounded_filtration_construction_failed",
                )
            )
            continue
        if rounded.exact_filtration_sha256 != audit.exact_filtration_sha256:
            comparisons.append(
                _not_run_case(
                    case_id,
                    construction=construction,
                    audit=audit,
                    connectivity_shadow=floating_shadow,
                    reason="exact_filtration_digest_mismatch",
                )
            )
            continue
        if rounded.simplex_count != audit.simplex_count:
            comparisons.append(
                _not_run_case(
                    case_id,
                    construction=construction,
                    audit=audit,
                    connectivity_shadow=floating_shadow,
                    reason="exact_filtration_simplex_count_mismatch",
                )
            )
            continue

        exact_shadow_report = run_case_benchmarks(
            case,
            config=config,
            methods=selected_methods,
            filtration=rounded.filtration,
        )
        primary_changes, primary_bridge_change = _report_differences(
            primary,
            exact_shadow_report,
        )
        value_changes, value_bridge_change = _report_differences(
            floating_shadow.shadow_report,
            exact_shadow_report,
        )
        selected_alpha_changes = _method_field_differences(
            floating_shadow.shadow_report,
            exact_shadow_report,
            (
                "alpha_squared",
                "alpha_radius_fraction",
                "selection_parameter_name",
                "selection_parameter_value",
            ),
        )
        objective_changes = _method_field_differences(
            floating_shadow.shadow_report,
            exact_shadow_report,
            ("objective_total", "objective_terms"),
        )
        endpoint_changes = _method_field_differences(
            floating_shadow.shadow_report,
            exact_shadow_report,
            ("endpoints",),
        )
        candidate_bookkeeping_changes = _method_field_differences(
            floating_shadow.shadow_report,
            exact_shadow_report,
            (
                "total_candidates_scanned",
                "candidate_alpha_squared_min",
                "candidate_alpha_squared_max",
                "candidate_count",
                "candidate_parameter_min",
                "candidate_parameter_max",
            ),
        )
        comparisons.append(
            ExactValueShadowCaseResult(
                case_id=case_id,
                status="evaluated",
                backend_accepted=True,
                backend_name=construction.backend_name,
                backend_version=construction.backend_version,
                backend_kernel=construction.backend_kernel,
                connectivity_matches_primary=(
                    floating_shadow.connectivity_matches_primary
                ),
                exact_audit_verified=True,
                exact_filtration_sha256=rounded.exact_filtration_sha256,
                simplex_count=rounded.simplex_count,
                shadow_ran=True,
                compared_method_count=len(selected_methods),
                connectivity_dependent_method_count=sum(
                    method is not BaselineID.B0_CONVEX_HULL
                    for method in selected_methods
                ),
                changed_methods_vs_primary=primary_changes,
                changed_methods_vs_floating_connectivity_shadow=value_changes,
                selected_alpha_changed_methods_vs_floating_connectivity_shadow=(
                    selected_alpha_changes
                ),
                objective_changed_methods_vs_floating_connectivity_shadow=(
                    objective_changes
                ),
                endpoint_changed_methods_vs_floating_connectivity_shadow=(
                    endpoint_changes
                ),
                candidate_bookkeeping_changed_methods_vs_floating_connectivity_shadow=(
                    candidate_bookkeeping_changes
                ),
                bridge_risk_probe_changed_vs_primary=primary_bridge_change,
                bridge_risk_probe_changed_vs_floating_connectivity_shadow=(
                    value_bridge_change
                ),
                all_nonruntime_outputs_match_primary=(
                    not primary_changes and not primary_bridge_change
                ),
                all_nonruntime_outputs_match_floating_connectivity_shadow=(
                    not value_changes and not value_bridge_change
                ),
                rejection_reasons=(),
                shadow_report=exact_shadow_report,
            )
        )

    return ExactValueShadowPanelResult(
        evaluation_split=construction_result.evaluation_split,
        backend_requested=construction_result.backend_requested,
        requested_case_count=len(materialized_cases),
        accepted_backend_case_count=construction_result.accepted_case_count,
        audited_case_count=filtration_audit.audited_case_count,
        floating_connectivity_shadow_case_count=(connectivity_shadow.shadow_case_count),
        cases=tuple(comparisons),
    )
