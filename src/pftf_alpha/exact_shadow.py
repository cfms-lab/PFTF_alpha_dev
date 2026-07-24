"""Evaluation-only application of host-validated exact connectivity."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .baselines import (
    BaselineID,
    BaselineResult,
    BenchmarkConfig,
    CaseBenchmark,
    run_case_benchmarks,
)
from .exact_backend import (
    ExactConstructionCaseResult,
    ExactConstructionPanelResult,
)
from .filtration import AlphaFiltration
from .synthetic import SyntheticCase

FLOAT_REL_TOLERANCE = 1.0e-12
FLOAT_ABS_TOLERANCE = 1.0e-15


@dataclass(frozen=True)
class ExactConnectivityShadowCaseResult:
    """One primary-vs-validated-connectivity shadow comparison."""

    case_id: str
    status: str
    backend_accepted: bool
    backend_name: str | None
    backend_version: str | None
    backend_kernel: str | None
    primary_top_simplex_count: int | None
    validated_top_simplex_count: int | None
    connectivity_matches_primary: bool | None
    shadow_ran: bool
    compared_method_count: int
    connectivity_dependent_method_count: int
    changed_methods: tuple[str, ...]
    bridge_risk_probe_changed: bool | None
    all_nonruntime_outputs_match: bool | None
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
            "primary_top_simplex_count": self.primary_top_simplex_count,
            "validated_top_simplex_count": self.validated_top_simplex_count,
            "connectivity_matches_primary": self.connectivity_matches_primary,
            "shadow_ran": self.shadow_ran,
            "compared_method_count": self.compared_method_count,
            "connectivity_dependent_method_count": (
                self.connectivity_dependent_method_count
            ),
            "changed_methods": list(self.changed_methods),
            "bridge_risk_probe_changed": self.bridge_risk_probe_changed,
            "all_nonruntime_outputs_match": self.all_nonruntime_outputs_match,
            "rejection_reasons": list(self.rejection_reasons),
            "shadow_report": (
                None if self.shadow_report is None else self.shadow_report.to_dict()
            ),
        }


@dataclass(frozen=True)
class ExactConnectivityShadowPanelResult:
    """Panel-level shadow result that cannot alter primary selection."""

    evaluation_split: str
    backend_requested: bool
    requested_case_count: int
    accepted_backend_case_count: int
    cases: tuple[ExactConnectivityShadowCaseResult, ...]

    @property
    def shadow_case_count(self) -> int:
        return sum(case.shadow_ran for case in self.cases)

    @property
    def output_difference_case_count(self) -> int:
        return sum(
            case.shadow_ran and not bool(case.all_nonruntime_outputs_match)
            for case in self.cases
        )

    @property
    def all_accepted_cases_evaluated(self) -> bool:
        return (
            self.accepted_backend_case_count > 0
            and self.shadow_case_count == self.accepted_backend_case_count
        )

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.backend_requested:
            reasons.append("no_exact_construction_backend")
        elif self.accepted_backend_case_count != self.requested_case_count:
            reasons.append("one_or_more_backend_results_rejected")
        if self.shadow_case_count != self.accepted_backend_case_count:
            reasons.append("one_or_more_accepted_connectivities_not_evaluated")
        reasons.append("exact_connectivity_shadow_not_deployed")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "evaluation_only_shadow_no_selection",
            "evaluation_split": self.evaluation_split,
            "connectivity_source": "host_validated_exact_backend_tetrahedra",
            "filtration_values": "floating_point_intrinsic_circumspheres",
            "comparison_float_relative_tolerance": FLOAT_REL_TOLERANCE,
            "comparison_float_absolute_tolerance": FLOAT_ABS_TOLERANCE,
            "backend_requested": self.backend_requested,
            "requested_case_count": self.requested_case_count,
            "accepted_backend_case_count": self.accepted_backend_case_count,
            "shadow_case_count": self.shadow_case_count,
            "output_difference_case_count": self.output_difference_case_count,
            "all_accepted_cases_evaluated": self.all_accepted_cases_evaluated,
            "primary_benchmark_results_changed": False,
            "selection_effect": "none",
            "promotion_supported": False,
            "blocking_reasons": list(self.blocking_reasons),
            "cases": [case.to_dict() for case in self.cases],
        }


def _result_without_runtime(result: BaselineResult) -> dict[str, object]:
    payload = result.to_dict()
    payload.pop("runtime_seconds")
    return payload


def _equivalent(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return math.isclose(
            left,
            right,
            rel_tol=FLOAT_REL_TOLERANCE,
            abs_tol=FLOAT_ABS_TOLERANCE,
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _equivalent(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _report_differences(
    primary: CaseBenchmark,
    shadow: CaseBenchmark,
) -> tuple[tuple[str, ...], bool]:
    primary_results = {result.method: result for result in primary.results}
    shadow_results = {result.method: result for result in shadow.results}
    if primary_results.keys() != shadow_results.keys():
        raise ValueError("primary and shadow reports must contain the same methods")
    changed_methods = tuple(
        method.value
        for method in primary_results
        if not _equivalent(
            _result_without_runtime(primary_results[method]),
            _result_without_runtime(shadow_results[method]),
        )
    )
    primary_probe = (
        None
        if primary.bridge_risk_probe is None
        else primary.bridge_risk_probe.to_dict()
    )
    shadow_probe = (
        None if shadow.bridge_risk_probe is None else shadow.bridge_risk_probe.to_dict()
    )
    return changed_methods, not _equivalent(primary_probe, shadow_probe)


def _not_run_case(
    case_id: str,
    backend_case: ExactConstructionCaseResult | None,
    reason: str,
) -> ExactConnectivityShadowCaseResult:
    return ExactConnectivityShadowCaseResult(
        case_id=case_id,
        status="not_run",
        backend_accepted=False if backend_case is None else backend_case.accepted,
        backend_name=None if backend_case is None else backend_case.backend_name,
        backend_version=None if backend_case is None else backend_case.backend_version,
        backend_kernel=None if backend_case is None else backend_case.backend_kernel,
        primary_top_simplex_count=None,
        validated_top_simplex_count=(
            None if backend_case is None else backend_case.top_simplex_count
        ),
        connectivity_matches_primary=None,
        shadow_ran=False,
        compared_method_count=0,
        connectivity_dependent_method_count=0,
        changed_methods=(),
        bridge_risk_probe_changed=None,
        all_nonruntime_outputs_match=None,
        rejection_reasons=(
            (reason,)
            if backend_case is None or not backend_case.rejection_reasons
            else backend_case.rejection_reasons
        ),
        shadow_report=None,
    )


def evaluate_exact_connectivity_shadow(
    cases: Iterable[SyntheticCase],
    primary_reports: Iterable[CaseBenchmark],
    *,
    construction_result: ExactConstructionPanelResult,
    config: BenchmarkConfig,
    methods: Iterable[BaselineID | str],
) -> ExactConnectivityShadowPanelResult:
    """Re-run methods on validated connectivity without changing primary reports."""

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
        raise ValueError("exact connectivity shadow requires a non-B0 method")

    backend_cases = {case.case_id: case for case in construction_result.cases}
    if len(backend_cases) != len(construction_result.cases):
        raise ValueError("construction result contains duplicate case identifiers")
    primary_by_id = {report.family: report for report in materialized_reports}
    if len(primary_by_id) != len(materialized_reports):
        raise ValueError("primary reports contain duplicate case identifiers")

    comparisons = []
    for case in materialized_cases:
        case_id = case.family.value
        primary_report = primary_by_id.get(case_id)
        if primary_report is None:
            raise ValueError(f"primary report missing for case {case_id}")
        backend_case = backend_cases.get(case_id)
        if backend_case is None:
            reason = (
                "no_exact_construction_backend"
                if not construction_result.backend_requested
                else "backend_result_missing"
            )
            comparisons.append(_not_run_case(case_id, None, reason))
            continue
        if not backend_case.accepted:
            comparisons.append(
                _not_run_case(case_id, backend_case, "backend_result_rejected")
            )
            continue
        if backend_case.validated_top_simplices is None:
            comparisons.append(
                _not_run_case(
                    case_id,
                    backend_case,
                    "validated_connectivity_missing",
                )
            )
            continue

        try:
            primary_filtration = AlphaFiltration.from_points(case.points)
            shadow_filtration = AlphaFiltration.from_top_simplices(
                case.points,
                backend_case.validated_top_simplices,
            )
        except (ArithmeticError, ValueError):
            comparisons.append(
                _not_run_case(
                    case_id,
                    backend_case,
                    "shadow_filtration_construction_failed",
                )
            )
            continue

        shadow_report = run_case_benchmarks(
            case,
            config=config,
            methods=selected_methods,
            filtration=shadow_filtration,
        )
        changed_methods, bridge_risk_changed = _report_differences(
            primary_report,
            shadow_report,
        )
        primary_cells = {
            tuple(sorted(int(vertex) for vertex in cell))
            for cell in primary_filtration.top_simplices
        }
        shadow_cells = {
            tuple(sorted(int(vertex) for vertex in cell))
            for cell in shadow_filtration.top_simplices
        }
        comparisons.append(
            ExactConnectivityShadowCaseResult(
                case_id=case_id,
                status="evaluated",
                backend_accepted=True,
                backend_name=backend_case.backend_name,
                backend_version=backend_case.backend_version,
                backend_kernel=backend_case.backend_kernel,
                primary_top_simplex_count=len(primary_cells),
                validated_top_simplex_count=len(shadow_cells),
                connectivity_matches_primary=primary_cells == shadow_cells,
                shadow_ran=True,
                compared_method_count=len(selected_methods),
                connectivity_dependent_method_count=sum(
                    method is not BaselineID.B0_CONVEX_HULL
                    for method in selected_methods
                ),
                changed_methods=changed_methods,
                bridge_risk_probe_changed=bridge_risk_changed,
                all_nonruntime_outputs_match=(
                    not changed_methods and not bridge_risk_changed
                ),
                rejection_reasons=(),
                shadow_report=shadow_report,
            )
        )

    return ExactConnectivityShadowPanelResult(
        evaluation_split=construction_result.evaluation_split,
        backend_requested=construction_result.backend_requested,
        requested_case_count=len(materialized_cases),
        accepted_backend_case_count=construction_result.accepted_case_count,
        cases=tuple(comparisons),
    )
