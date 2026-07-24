"""Audit exact-selected thresholds on the shared floating resampling path."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from .baselines import (
    BaselineID,
    BenchmarkConfig,
    _b3_critical_trace,
    _resampled_filtrations,
    _stability_loss,
    _unlabeled_geometry_loss,
)
from .exact_backend import ExactConstructionCaseResult, ExactConstructionPanelResult
from .exact_filtration import (
    ExactFiltrationCaseAudit,
    ExactFiltrationPanelAudit,
    exact_rounded_filtration,
)
from .exact_index_audit import (
    ExactCriticalIndexCaseAudit,
    ExactCriticalIndexPanelAudit,
    _boundary_sha256,
    _complex_sha256,
    _result,
)
from .exact_shadow import (
    FLOAT_ABS_TOLERANCE,
    FLOAT_REL_TOLERANCE,
    ExactConnectivityShadowCaseResult,
    ExactConnectivityShadowPanelResult,
)
from .exact_value_shadow import ExactValueShadowCaseResult, ExactValueShadowPanelResult
from .filtration import AlphaFiltration
from .surface import alpha_surface
from .synthetic import SyntheticCase


@dataclass(frozen=True)
class ExactResamplingRepeatAudit:
    """One shared floating resample evaluated at two selected thresholds."""

    repeat_index: int
    resampled_point_count: int
    resampled_top_simplex_count: int
    floating_threshold_complex_sha256: str
    exact_threshold_complex_sha256: str
    complex_matches: bool
    floating_threshold_boundary_sha256: str
    exact_threshold_boundary_sha256: str
    boundary_matches: bool
    floating_threshold_stability_loss: float
    exact_threshold_stability_loss: float
    stability_loss_matches: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExactResamplingThresholdCaseAudit:
    """Case-level isolation of the selected-alpha effect on B3 resampling."""

    case_id: str
    status: str
    backend_accepted: bool
    backend_name: str | None
    backend_version: str | None
    backend_kernel: str | None
    point_count: int
    top_simplex_count: int
    floating_selected_index: int | None
    exact_selected_index: int | None
    selected_index_matches: bool | None
    candidate_local_position: int | None
    floating_alpha_squared: float | None
    exact_alpha_squared: float | None
    alpha_absolute_difference: float | None
    floating_full_samples_sha256: str | None
    exact_full_samples_sha256: str | None
    full_surface_samples_match: bool | None
    full_surface_sample_count: int
    resample_repeat_count: int
    repeats: tuple[ExactResamplingRepeatAudit, ...]
    floating_reported_stability: float | None
    exact_reported_stability: float | None
    floating_recomputed_stability: float | None
    exact_threshold_recomputed_stability: float | None
    floating_recomputation_matches_report: bool | None
    exact_recomputation_matches_report: bool | None
    reported_stability_matches: bool | None
    recomputed_stability_matches: bool | None
    threshold_effect_reproduced: bool | None
    rejection_reasons: tuple[str, ...]

    @property
    def audited(self) -> bool:
        return self.status == "audited"

    @property
    def resampled_complex_difference_repeat_count(self) -> int:
        return sum(not repeat.complex_matches for repeat in self.repeats)

    @property
    def resampled_boundary_difference_repeat_count(self) -> int:
        return sum(not repeat.boundary_matches for repeat in self.repeats)

    @property
    def resampled_stability_difference_repeat_count(self) -> int:
        return sum(not repeat.stability_loss_matches for repeat in self.repeats)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["repeats"] = [repeat.to_dict() for repeat in self.repeats]
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["resampled_complex_difference_repeat_count"] = (
            self.resampled_complex_difference_repeat_count
        )
        payload["resampled_boundary_difference_repeat_count"] = (
            self.resampled_boundary_difference_repeat_count
        )
        payload["resampled_stability_difference_repeat_count"] = (
            self.resampled_stability_difference_repeat_count
        )
        return payload


@dataclass(frozen=True)
class ExactResamplingThresholdPanelAudit:
    """Panel audit that never constructs exact resampled connectivity."""

    evaluation_split: str
    backend_requested: bool
    requested_case_count: int
    accepted_backend_case_count: int
    cases: tuple[ExactResamplingThresholdCaseAudit, ...]

    @property
    def audited_case_count(self) -> int:
        return sum(case.audited for case in self.cases)

    @property
    def threshold_changed_case_count(self) -> int:
        return sum(
            case.audited
            and case.alpha_absolute_difference is not None
            and case.alpha_absolute_difference > 0.0
            for case in self.cases
        )

    @property
    def resampled_complex_difference_case_count(self) -> int:
        return sum(
            case.audited and case.resampled_complex_difference_repeat_count > 0
            for case in self.cases
        )

    @property
    def resampled_complex_difference_repeat_count(self) -> int:
        return sum(
            case.resampled_complex_difference_repeat_count
            for case in self.cases
            if case.audited
        )

    @property
    def resampled_boundary_difference_case_count(self) -> int:
        return sum(
            case.audited and case.resampled_boundary_difference_repeat_count > 0
            for case in self.cases
        )

    @property
    def resampled_boundary_difference_repeat_count(self) -> int:
        return sum(
            case.resampled_boundary_difference_repeat_count
            for case in self.cases
            if case.audited
        )

    @property
    def stability_difference_case_count(self) -> int:
        return sum(
            case.audited and case.reported_stability_matches is False
            for case in self.cases
        )

    @property
    def threshold_effect_reproduction_failure_case_count(self) -> int:
        return sum(
            case.audited and case.threshold_effect_reproduced is False
            for case in self.cases
        )

    @property
    def stability_difference_without_boundary_change_case_count(self) -> int:
        return sum(
            case.audited
            and case.reported_stability_matches is False
            and case.resampled_boundary_difference_repeat_count == 0
            for case in self.cases
        )

    @property
    def boundary_change_without_stability_difference_case_count(self) -> int:
        return sum(
            case.audited
            and case.reported_stability_matches is True
            and case.resampled_boundary_difference_repeat_count > 0
            for case in self.cases
        )

    @property
    def all_reported_stability_reproduced(self) -> bool:
        return (
            self.audited_case_count > 0
            and self.audited_case_count == self.accepted_backend_case_count
            and self.threshold_effect_reproduction_failure_case_count == 0
        )

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.backend_requested:
            reasons.append("no_exact_construction_backend")
        elif self.accepted_backend_case_count != self.requested_case_count:
            reasons.append("one_or_more_backend_results_rejected")
        if self.audited_case_count != self.accepted_backend_case_count:
            reasons.append("one_or_more_resampling_threshold_audits_not_completed")
        reasons.append("exact_resampling_threshold_audit_not_deployed")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "evaluation_only_exact_selected_threshold_resampling_audit",
            "evaluation_split": self.evaluation_split,
            "resampled_connectivity": "shared_floating_scipy_qhull",
            "resampled_filtration_values": "floating_point",
            "threshold_sources": [
                "same_connectivity_floating_selected_alpha",
                "same_connectivity_exact_rounded_selected_alpha",
            ],
            "exact_resampled_connectivity_constructed": False,
            "comparison_float_relative_tolerance": FLOAT_REL_TOLERANCE,
            "comparison_float_absolute_tolerance": FLOAT_ABS_TOLERANCE,
            "backend_requested": self.backend_requested,
            "requested_case_count": self.requested_case_count,
            "accepted_backend_case_count": self.accepted_backend_case_count,
            "audited_case_count": self.audited_case_count,
            "threshold_changed_case_count": self.threshold_changed_case_count,
            "resampled_complex_difference_case_count": (
                self.resampled_complex_difference_case_count
            ),
            "resampled_complex_difference_repeat_count": (
                self.resampled_complex_difference_repeat_count
            ),
            "resampled_boundary_difference_case_count": (
                self.resampled_boundary_difference_case_count
            ),
            "resampled_boundary_difference_repeat_count": (
                self.resampled_boundary_difference_repeat_count
            ),
            "stability_difference_case_count": self.stability_difference_case_count,
            "threshold_effect_reproduction_failure_case_count": (
                self.threshold_effect_reproduction_failure_case_count
            ),
            "stability_difference_without_boundary_change_case_count": (
                self.stability_difference_without_boundary_change_case_count
            ),
            "boundary_change_without_stability_difference_case_count": (
                self.boundary_change_without_stability_difference_case_count
            ),
            "all_reported_stability_reproduced": (
                self.all_reported_stability_reproduced
            ),
            "primary_benchmark_results_changed": False,
            "selection_effect": "none",
            "promotion_supported": False,
            "blocking_reasons": list(self.blocking_reasons),
            "cases": [case.to_dict() for case in self.cases],
        }


def _sample_sha256(samples: np.ndarray) -> str:
    canonical = np.ascontiguousarray(samples, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _matches(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=FLOAT_REL_TOLERANCE,
        abs_tol=FLOAT_ABS_TOLERANCE,
    )


def _not_audited_case(
    case: SyntheticCase,
    *,
    construction: ExactConstructionCaseResult | None,
    reason: str,
) -> ExactResamplingThresholdCaseAudit:
    rejection_reasons = (
        construction.rejection_reasons
        if construction is not None and construction.rejection_reasons
        else (reason,)
    )
    return ExactResamplingThresholdCaseAudit(
        case_id=case.family.value,
        status="not_audited",
        backend_accepted=bool(construction and construction.accepted),
        backend_name=None if construction is None else construction.backend_name,
        backend_version=None if construction is None else construction.backend_version,
        backend_kernel=None if construction is None else construction.backend_kernel,
        point_count=len(case.points),
        top_simplex_count=0 if construction is None else construction.top_simplex_count,
        floating_selected_index=None,
        exact_selected_index=None,
        selected_index_matches=None,
        candidate_local_position=None,
        floating_alpha_squared=None,
        exact_alpha_squared=None,
        alpha_absolute_difference=None,
        floating_full_samples_sha256=None,
        exact_full_samples_sha256=None,
        full_surface_samples_match=None,
        full_surface_sample_count=0,
        resample_repeat_count=0,
        repeats=(),
        floating_reported_stability=None,
        exact_reported_stability=None,
        floating_recomputed_stability=None,
        exact_threshold_recomputed_stability=None,
        floating_recomputation_matches_report=None,
        exact_recomputation_matches_report=None,
        reported_stability_matches=None,
        recomputed_stability_matches=None,
        threshold_effect_reproduced=None,
        rejection_reasons=rejection_reasons,
    )


def _unique_indexed(
    name: str,
    materialized: tuple[object, ...],
    indexed: dict[str, object],
) -> None:
    if len(indexed) != len(materialized):
        raise ValueError(f"{name} contains duplicate case identifiers")


def evaluate_exact_resampling_threshold_audit(
    cases: Iterable[SyntheticCase],
    *,
    construction_result: ExactConstructionPanelResult,
    filtration_audit: ExactFiltrationPanelAudit,
    connectivity_shadow: ExactConnectivityShadowPanelResult,
    value_shadow: ExactValueShadowPanelResult,
    critical_index_audit: ExactCriticalIndexPanelAudit,
    config: BenchmarkConfig,
) -> ExactResamplingThresholdPanelAudit:
    """Reproduce B3 stability using two thresholds on shared float resamples."""

    materialized_cases = tuple(cases)
    if not materialized_cases:
        raise ValueError("cases must be non-empty")
    if BaselineID.B3_PERSISTENCE_STABILITY.value not in (
        critical_index_audit.requested_methods
    ):
        raise ValueError("exact resampling threshold audit requires B3")
    if not (
        construction_result.evaluation_split
        == filtration_audit.evaluation_split
        == connectivity_shadow.evaluation_split
        == value_shadow.evaluation_split
        == critical_index_audit.evaluation_split
    ):
        raise ValueError("exact resampling prerequisites must use the same split")

    construction_by_id = {case.case_id: case for case in construction_result.cases}
    filtration_by_id = {case.case_id: case for case in filtration_audit.cases}
    connectivity_by_id = {case.case_id: case for case in connectivity_shadow.cases}
    value_by_id = {case.case_id: case for case in value_shadow.cases}
    critical_by_id = {case.case_id: case for case in critical_index_audit.cases}
    for name, materialized, indexed in (
        ("construction result", construction_result.cases, construction_by_id),
        ("filtration audit", filtration_audit.cases, filtration_by_id),
        ("connectivity shadow", connectivity_shadow.cases, connectivity_by_id),
        ("value shadow", value_shadow.cases, value_by_id),
        ("critical index audit", critical_index_audit.cases, critical_by_id),
    ):
        _unique_indexed(name, materialized, indexed)

    case_audits = []
    for case in materialized_cases:
        case_id = case.family.value
        construction: ExactConstructionCaseResult | None = construction_by_id.get(
            case_id
        )
        audit: ExactFiltrationCaseAudit | None = filtration_by_id.get(case_id)
        floating_case: ExactConnectivityShadowCaseResult | None = (
            connectivity_by_id.get(case_id)
        )
        exact_case: ExactValueShadowCaseResult | None = value_by_id.get(case_id)
        critical_case: ExactCriticalIndexCaseAudit | None = critical_by_id.get(case_id)
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
        if critical_case is None or not critical_case.audited:
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="critical_index_audit_missing",
                )
            )
            continue

        identities = tuple(
            identity
            for identity in critical_case.method_identities
            if identity.method == BaselineID.B3_PERSISTENCE_STABILITY.value
        )
        if len(identities) != 1:
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="b3_critical_identity_missing",
                )
            )
            continue
        identity = identities[0]
        if not (
            identity.selected_index_matches
            and identity.selected_complex_matches
            and identity.selected_boundary_matches
            and critical_case.b3_candidate_index_sequence_matches is True
        ):
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="b3_selection_identity_not_shared",
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
            floating_trace = _b3_critical_trace(
                floating_filtration,
                config.b3_candidate_budget,
            )
            exact_trace = _b3_critical_trace(
                exact_filtration,
                config.b3_candidate_budget,
            )
            floating_positions = np.flatnonzero(
                floating_trace.selected_indices == identity.floating_selected_index
            )
            exact_positions = np.flatnonzero(
                exact_trace.selected_indices == identity.exact_selected_index
            )
            if floating_positions.size != 1 or exact_positions.size != 1:
                raise ValueError("selected B3 critical index is not budgeted")
            floating_position = int(floating_positions[0])
            exact_position = int(exact_positions[0])
            if floating_position != exact_position:
                raise ValueError("selected B3 local candidate position differs")
            local_position = floating_position

            floating_result = _result(
                floating_case.shadow_report,
                BaselineID.B3_PERSISTENCE_STABILITY,
            )
            exact_result = _result(
                exact_case.shadow_report,
                BaselineID.B3_PERSISTENCE_STABILITY,
            )
            if (
                floating_result.alpha_squared is None
                or exact_result.alpha_squared is None
                or floating_result.objective_terms is None
                or exact_result.objective_terms is None
            ):
                raise ValueError("B3 reports must contain alpha and objective terms")
            if (
                floating_result.alpha_squared != identity.floating_alpha_squared
                or exact_result.alpha_squared != identity.exact_alpha_squared
            ):
                raise ValueError("B3 report and critical identity alpha differ")

            floating_mesh = alpha_surface(
                floating_filtration,
                identity.floating_alpha_squared,
            )
            exact_mesh = alpha_surface(
                exact_filtration,
                identity.exact_alpha_squared,
            )
            full_seed = config.seed + case.seed + 40_000 + local_position
            _, floating_samples = _unlabeled_geometry_loss(
                floating_mesh,
                case.points,
                case,
                config,
                seed=full_seed,
            )
            _, exact_samples = _unlabeled_geometry_loss(
                exact_mesh,
                case.points,
                case,
                config,
                seed=full_seed,
            )
            if not np.array_equal(floating_samples, exact_samples):
                raise ValueError("full surface samples differ")

            resampled = _resampled_filtrations(case, config)
            stability_seed = config.seed + case.seed + 50_000 + local_position
            repeat_audits = []
            floating_losses = []
            exact_losses = []
            for repeat_index, resampled_filtration in enumerate(resampled):
                floating_resampled_mesh = alpha_surface(
                    resampled_filtration,
                    identity.floating_alpha_squared,
                )
                exact_threshold_resampled_mesh = alpha_surface(
                    resampled_filtration,
                    identity.exact_alpha_squared,
                )
                repeat_seed = stability_seed + repeat_index
                floating_loss = _stability_loss(
                    floating_samples,
                    (floating_resampled_mesh,),
                    case,
                    config,
                    seed=repeat_seed,
                )
                exact_loss = _stability_loss(
                    exact_samples,
                    (exact_threshold_resampled_mesh,),
                    case,
                    config,
                    seed=repeat_seed,
                )
                floating_losses.append(floating_loss)
                exact_losses.append(exact_loss)
                floating_complex_sha256 = _complex_sha256(
                    resampled_filtration,
                    identity.floating_alpha_squared,
                )
                exact_complex_sha256 = _complex_sha256(
                    resampled_filtration,
                    identity.exact_alpha_squared,
                )
                floating_boundary_sha256 = _boundary_sha256(
                    resampled_filtration,
                    identity.floating_alpha_squared,
                )
                exact_boundary_sha256 = _boundary_sha256(
                    resampled_filtration,
                    identity.exact_alpha_squared,
                )
                repeat_audits.append(
                    ExactResamplingRepeatAudit(
                        repeat_index=repeat_index,
                        resampled_point_count=len(resampled_filtration.points),
                        resampled_top_simplex_count=len(
                            resampled_filtration.top_simplices
                        ),
                        floating_threshold_complex_sha256=(floating_complex_sha256),
                        exact_threshold_complex_sha256=exact_complex_sha256,
                        complex_matches=(
                            floating_complex_sha256 == exact_complex_sha256
                        ),
                        floating_threshold_boundary_sha256=(floating_boundary_sha256),
                        exact_threshold_boundary_sha256=exact_boundary_sha256,
                        boundary_matches=(
                            floating_boundary_sha256 == exact_boundary_sha256
                        ),
                        floating_threshold_stability_loss=floating_loss,
                        exact_threshold_stability_loss=exact_loss,
                        stability_loss_matches=_matches(
                            floating_loss,
                            exact_loss,
                        ),
                    )
                )
            floating_recomputed = float(np.mean(floating_losses))
            exact_recomputed = float(np.mean(exact_losses))
            floating_reported = floating_result.objective_terms.stability
            exact_reported = exact_result.objective_terms.stability
            floating_reproduced = _matches(floating_recomputed, floating_reported)
            exact_reproduced = _matches(exact_recomputed, exact_reported)
        except (ArithmeticError, ValueError):
            case_audits.append(
                _not_audited_case(
                    case,
                    construction=construction,
                    reason="resampling_threshold_audit_failed",
                )
            )
            continue

        case_audits.append(
            ExactResamplingThresholdCaseAudit(
                case_id=case_id,
                status="audited",
                backend_accepted=True,
                backend_name=construction.backend_name,
                backend_version=construction.backend_version,
                backend_kernel=construction.backend_kernel,
                point_count=len(case.points),
                top_simplex_count=construction.top_simplex_count,
                floating_selected_index=identity.floating_selected_index,
                exact_selected_index=identity.exact_selected_index,
                selected_index_matches=identity.selected_index_matches,
                candidate_local_position=local_position,
                floating_alpha_squared=identity.floating_alpha_squared,
                exact_alpha_squared=identity.exact_alpha_squared,
                alpha_absolute_difference=abs(
                    identity.floating_alpha_squared - identity.exact_alpha_squared
                ),
                floating_full_samples_sha256=_sample_sha256(floating_samples),
                exact_full_samples_sha256=_sample_sha256(exact_samples),
                full_surface_samples_match=True,
                full_surface_sample_count=len(floating_samples),
                resample_repeat_count=len(repeat_audits),
                repeats=tuple(repeat_audits),
                floating_reported_stability=floating_reported,
                exact_reported_stability=exact_reported,
                floating_recomputed_stability=floating_recomputed,
                exact_threshold_recomputed_stability=exact_recomputed,
                floating_recomputation_matches_report=floating_reproduced,
                exact_recomputation_matches_report=exact_reproduced,
                reported_stability_matches=_matches(
                    floating_reported,
                    exact_reported,
                ),
                recomputed_stability_matches=_matches(
                    floating_recomputed,
                    exact_recomputed,
                ),
                threshold_effect_reproduced=(floating_reproduced and exact_reproduced),
                rejection_reasons=(),
            )
        )

    return ExactResamplingThresholdPanelAudit(
        evaluation_split=construction_result.evaluation_split,
        backend_requested=construction_result.backend_requested,
        requested_case_count=len(materialized_cases),
        accepted_backend_case_count=construction_result.accepted_case_count,
        cases=tuple(case_audits),
    )
