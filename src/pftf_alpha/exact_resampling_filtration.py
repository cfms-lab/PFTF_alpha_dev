"""Evaluation-only exact connectivity and filtration audit for B3 resamples."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from .baselines import (
    BenchmarkConfig,
    _resampled_point_sets,
    _stability_loss,
    _unlabeled_geometry_loss,
)
from .exact_backend import (
    ExactConstructionCaseResult,
    ExactConstructionPanelResult,
    run_exact_construction_backend,
)
from .exact_filtration import audit_exact_filtration_case, exact_rounded_filtration
from .exact_index_audit import _boundary_sha256, _complex_sha256
from .exact_resampling_audit import (
    ExactResamplingThresholdCaseAudit,
    ExactResamplingThresholdPanelAudit,
)
from .exact_shadow import FLOAT_ABS_TOLERANCE, FLOAT_REL_TOLERANCE
from .filtration import AlphaFiltration
from .surface import alpha_surface
from .synthetic import SyntheticCase


@dataclass(frozen=True)
class ExactResamplingFiltrationRepeatAudit:
    """One deterministic resample reconstructed and filtered exactly."""

    repeat_index: int
    status: str
    resampled_point_count: int
    resampled_points_sha256: str
    backend_accepted: bool
    backend_name: str | None
    backend_version: str | None
    backend_kernel: str | None
    floating_top_simplex_count: int
    exact_top_simplex_count: int | None
    connectivity_matches: bool | None
    exact_filtration_audited: bool
    exact_filtration_sha256: str | None
    float_value_difference_count: int | None
    max_ulp_difference: int | None
    floating_threshold_complex_sha256: str | None
    exact_threshold_complex_sha256: str | None
    selected_complex_matches: bool | None
    floating_threshold_boundary_sha256: str | None
    exact_threshold_boundary_sha256: str | None
    selected_boundary_matches: bool | None
    floating_threshold_stability_loss: float | None
    exact_threshold_stability_loss: float | None
    stability_loss_matches: bool | None
    rejection_reasons: tuple[str, ...]

    @property
    def audited(self) -> bool:
        return self.status == "audited"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        return payload


@dataclass(frozen=True)
class ExactResamplingFiltrationCaseAudit:
    """Case-level exact-resampling comparison at the schema-23 B3 threshold."""

    case_id: str
    status: str
    backend_requested: bool
    point_count: int
    exact_alpha_squared: float | None
    candidate_local_position: int | None
    full_surface_samples_sha256: str | None
    full_surface_samples_match: bool | None
    full_surface_sample_count: int
    requested_repeat_count: int
    repeats: tuple[ExactResamplingFiltrationRepeatAudit, ...]
    threshold_audit_reported_stability: float | None
    threshold_audit_recomputed_stability: float | None
    floating_resampling_recomputed_stability: float | None
    exact_resampling_recomputed_stability: float | None
    floating_recomputation_matches_threshold_audit: bool | None
    exact_resampling_matches_threshold_audit: bool | None
    exact_resampling_stability_absolute_difference: float | None
    rejection_reasons: tuple[str, ...]

    @property
    def audited(self) -> bool:
        return self.status == "audited"

    @property
    def audited_repeat_count(self) -> int:
        return sum(repeat.audited for repeat in self.repeats)

    @property
    def connectivity_difference_repeat_count(self) -> int:
        return sum(
            repeat.audited and repeat.connectivity_matches is False
            for repeat in self.repeats
        )

    @property
    def selected_complex_difference_repeat_count(self) -> int:
        return sum(
            repeat.audited and repeat.selected_complex_matches is False
            for repeat in self.repeats
        )

    @property
    def selected_boundary_difference_repeat_count(self) -> int:
        return sum(
            repeat.audited and repeat.selected_boundary_matches is False
            for repeat in self.repeats
        )

    @property
    def stability_difference_repeat_count(self) -> int:
        return sum(
            repeat.audited and repeat.stability_loss_matches is False
            for repeat in self.repeats
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["repeats"] = [repeat.to_dict() for repeat in self.repeats]
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["audited_repeat_count"] = self.audited_repeat_count
        payload["connectivity_difference_repeat_count"] = (
            self.connectivity_difference_repeat_count
        )
        payload["selected_complex_difference_repeat_count"] = (
            self.selected_complex_difference_repeat_count
        )
        payload["selected_boundary_difference_repeat_count"] = (
            self.selected_boundary_difference_repeat_count
        )
        payload["stability_difference_repeat_count"] = (
            self.stability_difference_repeat_count
        )
        return payload


@dataclass(frozen=True)
class ExactResamplingFiltrationPanelAudit:
    """Panel result that keeps exact resampling outside primary selection."""

    evaluation_split: str
    backend_requested: bool
    requested_case_count: int
    requested_repeat_count: int
    cases: tuple[ExactResamplingFiltrationCaseAudit, ...]
    _exact_filtrations_by_case: tuple[
        tuple[str, tuple[AlphaFiltration, ...]], ...
    ] = field(default=(), repr=False, compare=False)

    def exact_resampled_filtrations(
        self,
        case_id: str,
    ) -> tuple[AlphaFiltration, ...] | None:
        """Return non-serialized exact resamples for a downstream shadow."""

        return next(
            (
                filtrations
                for stored_case_id, filtrations in self._exact_filtrations_by_case
                if stored_case_id == case_id
            ),
            None,
        )

    @property
    def audited_case_count(self) -> int:
        return sum(case.audited for case in self.cases)

    @property
    def audited_repeat_count(self) -> int:
        return sum(case.audited_repeat_count for case in self.cases)

    @property
    def rejected_repeat_count(self) -> int:
        return self.requested_repeat_count - self.audited_repeat_count

    @property
    def connectivity_difference_case_count(self) -> int:
        return sum(
            case.audited and case.connectivity_difference_repeat_count > 0
            for case in self.cases
        )

    @property
    def connectivity_difference_repeat_count(self) -> int:
        return sum(case.connectivity_difference_repeat_count for case in self.cases)

    @property
    def selected_complex_difference_case_count(self) -> int:
        return sum(
            case.audited and case.selected_complex_difference_repeat_count > 0
            for case in self.cases
        )

    @property
    def selected_complex_difference_repeat_count(self) -> int:
        return sum(
            case.selected_complex_difference_repeat_count for case in self.cases
        )

    @property
    def selected_boundary_difference_case_count(self) -> int:
        return sum(
            case.audited and case.selected_boundary_difference_repeat_count > 0
            for case in self.cases
        )

    @property
    def selected_boundary_difference_repeat_count(self) -> int:
        return sum(
            case.selected_boundary_difference_repeat_count for case in self.cases
        )

    @property
    def stability_difference_case_count(self) -> int:
        return sum(
            case.audited
            and case.exact_resampling_matches_threshold_audit is False
            for case in self.cases
        )

    @property
    def all_resamples_audited(self) -> bool:
        return (
            self.requested_repeat_count > 0
            and self.audited_repeat_count == self.requested_repeat_count
        )

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.backend_requested:
            reasons.append("no_exact_resampling_backend")
        elif not self.all_resamples_audited:
            reasons.append("one_or_more_exact_resampling_repeats_rejected")
        reasons.append("exact_resampling_filtration_audit_not_deployed")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": (
                "evaluation_only_exact_resampling_connectivity_and_filtration_audit"
            ),
            "evaluation_split": self.evaluation_split,
            "resampled_connectivity": "host_validated_exact_backend",
            "resampled_filtration_values": "correctly_rounded_exact_rationals",
            "threshold_source": "schema_23_exact_rounded_b3_selected_alpha",
            "backend_requested": self.backend_requested,
            "requested_case_count": self.requested_case_count,
            "audited_case_count": self.audited_case_count,
            "requested_repeat_count": self.requested_repeat_count,
            "audited_repeat_count": self.audited_repeat_count,
            "rejected_repeat_count": self.rejected_repeat_count,
            "connectivity_difference_case_count": (
                self.connectivity_difference_case_count
            ),
            "connectivity_difference_repeat_count": (
                self.connectivity_difference_repeat_count
            ),
            "selected_complex_difference_case_count": (
                self.selected_complex_difference_case_count
            ),
            "selected_complex_difference_repeat_count": (
                self.selected_complex_difference_repeat_count
            ),
            "selected_boundary_difference_case_count": (
                self.selected_boundary_difference_case_count
            ),
            "selected_boundary_difference_repeat_count": (
                self.selected_boundary_difference_repeat_count
            ),
            "stability_difference_case_count": self.stability_difference_case_count,
            "all_resamples_audited": self.all_resamples_audited,
            "comparison_float_relative_tolerance": FLOAT_REL_TOLERANCE,
            "comparison_float_absolute_tolerance": FLOAT_ABS_TOLERANCE,
            "primary_benchmark_results_changed": False,
            "selection_effect": "none",
            "promotion_supported": False,
            "blocking_reasons": list(self.blocking_reasons),
            "cases": [case.to_dict() for case in self.cases],
        }


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
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


def _canonical_cells(filtration: AlphaFiltration) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            tuple(sorted(int(vertex) for vertex in cell))
            for cell in filtration.top_simplices
        )
    )


def _rejected_repeat(
    repeat_index: int,
    points: np.ndarray,
    floating: AlphaFiltration,
    construction: ExactConstructionCaseResult | None,
    reasons: tuple[str, ...],
) -> ExactResamplingFiltrationRepeatAudit:
    return ExactResamplingFiltrationRepeatAudit(
        repeat_index=repeat_index,
        status="rejected",
        resampled_point_count=len(points),
        resampled_points_sha256=_array_sha256(points),
        backend_accepted=bool(construction and construction.accepted),
        backend_name=None if construction is None else construction.backend_name,
        backend_version=None if construction is None else construction.backend_version,
        backend_kernel=None if construction is None else construction.backend_kernel,
        floating_top_simplex_count=len(floating.top_simplices),
        exact_top_simplex_count=(
            None if construction is None else construction.top_simplex_count
        ),
        connectivity_matches=None,
        exact_filtration_audited=False,
        exact_filtration_sha256=None,
        float_value_difference_count=None,
        max_ulp_difference=None,
        floating_threshold_complex_sha256=None,
        exact_threshold_complex_sha256=None,
        selected_complex_matches=None,
        floating_threshold_boundary_sha256=None,
        exact_threshold_boundary_sha256=None,
        selected_boundary_matches=None,
        floating_threshold_stability_loss=None,
        exact_threshold_stability_loss=None,
        stability_loss_matches=None,
        rejection_reasons=reasons,
    )


def _not_audited_case(
    case: SyntheticCase,
    *,
    backend_requested: bool,
    config: BenchmarkConfig,
    threshold: ExactResamplingThresholdCaseAudit | None,
    repeats: tuple[ExactResamplingFiltrationRepeatAudit, ...] = (),
    full_surface_samples_sha256: str | None = None,
    full_surface_samples_match: bool | None = None,
    full_surface_sample_count: int = 0,
    reasons: tuple[str, ...],
) -> ExactResamplingFiltrationCaseAudit:
    return ExactResamplingFiltrationCaseAudit(
        case_id=case.family.value,
        status="not_audited",
        backend_requested=backend_requested,
        point_count=len(case.points),
        exact_alpha_squared=(
            None if threshold is None else threshold.exact_alpha_squared
        ),
        candidate_local_position=(
            None if threshold is None else threshold.candidate_local_position
        ),
        full_surface_samples_sha256=full_surface_samples_sha256,
        full_surface_samples_match=full_surface_samples_match,
        full_surface_sample_count=full_surface_sample_count,
        requested_repeat_count=config.resample_repeats,
        repeats=repeats,
        threshold_audit_reported_stability=(
            None if threshold is None else threshold.exact_reported_stability
        ),
        threshold_audit_recomputed_stability=(
            None
            if threshold is None
            else threshold.exact_threshold_recomputed_stability
        ),
        floating_resampling_recomputed_stability=None,
        exact_resampling_recomputed_stability=None,
        floating_recomputation_matches_threshold_audit=None,
        exact_resampling_matches_threshold_audit=None,
        exact_resampling_stability_absolute_difference=None,
        rejection_reasons=reasons,
    )


def evaluate_exact_resampling_filtration_audit(
    cases: Iterable[SyntheticCase],
    *,
    construction_result: ExactConstructionPanelResult,
    threshold_audit: ExactResamplingThresholdPanelAudit,
    config: BenchmarkConfig,
    backend_command: Sequence[str] | None,
    backend_timeout_seconds: float = 60.0,
) -> ExactResamplingFiltrationPanelAudit:
    """Reconstruct every B3 resample with exact connectivity and values."""

    materialized_cases = tuple(cases)
    if not materialized_cases:
        raise ValueError("cases must be non-empty")
    if construction_result.evaluation_split != threshold_audit.evaluation_split:
        raise ValueError("exact resampling prerequisites must use the same split")
    if not math.isfinite(backend_timeout_seconds) or backend_timeout_seconds <= 0.0:
        raise ValueError("backend_timeout_seconds must be finite and positive")

    construction_by_id = {
        construction.case_id: construction
        for construction in construction_result.cases
    }
    threshold_by_id = {case.case_id: case for case in threshold_audit.cases}
    if len(construction_by_id) != len(construction_result.cases):
        raise ValueError("construction result contains duplicate case identifiers")
    if len(threshold_by_id) != len(threshold_audit.cases):
        raise ValueError("threshold audit contains duplicate case identifiers")

    backend_requested = backend_command is not None
    case_audits = []
    runtime_filtrations = []
    for case in materialized_cases:
        case_id = case.family.value
        construction = construction_by_id.get(case_id)
        threshold = threshold_by_id.get(case_id)
        if backend_command is None:
            case_audits.append(
                _not_audited_case(
                    case,
                    backend_requested=False,
                    config=config,
                    threshold=threshold,
                    reasons=("no_exact_resampling_backend",),
                )
            )
            continue
        if (
            construction is None
            or not construction.accepted
            or construction.validated_top_simplices is None
        ):
            reasons = (
                ("full_exact_construction_missing",)
                if construction is None
                else construction.rejection_reasons
            )
            case_audits.append(
                _not_audited_case(
                    case,
                    backend_requested=True,
                    config=config,
                    threshold=threshold,
                    reasons=reasons,
                )
            )
            continue
        if (
            threshold is None
            or not threshold.audited
            or threshold.exact_alpha_squared is None
            or threshold.candidate_local_position is None
            or threshold.exact_reported_stability is None
            or threshold.exact_threshold_recomputed_stability is None
            or threshold.exact_full_samples_sha256 is None
            or threshold.threshold_effect_reproduced is not True
        ):
            case_audits.append(
                _not_audited_case(
                    case,
                    backend_requested=True,
                    config=config,
                    threshold=threshold,
                    reasons=("schema_23_threshold_audit_prerequisite_failed",),
                )
            )
            continue

        try:
            full_exact = exact_rounded_filtration(
                case.points,
                construction.validated_top_simplices,
            ).filtration
            full_mesh = alpha_surface(full_exact, threshold.exact_alpha_squared)
            full_seed = (
                config.seed
                + case.seed
                + 40_000
                + threshold.candidate_local_position
            )
            _, full_samples = _unlabeled_geometry_loss(
                full_mesh,
                case.points,
                case,
                config,
                seed=full_seed,
            )
            full_samples_sha256 = _array_sha256(full_samples)
            if full_samples_sha256 != threshold.exact_full_samples_sha256:
                raise ValueError("schema-23 full surface samples differ")
        except (ArithmeticError, np.linalg.LinAlgError, ValueError):
            case_audits.append(
                _not_audited_case(
                    case,
                    backend_requested=True,
                    config=config,
                    threshold=threshold,
                    reasons=("exact_full_surface_reconstruction_failed",),
                )
            )
            continue

        repeat_audits = []
        floating_losses = []
        exact_losses = []
        exact_filtrations = []
        point_sets = _resampled_point_sets(case, config)
        stability_seed = (
            config.seed + case.seed + 50_000 + threshold.candidate_local_position
        )
        for repeat_index, points in enumerate(point_sets):
            floating = AlphaFiltration.from_points(points)
            resample_id = f"{case_id}::resample::{repeat_index}"
            resampled_construction = run_exact_construction_backend(
                backend_command,
                resample_id,
                points,
                timeout_seconds=backend_timeout_seconds,
            )
            if (
                not resampled_construction.accepted
                or resampled_construction.validated_top_simplices is None
            ):
                repeat_audits.append(
                    _rejected_repeat(
                        repeat_index,
                        points,
                        floating,
                        resampled_construction,
                        resampled_construction.rejection_reasons,
                    )
                )
                continue
            filtration_audit = audit_exact_filtration_case(
                resample_id,
                points,
                resampled_construction,
            )
            if not filtration_audit.audited:
                repeat_audits.append(
                    _rejected_repeat(
                        repeat_index,
                        points,
                        floating,
                        resampled_construction,
                        filtration_audit.rejection_reasons,
                    )
                )
                continue

            try:
                exact = exact_rounded_filtration(
                    points,
                    resampled_construction.validated_top_simplices,
                ).filtration
                floating_mesh = alpha_surface(
                    floating,
                    threshold.exact_alpha_squared,
                )
                exact_mesh = alpha_surface(
                    exact,
                    threshold.exact_alpha_squared,
                )
                repeat_seed = stability_seed + repeat_index
                floating_loss = _stability_loss(
                    full_samples,
                    (floating_mesh,),
                    case,
                    config,
                    seed=repeat_seed,
                )
                exact_loss = _stability_loss(
                    full_samples,
                    (exact_mesh,),
                    case,
                    config,
                    seed=repeat_seed,
                )
                floating_complex_sha256 = _complex_sha256(
                    floating,
                    threshold.exact_alpha_squared,
                )
                exact_complex_sha256 = _complex_sha256(
                    exact,
                    threshold.exact_alpha_squared,
                )
                floating_boundary_sha256 = _boundary_sha256(
                    floating,
                    threshold.exact_alpha_squared,
                )
                exact_boundary_sha256 = _boundary_sha256(
                    exact,
                    threshold.exact_alpha_squared,
                )
            except (ArithmeticError, np.linalg.LinAlgError, ValueError):
                repeat_audits.append(
                    _rejected_repeat(
                        repeat_index,
                        points,
                        floating,
                        resampled_construction,
                        ("exact_resampling_evaluation_failed",),
                    )
                )
                continue

            floating_losses.append(floating_loss)
            exact_losses.append(exact_loss)
            exact_filtrations.append(exact)
            repeat_audits.append(
                ExactResamplingFiltrationRepeatAudit(
                    repeat_index=repeat_index,
                    status="audited",
                    resampled_point_count=len(points),
                    resampled_points_sha256=_array_sha256(points),
                    backend_accepted=True,
                    backend_name=resampled_construction.backend_name,
                    backend_version=resampled_construction.backend_version,
                    backend_kernel=resampled_construction.backend_kernel,
                    floating_top_simplex_count=len(floating.top_simplices),
                    exact_top_simplex_count=len(exact.top_simplices),
                    connectivity_matches=(
                        _canonical_cells(floating) == _canonical_cells(exact)
                    ),
                    exact_filtration_audited=True,
                    exact_filtration_sha256=(
                        filtration_audit.exact_filtration_sha256
                    ),
                    float_value_difference_count=(
                        filtration_audit.float_value_difference_count
                    ),
                    max_ulp_difference=filtration_audit.max_ulp_difference,
                    floating_threshold_complex_sha256=floating_complex_sha256,
                    exact_threshold_complex_sha256=exact_complex_sha256,
                    selected_complex_matches=(
                        floating_complex_sha256 == exact_complex_sha256
                    ),
                    floating_threshold_boundary_sha256=floating_boundary_sha256,
                    exact_threshold_boundary_sha256=exact_boundary_sha256,
                    selected_boundary_matches=(
                        floating_boundary_sha256 == exact_boundary_sha256
                    ),
                    floating_threshold_stability_loss=floating_loss,
                    exact_threshold_stability_loss=exact_loss,
                    stability_loss_matches=_matches(floating_loss, exact_loss),
                    rejection_reasons=(),
                )
            )

        repeat_tuple = tuple(repeat_audits)
        if len(floating_losses) != config.resample_repeats:
            case_audits.append(
                _not_audited_case(
                    case,
                    backend_requested=True,
                    config=config,
                    threshold=threshold,
                    repeats=repeat_tuple,
                    full_surface_samples_sha256=full_samples_sha256,
                    full_surface_samples_match=True,
                    full_surface_sample_count=len(full_samples),
                    reasons=("one_or_more_exact_resampling_repeats_rejected",),
                )
            )
            continue

        floating_stability = float(np.mean(floating_losses))
        exact_stability = float(np.mean(exact_losses))
        runtime_filtrations.append((case_id, tuple(exact_filtrations)))
        case_audits.append(
            ExactResamplingFiltrationCaseAudit(
                case_id=case_id,
                status="audited",
                backend_requested=True,
                point_count=len(case.points),
                exact_alpha_squared=threshold.exact_alpha_squared,
                candidate_local_position=threshold.candidate_local_position,
                full_surface_samples_sha256=full_samples_sha256,
                full_surface_samples_match=True,
                full_surface_sample_count=len(full_samples),
                requested_repeat_count=config.resample_repeats,
                repeats=repeat_tuple,
                threshold_audit_reported_stability=(
                    threshold.exact_reported_stability
                ),
                threshold_audit_recomputed_stability=(
                    threshold.exact_threshold_recomputed_stability
                ),
                floating_resampling_recomputed_stability=floating_stability,
                exact_resampling_recomputed_stability=exact_stability,
                floating_recomputation_matches_threshold_audit=_matches(
                    floating_stability,
                    threshold.exact_threshold_recomputed_stability,
                ),
                exact_resampling_matches_threshold_audit=_matches(
                    exact_stability,
                    threshold.exact_threshold_recomputed_stability,
                ),
                exact_resampling_stability_absolute_difference=abs(
                    exact_stability
                    - threshold.exact_threshold_recomputed_stability
                ),
                rejection_reasons=(),
            )
        )

    return ExactResamplingFiltrationPanelAudit(
        evaluation_split=construction_result.evaluation_split,
        backend_requested=backend_requested,
        requested_case_count=len(materialized_cases),
        requested_repeat_count=len(materialized_cases) * config.resample_repeats,
        cases=tuple(case_audits),
        _exact_filtrations_by_case=tuple(runtime_filtrations),
    )
