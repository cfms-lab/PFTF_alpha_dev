"""Deployed exact / validated fail-closed fallback (G4).

Unlike the schema 16-25 shadows, this module actually routes the selection
filtration. Per case it attempts an exact, host-validated Euclidean Delaunay
construction and deploys it; on any refusal it fails closed to a floating Qhull
construction that is explicitly labeled non-exact. It certifies only the base
Delaunay connectivity B4/B5/P1/P2 score, never the anisotropic PFTF complex, and
keeps ``promotion_supported`` false. See docs/G4_FAIL_CLOSED_DEPLOYMENT_DESIGN.md.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .baselines import BaselineID, BenchmarkConfig, CaseBenchmark, run_case_benchmarks
from .exact_backend import (
    exact_construction_request,
    run_exact_construction_backend,
    validate_exact_construction_response,
)
from .exact_python_backend import (
    MAX_EXACT_POINT_COUNT,
    ExactPythonBackendError,
    exact_backend_response,
)
from .filtration import AlphaFiltration
from .geometry import as_point_array
from .synthetic import PanelSplit, SyntheticCase, make_minimal_panel

EXACT_PROVENANCE = "exact_validated_connectivity"
FALLBACK_PROVENANCE = "conservative_floating_fallback"


@dataclass(frozen=True)
class G4CaseRouting:
    """Provenance of the deployed filtration for one case."""

    case_id: str
    point_count: int
    is_exact_certified: bool
    exact_backend_requested: bool
    provenance: str
    failure_reason: str | None
    top_simplex_count: int

    def __post_init__(self) -> None:
        if self.is_exact_certified:
            if self.provenance != EXACT_PROVENANCE:
                raise ValueError("certified routing must use the exact provenance")
            if self.failure_reason is not None:
                raise ValueError("certified routing cannot carry a failure reason")
        else:
            if self.provenance != FALLBACK_PROVENANCE:
                raise ValueError("uncertified routing must use the fallback provenance")
            if not self.failure_reason:
                raise ValueError("uncertified routing must record a failure reason")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _trigger_reason(points: np.ndarray, *, max_point_count: int) -> str | None:
    """Observed-data-only eligibility check; never reads labels or references.

    Duplicate and non-finite points are already refused by ``as_point_array``
    (the shared input contract) before routing, so neither the exact backend nor
    the floating fallback can be built from them; that refusal is intentional.
    """

    if points.shape[1] != 3:
        return "points_must_be_three_dimensional"
    if points.shape[0] < 4:
        return "at_least_four_points_required"
    if points.shape[0] > max_point_count:
        return "point_count_exceeds_exact_backend_limit"
    return None


def _fallback(
    points: np.ndarray,
    case_id: str,
    *,
    exact_backend_requested: bool,
    reason: str,
) -> tuple[AlphaFiltration, G4CaseRouting]:
    filtration = AlphaFiltration.from_points(points)
    routing = G4CaseRouting(
        case_id=case_id,
        point_count=int(points.shape[0]),
        is_exact_certified=False,
        exact_backend_requested=exact_backend_requested,
        provenance=FALLBACK_PROVENANCE,
        failure_reason=reason,
        top_simplex_count=int(filtration.top_simplices.shape[0]),
    )
    return filtration, routing


def route_case_filtration(
    case_id: str,
    points,
    *,
    backend_command: Sequence[str] | None = None,
    timeout_seconds: float = 60.0,
    max_point_count: int = MAX_EXACT_POINT_COUNT,
) -> tuple[AlphaFiltration, G4CaseRouting]:
    """Deploy exact-validated connectivity, or fail closed to floating Qhull."""

    if not case_id:
        raise ValueError("case_id must be non-empty")
    point_array = as_point_array(points)

    trigger_reason = _trigger_reason(point_array, max_point_count=max_point_count)
    if trigger_reason is not None:
        return _fallback(
            point_array, case_id, exact_backend_requested=False, reason=trigger_reason
        )

    if backend_command is None:
        request, _ = exact_construction_request(case_id, point_array)
        try:
            response = exact_backend_response(request)
        except ExactPythonBackendError as error:
            return _fallback(
                point_array, case_id, exact_backend_requested=True, reason=error.reason
            )
        case_result = validate_exact_construction_response(
            case_id, point_array, response
        )
    else:
        case_result = run_exact_construction_backend(
            backend_command, case_id, point_array, timeout_seconds=timeout_seconds
        )

    if case_result.accepted and case_result.validated_top_simplices is not None:
        cells = np.asarray(case_result.validated_top_simplices, dtype=np.int64)
        filtration = AlphaFiltration.from_top_simplices(point_array, cells)
        routing = G4CaseRouting(
            case_id=case_id,
            point_count=int(point_array.shape[0]),
            is_exact_certified=True,
            exact_backend_requested=True,
            provenance=EXACT_PROVENANCE,
            failure_reason=None,
            top_simplex_count=int(filtration.top_simplices.shape[0]),
        )
        return filtration, routing

    reason = (
        case_result.rejection_reasons[0]
        if case_result.rejection_reasons
        else "exact_construction_not_accepted"
    )
    return _fallback(
        point_array, case_id, exact_backend_requested=True, reason=reason
    )


def _case_identifier(case: SyntheticCase) -> str:
    return f"{case.family.value}:{case.split.value}:{case.seed}"


def run_g4_routed_case(
    case: SyntheticCase,
    *,
    config: BenchmarkConfig,
    methods: Iterable[BaselineID | str],
    backend_command: Sequence[str] | None = None,
    timeout_seconds: float = 60.0,
) -> tuple[CaseBenchmark, G4CaseRouting]:
    """Route the filtration, then run the benchmark on the deployed filtration."""

    filtration, routing = route_case_filtration(
        _case_identifier(case),
        case.points,
        backend_command=backend_command,
        timeout_seconds=timeout_seconds,
    )
    report = run_case_benchmarks(
        case, config=config, methods=methods, filtration=filtration
    )
    return report, routing


@dataclass(frozen=True)
class G4DeploymentPanelResult:
    evaluation_split: str
    routings: tuple[G4CaseRouting, ...]

    @property
    def exact_certified_case_count(self) -> int:
        return sum(routing.is_exact_certified for routing in self.routings)

    @property
    def fail_closed_case_count(self) -> int:
        return sum(not routing.is_exact_certified for routing in self.routings)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": "pftf_alpha_g4_fail_closed/v1",
            "role": "deployed_exact_or_conservative_fallback_selection_path",
            "evaluation_split": self.evaluation_split,
            "exact_construction_scope": (
                "euclidean_delaunay_connectivity_only_not_anisotropic"
            ),
            "certifies": "base_delaunay_connectivity_used_by_B4_B5_P1_P2",
            "does_not_certify": (
                "pftf_anisotropic_alpha_complex_selection_or_surface_false_safe"
            ),
            "changes_benchmark_selection": True,
            "case_count": len(self.routings),
            "exact_certified_case_count": self.exact_certified_case_count,
            "fail_closed_case_count": self.fail_closed_case_count,
            "no_uncertified_result_labeled_exact": all(
                routing.is_exact_certified == (routing.provenance == EXACT_PROVENANCE)
                and (routing.is_exact_certified or routing.failure_reason is not None)
                for routing in self.routings
            ),
            "promotion_supported": False,
            "claim_boundary": (
                "Deployed exact-validated Euclidean Delaunay connectivity where "
                "available, else an explicitly conservative floating fallback. This "
                "is not an exact anisotropic PFTF complex, CGAL parity, a general "
                "false-safe surface certificate, or higher-fidelity held-out "
                "evidence, and does not by itself justify promotion."
            ),
            "routings": [routing.to_dict() for routing in self.routings],
        }


def evaluate_g4_deployment_panel(
    cases: Iterable[SyntheticCase],
    *,
    config: BenchmarkConfig,
    methods: Iterable[BaselineID | str] = (
        BaselineID.B4_DENSITY_SCALED,
        BaselineID.P2_CONFIDENCE_FALLBACK,
    ),
    backend_command: Sequence[str] | None = None,
    timeout_seconds: float = 60.0,
) -> G4DeploymentPanelResult:
    """Run the deployed G4 routing across a panel and record provenance."""

    materialized = tuple(cases)
    if not materialized:
        raise ValueError("cases must be non-empty")
    selected_methods = tuple(methods)
    routings: list[G4CaseRouting] = []
    split = materialized[0].split.value
    for case in materialized:
        _, routing = run_g4_routed_case(
            case,
            config=config,
            methods=selected_methods,
            backend_command=backend_command,
            timeout_seconds=timeout_seconds,
        )
        routings.append(routing)
    return G4DeploymentPanelResult(evaluation_split=split, routings=tuple(routings))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deployed G4 exact/validated fail-closed selection path."
    )
    parser.add_argument("--point-count", type=int, default=96)
    parser.add_argument("--reference-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20_260_724)
    parser.add_argument(
        "--split",
        type=str,
        default=PanelSplit.HELD_OUT.value,
        choices=[split.value for split in PanelSplit],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/g4_fail_closed_deployment.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = BenchmarkConfig(seed=args.seed)
    cases = make_minimal_panel(
        split=PanelSplit(args.split),
        point_count=args.point_count,
        reference_count=args.reference_count,
        seed=args.seed,
    )
    result = evaluate_g4_deployment_panel(cases, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[g4] exact_certified={result.exact_certified_case_count} "
        f"fail_closed={result.fail_closed_case_count}",
        flush=True,
    )
    print(f"Wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
