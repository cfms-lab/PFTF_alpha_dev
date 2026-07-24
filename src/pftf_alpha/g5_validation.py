"""Frozen synthetic held-out robustness preflight for the G5 promotion gate."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .baselines import (
    BaselineID,
    BaselineResult,
    BenchmarkConfig,
    CaseBenchmark,
    run_case_benchmarks,
)
from .calibration import (
    AdaptiveCalibrationResult,
    P2ConfidenceCalibrationResult,
    calibrate_adaptive_multiplier,
    calibrate_p2_confidence_threshold,
)
from .synthetic import (
    PanelSplit,
    SyntheticCase,
    SyntheticFamily,
    make_minimal_panel,
    make_synthetic_case,
)

ADAPTIVE_METHODS = (
    BaselineID.B4_DENSITY_SCALED,
    BaselineID.B5_PCA_ANISOTROPIC,
    BaselineID.P1_PFTF_LOCAL_SPD,
    BaselineID.P2_CONFIDENCE_FALLBACK,
)
PFTF_METHODS = (
    BaselineID.P1_PFTF_LOCAL_SPD,
    BaselineID.P2_CONFIDENCE_FALLBACK,
)


@dataclass(frozen=True)
class G5Profile:
    """One declared held-out density/noise/geometry condition."""

    name: str
    point_count: int
    reference_count: int
    geometry_scale: float = 1.0
    noise_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must be non-empty")
        if self.point_count < 16:
            raise ValueError("profile point_count must be at least 16")
        if self.reference_count < self.point_count:
            raise ValueError("profile reference_count must be at least point_count")
        for name, value in {
            "geometry_scale": self.geometry_scale,
            "noise_scale": self.noise_scale,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"profile {name} must be finite and positive")

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenAdaptiveConfig:
    """Calibration-only result used unchanged for every held-out profile."""

    config: BenchmarkConfig
    confidence: P2ConfidenceCalibrationResult
    multipliers: tuple[AdaptiveCalibrationResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_split": PanelSplit.CALIBRATION.value,
            "confidence": self.confidence.to_dict(),
            "multipliers": [result.to_dict() for result in self.multipliers],
            "config": asdict(self.config),
        }


@dataclass(frozen=True)
class G5CaseBenchmark:
    profile: str
    repeat_index: int
    report: CaseBenchmark

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "repeat_index": self.repeat_index,
            **self.report.to_dict(),
        }


@dataclass(frozen=True)
class G5MethodSummary:
    profile: str
    method: BaselineID
    case_count: int
    mean_fscore: float
    std_fscore: float
    mean_normalized_chamfer_squared: float
    std_normalized_chamfer_squared: float
    mean_normalized_hausdorff: float
    std_normalized_hausdorff: float
    component_error_sum: int
    betti_error_sum: int
    topology_failure_case_count: int
    labeled_case_count: int
    labeled_false_bridge_edges_sum: int
    labeled_false_bridge_faces_sum: int
    runtime_seconds_sum: float
    mean_fallback_fraction: float | None
    mean_selected_fallback_fraction: float | None
    fallback_guard_violation_count: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["method"] = self.method.value
        return payload


@dataclass(frozen=True)
class G5CandidateComparison:
    """Matched candidate margin against the per-endpoint B4/B5 envelope."""

    profile: str
    candidate_method: BaselineID
    case_count: int
    mean_fscore_margin: float
    mean_geometry_loss_margin: float
    topology_burden_excess_sum: int
    labeled_false_bridge_edges_excess_sum: int
    labeled_false_bridge_faces_excess_sum: int
    fallback_guard_violation_count: int
    endpoint_noninferiority_supported: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_method"] = self.candidate_method.value
        return payload


@dataclass(frozen=True)
class G5ProfileShift:
    profile: str
    method: BaselineID
    case_count: int
    mean_fscore_change_from_base: float
    mean_geometry_loss_change_from_base: float
    topology_burden_change_sum: int
    labeled_false_bridge_edges_change_sum: int
    labeled_false_bridge_faces_change_sum: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["method"] = self.method.value
        return payload


@dataclass(frozen=True)
class G5PreflightResult:
    profiles: tuple[G5Profile, ...]
    repeat_count: int
    seed: int
    calibration: FrozenAdaptiveConfig
    cases: tuple[G5CaseBenchmark, ...]
    summaries: tuple[G5MethodSummary, ...]
    comparisons: tuple[G5CandidateComparison, ...]
    profile_shifts: tuple[G5ProfileShift, ...]
    endpoint_preflight_supported: bool
    promotion_supported: bool
    promotion_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": "pftf_alpha_g5_preflight/v1",
            "evaluation_role": "synthetic_frozen_held_out_preflight",
            "repeat_count": self.repeat_count,
            "seed": self.seed,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "selection_contract": {
                "calibration_split": PanelSplit.CALIBRATION.value,
                "evaluation_split": PanelSplit.HELD_OUT.value,
                "frozen_methods": [method.value for method in ADAPTIVE_METHODS],
                "held_out_reference_role": "evaluation_only",
                "held_out_tuning": "prohibited",
                "comparison_reference": "casewise_per_endpoint_B4_B5_envelope",
            },
            "profile_contract": {
                "base": "held_out_geometry_density_and_noise",
                "sparse": "two_thirds_observed_density_rounded_with_floor_16",
                "noisy": "held_out_observation_noise_times_2",
                "hard_geometry": "family_geometry_parameter_times_0.75",
                "paired_seed_policy": "same_seed_across_profiles_per_repeat_family",
            },
            "calibration": self.calibration.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "profile_shifts": [shift.to_dict() for shift in self.profile_shifts],
            "endpoint_preflight_supported": self.endpoint_preflight_supported,
            "promotion_supported": self.promotion_supported,
            "promotion_blockers": list(self.promotion_blockers),
            "claim_boundary": (
                "This deterministic synthetic robustness panel is a G5 preflight, "
                "not paper-level promotion evidence. It does not deploy or validate "
                "an exact fail-closed G4 fallback, replace real higher-fidelity "
                "held-out data, or provide confirmatory uncertainty estimates."
            ),
        }


def default_g5_profiles(
    *, point_count: int, reference_count: int
) -> tuple[G5Profile, ...]:
    """Build the four frozen profiles without inspecting held-out results."""

    sparse_count = max(16, int(round(point_count * (2.0 / 3.0))))
    return (
        G5Profile("base", point_count, reference_count),
        G5Profile("sparse", sparse_count, reference_count),
        G5Profile("noisy", point_count, reference_count, noise_scale=2.0),
        G5Profile("hard_geometry", point_count, reference_count, geometry_scale=0.75),
    )


def freeze_adaptive_config(
    calibration_cases: Iterable[SyntheticCase],
    *,
    config: BenchmarkConfig,
    candidate_budget: int,
    target_fallback_fraction: float,
) -> FrozenAdaptiveConfig:
    """Freeze P2 confidence and all adaptive multipliers on calibration only."""

    cases = tuple(calibration_cases)
    frozen_config = replace(
        config,
        b4_scale_multiplier=None,
        b5_scale_multiplier=None,
        p1_scale_multiplier=None,
        p2_scale_multiplier=None,
    )
    confidence = calibrate_p2_confidence_threshold(
        cases,
        config=frozen_config,
        target_fallback_fraction=target_fallback_fraction,
    )
    frozen_config = replace(
        frozen_config, p2_confidence_threshold=confidence.threshold
    )
    calibrations: list[AdaptiveCalibrationResult] = []
    field_by_method = {
        BaselineID.B4_DENSITY_SCALED: "b4_scale_multiplier",
        BaselineID.B5_PCA_ANISOTROPIC: "b5_scale_multiplier",
        BaselineID.P1_PFTF_LOCAL_SPD: "p1_scale_multiplier",
        BaselineID.P2_CONFIDENCE_FALLBACK: "p2_scale_multiplier",
    }
    for method in ADAPTIVE_METHODS:
        result = calibrate_adaptive_multiplier(
            cases,
            method,
            config=frozen_config,
            candidate_budget=candidate_budget,
        )
        calibrations.append(result)
        frozen_config = replace(
            frozen_config, **{field_by_method[method]: result.multiplier}
        )
    return FrozenAdaptiveConfig(frozen_config, confidence, tuple(calibrations))


def _profile_case(
    family: SyntheticFamily, profile: G5Profile, *, seed: int
) -> SyntheticCase:
    base = make_synthetic_case(
        family,
        split=PanelSplit.HELD_OUT,
        point_count=profile.point_count,
        reference_count=profile.reference_count,
        seed=seed,
    )
    if profile.geometry_scale == 1.0 and profile.noise_scale == 1.0:
        return base
    geometry_key = next(name for name in base.variation if name != "noise")
    return make_synthetic_case(
        family,
        split=PanelSplit.HELD_OUT,
        point_count=profile.point_count,
        reference_count=profile.reference_count,
        seed=seed,
        variation_overrides={
            geometry_key: base.variation[geometry_key] * profile.geometry_scale,
            "noise": base.variation["noise"] * profile.noise_scale,
        },
    )


def _result_map(report: CaseBenchmark) -> dict[BaselineID, BaselineResult]:
    return {result.method: result for result in report.results}


def _geometry_loss(result: BaselineResult) -> float:
    return (
        result.endpoints.normalized_chamfer_squared
        + result.endpoints.normalized_hausdorff
    )


def _topology_burden(result: BaselineResult) -> int:
    betti_error = result.endpoints.betti_error
    return result.endpoints.component_error + (
        0 if betti_error is None else betti_error
    )


def _optional_sum(values: Iterable[int | None]) -> int:
    return sum(0 if value is None else value for value in values)


def _diagnostic(result: BaselineResult, name: str) -> float | None:
    if result.method_diagnostics is None:
        return None
    return result.method_diagnostics.get(name)


def _method_summary(
    cases: tuple[G5CaseBenchmark, ...],
    profile: str,
    method: BaselineID,
) -> G5MethodSummary:
    results = tuple(
        _result_map(case.report)[method] for case in cases if case.profile == profile
    )
    fscores = np.asarray([result.endpoints.fscore for result in results])
    chamfers = np.asarray(
        [result.endpoints.normalized_chamfer_squared for result in results]
    )
    hausdorffs = np.asarray(
        [result.endpoints.normalized_hausdorff for result in results]
    )
    fallback = tuple(
        value
        for result in results
        if (value := _diagnostic(result, "fallback_fraction")) is not None
    )
    selected_fallback = tuple(
        value
        for result in results
        if (value := _diagnostic(result, "selected_fallback_fraction")) is not None
    )
    labeled_results = tuple(
        result
        for result in results
        if result.endpoints.labeled_false_bridge_edges is not None
    )
    return G5MethodSummary(
        profile=profile,
        method=method,
        case_count=len(results),
        mean_fscore=float(np.mean(fscores)),
        std_fscore=float(np.std(fscores)),
        mean_normalized_chamfer_squared=float(np.mean(chamfers)),
        std_normalized_chamfer_squared=float(np.std(chamfers)),
        mean_normalized_hausdorff=float(np.mean(hausdorffs)),
        std_normalized_hausdorff=float(np.std(hausdorffs)),
        component_error_sum=sum(result.endpoints.component_error for result in results),
        betti_error_sum=_optional_sum(
            result.endpoints.betti_error for result in results
        ),
        topology_failure_case_count=sum(
            _topology_burden(result) > 0 for result in results
        ),
        labeled_case_count=len(labeled_results),
        labeled_false_bridge_edges_sum=_optional_sum(
            result.endpoints.labeled_false_bridge_edges for result in labeled_results
        ),
        labeled_false_bridge_faces_sum=_optional_sum(
            result.endpoints.labeled_false_bridge_faces for result in labeled_results
        ),
        runtime_seconds_sum=sum(result.runtime_seconds for result in results),
        mean_fallback_fraction=(
            None if not fallback else float(np.mean(np.asarray(fallback)))
        ),
        mean_selected_fallback_fraction=(
            None
            if not selected_fallback
            else float(np.mean(np.asarray(selected_fallback)))
        ),
        fallback_guard_violation_count=int(
            sum(
                _diagnostic(result, "fallback_guard_violation_count") or 0.0
                for result in results
            )
        ),
    )


def _candidate_comparison(
    cases: tuple[G5CaseBenchmark, ...],
    profile: str,
    candidate: BaselineID,
) -> G5CandidateComparison:
    selected_cases = tuple(case for case in cases if case.profile == profile)
    fscore_margins: list[float] = []
    geometry_margins: list[float] = []
    topology_excess = 0
    edge_excess = 0
    face_excess = 0
    guard_violations = 0
    for case in selected_cases:
        results = _result_map(case.report)
        baselines = (
            results[BaselineID.B4_DENSITY_SCALED],
            results[BaselineID.B5_PCA_ANISOTROPIC],
        )
        result = results[candidate]
        fscore_margins.append(
            result.endpoints.fscore
            - max(baseline.endpoints.fscore for baseline in baselines)
        )
        geometry_margins.append(
            min(_geometry_loss(baseline) for baseline in baselines)
            - _geometry_loss(result)
        )
        topology_excess += _topology_burden(result) - min(
            _topology_burden(baseline) for baseline in baselines
        )
        edge_excess += (result.endpoints.labeled_false_bridge_edges or 0) - min(
            baseline.endpoints.labeled_false_bridge_edges or 0
            for baseline in baselines
        )
        face_excess += (result.endpoints.labeled_false_bridge_faces or 0) - min(
            baseline.endpoints.labeled_false_bridge_faces or 0
            for baseline in baselines
        )
        guard_violations += int(
            _diagnostic(result, "fallback_guard_violation_count") or 0.0
        )
    mean_fscore_margin = float(np.mean(np.asarray(fscore_margins)))
    mean_geometry_margin = float(np.mean(np.asarray(geometry_margins)))
    tolerance = 1.0e-12
    supported = (
        mean_fscore_margin >= -tolerance
        and mean_geometry_margin >= -tolerance
        and topology_excess <= 0
        and edge_excess <= 0
        and face_excess <= 0
        and guard_violations == 0
    )
    return G5CandidateComparison(
        profile=profile,
        candidate_method=candidate,
        case_count=len(selected_cases),
        mean_fscore_margin=mean_fscore_margin,
        mean_geometry_loss_margin=mean_geometry_margin,
        topology_burden_excess_sum=topology_excess,
        labeled_false_bridge_edges_excess_sum=edge_excess,
        labeled_false_bridge_faces_excess_sum=face_excess,
        fallback_guard_violation_count=guard_violations,
        endpoint_noninferiority_supported=supported,
    )


def _profile_shift(
    cases: tuple[G5CaseBenchmark, ...], profile: str, method: BaselineID
) -> G5ProfileShift:
    indexed = {
        (case.profile, case.repeat_index, case.report.family): _result_map(case.report)[
            method
        ]
        for case in cases
    }
    keys = tuple(
        (case.repeat_index, case.report.family)
        for case in cases
        if case.profile == profile
    )
    fscore_changes: list[float] = []
    geometry_changes: list[float] = []
    topology_change = 0
    edge_change = 0
    face_change = 0
    for repeat_index, family in keys:
        result = indexed[(profile, repeat_index, family)]
        base = indexed[("base", repeat_index, family)]
        fscore_changes.append(result.endpoints.fscore - base.endpoints.fscore)
        geometry_changes.append(_geometry_loss(result) - _geometry_loss(base))
        topology_change += _topology_burden(result) - _topology_burden(base)
        edge_change += (result.endpoints.labeled_false_bridge_edges or 0) - (
            base.endpoints.labeled_false_bridge_edges or 0
        )
        face_change += (result.endpoints.labeled_false_bridge_faces or 0) - (
            base.endpoints.labeled_false_bridge_faces or 0
        )
    return G5ProfileShift(
        profile=profile,
        method=method,
        case_count=len(keys),
        mean_fscore_change_from_base=float(np.mean(np.asarray(fscore_changes))),
        mean_geometry_loss_change_from_base=float(
            np.mean(np.asarray(geometry_changes))
        ),
        topology_burden_change_sum=topology_change,
        labeled_false_bridge_edges_change_sum=edge_change,
        labeled_false_bridge_faces_change_sum=face_change,
    )


def evaluate_g5_preflight(
    *,
    point_count: int = 96,
    reference_count: int = 4096,
    surface_sample_count: int = 512,
    candidate_budget: int = 24,
    adaptive_k_neighbors: int = 12,
    repeat_count: int = 3,
    target_fallback_fraction: float = 0.25,
    seed: int = 20_260_724,
    verbose: bool = False,
) -> G5PreflightResult:
    """Calibrate once, then evaluate all frozen held-out profile/seed pairs."""

    if repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    profiles = default_g5_profiles(
        point_count=point_count, reference_count=reference_count
    )
    initial_config = BenchmarkConfig(
        surface_sample_count=surface_sample_count,
        adaptive_k_neighbors=adaptive_k_neighbors,
        seed=seed,
    )
    calibration_cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=point_count,
        reference_count=reference_count,
        seed=seed,
    )
    if verbose:
        print("[g5] freezing P2 confidence and B4/B5/P1/P2 multipliers", flush=True)
    frozen = freeze_adaptive_config(
        calibration_cases,
        config=initial_config,
        candidate_budget=candidate_budget,
        target_fallback_fraction=target_fallback_fraction,
    )

    case_results: list[G5CaseBenchmark] = []
    for profile in profiles:
        if verbose:
            print(
                f"[g5] profile={profile.name} points={profile.point_count} "
                f"repeats={repeat_count}",
                flush=True,
            )
        for repeat_index in range(repeat_count):
            repeat_seed = seed + 100_003 * repeat_index
            for family_index, family in enumerate(SyntheticFamily):
                case = _profile_case(
                    family,
                    profile,
                    seed=repeat_seed + 10_007 * family_index,
                )
                report = run_case_benchmarks(
                    case, config=frozen.config, methods=ADAPTIVE_METHODS
                )
                if any(
                    result.uses_reference_for_selection for result in report.results
                ):
                    raise RuntimeError(
                        "held-out reference leaked into method selection"
                    )
                case_results.append(
                    G5CaseBenchmark(profile.name, repeat_index, report)
                )
    frozen_cases = tuple(case_results)
    summaries = tuple(
        _method_summary(frozen_cases, profile.name, method)
        for profile in profiles
        for method in ADAPTIVE_METHODS
    )
    comparisons = tuple(
        _candidate_comparison(frozen_cases, profile.name, method)
        for profile in profiles
        for method in PFTF_METHODS
    )
    shifts = tuple(
        _profile_shift(frozen_cases, profile.name, method)
        for profile in profiles
        if profile.name != "base"
        for method in ADAPTIVE_METHODS
    )
    p2_comparisons = tuple(
        comparison
        for comparison in comparisons
        if comparison.candidate_method is BaselineID.P2_CONFIDENCE_FALLBACK
    )
    endpoint_supported = all(
        comparison.endpoint_noninferiority_supported
        for comparison in p2_comparisons
    )
    blockers = [
        "g4_exact_or_validated_fail_closed_fallback_not_deployed",
        "synthetic_preflight_is_not_real_higher_fidelity_held_out_evidence",
        "confirmatory_uncertainty_estimates_not_run",
    ]
    if not endpoint_supported:
        blockers.append("p2_did_not_match_the_B4_B5_endpoint_envelope_in_all_profiles")
    return G5PreflightResult(
        profiles,
        repeat_count,
        seed,
        frozen,
        frozen_cases,
        summaries,
        comparisons,
        shifts,
        endpoint_supported,
        False,
        tuple(blockers),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen synthetic G5 held-out robustness preflight."
    )
    parser.add_argument("--point-count", type=int, default=96)
    parser.add_argument("--reference-count", type=int, default=4096)
    parser.add_argument("--surface-samples", type=int, default=512)
    parser.add_argument("--adaptive-calibration-budget", type=int, default=24)
    parser.add_argument("--adaptive-knn", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--p2-target-fallback-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20_260_724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/g5_frozen_held_out_preflight.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_g5_preflight(
        point_count=args.point_count,
        reference_count=args.reference_count,
        surface_sample_count=args.surface_samples,
        candidate_budget=args.adaptive_calibration_budget,
        adaptive_k_neighbors=args.adaptive_knn,
        repeat_count=args.repeats,
        target_fallback_fraction=args.p2_target_fallback_fraction,
        seed=args.seed,
        verbose=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[g5] endpoint_preflight_supported={result.endpoint_preflight_supported}; "
        f"promotion_supported={result.promotion_supported}",
        flush=True,
    )
    print(f"Wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
