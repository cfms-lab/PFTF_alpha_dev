"""Frozen Phase-8 sensor-style stress test for shared-trend inference."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .reacquisition import ReacquisitionConfig
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .shared_trend_inference import (
    SharedTrendConfig,
    SharedTrendDiagnostics,
    construct_shared_trend_surface,
)
from .surface import SurfaceEndpointMetrics, evaluate_surface
from .synthetic import PanelSplit, SyntheticCase, SyntheticFamily
from .two_layer_connectivity import (
    construct_two_layer_surface,
    route_two_layer_output,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class SensorStress(StrEnum):
    CONTROL = "control"
    UPPER_OCCLUSION = "upper_occlusion"
    IMBALANCED_75_25 = "imbalanced_75_25"
    ANISOTROPIC_NOISE = "anisotropic_noise"
    OUTLIERS_01 = "outliers_01"
    OUTLIERS_03 = "outliers_03"
    OUTLIERS_05 = "outliers_05"
    SINUSOIDAL = "sinusoidal"
    LOCAL_BUMP = "local_bump"

    @property
    def outlier_fraction(self) -> float:
        return {
            SensorStress.OUTLIERS_01: 0.01,
            SensorStress.OUTLIERS_03: 0.03,
            SensorStress.OUTLIERS_05: 0.05,
        }.get(self, 0.0)

    @property
    def is_outlier_stress(self) -> bool:
        return self.outlier_fraction > 0.0

    @property
    def is_nonquadratic(self) -> bool:
        return self in (SensorStress.SINUSOIDAL, SensorStress.LOCAL_BUMP)


DEFAULT_STRESSES = tuple(SensorStress)
DEFAULT_POINT_COUNTS = (96, 160, 256)


@dataclass(frozen=True)
class SensorStressCaseResult:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    outlier_count: int
    diagnostics: SharedTrendDiagnostics
    base_decision: SamplingGateDecision
    candidate_decision: SamplingGateDecision
    base_true_safe_output: bool
    candidate_true_safe_output: bool
    base_safe_accept: bool
    candidate_safe_accept: bool
    base_false_safe: bool
    candidate_false_safe: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress"] = self.stress.value
        payload["base_decision"] = self.base_decision.value
        payload["candidate_decision"] = self.candidate_decision.value
        return payload


@dataclass(frozen=True)
class SensorStressSummary:
    stress: SensorStress
    case_count: int
    candidate_accept_count: int
    candidate_safe_accept_count: int
    candidate_false_safe_count: int
    safe_acceptance_coverage: float
    group_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress"] = self.stress.value
        return payload


@dataclass(frozen=True)
class SensorDensitySummary:
    point_count: int
    case_count: int
    candidate_safe_accept_count: int
    candidate_false_safe_count: int
    non_outlier_safe_acceptance_coverage: float
    density_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SensorStressResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_candidate_source: str
    seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    base_gate_config: SamplingSufficiencyConfig
    shared_trend_config: SharedTrendConfig
    cases: tuple[SensorStressCaseResult, ...]
    stress_summaries: tuple[SensorStressSummary, ...]
    density_summaries: tuple[SensorDensitySummary, ...]
    case_count: int
    base_false_safe_count: int
    candidate_safe_accept_count: int
    candidate_false_safe_count: int
    non_outlier_safe_acceptance_coverage: float
    nonquadratic_safe_acceptance_coverage: float
    phase8_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_candidate_source": self.frozen_candidate_source,
            "seed": self.seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "base_gate_config": asdict(self.base_gate_config),
            "shared_trend_config": asdict(self.shared_trend_config),
            "cases": [case.to_dict() for case in self.cases],
            "stress_summaries": [row.to_dict() for row in self.stress_summaries],
            "density_summaries": [row.to_dict() for row in self.density_summaries],
            "case_count": self.case_count,
            "base_false_safe_count": self.base_false_safe_count,
            "candidate_safe_accept_count": self.candidate_safe_accept_count,
            "candidate_false_safe_count": self.candidate_false_safe_count,
            "non_outlier_safe_acceptance_coverage": (
                self.non_outlier_safe_acceptance_coverage
            ),
            "nonquadratic_safe_acceptance_coverage": (
                self.nonquadratic_safe_acceptance_coverage
            ),
            "phase8_supported": self.phase8_supported,
            "deployment_supported": self.deployment_supported,
        }


def _height(stress: SensorStress, xy: FloatArray, layer: int) -> FloatArray:
    x, y = xy[:, 0], xy[:, 1]
    if stress is SensorStress.SINUSOIDAL:
        middle = 0.22 * np.sin(np.pi * x) * np.cos(np.pi * y)
    elif stress is SensorStress.LOCAL_BUMP:
        distance = ((x - 0.25) ** 2 + (y + 0.15) ** 2) / 0.10
        middle = 0.34 * np.exp(-distance)
    else:
        middle = 0.24 * (x * x + y * y)
    return middle + (-0.40 if layer == 0 else 0.40)


def _sample_xy(
    count: int,
    layer: int,
    stress: SensorStress,
    rng: np.random.Generator,
) -> FloatArray:
    lower_x = -0.20 if stress is SensorStress.UPPER_OCCLUSION and layer == 1 else -1.0
    return np.column_stack(
        (
            rng.uniform(lower_x, 1.0, size=count),
            rng.uniform(-1.0, 1.0, size=count),
        )
    )


def _layer_counts(total: int, stress: SensorStress) -> tuple[int, int]:
    if stress is SensorStress.IMBALANCED_75_25:
        first = int(round(0.75 * total))
    else:
        first = total // 2
    return first, total - first


def make_sensor_stress_case(
    stress: SensorStress | str,
    point_count: int,
    *,
    reference_count: int = 2048,
    seed: int = 0,
) -> SyntheticCase:
    """Generate one sensor-style stress case with evaluation-only labels."""

    selected = SensorStress(stress)
    if point_count < 32:
        raise ValueError("point_count must be at least 32")
    if reference_count < point_count:
        raise ValueError("reference_count must be at least point_count")
    observed_rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed + 1_000_003)
    outlier_count = int(round(point_count * selected.outlier_fraction))
    surface_count = point_count - outlier_count

    def sample_surfaces(
        count: int,
        rng: np.random.Generator,
        *,
        noisy: bool,
    ) -> tuple[FloatArray, IntArray]:
        point_rows: list[FloatArray] = []
        label_rows: list[IntArray] = []
        for layer, layer_count in enumerate(_layer_counts(count, selected)):
            xy = _sample_xy(layer_count, layer, selected, rng)
            z = _height(selected, xy, layer)
            points = np.column_stack((xy, z))
            if noisy:
                if selected is SensorStress.ANISOTROPIC_NOISE:
                    scales = np.asarray((0.006, 0.006, 0.040))
                else:
                    scales = np.asarray((0.010, 0.010, 0.010))
                points = points + rng.normal(size=points.shape) * scales
            point_rows.append(points)
            label_rows.append(
                np.full(layer_count, layer, dtype=np.int64)
            )
        return np.vstack(point_rows), np.concatenate(label_rows)

    observed, labels = sample_surfaces(surface_count, observed_rng, noisy=True)
    if outlier_count:
        outliers = observed_rng.uniform(
            low=np.asarray((-1.0, -1.0, -0.65)),
            high=np.asarray((1.0, 1.0, 0.95)),
            size=(outlier_count, 3),
        )
        observed = np.vstack((observed, outliers))
        labels = np.concatenate(
            (labels, np.full(outlier_count, 2, dtype=np.int64))
        )
    reference, _ = sample_surfaces(
        reference_count,
        reference_rng,
        noisy=False,
    )
    characteristic_length = float(np.linalg.norm(np.ptp(reference, axis=0)))
    return SyntheticCase(
        family=SyntheticFamily.OPPOSING_SHEETS,
        split=PanelSplit.HELD_OUT,
        points=observed,
        reference_points=reference,
        expected_components=3 if outlier_count else 2,
        characteristic_length=characteristic_length,
        variation={
            "stress_index": float(list(SensorStress).index(selected)),
            "outlier_fraction": selected.outlier_fraction,
        },
        seed=seed,
        expected_surface_betti=(3, 0, 0) if outlier_count else (2, 0, 0),
        point_component_labels=labels,
    )


def _true_safe(metrics: SurfaceEndpointMetrics) -> bool:
    return bool(
        metrics.component_error == 0
        and int(metrics.labeled_false_bridge_edges or 0) == 0
        and int(metrics.labeled_false_bridge_faces or 0) == 0
    )


def _stress_summary(
    cases: Sequence[SensorStressCaseResult],
    stress: SensorStress,
) -> SensorStressSummary:
    rows = [case for case in cases if case.stress is stress]
    safe_accept = sum(case.candidate_safe_accept for case in rows)
    false_safe = sum(case.candidate_false_safe for case in rows)
    coverage = 0.0 if stress.is_outlier_stress else safe_accept / len(rows)
    passed = bool(
        false_safe == 0
        and (stress.is_outlier_stress or coverage >= 0.75)
    )
    return SensorStressSummary(
        stress=stress,
        case_count=len(rows),
        candidate_accept_count=sum(
            case.candidate_decision is SamplingGateDecision.ACCEPT for case in rows
        ),
        candidate_safe_accept_count=safe_accept,
        candidate_false_safe_count=false_safe,
        safe_acceptance_coverage=coverage,
        group_gate_passed=passed,
    )


def _density_summary(
    cases: Sequence[SensorStressCaseResult],
    point_count: int,
) -> SensorDensitySummary:
    rows = [case for case in cases if case.point_count == point_count]
    non_outlier = [case for case in rows if not case.stress.is_outlier_stress]
    safe_accept = sum(case.candidate_safe_accept for case in non_outlier)
    false_safe = sum(case.candidate_false_safe for case in rows)
    coverage = safe_accept / len(non_outlier)
    return SensorDensitySummary(
        point_count=point_count,
        case_count=len(rows),
        candidate_safe_accept_count=sum(
            case.candidate_safe_accept for case in rows
        ),
        candidate_false_safe_count=false_safe,
        non_outlier_safe_acceptance_coverage=coverage,
        density_gate_passed=bool(false_safe == 0 and coverage >= 0.75),
    )


def evaluate_sensor_stress(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    seed: int = 20600804,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
) -> SensorStressResult:
    """Run the frozen Phase-8 sensor-style transfer panel."""

    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    if repeats < 1 or not selected_counts or not selected_stresses:
        raise ValueError("counts/stresses must be non-empty and repeats positive")
    if any(count < 32 for count in selected_counts):
        raise ValueError("point counts must be at least 32")
    selected_base = base_gate_config or SamplingSufficiencyConfig(
        minimum_separation_snr=3.0
    )
    selected_trend = shared_trend_config or SharedTrendConfig(
        k_neighbors=selected_base.k_neighbors,
        minimum_cluster_fraction=selected_base.minimum_cluster_fraction,
        minimum_separation_snr=selected_base.minimum_separation_snr,
        cross_knn_threshold=selected_base.cross_knn_threshold,
    )
    reconstruction_config = ReacquisitionConfig(
        base_point_count=max(selected_counts),
        evaluation_reference_count=reference_count,
        candidate_pool_count=reference_count,
        added_point_counts=(1,),
        repeats=1,
        seed=seed,
        surface_sample_count=surface_sample_count,
        k_neighbors=selected_base.k_neighbors,
    )
    results: list[SensorStressCaseResult] = []
    for count_index, point_count in enumerate(selected_counts):
        for stress_index, stress in enumerate(selected_stresses):
            for repeat in range(repeats):
                case_seed = (
                    seed
                    + count_index * 1_000_003
                    + stress_index * 100_003
                    + repeat * 10_007
                )
                case = make_sensor_stress_case(
                    stress,
                    point_count,
                    reference_count=reference_count,
                    seed=case_seed,
                )
                base = construct_two_layer_surface(case.points, selected_base)
                candidate, diagnostics = construct_shared_trend_surface(
                    case.points,
                    selected_trend,
                )
                evaluations = []
                for construction, labels, expected_components in (
                    (base, base.inference.layer_ids, 2),
                    (candidate, candidate.inference.layer_ids, 2),
                    (base, case.point_component_labels, case.expected_components),
                    (
                        candidate,
                        case.point_component_labels,
                        case.expected_components,
                    ),
                ):
                    evaluations.append(
                        evaluate_surface(
                            construction.mesh,
                            case.reference_points,
                            expected_components=expected_components,
                            expected_betti=(expected_components, 0, 0),
                            vertex_component_labels=labels,
                            characteristic_length=case.characteristic_length,
                            sample_count=surface_sample_count,
                            threshold_fraction=(
                                reconstruction_config.fscore_threshold_fraction
                            ),
                            seed=case_seed + 41,
                        )
                    )
                base_inferred, candidate_inferred, base_truth, candidate_truth = (
                    evaluations
                )
                base_decision = route_two_layer_output(base, base_inferred)
                candidate_decision = route_two_layer_output(
                    candidate,
                    candidate_inferred,
                )
                base_safe = _true_safe(base_truth)
                candidate_safe = _true_safe(candidate_truth)
                base_accept = base_decision is SamplingGateDecision.ACCEPT
                candidate_accept = (
                    candidate_decision is SamplingGateDecision.ACCEPT
                )
                results.append(
                    SensorStressCaseResult(
                        stress=stress,
                        point_count=point_count,
                        repeat=repeat,
                        seed=case_seed,
                        outlier_count=int(
                            np.sum(case.point_component_labels == 2)
                        ),
                        diagnostics=diagnostics,
                        base_decision=base_decision,
                        candidate_decision=candidate_decision,
                        base_true_safe_output=base_safe,
                        candidate_true_safe_output=candidate_safe,
                        base_safe_accept=bool(base_accept and base_safe),
                        candidate_safe_accept=bool(
                            candidate_accept and candidate_safe
                        ),
                        base_false_safe=bool(base_accept and not base_safe),
                        candidate_false_safe=bool(
                            candidate_accept and not candidate_safe
                        ),
                    )
                )

    stress_summaries = tuple(
        _stress_summary(results, stress) for stress in selected_stresses
    )
    density_summaries = tuple(
        _density_summary(results, point_count) for point_count in selected_counts
    )
    non_outlier = [
        case for case in results if not case.stress.is_outlier_stress
    ]
    nonquadratic = [case for case in results if case.stress.is_nonquadratic]
    non_outlier_coverage = sum(
        case.candidate_safe_accept for case in non_outlier
    ) / len(non_outlier)
    nonquadratic_coverage = (
        0.0
        if not nonquadratic
        else sum(case.candidate_safe_accept for case in nonquadratic)
        / len(nonquadratic)
    )
    candidate_false = sum(case.candidate_false_safe for case in results)
    supported = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and repeats >= 8
        and len(results) == 216
        and sum(case.base_false_safe for case in results) > 0
        and candidate_false == 0
        and non_outlier_coverage >= 0.75
        and nonquadratic_coverage >= 0.75
        and all(row.group_gate_passed for row in stress_summaries)
        and all(row.density_gate_passed for row in density_summaries)
    )
    return SensorStressResult(
        artifact_schema="pftf_alpha_sensor_stress_phase8/v1",
        role="frozen_sensor_style_shared_trend_transfer_test",
        information_boundary=(
            "routes use observed coordinates only; stress identity, true labels, "
            "and clean dense references are evaluation-only"
        ),
        frozen_candidate_source="phase7_seed_20500804",
        seed=seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        base_gate_config=selected_base,
        shared_trend_config=selected_trend,
        cases=tuple(results),
        stress_summaries=stress_summaries,
        density_summaries=density_summaries,
        case_count=len(results),
        base_false_safe_count=sum(case.base_false_safe for case in results),
        candidate_safe_accept_count=sum(
            case.candidate_safe_accept for case in results
        ),
        candidate_false_safe_count=candidate_false,
        non_outlier_safe_acceptance_coverage=non_outlier_coverage,
        nonquadratic_safe_acceptance_coverage=nonquadratic_coverage,
        phase8_supported=supported,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20600804)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_sensor_stress(
        reference_count=args.reference,
        repeats=args.repeats,
        seed=args.seed,
        surface_sample_count=args.surface_samples,
    )
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
