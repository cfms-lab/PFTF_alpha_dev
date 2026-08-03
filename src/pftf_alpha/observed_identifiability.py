"""Observed-only feature identifiability audit for frozen Phase 14."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .local_insertion_influence import (
    LocalInsertionInfluenceConfig,
    LocalInsertionInfluenceEvidence,
    LocalInsertionInfluenceRawCase,
    estimate_local_insertion_influence,
    evaluate_local_insertion_influence_raw_panel,
)
from .multiscale_surface_consensus import (
    MultiscaleQuadraticConfig,
    estimate_multiscale_quadratic_consensus,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
from .sensor_stress import (
    DEFAULT_POINT_COUNTS,
    DEFAULT_STRESSES,
    SensorStress,
    make_sensor_stress_case,
)
from .shared_trend_inference import (
    SharedTrendConfig,
    construct_shared_trend_surface,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CALIBRATION_SEED = 21600804
HELD_OUT_SEED = 21700804

FEATURE_NAMES = (
    "log_point_count",
    "layer_count_imbalance",
    "median_within_layer_neighbor_distance",
    "percentile95_within_layer_neighbor_distance",
    "median_cross_layer_neighbor_distance",
    "layer_centroid_gap",
    "median_gap_direction_thickness",
    "median_insertion_influence",
    "percentile95_insertion_influence",
    "support_insertion_influence",
    "peak_insertion_influence",
    "median_multiscale_residual",
    "percentile95_multiscale_residual",
    "peak_multiscale_residual",
)


@dataclass(frozen=True)
class ObservedIdentifiabilitySignature:
    log_point_count: float
    layer_count_imbalance: float
    median_within_layer_neighbor_distance: float
    percentile95_within_layer_neighbor_distance: float
    median_cross_layer_neighbor_distance: float
    layer_centroid_gap: float
    median_gap_direction_thickness: float
    median_insertion_influence: float
    percentile95_insertion_influence: float
    support_insertion_influence: float
    peak_insertion_influence: float
    median_multiscale_residual: float
    percentile95_multiscale_residual: float
    peak_multiscale_residual: float

    def values(self) -> FloatArray:
        return np.asarray(
            [getattr(self, name) for name in FEATURE_NAMES],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RobustFeatureScaling:
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IdentifiabilityCaseResult:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    audit_class: str
    signature: ObservedIdentifiabilitySignature
    nearest_harmful_distance: float
    nearest_safe_distance: float
    nearest_harmful_seed: int
    nearest_safe_seed: int
    predicted_class: str
    classification_correct: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress"] = self.stress.value
        return payload


@dataclass(frozen=True)
class IdentifiabilityPanelResult:
    panel_role: str
    seed: int
    evaluation_mode: str
    cases: tuple[IdentifiabilityCaseResult, ...]
    audited_case_count: int
    harmful_case_count: int
    safe_focus_case_count: int
    correctly_identified_harmful_count: int
    correctly_identified_safe_count: int
    missed_harmful_count: int
    safe_false_alarm_count: int
    harmful_recall: float
    safe_specificity: float
    panel_identifiable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_role": self.panel_role,
            "seed": self.seed,
            "evaluation_mode": self.evaluation_mode,
            "cases": [case.to_dict() for case in self.cases],
            "audited_case_count": self.audited_case_count,
            "harmful_case_count": self.harmful_case_count,
            "safe_focus_case_count": self.safe_focus_case_count,
            "correctly_identified_harmful_count": (
                self.correctly_identified_harmful_count
            ),
            "correctly_identified_safe_count": self.correctly_identified_safe_count,
            "missed_harmful_count": self.missed_harmful_count,
            "safe_false_alarm_count": self.safe_false_alarm_count,
            "harmful_recall": self.harmful_recall,
            "safe_specificity": self.safe_specificity,
            "panel_identifiable": self.panel_identifiable,
        }


@dataclass(frozen=True)
class ObservedIdentifiabilityResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    calibration_seed: int
    held_out_seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    influence_config: LocalInsertionInfluenceConfig
    multiscale_config: MultiscaleQuadraticConfig
    feature_names: tuple[str, ...]
    scaling: RobustFeatureScaling
    calibration: IdentifiabilityPanelResult
    held_out: IdentifiabilityPanelResult
    full_protocol: bool
    feature_identifiable: bool
    guard_supported: bool
    trimmed_reconstruction_supported: bool
    real_scan_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_predecessor": self.frozen_predecessor,
            "calibration_seed": self.calibration_seed,
            "held_out_seed": self.held_out_seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "influence_config": asdict(self.influence_config),
            "multiscale_config": asdict(self.multiscale_config),
            "feature_names": list(self.feature_names),
            "scaling": self.scaling.to_dict(),
            "calibration": self.calibration.to_dict(),
            "held_out": self.held_out.to_dict(),
            "full_protocol": self.full_protocol,
            "feature_identifiable": self.feature_identifiable,
            "guard_supported": self.guard_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "real_scan_supported": self.real_scan_supported,
            "deployment_supported": self.deployment_supported,
        }


@dataclass(frozen=True)
class _SignatureCase:
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    audit_class: str
    signature: ObservedIdentifiabilitySignature


def estimate_observed_identifiability_signature(
    points: FloatArray,
    inferred_labels: IntArray,
    *,
    influence_evidence: LocalInsertionInfluenceEvidence | None = None,
    influence_config: LocalInsertionInfluenceConfig | None = None,
    multiscale_config: MultiscaleQuadraticConfig | None = None,
) -> ObservedIdentifiabilitySignature:
    """Compute the frozen 14-feature signature from observed data only."""

    point_array = np.asarray(points, dtype=np.float64)
    labels = np.asarray(inferred_labels, dtype=np.int64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if labels.shape != (point_array.shape[0],) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("inferred_labels must contain two aligned layers")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must be finite")

    selected_influence = (
        LocalInsertionInfluenceConfig()
        if influence_config is None
        else influence_config
    )
    selected_multiscale = (
        MultiscaleQuadraticConfig()
        if multiscale_config is None
        else multiscale_config
    )
    influence = (
        estimate_local_insertion_influence(
            point_array,
            labels,
            selected_influence,
        )
        if influence_evidence is None
        else influence_evidence
    )
    multiscale = estimate_multiscale_quadratic_consensus(
        point_array,
        labels,
        selected_multiscale,
    )

    epsilon = np.finfo(float).eps
    diagonal = max(float(np.linalg.norm(np.ptp(point_array, axis=0))), epsilon)
    layer_indices = tuple(np.flatnonzero(labels == layer) for layer in (0, 1))
    layer_points = tuple(point_array[indices] for indices in layer_indices)
    nearest_within = np.concatenate(
        tuple(
            cKDTree(values).query(values, k=2, workers=1)[0][:, 1]
            for values in layer_points
        )
    )
    cross_distances = np.concatenate(
        (
            cKDTree(layer_points[1]).query(layer_points[0], workers=1)[0],
            cKDTree(layer_points[0]).query(layer_points[1], workers=1)[0],
        )
    )
    centroids = tuple(np.mean(values, axis=0) for values in layer_points)
    centroid_delta = centroids[1] - centroids[0]
    centroid_gap = float(np.linalg.norm(centroid_delta))
    if centroid_gap > epsilon:
        gap_direction = centroid_delta / centroid_gap
    else:
        gap_direction = np.asarray([0.0, 0.0, 1.0])
    gap_thickness = np.concatenate(
        tuple(
            np.abs((values - centroid) @ gap_direction)
            for values, centroid in zip(layer_points, centroids, strict=True)
        )
    )
    imbalance = abs(layer_indices[0].size - layer_indices[1].size) / labels.size
    return ObservedIdentifiabilitySignature(
        log_point_count=float(np.log(point_array.shape[0])),
        layer_count_imbalance=float(imbalance),
        median_within_layer_neighbor_distance=float(
            np.median(nearest_within) / diagonal
        ),
        percentile95_within_layer_neighbor_distance=float(
            np.percentile(nearest_within, 95.0) / diagonal
        ),
        median_cross_layer_neighbor_distance=float(
            np.median(cross_distances) / diagonal
        ),
        layer_centroid_gap=centroid_gap / diagonal,
        median_gap_direction_thickness=float(
            np.median(gap_thickness) / diagonal
        ),
        median_insertion_influence=influence.median_standardized_influence,
        percentile95_insertion_influence=(
            influence.percentile95_standardized_influence
        ),
        support_insertion_influence=influence.support_standardized_influence,
        peak_insertion_influence=influence.peak_standardized_influence,
        median_multiscale_residual=multiscale.median_standardized_residual,
        percentile95_multiscale_residual=(
            multiscale.percentile95_standardized_residual
        ),
        peak_multiscale_residual=multiscale.maximum_standardized_residual,
    )


def fit_robust_feature_scaling(
    signatures: Sequence[ObservedIdentifiabilitySignature],
) -> RobustFeatureScaling:
    """Fit the predeclared feature-wise median and fallback scale."""

    if not signatures:
        raise ValueError("signatures must be non-empty")
    matrix = np.vstack([signature.values() for signature in signatures])
    medians = np.median(matrix, axis=0)
    mad_scales = 1.4826 * np.median(np.abs(matrix - medians), axis=0)
    iqr_scales = (
        np.percentile(matrix, 75.0, axis=0)
        - np.percentile(matrix, 25.0, axis=0)
    ) / 1.349
    fallback = np.maximum(0.05 * np.abs(medians), 1.0e-9)
    scales = np.where(
        mad_scales > 0.0,
        mad_scales,
        np.where(iqr_scales > 0.0, iqr_scales, fallback),
    )
    return RobustFeatureScaling(
        feature_names=FEATURE_NAMES,
        medians=tuple(float(value) for value in medians),
        scales=tuple(float(value) for value in scales),
    )


def _standardize(
    signature: ObservedIdentifiabilitySignature,
    scaling: RobustFeatureScaling,
) -> FloatArray:
    return (
        signature.values() - np.asarray(scaling.medians, dtype=np.float64)
    ) / np.asarray(scaling.scales, dtype=np.float64)


def _audit_class(row: LocalInsertionInfluenceRawCase) -> str | None:
    accepted = row.unguarded_decision is SamplingGateDecision.ACCEPT
    if (
        accepted
        and row.stress.is_outlier_stress
        and row.endpoint.geometry_topology_harm_present
    ):
        return "harmful"
    if (
        accepted
        and row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
        and not row.endpoint.geometry_topology_harm_present
    ):
        return "safe_focus"
    return None


def _signature_cases(
    raw_rows: tuple[LocalInsertionInfluenceRawCase, ...],
    *,
    reference_count: int,
    shared_trend_config: SharedTrendConfig | None,
    influence_config: LocalInsertionInfluenceConfig,
    multiscale_config: MultiscaleQuadraticConfig,
) -> tuple[_SignatureCase, ...]:
    results: list[_SignatureCase] = []
    for row in raw_rows:
        audit_class = _audit_class(row)
        if audit_class is None:
            continue
        case = make_sensor_stress_case(
            row.stress,
            row.point_count,
            reference_count=max(reference_count, row.point_count),
            seed=row.seed,
        )
        construction, _ = construct_shared_trend_surface(
            case.points,
            shared_trend_config,
        )
        signature = estimate_observed_identifiability_signature(
            case.points,
            construction.inference.layer_ids,
            influence_evidence=row.evidence,
            influence_config=influence_config,
            multiscale_config=multiscale_config,
        )
        results.append(
            _SignatureCase(
                stress=row.stress,
                point_count=row.point_count,
                repeat=row.repeat,
                seed=row.seed,
                audit_class=audit_class,
                signature=signature,
            )
        )
    return tuple(results)


def _nearest_class(
    query: _SignatureCase,
    references: tuple[_SignatureCase, ...],
    scaling: RobustFeatureScaling,
    *,
    exclude_seed: int | None,
) -> tuple[float, float, int, int, str]:
    query_values = _standardize(query.signature, scaling)
    nearest: dict[str, tuple[float, int]] = {}
    for audit_class in ("harmful", "safe_focus"):
        candidates = [
            reference
            for reference in references
            if reference.audit_class == audit_class
            and (exclude_seed is None or reference.seed != exclude_seed)
        ]
        if not candidates:
            raise ValueError(f"no {audit_class} reference remains after exclusion")
        distances = [
            float(
                np.linalg.norm(
                    query_values - _standardize(candidate.signature, scaling)
                )
            )
            for candidate in candidates
        ]
        index = int(np.argmin(distances))
        nearest[audit_class] = (distances[index], candidates[index].seed)
    harmful_distance, harmful_seed = nearest["harmful"]
    safe_distance, safe_seed = nearest["safe_focus"]
    predicted = "harmful" if harmful_distance <= safe_distance else "safe_focus"
    return harmful_distance, safe_distance, harmful_seed, safe_seed, predicted


def _panel_result(
    cases: tuple[_SignatureCase, ...],
    references: tuple[_SignatureCase, ...],
    scaling: RobustFeatureScaling,
    *,
    panel_role: str,
    seed: int,
    leave_one_out: bool,
) -> IdentifiabilityPanelResult:
    harmful_count = sum(case.audit_class == "harmful" for case in cases)
    safe_count = sum(case.audit_class == "safe_focus" for case in cases)
    required_references = 2 if leave_one_out else 1
    reference_harmful = sum(
        case.audit_class == "harmful" for case in references
    )
    reference_safe = sum(
        case.audit_class == "safe_focus" for case in references
    )
    if (
        reference_harmful < required_references
        or reference_safe < required_references
    ):
        return IdentifiabilityPanelResult(
            panel_role=panel_role,
            seed=seed,
            evaluation_mode=(
                "calibration_leave_one_out"
                if leave_one_out
                else "held_out_nearest_calibration_class"
            ),
            cases=(),
            audited_case_count=len(cases),
            harmful_case_count=harmful_count,
            safe_focus_case_count=safe_count,
            correctly_identified_harmful_count=0,
            correctly_identified_safe_count=0,
            missed_harmful_count=harmful_count,
            safe_false_alarm_count=safe_count,
            harmful_recall=0.0,
            safe_specificity=0.0,
            panel_identifiable=False,
        )
    results: list[IdentifiabilityCaseResult] = []
    for case in cases:
        harmful_distance, safe_distance, harmful_seed, safe_seed, predicted = (
            _nearest_class(
                case,
                references,
                scaling,
                exclude_seed=case.seed if leave_one_out else None,
            )
        )
        results.append(
            IdentifiabilityCaseResult(
                stress=case.stress,
                point_count=case.point_count,
                repeat=case.repeat,
                seed=case.seed,
                audit_class=case.audit_class,
                signature=case.signature,
                nearest_harmful_distance=harmful_distance,
                nearest_safe_distance=safe_distance,
                nearest_harmful_seed=harmful_seed,
                nearest_safe_seed=safe_seed,
                predicted_class=predicted,
                classification_correct=predicted == case.audit_class,
            )
        )
    harmful = [case for case in results if case.audit_class == "harmful"]
    safe = [case for case in results if case.audit_class == "safe_focus"]
    correct_harmful = sum(case.classification_correct for case in harmful)
    correct_safe = sum(case.classification_correct for case in safe)
    harmful_recall = 0.0 if not harmful else correct_harmful / len(harmful)
    safe_specificity = 0.0 if not safe else correct_safe / len(safe)
    identifiable = bool(
        harmful
        and safe
        and harmful_recall == 1.0
        and safe_specificity >= 0.90
    )
    return IdentifiabilityPanelResult(
        panel_role=panel_role,
        seed=seed,
        evaluation_mode=(
            "calibration_leave_one_out"
            if leave_one_out
            else "held_out_nearest_calibration_class"
        ),
        cases=tuple(results),
        audited_case_count=len(results),
        harmful_case_count=len(harmful),
        safe_focus_case_count=len(safe),
        correctly_identified_harmful_count=correct_harmful,
        correctly_identified_safe_count=correct_safe,
        missed_harmful_count=len(harmful) - correct_harmful,
        safe_false_alarm_count=len(safe) - correct_safe,
        harmful_recall=harmful_recall,
        safe_specificity=safe_specificity,
        panel_identifiable=identifiable,
    )


def evaluate_observed_identifiability(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    calibration_seed: int = CALIBRATION_SEED,
    held_out_seed: int = HELD_OUT_SEED,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
    influence_config: LocalInsertionInfluenceConfig | None = None,
    multiscale_config: MultiscaleQuadraticConfig | None = None,
) -> ObservedIdentifiabilityResult:
    """Execute the frozen calibration/held-out identifiability audit."""

    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_influence = (
        LocalInsertionInfluenceConfig()
        if influence_config is None
        else influence_config
    )
    selected_multiscale = (
        MultiscaleQuadraticConfig()
        if multiscale_config is None
        else multiscale_config
    )
    if repeats < 1 or not selected_counts or not selected_stresses:
        raise ValueError("counts/stresses must be non-empty and repeats positive")
    if calibration_seed == held_out_seed:
        raise ValueError("calibration and held-out seeds must differ")
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and repeats == 8
        and reference_count == 2048
        and surface_sample_count == 256
        and calibration_seed == CALIBRATION_SEED
        and held_out_seed == HELD_OUT_SEED
    )
    common = {
        "point_counts": selected_counts,
        "stresses": selected_stresses,
        "reference_count": reference_count,
        "repeats": repeats,
        "surface_sample_count": surface_sample_count,
        "base_gate_config": base_gate_config,
        "shared_trend_config": shared_trend_config,
        "influence_config": selected_influence,
    }
    calibration_raw = evaluate_local_insertion_influence_raw_panel(
        seed=calibration_seed,
        **common,
    )
    held_out_raw = evaluate_local_insertion_influence_raw_panel(
        seed=held_out_seed,
        **common,
    )
    signature_common = {
        "reference_count": reference_count,
        "shared_trend_config": shared_trend_config,
        "influence_config": selected_influence,
        "multiscale_config": selected_multiscale,
    }
    calibration_cases = _signature_cases(calibration_raw, **signature_common)
    held_out_cases = _signature_cases(held_out_raw, **signature_common)
    if calibration_cases:
        scaling = fit_robust_feature_scaling(
            [case.signature for case in calibration_cases]
        )
    else:
        scaling = RobustFeatureScaling(
            feature_names=FEATURE_NAMES,
            medians=tuple(0.0 for _ in FEATURE_NAMES),
            scales=tuple(1.0 for _ in FEATURE_NAMES),
        )
    calibration = _panel_result(
        calibration_cases,
        calibration_cases,
        scaling,
        panel_role="calibration",
        seed=calibration_seed,
        leave_one_out=True,
    )
    held_out = _panel_result(
        held_out_cases,
        calibration_cases,
        scaling,
        panel_role="held_out",
        seed=held_out_seed,
        leave_one_out=False,
    )
    identifiable = bool(
        full_protocol
        and calibration.panel_identifiable
        and held_out.panel_identifiable
    )
    return ObservedIdentifiabilityResult(
        artifact_schema="pftf_alpha_observed_identifiability_phase14/v1",
        role="diagnostic_observed_signature_nearest_class_audit",
        information_boundary=(
            "signatures use observed coordinates and inferred layers only; "
            "stress and harm labels define evaluation classes after extraction"
        ),
        frozen_predecessor="phase13_seeds_21300804_21400804_21500804_negative",
        calibration_seed=calibration_seed,
        held_out_seed=held_out_seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        influence_config=selected_influence,
        multiscale_config=selected_multiscale,
        feature_names=FEATURE_NAMES,
        scaling=scaling,
        calibration=calibration,
        held_out=held_out,
        full_protocol=full_protocol,
        feature_identifiable=identifiable,
        guard_supported=False,
        trimmed_reconstruction_supported=False,
        real_scan_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--held-out-seed", type=int, default=HELD_OUT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_observed_identifiability(
        reference_count=args.reference,
        repeats=args.repeats,
        calibration_seed=args.calibration_seed,
        held_out_seed=args.held_out_seed,
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
