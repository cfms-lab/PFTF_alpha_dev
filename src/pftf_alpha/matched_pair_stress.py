"""Correspondence/registration stress audit for Phase-17 matched pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .conservative_influence_calibration import InfluenceFeatureCohort
from .local_insertion_influence import InfluenceRectangle
from .local_spatial_displacement import (
    LocalSpatialDisplacementEvidence,
    estimate_local_spatial_displacement_evidence,
)
from .local_surface_consensus import (
    GeometryTopologyHarmEndpoint,
    LocalSurfaceConsensusConfig,
    evaluate_geometry_topology_harm,
)
from .matched_pair_consistency import (
    MatchedPairConfig,
    MatchedPairEvidence,
    estimate_matched_pair_evidence,
    make_matched_repeat_observation,
)
from .sampling_gate import (
    ParallelLayerInference,
    SamplingGateDecision,
    SamplingSufficiencyConfig,
)
from .sensor_stress import (
    DEFAULT_POINT_COUNTS,
    DEFAULT_STRESSES,
    SensorStress,
    evaluate_sensor_stress,
    make_sensor_stress_case,
)
from .shared_trend_inference import (
    SharedTrendConfig,
    construct_shared_trend_surface,
)
from .two_layer_connectivity import construct_two_layer_surface_from_inference

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CALIBRATION_A_SEED = 22700804
CALIBRATION_B_SEED = 22800804
FINAL_HELD_OUT_SEED = 22900804


class MatchedPairStressProfile(StrEnum):
    EXACT = "exact"
    REGISTRATION_0P5DEG = "registration_0p5deg"
    MISSING_10PCT = "missing_10pct"
    MISMATCH_02 = "mismatch_02"
    COMBINED = "combined"


@dataclass(frozen=True)
class MatchedPairStressSpec:
    profile: MatchedPairStressProfile
    rotation_degrees: float
    missing_fraction: float
    mismatch_fraction: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.rotation_degrees) or self.rotation_degrees < 0.0:
            raise ValueError("rotation_degrees must be finite and non-negative")
        for name, value in (
            ("missing_fraction", self.missing_fraction),
            ("mismatch_fraction", self.mismatch_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value < 0.5:
                raise ValueError(f"{name} must lie in [0, 0.5)")


DEFAULT_STRESS_SPECS = (
    MatchedPairStressSpec(MatchedPairStressProfile.EXACT, 0.0, 0.0, 0.0),
    MatchedPairStressSpec(
        MatchedPairStressProfile.REGISTRATION_0P5DEG,
        0.5,
        0.0,
        0.0,
    ),
    MatchedPairStressSpec(
        MatchedPairStressProfile.MISSING_10PCT,
        0.0,
        0.10,
        0.0,
    ),
    MatchedPairStressSpec(
        MatchedPairStressProfile.MISMATCH_02,
        0.0,
        0.0,
        0.02,
    ),
    MatchedPairStressSpec(
        MatchedPairStressProfile.COMBINED,
        0.5,
        0.10,
        0.02,
    ),
)


@dataclass(frozen=True)
class MatchedPairStressConfig:
    perturbation_seed_offset: int = 900000007
    profile_seed_stride: int = 1000003

    def __post_init__(self) -> None:
        if self.perturbation_seed_offset <= 0 or self.profile_seed_stride <= 0:
            raise ValueError("stress seed controls must be positive")


@dataclass(frozen=True)
class PerturbedMatchedPairs:
    primary_points: FloatArray
    repeat_points: FloatArray
    primary_ids: IntArray
    repeat_source_ids: IntArray
    retained_pair_count: int
    missing_pair_count: int
    mismatched_pair_count: int
    rotation_degrees: float
    rotation_axis: tuple[float, float, float]
    presented_pair_map_sha256: str


@dataclass(frozen=True)
class MatchedPairStressRawCase:
    profile: MatchedPairStressProfile
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    perturbation_seed: int
    repeat_transient_outlier_count: int
    repeat_transient_outlier_index_sha256: str
    retained_pair_count: int
    missing_pair_count: int
    mismatched_pair_count: int
    rotation_degrees: float
    rotation_axis: tuple[float, float, float]
    presented_pair_map_sha256: str
    evidence: MatchedPairEvidence
    endpoint: GeometryTopologyHarmEndpoint
    matched_subset_endpoint: GeometryTopologyHarmEndpoint
    frozen_partition_endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision
    local_spatial_evidence: LocalSpatialDisplacementEvidence | None = None


@dataclass(frozen=True)
class MatchedPairStressCaseResult:
    profile: MatchedPairStressProfile
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    perturbation_seed: int
    repeat_transient_outlier_count: int
    repeat_transient_outlier_index_sha256: str
    retained_pair_count: int
    missing_pair_count: int
    mismatched_pair_count: int
    rotation_degrees: float
    rotation_axis: tuple[float, float, float]
    presented_pair_map_sha256: str
    evidence: MatchedPairEvidence
    endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision
    guarded_decision: SamplingGateDecision
    unguarded_safe_accept: bool
    guarded_safe_accept: bool
    unguarded_harmful_outlier_false_safe: bool
    guarded_harmful_outlier_false_safe: bool
    unguarded_provenance_violation_accept: bool
    guarded_provenance_violation_accept: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        payload["stress"] = self.stress.value
        payload["unguarded_decision"] = self.unguarded_decision.value
        payload["guarded_decision"] = self.guarded_decision.value
        return payload


@dataclass(frozen=True)
class MatchedPairProfileSummary:
    profile: MatchedPairStressProfile
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    all_stress_unguarded_safe_accept_count: int
    all_stress_guarded_safe_accept_count: int
    profile_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        return payload


@dataclass(frozen=True)
class MatchedPairStressPanel:
    panel_role: str
    seed: int
    rectangle: InfluenceRectangle | None
    cases: tuple[MatchedPairStressCaseResult, ...]
    profile_summaries: tuple[MatchedPairProfileSummary, ...]
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    all_stress_unguarded_safe_accept_count: int
    all_stress_guarded_safe_accept_count: int
    full_protocol: bool
    panel_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_role": self.panel_role,
            "seed": self.seed,
            "rectangle": None if self.rectangle is None else self.rectangle.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "profile_summaries": [
                summary.to_dict() for summary in self.profile_summaries
            ],
            "case_count": self.case_count,
            "unguarded_harmful_outlier_false_safe_count": (
                self.unguarded_harmful_outlier_false_safe_count
            ),
            "guarded_harmful_outlier_false_safe_count": (
                self.guarded_harmful_outlier_false_safe_count
            ),
            "focus_unguarded_safe_accept_count": (
                self.focus_unguarded_safe_accept_count
            ),
            "focus_guarded_safe_accept_count": self.focus_guarded_safe_accept_count,
            "focus_safe_accept_retention": self.focus_safe_accept_retention,
            "all_stress_unguarded_safe_accept_count": (
                self.all_stress_unguarded_safe_accept_count
            ),
            "all_stress_guarded_safe_accept_count": (
                self.all_stress_guarded_safe_accept_count
            ),
            "full_protocol": self.full_protocol,
            "panel_gate_passed": self.panel_gate_passed,
        }


@dataclass(frozen=True)
class MatchedPairStressResult:
    artifact_schema: str
    role: str
    information_boundary: str
    frozen_predecessor: str
    calibration_a_seed: int
    calibration_b_seed: int
    final_held_out_seed: int
    reference_count: int
    repeats: int
    surface_sample_count: int
    point_counts: tuple[int, ...]
    stresses: tuple[SensorStress, ...]
    matched_pair_config: MatchedPairConfig
    stress_config: MatchedPairStressConfig
    profile_specs: tuple[MatchedPairStressSpec, ...]
    rectangle_selection_rule: str
    selected_rectangle: InfluenceRectangle | None
    calibration_a: MatchedPairStressPanel
    calibration_b: MatchedPairStressPanel
    final_held_out: MatchedPairStressPanel | None
    phase18_supported: bool
    correspondence_stress_synthetic_supported: bool
    real_correspondence_supported: bool
    real_paired_scan_supported: bool
    trimmed_reconstruction_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "role": self.role,
            "information_boundary": self.information_boundary,
            "frozen_predecessor": self.frozen_predecessor,
            "calibration_a_seed": self.calibration_a_seed,
            "calibration_b_seed": self.calibration_b_seed,
            "final_held_out_seed": self.final_held_out_seed,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "surface_sample_count": self.surface_sample_count,
            "point_counts": list(self.point_counts),
            "stresses": [stress.value for stress in self.stresses],
            "matched_pair_config": asdict(self.matched_pair_config),
            "stress_config": asdict(self.stress_config),
            "profile_specs": [
                {
                    **asdict(spec),
                    "profile": spec.profile.value,
                }
                for spec in self.profile_specs
            ],
            "rectangle_selection_rule": self.rectangle_selection_rule,
            "selected_rectangle": (
                None
                if self.selected_rectangle is None
                else self.selected_rectangle.to_dict()
            ),
            "calibration_a": self.calibration_a.to_dict(),
            "calibration_b": self.calibration_b.to_dict(),
            "final_held_out": (
                None
                if self.final_held_out is None
                else self.final_held_out.to_dict()
            ),
            "phase18_supported": self.phase18_supported,
            "correspondence_stress_synthetic_supported": (
                self.correspondence_stress_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def _rotation_matrix(axis: FloatArray, angle_degrees: float) -> FloatArray:
    angle = math.radians(angle_degrees)
    first, second, third = axis
    cross = np.asarray(
        (
            (0.0, -third, second),
            (third, 0.0, -first),
            (-second, first, 0.0),
        ),
        dtype=np.float64,
    )
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )


def perturb_matched_pairs(
    primary_points: FloatArray,
    repeat_points: FloatArray,
    spec: MatchedPairStressSpec,
    *,
    seed: int,
) -> PerturbedMatchedPairs:
    """Apply observed-only registration, mismatch, and missing-pair stress."""

    primary = np.asarray(primary_points, dtype=np.float64)
    repeat = np.asarray(repeat_points, dtype=np.float64)
    if primary.ndim != 2 or primary.shape[1] != 3:
        raise ValueError("primary_points must have shape (n, 3)")
    if repeat.shape != primary.shape:
        raise ValueError("repeat_points must initially align one-to-one")
    if primary.shape[0] < 4:
        raise ValueError("at least four pairs are required for stress")
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(repeat)):
        raise ValueError("matched points must be finite")

    rng = np.random.default_rng(seed)
    presented_repeat = repeat.copy()
    if spec.rotation_degrees > 0.0:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        center = np.mean(presented_repeat, axis=0)
        rotation = _rotation_matrix(axis, spec.rotation_degrees)
        presented_repeat = (presented_repeat - center) @ rotation.T + center
    else:
        axis = np.asarray((0.0, 0.0, 1.0))

    pair_count = primary.shape[0]
    primary_ids = np.arange(pair_count, dtype=np.int64)
    repeat_source_ids = primary_ids.copy()
    mismatch_count = int(round(spec.mismatch_fraction * pair_count))
    if spec.mismatch_fraction > 0.0:
        mismatch_count = max(2, mismatch_count)
        mismatch_indices = np.sort(
            rng.choice(pair_count, size=mismatch_count, replace=False)
        )
        shifted_sources = np.roll(mismatch_indices, 1)
        original_repeat = presented_repeat.copy()
        presented_repeat[mismatch_indices] = original_repeat[shifted_sources]
        repeat_source_ids[mismatch_indices] = shifted_sources

    missing_count = int(round(spec.missing_fraction * pair_count))
    if missing_count:
        missing_indices = rng.choice(pair_count, size=missing_count, replace=False)
        keep = np.ones(pair_count, dtype=bool)
        keep[missing_indices] = False
    else:
        keep = np.ones(pair_count, dtype=bool)
    retained_primary_ids = primary_ids[keep]
    retained_repeat_source_ids = repeat_source_ids[keep]
    presented_map = np.column_stack(
        (retained_primary_ids, retained_repeat_source_ids)
    ).astype("<i8", copy=False)
    digest = hashlib.sha256(presented_map.tobytes()).hexdigest()
    return PerturbedMatchedPairs(
        primary_points=primary[keep],
        repeat_points=presented_repeat[keep],
        primary_ids=retained_primary_ids,
        repeat_source_ids=retained_repeat_source_ids,
        retained_pair_count=int(np.sum(keep)),
        missing_pair_count=missing_count,
        mismatched_pair_count=int(
            np.sum(retained_primary_ids != retained_repeat_source_ids)
        ),
        rotation_degrees=spec.rotation_degrees,
        rotation_axis=tuple(float(value) for value in axis),
        presented_pair_map_sha256=digest,
    )


def _as_feature_array(
    values: Sequence[tuple[float, float]],
    *,
    name: str,
) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if not result.size:
        return np.empty((0, 2), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must contain peak/support pairs")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _accepted_count(
    features: FloatArray,
    peak_threshold: float,
    support_threshold: float,
) -> int:
    return int(
        np.sum(
            (features[:, 0] <= peak_threshold)
            & (features[:, 1] <= support_threshold)
        )
    )


def calibrate_profile_aware_rectangle(
    groups: Sequence[InfluenceFeatureCohort],
) -> InfluenceRectangle | None:
    """Select one zero-harm rectangle by worst cohort/profile retention."""

    harmful = tuple(
        _as_feature_array(group.harmful, name=f"group_{index}_harmful")
        for index, group in enumerate(groups)
    )
    focus_safe = tuple(
        _as_feature_array(group.focus_safe, name=f"group_{index}_focus_safe")
        for index, group in enumerate(groups)
    )
    all_safe = tuple(
        _as_feature_array(group.all_safe, name=f"group_{index}_all_safe")
        for index, group in enumerate(groups)
    )
    if not groups or any(not values.shape[0] for values in harmful):
        return None
    combined_harmful = np.vstack(harmful)
    peak_candidates = sorted(
        {math.inf}
        | {
            float(np.nextafter(value, -np.inf))
            for value in combined_harmful[:, 0].tolist()
        }
    )
    best: InfluenceRectangle | None = None
    best_key: tuple[float, int, float, int, float, float] | None = None
    for peak_threshold in peak_candidates:
        eligible = combined_harmful[
            combined_harmful[:, 0] <= peak_threshold,
            1,
        ]
        support_threshold = (
            math.inf
            if not eligible.size
            else float(np.nextafter(np.min(eligible), -np.inf))
        )
        focus_counts = tuple(
            _accepted_count(values, peak_threshold, support_threshold)
            for values in focus_safe
        )
        all_counts = tuple(
            _accepted_count(values, peak_threshold, support_threshold)
            for values in all_safe
        )
        focus_retentions = tuple(
            0.0 if not values.shape[0] else count / values.shape[0]
            for count, values in zip(focus_counts, focus_safe, strict=True)
        )
        all_retentions = tuple(
            0.0 if not values.shape[0] else count / values.shape[0]
            for count, values in zip(all_counts, all_safe, strict=True)
        )
        key = (
            min(focus_retentions),
            sum(focus_counts),
            min(all_retentions),
            sum(all_counts),
            peak_threshold,
            support_threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = InfluenceRectangle(
                peak_threshold=peak_threshold,
                support_threshold=support_threshold,
                retained_focus_safe_count=sum(focus_counts),
                retained_all_safe_count=sum(all_counts),
            )
    return best


def _features(row: MatchedPairStressRawCase) -> tuple[float, float]:
    return (
        row.evidence.peak_standardized_displacement,
        row.evidence.support_standardized_displacement,
    )


def _feature_group(
    rows: Sequence[MatchedPairStressRawCase],
) -> InfluenceFeatureCohort:
    harmful = tuple(
        _features(row)
        for row in rows
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
        and row.stress.is_outlier_stress
        and row.endpoint.geometry_topology_harm_present
    )
    safe_rows = tuple(
        row
        for row in rows
        if row.unguarded_decision is SamplingGateDecision.ACCEPT
        and not row.endpoint.geometry_topology_harm_present
    )
    return InfluenceFeatureCohort(
        harmful=harmful,
        focus_safe=tuple(
            _features(row)
            for row in safe_rows
            if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
        ),
        all_safe=tuple(_features(row) for row in safe_rows),
    )


def _raw_panel(
    *,
    point_counts: tuple[int, ...],
    stresses: tuple[SensorStress, ...],
    reference_count: int,
    repeats: int,
    seed: int,
    surface_sample_count: int,
    base_gate_config: SamplingSufficiencyConfig | None,
    shared_trend_config: SharedTrendConfig | None,
    matched_pair_config: MatchedPairConfig,
    stress_config: MatchedPairStressConfig,
    profile_specs: tuple[MatchedPairStressSpec, ...],
) -> tuple[MatchedPairStressRawCase, ...]:
    base_result = evaluate_sensor_stress(
        point_counts=point_counts,
        stresses=stresses,
        reference_count=reference_count,
        repeats=repeats,
        seed=seed,
        surface_sample_count=surface_sample_count,
        base_gate_config=base_gate_config,
        shared_trend_config=shared_trend_config,
    )
    harm_config = LocalSurfaceConsensusConfig(
        harmful_distance_fraction=matched_pair_config.harmful_distance_fraction
    )
    rows: list[MatchedPairStressRawCase] = []
    for case_row in base_result.cases:
        primary = make_sensor_stress_case(
            case_row.stress,
            case_row.point_count,
            reference_count=max(reference_count, case_row.point_count),
            seed=case_row.seed,
        )
        primary_construction, _ = construct_shared_trend_surface(
            primary.points,
            shared_trend_config,
        )
        replicate_seed = case_row.seed + matched_pair_config.replicate_seed_offset
        matched_repeat = make_matched_repeat_observation(
            primary.points,
            primary.point_component_labels,
            case_row.stress,
            seed=replicate_seed,
        )
        endpoint = evaluate_geometry_topology_harm(
            primary_construction.mesh,
            primary.reference_points,
            primary.point_component_labels,
            characteristic_length=primary.characteristic_length,
            config=harm_config,
        )
        for profile_index, spec in enumerate(profile_specs):
            perturbation_seed = (
                case_row.seed
                + stress_config.perturbation_seed_offset
                + profile_index * stress_config.profile_seed_stride
            )
            perturbed = perturb_matched_pairs(
                primary.points,
                matched_repeat.points,
                spec,
                seed=perturbation_seed,
            )
            evidence = replace(
                estimate_matched_pair_evidence(
                    perturbed.primary_points,
                    perturbed.repeat_points,
                    matched_pair_config,
                ),
                information_boundary=(
                    "ordered_primary_and_repeat_coordinates_with_presented_"
                    "pair_order_only; pairing_correctness_unknown_to_route"
                ),
            )
            local_spatial_evidence = estimate_local_spatial_displacement_evidence(
                perturbed.primary_points,
                perturbed.repeat_points,
                matched_pair_config,
            )
            if perturbed.missing_pair_count:
                matched_subset_construction, _ = construct_shared_trend_surface(
                    perturbed.primary_points,
                    shared_trend_config,
                )
                matched_subset_endpoint = evaluate_geometry_topology_harm(
                    matched_subset_construction.mesh,
                    primary.reference_points,
                    primary.point_component_labels[perturbed.primary_ids],
                    characteristic_length=primary.characteristic_length,
                    config=harm_config,
                )
                frozen_layer_ids = primary_construction.inference.layer_ids[
                    perturbed.primary_ids
                ]
                frozen_layer_counts = tuple(
                    int(np.sum(frozen_layer_ids == layer)) for layer in range(2)
                )
                frozen_evidence = replace(
                    primary_construction.inference.evidence,
                    information_boundary=(
                        "full_primary_observed_shared_trend_partition_"
                        "restricted_to_retained_ids"
                    ),
                    point_count=int(frozen_layer_ids.size),
                    cluster_sizes=frozen_layer_counts,
                    minimum_cluster_fraction=(
                        min(frozen_layer_counts) / frozen_layer_ids.size
                    ),
                )
                frozen_partition_construction = (
                    construct_two_layer_surface_from_inference(
                        perturbed.primary_points,
                        ParallelLayerInference(
                            layer_ids=frozen_layer_ids,
                            evidence=frozen_evidence,
                        ),
                    )
                )
                frozen_partition_endpoint = evaluate_geometry_topology_harm(
                    frozen_partition_construction.mesh,
                    primary.reference_points,
                    primary.point_component_labels[perturbed.primary_ids],
                    characteristic_length=primary.characteristic_length,
                    config=harm_config,
                )
            else:
                matched_subset_endpoint = endpoint
                frozen_partition_endpoint = endpoint
            rows.append(
                MatchedPairStressRawCase(
                    profile=spec.profile,
                    stress=case_row.stress,
                    point_count=case_row.point_count,
                    repeat=case_row.repeat,
                    seed=case_row.seed,
                    replicate_seed=replicate_seed,
                    perturbation_seed=perturbation_seed,
                    repeat_transient_outlier_count=int(
                        matched_repeat.transient_outlier_indices.size
                    ),
                    repeat_transient_outlier_index_sha256=(
                        matched_repeat.transient_outlier_index_sha256
                    ),
                    retained_pair_count=perturbed.retained_pair_count,
                    missing_pair_count=perturbed.missing_pair_count,
                    mismatched_pair_count=perturbed.mismatched_pair_count,
                    rotation_degrees=perturbed.rotation_degrees,
                    rotation_axis=perturbed.rotation_axis,
                    presented_pair_map_sha256=(
                        perturbed.presented_pair_map_sha256
                    ),
                    evidence=evidence,
                    endpoint=endpoint,
                    matched_subset_endpoint=matched_subset_endpoint,
                    frozen_partition_endpoint=frozen_partition_endpoint,
                    unguarded_decision=case_row.candidate_decision,
                    local_spatial_evidence=local_spatial_evidence,
                )
            )
    return tuple(rows)


def _profile_summary(
    rows: Sequence[MatchedPairStressCaseResult],
    profile: MatchedPairStressProfile,
    *,
    full_protocol: bool,
    rectangle: InfluenceRectangle | None,
) -> MatchedPairProfileSummary:
    selected = [row for row in rows if row.profile is profile]
    focus = [
        row
        for row in selected
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    retention = 0.0 if not unguarded_focus else guarded_focus / unguarded_focus
    unguarded_harm = sum(
        row.unguarded_harmful_outlier_false_safe for row in selected
    )
    guarded_harm = sum(
        row.guarded_harmful_outlier_false_safe for row in selected
    )
    all_safe = [row for row in selected if row.unguarded_safe_accept]
    return MatchedPairProfileSummary(
        profile=profile,
        case_count=len(selected),
        unguarded_harmful_outlier_false_safe_count=unguarded_harm,
        guarded_harmful_outlier_false_safe_count=guarded_harm,
        unguarded_provenance_violation_accept_count=sum(
            row.unguarded_provenance_violation_accept for row in selected
        ),
        guarded_provenance_violation_accept_count=sum(
            row.guarded_provenance_violation_accept for row in selected
        ),
        focus_unguarded_safe_accept_count=unguarded_focus,
        focus_guarded_safe_accept_count=guarded_focus,
        focus_safe_accept_retention=retention,
        all_stress_unguarded_safe_accept_count=len(all_safe),
        all_stress_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in all_safe
        ),
        profile_gate_passed=bool(
            full_protocol
            and rectangle is not None
            and unguarded_harm > 0
            and guarded_harm == 0
            and retention >= 0.90
        ),
    )


def _materialize_panel(
    raw_rows: tuple[MatchedPairStressRawCase, ...],
    *,
    panel_role: str,
    seed: int,
    rectangle: InfluenceRectangle | None,
    full_protocol: bool,
    profile_specs: tuple[MatchedPairStressSpec, ...],
) -> MatchedPairStressPanel:
    rows: list[MatchedPairStressCaseResult] = []
    for raw in raw_rows:
        unguarded_accept = raw.unguarded_decision is SamplingGateDecision.ACCEPT
        consistent = bool(
            rectangle is not None
            and raw.evidence.peak_standardized_displacement
            <= rectangle.peak_threshold
            and raw.evidence.support_standardized_displacement
            <= rectangle.support_threshold
        )
        guarded_decision = (
            SamplingGateDecision.UNSUPPORTED
            if unguarded_accept and not consistent
            else raw.unguarded_decision
        )
        guarded_accept = guarded_decision is SamplingGateDecision.ACCEPT
        harmful_outlier = bool(
            raw.stress.is_outlier_stress
            and raw.endpoint.geometry_topology_harm_present
        )
        rows.append(
            MatchedPairStressCaseResult(
                profile=raw.profile,
                stress=raw.stress,
                point_count=raw.point_count,
                repeat=raw.repeat,
                seed=raw.seed,
                replicate_seed=raw.replicate_seed,
                perturbation_seed=raw.perturbation_seed,
                repeat_transient_outlier_count=raw.repeat_transient_outlier_count,
                repeat_transient_outlier_index_sha256=(
                    raw.repeat_transient_outlier_index_sha256
                ),
                retained_pair_count=raw.retained_pair_count,
                missing_pair_count=raw.missing_pair_count,
                mismatched_pair_count=raw.mismatched_pair_count,
                rotation_degrees=raw.rotation_degrees,
                rotation_axis=raw.rotation_axis,
                presented_pair_map_sha256=raw.presented_pair_map_sha256,
                evidence=raw.evidence,
                endpoint=raw.endpoint,
                unguarded_decision=raw.unguarded_decision,
                guarded_decision=guarded_decision,
                unguarded_safe_accept=bool(
                    unguarded_accept
                    and not raw.endpoint.geometry_topology_harm_present
                ),
                guarded_safe_accept=bool(
                    guarded_accept
                    and not raw.endpoint.geometry_topology_harm_present
                ),
                unguarded_harmful_outlier_false_safe=bool(
                    unguarded_accept and harmful_outlier
                ),
                guarded_harmful_outlier_false_safe=bool(
                    guarded_accept and harmful_outlier
                ),
                unguarded_provenance_violation_accept=bool(
                    unguarded_accept
                    and raw.endpoint.provenance_violation_present
                ),
                guarded_provenance_violation_accept=bool(
                    guarded_accept
                    and raw.endpoint.provenance_violation_present
                ),
            )
        )
    profiles = tuple(spec.profile for spec in profile_specs)
    summaries = tuple(
        _profile_summary(
            rows,
            profile,
            full_protocol=full_protocol,
            rectangle=rectangle,
        )
        for profile in profiles
    )
    focus = [
        row
        for row in rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    all_safe = [row for row in rows if row.unguarded_safe_accept]
    return MatchedPairStressPanel(
        panel_role=panel_role,
        seed=seed,
        rectangle=rectangle,
        cases=tuple(rows),
        profile_summaries=summaries,
        case_count=len(rows),
        unguarded_harmful_outlier_false_safe_count=sum(
            row.unguarded_harmful_outlier_false_safe for row in rows
        ),
        guarded_harmful_outlier_false_safe_count=sum(
            row.guarded_harmful_outlier_false_safe for row in rows
        ),
        focus_unguarded_safe_accept_count=unguarded_focus,
        focus_guarded_safe_accept_count=guarded_focus,
        focus_safe_accept_retention=(
            0.0 if not unguarded_focus else guarded_focus / unguarded_focus
        ),
        all_stress_unguarded_safe_accept_count=len(all_safe),
        all_stress_guarded_safe_accept_count=sum(
            row.guarded_safe_accept for row in all_safe
        ),
        full_protocol=full_protocol,
        panel_gate_passed=bool(
            full_protocol
            and len(summaries) == len(profile_specs)
            and all(summary.profile_gate_passed for summary in summaries)
        ),
    )


def evaluate_matched_pair_stress(
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    stresses: Sequence[SensorStress | str] = DEFAULT_STRESSES,
    reference_count: int = 2048,
    repeats: int = 8,
    calibration_a_seed: int = CALIBRATION_A_SEED,
    calibration_b_seed: int = CALIBRATION_B_SEED,
    final_held_out_seed: int = FINAL_HELD_OUT_SEED,
    surface_sample_count: int = 256,
    base_gate_config: SamplingSufficiencyConfig | None = None,
    shared_trend_config: SharedTrendConfig | None = None,
    matched_pair_config: MatchedPairConfig | None = None,
    stress_config: MatchedPairStressConfig | None = None,
    profile_specs: Sequence[MatchedPairStressSpec] = DEFAULT_STRESS_SPECS,
) -> MatchedPairStressResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_matched = (
        MatchedPairConfig() if matched_pair_config is None else matched_pair_config
    )
    selected_stress = (
        MatchedPairStressConfig() if stress_config is None else stress_config
    )
    selected_profiles = tuple(profile_specs)
    if not selected_profiles or len(
        {spec.profile for spec in selected_profiles}
    ) != len(selected_profiles):
        raise ValueError("profile specs must be non-empty with unique profiles")
    seeds = (calibration_a_seed, calibration_b_seed, final_held_out_seed)
    if repeats < 1 or not selected_counts or not selected_stresses:
        raise ValueError("counts/stresses must be non-empty and repeats positive")
    if len(set(seeds)) != 3:
        raise ValueError("all calibration and held-out seeds must differ")
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and repeats == 8
        and reference_count == 2048
        and surface_sample_count == 256
        and seeds
        == (CALIBRATION_A_SEED, CALIBRATION_B_SEED, FINAL_HELD_OUT_SEED)
        and selected_matched == MatchedPairConfig()
        and selected_stress == MatchedPairStressConfig()
        and selected_profiles == DEFAULT_STRESS_SPECS
    )
    common = {
        "point_counts": selected_counts,
        "stresses": selected_stresses,
        "reference_count": reference_count,
        "repeats": repeats,
        "surface_sample_count": surface_sample_count,
        "base_gate_config": base_gate_config,
        "shared_trend_config": shared_trend_config,
        "matched_pair_config": selected_matched,
        "stress_config": selected_stress,
        "profile_specs": selected_profiles,
    }
    calibration_a_raw = _raw_panel(seed=calibration_a_seed, **common)
    calibration_b_raw = _raw_panel(seed=calibration_b_seed, **common)
    calibration_groups = tuple(
        _feature_group(
            [row for row in raw_rows if row.profile is spec.profile]
        )
        for raw_rows in (calibration_a_raw, calibration_b_raw)
        for spec in selected_profiles
    )
    rectangle = calibrate_profile_aware_rectangle(calibration_groups)
    calibration_a = _materialize_panel(
        calibration_a_raw,
        panel_role="calibration_a",
        seed=calibration_a_seed,
        rectangle=rectangle,
        full_protocol=full_protocol,
        profile_specs=selected_profiles,
    )
    calibration_b = _materialize_panel(
        calibration_b_raw,
        panel_role="calibration_b",
        seed=calibration_b_seed,
        rectangle=rectangle,
        full_protocol=full_protocol,
        profile_specs=selected_profiles,
    )
    calibration_passed = bool(
        calibration_a.panel_gate_passed and calibration_b.panel_gate_passed
    )
    final_held_out: MatchedPairStressPanel | None = None
    if calibration_passed:
        final_raw = _raw_panel(seed=final_held_out_seed, **common)
        final_held_out = _materialize_panel(
            final_raw,
            panel_role="final_held_out",
            seed=final_held_out_seed,
            rectangle=rectangle,
            full_protocol=full_protocol,
            profile_specs=selected_profiles,
        )
    supported = bool(
        calibration_passed
        and final_held_out is not None
        and final_held_out.panel_gate_passed
    )
    return MatchedPairStressResult(
        artifact_schema="pftf_alpha_matched_pair_stress_phase18/v1",
        role="matched_pair_correspondence_and_registration_stress_audit",
        information_boundary=(
            "route trusts presented pair order and coordinates; mismatch truth, "
            "missing IDs, source labels, stress, and endpoints are evaluation-only"
        ),
        frozen_predecessor="phase17_seeds_22400804_22500804_22600804_positive",
        calibration_a_seed=calibration_a_seed,
        calibration_b_seed=calibration_b_seed,
        final_held_out_seed=final_held_out_seed,
        reference_count=reference_count,
        repeats=repeats,
        surface_sample_count=surface_sample_count,
        point_counts=selected_counts,
        stresses=selected_stresses,
        matched_pair_config=selected_matched,
        stress_config=selected_stress,
        profile_specs=selected_profiles,
        rectangle_selection_rule=(
            "zero harm in every calibration cohort/profile group; maximize "
            "worst group focus retention, total focus, worst all-safe retention, "
            "total all-safe, then peak and support thresholds"
        ),
        selected_rectangle=rectangle,
        calibration_a=calibration_a,
        calibration_b=calibration_b,
        final_held_out=final_held_out,
        phase18_supported=supported,
        correspondence_stress_synthetic_supported=supported,
        real_correspondence_supported=False,
        real_paired_scan_supported=False,
        trimmed_reconstruction_supported=False,
        deployment_supported=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--surface-samples", type=int, default=256)
    parser.add_argument("--calibration-a-seed", type=int, default=CALIBRATION_A_SEED)
    parser.add_argument("--calibration-b-seed", type=int, default=CALIBRATION_B_SEED)
    parser.add_argument(
        "--final-held-out-seed",
        type=int,
        default=FINAL_HELD_OUT_SEED,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_matched_pair_stress(
        reference_count=args.reference,
        repeats=args.repeats,
        calibration_a_seed=args.calibration_a_seed,
        calibration_b_seed=args.calibration_b_seed,
        final_held_out_seed=args.final_held_out_seed,
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
