"""Tangential pair-confidence filtering for the frozen Phase-19 audit."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .local_insertion_influence import InfluenceRectangle
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
from .matched_pair_stress import (
    DEFAULT_STRESS_SPECS,
    MatchedPairStressConfig,
    MatchedPairStressProfile,
    MatchedPairStressSpec,
    perturb_matched_pairs,
)
from .sampling_gate import SamplingGateDecision, SamplingSufficiencyConfig
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

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CALIBRATION_A_SEED = 23000804
CALIBRATION_B_SEED = 23100804
FINAL_HELD_OUT_SEED = 23200804
PHASE18_FINAL_HELD_OUT_SEED = 22900804
FROZEN_PHASE18_RECTANGLE = InfluenceRectangle(
    peak_threshold=10.922625244331805,
    support_threshold=math.inf,
    retained_focus_safe_count=259,
    retained_all_safe_count=725,
)


@dataclass(frozen=True)
class PairConfidenceConfig:
    local_neighbor_count: int = 12
    alignment_trim_fraction: float = 0.20
    alignment_iterations: int = 3
    minimum_spacing_fraction: float = 0.002
    correct_pair_retention_gate: float = 0.99
    minimum_filtered_pair_count: int = 8

    def __post_init__(self) -> None:
        if self.local_neighbor_count < 3:
            raise ValueError("local_neighbor_count must be at least three")
        if not 0.0 <= self.alignment_trim_fraction < 0.5:
            raise ValueError("alignment_trim_fraction must lie in [0, 0.5)")
        if self.alignment_iterations < 1:
            raise ValueError("alignment_iterations must be positive")
        if not math.isfinite(self.minimum_spacing_fraction):
            raise ValueError("minimum_spacing_fraction must be finite")
        if self.minimum_spacing_fraction <= 0.0:
            raise ValueError("minimum_spacing_fraction must be positive")
        if not 0.0 < self.correct_pair_retention_gate <= 1.0:
            raise ValueError("correct_pair_retention_gate must lie in (0, 1]")
        if self.minimum_filtered_pair_count < 2:
            raise ValueError("minimum_filtered_pair_count must be at least two")


@dataclass(frozen=True)
class PairConfidenceScores:
    scores: FloatArray
    tangential_residuals: FloatArray
    local_tangent_spacings: FloatArray
    aligned_repeat_points: FloatArray
    alignment_rotation_degrees: float
    alignment_translation: tuple[float, float, float]
    alignment_inlier_count: int
    observed_characteristic_length: float


@dataclass(frozen=True)
class PairScoreCohort:
    profile: MatchedPairStressProfile
    scores: FloatArray
    correct_mask: BoolArray
    mismatch_required: bool


@dataclass(frozen=True)
class PairCutoffCalibration:
    cutoff: float
    minimum_mismatch_score: float
    presented_correct_pair_count: int
    retained_correct_pair_count: int
    presented_mismatch_pair_count: int
    retained_mismatch_pair_count: int
    correct_pair_retention: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TangentialPairRawCase:
    profile: MatchedPairStressProfile
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    perturbation_seed: int
    repeat_transient_outlier_count: int
    repeat_transient_outlier_index_sha256: str
    missing_pair_count: int
    rotation_degrees: float
    rotation_axis: tuple[float, float, float]
    presented_pair_map_sha256: str
    primary_ids: NDArray[np.int64]
    repeat_source_ids: NDArray[np.int64]
    primary_points: FloatArray
    repeat_points: FloatArray
    correct_mask: BoolArray
    confidence: PairConfidenceScores
    endpoint: GeometryTopologyHarmEndpoint
    unguarded_decision: SamplingGateDecision


@dataclass(frozen=True)
class TangentialPairCaseResult:
    profile: MatchedPairStressProfile
    stress: SensorStress
    point_count: int
    repeat: int
    seed: int
    replicate_seed: int
    perturbation_seed: int
    repeat_transient_outlier_count: int
    repeat_transient_outlier_index_sha256: str
    missing_pair_count: int
    rotation_degrees: float
    rotation_axis: tuple[float, float, float]
    presented_pair_map_sha256: str
    presented_pair_count: int
    presented_correct_pair_count: int
    presented_mismatch_pair_count: int
    filtered_pair_count: int
    retained_correct_pair_count: int
    retained_mismatch_pair_count: int
    correct_pair_retention: float
    pair_score_minimum: float
    pair_score_median: float
    pair_score_percentile95: float
    pair_score_maximum: float
    minimum_truth_mismatch_score: float | None
    maximum_truth_correct_score: float
    alignment_rotation_degrees: float
    alignment_translation: tuple[float, float, float]
    alignment_inlier_count: int
    evidence: MatchedPairEvidence | None
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
class TangentialPairProfileSummary:
    profile: MatchedPairStressProfile
    case_count: int
    presented_correct_pair_count: int
    retained_correct_pair_count: int
    correct_pair_retention: float
    presented_mismatch_pair_count: int
    retained_mismatch_pair_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    unguarded_provenance_violation_accept_count: int
    guarded_provenance_violation_accept_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    pair_gate_passed: bool
    profile_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile"] = self.profile.value
        return payload


@dataclass(frozen=True)
class TangentialPairPanel:
    panel_role: str
    seed: int
    pair_cutoff: float
    frozen_rectangle: InfluenceRectangle
    cases: tuple[TangentialPairCaseResult, ...]
    profile_summaries: tuple[TangentialPairProfileSummary, ...]
    case_count: int
    unguarded_harmful_outlier_false_safe_count: int
    guarded_harmful_outlier_false_safe_count: int
    focus_unguarded_safe_accept_count: int
    focus_guarded_safe_accept_count: int
    focus_safe_accept_retention: float
    full_protocol: bool
    panel_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_role": self.panel_role,
            "seed": self.seed,
            "pair_cutoff": self.pair_cutoff,
            "frozen_rectangle": self.frozen_rectangle.to_dict(),
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
            "full_protocol": self.full_protocol,
            "panel_gate_passed": self.panel_gate_passed,
        }


@dataclass(frozen=True)
class TangentialPairConfidenceResult:
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
    confidence_config: PairConfidenceConfig
    profile_specs: tuple[MatchedPairStressSpec, ...]
    frozen_rectangle: InfluenceRectangle
    pair_cutoff_calibration: PairCutoffCalibration | None
    calibration_a: TangentialPairPanel | None
    calibration_b: TangentialPairPanel | None
    final_held_out: TangentialPairPanel | None
    phase19_supported: bool
    tangential_pair_confidence_synthetic_supported: bool
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
            "confidence_config": asdict(self.confidence_config),
            "profile_specs": [
                {**asdict(spec), "profile": spec.profile.value}
                for spec in self.profile_specs
            ],
            "frozen_rectangle": self.frozen_rectangle.to_dict(),
            "pair_cutoff_calibration": (
                None
                if self.pair_cutoff_calibration is None
                else self.pair_cutoff_calibration.to_dict()
            ),
            "calibration_a": (
                None if self.calibration_a is None else self.calibration_a.to_dict()
            ),
            "calibration_b": (
                None if self.calibration_b is None else self.calibration_b.to_dict()
            ),
            "final_held_out": (
                None if self.final_held_out is None else self.final_held_out.to_dict()
            ),
            "phase19_supported": self.phase19_supported,
            "tangential_pair_confidence_synthetic_supported": (
                self.tangential_pair_confidence_synthetic_supported
            ),
            "real_correspondence_supported": self.real_correspondence_supported,
            "real_paired_scan_supported": self.real_paired_scan_supported,
            "trimmed_reconstruction_supported": (
                self.trimmed_reconstruction_supported
            ),
            "deployment_supported": self.deployment_supported,
        }


def _as_points(values: FloatArray, *, name: str) -> FloatArray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if points.shape[0] < 4:
        raise ValueError(f"{name} must contain at least four points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _fit_rigid_transform(
    primary: FloatArray,
    repeat: FloatArray,
    indices: NDArray[np.int64],
) -> tuple[FloatArray, FloatArray]:
    selected_primary = primary[indices]
    selected_repeat = repeat[indices]
    primary_center = np.mean(selected_primary, axis=0)
    repeat_center = np.mean(selected_repeat, axis=0)
    centered_primary = selected_primary - primary_center
    centered_repeat = selected_repeat - repeat_center
    left, _, right_transpose = np.linalg.svd(
        centered_repeat.T @ centered_primary
    )
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transpose[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    translation = primary_center - repeat_center @ rotation.T
    return rotation, translation


def _robust_rigid_alignment(
    primary: FloatArray,
    repeat: FloatArray,
    config: PairConfidenceConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64]]:
    count = primary.shape[0]
    retained = max(3, int(math.ceil((1.0 - config.alignment_trim_fraction) * count)))
    inliers = np.arange(count, dtype=np.int64)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    aligned = repeat.copy()
    for _ in range(config.alignment_iterations):
        rotation, translation = _fit_rigid_transform(primary, repeat, inliers)
        aligned = repeat @ rotation.T + translation
        residuals = np.linalg.norm(primary - aligned, axis=1)
        inliers = np.argsort(residuals, kind="stable")[:retained].astype(
            np.int64,
            copy=False,
        )
    return aligned, rotation, translation, inliers


def _local_normals_and_spacings(
    primary: FloatArray,
    *,
    neighbor_count: int,
    minimum_spacing: float,
) -> tuple[FloatArray, FloatArray]:
    count = primary.shape[0]
    selected_count = min(neighbor_count, count - 1)
    _, neighbor_rows = cKDTree(primary).query(primary, k=selected_count + 1)
    neighbor_rows = np.asarray(neighbor_rows, dtype=np.int64)
    normals = np.empty_like(primary)
    spacings = np.empty(count, dtype=np.float64)
    for index in range(count):
        indices = neighbor_rows[index]
        indices = indices[indices != index][:selected_count]
        neighborhood = np.vstack((primary[index], primary[indices]))
        centered = neighborhood - np.mean(neighborhood, axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        normal = vectors[:, 0]
        normal /= max(float(np.linalg.norm(normal)), np.finfo(float).eps)
        offsets = primary[indices] - primary[index]
        tangent = offsets - np.outer(offsets @ normal, normal)
        distances = np.linalg.norm(tangent, axis=1)
        positive = distances[distances > np.finfo(float).eps]
        spacing = minimum_spacing if not positive.size else float(np.median(positive))
        normals[index] = normal
        spacings[index] = max(spacing, minimum_spacing)
    return normals, spacings


def tangential_pair_confidence_scores(
    primary_points: FloatArray,
    repeat_points: FloatArray,
    config: PairConfidenceConfig | None = None,
) -> PairConfidenceScores:
    """Score presented pairs without consuming pair correctness or source truth."""

    selected = PairConfidenceConfig() if config is None else config
    primary = _as_points(primary_points, name="primary_points")
    repeat = _as_points(repeat_points, name="repeat_points")
    if repeat.shape != primary.shape:
        raise ValueError("repeat_points must match primary_points shape")
    aligned, rotation, translation, inliers = _robust_rigid_alignment(
        primary,
        repeat,
        selected,
    )
    pooled = np.vstack((primary, aligned))
    characteristic_length = max(
        float(np.linalg.norm(np.ptp(pooled, axis=0))),
        np.finfo(float).eps,
    )
    normals, spacings = _local_normals_and_spacings(
        primary,
        neighbor_count=selected.local_neighbor_count,
        minimum_spacing=selected.minimum_spacing_fraction * characteristic_length,
    )
    residuals = primary - aligned
    tangential = residuals - np.sum(residuals * normals, axis=1)[:, None] * normals
    tangential_norms = np.linalg.norm(tangential, axis=1)
    scores = tangential_norms / spacings
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return PairConfidenceScores(
        scores=scores,
        tangential_residuals=tangential_norms,
        local_tangent_spacings=spacings,
        aligned_repeat_points=aligned,
        alignment_rotation_degrees=math.degrees(math.acos(cosine)),
        alignment_translation=tuple(float(value) for value in translation),
        alignment_inlier_count=int(inliers.size),
        observed_characteristic_length=characteristic_length,
    )


def calibrate_pair_confidence_cutoff(
    groups: Sequence[PairScoreCohort],
) -> PairCutoffCalibration | None:
    """Choose the largest cutoff strictly below every calibration mismatch."""

    if not groups:
        return None
    mismatch_scores: list[FloatArray] = []
    correct_parts: list[FloatArray] = []
    for group in groups:
        scores = np.asarray(group.scores, dtype=np.float64)
        correct = np.asarray(group.correct_mask, dtype=np.bool_)
        if scores.ndim != 1 or correct.shape != scores.shape:
            raise ValueError("pair-score groups must contain aligned 1D arrays")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
            raise ValueError("pair confidence scores must be finite and non-negative")
        mismatches = scores[~correct]
        if group.mismatch_required and not mismatches.size:
            return None
        if mismatches.size:
            mismatch_scores.append(mismatches)
        correct_parts.append(scores[correct])
    if not mismatch_scores:
        return None
    all_mismatches = np.concatenate(mismatch_scores)
    all_correct = np.concatenate(correct_parts)
    minimum_mismatch = float(np.min(all_mismatches))
    cutoff = float(np.nextafter(minimum_mismatch, -np.inf))
    retained_correct = int(np.sum(all_correct <= cutoff))
    retained_mismatch = int(np.sum(all_mismatches <= cutoff))
    return PairCutoffCalibration(
        cutoff=cutoff,
        minimum_mismatch_score=minimum_mismatch,
        presented_correct_pair_count=int(all_correct.size),
        retained_correct_pair_count=retained_correct,
        presented_mismatch_pair_count=int(all_mismatches.size),
        retained_mismatch_pair_count=retained_mismatch,
        correct_pair_retention=(
            0.0 if not all_correct.size else retained_correct / all_correct.size
        ),
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
    confidence_config: PairConfidenceConfig,
    profile_specs: tuple[MatchedPairStressSpec, ...],
) -> tuple[TangentialPairRawCase, ...]:
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
    rows: list[TangentialPairRawCase] = []
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
        endpoint = evaluate_geometry_topology_harm(
            primary_construction.mesh,
            primary.reference_points,
            primary.point_component_labels,
            characteristic_length=primary.characteristic_length,
            config=harm_config,
        )
        replicate_seed = case_row.seed + matched_pair_config.replicate_seed_offset
        matched_repeat = make_matched_repeat_observation(
            primary.points,
            primary.point_component_labels,
            case_row.stress,
            seed=replicate_seed,
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
            confidence = tangential_pair_confidence_scores(
                perturbed.primary_points,
                perturbed.repeat_points,
                confidence_config,
            )
            rows.append(
                TangentialPairRawCase(
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
                    missing_pair_count=perturbed.missing_pair_count,
                    rotation_degrees=perturbed.rotation_degrees,
                    rotation_axis=perturbed.rotation_axis,
                    presented_pair_map_sha256=perturbed.presented_pair_map_sha256,
                    primary_ids=perturbed.primary_ids,
                    repeat_source_ids=perturbed.repeat_source_ids,
                    primary_points=perturbed.primary_points,
                    repeat_points=perturbed.repeat_points,
                    correct_mask=(
                        perturbed.primary_ids == perturbed.repeat_source_ids
                    ),
                    confidence=confidence,
                    endpoint=endpoint,
                    unguarded_decision=case_row.candidate_decision,
                )
            )
    return tuple(rows)


def _calibration_groups(
    panel_rows: Sequence[tuple[str, Sequence[TangentialPairRawCase]]],
    profile_specs: tuple[MatchedPairStressSpec, ...],
) -> tuple[PairScoreCohort, ...]:
    groups: list[PairScoreCohort] = []
    for _, rows in panel_rows:
        for spec in profile_specs:
            selected = [row for row in rows if row.profile is spec.profile]
            groups.append(
                PairScoreCohort(
                    profile=spec.profile,
                    scores=np.concatenate(
                        [row.confidence.scores for row in selected]
                    ),
                    correct_mask=np.concatenate(
                        [row.correct_mask for row in selected]
                    ),
                    mismatch_required=spec.mismatch_fraction > 0.0,
                )
            )
    return tuple(groups)


def _materialize_case(
    raw: TangentialPairRawCase,
    *,
    pair_cutoff: float,
    rectangle: InfluenceRectangle,
    confidence_config: PairConfidenceConfig,
    matched_pair_config: MatchedPairConfig,
) -> TangentialPairCaseResult:
    keep = raw.confidence.scores <= pair_cutoff
    filtered_count = int(np.sum(keep))
    evidence: MatchedPairEvidence | None = None
    consistent = False
    if filtered_count >= confidence_config.minimum_filtered_pair_count:
        evidence = replace(
            estimate_matched_pair_evidence(
                raw.primary_points[keep],
                raw.repeat_points[keep],
                matched_pair_config,
            ),
            information_boundary=(
                "presented_pairs_retained_by_observed_tangential_confidence_"
                "only; pairing_truth_unknown_to_route"
            ),
        )
        consistent = bool(
            evidence.peak_standardized_displacement <= rectangle.peak_threshold
            and evidence.support_standardized_displacement
            <= rectangle.support_threshold
        )
    unguarded_accept = raw.unguarded_decision is SamplingGateDecision.ACCEPT
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
    correct_count = int(np.sum(raw.correct_mask))
    retained_correct = int(np.sum(keep & raw.correct_mask))
    scores = raw.confidence.scores
    return TangentialPairCaseResult(
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
        missing_pair_count=raw.missing_pair_count,
        rotation_degrees=raw.rotation_degrees,
        rotation_axis=raw.rotation_axis,
        presented_pair_map_sha256=raw.presented_pair_map_sha256,
        presented_pair_count=int(scores.size),
        presented_correct_pair_count=correct_count,
        presented_mismatch_pair_count=int(np.sum(~raw.correct_mask)),
        filtered_pair_count=filtered_count,
        retained_correct_pair_count=retained_correct,
        retained_mismatch_pair_count=int(np.sum(keep & ~raw.correct_mask)),
        correct_pair_retention=(
            0.0 if not correct_count else retained_correct / correct_count
        ),
        pair_score_minimum=float(np.min(scores)),
        pair_score_median=float(np.median(scores)),
        pair_score_percentile95=float(np.percentile(scores, 95.0)),
        pair_score_maximum=float(np.max(scores)),
        minimum_truth_mismatch_score=(
            None
            if np.all(raw.correct_mask)
            else float(np.min(scores[~raw.correct_mask]))
        ),
        maximum_truth_correct_score=float(np.max(scores[raw.correct_mask])),
        alignment_rotation_degrees=raw.confidence.alignment_rotation_degrees,
        alignment_translation=raw.confidence.alignment_translation,
        alignment_inlier_count=raw.confidence.alignment_inlier_count,
        evidence=evidence,
        endpoint=raw.endpoint,
        unguarded_decision=raw.unguarded_decision,
        guarded_decision=guarded_decision,
        unguarded_safe_accept=bool(
            unguarded_accept and not raw.endpoint.geometry_topology_harm_present
        ),
        guarded_safe_accept=bool(
            guarded_accept and not raw.endpoint.geometry_topology_harm_present
        ),
        unguarded_harmful_outlier_false_safe=bool(
            unguarded_accept and harmful_outlier
        ),
        guarded_harmful_outlier_false_safe=bool(
            guarded_accept and harmful_outlier
        ),
        unguarded_provenance_violation_accept=bool(
            unguarded_accept and raw.endpoint.provenance_violation_present
        ),
        guarded_provenance_violation_accept=bool(
            guarded_accept and raw.endpoint.provenance_violation_present
        ),
    )


def _profile_summary(
    rows: Sequence[TangentialPairCaseResult],
    spec: MatchedPairStressSpec,
    *,
    full_protocol: bool,
    confidence_config: PairConfidenceConfig,
) -> TangentialPairProfileSummary:
    selected = [row for row in rows if row.profile is spec.profile]
    correct = sum(row.presented_correct_pair_count for row in selected)
    retained_correct = sum(row.retained_correct_pair_count for row in selected)
    mismatches = sum(row.presented_mismatch_pair_count for row in selected)
    retained_mismatches = sum(row.retained_mismatch_pair_count for row in selected)
    correct_retention = 0.0 if not correct else retained_correct / correct
    focus = [
        row
        for row in selected
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    focus_retention = (
        0.0 if not unguarded_focus else guarded_focus / unguarded_focus
    )
    unguarded_harm = sum(
        row.unguarded_harmful_outlier_false_safe for row in selected
    )
    guarded_harm = sum(
        row.guarded_harmful_outlier_false_safe for row in selected
    )
    pair_gate = bool(
        correct_retention >= confidence_config.correct_pair_retention_gate
        and retained_mismatches == 0
        and (spec.mismatch_fraction == 0.0 or mismatches > 0)
    )
    return TangentialPairProfileSummary(
        profile=spec.profile,
        case_count=len(selected),
        presented_correct_pair_count=correct,
        retained_correct_pair_count=retained_correct,
        correct_pair_retention=correct_retention,
        presented_mismatch_pair_count=mismatches,
        retained_mismatch_pair_count=retained_mismatches,
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
        focus_safe_accept_retention=focus_retention,
        pair_gate_passed=pair_gate,
        profile_gate_passed=bool(
            full_protocol
            and pair_gate
            and unguarded_harm > 0
            and guarded_harm == 0
            and focus_retention >= 0.90
        ),
    )


def _materialize_panel(
    raw_rows: tuple[TangentialPairRawCase, ...],
    *,
    panel_role: str,
    seed: int,
    pair_cutoff: float,
    rectangle: InfluenceRectangle,
    confidence_config: PairConfidenceConfig,
    matched_pair_config: MatchedPairConfig,
    profile_specs: tuple[MatchedPairStressSpec, ...],
    full_protocol: bool,
) -> TangentialPairPanel:
    rows = tuple(
        _materialize_case(
            raw,
            pair_cutoff=pair_cutoff,
            rectangle=rectangle,
            confidence_config=confidence_config,
            matched_pair_config=matched_pair_config,
        )
        for raw in raw_rows
    )
    summaries = tuple(
        _profile_summary(
            rows,
            spec,
            full_protocol=full_protocol,
            confidence_config=confidence_config,
        )
        for spec in profile_specs
    )
    focus = [
        row
        for row in rows
        if row.stress in (SensorStress.CONTROL, SensorStress.LOCAL_BUMP)
    ]
    unguarded_focus = sum(row.unguarded_safe_accept for row in focus)
    guarded_focus = sum(row.guarded_safe_accept for row in focus)
    return TangentialPairPanel(
        panel_role=panel_role,
        seed=seed,
        pair_cutoff=pair_cutoff,
        frozen_rectangle=rectangle,
        cases=rows,
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
        full_protocol=full_protocol,
        panel_gate_passed=bool(
            full_protocol
            and len(summaries) == len(profile_specs)
            and all(summary.profile_gate_passed for summary in summaries)
        ),
    )


def evaluate_tangential_pair_confidence(
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
    confidence_config: PairConfidenceConfig | None = None,
    profile_specs: Sequence[MatchedPairStressSpec] = DEFAULT_STRESS_SPECS,
) -> TangentialPairConfidenceResult:
    selected_counts = tuple(int(value) for value in point_counts)
    selected_stresses = tuple(SensorStress(value) for value in stresses)
    selected_matched = (
        MatchedPairConfig() if matched_pair_config is None else matched_pair_config
    )
    selected_stress = (
        MatchedPairStressConfig() if stress_config is None else stress_config
    )
    selected_confidence = (
        PairConfidenceConfig() if confidence_config is None else confidence_config
    )
    selected_profiles = tuple(profile_specs)
    seeds = (calibration_a_seed, calibration_b_seed, final_held_out_seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("calibration and final seeds must differ")
    if PHASE18_FINAL_HELD_OUT_SEED in seeds:
        raise ValueError("Phase-18 unopened final seed must not be reused")
    if reference_count < 1 or repeats < 1 or surface_sample_count < 1:
        raise ValueError("panel sizes must be positive")
    if not selected_counts or min(selected_counts) < 4:
        raise ValueError("point_counts must contain values of at least four")
    if not selected_stresses or not selected_profiles:
        raise ValueError("stresses and profile_specs must not be empty")
    if len({spec.profile for spec in selected_profiles}) != len(selected_profiles):
        raise ValueError("stress profiles must be unique")
    full_protocol = bool(
        selected_counts == DEFAULT_POINT_COUNTS
        and selected_stresses == DEFAULT_STRESSES
        and selected_profiles == DEFAULT_STRESS_SPECS
        and repeats >= 8
        and reference_count >= 2048
        and surface_sample_count >= 256
        and seeds
        == (CALIBRATION_A_SEED, CALIBRATION_B_SEED, FINAL_HELD_OUT_SEED)
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
        "confidence_config": selected_confidence,
        "profile_specs": selected_profiles,
    }
    raw_a = _raw_panel(seed=calibration_a_seed, **common)
    raw_b = _raw_panel(seed=calibration_b_seed, **common)
    cutoff_calibration = calibrate_pair_confidence_cutoff(
        _calibration_groups(
            (("calibration_a", raw_a), ("calibration_b", raw_b)),
            selected_profiles,
        )
    )
    panel_a: TangentialPairPanel | None = None
    panel_b: TangentialPairPanel | None = None
    final_panel: TangentialPairPanel | None = None
    if cutoff_calibration is not None:
        panel_a = _materialize_panel(
            raw_a,
            panel_role="calibration_a",
            seed=calibration_a_seed,
            pair_cutoff=cutoff_calibration.cutoff,
            rectangle=FROZEN_PHASE18_RECTANGLE,
            confidence_config=selected_confidence,
            matched_pair_config=selected_matched,
            profile_specs=selected_profiles,
            full_protocol=full_protocol,
        )
        panel_b = _materialize_panel(
            raw_b,
            panel_role="calibration_b",
            seed=calibration_b_seed,
            pair_cutoff=cutoff_calibration.cutoff,
            rectangle=FROZEN_PHASE18_RECTANGLE,
            confidence_config=selected_confidence,
            matched_pair_config=selected_matched,
            profile_specs=selected_profiles,
            full_protocol=full_protocol,
        )
        calibration_passed = bool(
            panel_a.panel_gate_passed and panel_b.panel_gate_passed
        )
        if calibration_passed:
            raw_final = _raw_panel(seed=final_held_out_seed, **common)
            final_panel = _materialize_panel(
                raw_final,
                panel_role="final_held_out",
                seed=final_held_out_seed,
                pair_cutoff=cutoff_calibration.cutoff,
                rectangle=FROZEN_PHASE18_RECTANGLE,
                confidence_config=selected_confidence,
                matched_pair_config=selected_matched,
                profile_specs=selected_profiles,
                full_protocol=full_protocol,
            )
    supported = bool(
        panel_a is not None
        and panel_b is not None
        and panel_a.panel_gate_passed
        and panel_b.panel_gate_passed
        and final_panel is not None
        and final_panel.panel_gate_passed
    )
    return TangentialPairConfidenceResult(
        artifact_schema="pftf_alpha_tangential_pair_confidence_phase19/v1",
        role="synthetic_supervised_tangential_pair_confidence_audit",
        information_boundary=(
            "route uses presented coordinates and row pairing only; pair-source "
            "truth calibrates/evaluates cutoff and endpoints remain evaluation-only"
        ),
        frozen_predecessor=(
            "phase18_correspondence_stress_negative_final_seed_unopened"
        ),
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
        confidence_config=selected_confidence,
        profile_specs=selected_profiles,
        frozen_rectangle=FROZEN_PHASE18_RECTANGLE,
        pair_cutoff_calibration=cutoff_calibration,
        calibration_a=panel_a,
        calibration_b=panel_b,
        final_held_out=final_panel,
        phase19_supported=supported,
        tangential_pair_confidence_synthetic_supported=supported,
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
    parser.add_argument("--final-seed", type=int, default=FINAL_HELD_OUT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_tangential_pair_confidence(
        reference_count=args.reference,
        repeats=args.repeats,
        calibration_a_seed=args.calibration_a_seed,
        calibration_b_seed=args.calibration_b_seed,
        final_held_out_seed=args.final_seed,
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
