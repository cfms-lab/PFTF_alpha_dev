"""Pre-flight diagnostic for the P3 resampling-consensus bridge-excision idea.

This module answers one question cheaply, before the full P3 ablation is built:
on the six-case calibration panel, at a frozen P2 operating point, are labeled
cross-component (bridge) boundary cells *less resampling-persistent* than
same-component boundary cells?

Per-cell resampling persistence is the genuinely new primitive P3 would rely on.
It does not exist elsewhere: B3 only computes a scalar global stability term.
Component labels are used here **only** to score the diagnostic separation (AUC);
they never affect the persistence signal, the frozen operating point, or the
resample draws. This is an evaluation-only probe and never a promotion result.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .adaptive import (
    boundary_bridge_localization,
    pftf_confidence_fallback_filtration,
)
from .baselines import BaselineID, BenchmarkConfig, _binary_auc
from .calibration import (
    calibrate_adaptive_multiplier,
    calibrate_p2_confidence_threshold,
)
from .filtration import AlphaFiltration
from .synthetic import PanelSplit, SyntheticCase, make_minimal_panel

_KEEP_SEED_STRIDE = 100_003


def resampled_index_sets(
    point_count: int,
    *,
    keep_fraction: float,
    repeats: int,
    seed: int,
    min_points: int = 5,
) -> tuple[np.ndarray, ...]:
    """Deterministic retained-index subsets (the base indices, unlike B3).

    ``baselines._resampled_point_sets`` discards the base indices it draws.  The
    persistence signal needs them to map a resample's selected cells back to base
    tetrahedra, so this returns the retained index arrays directly.
    """

    if point_count < min_points:
        raise ValueError("point_count is too small to resample")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1]")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    kept = max(min_points, int(round(keep_fraction * point_count)))
    sample_size = min(point_count, kept)
    rng = np.random.default_rng(seed)
    subsets: list[np.ndarray] = []
    for _ in range(repeats):
        indices = np.sort(rng.choice(point_count, size=sample_size, replace=False))
        subsets.append(np.ascontiguousarray(indices, dtype=np.int64))
    return tuple(subsets)


def _p2_adaptive(
    points: np.ndarray, *, config: BenchmarkConfig, confidence_threshold: float
):
    filtration = AlphaFiltration.from_points(points)
    return pftf_confidence_fallback_filtration(
        filtration,
        k_neighbors=config.adaptive_k_neighbors,
        relation_gain=config.p1_relation_gain,
        max_condition_number=config.p1_max_condition_number,
        density_contrast_scale=config.p1_density_contrast_scale,
        receiver_imbalance_weight=config.p1_receiver_imbalance_weight,
        confidence_threshold=confidence_threshold,
    )


def _resample_selection(
    points: np.ndarray,
    base_indices: np.ndarray,
    *,
    config: BenchmarkConfig,
    confidence_threshold: float,
    multiplier: float,
) -> tuple[frozenset[int], set[frozenset[int]]]:
    """Rebuild P2 on one subset; return (present base indices, selected base cells)."""

    adaptive = _p2_adaptive(
        points, config=config, confidence_threshold=confidence_threshold
    )
    selected = adaptive.scores <= multiplier
    present = frozenset(int(index) for index in base_indices)
    selected_cells: set[frozenset[int]] = set()
    for cell in adaptive.top_simplices[selected]:
        selected_cells.add(frozenset(int(base_indices[vertex]) for vertex in cell))
    return present, selected_cells


@dataclass(frozen=True)
class _OwnerRow:
    persistence: float | None
    mixed: bool
    flagged: bool


def _case_owner_rows(
    case: SyntheticCase,
    *,
    config: BenchmarkConfig,
    confidence_threshold: float,
    multiplier: float,
    keep_fraction: float,
    repeats: int,
    min_evaluations: int,
    seed: int,
) -> list[_OwnerRow]:
    base = _p2_adaptive(
        case.points, config=config, confidence_threshold=confidence_threshold
    )
    try:
        localization = boundary_bridge_localization(
            base,
            scale_multiplier=multiplier,
            k_neighbors=config.adaptive_k_neighbors,
            normal_coherence_threshold=config.bridge_probe_normal_coherence_threshold,
            normal_edge_threshold=config.bridge_probe_normal_edge_threshold,
            length_edge_threshold=config.bridge_probe_length_edge_threshold,
        )
    except ValueError:
        return []
    if localization.owner_cell_indices.size == 0:
        return []

    max_risk_by_owner: dict[int, float] = {}
    for owner, risk in zip(
        localization.owner_cell_indices.tolist(),
        localization.boundary_face_risk.tolist(),
        strict=True,
    ):
        owner_index = int(owner)
        max_risk_by_owner[owner_index] = max(
            max_risk_by_owner.get(owner_index, 0.0), float(risk)
        )

    subsets = resampled_index_sets(
        case.points.shape[0],
        keep_fraction=keep_fraction,
        repeats=repeats,
        seed=seed,
    )
    resample_selections = [
        _resample_selection(
            case.points[indices],
            indices,
            config=config,
            confidence_threshold=confidence_threshold,
            multiplier=multiplier,
        )
        for indices in subsets
    ]

    labels = case.point_component_labels
    rows: list[_OwnerRow] = []
    for owner_index, max_risk in max_risk_by_owner.items():
        cell = base.top_simplices[owner_index]
        cell_set = frozenset(int(vertex) for vertex in cell)
        mixed = len({int(labels[vertex]) for vertex in cell}) > 1
        evaluable = 0
        supported = 0
        for present, selected_cells in resample_selections:
            if cell_set <= present:
                evaluable += 1
                if cell_set in selected_cells:
                    supported += 1
        persistence = (
            None if evaluable < min_evaluations else supported / evaluable
        )
        rows.append(_OwnerRow(persistence, mixed, max_risk > 1.0))
    return rows


@dataclass(frozen=True)
class KeepFractionSummary:
    keep_fraction: float
    owner_count: int
    defined_persistence_count: int
    coverage_fraction: float
    mixed_owner_count: int
    same_owner_count: int
    mean_persistence_mixed: float | None
    mean_persistence_same: float | None
    instability_auc_all: float | None
    flagged_owner_count: int
    flagged_mixed_count: int
    instability_auc_flagged: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _summarize(rows: list[_OwnerRow], keep_fraction: float) -> KeepFractionSummary:
    defined = [row for row in rows if row.persistence is not None]
    persistence = np.asarray([row.persistence for row in defined], dtype=np.float64)
    mixed = np.asarray([row.mixed for row in defined], dtype=np.bool_)
    flagged = np.asarray([row.flagged for row in defined], dtype=np.bool_)
    instability = 1.0 - persistence
    mixed_persistence = persistence[mixed]
    same_persistence = persistence[~mixed]
    flagged_instability = instability[flagged]
    flagged_mixed = mixed[flagged]
    return KeepFractionSummary(
        keep_fraction=keep_fraction,
        owner_count=len(rows),
        defined_persistence_count=len(defined),
        coverage_fraction=(len(defined) / len(rows)) if rows else 0.0,
        mixed_owner_count=int(np.count_nonzero(mixed)),
        same_owner_count=int(np.count_nonzero(~mixed)),
        mean_persistence_mixed=(
            float(np.mean(mixed_persistence)) if mixed_persistence.size else None
        ),
        mean_persistence_same=(
            float(np.mean(same_persistence)) if same_persistence.size else None
        ),
        instability_auc_all=(
            None if not defined else _binary_auc(instability, mixed)
        ),
        flagged_owner_count=int(np.count_nonzero(flagged)),
        flagged_mixed_count=int(np.count_nonzero(flagged_mixed)),
        instability_auc_flagged=(
            None
            if flagged_instability.size == 0
            else _binary_auc(flagged_instability, flagged_mixed)
        ),
    )


@dataclass(frozen=True)
class BridgePersistenceProbeResult:
    point_count: int
    reference_count: int
    repeats: int
    min_evaluations: int
    seed: int
    confidence_threshold: float
    p2_multiplier: float
    keep_summaries: tuple[KeepFractionSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": "pftf_alpha_bridge_persistence_probe/v1",
            "evaluation_role": "diagnostic_only",
            "signal_inputs": "observed_points_and_frozen_calibration_only",
            "labels_role": "diagnostic_auc_only_never_signal",
            "promotion_supported": False,
            "point_count": self.point_count,
            "reference_count": self.reference_count,
            "repeats": self.repeats,
            "min_evaluations": self.min_evaluations,
            "seed": self.seed,
            "confidence_threshold": self.confidence_threshold,
            "p2_multiplier": self.p2_multiplier,
            "keep_summaries": [summary.to_dict() for summary in self.keep_summaries],
        }


def run_bridge_persistence_probe(
    *,
    point_count: int = 80,
    reference_count: int = 2048,
    candidate_budget: int = 12,
    repeats: int = 8,
    keep_fractions: Sequence[float] = (0.90, 0.75),
    min_evaluations: int = 3,
    target_fallback_fraction: float = 0.25,
    seed: int = 20_260_724,
    verbose: bool = False,
) -> BridgePersistenceProbeResult:
    """Freeze a P2 operating point, then measure per-cell persistence separation."""

    config = BenchmarkConfig(seed=seed)
    cases = make_minimal_panel(
        split=PanelSplit.CALIBRATION,
        point_count=point_count,
        reference_count=reference_count,
        seed=seed,
    )
    frozen = replace(
        config,
        b4_scale_multiplier=None,
        b5_scale_multiplier=None,
        p1_scale_multiplier=None,
        p2_scale_multiplier=None,
    )
    confidence = calibrate_p2_confidence_threshold(
        cases, config=frozen, target_fallback_fraction=target_fallback_fraction
    )
    frozen = replace(frozen, p2_confidence_threshold=confidence.threshold)
    multiplier = calibrate_adaptive_multiplier(
        cases,
        BaselineID.P2_CONFIDENCE_FALLBACK,
        config=frozen,
        candidate_budget=candidate_budget,
    ).multiplier
    if verbose:
        print(
            f"[probe] frozen confidence={confidence.threshold:.6f} "
            f"p2_multiplier={multiplier:.6f}",
            flush=True,
        )

    summaries: list[KeepFractionSummary] = []
    for keep_index, keep_fraction in enumerate(keep_fractions):
        rows: list[_OwnerRow] = []
        for case in cases:
            rows.extend(
                _case_owner_rows(
                    case,
                    config=frozen,
                    confidence_threshold=confidence.threshold,
                    multiplier=multiplier,
                    keep_fraction=keep_fraction,
                    repeats=repeats,
                    min_evaluations=min_evaluations,
                    seed=seed + case.seed + 700_000 + keep_index * _KEEP_SEED_STRIDE,
                )
            )
        summary = _summarize(rows, keep_fraction)
        summaries.append(summary)
        if verbose:
            print(
                f"[probe] keep={keep_fraction:.2f} "
                f"coverage={summary.coverage_fraction:.3f} "
                f"auc_all={summary.instability_auc_all} "
                f"auc_flagged={summary.instability_auc_flagged}",
                flush=True,
            )
    return BridgePersistenceProbeResult(
        point_count=point_count,
        reference_count=reference_count,
        repeats=repeats,
        min_evaluations=min_evaluations,
        seed=seed,
        confidence_threshold=confidence.threshold,
        p2_multiplier=multiplier,
        keep_summaries=tuple(summaries),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the P3 bridge-persistence pre-flight diagnostic."
    )
    parser.add_argument("--point-count", type=int, default=80)
    parser.add_argument("--reference-count", type=int, default=2048)
    parser.add_argument("--candidate-budget", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--min-evaluations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20_260_724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/bridge_persistence_probe.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_bridge_persistence_probe(
        point_count=args.point_count,
        reference_count=args.reference_count,
        candidate_budget=args.candidate_budget,
        repeats=args.repeats,
        min_evaluations=args.min_evaluations,
        seed=args.seed,
        verbose=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
