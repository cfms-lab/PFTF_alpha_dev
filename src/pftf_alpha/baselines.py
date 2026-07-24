"""B0-P2 benchmark runners with explicit selection-information boundaries."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from time import perf_counter

import numpy as np

from .adaptive import (
    AdaptiveCellFiltration,
    density_scaled_filtration,
    pca_anisotropic_filtration,
    pftf_confidence_fallback_filtration,
    pftf_local_metric_filtration,
)
from .filtration import AlphaFiltration
from .selection import (
    AlphaEvaluation,
    ObjectiveTerms,
    ObjectiveWeights,
    select_best_alpha,
)
from .surface import (
    SurfaceEndpointMetrics,
    SurfaceMesh,
    alpha_surface,
    convex_hull_surface,
    evaluate_surface,
    mesh_statistics,
    sample_triangle_mesh,
    surface_distance_metrics,
)
from .synthetic import SyntheticCase


class BaselineID(StrEnum):
    B0_CONVEX_HULL = "B0"
    B1_FIXED_ALPHA = "B1"
    B2_CRITICAL_ORACLE = "B2"
    B3_PERSISTENCE_STABILITY = "B3"
    B4_DENSITY_SCALED = "B4"
    B5_PCA_ANISOTROPIC = "B5"
    P1_PFTF_LOCAL_SPD = "P1"
    P2_CONFIDENCE_FALLBACK = "P2"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Shared budgets and declared objective weights."""

    fixed_alpha_radius_fraction: float = 0.12
    surface_sample_count: int = 256
    fscore_threshold_fraction: float = 0.025
    resample_fraction: float = 0.90
    resample_repeats: int = 2
    b3_candidate_budget: int = 24
    adaptive_k_neighbors: int = 12
    b4_scale_multiplier: float | None = None
    b5_scale_multiplier: float | None = None
    b5_max_normal_penalty: float = 4.0
    p1_scale_multiplier: float | None = None
    p1_relation_gain: float = 2.0
    p1_max_condition_number: float = 9.0
    p1_density_contrast_scale: float = 0.5
    p1_receiver_imbalance_weight: float = 0.5
    p2_scale_multiplier: float | None = None
    p2_confidence_threshold: float = 0.5
    b2_weights: ObjectiveWeights = ObjectiveWeights(
        geometry=1.0,
        topology=1.0,
        stability=0.0,
        complexity=0.05,
    )
    b3_weights: ObjectiveWeights = ObjectiveWeights(
        geometry=1.0,
        topology=0.5,
        stability=1.0,
        complexity=0.05,
    )
    adaptive_weights: ObjectiveWeights = ObjectiveWeights(
        geometry=1.0,
        topology=1.0,
        stability=0.0,
        complexity=0.05,
    )
    seed: int = 0

    def __post_init__(self) -> None:
        fractions = {
            "fixed_alpha_radius_fraction": self.fixed_alpha_radius_fraction,
            "fscore_threshold_fraction": self.fscore_threshold_fraction,
            "resample_fraction": self.resample_fraction,
        }
        for name, value in fractions.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.resample_fraction > 1.0:
            raise ValueError("resample_fraction cannot exceed 1")
        for name, value in {
            "surface_sample_count": self.surface_sample_count,
            "resample_repeats": self.resample_repeats,
            "b3_candidate_budget": self.b3_candidate_budget,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.adaptive_k_neighbors < 3:
            raise ValueError("adaptive_k_neighbors must be at least three")
        for name, value in {
            "b4_scale_multiplier": self.b4_scale_multiplier,
            "b5_scale_multiplier": self.b5_scale_multiplier,
            "p1_scale_multiplier": self.p1_scale_multiplier,
            "p2_scale_multiplier": self.p2_scale_multiplier,
        }.items():
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be None or finite and positive")
        if (
            not math.isfinite(self.b5_max_normal_penalty)
            or self.b5_max_normal_penalty < 1.0
        ):
            raise ValueError("b5_max_normal_penalty must be finite and at least one")
        for name, value in {
            "p1_relation_gain": self.p1_relation_gain,
            "p1_density_contrast_scale": self.p1_density_contrast_scale,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.p1_max_condition_number)
            or self.p1_max_condition_number < 1.0
        ):
            raise ValueError("p1_max_condition_number must be finite and at least one")
        if not math.isfinite(self.p1_receiver_imbalance_weight) or (
            self.p1_receiver_imbalance_weight < 0.0
        ):
            raise ValueError("p1_receiver_imbalance_weight must be non-negative")
        if not math.isfinite(self.p2_confidence_threshold) or not (
            0.0 <= self.p2_confidence_threshold <= 1.0
        ):
            raise ValueError("p2_confidence_threshold must lie in [0, 1]")


@dataclass(frozen=True)
class BaselineResult:
    method: BaselineID
    selection_mode: str
    uses_reference_for_selection: bool
    alpha_squared: float | None
    alpha_radius_fraction: float | None
    total_candidates_scanned: int
    candidate_alpha_squared_min: float | None
    candidate_alpha_squared_max: float | None
    candidate_count: int
    objective_total: float | None
    objective_terms: ObjectiveTerms | None
    endpoints: SurfaceEndpointMetrics
    runtime_seconds: float
    selection_parameter_name: str | None = None
    selection_parameter_value: float | None = None
    candidate_parameter_min: float | None = None
    candidate_parameter_max: float | None = None
    method_diagnostics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "selection_mode": self.selection_mode,
            "uses_reference_for_selection": self.uses_reference_for_selection,
            "alpha_squared": self.alpha_squared,
            "alpha_radius_fraction": self.alpha_radius_fraction,
            "total_candidates_scanned": self.total_candidates_scanned,
            "candidate_alpha_squared_min": self.candidate_alpha_squared_min,
            "candidate_alpha_squared_max": self.candidate_alpha_squared_max,
            "candidate_count": self.candidate_count,
            "objective_total": self.objective_total,
            "objective_terms": (
                None if self.objective_terms is None else asdict(self.objective_terms)
            ),
            "endpoints": self.endpoints.to_dict(),
            "runtime_seconds": self.runtime_seconds,
            "selection_parameter_name": self.selection_parameter_name,
            "selection_parameter_value": self.selection_parameter_value,
            "candidate_parameter_min": self.candidate_parameter_min,
            "candidate_parameter_max": self.candidate_parameter_max,
            "method_diagnostics": self.method_diagnostics,
        }


@dataclass(frozen=True)
class CaseBenchmark:
    family: str
    split: str
    seed: int
    point_count: int
    reference_count: int
    expected_components: int
    characteristic_length: float
    expected_surface_betti: tuple[int, int, int]
    variation: dict[str, float]
    results: tuple[BaselineResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "split": self.split,
            "seed": self.seed,
            "point_count": self.point_count,
            "reference_count": self.reference_count,
            "expected_components": self.expected_components,
            "characteristic_length": self.characteristic_length,
            "expected_surface_betti": list(self.expected_surface_betti),
            "variation": self.variation,
            "results": [result.to_dict() for result in self.results],
        }


def _endpoint_metrics(
    mesh: SurfaceMesh,
    case: SyntheticCase,
    config: BenchmarkConfig,
    *,
    seed_offset: int,
) -> SurfaceEndpointMetrics:
    return evaluate_surface(
        mesh,
        case.reference_points,
        expected_components=case.expected_components,
        expected_betti=case.expected_surface_betti,
        characteristic_length=case.characteristic_length,
        sample_count=config.surface_sample_count,
        threshold_fraction=config.fscore_threshold_fraction,
        seed=config.seed + case.seed + seed_offset,
    )


def _b0(case: SyntheticCase, config: BenchmarkConfig) -> BaselineResult:
    started = perf_counter()
    mesh = convex_hull_surface(case.points)
    endpoints = _endpoint_metrics(mesh, case, config, seed_offset=100)
    return BaselineResult(
        method=BaselineID.B0_CONVEX_HULL,
        selection_mode="not_applicable",
        uses_reference_for_selection=False,
        alpha_squared=None,
        alpha_radius_fraction=None,
        total_candidates_scanned=1,
        candidate_alpha_squared_min=None,
        candidate_alpha_squared_max=None,
        candidate_count=1,
        objective_total=None,
        objective_terms=None,
        endpoints=endpoints,
        runtime_seconds=perf_counter() - started,
    )


def _b1(
    filtration: AlphaFiltration,
    case: SyntheticCase,
    config: BenchmarkConfig,
) -> BaselineResult:
    started = perf_counter()
    alpha_radius = config.fixed_alpha_radius_fraction * case.characteristic_length
    alpha_squared = alpha_radius * alpha_radius
    mesh = alpha_surface(filtration, alpha_squared)
    endpoints = _endpoint_metrics(mesh, case, config, seed_offset=200)
    return BaselineResult(
        method=BaselineID.B1_FIXED_ALPHA,
        selection_mode="fixed_normalized_radius",
        uses_reference_for_selection=False,
        alpha_squared=alpha_squared,
        alpha_radius_fraction=config.fixed_alpha_radius_fraction,
        total_candidates_scanned=1,
        candidate_alpha_squared_min=alpha_squared,
        candidate_alpha_squared_max=alpha_squared,
        candidate_count=1,
        objective_total=None,
        objective_terms=None,
        endpoints=endpoints,
        runtime_seconds=perf_counter() - started,
    )


def _b2(
    filtration: AlphaFiltration,
    case: SyntheticCase,
    config: BenchmarkConfig,
) -> BaselineResult:
    """Exhaustive top-simplex critical scan using dense reference labels."""

    started = perf_counter()
    candidates = filtration.critical_values(dimensions=[filtration.ambient_dimension])
    if candidates.size == 0:
        raise ValueError("B2 requires at least one top-simplex critical value")

    rows: list[tuple[float, SurfaceEndpointMetrics]] = []
    maximum_faces = 1
    for index, alpha_squared in enumerate(candidates):
        mesh = alpha_surface(filtration, float(alpha_squared))
        endpoints = _endpoint_metrics(
            mesh,
            case,
            config,
            seed_offset=10_000 + index,
        )
        maximum_faces = max(maximum_faces, endpoints.faces)
        rows.append((float(alpha_squared), endpoints))

    evaluations: list[AlphaEvaluation] = []
    for alpha_squared, endpoints in rows:
        nonmanifold_fraction = endpoints.nonmanifold_edges / max(endpoints.edges, 1)
        terms = ObjectiveTerms(
            geometry=(
                endpoints.normalized_chamfer_squared + endpoints.normalized_hausdorff
            ),
            topology=float(endpoints.component_error),
            stability=0.0,
            complexity=(nonmanifold_fraction + endpoints.faces / maximum_faces),
        )
        evaluations.append(
            AlphaEvaluation(
                alpha_squared=alpha_squared,
                terms=terms,
                total=config.b2_weights.apply(terms),
                statistics=filtration.statistics(alpha_squared),
            )
        )

    selected = select_best_alpha(evaluations)
    selected_index = next(
        index
        for index, (alpha_squared, _) in enumerate(rows)
        if alpha_squared == selected.alpha_squared
    )
    endpoints = rows[selected_index][1]
    return BaselineResult(
        method=BaselineID.B2_CRITICAL_ORACLE,
        selection_mode="dense_reference_oracle",
        uses_reference_for_selection=True,
        alpha_squared=selected.alpha_squared,
        candidate_count=len(evaluations),
        alpha_radius_fraction=(
            math.sqrt(selected.alpha_squared) / case.characteristic_length
        ),
        total_candidates_scanned=len(evaluations),
        candidate_alpha_squared_min=float(candidates[0]),
        candidate_alpha_squared_max=float(candidates[-1]),
        objective_total=selected.total,
        objective_terms=selected.terms,
        endpoints=endpoints,
        runtime_seconds=perf_counter() - started,
    )


def _candidate_indices(
    total: int,
    budget: int,
    *,
    priority: Iterable[int] = (),
) -> np.ndarray:
    if total <= budget:
        return np.arange(total, dtype=np.int64)

    chosen: list[int] = []
    priority_budget = max(1, budget // 2)
    for index in priority:
        candidate = int(index)
        if 0 <= candidate < total and candidate not in chosen:
            chosen.append(candidate)
        if len(chosen) >= priority_budget:
            break
    even_indices = np.unique(
        np.rint(np.linspace(0, total - 1, num=budget)).astype(np.int64)
    )
    for index in even_indices:
        candidate = int(index)
        if candidate not in chosen:
            chosen.append(candidate)
        if len(chosen) >= budget:
            break
    return np.asarray(sorted(chosen), dtype=np.int64)


def _plateau_persistence(
    candidates: np.ndarray,
    signatures: list[tuple[int, int]],
) -> np.ndarray:
    """Normalized log-radius width of each contiguous topology plateau."""

    radii = np.sqrt(candidates)
    log_radii = np.log(np.maximum(radii, np.finfo(np.float64).tiny))
    widths = np.zeros(candidates.shape[0], dtype=np.float64)
    start = 0
    while start < len(signatures):
        end = start
        while end + 1 < len(signatures) and signatures[end + 1] == signatures[start]:
            end += 1
        if end + 1 < len(signatures):
            lower = log_radii[start]
            upper = log_radii[end + 1]
            widths[start : end + 1] = max(upper - lower, 0.0)
        start = end + 1
    maximum = float(np.max(widths))
    if maximum <= 0.0:
        return np.zeros_like(widths)
    return widths / maximum


def _unlabeled_geometry_loss(
    mesh: SurfaceMesh,
    observed_points: np.ndarray,
    case: SyntheticCase,
    config: BenchmarkConfig,
    *,
    seed: int,
) -> tuple[float, np.ndarray]:
    sampled = sample_triangle_mesh(mesh, config.surface_sample_count, seed=seed)
    if sampled.shape[0] == 0:
        return 4.0, sampled
    distances = surface_distance_metrics(
        sampled,
        observed_points,
        threshold=(config.fscore_threshold_fraction * case.characteristic_length),
    )
    return (
        distances.chamfer_squared / case.characteristic_length**2,
        sampled,
    )


def _stability_loss(
    full_samples: np.ndarray,
    resampled_meshes: Iterable[SurfaceMesh],
    case: SyntheticCase,
    config: BenchmarkConfig,
    *,
    seed: int,
) -> float:
    if full_samples.shape[0] == 0:
        return 4.0
    losses: list[float] = []
    for repeat, mesh in enumerate(resampled_meshes):
        samples = sample_triangle_mesh(
            mesh,
            config.surface_sample_count,
            seed=seed + repeat + 1,
        )
        if samples.shape[0] == 0:
            losses.append(4.0)
            continue
        distances = surface_distance_metrics(
            full_samples,
            samples,
            threshold=(config.fscore_threshold_fraction * case.characteristic_length),
        )
        losses.append(distances.chamfer_squared / case.characteristic_length**2)
    return float(np.mean(losses))


def _resampled_filtrations(
    case: SyntheticCase,
    config: BenchmarkConfig,
) -> tuple[AlphaFiltration, ...]:
    rng = np.random.default_rng(config.seed + case.seed + 30_000)
    sample_size = max(4, int(round(config.resample_fraction * case.points.shape[0])))
    result: list[AlphaFiltration] = []
    for _ in range(config.resample_repeats):
        indices = rng.choice(case.points.shape[0], size=sample_size, replace=False)
        result.append(AlphaFiltration.from_points(case.points[indices]))
    return tuple(result)


def _b3(
    filtration: AlphaFiltration,
    case: SyntheticCase,
    config: BenchmarkConfig,
) -> BaselineResult:
    """Unlabeled topology-persistence and resampling-stability selection."""

    started = perf_counter()
    all_candidates = filtration.critical_values(
        dimensions=[filtration.ambient_dimension]
    )
    if all_candidates.size == 0:
        raise ValueError("B3 requires at least one top-simplex critical value")

    signatures: list[tuple[int, int]] = []
    for alpha_squared in all_candidates:
        statistics = mesh_statistics(alpha_surface(filtration, float(alpha_squared)))
        signatures.append(
            (
                statistics.connected_components,
                statistics.euler_characteristic,
            )
        )
    persistence = _plateau_persistence(all_candidates, signatures)
    priority_indices: list[int] = []
    plateau_start = 0
    while plateau_start < len(signatures):
        plateau_end = plateau_start
        while (
            plateau_end + 1 < len(signatures)
            and signatures[plateau_end + 1] == signatures[plateau_start]
        ):
            plateau_end += 1
        representative = (plateau_start + plateau_end) // 2
        if persistence[representative] > 0.0:
            priority_indices.append(representative)
        plateau_start = plateau_end + 1
    priority_indices.sort(key=lambda index: persistence[index], reverse=True)
    selected_indices = _candidate_indices(
        all_candidates.shape[0],
        config.b3_candidate_budget,
        priority=priority_indices,
    )
    resampled = _resampled_filtrations(case, config)

    materialized: list[tuple[int, float, SurfaceMesh, float, np.ndarray]] = []
    maximum_faces = 1
    for local_index, candidate_index in enumerate(selected_indices):
        alpha_squared = float(all_candidates[candidate_index])
        mesh = alpha_surface(filtration, alpha_squared)
        statistics = mesh_statistics(mesh)
        maximum_faces = max(maximum_faces, statistics.faces)
        geometry_loss, full_samples = _unlabeled_geometry_loss(
            mesh,
            case.points,
            case,
            config,
            seed=config.seed + case.seed + 40_000 + local_index,
        )
        materialized.append(
            (
                int(candidate_index),
                alpha_squared,
                mesh,
                geometry_loss,
                full_samples,
            )
        )

    evaluations: list[AlphaEvaluation] = []
    for local_index, (
        candidate_index,
        alpha_squared,
        mesh,
        geometry_loss,
        full_samples,
    ) in enumerate(materialized):
        statistics = mesh_statistics(mesh)
        resampled_meshes = (
            alpha_surface(resampled_filtration, alpha_squared)
            for resampled_filtration in resampled
        )
        stability_loss = _stability_loss(
            full_samples,
            resampled_meshes,
            case,
            config,
            seed=config.seed + case.seed + 50_000 + local_index,
        )
        terms = ObjectiveTerms(
            geometry=geometry_loss,
            topology=float(1.0 - persistence[candidate_index]),
            stability=stability_loss,
            complexity=(
                statistics.nonmanifold_edges / max(statistics.edges, 1)
                + statistics.faces / maximum_faces
            ),
        )
        evaluations.append(
            AlphaEvaluation(
                alpha_squared=alpha_squared,
                terms=terms,
                total=config.b3_weights.apply(terms),
                statistics=filtration.statistics(alpha_squared),
            )
        )

    selected = select_best_alpha(evaluations)
    selected_mesh = next(
        mesh
        for _, alpha_squared, mesh, _, _ in materialized
        if alpha_squared == selected.alpha_squared
    )
    endpoints = _endpoint_metrics(selected_mesh, case, config, seed_offset=60_000)
    return BaselineResult(
        method=BaselineID.B3_PERSISTENCE_STABILITY,
        selection_mode="unlabeled_persistence_resampling",
        uses_reference_for_selection=False,
        alpha_squared=selected.alpha_squared,
        candidate_count=len(evaluations),
        alpha_radius_fraction=(
            math.sqrt(selected.alpha_squared) / case.characteristic_length
        ),
        total_candidates_scanned=int(all_candidates.shape[0]),
        candidate_alpha_squared_min=float(all_candidates[0]),
        candidate_alpha_squared_max=float(all_candidates[-1]),
        objective_total=selected.total,
        objective_terms=selected.terms,
        endpoints=endpoints,
        runtime_seconds=perf_counter() - started,
    )


def _adaptive_objective_terms(
    endpoints: SurfaceEndpointMetrics,
    *,
    maximum_faces: int,
) -> ObjectiveTerms:
    nonmanifold_fraction = endpoints.nonmanifold_edges / max(endpoints.edges, 1)
    return ObjectiveTerms(
        geometry=(
            endpoints.normalized_chamfer_squared + endpoints.normalized_hausdorff
        ),
        topology=float(endpoints.component_error),
        stability=0.0,
        complexity=nonmanifold_fraction + endpoints.faces / maximum_faces,
    )


def _adaptive_baseline(
    adaptive: AdaptiveCellFiltration,
    method: BaselineID,
    case: SyntheticCase,
    config: BenchmarkConfig,
    *,
    fixed_multiplier: float | None,
    seed_offset: int,
    started: float,
) -> BaselineResult:
    """Select a B4/B5/P1/P2 multiplier or apply one frozen value."""

    critical_values = adaptive.critical_values()
    if critical_values.size == 0:
        raise ValueError(f"{method.value} has no adaptive critical values")
    if fixed_multiplier is None:
        candidates = critical_values
        selection_mode = "dense_reference_adaptive_oracle"
        uses_reference = True
    else:
        candidates = np.asarray([fixed_multiplier], dtype=np.float64)
        selection_mode = "frozen_local_scale_multiplier"
        uses_reference = False

    rows: list[tuple[float, SurfaceEndpointMetrics]] = []
    maximum_faces = 1
    for index, multiplier in enumerate(candidates):
        mesh = adaptive.surface_at(float(multiplier))
        endpoints = _endpoint_metrics(
            mesh,
            case,
            config,
            seed_offset=seed_offset + index,
        )
        maximum_faces = max(maximum_faces, endpoints.faces)
        rows.append((float(multiplier), endpoints))

    scored: list[tuple[float, ObjectiveTerms, float, SurfaceEndpointMetrics]] = []
    for multiplier, endpoints in rows:
        terms = _adaptive_objective_terms(endpoints, maximum_faces=maximum_faces)
        scored.append(
            (
                multiplier,
                terms,
                config.adaptive_weights.apply(terms),
                endpoints,
            )
        )
    selected_multiplier, selected_terms, selected_total, selected_endpoints = min(
        scored,
        key=lambda row: (row[2], row[1].complexity, row[0]),
    )
    return BaselineResult(
        method=method,
        selection_mode=selection_mode,
        uses_reference_for_selection=uses_reference,
        alpha_squared=None,
        alpha_radius_fraction=None,
        total_candidates_scanned=len(candidates),
        candidate_alpha_squared_min=None,
        candidate_alpha_squared_max=None,
        candidate_count=len(candidates),
        objective_total=selected_total,
        objective_terms=selected_terms,
        endpoints=selected_endpoints,
        runtime_seconds=perf_counter() - started,
        selection_parameter_name="local_scale_multiplier",
        selection_parameter_value=selected_multiplier,
        candidate_parameter_min=float(candidates[0]),
        candidate_parameter_max=float(candidates[-1]),
        method_diagnostics=adaptive.diagnostics_at(selected_multiplier),
    )


def _b4(
    filtration: AlphaFiltration,
    case: SyntheticCase,
    config: BenchmarkConfig,
) -> BaselineResult:
    started = perf_counter()
    adaptive = density_scaled_filtration(
        filtration, k_neighbors=config.adaptive_k_neighbors
    )
    return _adaptive_baseline(
        adaptive,
        BaselineID.B4_DENSITY_SCALED,
        case,
        config,
        fixed_multiplier=config.b4_scale_multiplier,
        seed_offset=70_000,
        started=started,
    )


def _b5(
    filtration: AlphaFiltration,
    case: SyntheticCase,
    config: BenchmarkConfig,
) -> BaselineResult:
    started = perf_counter()
    adaptive = pca_anisotropic_filtration(
        filtration,
        k_neighbors=config.adaptive_k_neighbors,
        max_normal_penalty=config.b5_max_normal_penalty,
    )
    return _adaptive_baseline(
        adaptive,
        BaselineID.B5_PCA_ANISOTROPIC,
        case,
        config,
        fixed_multiplier=config.b5_scale_multiplier,
        seed_offset=80_000,
        started=started,
    )


def _p1(
    filtration: AlphaFiltration,
    case: SyntheticCase,
    config: BenchmarkConfig,
) -> BaselineResult:
    started = perf_counter()
    adaptive = pftf_local_metric_filtration(
        filtration,
        k_neighbors=config.adaptive_k_neighbors,
        relation_gain=config.p1_relation_gain,
        max_condition_number=config.p1_max_condition_number,
        density_contrast_scale=config.p1_density_contrast_scale,
        receiver_imbalance_weight=config.p1_receiver_imbalance_weight,
    )
    return _adaptive_baseline(
        adaptive,
        BaselineID.P1_PFTF_LOCAL_SPD,
        case,
        config,
        fixed_multiplier=config.p1_scale_multiplier,
        seed_offset=90_000,
        started=started,
    )


def _p2(
    filtration: AlphaFiltration,
    case: SyntheticCase,
    config: BenchmarkConfig,
) -> BaselineResult:
    started = perf_counter()
    adaptive = pftf_confidence_fallback_filtration(
        filtration,
        k_neighbors=config.adaptive_k_neighbors,
        relation_gain=config.p1_relation_gain,
        max_condition_number=config.p1_max_condition_number,
        density_contrast_scale=config.p1_density_contrast_scale,
        receiver_imbalance_weight=config.p1_receiver_imbalance_weight,
        confidence_threshold=config.p2_confidence_threshold,
    )
    return _adaptive_baseline(
        adaptive,
        BaselineID.P2_CONFIDENCE_FALLBACK,
        case,
        config,
        fixed_multiplier=config.p2_scale_multiplier,
        seed_offset=100_000,
        started=started,
    )


def run_case_benchmarks(
    case: SyntheticCase,
    *,
    config: BenchmarkConfig | None = None,
    methods: Iterable[BaselineID | str] = tuple(BaselineID),
) -> CaseBenchmark:
    """Run selected B0-P2 methods on one frozen synthetic case."""
    if config is None:
        config = BenchmarkConfig()

    selected_methods = tuple(BaselineID(method) for method in methods)
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("methods must not contain duplicates")
    needs_filtration = any(
        method is not BaselineID.B0_CONVEX_HULL for method in selected_methods
    )
    filtration = AlphaFiltration.from_points(case.points) if needs_filtration else None

    results: list[BaselineResult] = []
    for method in selected_methods:
        if method is BaselineID.B0_CONVEX_HULL:
            result = _b0(case, config)
        elif method is BaselineID.B1_FIXED_ALPHA:
            assert filtration is not None
            result = _b1(filtration, case, config)
        elif method is BaselineID.B2_CRITICAL_ORACLE:
            assert filtration is not None
            result = _b2(filtration, case, config)
        elif method is BaselineID.B3_PERSISTENCE_STABILITY:
            assert filtration is not None
            result = _b3(filtration, case, config)
        elif method is BaselineID.B4_DENSITY_SCALED:
            assert filtration is not None
            result = _b4(filtration, case, config)
        elif method is BaselineID.B5_PCA_ANISOTROPIC:
            assert filtration is not None
            result = _b5(filtration, case, config)
        elif method is BaselineID.P1_PFTF_LOCAL_SPD:
            assert filtration is not None
            result = _p1(filtration, case, config)
        else:
            assert filtration is not None
            result = _p2(filtration, case, config)
        results.append(result)

    return CaseBenchmark(
        family=case.family.value,
        split=case.split.value,
        seed=case.seed,
        point_count=case.points.shape[0],
        reference_count=case.reference_points.shape[0],
        expected_components=case.expected_components,
        characteristic_length=case.characteristic_length,
        expected_surface_betti=case.expected_surface_betti,
        variation=dict(case.variation),
        results=tuple(results),
    )
