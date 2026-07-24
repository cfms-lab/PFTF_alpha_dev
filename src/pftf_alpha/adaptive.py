"""Closure-preserving adaptive Delaunay methods B4, B5, P1, and P2."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .filtration import AlphaFiltration
from .geometry import intrinsic_circumsphere
from .metrics import metric_circumradius_squared
from .pftf import pftf_relation_field
from .surface import SurfaceMesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class LocalNeighborhoodGeometry:
    """kNN scale and PCA frame used by prior-art adaptive baselines."""

    scales: FloatArray
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    planarity: FloatArray
    k_neighbors: int


@dataclass(frozen=True)
class GeometricBridgeRisk:
    """Reference-free per-cell risk signals for two bridge failure modes."""

    risk: FloatArray
    normal_signal: FloatArray
    length_signal: FloatArray
    normal_coherence: float
    route: str
    normal_coherence_threshold: float
    normal_edge_threshold: float
    length_edge_threshold: float
    k_neighbors: int

    def __post_init__(self) -> None:
        risk = np.asarray(self.risk, dtype=np.float64)
        normal = np.asarray(self.normal_signal, dtype=np.float64)
        length = np.asarray(self.length_signal, dtype=np.float64)
        if risk.ndim != 1 or normal.shape != risk.shape or length.shape != risk.shape:
            raise ValueError("bridge-risk arrays must be one-dimensional and aligned")
        if any(
            not np.all(np.isfinite(values)) or np.any(values < 0.0)
            for values in (risk, normal, length)
        ):
            raise ValueError("bridge-risk arrays must be finite and non-negative")
        if self.route not in {"parallel_normal", "long_edge"}:
            raise ValueError("bridge-risk route is invalid")
        if not math.isfinite(self.normal_coherence) or not (
            0.0 <= self.normal_coherence <= 1.0
        ):
            raise ValueError("normal_coherence must lie in [0, 1]")
        if not math.isfinite(self.normal_coherence_threshold) or not (
            0.0 < self.normal_coherence_threshold <= 1.0
        ):
            raise ValueError("normal_coherence_threshold must lie in (0, 1]")
        for name, value in (
            ("normal_edge_threshold", self.normal_edge_threshold),
            ("length_edge_threshold", self.length_edge_threshold),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.k_neighbors < 3:
            raise ValueError("k_neighbors must be at least three")
        object.__setattr__(self, "risk", np.ascontiguousarray(risk))
        object.__setattr__(self, "normal_signal", np.ascontiguousarray(normal))
        object.__setattr__(self, "length_signal", np.ascontiguousarray(length))


@dataclass(frozen=True)
class BoundaryBridgeLocalization:
    """Label-free bridge-risk values on one selected tetrahedral boundary."""

    boundary_faces: IntArray
    boundary_face_risk: FloatArray
    boundary_edges: IntArray
    boundary_edge_risk: FloatArray
    owner_cell_indices: IntArray
    owner_dual_degree: IntArray
    owner_boundary_face_count: IntArray
    owner_articulation_mask: BoolArray
    owner_dual_bridge_fraction: FloatArray
    route: str
    normal_coherence: float
    normal_coherence_threshold: float
    normal_edge_threshold: float
    length_edge_threshold: float
    k_neighbors: int
    selected_cell_count: int
    selected_dual_component_count: int
    selected_dual_edge_count: int
    selected_dual_bridge_edge_count: int
    selected_dual_articulation_cell_count: int

    def __post_init__(self) -> None:
        faces = np.asarray(self.boundary_faces, dtype=np.int64)
        face_risk = np.asarray(self.boundary_face_risk, dtype=np.float64)
        edges = np.asarray(self.boundary_edges, dtype=np.int64)
        edge_risk = np.asarray(self.boundary_edge_risk, dtype=np.float64)
        owners = np.asarray(self.owner_cell_indices, dtype=np.int64)
        degrees = np.asarray(self.owner_dual_degree, dtype=np.int64)
        boundary_counts = np.asarray(self.owner_boundary_face_count, dtype=np.int64)
        articulations = np.asarray(self.owner_articulation_mask, dtype=bool)
        bridge_fractions = np.asarray(self.owner_dual_bridge_fraction, dtype=np.float64)
        face_count = faces.shape[0]
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("boundary_faces must have shape (m, 3)")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("boundary_edges must have shape (n, 2)")
        if face_risk.shape != (face_count,) or edge_risk.shape != (edges.shape[0],):
            raise ValueError("boundary risk arrays must align with faces and edges")
        for name, values in (
            ("owner_cell_indices", owners),
            ("owner_dual_degree", degrees),
            ("owner_boundary_face_count", boundary_counts),
            ("owner_articulation_mask", articulations),
            ("owner_dual_bridge_fraction", bridge_fractions),
        ):
            if values.shape != (face_count,):
                raise ValueError(f"{name} must align with boundary faces")
        if any(
            not np.all(np.isfinite(values)) or np.any(values < 0.0)
            for values in (face_risk, edge_risk, bridge_fractions)
        ):
            raise ValueError("boundary bridge risks must be finite and non-negative")
        if np.any(degrees < 0) or np.any(boundary_counts < 1):
            raise ValueError("boundary owner connectivity counts are invalid")
        if self.route not in {"parallel_normal", "long_edge"}:
            raise ValueError("boundary bridge-risk route is invalid")
        if self.selected_cell_count < 0 or self.selected_dual_component_count < 0:
            raise ValueError("selected-cell connectivity counts must be non-negative")
        if self.selected_cell_count == 0:
            if self.selected_dual_component_count != 0 or face_count or edges.shape[0]:
                raise ValueError(
                    "empty localization must have no boundary or dual graph"
                )
        elif self.selected_dual_component_count < 1:
            raise ValueError(
                "non-empty localization requires a selected dual component"
            )
        for value in (
            self.selected_dual_edge_count,
            self.selected_dual_bridge_edge_count,
            self.selected_dual_articulation_cell_count,
        ):
            if value < 0:
                raise ValueError("dual-connectivity counts must be non-negative")
        object.__setattr__(self, "boundary_faces", np.ascontiguousarray(faces))
        object.__setattr__(self, "boundary_face_risk", np.ascontiguousarray(face_risk))
        object.__setattr__(self, "boundary_edges", np.ascontiguousarray(edges))
        object.__setattr__(self, "boundary_edge_risk", np.ascontiguousarray(edge_risk))
        object.__setattr__(self, "owner_cell_indices", np.ascontiguousarray(owners))
        object.__setattr__(self, "owner_dual_degree", np.ascontiguousarray(degrees))
        object.__setattr__(
            self, "owner_boundary_face_count", np.ascontiguousarray(boundary_counts)
        )
        object.__setattr__(
            self, "owner_articulation_mask", np.ascontiguousarray(articulations)
        )
        object.__setattr__(
            self, "owner_dual_bridge_fraction", np.ascontiguousarray(bridge_fractions)
        )


@dataclass(frozen=True)
class BoundaryRiskRegionAnalysis:
    """Connected risky-face regions and safe-backbone cut evidence."""

    localization: BoundaryBridgeLocalization
    flagged_face_region_ids: IntArray
    region_face_counts: IntArray
    region_owner_counts: IntArray
    region_risk_mass: FloatArray
    region_max_risk: FloatArray
    boundary_vertex_indices: IntArray
    safe_vertex_component_ids: IntArray
    flagged_edge_cut_mask: BoolArray
    risk_threshold: float
    safe_boundary_component_count: int

    def __post_init__(self) -> None:
        region_ids = np.asarray(self.flagged_face_region_ids, dtype=np.int64)
        face_counts = np.asarray(self.region_face_counts, dtype=np.int64)
        owner_counts = np.asarray(self.region_owner_counts, dtype=np.int64)
        risk_mass = np.asarray(self.region_risk_mass, dtype=np.float64)
        max_risk = np.asarray(self.region_max_risk, dtype=np.float64)
        vertices = np.asarray(self.boundary_vertex_indices, dtype=np.int64)
        safe_components = np.asarray(
            self.safe_vertex_component_ids,
            dtype=np.int64,
        )
        cut_mask = np.asarray(self.flagged_edge_cut_mask, dtype=bool)
        region_count = face_counts.size
        if region_ids.shape != (self.localization.boundary_faces.shape[0],):
            raise ValueError("flagged_face_region_ids must align with boundary faces")
        if any(
            values.shape != (region_count,)
            for values in (owner_counts, risk_mass, max_risk)
        ):
            raise ValueError("risk-region summaries must be aligned vectors")
        if np.any(region_ids < -1) or np.any(region_ids >= region_count):
            raise ValueError("flagged face region ids are invalid")
        if region_count and not np.array_equal(
            np.unique(region_ids[region_ids >= 0]),
            np.arange(region_count),
        ):
            raise ValueError("risk-region ids must be dense and represented")
        if np.any(face_counts <= 0) or np.any(owner_counts <= 0):
            raise ValueError("risk regions must contain faces and owners")
        if not np.all(np.isfinite(risk_mass)) or np.any(risk_mass < 0.0):
            raise ValueError("region risk mass must be finite and non-negative")
        threshold = _finite_nonnegative(self.risk_threshold, "risk_threshold")
        if not np.all(np.isfinite(max_risk)) or np.any(max_risk <= threshold):
            raise ValueError("every risk region must exceed the risk threshold")
        if vertices.ndim != 1 or np.unique(vertices).size != vertices.size:
            raise ValueError("boundary_vertex_indices must be a unique vector")
        if safe_components.shape != vertices.shape:
            raise ValueError("safe component ids must align with boundary vertices")
        if cut_mask.shape != (self.localization.boundary_edges.shape[0],):
            raise ValueError("flagged_edge_cut_mask must align with boundary edges")
        if np.any(cut_mask & (self.localization.boundary_edge_risk <= threshold)):
            raise ValueError("safe or threshold edges cannot enter the risky cut set")
        if vertices.size == 0:
            if self.safe_boundary_component_count != 0:
                raise ValueError("empty boundary graph cannot have safe components")
        else:
            if self.safe_boundary_component_count < 1:
                raise ValueError("non-empty boundary graph needs a safe component")
            if not np.array_equal(
                np.unique(safe_components),
                np.arange(self.safe_boundary_component_count),
            ):
                raise ValueError("safe component ids must be dense and represented")
        object.__setattr__(
            self,
            "flagged_face_region_ids",
            np.ascontiguousarray(region_ids),
        )
        object.__setattr__(
            self, "region_face_counts", np.ascontiguousarray(face_counts)
        )
        object.__setattr__(
            self,
            "region_owner_counts",
            np.ascontiguousarray(owner_counts),
        )
        object.__setattr__(self, "region_risk_mass", np.ascontiguousarray(risk_mass))
        object.__setattr__(self, "region_max_risk", np.ascontiguousarray(max_risk))
        object.__setattr__(
            self,
            "boundary_vertex_indices",
            np.ascontiguousarray(vertices),
        )
        object.__setattr__(
            self,
            "safe_vertex_component_ids",
            np.ascontiguousarray(safe_components),
        )
        object.__setattr__(
            self,
            "flagged_edge_cut_mask",
            np.ascontiguousarray(cut_mask),
        )
        object.__setattr__(self, "risk_threshold", threshold)


@dataclass(frozen=True)
class AdaptiveCellFiltration:
    """Dimensionless score for every Delaunay tetrahedron.

    A threshold selects top cells. Taking the complete closure of those cells
    guarantees a global simplicial subcomplex even though scores came from
    spatially varying local scales or metrics.
    """

    points: FloatArray
    top_simplices: IntArray
    scores: FloatArray
    method: str
    diagnostics: Mapping[str, float]
    cell_confidence: FloatArray | None = None
    fallback_mask: BoolArray | None = None
    guard_scores: FloatArray | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        cells = np.asarray(self.top_simplices, dtype=np.int64)
        scores = np.asarray(self.scores, dtype=np.float64)
        confidence = (
            None
            if self.cell_confidence is None
            else np.asarray(self.cell_confidence, dtype=np.float64)
        )
        fallback = (
            None
            if self.fallback_mask is None
            else np.asarray(self.fallback_mask, dtype=bool)
        )
        guard_scores = (
            None
            if self.guard_scores is None
            else np.asarray(self.guard_scores, dtype=np.float64)
        )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("adaptive filtration requires points with shape (n, 3)")
        if cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError("top_simplices must have shape (m, 4)")
        if scores.shape != (cells.shape[0],):
            raise ValueError("scores must have shape (m,)")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
            raise ValueError("adaptive scores must be finite and non-negative")
        if confidence is not None and (
            confidence.shape != scores.shape
            or not np.all(np.isfinite(confidence))
            or np.any((confidence < 0.0) | (confidence > 1.0))
        ):
            raise ValueError("cell_confidence must have score shape and lie in [0, 1]")
        if fallback is not None and fallback.shape != scores.shape:
            raise ValueError("fallback_mask must have the same shape as scores")
        if guard_scores is not None and (
            guard_scores.shape != scores.shape
            or not np.all(np.isfinite(guard_scores))
            or np.any(guard_scores < 0.0)
        ):
            raise ValueError(
                "guard_scores must have score shape and be finite and non-negative"
            )
        object.__setattr__(self, "points", np.ascontiguousarray(points))
        object.__setattr__(self, "top_simplices", np.ascontiguousarray(cells))
        object.__setattr__(self, "scores", np.ascontiguousarray(scores))
        if confidence is not None:
            object.__setattr__(
                self,
                "cell_confidence",
                np.ascontiguousarray(confidence),
            )
        if fallback is not None:
            object.__setattr__(
                self,
                "fallback_mask",
                np.ascontiguousarray(fallback),
            )
        if guard_scores is not None:
            object.__setattr__(
                self,
                "guard_scores",
                np.ascontiguousarray(guard_scores),
            )

    def critical_values(self) -> FloatArray:
        return np.unique(self.scores)

    def selected_cell_count(self, scale_multiplier: float) -> int:
        threshold = _finite_nonnegative(scale_multiplier, "scale_multiplier")
        return int(np.count_nonzero(self.scores <= threshold))

    def surface_at(self, scale_multiplier: float) -> SurfaceMesh:
        """Regularized boundary of selected tetrahedra and their closure."""
        threshold = _finite_nonnegative(scale_multiplier, "scale_multiplier")
        selected = self.scores <= threshold
        face_counts: Counter[tuple[int, int, int]] = Counter()
        for cell in self.top_simplices[selected]:
            ordered = tuple(sorted(int(vertex) for vertex in cell))
            face_counts.update(combinations(ordered, 3))
        boundary = sorted(
            face for face, incidence in face_counts.items() if incidence == 1
        )
        faces = (
            np.asarray(boundary, dtype=np.int64)
            if boundary
            else np.empty((0, 3), dtype=np.int64)
        )
        return SurfaceMesh(vertices=self.points, faces=faces)

    def diagnostics_at(self, scale_multiplier: float) -> dict[str, float]:
        """Combine static diagnostics with closure and guard evidence."""

        threshold = _finite_nonnegative(scale_multiplier, "scale_multiplier")
        selected = self.scores <= threshold
        selected_count = int(np.count_nonzero(selected))
        result = dict(self.diagnostics)
        result["selected_cell_count"] = float(selected_count)

        selected_cells = self.top_simplices[selected]
        closure_vertices: set[int] = set()
        closure_edges: set[tuple[int, int]] = set()
        face_counts: Counter[tuple[int, int, int]] = Counter()
        for cell in selected_cells:
            ordered = tuple(sorted(int(vertex) for vertex in cell))
            closure_vertices.update(ordered)
            closure_edges.update(combinations(ordered, 2))
            face_counts.update(combinations(ordered, 3))
        result.update(
            {
                "closure_vertex_count": float(len(closure_vertices)),
                "closure_edge_count": float(len(closure_edges)),
                "closure_face_count": float(len(face_counts)),
                "boundary_face_count": float(
                    sum(incidence == 1 for incidence in face_counts.values())
                ),
                "face_incidence_over_two_count": float(
                    sum(incidence > 2 for incidence in face_counts.values())
                ),
                "downward_closure_complete": 1.0,
            }
        )

        if self.cell_confidence is not None:
            selected_confidence = self.cell_confidence[selected]
            if selected_confidence.size:
                result.update(
                    {
                        "selected_confidence_min": float(np.min(selected_confidence)),
                        "selected_confidence_median": float(
                            np.median(selected_confidence)
                        ),
                        "selected_confidence_max": float(np.max(selected_confidence)),
                    }
                )
            else:
                result.update(
                    {
                        "selected_confidence_min": 0.0,
                        "selected_confidence_median": 0.0,
                        "selected_confidence_max": 0.0,
                    }
                )
        if self.fallback_mask is not None:
            fallback_count = int(np.count_nonzero(self.fallback_mask))
            selected_fallback_count = int(
                np.count_nonzero(self.fallback_mask & selected)
            )
            result["fallback_cell_count"] = float(fallback_count)
            result["fallback_fraction"] = fallback_count / max(len(self.scores), 1)
            result["selected_fallback_cell_count"] = float(selected_fallback_count)
            result["selected_fallback_fraction"] = selected_fallback_count / max(
                selected_count, 1
            )
        if self.guard_scores is not None:
            guarded_selected = selected
            if self.fallback_mask is not None:
                guarded_selected = guarded_selected & self.fallback_mask
            tolerance = (
                16.0 * np.finfo(np.float64).eps * np.maximum(self.guard_scores, 1.0)
            )
            selected_guard_violations = guarded_selected & (
                self.guard_scores > threshold + tolerance
            )
            guarded_selected_count = int(np.count_nonzero(guarded_selected))
            result["selected_guard_violation_count"] = float(
                np.count_nonzero(selected_guard_violations)
            )
            result["selected_guard_violation_fraction"] = float(
                np.count_nonzero(selected_guard_violations)
                / max(guarded_selected_count, 1)
            )
        return result


@dataclass(frozen=True)
class BoundaryOwnerIntervention:
    """One label-free, boundary-recomputing cell-removal intervention."""

    filtration: AdaptiveCellFiltration
    requested_rounds: int
    executed_rounds: int
    removed_cell_indices: IntArray
    removed_cells_per_round: tuple[int, ...]
    risk_threshold: float
    initial_selected_cell_count: int
    final_selected_cell_count: int
    initial_flagged_face_count: int
    final_flagged_face_count: int
    initial_flagged_edge_count: int
    final_flagged_edge_count: int
    boundary_recomputation_count: int
    stopping_reason: str

    def __post_init__(self) -> None:
        removed = np.asarray(self.removed_cell_indices, dtype=np.int64)
        if removed.ndim != 1:
            raise ValueError("removed_cell_indices must be one-dimensional")
        if np.unique(removed).size != removed.size:
            raise ValueError("removed_cell_indices must be unique")
        if self.requested_rounds < 0 or self.executed_rounds < 0:
            raise ValueError("intervention round counts must be non-negative")
        if self.executed_rounds > self.requested_rounds:
            raise ValueError("executed rounds cannot exceed requested rounds")
        if len(self.removed_cells_per_round) != self.executed_rounds:
            raise ValueError("removed_cells_per_round must align with executed rounds")
        if any(count <= 0 for count in self.removed_cells_per_round):
            raise ValueError("every executed round must remove at least one cell")
        if sum(self.removed_cells_per_round) != removed.size:
            raise ValueError("per-round removal counts must match removed indices")
        if min(self.initial_selected_cell_count, self.final_selected_cell_count) < 0:
            raise ValueError("selected-cell counts must be non-negative")
        if (
            self.initial_selected_cell_count - self.final_selected_cell_count
            != removed.size
        ):
            raise ValueError("selected-cell change must match removed indices")
        if (
            min(
                self.initial_flagged_face_count,
                self.final_flagged_face_count,
                self.initial_flagged_edge_count,
                self.final_flagged_edge_count,
            )
            < 0
        ):
            raise ValueError("flagged boundary counts must be non-negative")
        threshold = _finite_nonnegative(self.risk_threshold, "risk_threshold")
        if self.boundary_recomputation_count != self.executed_rounds + 1:
            raise ValueError(
                "boundary must be computed initially and after every round"
            )
        if self.stopping_reason not in {
            "round_budget",
            "no_flagged_faces",
            "empty_selection",
        }:
            raise ValueError("invalid intervention stopping reason")
        object.__setattr__(self, "removed_cell_indices", np.ascontiguousarray(removed))
        object.__setattr__(self, "risk_threshold", threshold)


@dataclass(frozen=True)
class BoundaryRegionCutIntervention:
    """One-shot connected-region or safe-backbone cut candidate."""

    filtration: AdaptiveCellFiltration
    analysis: BoundaryRiskRegionAnalysis
    strategy: str
    removed_cell_indices: IntArray
    initial_selected_cell_count: int
    final_selected_cell_count: int
    candidate_face_count: int
    stopping_reason: str

    def __post_init__(self) -> None:
        removed = np.asarray(self.removed_cell_indices, dtype=np.int64)
        if removed.ndim != 1 or np.unique(removed).size != removed.size:
            raise ValueError("removed_cell_indices must be a unique vector")
        if self.strategy not in {
            "baseline",
            "largest_risk_region",
            "safe_backbone_cut",
        }:
            raise ValueError("invalid boundary region/cut strategy")
        if min(self.initial_selected_cell_count, self.final_selected_cell_count) < 0:
            raise ValueError("selected-cell counts must be non-negative")
        if (
            self.initial_selected_cell_count - self.final_selected_cell_count
            != removed.size
        ):
            raise ValueError("selected-cell change must match removed indices")
        if self.candidate_face_count < 0:
            raise ValueError("candidate_face_count must be non-negative")
        if self.stopping_reason not in {
            "baseline",
            "candidate_applied",
            "no_candidate",
        }:
            raise ValueError("invalid region/cut intervention stopping reason")
        if self.strategy == "baseline":
            if removed.size or self.candidate_face_count:
                raise ValueError("baseline cannot remove cells or select faces")
            if self.stopping_reason != "baseline":
                raise ValueError("baseline must use the baseline stopping reason")
        elif removed.size:
            if not self.candidate_face_count:
                raise ValueError("an applied candidate must contain boundary faces")
            if self.stopping_reason != "candidate_applied":
                raise ValueError("removed cells require candidate_applied")
        elif self.stopping_reason != "no_candidate":
            raise ValueError("an empty non-baseline candidate must report no_candidate")
        object.__setattr__(self, "removed_cell_indices", np.ascontiguousarray(removed))


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_k(point_count: int, k_neighbors: int) -> int:
    if not 3 <= k_neighbors < point_count:
        raise ValueError("k_neighbors must satisfy 3 <= k_neighbors < point count")
    return int(k_neighbors)


def knn_scales(points: FloatArray, *, k_neighbors: int) -> FloatArray:
    """Median nonzero kNN distance at every observed point."""

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    selected_k = _validate_k(point_array.shape[0], k_neighbors)
    distances = cKDTree(point_array).query(point_array, k=selected_k + 1, workers=1)[0][
        :, 1:
    ]
    scales = np.median(distances, axis=1)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("kNN scales must be finite and positive")
    return np.ascontiguousarray(scales)


def local_neighborhood_geometry(
    points: FloatArray,
    *,
    k_neighbors: int,
) -> LocalNeighborhoodGeometry:
    """Fit local PCA frames without using reference geometry."""

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    selected_k = _validate_k(point_array.shape[0], k_neighbors)
    distances, neighbor_indices = cKDTree(point_array).query(
        point_array, k=selected_k + 1, workers=1
    )
    scales = np.median(distances[:, 1:], axis=1)
    eigenvalues = np.empty((point_array.shape[0], 3), dtype=np.float64)
    eigenvectors = np.empty((point_array.shape[0], 3, 3), dtype=np.float64)
    for point_index, indices in enumerate(neighbor_indices[:, 1:]):
        neighborhood = point_array[indices]
        centered = neighborhood - np.mean(neighborhood, axis=0)
        covariance = centered.T @ centered / selected_k
        values, vectors = np.linalg.eigh(covariance)
        eigenvalues[point_index] = np.maximum(values, 0.0)
        eigenvectors[point_index] = vectors

    denominator = np.maximum(eigenvalues[:, 2], np.finfo(np.float64).eps)
    planarity = np.clip(
        (eigenvalues[:, 1] - eigenvalues[:, 0]) / denominator,
        0.0,
        1.0,
    )
    return LocalNeighborhoodGeometry(
        scales=np.ascontiguousarray(scales),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        planarity=planarity,
        k_neighbors=selected_k,
    )


def geometric_bridge_risk(
    filtration: AlphaFiltration,
    *,
    k_neighbors: int,
    normal_coherence_threshold: float = 0.9,
    normal_edge_threshold: float = 0.02,
    length_edge_threshold: float = 1.8,
) -> GeometricBridgeRisk:
    """Estimate bridge-prone cells without references or component labels.

    Highly coherent normal fields use a parallel-sheet signal: an edge is
    suspicious when it follows both endpoint normals, the normals agree up to
    sign, and both neighborhoods are planar. Other shapes use the second-longest
    edge normalized by endpoint kNN scales, which is robust to one sliver edge.
    A risk above one exceeds the declared route-specific threshold.
    """

    if filtration.ambient_dimension != 3:
        raise ValueError("geometric bridge risk requires a 3D Delaunay filtration")
    if not math.isfinite(normal_coherence_threshold) or not (
        0.0 < normal_coherence_threshold <= 1.0
    ):
        raise ValueError("normal_coherence_threshold must lie in (0, 1]")
    for name, value in (
        ("normal_edge_threshold", normal_edge_threshold),
        ("length_edge_threshold", length_edge_threshold),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    geometry = local_neighborhood_geometry(
        filtration.points,
        k_neighbors=k_neighbors,
    )
    normals = geometry.eigenvectors[:, :, 0]
    orientation_tensor = np.mean(
        normals[:, :, None] * normals[:, None, :],
        axis=0,
    )
    normal_coherence = float(np.linalg.eigvalsh(orientation_tensor)[-1])

    cell_count = filtration.top_simplices.shape[0]
    normal_signal = np.empty(cell_count, dtype=np.float64)
    length_signal = np.empty(cell_count, dtype=np.float64)
    for cell_index, cell in enumerate(filtration.top_simplices):
        edge_normal_signals: list[float] = []
        normalized_edge_lengths: list[float] = []
        for first, second in combinations((int(vertex) for vertex in cell), 2):
            displacement = filtration.points[second] - filtration.points[first]
            edge_length = float(np.linalg.norm(displacement))
            direction = displacement / edge_length
            first_alignment = abs(float(direction @ normals[first]))
            second_alignment = abs(float(direction @ normals[second]))
            normal_parallelism = abs(float(normals[first] @ normals[second]))
            edge_planarity = math.sqrt(
                float(geometry.planarity[first] * geometry.planarity[second])
            )
            edge_normal_signals.append(
                edge_planarity * normal_parallelism * first_alignment * second_alignment
            )
            normalized_edge_lengths.append(
                edge_length
                / math.sqrt(float(geometry.scales[first] * geometry.scales[second]))
            )
        normal_signal[cell_index] = float(np.mean(edge_normal_signals))
        length_signal[cell_index] = sorted(normalized_edge_lengths, reverse=True)[1]

    if normal_coherence >= normal_coherence_threshold:
        route = "parallel_normal"
        risk = normal_signal / normal_edge_threshold
    else:
        route = "long_edge"
        risk = length_signal / length_edge_threshold
    return GeometricBridgeRisk(
        risk=risk,
        normal_signal=normal_signal,
        length_signal=length_signal,
        normal_coherence=normal_coherence,
        route=route,
        normal_coherence_threshold=float(normal_coherence_threshold),
        normal_edge_threshold=float(normal_edge_threshold),
        length_edge_threshold=float(length_edge_threshold),
        k_neighbors=geometry.k_neighbors,
    )


def _dual_cut_structure(
    adjacency: dict[int, set[int]],
) -> tuple[int, set[int], set[tuple[int, int]]]:
    """Return component, articulation, and bridge structure of an undirected graph."""

    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    child_count: Counter[int] = Counter()
    articulation_cells: set[int] = set()
    bridge_edges: set[tuple[int, int]] = set()
    time = 0
    component_count = 0

    for root in sorted(adjacency):
        if root in discovery:
            continue
        component_count += 1
        time += 1
        discovery[root] = time
        low[root] = time
        parent[root] = None
        stack: list[tuple[int, Iterator[int]]] = [(root, iter(sorted(adjacency[root])))]
        while stack:
            node, neighbors = stack[-1]
            try:
                neighbor = next(neighbors)
            except StopIteration:
                stack.pop()
                ancestor = parent[node]
                if ancestor is None:
                    if child_count[node] > 1:
                        articulation_cells.add(node)
                    continue
                low[ancestor] = min(low[ancestor], low[node])
                if parent[ancestor] is not None and low[node] >= discovery[ancestor]:
                    articulation_cells.add(ancestor)
                if low[node] > discovery[ancestor]:
                    bridge_edges.add(tuple(sorted((ancestor, node))))
                continue

            if neighbor not in discovery:
                parent[neighbor] = node
                child_count[node] += 1
                time += 1
                discovery[neighbor] = time
                low[neighbor] = time
                stack.append((neighbor, iter(sorted(adjacency[neighbor]))))
            elif neighbor != parent[node]:
                low[node] = min(low[node], discovery[neighbor])

    return component_count, articulation_cells, bridge_edges


def boundary_bridge_localization(
    base: AdaptiveCellFiltration,
    *,
    scale_multiplier: float,
    k_neighbors: int,
    normal_coherence_threshold: float = 0.9,
    normal_edge_threshold: float = 0.02,
    length_edge_threshold: float = 1.8,
) -> BoundaryBridgeLocalization:
    """Localize bridge-prone edges and faces on the selected output boundary.

    The primary edge risk uses the same reference-free route as the cell probe.
    A boundary face receives the maximum risk of its three edges. Selected-cell
    dual articulation and bridge incidence are reported separately rather than
    fused into the geometric score, so weak cut-graph evidence cannot hide a
    strong boundary signal or affect selection.
    """

    threshold = _finite_nonnegative(scale_multiplier, "scale_multiplier")
    if not math.isfinite(normal_coherence_threshold) or not (
        0.0 < normal_coherence_threshold <= 1.0
    ):
        raise ValueError("normal_coherence_threshold must lie in (0, 1]")
    for name, value in (
        ("normal_edge_threshold", normal_edge_threshold),
        ("length_edge_threshold", length_edge_threshold),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    selected_indices = np.flatnonzero(base.scores <= threshold)
    if selected_indices.size == 0:
        geometry = local_neighborhood_geometry(base.points, k_neighbors=k_neighbors)
        normals = geometry.eigenvectors[:, :, 0]
        orientation_tensor = np.mean(
            normals[:, :, None] * normals[:, None, :],
            axis=0,
        )
        normal_coherence = float(np.linalg.eigvalsh(orientation_tensor)[-1])
        route = (
            "parallel_normal"
            if normal_coherence >= normal_coherence_threshold
            else "long_edge"
        )
        return BoundaryBridgeLocalization(
            boundary_faces=np.empty((0, 3), dtype=np.int64),
            boundary_face_risk=np.empty(0, dtype=np.float64),
            boundary_edges=np.empty((0, 2), dtype=np.int64),
            boundary_edge_risk=np.empty(0, dtype=np.float64),
            owner_cell_indices=np.empty(0, dtype=np.int64),
            owner_dual_degree=np.empty(0, dtype=np.int64),
            owner_boundary_face_count=np.empty(0, dtype=np.int64),
            owner_articulation_mask=np.empty(0, dtype=bool),
            owner_dual_bridge_fraction=np.empty(0, dtype=np.float64),
            route=route,
            normal_coherence=normal_coherence,
            normal_coherence_threshold=float(normal_coherence_threshold),
            normal_edge_threshold=float(normal_edge_threshold),
            length_edge_threshold=float(length_edge_threshold),
            k_neighbors=geometry.k_neighbors,
            selected_cell_count=0,
            selected_dual_component_count=0,
            selected_dual_edge_count=0,
            selected_dual_bridge_edge_count=0,
            selected_dual_articulation_cell_count=0,
        )
    face_owners: dict[tuple[int, int, int], list[int]] = {}
    for cell_index in selected_indices:
        cell = tuple(sorted(int(vertex) for vertex in base.top_simplices[cell_index]))
        for face in combinations(cell, 3):
            face_owners.setdefault(face, []).append(int(cell_index))

    adjacency = {int(cell_index): set() for cell_index in selected_indices}
    for owners in face_owners.values():
        if len(owners) == 2:
            first, second = owners
            adjacency[first].add(second)
            adjacency[second].add(first)
        elif len(owners) > 2:
            raise ValueError("selected Delaunay face has more than two cofaces")
    dual_components, articulation_cells, bridge_edges = _dual_cut_structure(adjacency)

    boundary_records = sorted(
        (face, owners[0]) for face, owners in face_owners.items() if len(owners) == 1
    )
    if not boundary_records:
        raise ValueError("selected cells have no regularized boundary")
    boundary_faces = np.asarray(
        [face for face, _ in boundary_records],
        dtype=np.int64,
    )
    owner_cell_indices = np.asarray(
        [owner for _, owner in boundary_records],
        dtype=np.int64,
    )
    boundary_edges = np.asarray(
        sorted(
            {
                tuple(sorted(edge))
                for face in boundary_faces.tolist()
                for edge in combinations(face, 2)
            }
        ),
        dtype=np.int64,
    )

    geometry = local_neighborhood_geometry(base.points, k_neighbors=k_neighbors)
    normals = geometry.eigenvectors[:, :, 0]
    orientation_tensor = np.mean(
        normals[:, :, None] * normals[:, None, :],
        axis=0,
    )
    normal_coherence = float(np.linalg.eigvalsh(orientation_tensor)[-1])
    route = (
        "parallel_normal"
        if normal_coherence >= normal_coherence_threshold
        else "long_edge"
    )

    edge_risk_lookup: dict[tuple[int, int], float] = {}
    for first, second in boundary_edges:
        edge = (int(first), int(second))
        displacement = base.points[second] - base.points[first]
        edge_length = float(np.linalg.norm(displacement))
        direction = displacement / edge_length
        normalized_length = edge_length / math.sqrt(
            float(geometry.scales[first] * geometry.scales[second])
        )
        normal_signal = (
            math.sqrt(float(geometry.planarity[first] * geometry.planarity[second]))
            * abs(float(normals[first] @ normals[second]))
            * abs(float(direction @ normals[first]))
            * abs(float(direction @ normals[second]))
        )
        edge_risk_lookup[edge] = (
            normal_signal / normal_edge_threshold
            if route == "parallel_normal"
            else normalized_length / length_edge_threshold
        )
    boundary_edge_risk = np.asarray(
        [
            edge_risk_lookup[tuple(int(vertex) for vertex in edge)]
            for edge in boundary_edges
        ],
        dtype=np.float64,
    )
    boundary_face_risk = np.asarray(
        [
            max(
                edge_risk_lookup[tuple(sorted((int(first), int(second))))]
                for first, second in combinations(face, 2)
            )
            for face in boundary_faces
        ],
        dtype=np.float64,
    )

    owner_boundary_counts = Counter(int(owner) for owner in owner_cell_indices)
    bridge_incidence: Counter[int] = Counter()
    for first, second in bridge_edges:
        bridge_incidence[first] += 1
        bridge_incidence[second] += 1
    owner_dual_degree = np.asarray(
        [len(adjacency[int(owner)]) for owner in owner_cell_indices],
        dtype=np.int64,
    )
    owner_boundary_face_count = np.asarray(
        [owner_boundary_counts[int(owner)] for owner in owner_cell_indices],
        dtype=np.int64,
    )
    owner_articulation_mask = np.asarray(
        [int(owner) in articulation_cells for owner in owner_cell_indices],
        dtype=bool,
    )
    owner_dual_bridge_fraction = np.asarray(
        [
            bridge_incidence[int(owner)] / max(len(adjacency[int(owner)]), 1)
            for owner in owner_cell_indices
        ],
        dtype=np.float64,
    )
    return BoundaryBridgeLocalization(
        boundary_faces=boundary_faces,
        boundary_face_risk=boundary_face_risk,
        boundary_edges=boundary_edges,
        boundary_edge_risk=boundary_edge_risk,
        owner_cell_indices=owner_cell_indices,
        owner_dual_degree=owner_dual_degree,
        owner_boundary_face_count=owner_boundary_face_count,
        owner_articulation_mask=owner_articulation_mask,
        owner_dual_bridge_fraction=owner_dual_bridge_fraction,
        route=route,
        normal_coherence=normal_coherence,
        normal_coherence_threshold=float(normal_coherence_threshold),
        normal_edge_threshold=float(normal_edge_threshold),
        length_edge_threshold=float(length_edge_threshold),
        k_neighbors=geometry.k_neighbors,
        selected_cell_count=int(selected_indices.size),
        selected_dual_component_count=dual_components,
        selected_dual_edge_count=sum(len(neighbors) for neighbors in adjacency.values())
        // 2,
        selected_dual_bridge_edge_count=len(bridge_edges),
        selected_dual_articulation_cell_count=len(articulation_cells),
    )


def boundary_risk_region_analysis(
    base: AdaptiveCellFiltration,
    *,
    scale_multiplier: float,
    risk_threshold: float = 1.0,
    k_neighbors: int,
    normal_coherence_threshold: float = 0.9,
    normal_edge_threshold: float = 0.02,
    length_edge_threshold: float = 1.8,
) -> BoundaryRiskRegionAnalysis:
    """Group risky faces through risky edges and audit the safe boundary graph."""

    selected_risk_threshold = _finite_nonnegative(
        risk_threshold,
        "risk_threshold",
    )
    localization = boundary_bridge_localization(
        base,
        scale_multiplier=scale_multiplier,
        k_neighbors=k_neighbors,
        normal_coherence_threshold=normal_coherence_threshold,
        normal_edge_threshold=normal_edge_threshold,
        length_edge_threshold=length_edge_threshold,
    )
    face_region_ids = np.full(
        localization.boundary_faces.shape[0],
        -1,
        dtype=np.int64,
    )
    edge_risk = {
        tuple(int(vertex) for vertex in edge): float(risk)
        for edge, risk in zip(
            localization.boundary_edges,
            localization.boundary_edge_risk,
            strict=True,
        )
    }
    flagged_faces = np.flatnonzero(
        localization.boundary_face_risk > selected_risk_threshold
    )
    edge_to_flagged_faces: dict[tuple[int, int], list[int]] = {}
    for face_index in flagged_faces:
        face = tuple(
            sorted(int(vertex) for vertex in localization.boundary_faces[face_index])
        )
        for edge in combinations(face, 2):
            if edge_risk[edge] > selected_risk_threshold:
                edge_to_flagged_faces.setdefault(edge, []).append(int(face_index))
    face_adjacency = {int(face_index): set() for face_index in flagged_faces}
    for members in edge_to_flagged_faces.values():
        for face_index in members:
            face_adjacency[face_index].update(
                other for other in members if other != face_index
            )

    regions: list[list[int]] = []
    visited_faces: set[int] = set()
    for start in sorted(face_adjacency):
        if start in visited_faces:
            continue
        visited_faces.add(start)
        stack = [start]
        region: list[int] = []
        while stack:
            face_index = stack.pop()
            region.append(face_index)
            for neighbor in sorted(face_adjacency[face_index], reverse=True):
                if neighbor not in visited_faces:
                    visited_faces.add(neighbor)
                    stack.append(neighbor)
        region_id = len(regions)
        ordered_region = sorted(region)
        face_region_ids[ordered_region] = region_id
        regions.append(ordered_region)

    region_face_counts = np.asarray(
        [len(region) for region in regions],
        dtype=np.int64,
    )
    region_owner_counts = np.asarray(
        [np.unique(localization.owner_cell_indices[region]).size for region in regions],
        dtype=np.int64,
    )
    region_risk_mass = np.asarray(
        [
            np.sum(localization.boundary_face_risk[region] - selected_risk_threshold)
            for region in regions
        ],
        dtype=np.float64,
    )
    region_max_risk = np.asarray(
        [np.max(localization.boundary_face_risk[region]) for region in regions],
        dtype=np.float64,
    )

    boundary_vertices = np.asarray(
        sorted(
            {int(vertex) for edge in localization.boundary_edges for vertex in edge}
        ),
        dtype=np.int64,
    )
    vertex_position = {
        int(vertex): position for position, vertex in enumerate(boundary_vertices)
    }
    safe_adjacency = [set() for _ in boundary_vertices]
    for edge, risk in zip(
        localization.boundary_edges,
        localization.boundary_edge_risk,
        strict=True,
    ):
        if risk <= selected_risk_threshold:
            first = vertex_position[int(edge[0])]
            second = vertex_position[int(edge[1])]
            safe_adjacency[first].add(second)
            safe_adjacency[second].add(first)
    safe_component_ids = np.full(boundary_vertices.size, -1, dtype=np.int64)
    safe_component_count = 0
    for start in range(boundary_vertices.size):
        if safe_component_ids[start] >= 0:
            continue
        safe_component_ids[start] = safe_component_count
        stack = [start]
        while stack:
            position = stack.pop()
            for neighbor in sorted(safe_adjacency[position], reverse=True):
                if safe_component_ids[neighbor] < 0:
                    safe_component_ids[neighbor] = safe_component_count
                    stack.append(neighbor)
        safe_component_count += 1
    flagged_edge_cut_mask = np.asarray(
        [
            risk > selected_risk_threshold
            and safe_component_ids[vertex_position[int(edge[0])]]
            != safe_component_ids[vertex_position[int(edge[1])]]
            for edge, risk in zip(
                localization.boundary_edges,
                localization.boundary_edge_risk,
                strict=True,
            )
        ],
        dtype=bool,
    )
    return BoundaryRiskRegionAnalysis(
        localization=localization,
        flagged_face_region_ids=face_region_ids,
        region_face_counts=region_face_counts,
        region_owner_counts=region_owner_counts,
        region_risk_mass=region_risk_mass,
        region_max_risk=region_max_risk,
        boundary_vertex_indices=boundary_vertices,
        safe_vertex_component_ids=safe_component_ids,
        flagged_edge_cut_mask=flagged_edge_cut_mask,
        risk_threshold=selected_risk_threshold,
        safe_boundary_component_count=safe_component_count,
    )


def boundary_region_cut_intervention(
    base: AdaptiveCellFiltration,
    *,
    scale_multiplier: float,
    strategy: str,
    risk_threshold: float = 1.0,
    k_neighbors: int,
    normal_coherence_threshold: float = 0.9,
    normal_edge_threshold: float = 0.02,
    length_edge_threshold: float = 1.8,
) -> BoundaryRegionCutIntervention:
    """Construct one label-free connected-region or safe-cut candidate."""

    if strategy not in {
        "baseline",
        "largest_risk_region",
        "safe_backbone_cut",
    }:
        raise ValueError("invalid boundary region/cut strategy")
    threshold = _finite_nonnegative(scale_multiplier, "scale_multiplier")
    analysis = boundary_risk_region_analysis(
        base,
        scale_multiplier=threshold,
        risk_threshold=risk_threshold,
        k_neighbors=k_neighbors,
        normal_coherence_threshold=normal_coherence_threshold,
        normal_edge_threshold=normal_edge_threshold,
        length_edge_threshold=length_edge_threshold,
    )
    localization = analysis.localization
    candidate_face_mask = np.zeros(
        localization.boundary_faces.shape[0],
        dtype=bool,
    )
    if strategy == "largest_risk_region" and analysis.region_face_counts.size:
        region_id = max(
            range(analysis.region_face_counts.size),
            key=lambda index: (
                float(analysis.region_risk_mass[index]),
                float(analysis.region_max_risk[index]),
                int(analysis.region_face_counts[index]),
                -index,
            ),
        )
        candidate_face_mask = analysis.flagged_face_region_ids == region_id
    elif strategy == "safe_backbone_cut":
        cut_edges = {
            tuple(int(vertex) for vertex in edge)
            for edge in localization.boundary_edges[analysis.flagged_edge_cut_mask]
        }
        for face_index, face in enumerate(localization.boundary_faces):
            ordered_face = tuple(sorted(int(vertex) for vertex in face))
            candidate_face_mask[face_index] = any(
                edge in cut_edges for edge in combinations(ordered_face, 2)
            )

    removed = np.unique(localization.owner_cell_indices[candidate_face_mask]).astype(
        np.int64, copy=False
    )
    candidate_face_count = int(np.count_nonzero(candidate_face_mask))
    initial_selected = localization.selected_cell_count
    if strategy == "baseline":
        return BoundaryRegionCutIntervention(
            filtration=base,
            analysis=analysis,
            strategy=strategy,
            removed_cell_indices=np.empty(0, dtype=np.int64),
            initial_selected_cell_count=initial_selected,
            final_selected_cell_count=initial_selected,
            candidate_face_count=0,
            stopping_reason="baseline",
        )
    if removed.size == 0:
        return BoundaryRegionCutIntervention(
            filtration=base,
            analysis=analysis,
            strategy=strategy,
            removed_cell_indices=removed,
            initial_selected_cell_count=initial_selected,
            final_selected_cell_count=initial_selected,
            candidate_face_count=0,
            stopping_reason="no_candidate",
        )

    replacement = float(np.nextafter(threshold, np.inf))
    if not math.isfinite(replacement):
        raise ValueError("scale_multiplier leaves no finite score for cell removal")
    scores = np.array(base.scores, copy=True)
    scores[removed] = replacement
    diagnostics = dict(base.diagnostics)
    diagnostics.update(
        {
            "boundary_region_cut_strategy_largest_region": float(
                strategy == "largest_risk_region"
            ),
            "boundary_region_cut_strategy_safe_backbone": float(
                strategy == "safe_backbone_cut"
            ),
            "boundary_region_count": float(analysis.region_face_counts.size),
            "boundary_region_largest_face_count": float(
                np.max(analysis.region_face_counts, initial=0)
            ),
            "boundary_safe_component_count": float(
                analysis.safe_boundary_component_count
            ),
            "boundary_safe_cut_edge_count": float(
                np.count_nonzero(analysis.flagged_edge_cut_mask)
            ),
            "boundary_region_cut_candidate_face_count": float(candidate_face_count),
            "boundary_region_cut_removed_cell_count": float(removed.size),
        }
    )
    filtration = AdaptiveCellFiltration(
        points=base.points,
        top_simplices=base.top_simplices,
        scores=scores,
        method=f"{base.method}_boundary_region_cut",
        diagnostics=diagnostics,
        cell_confidence=base.cell_confidence,
        fallback_mask=base.fallback_mask,
        guard_scores=base.guard_scores,
    )
    return BoundaryRegionCutIntervention(
        filtration=filtration,
        analysis=analysis,
        strategy=strategy,
        removed_cell_indices=removed,
        initial_selected_cell_count=initial_selected,
        final_selected_cell_count=initial_selected - int(removed.size),
        candidate_face_count=candidate_face_count,
        stopping_reason="candidate_applied",
    )


def iterative_boundary_owner_intervention(
    base: AdaptiveCellFiltration,
    *,
    scale_multiplier: float,
    max_rounds: int,
    risk_threshold: float = 1.0,
    k_neighbors: int,
    normal_coherence_threshold: float = 0.9,
    normal_edge_threshold: float = 0.02,
    length_edge_threshold: float = 1.8,
) -> BoundaryOwnerIntervention:
    """Remove owners of risky boundary faces and recompute after each round.

    The intervention uses only observed geometry and the frozen P2 boundary.
    Every round removes all unique owners of faces whose risk exceeds the fixed
    threshold. Labels and reference geometry are intentionally absent here and
    may only be used by a separate calibration-only promotion gate.
    """

    threshold = _finite_nonnegative(scale_multiplier, "scale_multiplier")
    selected_risk_threshold = _finite_nonnegative(
        risk_threshold,
        "risk_threshold",
    )
    if isinstance(max_rounds, bool) or int(max_rounds) != max_rounds:
        raise ValueError("max_rounds must be a non-negative integer")
    requested_rounds = int(max_rounds)
    if requested_rounds < 0:
        raise ValueError("max_rounds must be a non-negative integer")

    def localize(filtration: AdaptiveCellFiltration) -> BoundaryBridgeLocalization:
        return boundary_bridge_localization(
            filtration,
            scale_multiplier=threshold,
            k_neighbors=k_neighbors,
            normal_coherence_threshold=normal_coherence_threshold,
            normal_edge_threshold=normal_edge_threshold,
            length_edge_threshold=length_edge_threshold,
        )

    initial = localize(base)
    current = initial
    removed_rounds: list[int] = []
    removed_indices: list[int] = []
    scores = np.array(base.scores, copy=True)
    stopping_reason = "round_budget"

    if requested_rounds == 0:
        initial_flagged_faces = int(
            np.count_nonzero(initial.boundary_face_risk > selected_risk_threshold)
        )
        initial_flagged_edges = int(
            np.count_nonzero(initial.boundary_edge_risk > selected_risk_threshold)
        )
        return BoundaryOwnerIntervention(
            filtration=base,
            requested_rounds=0,
            executed_rounds=0,
            removed_cell_indices=np.empty(0, dtype=np.int64),
            removed_cells_per_round=(),
            risk_threshold=selected_risk_threshold,
            initial_selected_cell_count=initial.selected_cell_count,
            final_selected_cell_count=initial.selected_cell_count,
            initial_flagged_face_count=initial_flagged_faces,
            final_flagged_face_count=initial_flagged_faces,
            initial_flagged_edge_count=initial_flagged_edges,
            final_flagged_edge_count=initial_flagged_edges,
            boundary_recomputation_count=1,
            stopping_reason=stopping_reason,
        )

    replacement = float(np.nextafter(threshold, np.inf))
    if not math.isfinite(replacement):
        raise ValueError("scale_multiplier leaves no finite score for cell removal")
    working = base
    for _ in range(requested_rounds):
        if current.selected_cell_count == 0:
            stopping_reason = "empty_selection"
            break
        flagged = current.boundary_face_risk > selected_risk_threshold
        if not np.any(flagged):
            stopping_reason = "no_flagged_faces"
            break
        owners = np.unique(current.owner_cell_indices[flagged])
        if owners.size == 0:
            stopping_reason = "no_flagged_faces"
            break
        scores[owners] = replacement
        removed_indices.extend(int(owner) for owner in owners)
        removed_rounds.append(int(owners.size))
        working = AdaptiveCellFiltration(
            points=base.points,
            top_simplices=base.top_simplices,
            scores=scores,
            method=f"{base.method}_boundary_owner_intervention",
            diagnostics=base.diagnostics,
            cell_confidence=base.cell_confidence,
            fallback_mask=base.fallback_mask,
            guard_scores=base.guard_scores,
        )
        current = localize(working)

    removed = np.asarray(sorted(removed_indices), dtype=np.int64)
    initial_flagged_faces = int(
        np.count_nonzero(initial.boundary_face_risk > selected_risk_threshold)
    )
    final_flagged_faces = int(
        np.count_nonzero(current.boundary_face_risk > selected_risk_threshold)
    )
    initial_flagged_edges = int(
        np.count_nonzero(initial.boundary_edge_risk > selected_risk_threshold)
    )
    final_flagged_edges = int(
        np.count_nonzero(current.boundary_edge_risk > selected_risk_threshold)
    )
    diagnostics = dict(base.diagnostics)
    diagnostics.update(
        {
            "boundary_intervention_requested_rounds": float(requested_rounds),
            "boundary_intervention_executed_rounds": float(len(removed_rounds)),
            "boundary_intervention_removed_cell_count": float(removed.size),
            "boundary_intervention_removed_cell_fraction": float(
                removed.size / max(initial.selected_cell_count, 1)
            ),
            "boundary_intervention_risk_threshold": selected_risk_threshold,
            "boundary_intervention_initial_selected_cell_count": float(
                initial.selected_cell_count
            ),
            "boundary_intervention_final_selected_cell_count": float(
                current.selected_cell_count
            ),
            "boundary_intervention_initial_flagged_face_count": float(
                initial_flagged_faces
            ),
            "boundary_intervention_final_flagged_face_count": float(
                final_flagged_faces
            ),
            "boundary_intervention_initial_flagged_edge_count": float(
                initial_flagged_edges
            ),
            "boundary_intervention_final_flagged_edge_count": float(
                final_flagged_edges
            ),
            "boundary_intervention_boundary_recomputation_count": float(
                len(removed_rounds) + 1
            ),
        }
    )
    if removed.size:
        working = AdaptiveCellFiltration(
            points=base.points,
            top_simplices=base.top_simplices,
            scores=scores,
            method=f"{base.method}_boundary_owner_intervention",
            diagnostics=diagnostics,
            cell_confidence=base.cell_confidence,
            fallback_mask=base.fallback_mask,
            guard_scores=base.guard_scores,
        )
    return BoundaryOwnerIntervention(
        filtration=working,
        requested_rounds=requested_rounds,
        executed_rounds=len(removed_rounds),
        removed_cell_indices=removed,
        removed_cells_per_round=tuple(removed_rounds),
        risk_threshold=selected_risk_threshold,
        initial_selected_cell_count=initial.selected_cell_count,
        final_selected_cell_count=current.selected_cell_count,
        initial_flagged_face_count=initial_flagged_faces,
        final_flagged_face_count=final_flagged_faces,
        initial_flagged_edge_count=initial_flagged_edges,
        final_flagged_edge_count=final_flagged_edges,
        boundary_recomputation_count=len(removed_rounds) + 1,
        stopping_reason=stopping_reason,
    )


def bridge_penalized_filtration(
    base: AdaptiveCellFiltration,
    bridge_risk: GeometricBridgeRisk,
    *,
    strength: float,
) -> AdaptiveCellFiltration:
    """Apply a label-free soft bridge penalty without changing the cell complex.

    The multiplicative factor is ``1 + strength * max(risk - 1, 0)``. Thus
    strength zero is exactly the base filtration, cells below the diagnostic
    threshold are unchanged, and penalized cell scores can only increase.
    """

    selected_strength = _finite_nonnegative(strength, "strength")
    if bridge_risk.risk.shape != base.scores.shape:
        raise ValueError("bridge risk must align with the base filtration scores")
    excess = np.maximum(bridge_risk.risk - 1.0, 0.0)
    factors = 1.0 + selected_strength * excess
    scores = base.scores * factors
    if not np.all(np.isfinite(scores)):
        raise ValueError("bridge-penalized scores must be finite")

    diagnostics = dict(base.diagnostics)
    diagnostics.update(
        {
            "bridge_penalty_strength": selected_strength,
            "bridge_risk_flagged_fraction": float(np.mean(bridge_risk.risk > 1.0)),
            "bridge_penalty_changed_fraction": float(np.mean(factors > 1.0)),
            "bridge_penalty_factor_max": float(np.max(factors)),
            "bridge_risk_mean": float(np.mean(bridge_risk.risk)),
            "bridge_risk_max": float(np.max(bridge_risk.risk)),
            "bridge_route_parallel_normal": float(
                bridge_risk.route == "parallel_normal"
            ),
        }
    )
    return AdaptiveCellFiltration(
        points=base.points,
        top_simplices=base.top_simplices,
        scores=scores,
        method=f"{base.method}_bridge_penalty",
        diagnostics=diagnostics,
        cell_confidence=base.cell_confidence,
        fallback_mask=base.fallback_mask,
        guard_scores=base.guard_scores,
    )


def density_scaled_filtration(
    filtration: AlphaFiltration,
    *,
    k_neighbors: int,
) -> AdaptiveCellFiltration:
    """B4 score: Euclidean circumradius divided by local kNN spacing."""

    if filtration.ambient_dimension != 3:
        raise ValueError("B4 requires a 3D Delaunay filtration")
    scales = knn_scales(filtration.points, k_neighbors=k_neighbors)
    scores = np.empty(filtration.top_simplices.shape[0], dtype=np.float64)
    for cell_index, cell in enumerate(filtration.top_simplices):
        radius_squared = intrinsic_circumsphere(filtration.points[cell]).radius_squared
        simplex_scale = float(np.exp(np.mean(np.log(scales[cell]))))
        scores[cell_index] = math.sqrt(radius_squared) / simplex_scale
    return AdaptiveCellFiltration(
        points=filtration.points,
        top_simplices=filtration.top_simplices,
        scores=scores,
        method="B4_knn_density_scaled",
        diagnostics={
            "k_neighbors": float(k_neighbors),
            "spacing_min": float(np.min(scales)),
            "spacing_median": float(np.median(scales)),
            "spacing_max": float(np.max(scales)),
        },
    )


def pca_anisotropic_filtration(
    filtration: AlphaFiltration,
    *,
    k_neighbors: int,
    max_normal_penalty: float,
) -> AdaptiveCellFiltration:
    """B5 score in a density-normalized PCA anisotropic SPD metric."""

    if filtration.ambient_dimension != 3:
        raise ValueError("B5 requires a 3D Delaunay filtration")
    if not math.isfinite(max_normal_penalty) or max_normal_penalty < 1.0:
        raise ValueError("max_normal_penalty must be finite and at least one")
    geometry = local_neighborhood_geometry(filtration.points, k_neighbors=k_neighbors)
    metrics = np.empty((filtration.points.shape[0], 3, 3), dtype=np.float64)
    penalties = 1.0 + (max_normal_penalty - 1.0) * geometry.planarity
    for point_index in range(filtration.points.shape[0]):
        eigenbasis = geometry.eigenvectors[point_index]
        metric_eigenvalues = np.array(
            [penalties[point_index] ** 2, 1.0, 1.0],
            dtype=np.float64,
        )
        metric = eigenbasis @ np.diag(metric_eigenvalues) @ eigenbasis.T
        metrics[point_index] = metric / geometry.scales[point_index] ** 2

    scores = np.empty(filtration.top_simplices.shape[0], dtype=np.float64)
    for cell_index, cell in enumerate(filtration.top_simplices):
        simplex_metric = np.mean(metrics[cell], axis=0)
        radius_squared = metric_circumradius_squared(
            filtration.points[cell], simplex_metric
        )
        scores[cell_index] = math.sqrt(radius_squared)
    return AdaptiveCellFiltration(
        points=filtration.points,
        top_simplices=filtration.top_simplices,
        scores=scores,
        method="B5_pca_anisotropic",
        diagnostics={
            "k_neighbors": float(k_neighbors),
            "max_normal_penalty": float(max_normal_penalty),
            "planarity_min": float(np.min(geometry.planarity)),
            "planarity_median": float(np.median(geometry.planarity)),
            "planarity_max": float(np.max(geometry.planarity)),
        },
    )


def pftf_local_metric_filtration(
    filtration: AlphaFiltration,
    *,
    k_neighbors: int,
    relation_gain: float,
    max_condition_number: float,
    density_contrast_scale: float,
    receiver_imbalance_weight: float,
) -> AdaptiveCellFiltration:
    """P1 score from a bounded directed-relation local SPD field.

    P1 uses confidence to blend uncertain point metrics toward the
    density-scaled identity. A hard confidence threshold and exact trusted
    fallback are intentionally reserved for P2.
    """

    if filtration.ambient_dimension != 3:
        raise ValueError("P1 requires a 3D Delaunay filtration")
    relation = pftf_relation_field(
        filtration.points,
        k_neighbors=k_neighbors,
        relation_gain=relation_gain,
        max_condition_number=max_condition_number,
        density_contrast_scale=density_contrast_scale,
        receiver_imbalance_weight=receiver_imbalance_weight,
    )
    field = relation.metric_field
    scores = np.empty(filtration.top_simplices.shape[0], dtype=np.float64)
    simplex_confidence = np.empty_like(scores)
    used_fallback = np.zeros(scores.shape[0], dtype=bool)
    for cell_index, cell in enumerate(filtration.top_simplices):
        simplex_scale = float(np.exp(np.mean(np.log(relation.scales[cell]))))
        decision = field.metric_for_simplex(
            cell,
            confidence_threshold=0.0,
            fallback_metric=np.eye(3, dtype=np.float64) / simplex_scale**2,
        )
        radius_squared = metric_circumradius_squared(
            filtration.points[cell],
            decision.metric,
            minimum_eigenvalue=field.minimum_eigenvalue,
        )
        scores[cell_index] = math.sqrt(radius_squared)
        simplex_confidence[cell_index] = decision.confidence
        used_fallback[cell_index] = decision.used_fallback

    normalized_metrics = field.matrices * relation.scales[:, None, None] ** 2
    metric_eigenvalues = np.linalg.eigvalsh(normalized_metrics)
    metric_condition = metric_eigenvalues[:, -1] / metric_eigenvalues[:, 0]
    return AdaptiveCellFiltration(
        points=filtration.points,
        top_simplices=filtration.top_simplices,
        scores=scores,
        method="P1_pftf_local_spd",
        diagnostics={
            "k_neighbors": float(k_neighbors),
            "relation_gain": float(relation_gain),
            "max_condition_number": float(max_condition_number),
            "density_contrast_scale": float(density_contrast_scale),
            "receiver_imbalance_weight": float(receiver_imbalance_weight),
            "point_confidence_min": float(np.min(field.confidence)),
            "point_confidence_median": float(np.median(field.confidence)),
            "point_confidence_max": float(np.max(field.confidence)),
            "simplex_confidence_min": float(np.min(simplex_confidence)),
            "simplex_confidence_median": float(np.median(simplex_confidence)),
            "simplex_confidence_max": float(np.max(simplex_confidence)),
            "relation_strength_median": float(np.median(relation.relation_strength)),
            "reciprocity_median": float(np.median(relation.reciprocity)),
            "metric_condition_max": float(np.max(metric_condition)),
            "fallback_fraction": float(np.mean(used_fallback)),
        },
        cell_confidence=simplex_confidence,
        fallback_mask=used_fallback,
    )


def pftf_confidence_fallback_filtration(
    filtration: AlphaFiltration,
    *,
    k_neighbors: int,
    relation_gain: float,
    max_condition_number: float,
    density_contrast_scale: float,
    receiver_imbalance_weight: float,
    confidence_threshold: float,
) -> AdaptiveCellFiltration:
    """P2 conservative guard using B4 on low-confidence P1 cells.

    For a low-confidence cell the combined score is the maximum of the P1 and
    trusted B4 scores. Therefore the cell must pass both tests under a shared
    threshold. This is a conservative SciPy prototype, not an exact CGAL
    fallback.
    """

    if not math.isfinite(confidence_threshold) or not (
        0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("confidence_threshold must lie in [0, 1]")
    p1 = pftf_local_metric_filtration(
        filtration,
        k_neighbors=k_neighbors,
        relation_gain=relation_gain,
        max_condition_number=max_condition_number,
        density_contrast_scale=density_contrast_scale,
        receiver_imbalance_weight=receiver_imbalance_weight,
    )
    trusted = density_scaled_filtration(
        filtration,
        k_neighbors=k_neighbors,
    )
    assert p1.cell_confidence is not None
    low_confidence = p1.cell_confidence < confidence_threshold
    scores = p1.scores.copy()
    scores[low_confidence] = np.maximum(
        p1.scores[low_confidence],
        trusted.scores[low_confidence],
    )
    tolerance = 16.0 * np.finfo(np.float64).eps * np.maximum(scores, 1.0)
    guard_violations = low_confidence & (
        (scores + tolerance < p1.scores) | (scores + tolerance < trusted.scores)
    )
    trusted_dominant = low_confidence & (trusted.scores > p1.scores)
    low_confidence_count = int(np.count_nonzero(low_confidence))
    diagnostics = dict(p1.diagnostics)
    diagnostics.update(
        {
            "confidence_threshold": float(confidence_threshold),
            "fallback_fraction": float(np.mean(low_confidence)),
            "fallback_score_dominant_fraction": float(
                np.count_nonzero(trusted_dominant) / max(low_confidence_count, 1)
            ),
            "fallback_guard_violation_count": float(np.count_nonzero(guard_violations)),
        }
    )
    return AdaptiveCellFiltration(
        points=filtration.points,
        top_simplices=filtration.top_simplices,
        scores=scores,
        method="P2_pftf_confidence_b4_guard",
        diagnostics=diagnostics,
        cell_confidence=p1.cell_confidence,
        fallback_mask=low_confidence,
        guard_scores=trusted.scores,
    )
