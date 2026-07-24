"""Surface extraction and benchmark endpoints for 3D baselines."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .filtration import AlphaFiltration, BoundaryMode

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SurfaceMesh:
    """Triangular surface referencing a point array."""

    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("surface vertices must have shape (n, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("surface faces must have shape (m, 3)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("surface vertices must be finite")
        if faces.size and (np.min(faces) < 0 or np.max(faces) >= vertices.shape[0]):
            raise ValueError("surface face index is out of range")
        if any(len(set(face)) != 3 for face in faces.tolist()):
            raise ValueError("surface faces must contain three distinct vertices")
        canonical_faces = np.sort(faces, axis=1)
        if canonical_faces.shape[0] != np.unique(canonical_faces, axis=0).shape[0]:
            raise ValueError("surface faces must be unique as unoriented simplices")
        object.__setattr__(self, "vertices", np.ascontiguousarray(vertices))
        object.__setattr__(self, "faces", np.ascontiguousarray(faces))


@dataclass(frozen=True)
class MeshStatistics:
    used_vertices: int
    edges: int
    faces: int
    connected_components: int
    euler_characteristic: int
    betti_0: int
    betti_1: int
    betti_2: int
    boundary_edges: int
    nonmanifold_edges: int
    watertight: bool


@dataclass(frozen=True)
class SurfaceDistanceMetrics:
    chamfer_squared: float
    hausdorff: float
    precision: float
    recall: float
    fscore: float


@dataclass(frozen=True)
class SurfaceEndpointMetrics:
    """Geometry and combinatorial surface-topology endpoints."""

    chamfer_squared: float
    normalized_chamfer_squared: float
    hausdorff: float
    normalized_hausdorff: float
    precision: float
    recall: float
    fscore: float
    connected_components: int
    component_error: int
    betti_0: int
    betti_1: int
    betti_2: int
    expected_betti_0: int | None
    expected_betti_1: int | None
    expected_betti_2: int | None
    betti_error: int | None
    false_bridges: int
    false_splits: int
    used_vertices: int
    edges: int
    faces: int
    euler_characteristic: int
    boundary_edges: int
    nonmanifold_edges: int
    watertight: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


def convex_hull_surface(points: ArrayLike) -> SurfaceMesh:
    """B0: triangular boundary of the 3D convex hull."""

    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("convex_hull_surface requires points with shape (n, 3)")
    try:
        hull = ConvexHull(vertices)
    except QhullError as error:
        raise ValueError(f"convex hull construction failed: {error}") from error
    return SurfaceMesh(vertices=vertices, faces=np.asarray(hull.simplices))


def alpha_surface(
    filtration: AlphaFiltration,
    alpha_squared: float,
    *,
    mode: BoundaryMode | str = BoundaryMode.REGULARIZED,
) -> SurfaceMesh:
    """Extract a triangular boundary from a 3D alpha filtration."""

    if filtration.ambient_dimension != 3:
        raise ValueError("surface extraction requires a 3D alpha filtration")
    faces = filtration.boundary_facets_at(alpha_squared, mode=mode)
    return SurfaceMesh(vertices=filtration.points, faces=faces)


def mesh_statistics(mesh: SurfaceMesh) -> MeshStatistics:
    """Compute face connectivity, edge incidence, and GF(2) Betti numbers."""

    if mesh.faces.shape[0] == 0:
        return MeshStatistics(
            used_vertices=0,
            edges=0,
            faces=0,
            connected_components=0,
            euler_characteristic=0,
            betti_0=0,
            betti_1=0,
            betti_2=0,
            boundary_edges=0,
            nonmanifold_edges=0,
            watertight=False,
        )

    used_vertices = np.unique(mesh.faces)
    parent = {int(vertex): int(vertex) for vertex in used_vertices}

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_counts: Counter[tuple[int, int]] = Counter()
    for face in mesh.faces:
        first, second, third = (int(vertex) for vertex in face)
        union(first, second)
        union(second, third)
        union(third, first)
        edge_counts.update(
            (
                tuple(sorted((first, second))),
                tuple(sorted((second, third))),
                tuple(sorted((third, first))),
            )
        )

    components = len({find(int(vertex)) for vertex in used_vertices})
    boundary_edges = sum(count == 1 for count in edge_counts.values())
    nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
    euler_characteristic = len(used_vertices) - len(edge_counts) + mesh.faces.shape[0]

    edge_indices = {edge: index for index, edge in enumerate(edge_counts)}
    boundary_columns: list[int] = []
    for face in mesh.faces:
        first, second, third = (int(vertex) for vertex in face)
        column = 0
        for edge in (
            tuple(sorted((first, second))),
            tuple(sorted((second, third))),
            tuple(sorted((third, first))),
        ):
            column ^= 1 << edge_indices[edge]
        boundary_columns.append(column)
    rank_boundary_2 = _gf2_rank(boundary_columns)
    rank_boundary_1 = len(used_vertices) - components
    betti_0 = components
    betti_1 = len(edge_counts) - rank_boundary_1 - rank_boundary_2
    betti_2 = mesh.faces.shape[0] - rank_boundary_2

    return MeshStatistics(
        used_vertices=int(len(used_vertices)),
        edges=int(len(edge_counts)),
        faces=int(mesh.faces.shape[0]),
        connected_components=components,
        euler_characteristic=int(euler_characteristic),
        betti_0=int(betti_0),
        betti_1=int(betti_1),
        betti_2=int(betti_2),
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        watertight=boundary_edges == 0 and nonmanifold_edges == 0,
    )


def _gf2_rank(columns: list[int]) -> int:
    """Rank of bit-packed vectors over GF(2)."""

    pivots: dict[int, int] = {}
    for column in columns:
        value = int(column)
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def sample_triangle_mesh(
    mesh: SurfaceMesh,
    sample_count: int,
    *,
    seed: int = 0,
) -> FloatArray:
    """Area-weighted deterministic sampling of a triangular surface."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if mesh.faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > np.finfo(np.float64).eps
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    triangles = triangles[valid]
    areas = areas[valid]

    rng = np.random.default_rng(seed)
    selected = rng.choice(
        triangles.shape[0],
        size=sample_count,
        replace=True,
        p=areas / np.sum(areas),
    )
    chosen = triangles[selected]
    first = np.sqrt(rng.uniform(size=sample_count))
    second = rng.uniform(size=sample_count)
    weight_a = 1.0 - first
    weight_b = first * (1.0 - second)
    weight_c = first * second
    return (
        weight_a[:, None] * chosen[:, 0]
        + weight_b[:, None] * chosen[:, 1]
        + weight_c[:, None] * chosen[:, 2]
    )


def surface_distance_metrics(
    predicted_points: ArrayLike,
    reference_points: ArrayLike,
    *,
    threshold: float,
) -> SurfaceDistanceMetrics:
    """Symmetric Chamfer, Hausdorff, and thresholded surface F-score."""

    predicted = np.asarray(predicted_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    for name, values in (
        ("predicted_points", predicted),
        ("reference_points", reference),
    ):
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError(f"{name} must have non-empty shape (n, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite coordinates")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")

    reference_tree = cKDTree(reference)
    predicted_tree = cKDTree(predicted)
    predicted_to_reference = reference_tree.query(predicted, workers=1)[0]
    reference_to_predicted = predicted_tree.query(reference, workers=1)[0]
    precision = float(np.mean(predicted_to_reference <= threshold))
    recall = float(np.mean(reference_to_predicted <= threshold))
    fscore = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return SurfaceDistanceMetrics(
        chamfer_squared=float(
            np.mean(predicted_to_reference**2) + np.mean(reference_to_predicted**2)
        ),
        hausdorff=float(
            max(
                np.max(predicted_to_reference),
                np.max(reference_to_predicted),
            )
        ),
        precision=precision,
        recall=recall,
        fscore=fscore,
    )


def evaluate_surface(
    mesh: SurfaceMesh,
    reference_points: ArrayLike,
    *,
    expected_components: int,
    characteristic_length: float,
    sample_count: int,
    threshold_fraction: float,
    seed: int,
    expected_betti: tuple[int, int, int] | None = None,
) -> SurfaceEndpointMetrics:
    """Evaluate a mesh against a dense reference without hiding empty outputs."""

    if expected_components < 1:
        raise ValueError("expected_components must be positive")
    if expected_betti is not None and (
        len(expected_betti) != 3
        or any(
            not isinstance(value, (int, np.integer)) or value < 0
            for value in expected_betti
        )
    ):
        raise ValueError("expected_betti must contain three non-negative integers")
    if not math.isfinite(characteristic_length) or characteristic_length <= 0.0:
        raise ValueError("characteristic_length must be finite and positive")
    if not math.isfinite(threshold_fraction) or threshold_fraction <= 0.0:
        raise ValueError("threshold_fraction must be finite and positive")

    statistics = mesh_statistics(mesh)
    predicted = sample_triangle_mesh(mesh, sample_count, seed=seed)
    if predicted.shape[0] == 0:
        distances = SurfaceDistanceMetrics(
            chamfer_squared=4.0 * characteristic_length**2,
            hausdorff=2.0 * characteristic_length,
            precision=0.0,
            recall=0.0,
            fscore=0.0,
        )
    else:
        distances = surface_distance_metrics(
            predicted,
            reference_points,
            threshold=threshold_fraction * characteristic_length,
        )

    component_error = abs(statistics.connected_components - expected_components)
    expected_betti_values = (
        (None, None, None) if expected_betti is None else expected_betti
    )
    betti_error = (
        None
        if expected_betti is None
        else sum(
            abs(actual - expected)
            for actual, expected in zip(
                (statistics.betti_0, statistics.betti_1, statistics.betti_2),
                expected_betti,
                strict=True,
            )
        )
    )
    return SurfaceEndpointMetrics(
        chamfer_squared=distances.chamfer_squared,
        normalized_chamfer_squared=(
            distances.chamfer_squared / characteristic_length**2
        ),
        hausdorff=distances.hausdorff,
        normalized_hausdorff=distances.hausdorff / characteristic_length,
        precision=distances.precision,
        recall=distances.recall,
        fscore=distances.fscore,
        connected_components=statistics.connected_components,
        component_error=component_error,
        betti_0=statistics.betti_0,
        betti_1=statistics.betti_1,
        betti_2=statistics.betti_2,
        expected_betti_0=expected_betti_values[0],
        expected_betti_1=expected_betti_values[1],
        expected_betti_2=expected_betti_values[2],
        betti_error=betti_error,
        false_bridges=max(expected_components - statistics.connected_components, 0),
        false_splits=max(statistics.connected_components - expected_components, 0),
        used_vertices=statistics.used_vertices,
        edges=statistics.edges,
        faces=statistics.faces,
        euler_characteristic=statistics.euler_characteristic,
        boundary_edges=statistics.boundary_edges,
        nonmanifold_edges=statistics.nonmanifold_edges,
        watertight=statistics.watertight,
    )
