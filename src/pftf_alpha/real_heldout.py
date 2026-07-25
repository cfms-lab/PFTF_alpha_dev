"""P3 — frozen real-data held-out evaluation (Gate 1).

Loads real meshes, samples point clouds, freezes B4/B5/P1/P2/M1 on a calibration
mesh set, and tests each PFTF candidate against the strict casewise B4/B5 envelope
on a disjoint held-out mesh set. See docs/P3_REAL_HELDOUT_DESIGN.md. This is a real
higher-fidelity comparison on a local corpus, not a licensed leaderboard;
``promotion_supported`` is true only if a candidate clears the declared envelope.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adaptive import (
    AdaptiveCellFiltration,
    density_scaled_filtration,
    pca_anisotropic_filtration,
    pftf_confidence_fallback_filtration,
    pftf_local_metric_filtration,
)
from .baselines import BaselineID, BenchmarkConfig
from .calibration import calibrate_p2_confidence_threshold
from .filtration import AlphaFiltration
from .selection import ObjectiveTerms
from .surface import (
    SurfaceEndpointMetrics,
    SurfaceMesh,
    evaluate_surface,
    mesh_statistics,
    sample_triangle_mesh,
)
from .synthetic import PanelSplit, make_minimal_panel
from .weighted_alpha import PointSubmersionError, weighted_alpha_filtration

FloatArray = np.ndarray
CANDIDATES = (
    BaselineID.P1_PFTF_LOCAL_SPD,
    BaselineID.P2_CONFIDENCE_FALLBACK,
)
M1_METHOD = "M1_weighted_alpha"
M1_WEIGHT_SCALE = 0.375


# --------------------------------------------------------------------------- I/O
def _load_stl(path: Path) -> tuple[FloatArray, np.ndarray]:
    data = path.read_bytes()
    triangles: FloatArray | None = None
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + count * 50:
            raw = np.frombuffer(data[84:], dtype=np.uint8).reshape(count, 50)
            coords = np.frombuffer(
                np.ascontiguousarray(raw[:, 12:48]).tobytes(), dtype="<f4"
            )
            triangles = coords.reshape(count, 3, 3).astype(np.float64)
    if triangles is None:
        rows = [
            [float(x) for x in line.split()[1:4]]
            for line in data.decode("ascii", errors="ignore").splitlines()
            if line.strip().startswith("vertex")
        ]
        triangles = np.asarray(rows, dtype=np.float64).reshape(-1, 3, 3)
    flat = triangles.reshape(-1, 3)
    unique, inverse = np.unique(
        np.round(flat, 6), axis=0, return_inverse=True
    )
    faces = np.asarray(inverse, dtype=np.int64).reshape(-1, 3)
    return unique, faces


def load_mesh(path: str | Path) -> SurfaceMesh:
    """Load a mesh from a ``.npz`` (vertices/facets) or ``.stl`` file."""

    resolved = Path(path)
    if resolved.suffix.lower() == ".npz":
        payload = np.load(resolved)
        vertices = np.asarray(payload["vertices"], dtype=np.float64)
        faces = np.asarray(payload["facets"], dtype=np.int64)
    elif resolved.suffix.lower() == ".stl":
        vertices, faces = _load_stl(resolved)
    else:
        raise ValueError(f"unsupported mesh format: {resolved.suffix}")
    distinct = np.array(
        [len(set(face.tolist())) == 3 for face in faces], dtype=bool
    )
    faces = np.unique(np.sort(faces[distinct], axis=1), axis=0)
    return SurfaceMesh(vertices=vertices, faces=faces)


def _normalized(mesh: SurfaceMesh) -> SurfaceMesh:
    vertices = mesh.vertices
    center = 0.5 * (vertices.max(axis=0) + vertices.min(axis=0))
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("mesh has degenerate extent")
    return SurfaceMesh(vertices=(vertices - center) / diagonal, faces=mesh.faces)


# ---------------------------------------------------------------------- cases
@dataclass(frozen=True)
class RealCase:
    name: str
    points: FloatArray
    reference_points: FloatArray
    characteristic_length: float
    expected_components: int
    expected_surface_betti: tuple[int, int, int]
    density: int
    noise_fraction: float
    seed: int


def make_real_case(
    mesh: SurfaceMesh,
    name: str,
    *,
    observed_count: int,
    reference_count: int,
    noise_fraction: float,
    seed: int,
) -> RealCase:
    normalized = _normalized(mesh)
    statistics = mesh_statistics(normalized)
    reference = sample_triangle_mesh(normalized, reference_count, seed=7 * seed + 1)
    observed = sample_triangle_mesh(normalized, observed_count, seed=7 * seed + 2)
    rng = np.random.default_rng(7 * seed + 3)
    observed = observed + rng.normal(scale=noise_fraction, size=observed.shape)
    return RealCase(
        name=name,
        points=np.ascontiguousarray(observed),
        reference_points=np.ascontiguousarray(reference),
        characteristic_length=1.0,
        expected_components=max(statistics.connected_components, 1),
        expected_surface_betti=(
            statistics.betti_0,
            statistics.betti_1,
            statistics.betti_2,
        ),
        density=observed_count,
        noise_fraction=noise_fraction,
        seed=seed,
    )


# ------------------------------------------------------------------- methods
def _adaptive(
    method: str, points: FloatArray, config: BenchmarkConfig, *, p2_threshold: float
) -> AdaptiveCellFiltration:
    k = config.adaptive_k_neighbors
    if method == M1_METHOD:
        return weighted_alpha_filtration(
            points, k_neighbors=k, weight_scale=M1_WEIGHT_SCALE
        )
    filtration = AlphaFiltration.from_points(points)
    if method == BaselineID.B4_DENSITY_SCALED.value:
        return density_scaled_filtration(filtration, k_neighbors=k)
    if method == BaselineID.B5_PCA_ANISOTROPIC.value:
        return pca_anisotropic_filtration(
            filtration, k_neighbors=k, max_normal_penalty=config.b5_max_normal_penalty
        )
    if method == BaselineID.P1_PFTF_LOCAL_SPD.value:
        return pftf_local_metric_filtration(
            filtration, k_neighbors=k, relation_gain=config.p1_relation_gain,
            max_condition_number=config.p1_max_condition_number,
            density_contrast_scale=config.p1_density_contrast_scale,
            receiver_imbalance_weight=config.p1_receiver_imbalance_weight,
        )
    return pftf_confidence_fallback_filtration(
        filtration, k_neighbors=k, relation_gain=config.p1_relation_gain,
        max_condition_number=config.p1_max_condition_number,
        density_contrast_scale=config.p1_density_contrast_scale,
        receiver_imbalance_weight=config.p1_receiver_imbalance_weight,
        confidence_threshold=p2_threshold,
    )


METHODS = (
    BaselineID.B4_DENSITY_SCALED.value,
    BaselineID.B5_PCA_ANISOTROPIC.value,
    BaselineID.P1_PFTF_LOCAL_SPD.value,
    BaselineID.P2_CONFIDENCE_FALLBACK.value,
    M1_METHOD,
)


def _endpoints(
    adaptive: AdaptiveCellFiltration, multiplier: float, case: RealCase, config
) -> SurfaceEndpointMetrics:
    return evaluate_surface(
        adaptive.surface_at(multiplier), case.reference_points,
        expected_components=case.expected_components,
        characteristic_length=case.characteristic_length,
        expected_betti=case.expected_surface_betti,
        sample_count=config.surface_sample_count,
        threshold_fraction=config.fscore_threshold_fraction, seed=7 * case.seed + 5,
    )


def _objective(endpoints: SurfaceEndpointMetrics, config) -> float:
    terms = ObjectiveTerms(
        geometry=endpoints.normalized_chamfer_squared + endpoints.normalized_hausdorff,
        topology=float(endpoints.component_error),
        stability=0.0,
        complexity=endpoints.nonmanifold_edges / max(endpoints.edges, 1),
    )
    return config.adaptive_weights.apply(terms)


def _freeze_multiplier(
    method: str, cases: Sequence[RealCase], config, *, p2_threshold: float,
    candidate_budget: int,
) -> float:
    adaptives = []
    for case in cases:
        try:
            built = _adaptive(method, case.points, config, p2_threshold=p2_threshold)
        except PointSubmersionError:
            continue
        adaptives.append((case, built))
    pooled = np.concatenate([a.critical_values() for _, a in adaptives])
    positive = pooled[pooled > 0.0]
    lower = float(np.quantile(positive, 0.02))
    upper = float(np.quantile(positive, 0.95))
    candidates = (
        np.asarray([lower]) if upper <= lower
        else np.geomspace(lower, upper, num=candidate_budget)
    )
    best_m, best_obj = float(candidates[0]), float("inf")
    for multiplier in candidates:
        objs = [
            _objective(_endpoints(a, float(multiplier), case, config), config)
            for case, a in adaptives
        ]
        mean_obj = float(np.mean(objs))
        if mean_obj < best_obj:
            best_obj, best_m = mean_obj, float(multiplier)
    return best_m


# ------------------------------------------------------------------ compare
def _geom(ep: SurfaceEndpointMetrics) -> float:
    return ep.normalized_chamfer_squared + ep.normalized_hausdorff


def _topo(ep: SurfaceEndpointMetrics) -> int:
    return ep.component_error + int(ep.betti_error or 0)


@dataclass(frozen=True)
class CandidateGateResult:
    method: str
    case_count: int
    mean_fscore_margin: float
    fscore_margin_ci95: tuple[float, float]
    mean_geometry_margin: float
    topology_excess_sum: int
    geometry_cases_cleared: int
    full_cases_cleared: int
    clears_geometry_envelope: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "case_count": self.case_count,
            "mean_fscore_margin": self.mean_fscore_margin,
            "fscore_margin_ci95": list(self.fscore_margin_ci95),
            "mean_geometry_margin": self.mean_geometry_margin,
            "topology_excess_sum": self.topology_excess_sum,
            "geometry_cases_cleared": self.geometry_cases_cleared,
            "full_cases_cleared": self.full_cases_cleared,
            "clears_geometry_envelope": self.clears_geometry_envelope,
        }


def _bootstrap_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(values, size=values.size, replace=True)))
        for _ in range(1000)
    ]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


@dataclass(frozen=True)
class P3Result:
    calibration_meshes: tuple[str, ...]
    held_out_meshes: tuple[str, ...]
    frozen_multipliers: dict[str, float]
    p2_confidence_threshold: float
    method_mean_fscore: dict[str, float]
    method_mean_geometry: dict[str, float]
    candidates: tuple[CandidateGateResult, ...]
    promotion_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": "pftf_alpha_p3_real_heldout/v1",
            "evaluation_role": "frozen_real_data_held_out",
            "calibration_meshes": list(self.calibration_meshes),
            "held_out_meshes": list(self.held_out_meshes),
            "frozen_multipliers": self.frozen_multipliers,
            "p2_confidence_threshold": self.p2_confidence_threshold,
            "method_mean_fscore": self.method_mean_fscore,
            "method_mean_geometry_loss": self.method_mean_geometry,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "promotion_supported": self.promotion_supported,
            "claim_boundary": (
                "Real higher-fidelity held-out comparison on a local mesh corpus, "
                "not a licensed leaderboard; topology endpoints on complex shells "
                "are noisy. Does not add exact/CGAL parity. promotion_supported is "
                "true only if a candidate clears the declared envelope."
            ),
        }


def evaluate_p3(
    *,
    calibration_meshes: Sequence[tuple[str, str]],
    held_out_meshes: Sequence[tuple[str, str]],
    observed_counts: Sequence[int] = (900, 1500),
    noise_fractions: Sequence[float] = (0.004, 0.012),
    reference_count: int = 8000,
    repeats: int = 2,
    candidate_budget: int = 10,
    seed: int = 20_260_725,
) -> P3Result:
    config = BenchmarkConfig(seed=seed, surface_sample_count=1024)

    def build(meshes, base):
        cases = []
        for mi, (name, path) in enumerate(meshes):
            mesh = load_mesh(path)
            for di, obs in enumerate(observed_counts):
                for ni, noise in enumerate(noise_fractions):
                    for rep in range(repeats):
                        cid = base + 1000 * mi + 100 * di + 10 * ni + rep
                        cases.append(
                            make_real_case(
                                mesh, f"{name}|n{obs}|e{noise}|r{rep}",
                                observed_count=obs, reference_count=reference_count,
                                noise_fraction=noise, seed=cid,
                            )
                        )
        return cases

    cal_cases = build(calibration_meshes, 1)
    hold_cases = build(held_out_meshes, 500_000)

    synthetic_cal = make_minimal_panel(
        split=PanelSplit.CALIBRATION, point_count=96, reference_count=2048, seed=seed
    )
    p2_threshold = calibrate_p2_confidence_threshold(
        synthetic_cal, config=config, target_fallback_fraction=0.25
    ).threshold

    frozen = {
        method: _freeze_multiplier(
            method, cal_cases, config, p2_threshold=p2_threshold,
            candidate_budget=candidate_budget,
        )
        for method in METHODS
    }

    per_method_eps: dict[str, list[SurfaceEndpointMetrics]] = {m: [] for m in METHODS}
    for case in hold_cases:
        for method in METHODS:
            try:
                adaptive = _adaptive(
                    method, case.points, config, p2_threshold=p2_threshold
                )
            except PointSubmersionError:
                adaptive = _adaptive(
                    BaselineID.B4_DENSITY_SCALED.value, case.points, config,
                    p2_threshold=p2_threshold,
                )
                per_method_eps[method].append(
                    _endpoints(adaptive, frozen[BaselineID.B4_DENSITY_SCALED.value],
                               case, config)
                )
                continue
            per_method_eps[method].append(
                _endpoints(adaptive, frozen[method], case, config)
            )

    b4 = per_method_eps[BaselineID.B4_DENSITY_SCALED.value]
    b5 = per_method_eps[BaselineID.B5_PCA_ANISOTROPIC.value]
    candidates = []
    for method in (*[c.value for c in CANDIDATES], M1_METHOD):
        rows = per_method_eps[method]
        count = len(rows)
        fmarg = np.array([
            rows[i].fscore - max(b4[i].fscore, b5[i].fscore) for i in range(count)
        ])
        gmarg = np.array([
            min(_geom(b4[i]), _geom(b5[i])) - _geom(rows[i]) for i in range(count)
        ])
        texc = sum(
            _topo(rows[i]) - min(_topo(b4[i]), _topo(b5[i])) for i in range(count)
        )
        geo_clear = sum(fmarg[i] >= 0 and gmarg[i] >= 0 for i in range(count))
        full_clear = sum(
            fmarg[i] >= 0 and gmarg[i] >= 0
            and _topo(rows[i]) <= min(_topo(b4[i]), _topo(b5[i]))
            for i in range(len(rows))
        )
        candidates.append(CandidateGateResult(
            method=method, case_count=len(rows),
            mean_fscore_margin=float(np.mean(fmarg)),
            fscore_margin_ci95=_bootstrap_ci(fmarg, seed=seed + 1),
            mean_geometry_margin=float(np.mean(gmarg)),
            topology_excess_sum=int(texc),
            geometry_cases_cleared=int(geo_clear),
            full_cases_cleared=int(full_clear),
            clears_geometry_envelope=bool(
                np.mean(fmarg) >= 0 and np.mean(gmarg) >= 0
            ),
        ))

    promotion = any(
        c.clears_geometry_envelope and c.topology_excess_sum <= 0 for c in candidates
    )
    return P3Result(
        calibration_meshes=tuple(n for n, _ in calibration_meshes),
        held_out_meshes=tuple(n for n, _ in held_out_meshes),
        frozen_multipliers=frozen,
        p2_confidence_threshold=p2_threshold,
        method_mean_fscore={
            m: float(np.mean([e.fscore for e in per_method_eps[m]])) for m in METHODS
        },
        method_mean_geometry={
            m: float(np.mean([_geom(e) for e in per_method_eps[m]])) for m in METHODS
        },
        candidates=tuple(candidates),
        promotion_supported=promotion,
    )


_DATA = "D:/__SFTF_Projects(2026)/sftf_Mesh_Data"
_DEFAULT_CAL = (
    ("thingi_271867", f"{_DATA}/271867.npz"),
    ("thingi_56629", f"{_DATA}/56629.npz"),
    ("g5_sphere", f"{_DATA}/g5test/Group_A2_sphere.stl"),
)
_DEFAULT_HOLD = (
    ("thingi_41909", f"{_DATA}/41909.npz"),
    ("thingi_44704", f"{_DATA}/44704.npz"),
    ("thingi_46774", f"{_DATA}/46774.npz"),
    ("g5_torus", f"{_DATA}/g5test/Group_A5_torus.stl"),
    ("g5_cylinder", f"{_DATA}/g5test/Group_A3_cylinder.stl"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the P3 real-data held-out eval.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--candidate-budget", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark-out/p3_real_heldout.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = evaluate_p3(
        calibration_meshes=_DEFAULT_CAL, held_out_meshes=_DEFAULT_HOLD,
        repeats=args.repeats, candidate_budget=args.candidate_budget, seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[p3] promotion_supported={result.promotion_supported}", flush=True)
    for c in result.candidates:
        print(f"[p3] {c.method}: meanF_margin={c.mean_fscore_margin:+.4f} "
              f"geo_clear={c.geometry_cases_cleared}/{c.case_count}", flush=True)
    print(f"Wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
