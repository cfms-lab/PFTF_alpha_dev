"""Train and audit the Phase-48 PFTF-conditioned quadratic-shear map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .filtration import AlphaFiltration
from .integrable_spatial_alpha import QuadraticShearMap3D
from .learned_pftf_coordinate_map_protocol import (
    CALIBRATION_SEEDS,
    CALIBRATION_STRENGTHS,
    FAMILIES,
    GEOMETRY_BASELINE_FEATURES,
    HELD_OUT_SEEDS,
    HELD_OUT_STRENGTHS,
    K_NEIGHBORS,
    MAP_STRENGTH_BOUNDS,
    PFTF_FEATURES,
    POINT_COUNT,
    REFERENCE_COUNT,
    RIDGE_GRID,
    TRAIN_SEEDS,
    TRAIN_STRENGTHS,
    preregister_learned_pftf_coordinate_map,
)
from .pftf import pftf_relation_field
from .synthetic import PanelSplit, SyntheticFamily, make_synthetic_case

FloatArray = NDArray[np.float64]
RESULT_SCHEMA = "pftf_alpha_learned_pftf_coordinate_map_phase48/v1"


@dataclass(frozen=True)
class MapRecoveryCase:
    family: str
    split: str
    seed: int
    true_strength: float
    latent_points: FloatArray
    observed_points: FloatArray


@dataclass(frozen=True)
class StandardizedRidgeModel:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    ridge_penalty: float
    prediction_bounds: tuple[float, float]

    def predict(self, features: ArrayLike) -> FloatArray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("features have the wrong number of columns")
        centered = (values - np.asarray(self.feature_mean)) / np.asarray(
            self.feature_scale
        )
        raw = self.intercept + centered @ np.asarray(self.coefficients)
        return np.clip(raw, *self.prediction_bounds)


@dataclass(frozen=True)
class MethodEndpoints:
    coefficient_mae: float
    coordinate_rms_error: float
    delaunay_top_cell_jaccard: float


@dataclass(frozen=True)
class LearnedPFTFCoordinateMapResult:
    artifact_schema: str
    protocol: dict[str, object]
    train_case_count: int
    calibration_case_count: int
    held_out_case_count: int
    pftf_model: StandardizedRidgeModel
    geometry_model: StandardizedRidgeModel
    identity: MethodEndpoints
    train_mean: MethodEndpoints
    geometry: MethodEndpoints
    pftf: MethodEndpoints
    oracle: MethodEndpoints
    train_mean_strength: float
    maximum_inverse_roundtrip_error: float
    minimum_jacobian_determinant: float
    maximum_jacobian_determinant: float
    bounded_prediction_count: int
    total_prediction_count: int
    construction_gate_passed: bool
    coefficient_gate_passed: bool
    coordinate_gate_passed: bool
    connectivity_gate_passed: bool
    pftf_conditioning_value_supported: bool
    learned_invertible_quadratic_shear_supported: bool
    arbitrary_point_local_spd_complex_supported: bool
    general_nonlinear_map_learner_supported: bool
    point_local_alpha_field_supported: bool
    global_alpha_selection_supported: bool
    reconstruction_advantage_supported: bool
    topology_correctness_supported: bool
    real_scan_transfer_supported: bool
    exact_predicates_supported: bool
    deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalized_cloud(points: ArrayLike) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("points must be a finite array with shape (n, 3)")
    centered = values - np.mean(values, axis=0)
    rms_radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if not math.isfinite(rms_radius) or rms_radius <= 0.0:
        raise ValueError("points must have positive RMS radius")
    return np.ascontiguousarray(centered / rms_radius)


def make_map_recovery_cases(
    *,
    split: PanelSplit,
    seeds: tuple[int, ...],
    strengths: tuple[float, ...],
    families: tuple[str, ...] = FAMILIES,
    point_count: int = POINT_COUNT,
    reference_count: int = REFERENCE_COUNT,
) -> tuple[MapRecoveryCase, ...]:
    cases: list[MapRecoveryCase] = []
    for family_name in families:
        family = SyntheticFamily(family_name)
        for seed in seeds:
            source = make_synthetic_case(
                family,
                split=split,
                point_count=point_count,
                reference_count=reference_count,
                seed=seed,
            )
            latent = _normalized_cloud(source.points)
            for strength in strengths:
                if not MAP_STRENGTH_BOUNDS[0] <= strength <= MAP_STRENGTH_BOUNDS[1]:
                    raise ValueError("strength lies outside the declared map family")
                observed = QuadraticShearMap3D(-strength).forward(latent)
                cases.append(
                    MapRecoveryCase(
                        family=family.value,
                        split=split.value,
                        seed=int(seed),
                        true_strength=float(strength),
                        latent_points=latent,
                        observed_points=observed,
                    )
                )
    return tuple(cases)


def pftf_summary_features(
    points: ArrayLike,
    *,
    k_neighbors: int = K_NEIGHBORS,
) -> FloatArray:
    field = pftf_relation_field(points, k_neighbors=k_neighbors)
    xy = field.relation_tensors[:, 0, 1]
    scales = np.log(field.scales)
    return np.asarray(
        (
            np.mean(xy),
            np.std(xy),
            np.median(field.relation_strength),
            np.quantile(field.relation_strength, 0.90),
            np.mean(field.metric_field.confidence),
            np.mean(field.reciprocity),
            np.std(scales),
        ),
        dtype=np.float64,
    )


def geometry_summary_features(points: ArrayLike) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("points must be a finite array with shape (n, 3)")
    centered = values - np.mean(values, axis=0)
    covariance = centered.T @ centered / len(centered)
    variances = np.maximum(np.diag(covariance), np.finfo(np.float64).tiny)
    standard_deviations = np.sqrt(variances)
    normalized_xy = covariance[0, 1] / (
        standard_deviations[0] * standard_deviations[1]
    )
    standardized = centered / standard_deviations
    axis_variance_ratio = float(np.max(variances) / np.min(variances))
    return np.asarray(
        (
            normalized_xy,
            np.mean(standardized[:, 0] ** 3),
            np.mean(standardized[:, 1] ** 3),
            axis_variance_ratio,
        ),
        dtype=np.float64,
    )


def _feature_matrix(
    cases: tuple[MapRecoveryCase, ...],
    feature_kind: str,
) -> FloatArray:
    extractor = {
        "pftf": pftf_summary_features,
        "geometry": geometry_summary_features,
    }.get(feature_kind)
    if extractor is None:
        raise ValueError("feature_kind must be 'pftf' or 'geometry'")
    return np.vstack([extractor(case.observed_points) for case in cases])


def _fit_ridge(
    features: FloatArray,
    targets: FloatArray,
    *,
    feature_names: tuple[str, ...],
    penalty: float,
) -> StandardizedRidgeModel:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != labels.shape[0]:
        raise ValueError("features and targets must have matching rows")
    feature_mean = np.mean(values, axis=0)
    feature_scale = np.std(values, axis=0)
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    standardized = (values - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(labels)), standardized))
    gram = design.T @ design
    regularizer = np.diag((0.0, *([float(penalty)] * values.shape[1])))
    solution = np.linalg.pinv(gram + regularizer) @ design.T @ labels
    return StandardizedRidgeModel(
        feature_names=feature_names,
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        intercept=float(solution[0]),
        coefficients=tuple(float(value) for value in solution[1:]),
        ridge_penalty=float(penalty),
        prediction_bounds=MAP_STRENGTH_BOUNDS,
    )


def _select_model(
    train_features: FloatArray,
    train_targets: FloatArray,
    calibration_features: FloatArray,
    calibration_targets: FloatArray,
    *,
    feature_names: tuple[str, ...],
) -> StandardizedRidgeModel:
    candidates = tuple(
        _fit_ridge(
            train_features,
            train_targets,
            feature_names=feature_names,
            penalty=penalty,
        )
        for penalty in RIDGE_GRID
    )
    return min(
        candidates,
        key=lambda model: (
            float(
                np.mean(
                    np.abs(model.predict(calibration_features) - calibration_targets)
                )
            ),
            -model.ridge_penalty,
        ),
    )


def _top_cell_set(points: FloatArray) -> set[tuple[int, ...]]:
    cells = np.sort(AlphaFiltration.from_points(points).top_simplices, axis=1)
    return {tuple(int(index) for index in cell) for cell in cells}


def _jaccard(
    first: set[tuple[int, ...]],
    second: set[tuple[int, ...]],
) -> float:
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def _method_endpoints(
    cases: tuple[MapRecoveryCase, ...],
    predictions: FloatArray,
    latent_top_cells: tuple[set[tuple[int, ...]], ...],
) -> MethodEndpoints:
    coefficient_errors: list[float] = []
    coordinate_errors: list[float] = []
    jaccards: list[float] = []
    for case, prediction, latent_cells in zip(
        cases, predictions, latent_top_cells, strict=True
    ):
        corrected = QuadraticShearMap3D(float(prediction)).forward(
            case.observed_points
        )
        coefficient_errors.append(abs(float(prediction) - case.true_strength))
        coordinate_errors.append(
            float(
                np.sqrt(
                    np.mean(np.sum((corrected - case.latent_points) ** 2, axis=1))
                )
            )
        )
        jaccards.append(_jaccard(_top_cell_set(corrected), latent_cells))
    return MethodEndpoints(
        coefficient_mae=float(np.mean(coefficient_errors)),
        coordinate_rms_error=float(np.mean(coordinate_errors)),
        delaunay_top_cell_jaccard=float(np.mean(jaccards)),
    )


def _strictly_better(
    candidate: float,
    first: float,
    second: float,
    *,
    higher_is_better: bool = False,
) -> bool:
    if higher_is_better:
        return candidate > first and candidate > second
    return candidate < first and candidate < second


@lru_cache(maxsize=1)
def evaluate_learned_pftf_coordinate_map() -> LearnedPFTFCoordinateMapResult:
    protocol = preregister_learned_pftf_coordinate_map()
    train_cases = make_map_recovery_cases(
        split=PanelSplit.TRAIN,
        seeds=TRAIN_SEEDS,
        strengths=TRAIN_STRENGTHS,
    )
    calibration_cases = make_map_recovery_cases(
        split=PanelSplit.CALIBRATION,
        seeds=CALIBRATION_SEEDS,
        strengths=CALIBRATION_STRENGTHS,
    )
    held_out_cases = make_map_recovery_cases(
        split=PanelSplit.HELD_OUT,
        seeds=HELD_OUT_SEEDS,
        strengths=HELD_OUT_STRENGTHS,
    )
    train_targets = np.asarray(
        [case.true_strength for case in train_cases], dtype=np.float64
    )
    calibration_targets = np.asarray(
        [case.true_strength for case in calibration_cases], dtype=np.float64
    )
    held_out_targets = np.asarray(
        [case.true_strength for case in held_out_cases], dtype=np.float64
    )

    train_pftf = _feature_matrix(train_cases, "pftf")
    calibration_pftf = _feature_matrix(calibration_cases, "pftf")
    held_out_pftf = _feature_matrix(held_out_cases, "pftf")
    train_geometry = _feature_matrix(train_cases, "geometry")
    calibration_geometry = _feature_matrix(calibration_cases, "geometry")
    held_out_geometry = _feature_matrix(held_out_cases, "geometry")

    pftf_model = _select_model(
        train_pftf,
        train_targets,
        calibration_pftf,
        calibration_targets,
        feature_names=PFTF_FEATURES,
    )
    geometry_model = _select_model(
        train_geometry,
        train_targets,
        calibration_geometry,
        calibration_targets,
        feature_names=GEOMETRY_BASELINE_FEATURES,
    )
    pftf_predictions = pftf_model.predict(held_out_pftf)
    geometry_predictions = geometry_model.predict(held_out_geometry)
    train_mean_strength = float(np.mean(train_targets))
    identity_predictions = np.zeros_like(held_out_targets)
    train_mean_predictions = np.full_like(held_out_targets, train_mean_strength)

    latent_top_cells = tuple(
        _top_cell_set(case.latent_points) for case in held_out_cases
    )
    identity = _method_endpoints(
        held_out_cases, identity_predictions, latent_top_cells
    )
    train_mean = _method_endpoints(
        held_out_cases, train_mean_predictions, latent_top_cells
    )
    geometry = _method_endpoints(
        held_out_cases, geometry_predictions, latent_top_cells
    )
    pftf = _method_endpoints(held_out_cases, pftf_predictions, latent_top_cells)
    oracle = MethodEndpoints(
        coefficient_mae=0.0,
        coordinate_rms_error=0.0,
        delaunay_top_cell_jaccard=1.0,
    )

    roundtrip_errors: list[float] = []
    determinants: list[float] = []
    bounded_count = 0
    for case, prediction in zip(held_out_cases, pftf_predictions, strict=True):
        coordinate_map = QuadraticShearMap3D(float(prediction))
        transformed = coordinate_map.forward(case.observed_points)
        recovered = coordinate_map.inverse(transformed)
        roundtrip_errors.append(float(np.max(np.abs(recovered - case.observed_points))))
        determinants.extend(
            float(value)
            for value in np.linalg.det(
                coordinate_map.jacobians(case.observed_points)
            )
        )
        bounded_count += int(
            MAP_STRENGTH_BOUNDS[0]
            <= float(prediction)
            <= MAP_STRENGTH_BOUNDS[1]
        )

    maximum_roundtrip_error = max(roundtrip_errors)
    minimum_determinant = min(determinants)
    maximum_determinant = max(determinants)
    construction_gate = (
        bounded_count == len(held_out_cases)
        and maximum_roundtrip_error <= 1.0e-12
        and minimum_determinant > 0.0
        and maximum_determinant == 1.0
    )
    coefficient_gate = _strictly_better(
        pftf.coefficient_mae,
        train_mean.coefficient_mae,
        geometry.coefficient_mae,
    )
    coordinate_gate = _strictly_better(
        pftf.coordinate_rms_error,
        train_mean.coordinate_rms_error,
        geometry.coordinate_rms_error,
    )
    connectivity_gate = _strictly_better(
        pftf.delaunay_top_cell_jaccard,
        train_mean.delaunay_top_cell_jaccard,
        geometry.delaunay_top_cell_jaccard,
        higher_is_better=True,
    )
    pftf_supported = (
        construction_gate
        and coefficient_gate
        and coordinate_gate
        and connectivity_gate
    )
    return LearnedPFTFCoordinateMapResult(
        artifact_schema=RESULT_SCHEMA,
        protocol=protocol.to_dict(),
        train_case_count=len(train_cases),
        calibration_case_count=len(calibration_cases),
        held_out_case_count=len(held_out_cases),
        pftf_model=pftf_model,
        geometry_model=geometry_model,
        identity=identity,
        train_mean=train_mean,
        geometry=geometry,
        pftf=pftf,
        oracle=oracle,
        train_mean_strength=train_mean_strength,
        maximum_inverse_roundtrip_error=maximum_roundtrip_error,
        minimum_jacobian_determinant=minimum_determinant,
        maximum_jacobian_determinant=maximum_determinant,
        bounded_prediction_count=bounded_count,
        total_prediction_count=len(held_out_cases),
        construction_gate_passed=construction_gate,
        coefficient_gate_passed=coefficient_gate,
        coordinate_gate_passed=coordinate_gate,
        connectivity_gate_passed=connectivity_gate,
        pftf_conditioning_value_supported=pftf_supported,
        learned_invertible_quadratic_shear_supported=pftf_supported,
        arbitrary_point_local_spd_complex_supported=False,
        general_nonlinear_map_learner_supported=False,
        point_local_alpha_field_supported=False,
        global_alpha_selection_supported=False,
        reconstruction_advantage_supported=False,
        topology_correctness_supported=False,
        real_scan_transfer_supported=False,
        exact_predicates_supported=False,
        deployment_supported=False,
    )


def write_result(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        evaluate_learned_pftf_coordinate_map().to_dict(),
        indent=2,
        sort_keys=True,
    ) + "\n"
    output.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-out/learned_pftf_coordinate_map_phase48.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_learned_pftf_coordinate_map()
    digest = write_result(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")
    print(f"pftf_ridge_penalty={result.pftf_model.ridge_penalty:.12g}")
    print(f"geometry_ridge_penalty={result.geometry_model.ridge_penalty:.12g}")
    print(f"pftf_coefficient_mae={result.pftf.coefficient_mae:.12g}")
    print(f"geometry_coefficient_mae={result.geometry.coefficient_mae:.12g}")
    print(f"pftf_coordinate_rms={result.pftf.coordinate_rms_error:.12g}")
    print(f"geometry_coordinate_rms={result.geometry.coordinate_rms_error:.12g}")
    print(f"pftf_top_cell_jaccard={result.pftf.delaunay_top_cell_jaccard:.12g}")
    print(
        "geometry_top_cell_jaccard="
        f"{result.geometry.delaunay_top_cell_jaccard:.12g}"
    )
    print(
        "pftf_conditioning_value_supported="
        f"{str(result.pftf_conditioning_value_supported).lower()}"
    )


if __name__ == "__main__":
    main()
