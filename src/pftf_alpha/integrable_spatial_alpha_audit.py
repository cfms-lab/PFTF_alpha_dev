"""Run the frozen Phase-47 integrable nonlinear spatial-alpha audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .affine_spd_alpha import global_affine_spd_alpha
from .filtration import AlphaFiltration
from .integrable_spatial_alpha import (
    AffineCoordinateMap3D,
    NonIntegrableJacobianError,
    QuadraticShearMap3D,
    audit_jacobian_integrability,
    coordinate_map_spatial_alpha,
    numerical_jacobians,
    require_integrable_jacobian_field,
)
from .integrable_spatial_alpha_protocol import (
    FINITE_DIFFERENCE_STEP,
    INTEGRABILITY_ABSOLUTE_TOLERANCE,
    INVERSE_ROUNDTRIP_TOLERANCE,
    JACOBIAN_ABSOLUTE_TOLERANCE,
    MINIMUM_JACOBIAN_DETERMINANT,
    MINIMUM_METRIC_VARIATION,
    NONINTEGRABLE_FIELD_STRENGTH,
    QUADRATIC_SHEAR_STRENGTH,
    SCORE_ABSOLUTE_TOLERANCE,
    SCORE_RELATIVE_TOLERANCE,
    preregister_integrable_spatial_alpha,
)

RESULT_SCHEMA = "pftf_alpha_integrable_spatial_alpha_audit_phase47/v1"


@dataclass(frozen=True)
class FiltrationComparison:
    connectivity_equal: bool
    simplex_keys_equal: bool
    scores_equal: bool
    maximum_absolute_score_error: float
    maximum_relative_score_error: float

    @property
    def passed(self) -> bool:
        return self.connectivity_equal and self.simplex_keys_equal and self.scores_equal


@dataclass(frozen=True)
class SpatialAlphaControlResult:
    name: str
    passed: bool
    comparison: FiltrationComparison | None
    observed_value: float
    threshold: float
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IntegrableSpatialAlphaAuditResult:
    artifact_schema: str
    protocol_schema: str
    protocol_sha256: str
    controls: tuple[SpatialAlphaControlResult, ...]
    passed_control_count: int
    total_control_count: int
    connectivity_symmetric_difference_count: int
    generic_rotation_connectivity_equal: bool
    generic_rotation_maximum_relative_score_error: float
    analytic_integrable_spatial_spd_complex_supported: bool
    arbitrary_point_local_spd_complex_supported: bool
    point_local_alpha_field_supported: bool
    pftf_conditioned_spatial_alpha_supported: bool
    generic_floating_rigid_score_invariance_supported: bool
    exact_integrable_spatial_predicates_supported: bool
    spatial_alpha_reconstruction_advantage_supported: bool
    spatial_alpha_topology_correctness_supported: bool
    spatial_alpha_real_scan_transfer_supported: bool
    spatial_alpha_deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["controls"] = [control.to_dict() for control in self.controls]
        return payload


def _canonical_cells(filtration: AlphaFiltration) -> set[tuple[int, ...]]:
    return {
        tuple(sorted(int(vertex) for vertex in cell))
        for cell in filtration.top_simplices
    }


def _record_values(filtration: AlphaFiltration) -> dict[tuple[int, ...], float]:
    return {record.vertices: record.alpha_squared for record in filtration.records}


def _compare_filtrations(
    left: AlphaFiltration,
    right: AlphaFiltration,
) -> FiltrationComparison:
    connectivity_equal = _canonical_cells(left) == _canonical_cells(right)
    left_values = _record_values(left)
    right_values = _record_values(right)
    simplex_keys_equal = left_values.keys() == right_values.keys()
    if simplex_keys_equal:
        keys = sorted(left_values)
        absolute_errors = np.asarray(
            [abs(left_values[key] - right_values[key]) for key in keys]
        )
        denominators = np.asarray(
            [max(abs(right_values[key]), np.finfo(float).tiny) for key in keys]
        )
        relative_errors = absolute_errors / denominators
        maximum_absolute = float(np.max(absolute_errors, initial=0.0))
        maximum_relative = float(np.max(relative_errors, initial=0.0))
        scores_equal = bool(
            all(
                np.isclose(
                    left_values[key],
                    right_values[key],
                    rtol=SCORE_RELATIVE_TOLERANCE,
                    atol=SCORE_ABSOLUTE_TOLERANCE,
                )
                for key in keys
            )
        )
    else:
        maximum_absolute = float("inf")
        maximum_relative = float("inf")
        scores_equal = False
    return FiltrationComparison(
        connectivity_equal=connectivity_equal,
        simplex_keys_equal=simplex_keys_equal,
        scores_equal=scores_equal,
        maximum_absolute_score_error=maximum_absolute,
        maximum_relative_score_error=maximum_relative,
    )


def _nonintegrable_jacobians(points: np.ndarray) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    result = np.repeat(np.eye(3)[None, :, :], len(point_array), axis=0)
    result[:, 0, 1] = NONINTEGRABLE_FIELD_STRENGTH * point_array[:, 1]
    return result


def _protocol_digest() -> str:
    text = json.dumps(
        preregister_integrable_spatial_alpha().to_dict(),
        indent=2,
        sort_keys=True,
    ) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate_integrable_spatial_alpha_audit() -> IntegrableSpatialAlphaAuditResult:
    protocol = preregister_integrable_spatial_alpha()
    points = np.random.default_rng(protocol.audit_seed).normal(
        size=(protocol.point_count, protocol.ambient_dimension)
    )
    euclidean = AlphaFiltration.from_points(points)

    zero_map = QuadraticShearMap3D(strength=0.0)
    zero_construction = coordinate_map_spatial_alpha(
        points,
        zero_map,
        minimum_jacobian_determinant=MINIMUM_JACOBIAN_DETERMINANT,
        inverse_roundtrip_tolerance=INVERSE_ROUNDTRIP_TOLERANCE,
    )
    zero_comparison = _compare_filtrations(zero_construction.filtration, euclidean)

    affine_factor = np.asarray(
        [[1.20, 0.0, 0.0], [0.15, 0.90, 0.0], [0.05, -0.10, 1.10]],
        dtype=np.float64,
    )
    affine_map = AffineCoordinateMap3D(
        factor=affine_factor,
        offset=np.asarray([0.25, -0.40, 0.10]),
    )
    affine_construction = coordinate_map_spatial_alpha(
        points,
        affine_map,
        minimum_jacobian_determinant=MINIMUM_JACOBIAN_DETERMINANT,
        inverse_roundtrip_tolerance=INVERSE_ROUNDTRIP_TOLERANCE,
    )
    phase46 = global_affine_spd_alpha(
        points,
        affine_factor @ affine_factor.T,
    )
    affine_comparison = _compare_filtrations(
        affine_construction.filtration,
        phase46.filtration,
    )

    shear_map = QuadraticShearMap3D(strength=QUADRATIC_SHEAR_STRENGTH)
    shear = coordinate_map_spatial_alpha(
        points,
        shear_map,
        minimum_jacobian_determinant=MINIMUM_JACOBIAN_DETERMINANT,
        inverse_roundtrip_tolerance=INVERSE_ROUNDTRIP_TOLERANCE,
    )
    explicit = AlphaFiltration.from_points(shear_map.forward(points))
    explicit_comparison = _compare_filtrations(shear.filtration, explicit)

    numerical = numerical_jacobians(
        shear_map,
        points,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
    )
    maximum_jacobian_error = float(np.max(np.abs(shear.jacobians - numerical)))

    euclidean_cells = _canonical_cells(euclidean)
    shear_cells = _canonical_cells(shear.filtration)
    connectivity_difference = len(euclidean_cells.symmetric_difference(shear_cells))

    shear_integrability = audit_jacobian_integrability(
        shear_map.jacobians,
        points,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        absolute_tolerance=INTEGRABILITY_ABSOLUTE_TOLERANCE,
        minimum_jacobian_determinant=MINIMUM_JACOBIAN_DETERMINANT,
    )
    nonintegrable = audit_jacobian_integrability(
        _nonintegrable_jacobians,
        points,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        absolute_tolerance=INTEGRABILITY_ABSOLUTE_TOLERANCE,
        minimum_jacobian_determinant=MINIMUM_JACOBIAN_DETERMINANT,
    )
    observed_rejection = False
    try:
        require_integrable_jacobian_field(
            _nonintegrable_jacobians,
            points,
            finite_difference_step=FINITE_DIFFERENCE_STEP,
            absolute_tolerance=INTEGRABILITY_ABSOLUTE_TOLERANCE,
            minimum_jacobian_determinant=MINIMUM_JACOBIAN_DETERMINANT,
        )
    except NonIntegrableJacobianError:
        observed_rejection = True

    generic_angle = 0.43
    generic_cosine = float(np.cos(generic_angle))
    generic_sine = float(np.sin(generic_angle))
    generic_rotation = np.asarray(
        [
            [generic_cosine, -generic_sine, 0.0],
            [generic_sine, generic_cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    generic_rigid = AlphaFiltration.from_points(
        shear.transformed_points @ generic_rotation
    )
    generic_rigid_comparison = _compare_filtrations(
        shear.filtration,
        generic_rigid,
    )

    exact_rotation = np.asarray(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rigid_points = shear.transformed_points @ exact_rotation
    rigid = AlphaFiltration.from_points(rigid_points)
    rigid_comparison = _compare_filtrations(shear.filtration, rigid)

    controls = (
        SpatialAlphaControlResult(
            name="zero_shear_reproduces_euclidean",
            passed=zero_comparison.passed,
            comparison=zero_comparison,
            observed_value=zero_comparison.maximum_relative_score_error,
            threshold=SCORE_RELATIVE_TOLERANCE,
            detail="zero-strength nonlinear-map boundary",
        ),
        SpatialAlphaControlResult(
            name="affine_map_reproduces_phase46",
            passed=affine_comparison.passed,
            comparison=affine_comparison,
            observed_value=affine_comparison.maximum_relative_score_error,
            threshold=SCORE_RELATIVE_TOLERANCE,
            detail="constant Jacobian boundary versus global affine-SPD alpha",
        ),
        SpatialAlphaControlResult(
            name="nonlinear_map_matches_explicit_transform",
            passed=explicit_comparison.passed,
            comparison=explicit_comparison,
            observed_value=explicit_comparison.maximum_relative_score_error,
            threshold=SCORE_RELATIVE_TOLERANCE,
            detail="single transformed-coordinate Delaunay/alpha construction",
        ),
        SpatialAlphaControlResult(
            name="inverse_and_positive_jacobian",
            passed=(
                shear.maximum_inverse_roundtrip_error
                <= INVERSE_ROUNDTRIP_TOLERANCE
                and shear.minimum_jacobian_determinant
                >= MINIMUM_JACOBIAN_DETERMINANT
            ),
            comparison=None,
            observed_value=shear.maximum_inverse_roundtrip_error,
            threshold=INVERSE_ROUNDTRIP_TOLERANCE,
            detail=(
                "roundtrip error; minimum determinant="
                f"{shear.minimum_jacobian_determinant:.12g}"
            ),
        ),
        SpatialAlphaControlResult(
            name="analytic_jacobian_matches_finite_difference",
            passed=maximum_jacobian_error <= JACOBIAN_ABSOLUTE_TOLERANCE,
            comparison=None,
            observed_value=maximum_jacobian_error,
            threshold=JACOBIAN_ABSOLUTE_TOLERANCE,
            detail="maximum absolute Jacobian entry error",
        ),
        SpatialAlphaControlResult(
            name="spatial_spd_variation_changes_connectivity",
            passed=(
                shear.minimum_metric_eigenvalue > 0.0
                and shear.maximum_relative_metric_variation
                >= MINIMUM_METRIC_VARIATION
                and connectivity_difference > 0
            ),
            comparison=None,
            observed_value=shear.maximum_relative_metric_variation,
            threshold=MINIMUM_METRIC_VARIATION,
            detail=(
                f"minimum eigenvalue={shear.minimum_metric_eigenvalue:.12g}; "
                f"connectivity symmetric difference={connectivity_difference}"
            ),
        ),
        SpatialAlphaControlResult(
            name="integrability_accepts_shear_rejects_incompatible_field",
            passed=(
                shear_integrability.compatible
                and not nonintegrable.compatible
                and observed_rejection
            ),
            comparison=None,
            observed_value=nonintegrable.maximum_mixed_partial_residual,
            threshold=INTEGRABILITY_ABSOLUTE_TOLERANCE,
            detail=(
                "incompatible residual; shear residual="
                f"{shear_integrability.maximum_mixed_partial_residual:.12g}"
            ),
        ),
        SpatialAlphaControlResult(
            name="rigid_output_invariance",
            passed=rigid_comparison.passed,
            comparison=rigid_comparison,
            observed_value=rigid_comparison.maximum_relative_score_error,
            threshold=SCORE_RELATIVE_TOLERANCE,
            detail="post-composed exactly representable half-turn rotation",
        ),
    )
    passed_count = sum(control.passed for control in controls)
    supported = passed_count == len(controls)
    return IntegrableSpatialAlphaAuditResult(
        artifact_schema=RESULT_SCHEMA,
        protocol_schema=protocol.artifact_schema,
        protocol_sha256=_protocol_digest(),
        controls=controls,
        passed_control_count=passed_count,
        total_control_count=len(controls),
        connectivity_symmetric_difference_count=connectivity_difference,
        generic_rotation_connectivity_equal=(
            generic_rigid_comparison.connectivity_equal
        ),
        generic_rotation_maximum_relative_score_error=(
            generic_rigid_comparison.maximum_relative_score_error
        ),
        analytic_integrable_spatial_spd_complex_supported=supported,
        arbitrary_point_local_spd_complex_supported=False,
        point_local_alpha_field_supported=False,
        pftf_conditioned_spatial_alpha_supported=False,
        generic_floating_rigid_score_invariance_supported=False,
        exact_integrable_spatial_predicates_supported=False,
        spatial_alpha_reconstruction_advantage_supported=False,
        spatial_alpha_topology_correctness_supported=False,
        spatial_alpha_real_scan_transfer_supported=False,
        spatial_alpha_deployment_supported=False,
    )


def write_audit(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        evaluate_integrable_spatial_alpha_audit().to_dict(),
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
        default=Path("benchmark-out/integrable_spatial_alpha_audit_phase47.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_audit(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
