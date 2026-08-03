"""Run the frozen Phase-46 affine-SPD construction compatibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .affine_spd_alpha import (
    IncompatibleLocalMetricError,
    audit_global_metric_compatibility,
    global_affine_spd_alpha,
    global_affine_spd_alpha_from_field,
)
from .affine_spd_alpha_protocol import (
    METRIC_COMPATIBILITY_ABSOLUTE_TOLERANCE,
    METRIC_COMPATIBILITY_RELATIVE_TOLERANCE,
    SCORE_ABSOLUTE_TOLERANCE,
    SCORE_RELATIVE_TOLERANCE,
    preregister_affine_spd_alpha,
)
from .filtration import AlphaFiltration
from .metrics import LocalMetricField

RESULT_SCHEMA = "pftf_alpha_affine_spd_alpha_audit_phase46/v1"


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
class AffineSPDControlResult:
    name: str
    passed: bool
    expected_rejection: bool
    observed_rejection: bool
    comparison: FiltrationComparison | None
    metric_maximum_relative_deviation: float
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AffineSPDAlphaAuditResult:
    artifact_schema: str
    protocol_schema: str
    protocol_sha256: str
    controls: tuple[AffineSPDControlResult, ...]
    passed_control_count: int
    total_control_count: int
    global_affine_spd_complex_supported: bool
    spatially_varying_spd_complex_supported: bool
    point_local_alpha_field_supported: bool
    exact_affine_spd_predicates_supported: bool
    affine_spd_reconstruction_advantage_supported: bool
    affine_spd_topology_correctness_supported: bool
    affine_spd_real_scan_transfer_supported: bool
    affine_spd_deployment_supported: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["controls"] = [control.to_dict() for control in self.controls]
        return payload


def _canonical_cells(filtration: AlphaFiltration) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            tuple(sorted(int(vertex) for vertex in cell))
            for cell in filtration.top_simplices
        )
    )


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
            [abs(left_values[key] - right_values[key]) for key in keys],
            dtype=np.float64,
        )
        denominators = np.asarray(
            [max(abs(right_values[key]), np.finfo(float).tiny) for key in keys],
            dtype=np.float64,
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


def _rotation_z(angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _protocol_digest() -> str:
    text = json.dumps(
        preregister_affine_spd_alpha().to_dict(),
        indent=2,
        sort_keys=True,
    ) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate_affine_spd_alpha_audit() -> AffineSPDAlphaAuditResult:
    protocol = preregister_affine_spd_alpha()
    rng = np.random.default_rng(protocol.audit_seed)
    points = rng.normal(size=(protocol.point_count, protocol.ambient_dimension))

    euclidean = AlphaFiltration.from_points(points)
    identity = global_affine_spd_alpha(points, np.eye(3)).filtration
    identity_comparison = _compare_filtrations(identity, euclidean)

    rotation = _rotation_z(0.37)
    anisotropic_metric = rotation @ np.diag([4.0, 1.0, 0.25]) @ rotation.T
    anisotropic = global_affine_spd_alpha(points, anisotropic_metric)
    explicit = AlphaFiltration.from_points(
        points @ np.linalg.cholesky(anisotropic_metric)
    )
    explicit_comparison = _compare_filtrations(anisotropic.filtration, explicit)

    coordinate_map = np.asarray(
        [[1.20, 0.15, -0.05], [0.05, 0.80, 0.10], [0.02, -0.08, 1.10]],
        dtype=np.float64,
    )
    inverse_map = np.linalg.inv(coordinate_map)
    covariant_metric = inverse_map @ anisotropic_metric @ inverse_map.T
    reparameterized = global_affine_spd_alpha(
        points @ coordinate_map,
        covariant_metric,
    )
    covariance_comparison = _compare_filtrations(
        anisotropic.filtration,
        reparameterized.filtration,
    )

    constant_field = LocalMetricField(
        matrices=np.repeat(anisotropic_metric[None, :, :], len(points), axis=0),
        confidence=np.ones(len(points), dtype=np.float64),
    )
    constant_compatibility = audit_global_metric_compatibility(
        constant_field,
        relative_tolerance=METRIC_COMPATIBILITY_RELATIVE_TOLERANCE,
        absolute_tolerance=METRIC_COMPATIBILITY_ABSOLUTE_TOLERANCE,
    )
    from_constant_field = global_affine_spd_alpha_from_field(
        points,
        constant_field,
        relative_tolerance=METRIC_COMPATIBILITY_RELATIVE_TOLERANCE,
        absolute_tolerance=METRIC_COMPATIBILITY_ABSOLUTE_TOLERANCE,
    )
    constant_comparison = _compare_filtrations(
        from_constant_field.filtration,
        anisotropic.filtration,
    )

    angles = np.linspace(-0.6, 0.6, len(points))
    varying_matrices = np.stack(
        [
            _rotation_z(float(angle))
            @ np.diag([4.0, 1.0, 0.25])
            @ _rotation_z(float(angle)).T
            for angle in angles
        ]
    )
    varying_field = LocalMetricField(
        matrices=varying_matrices,
        confidence=np.ones(len(points), dtype=np.float64),
    )
    varying_compatibility = audit_global_metric_compatibility(
        varying_field,
        relative_tolerance=METRIC_COMPATIBILITY_RELATIVE_TOLERANCE,
        absolute_tolerance=METRIC_COMPATIBILITY_ABSOLUTE_TOLERANCE,
    )
    observed_rejection = False
    try:
        global_affine_spd_alpha_from_field(
            points,
            varying_field,
            relative_tolerance=METRIC_COMPATIBILITY_RELATIVE_TOLERANCE,
            absolute_tolerance=METRIC_COMPATIBILITY_ABSOLUTE_TOLERANCE,
        )
    except IncompatibleLocalMetricError:
        observed_rejection = True

    controls = (
        AffineSPDControlResult(
            name="identity_reproduces_euclidean",
            passed=identity_comparison.passed,
            expected_rejection=False,
            observed_rejection=False,
            comparison=identity_comparison,
            metric_maximum_relative_deviation=0.0,
            detail="identity global metric versus Euclidean AlphaFiltration",
        ),
        AffineSPDControlResult(
            name="constant_anisotropic_matches_explicit_transform",
            passed=explicit_comparison.passed,
            expected_rejection=False,
            observed_rejection=False,
            comparison=explicit_comparison,
            metric_maximum_relative_deviation=0.0,
            detail="global construction versus explicit y=xL filtration",
        ),
        AffineSPDControlResult(
            name="affine_coordinate_covariance",
            passed=covariance_comparison.passed,
            expected_rejection=False,
            observed_rejection=False,
            comparison=covariance_comparison,
            metric_maximum_relative_deviation=0.0,
            detail="x'=xA and M'=A^-1 M A^-T",
        ),
        AffineSPDControlResult(
            name="constant_local_field_reduces_to_global_metric",
            passed=constant_compatibility.compatible and constant_comparison.passed,
            expected_rejection=False,
            observed_rejection=False,
            comparison=constant_comparison,
            metric_maximum_relative_deviation=(
                constant_compatibility.maximum_relative_deviation
            ),
            detail="constant LocalMetricField accepted by compatibility guard",
        ),
        AffineSPDControlResult(
            name="rotating_local_field_fails_closed",
            passed=(not varying_compatibility.compatible) and observed_rejection,
            expected_rejection=True,
            observed_rejection=observed_rejection,
            comparison=None,
            metric_maximum_relative_deviation=(
                varying_compatibility.maximum_relative_deviation
            ),
            detail="varying point metrics require more than one global transform",
        ),
    )
    passed_count = sum(control.passed for control in controls)
    all_controls_pass = passed_count == len(controls)
    return AffineSPDAlphaAuditResult(
        artifact_schema=RESULT_SCHEMA,
        protocol_schema=protocol.artifact_schema,
        protocol_sha256=_protocol_digest(),
        controls=controls,
        passed_control_count=passed_count,
        total_control_count=len(controls),
        global_affine_spd_complex_supported=all_controls_pass,
        spatially_varying_spd_complex_supported=False,
        point_local_alpha_field_supported=False,
        exact_affine_spd_predicates_supported=False,
        affine_spd_reconstruction_advantage_supported=False,
        affine_spd_topology_correctness_supported=False,
        affine_spd_real_scan_transfer_supported=False,
        affine_spd_deployment_supported=False,
    )


def write_audit(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        evaluate_affine_spd_alpha_audit().to_dict(),
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
        default=Path("benchmark-out/affine_spd_alpha_audit_phase46.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_audit(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
