"""Phase-47 preregistration for an integrable nonlinear spatial alpha control."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL_SCHEMA = "pftf_alpha_integrable_spatial_alpha_protocol_phase47/v1"
AUDIT_SEED = 47_001
POINT_COUNT = 56
AMBIENT_DIMENSION = 3
QUADRATIC_SHEAR_STRENGTH = 0.20
NONINTEGRABLE_FIELD_STRENGTH = 0.35
FINITE_DIFFERENCE_STEP = 1.0e-6
JACOBIAN_ABSOLUTE_TOLERANCE = 1.0e-8
INTEGRABILITY_ABSOLUTE_TOLERANCE = 1.0e-8
INVERSE_ROUNDTRIP_TOLERANCE = 1.0e-12
MINIMUM_JACOBIAN_DETERMINANT = 0.50
MINIMUM_METRIC_VARIATION = 0.05
SCORE_RELATIVE_TOLERANCE = 5.0e-10
SCORE_ABSOLUTE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class IntegrableSpatialAlphaProtocol:
    artifact_schema: str
    role: str
    audit_seed: int
    point_count: int
    ambient_dimension: int
    coordinate_map: str
    inverse_map: str
    row_jacobian: str
    induced_metric: str
    local_integrability_condition: str
    nonintegrable_control: str
    quadratic_shear_strength: float
    nonintegrable_field_strength: float
    finite_difference_step: float
    jacobian_absolute_tolerance: float
    integrability_absolute_tolerance: float
    inverse_roundtrip_tolerance: float
    minimum_jacobian_determinant: float
    minimum_metric_variation: float
    score_relative_tolerance: float
    score_absolute_tolerance: float
    frozen_controls: tuple[str, ...]
    validation_gate: str
    reference_boundary: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["frozen_controls"] = list(self.frozen_controls)
        return payload


def preregister_integrable_spatial_alpha() -> IntegrableSpatialAlphaProtocol:
    return IntegrableSpatialAlphaProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="pre_implementation_integrable_nonlinear_spatial_alpha_audit",
        audit_seed=AUDIT_SEED,
        point_count=POINT_COUNT,
        ambient_dimension=AMBIENT_DIMENSION,
        coordinate_map="Phi(x,y,z) = (x, y + s*x^2, z)",
        inverse_map="Phi^-1(u,v,w) = (u, v - s*u^2, w)",
        row_jacobian=(
            "J_Phi[axis_in,axis_out] = d Phi_axis_out / d x_axis_in; "
            "row displacements obey dPhi = dx J_Phi"
        ),
        induced_metric="M(x) = J_Phi(x) J_Phi(x)^T",
        local_integrability_condition=(
            "for every output b and input axes a,c, d J[a,b]/d x[c] = "
            "d J[c,b]/d x[a]; this necessary mixed-partial condition is "
            "audited numerically but is not by itself a global injectivity proof"
        ),
        nonintegrable_control=(
            "J[0,1] = q*y, J[0,0] = J[1,1] = J[2,2] = 1, all other "
            "entries zero; mixed-partial residual is q"
        ),
        quadratic_shear_strength=QUADRATIC_SHEAR_STRENGTH,
        nonintegrable_field_strength=NONINTEGRABLE_FIELD_STRENGTH,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        jacobian_absolute_tolerance=JACOBIAN_ABSOLUTE_TOLERANCE,
        integrability_absolute_tolerance=INTEGRABILITY_ABSOLUTE_TOLERANCE,
        inverse_roundtrip_tolerance=INVERSE_ROUNDTRIP_TOLERANCE,
        minimum_jacobian_determinant=MINIMUM_JACOBIAN_DETERMINANT,
        minimum_metric_variation=MINIMUM_METRIC_VARIATION,
        score_relative_tolerance=SCORE_RELATIVE_TOLERANCE,
        score_absolute_tolerance=SCORE_ABSOLUTE_TOLERANCE,
        frozen_controls=(
            "zero-strength quadratic shear reproduces Euclidean alpha",
            "a constant affine coordinate map reproduces the Phase-46 "
            "global-SPD construction",
            "nonzero quadratic shear reproduces explicit "
            "AlphaFiltration.from_points(Phi(points))",
            "quadratic shear passes analytic inverse roundtrip and positive-"
            "Jacobian gates",
            "declared analytic Jacobians match central finite differences",
            "induced metrics are SPD, spatially varying, and change Delaunay "
            "connectivity",
            "mixed-partial audit accepts the shear Jacobian and rejects the "
            "frozen nonintegrable field",
            "post-composition by one rigid output transform preserves "
            "connectivity and filtration values",
        ),
        validation_gate=(
            "all eight controls must pass; every filtration comparison requires "
            "identical canonical top-simplex and simplex-key sets plus scores "
            "within frozen tolerances; the nonlinear map must meet roundtrip, "
            "Jacobian, determinant, metric-variation, and connectivity-change "
            "gates; the incompatible Jacobian field must fail closed"
        ),
        reference_boundary=(
            "the audit consumes only generated coordinates and declared maps or "
            "Jacobians; no reference surface, reconstruction endpoint, family "
            "label, or held-out Phase-43--45 evidence is used"
        ),
        claim_boundary=(
            "a positive result supports only a floating-point analytic nonlinear "
            "coordinate-map alpha complex and its Jacobian-induced spatial SPD "
            "metrics in the declared quadratic-shear control; it does not support "
            "arbitrary point-local metrics, a learned or PFTF-conditioned map, "
            "exact predicates, reconstruction advantage, topology correctness, "
            "real-scan transfer, or deployment"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_integrable_spatial_alpha().to_dict(),
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
        default=Path("benchmark-out/integrable_spatial_alpha_protocol_phase47.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_protocol(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
