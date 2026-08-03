"""Phase-46 preregistration for a globally compatible affine-SPD control."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL_SCHEMA = "pftf_alpha_affine_spd_alpha_protocol_phase46/v1"
AUDIT_SEED = 46_001
POINT_COUNT = 48
AMBIENT_DIMENSION = 3
CONNECTIVITY_TOLERANCE = 0.0
SCORE_RELATIVE_TOLERANCE = 5.0e-10
SCORE_ABSOLUTE_TOLERANCE = 1.0e-12
METRIC_COMPATIBILITY_RELATIVE_TOLERANCE = 1.0e-10
METRIC_COMPATIBILITY_ABSOLUTE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class AffineSPDAlphaProtocol:
    artifact_schema: str
    role: str
    audit_seed: int
    point_count: int
    ambient_dimension: int
    compatibility_condition: str
    construction: str
    coordinate_covariance_rule: str
    frozen_controls: tuple[str, ...]
    frozen_incompatible_field: str
    connectivity_tolerance: float
    score_relative_tolerance: float
    score_absolute_tolerance: float
    metric_compatibility_relative_tolerance: float
    metric_compatibility_absolute_tolerance: float
    validation_gate: str
    reference_boundary: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["frozen_controls"] = list(self.frozen_controls)
        return payload


def preregister_affine_spd_alpha() -> AffineSPDAlphaProtocol:
    return AffineSPDAlphaProtocol(
        artifact_schema=PROTOCOL_SCHEMA,
        role="pre_implementation_affine_spd_complex_compatibility_audit",
        audit_seed=AUDIT_SEED,
        point_count=POINT_COUNT,
        ambient_dimension=AMBIENT_DIMENSION,
        compatibility_condition=(
            "there exists one invertible matrix L shared by every point such "
            "that M_i = L L^T within the frozen tolerances; equivalently the "
            "point metric field is constant and globally affine-representable"
        ),
        construction=(
            "transform row-vector coordinates by y = x L, build the ordinary "
            "Euclidean Delaunay alpha filtration in y, and retain its indexed "
            "simplices and filtration values on the original coordinates"
        ),
        coordinate_covariance_rule=(
            "under x_prime = x A, use M_prime = A^-1 M A^-T so squared metric "
            "distances, connectivity, and alpha filtration values are invariant"
        ),
        frozen_controls=(
            "identity metric reproduces the Euclidean alpha filtration",
            "one constant rotated anisotropic SPD metric reproduces the "
            "explicit transformed-coordinate filtration",
            "one invertible affine reparameterization preserves connectivity "
            "and filtration values under the covariance rule",
            "a constant LocalMetricField is accepted and reproduces the global "
            "construction",
            "a spatially rotating LocalMetricField is rejected as not "
            "representable by one global affine transform",
        ),
        frozen_incompatible_field=(
            "M_i = R_z(theta_i) diag(4, 1, 0.25) R_z(theta_i)^T with theta_i "
            "spanning [-0.6, 0.6] in point-index order"
        ),
        connectivity_tolerance=CONNECTIVITY_TOLERANCE,
        score_relative_tolerance=SCORE_RELATIVE_TOLERANCE,
        score_absolute_tolerance=SCORE_ABSOLUTE_TOLERANCE,
        metric_compatibility_relative_tolerance=(
            METRIC_COMPATIBILITY_RELATIVE_TOLERANCE
        ),
        metric_compatibility_absolute_tolerance=METRIC_COMPATIBILITY_ABSOLUTE_TOLERANCE,
        validation_gate=(
            "all five controls must pass; identity, explicit-transform, affine-"
            "covariance, and constant-field cases require identical canonical "
            "top-simplex sets and all simplex filtration values within frozen "
            "tolerances, while the rotating field must fail closed before a "
            "filtration is constructed"
        ),
        reference_boundary=(
            "the audit consumes only generated point coordinates and declared "
            "SPD matrices; it uses no reference surface or reconstruction endpoint"
        ),
        claim_boundary=(
            "a positive audit supports only the mathematical and floating-point "
            "implementation validity of a constant global affine-SPD alpha "
            "control; it is not an exact-predicate result, a spatially varying "
            "local-SPD complex, a performance advantage, topology correctness, "
            "real-scan transfer, or deployment evidence"
        ),
    )


def write_protocol(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        preregister_affine_spd_alpha().to_dict(),
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
        default=Path("benchmark-out/affine_spd_alpha_protocol_phase46.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    digest = write_protocol(args.output)
    print(f"wrote {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
