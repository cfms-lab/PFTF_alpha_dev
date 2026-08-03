# Phase 46: global affine-SPD alpha compatibility audit

## Outcome

The frozen construction audit passes all five controls. One constant SPD
metric now has a coherent alpha-complex implementation: transform every point
through one shared affine map, build Delaunay connectivity and alpha filtration
values in that transformed coordinate system, and retain the indexed complex
on the original coordinates.

This is a positive construction-invariant result, not a reconstruction
performance result. The spatially varying local-SPD claim remains unsupported.

## Frozen protocol and implementation

- Protocol commits: `d48fc0b`, followed by serialization-only commit
  `afc7734` to make the UTF-8/LF artifact hash operating-system independent.
- Protocol SHA-256:
  `6b49428a641e1db1db88d1a8cd5da1e3672490a9409dfd5c5f9b64036a632b06`.
- Implementation commit: `a545945`.
- Seed: `46001`; 48 points in 3D.
- Compatibility condition: every point metric must equal one shared
  `M = L L^T` within `rtol=1e-10`, `atol=1e-12`.
- Score comparison: `rtol=5e-10`, `atol=1e-12`.

The implementation uses row-vector coordinates `y = x L`. It computes both
Delaunay connectivity and all simplex alpha values through
`AlphaFiltration.from_points(y)`, then reuses the same index records with the
original point coordinates. It does not rescore cells from an unrelated
Euclidean triangulation.

## Control results

| Control | Connectivity | Maximum score error | Result |
|---|---:|---:|---|
| Identity metric versus Euclidean alpha | equal | absolute 0, relative 0 | pass |
| Constant anisotropic metric versus explicit `y=xL` | equal | absolute 0, relative 0 | pass |
| Affine coordinate covariance | equal | absolute `2.4835e-6`, relative `1.4194e-10` | pass |
| Constant `LocalMetricField` versus global construction | equal | absolute 0, relative 0 | pass |
| Spatially rotating `LocalMetricField` | construction rejected | metric relative deviation `0.957302` | pass, fail-closed |

The affine-covariance absolute error occurs on large squared-radius values; the
frozen relative gate is the scale-aware test and remains satisfied. The audit
artifact is byte-identical across two complete executions. Result SHA-256:
`2c3ad38e32754d04e31c15b5388988a87f7b4abd4d49f4ec540bcaba29822198`.

## Interpretation

`global_affine_spd_complex_supported=true` means only that the constant-metric
control is mathematically coherent and implemented consistently in floating
point. It does not mean that one constant anisotropy improves reconstruction.

The rejection of the rotating field is narrower than an impossibility result.
It says that arbitrary per-point matrices cannot be passed to this **single
affine transform** construction. A future spatial construction could instead
use one globally injective nonlinear coordinate map `Phi`. Under the repository
row-vector convention its local metric must be induced by the map's Jacobian,
`M(x) = J_Phi(x) J_Phi(x)^T`; independently chosen local factors need not obey
the required cross-partial integrability or global injectivity conditions.

Therefore the following flags remain false:

- `spatially_varying_spd_complex_supported`;
- `point_local_alpha_field_supported`;
- `exact_affine_spd_predicates_supported`;
- `affine_spd_reconstruction_advantage_supported`;
- `affine_spd_topology_correctness_supported`;
- `affine_spd_real_scan_transfer_supported`;
- `affine_spd_deployment_supported`.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.affine_spd_alpha_protocol --output benchmark-out\affine_spd_alpha_protocol_phase46.json
.\.venv\Scripts\python.exe -m pftf_alpha.affine_spd_alpha_audit --output benchmark-out\affine_spd_alpha_audit_phase46.json
```

The next distinct phase should preregister an **integrable nonlinear metric
control**: define one analytic injective `Phi`, build the Euclidean alpha
complex on `Phi(points)`, verify coordinate and filtration invariants, and only
then consider learning or observing a PFTF-conditioned map on separate data.
