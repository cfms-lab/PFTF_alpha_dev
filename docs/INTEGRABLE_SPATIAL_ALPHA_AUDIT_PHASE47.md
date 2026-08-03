# Phase 47: integrable nonlinear spatial-alpha audit

## Outcome

All eight frozen construction controls pass. The repository now contains a
coherent spatially varying SPD alpha complex for one bounded analytic regime:
an explicit globally injective quadratic-shear coordinate map. Delaunay
connectivity and every alpha-filtration value are computed together in the
single transformed coordinate system.

This is positive construction evidence. It is not reconstruction-performance,
arbitrary local-metric, learned PFTF, exact-predicate, or real-scan evidence.

## Frozen protocol and implementation

- Protocol commit: `a0022b6`.
- Protocol SHA-256:
  `c8ead6b3bbf4850708fb84d6d4b83ca647a491c11728eca2de439ec4e835fa38`.
- Implementation commit: `434c873`.
- Audit seed: `47001`; 56 generated 3D points.
- Map: `Phi(x,y,z)=(x, y+0.20*x^2, z)`.
- Inverse: `Phi^-1(u,v,w)=(u, v-0.20*u^2, w)`.
- Row-convention metric: `M(x)=J_Phi(x) J_Phi(x)^T`.
- Score tolerance: `rtol=5e-10`, `atol=1e-12`.

## Frozen control results

| Control | Observed result | Decision |
|---|---:|---|
| Zero shear versus Euclidean alpha | identical connectivity and scores | pass |
| Affine map versus Phase 46 | relative score error `2.4291e-11` | pass |
| Nonlinear construction versus explicit `AlphaFiltration(Phi(points))` | identical | pass |
| Inverse and positive Jacobian | roundtrip `2.2204e-16`, minimum determinant `1` | pass |
| Analytic versus finite-difference Jacobian | maximum error `2.1434e-10` | pass |
| Spatial SPD variation and connectivity | variation `1.303642`, minimum eigenvalue `0.294965`, top-cell symmetric difference `133` | pass |
| Integrability acceptance/rejection | shear residual `0`; incompatible residual `0.35000000004` and rejected | pass |
| Exactly representable half-turn output rotation | identical connectivity and scores | pass |

The artifact is byte-identical across two complete executions. Result SHA-256:
`7228d2f00314f88a7fb13cb6786bda7b14eacd0c271b0e53b3f0b3851fc75ce8`.

## Floating-point rigid-rotation boundary

An additional diagnostic post-composed the transformed coordinates with a
generic `0.43 rad` floating rotation. Connectivity remained identical, but the
maximum relative filtration-value difference was `3.5326e-9`, above the frozen
`5e-10` tolerance. The discrepancy comes from floating intrinsic-circumsphere
solves on ill-conditioned, very-large-radius simplices.

It is not used to relax the gate. The frozen exactly representable half-turn
control passes, while
`generic_floating_rigid_score_invariance_supported=false`. Exact predicates or
a better-conditioned geometry kernel would be needed for a stronger numerical
invariance claim.

## Interpretation and claim boundary

`analytic_integrable_spatial_spd_complex_supported=true` means that a declared
nonlinear diffeomorphism can induce a coherent spatial metric and one associated
alpha complex. This is genuinely different from B5/P1, which average local SPD
matrices only to score fixed Euclidean Delaunay cells.

The mixed-partial audit is only a necessary local condition. An arbitrary field
that passes it is not automatically globally injective, and SPD matrices alone
do not identify a unique coordinate map. Phase 47 is positive because the
quadratic shear also has an explicit global inverse.

The following claims remain false:

- `arbitrary_point_local_spd_complex_supported`;
- `point_local_alpha_field_supported`;
- `pftf_conditioned_spatial_alpha_supported`;
- `generic_floating_rigid_score_invariance_supported`;
- `exact_integrable_spatial_predicates_supported`;
- `spatial_alpha_reconstruction_advantage_supported`;
- `spatial_alpha_topology_correctness_supported`;
- `spatial_alpha_real_scan_transfer_supported`;
- `spatial_alpha_deployment_supported`.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.integrable_spatial_alpha_protocol --output benchmark-out\integrable_spatial_alpha_protocol_phase47.json
.\.venv\Scripts\python.exe -m pftf_alpha.integrable_spatial_alpha_audit --output benchmark-out\integrable_spatial_alpha_audit_phase47.json
```

The next distinct phase should not tune the quadratic shear. It should define a
separate training-only procedure that learns a globally invertible coordinate
map from observed confidence/PFTF features, freeze that map family and all
regularization on training/calibration data, and validate once on a new panel.
