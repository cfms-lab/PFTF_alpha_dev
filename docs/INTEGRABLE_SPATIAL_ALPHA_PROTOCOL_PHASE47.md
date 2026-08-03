# Phase 47: integrable nonlinear spatial-alpha protocol

## Motivation

Phase 46 proved that one constant SPD metric gives a coherent alpha complex
through a shared affine coordinate transform. It also correctly rejected a
spatially rotating point field from that affine-only path. Phase 47 tests the
next sufficient construction: one globally injective nonlinear coordinate map
whose Jacobian induces a spatially varying SPD metric.

This is a construction audit, not a reconstruction benchmark. No Phase 43--45
held-out endpoint is reopened.

## Frozen analytic map

For strength `s=0.20`, use

`Phi(x,y,z) = (x, y + s*x^2, z)`.

Its explicit inverse is

`Phi^-1(u,v,w) = (u, v - s*u^2, w)`.

Using row-vector displacements, `dPhi = dx J_Phi`, and the induced local metric
is `M(x)=J_Phi(x) J_Phi(x)^T`. The determinant is identically one, and the
explicit inverse establishes global injectivity for this declared map.

The alpha complex is built once as the ordinary Euclidean Delaunay/alpha
filtration of `Phi(points)`. Its indexed simplices and filtration values are
then attached to the original coordinates. Local matrices are diagnostics of
the map; they are not independently averaged or used to rescore an unrelated
Euclidean triangulation.

## Frozen necessary integrability audit

For each output coordinate `b` and input axes `a,c`, a Jacobian field of one
smooth map must satisfy the mixed-partial condition

`d J[a,b]/d x[c] = d J[c,b]/d x[a]`.

Central differences use step `1e-6`; the accepted residual is at most `1e-8`.
The frozen rejection control sets `J[0,1]=0.35*y` with an identity diagonal.
Its mixed-partial residual is `0.35`, so it must fail closed. Passing this local
condition alone is not treated as proof of global injectivity; the accepted
quadratic shear separately supplies an explicit inverse.

## Frozen audit

- Seed: `47001`; 56 generated 3D points.
- Score tolerance: `rtol=5e-10`, `atol=1e-12`.
- Inverse roundtrip maximum: `1e-12`.
- Minimum Jacobian determinant: `0.50`.
- Minimum relative metric variation: `0.05`.
- Controls:
  1. zero shear equals Euclidean alpha;
  2. an affine map equals the Phase-46 construction;
  3. nonzero shear equals an explicit transformed-coordinate filtration;
  4. inverse roundtrip and determinant gates;
  5. analytic versus finite-difference Jacobian;
  6. spatial SPD variation and a connectivity change;
  7. integrable acceptance and nonintegrable rejection;
  8. rigid output-coordinate invariance.

All eight controls must pass.

## Claim boundary

A positive audit supports only the declared analytic nonlinear coordinate-map
alpha construction and its induced spatial SPD metrics. It does not establish
an arbitrary point-local metric complex, a learned or PFTF-conditioned map,
exact predicates, reconstruction advantage, topology correctness, real-scan
transfer, or deployment.
