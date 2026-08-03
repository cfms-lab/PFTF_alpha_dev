# Phase 46: global affine-SPD alpha compatibility protocol

## Motivation

Phase 45 changed regular-triangulation connectivity but did not improve the
frozen reconstruction objective. Its scalar confidence-to-power relation must
not be retuned on the opened held-out panel. The scientifically distinct next
question is more basic: when does an SPD metric field define one coherent alpha
complex rather than unrelated per-cell scores on a Euclidean triangulation?

## Frozen compatibility condition

This phase accepts a point metric field only when there is one invertible
matrix `L`, shared by every point, such that

`M_i = L L^T`

within the frozen absolute and relative tolerances. Thus the accepted field is
constant and globally affine-representable. For row-vector coordinates, set
`y = x L`, build the ordinary Euclidean Delaunay alpha filtration in `y`, and
attach the resulting indexed simplices and filtration values to the original
coordinates.

Under an invertible coordinate change `x_prime = x A`, the compatible metric is

`M_prime = A^-1 M A^-T`.

This preserves every squared metric distance and therefore must preserve the
canonical top-simplex set and all simplex filtration values.

## Frozen audit

- Seed: `46001`.
- Dimension: 3.
- Point count: 48.
- Connectivity comparison: exact equality of canonical index sets.
- Score comparison: relative tolerance `5e-10`, absolute tolerance `1e-12`.
- Metric compatibility: relative tolerance `1e-10`, absolute tolerance `1e-12`.
- Controls:
  1. identity metric versus the Euclidean alpha filtration;
  2. a constant rotated anisotropic SPD metric versus an explicit coordinate
     transform;
  3. an invertible affine reparameterization with the covariant metric;
  4. a constant `LocalMetricField` accepted through the compatibility guard;
  5. a rotating local field rejected before filtration construction.

The incompatible field is fixed as
`M_i = R_z(theta_i) diag(4, 1, 0.25) R_z(theta_i)^T`, with `theta_i` spanning
`[-0.6, 0.6]` in point-index order.

All five controls must pass. No reconstruction/reference endpoint is inspected.

## Claim boundary

A positive result supports a mathematically coherent, floating-point
**constant global affine-SPD control**. It does not support exact predicates, a
spatially varying local-SPD/anisotropic complex, reconstruction improvement,
topology correctness, real-scan transfer, or deployment. In particular, the
existing B5/P1 per-cell metric averages remain fixed-Euclidean-connectivity
score constructions and are not reclassified by this phase.
