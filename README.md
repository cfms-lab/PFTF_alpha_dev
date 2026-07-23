# PFTF_alpha_dev

PFTF-guided local anisotropic alpha fields for robust point-cloud shape
reconstruction.

## Status

- Quality: **ToDo**
- Stage: design and prior-art boundary
- No implementation benchmark or paper draft exists yet.
- No PDF or DOCX is generated at this stage.

## Core decision

A single global alpha is a scalar scale-selection problem. Because an
alpha-complex changes only at a finite set of Delaunay critical values, the
strong practical baseline is to enumerate those values and select one by
geometry, topology, and resampling-stability criteria.

The PFTF contribution is therefore **not** framed as “use a tensor to find one
global alpha.” The project instead studies whether a PFTF field can construct a
local symmetric positive-definite metric, or a position- and direction-aware
alpha field, that handles:

- non-uniform sampling density;
- anisotropic surface spacing;
- thin gaps and nearby opposing sheets;
- curvature and normal uncertainty;
- noise and missing observations.

## Proposed method

At point or region \(i\), construct a valid metric

\[
M_i=L_iL_i^T+\varepsilon I
\]

from local directional relations. Candidate edges or simplices are then tested
in the induced metric rather than with one Euclidean ball. During calibration,
the hard alpha gate can be relaxed as

\[
g_\tau(\alpha,T)
=
\operatorname{sigmoid}
\left(\frac{\alpha-r_{\tau,M}^{\,2}}{T}\right),
\]

where \(r_{\tau,M}^{\,2}\) is the metric circumsphere quantity. Away from exact
ties, \(T\rightarrow0^+\) recovers the hard inclusion decision.

Only the symmetric positive-definite part may define a metric. Signed or
asymmetric PFTF information must be mapped to scale, confidence, or a separate
penalty; it must not be presented directly as a geometric distance.

## Novelty boundary

Density-scaled and normal-driven anisotropic alpha shapes already exist.
Accordingly, local density, point normals, or an ellipsoidal alpha-ball alone
are baselines rather than a new contribution. A publishable PFTF result must
demonstrate additional value through at least one of:

- routed source-to-receiver relations beyond local PCA or normal averaging;
- a consistent soft-to-hard gate analysis;
- joint geometry, topology, and stability calibration;
- uncertainty-aware fail-closed routing to a trusted baseline;
- held-out gains over both density-adaptive and prior anisotropic alpha methods.

## First gate

Implement and compare:

1. convex hull;
2. fixed global alpha;
3. critical-alpha scan with topology and stability selection;
4. k-nearest-neighbor density-scaled local alpha;
5. normal-based anisotropic alpha;
6. PFTF local metric with confidence and exact fallback.

See [docs/PFTF_ALPHA_RESEARCH_PLAN.md](docs/PFTF_ALPHA_RESEARCH_PLAN.md) and
[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).

## Safety boundary

- There is no universal ground-truth alpha without a declared downstream
  objective.
- A one-component criterion is invalid for deliberately disconnected objects.
- Standard weighted alpha shapes do not automatically provide arbitrary
  spatially varying SPD metrics.
- Soft membership is a calibration surrogate, not a replacement for exact
  alpha-complex construction at evaluation time.
- Success against a fixed global alpha alone is insufficient; adaptive and
  anisotropic baselines are mandatory.

## References

- H. Edelsbrunner and E. P. Mücke, “Three-dimensional alpha shapes,”
  *ACM Transactions on Graphics*, 1994.
  <https://pub.ista.ac.at/~edels/Papers/1994-04-3DAlphaShapes.pdf>
- CGAL, “3D Alpha Shapes.”
  <https://doc.cgal.org/latest/Alpha_shapes_3/index.html>
- M. Teichmann and M. Capps, “Surface reconstruction with anisotropic
  density-scaled alpha shapes,” *IEEE Visualization*, 1998.
  <https://doi.org/10.1109/VISUAL.1998.745286>
