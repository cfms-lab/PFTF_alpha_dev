# PFTF_alpha_dev

PFTF-guided local anisotropic alpha fields for robust point-cloud shape
reconstruction.

## Status

- Quality: **ToDo**
- Stage: G0-G3 complete plus a G4 conservative P2 fallback prototype smoke
- B0-P2 calibration and frozen held-out smoke runs exist; exact CGAL fallback
  does not.
- No PDF or DOCX is generated at this stage.

## Implemented first slice

The current Python prototype provides:

- an explicit radius versus squared-radius convention, with squared physical
  radius used internally;
- a 2D/3D floating-point Delaunay alpha filtration and finite critical-value
  enumeration;
- general and regularized codimension-one boundary extraction;
- cheap complex diagnostics: component count, Euler characteristic, GF(2)
  surface Betti numbers, simplex counts, and boundary-facet count;
- an auditable geometry/topology/stability/complexity objective scan;
- SPD construction \(M_i=L_iL_i^T+\varepsilon I\), local metric evaluation,
  confidence-aware fallback, and soft/hard alpha gates;
- deterministic train/calibration/held-out variants of the six required 3D
  synthetic families;
- B0 convex hull, B1 fixed normalized alpha, B2 exhaustive reference oracle,
  and B3 unlabeled persistence/resampling selection;
- B4 density-normalized top-cell scores using kNN spacing and complete
  downward closure;
- B5 local-PCA SPD scores with density normalization and a planarity-weighted
  normal penalty;
- P1 directed receiver/source scale messages and receiver imbalance, projected
  through bounded log-eigenvalues to a confidence-softened local SPD field;
- P1 confidence, reciprocity, relation-strength, metric-condition, and numeric
  fallback diagnostics;
- P2 with an explicit or reference-free calibration-only confidence threshold:
  a low-confidence cell must pass both P1 and trusted B4 by using
  `max(P1 score, B4 score)`;
- confidence-first calibration that freezes a pooled simplex-confidence
  threshold before selecting one B4/B5/P1/P2 multiplier on all six calibration
  cases, plus selected-set guard and complete-closure diagnostics;
- sampled-surface Chamfer, Hausdorff, F-score, component/Betti error,
  edge-incidence, and watertightness endpoints with a JSON audit trail.

This is a SciPy research baseline, not the later exact-predicate CGAL evaluation
path. B4/B5/P1/P2 reuse one Euclidean Delaunay tetrahedralization and change a
dimensionless top-cell filtration score; accepted cells are closed downward
before their boundary is extracted. P1 is a deterministic PFTF-style relation
prototype, not a trained field. P2 is a conservative B4 guard on that fixed
complex, not an exact CGAL fallback. The current path is auditable and
scale-invariant, but it is not an exact anisotropic Delaunay construction or a
globally consistent spatially varying metric complex.

Set up and verify:

```powershell
uv sync --dev
uv run ruff check src tests examples
uv run pytest
uv run python examples/critical_alpha_demo.py
uv run python -m pftf_alpha.benchmark --split held_out --calibrate-adaptive
```

The benchmark result records whether reference data influenced selection. B2
uses the dense reference as a declared per-case oracle. With
`--calibrate-adaptive`, B4/B5/P1/P2 multipliers use dense references only on the
separate calibration panel, freeze one shared multiplier per method, and record
reference-free frozen selection on every requested held-out case. P2 confidence
threshold calibration uses only input-point confidence and does not read dense
references. Without an
explicit or calibrated multiplier, these local methods remain declared per-case
oracles for diagnostic comparison. B3 uses the dense reference only after
selection to report endpoints.

The current topology implementation computes GF(2) surface Betti numbers
`(beta_0, beta_1, beta_2)` from each triangular complex and records L1 Betti
error against a declared synthetic target. The six targets are U concavity
`(1,1,0)`, opposing sheets `(2,0,0)`, torus `(1,2,1)`, disconnected spheres
`(2,0,2)`, sharp crease `(1,0,0)`, and the full-sphere target for missing patch
`(1,0,1)`. These targets are evaluation-only: B2/adaptive calibration still
uses its existing component-based topology term, so target Betti numbers do not
enter reference-free selection. The JSON `false_bridges` and `false_splits`
fields remain component-merge/split proxies, not exact handle or bridge counts.

Exact CGAL construction, exact false-safe endpoints, and globally consistent
spatially varying metrics remain pending. B4/B5/P1/P2 are filtration-score
methods on a fixed Euclidean Delaunay complex, not exact anisotropic alpha-shape
constructions. In the current smoke, the reference-free target fallback
fraction 25% produced threshold 0.268687 and 25.84% calibration fallback.
Held-out fallback was 7.49-48.99% of all cells and 11.01-42.31% of selected
cells, with zero score/selected-set guard violations and zero closure-incidence
failures. Betti-error sums were B4 20, B5 27, P1 25, and P2 25; P1/P2 recovered
the torus target exactly but not the other five targets. P2 still underperformed
P1 on mean F-score and Chamfer, and these checks are not evidence of zero
false-safe cases or of promotion. Very large critical values caused by
surface-sampling sliver tetrahedra are retained and reported rather than
silently clipped.

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
6. PFTF local metric with a confidence/B4 guard prototype; exact fallback pending.

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
