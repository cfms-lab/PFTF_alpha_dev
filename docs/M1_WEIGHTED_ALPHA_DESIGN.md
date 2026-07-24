# M1 (B6) — weighted / regular alpha complex (predeclared design)

Status: **predeclared design; construction spike + ceiling probe done, mildly positive.**
Date: 2026-07-24 (Asia/Seoul).

M1 is the first method that escapes the shared limitation of B4/B5/P1/P2: they all
select subcomplexes of **one fixed isotropic Delaunay triangulation** by a per-cell
scalar, so a false-bridge tetrahedron baked into that connectivity cannot be
removed by rescoring without collateral damage. M1 changes the **connectivity**
itself by moving to a weighted (regular / power) Delaunay triangulation with
observed-geometry-derived per-point weights. It stays exactly computable (power
diagrams lift to a convex hull), so it composes with the deployed G4 exact
fallback rather than breaking it.

## 1. Construction (spike-verified)

The regular triangulation of weighted points is the lower convex hull of the
lift `p_i -> (p_i, ||p_i||^2 - w_i)` in one higher dimension. Spike results:
`w = 0` reproduces SciPy Delaunay exactly (175/175 cells); nonzero weights change
connectivity monotonically with weight magnitude; every output is a valid
triangulation. Implementation: `scipy.spatial.ConvexHull` on the 4D lift, keep
facets whose lift-axis normal points down.

**Point-coverage constraint.** Large weights submerge points out of the regular
triangulation, which violates the pipeline's "every input point is used"
invariant (`from_top_simplices`). M1 therefore **caps the weight scale** so all
points remain; a scale that would submerge any point is rejected during
calibration and never frozen.

## 2. Weights (predeclared)

Density weights only, for this first version (the orientation lever is M2):
`w_i = (s * spacing_i)^2`, where `spacing_i` is the geometric kNN spacing already
used by B4 (`knn_scales`, `adaptive.py:608`) and `s` is a single frozen
non-negative scale. `s = 0` is exactly B4's connectivity, so M1 strictly
generalizes B4. No labels, references, or Betti targets enter the weights.

## 3. Scoring and selection

M1 scores each regular-triangulation cell by its **proper weighted (power)
circumradius** normalized by local kNN spacing, then wraps the scores in
`AdaptiveCellFiltration` and selects by one frozen multiplier exactly as B4/B5
do. The weighted orthoradius reduces to the ordinary circumradius when weights
are zero, so `weight_scale = 0` remains identical to B4. The surface, endpoints,
downward closure, and diagnostics are unchanged.

**Manifold refinement (2026-07-24, probe-confirmed).** An earlier version scored
the regular connectivity with B4's *ordinary* circumradius; that mismatch
produced extra nonmanifold edges and higher Betti error. Switching to the proper
weighted-power circumradius drove nonmanifold edges toward zero and lowered Betti
error at fixed connectivity (e.g. `missing_patch` Betti 36 -> 11, `u_concavity`
nonmanifold 5 -> 0), while keeping or improving F-score. This is the version M1
now uses.

## 4. Predeclared calibration ablation

- Weight-scale grid `s in {0.0, 0.125, 0.25, 0.375, 0.5}` (0.0 = B4 connectivity;
  values above ~0.5 risk point submersion and are excluded up front).
- For each `s`, freeze one density multiplier on the six-case calibration panel
  with the existing `calibrate_adaptive_multiplier` pattern, evaluate the panel,
  and reject any `s` that submerges a point in any case.
- **Freeze rule:** pick the single `s` that does not regress against **either B4
  or B5** on the predeclared endpoints (normalized geometry loss, component
  error, surface Betti error, labeled false-bridge edges and faces, nonmanifold
  edges) and that strictly improves at least one geometry-or-topology endpoint on
  at least one family. Ties broken by (best mean geometry, then smallest `s`). If
  only `s = 0` qualifies, M1 collapses to B4 and is reported as no improvement.
- Evaluate the frozen `s` unchanged on the four-profile / three-seed G5 panel with
  the existing strict casewise B4/B5 envelope.

## 5. Ceiling-probe evidence (calibration only)

At capped weights M1 offers a mild, real advantage without clear regression:
`disconnected_parts` best-achievable F-score rose 0.555 -> 0.624 with Betti error
5 -> 2 at `s = 0.5`; single-component families were unregressed at `s <= 0.25`;
`opposing_sheets` (thin gap) was unchanged, as expected. The density weight is
partly redundant with B4's density scoring, which bounds M1's ceiling; the
thin-gap bridge is explicitly left to M2 (oriented normals).

## 5.1 Frozen ablation outcome (2026-07-24)

With the proper weighted scoring and a reference-free frozen multiplier, M1 at
`weight_scale = 0.375` **strictly dominates B4 on every declared endpoint**:
F-score 0.365 -> 0.384, geometry loss 0.1802 -> 0.1730, labeled bridge edges
48 -> 46, with identical component error (2), Betti error (9), and nonmanifold
edges (0). This is the first method improvement over B4 in the investigation.

It does **not** clear the strict "beyond both B4 and B5 on every endpoint" bar:
B5 has a slightly higher mean F-score (0.391) and one lower component error, even
though M1 is far better than B5 on Betti error (9 vs 105) and manifoldness
(0 vs 175). So `m1_promotes_over_baselines = false` while `m1_dominates_b4 =
true`. The residual F-score gap versus B5's dense, topologically degenerate
reconstruction is the remaining target — plausibly for M2 (oriented normals),
which can add surface detail without the topology cost.

## 6. Why this is different from prior negative evidence

- Unlike schemas 12/14/15 and the P1 probes, M1 does **not** rescore or surgically
  edit a fixed complex; it produces a **different triangulation**, so the achievable
  complexes are no longer confined to subcomplexes of one Delaunay diagram.
- Unlike multiplier backoff, M1 changes *which* tetrahedra exist rather than *how
  many* of a fixed set are kept, so it can drop a large-gap bridge tetrahedron
  outright instead of trading it against real surface.

## 7. Claim boundary

M1 is a floating-point research construction like B4/B5 (the regular triangulation
is computed with Qhull; the exact-weighted-alpha variant that would compose with
G4 is future work). It certifies nothing about exactness and does not by itself
justify promotion: promotion still requires frozen higher-fidelity held-out value
beyond both B4 and B5, and the thin-gap failure mode remains open until M2.
`promotion_supported` stays false.
