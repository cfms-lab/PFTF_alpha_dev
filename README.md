# PFTF_alpha_dev

PFTF-guided local anisotropic alpha fields for robust point-cloud shape
reconstruction.

## Status

- Quality: **ToDo**
- Stage: G0-G3 complete plus a G4 conservative P2 fallback prototype,
  evaluation-only cell/boundary bridge diagnostics, a calibration-only penalty
  ablation, a rejected boundary-owner intervention, and a rejected connected
  boundary risk-region/cut audit, plus an exact-predicate construction-readiness
  preflight and a fail-closed optional exact-backend handoff validator
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
- a reference-free per-cell bridge-risk probe that routes coherent normal fields
  to a parallel-normal signal and other fields to a density-normalized
  second-longest-edge signal; component labels are used only after scoring;
- a zero-strength-exact multiplicative P2 penalty primitive and calibration-only
  curve that audits proxy/endpoint alignment without changing benchmark
  selection or deploying a held-out penalty;
- a frozen-P2 boundary bridge localizer that scores output edges/faces from
  observed geometry, records selected-cell dual articulation/bridge structure
  separately, and uses component labels only for evaluation;
- a calibration-only iterative boundary-owner pruning audit that removes all
  owners of currently risky boundary faces, recomputes the boundary after each
  round, and requires the existing objective, geometry, component, Betti, and
  labeled-bridge promotion gates without changing held-out P2 selection;
- a calibration-only connected boundary-risk-region and safe-backbone-cut audit
  that joins flagged faces only through flagged edges, detects risky edges between
  safe-edge vertex components, and evaluates fixed structural candidates without
  changing held-out P2 selection;
- an exact-predicate readiness audit that promotes each binary64 input
  coordinate to its exact rational value, checks tetrahedron orientation and
  interior-facet in-sphere signs, and blocks promotion because it audits
  SciPy/Qhull connectivity rather than constructing an exact complex;
- a versioned exact-construction backend protocol that sends canonical
  exact-rational coordinate pairs over JSON, binds responses to the request
  SHA-256, requires backend name/version/kernel attestation, and revalidates all
  returned cells for point coverage, face incidence, convex-hull support, exact
  volume coverage, orientation, and in-sphere predicates without applying them
  to benchmark selection;
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
Each synthetic input point also carries an evaluation-only surface-component
label. The benchmark counts unique output-mesh edges and faces that span those
labels; a nonzero count is a direct discrete false-bridge witness for the two
declared multi-component families. It is not a general handle-localization or
exact-CGAL certificate.

Schema 11 adds this bridge-risk probe as an evaluation-only diagnostic. The
risk calculation accepts only observed points and the fixed Delaunay cells;
synthetic component labels are applied afterward for AUC, recall, and
false-positive-rate measurement. A risk above one means that the selected
route-specific threshold was exceeded. The probe does not alter B0-P2
selection, filtration scores, or extracted surfaces.

Exact CGAL construction, exact false-safe endpoints, and globally consistent
spatially varying metrics remain pending. B4/B5/P1/P2 are filtration-score
methods on a fixed Euclidean Delaunay complex, not exact anisotropic alpha-shape
constructions. In the current smoke, the reference-free target fallback
fraction 25% produced threshold 0.268687 and 25.84% calibration fallback.
Held-out fallback was 7.49-48.99% of all cells and 11.01-42.31% of selected
cells, with zero score/selected-set guard violations and zero closure-incidence
failures. Betti-error sums were B4 20, B5 27, P1 25, and P2 25; P1/P2 recovered
the torus target exactly but not the other five targets. P2 retained 39
cross-sheet edges/39 mixed faces for opposing sheets and 18 cross-part edges/19
mixed faces for disconnected parts. B2 reached two components on opposing
sheets but still retained 46 cross-sheet edges, demonstrating that component
count alone can be a false negative. Selection parameters and prior endpoints
were unchanged by this evaluation-only audit. P2 still underperformed P1 on
mean F-score and Chamfer, and these checks are not evidence of zero false-safe
cases or of promotion. Very large critical values caused by surface-sampling
sliver tetrahedra are retained and reported rather than silently clipped.

With frozen probe thresholds `(normal coherence, normal edge, normalized
length) = (0.9, 0.02, 1.8)`, the held-out opposing-sheets case reached AUC
0.99490, recall 0.87402, and FPR 0.02817; disconnected parts reached AUC 1.0,
recall 1.0, and FPR 0.0. Single-component flagged fractions still ranged from
0.0625 to 0.3298, so the probe is not promoted to a hard exclusion rule.
Comparison with the schema-10 artifact confirmed that all 48 pre-existing
non-runtime method results were unchanged.

Schema 12 adds `--evaluate-bridge-penalty`, which evaluates
`score * (1 + strength * max(risk - 1, 0))` on the calibration panel while
leaving every requested B0-P2 result unchanged. At strengths
`[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]`, low strengths improved the
mean geometry/objective slightly but increased labeled false-bridge edges;
strength 0.8 reduced edges from 34 to 25 but regressed geometry and Betti error.
No positive strength passed all promotion gates. Selected flagged-cell count and
selected mean risk each had Spearman correlation -0.26998 with output bridge
edges, so per-cell risk suppression is not an adequate boundary-safety proxy.

Schema 13 adds `--evaluate-boundary-bridges`. On the frozen held-out P2 output,
the route-specific boundary-edge risk and maximum incident-edge face risk use
only observed points and the already selected P2 boundary. Across 550 boundary
faces, the fixed `risk > 1` rule reached AUC 0.99523, recall 0.96552, and FPR
0.02439; across 807 boundary edges it reached AUC 0.98905, recall 0.89474, and
FPR 0.00933. It detected all 19 mixed disconnected-parts faces and 37 of 39
mixed opposing-sheets faces. The separate selected-cell dual bottleneck score
reached pooled face AUC 0.58274, including no cut signal for opposing sheets,
so it remains audit-only and is not fused into the geometric risk. All 48
pre-existing non-runtime B0-P2 results and the schema-11 cell probe remained
unchanged. This is localization evidence, not a deployed repair or an exact
false-safe certificate.

Schema 14 adds `--evaluate-boundary-intervention`. On the calibration panel it
removes all unique owners of current boundary faces with `risk > 1`, then
recomputes the boundary after every round; references and component labels enter
only the promotion gate. At rounds `[0, 1, 2, 4]`, the baseline objective,
geometry term, Betti error, and labeled false-bridge edge/face counts were
0.41890, 0.20157, 25, and 34/35. One round slightly improved objective and
geometry but increased bridges to 50/53. Four rounds removed 121 of 579 selected
cells and reduced bridges to 11/12, but regressed objective and geometry to
0.45821 and 0.24757. No positive depth passed every gate, so boundary-owner
peeling is rejected and remains undeployed. All 48 held-out non-runtime B0-P2
results, the schema-11 cell probe, and schema-13 localization were unchanged.

Schema 15 adds `--evaluate-boundary-region-cuts`. On the calibration panel it
found 13 flagged-edge-connected face regions, with a largest region of 25 faces.
The `largest_risk_region` candidate removed 28 cells across five cases, improved
Betti error from 25 to 22, but regressed objective/geometry from
0.41018/0.19247 to 0.41441/0.19727 and increased labeled bridge edges/faces from
34/35 to 43/45. Removing flagged edges left seven safe-edge vertex components in
total, but no flagged edge joined distinct safe components, so
`safe_backbone_cut` produced no candidate and remained exactly equal to baseline.
No strategy passed all promotion gates; neither was frozen or deployed. All 48
held-out non-runtime B0-P2 results, the schema-11 cell probe, and schema-13
localization remained unchanged. Simple boundary-region and cut heuristics are
therefore exhausted; exact-construction fallback is the next implementation path.

Schema 16 adds `--evaluate-exact-predicates` as a non-selecting G4 preflight.
It interprets all 288 held-out binary64 coordinates as exact rationals and
audits 1,131 supplied Qhull tetrahedra and 2,094 interior facets with exact
3D orientation and in-sphere signs. The frozen panel had zero degenerate
tetrahedra, exact cospherical facets, local-Delaunay violations, non-manifold
facets, or float/exact sign disagreements, so the supplied connectivity is
predicate-consistent for this dataset. This does not reconstruct the
triangulation, produce an exact alpha complex, or certify CGAL behavior.
`exact_construction_backend_integrated=false`, promotion remains blocked, and
all 48 prior non-runtime B0-P2 results plus the bridge probe stayed exactly
unchanged. The fixed artifact is
`benchmark-out/smoke_b0_p2_exact_predicate_audit_held_out.json`
(SHA-256 `f82ca6feae290c4b47c044601ecb0c6d2a07555995fc7a71ae6c2eec74951e9b`).

Schema 17 adds `--evaluate-exact-construction` and an optional explicit backend
command. The protocol sends one canonical JSON request per case, encodes every
coordinate as an exact numerator/denominator pair, binds the response to the
request SHA-256, and requires backend name, version, kernel, and exact-
construction attestation. The host then checks point coverage, face incidence,
convex-hull supporting facets, exact tetrahedral-versus-boundary volume, and the
schema-16 exact orientation/in-sphere predicates. The frozen held-out run
provided no backend, so `backend_requested=false`, accepted zero cases, changed
no selection, and remained blocked by `no_exact_construction_backend`. Its
schema-16 predicate result, config, bridge probe, and all 48 non-runtime B0-P2
results stayed exactly unchanged. Protocol fixtures verify acceptance of a
complete bipyramid and rejection of request-hash mismatch, missing-point, and
local-Delaunay-violation responses. Even a validated backend result is not yet
applied to benchmark selection, so it cannot support promotion. The fixed
artifact is
`benchmark-out/smoke_b0_p2_exact_backend_handoff_held_out.json`
(SHA-256 `31aa04659eab4eeee0977f388facc7e37cc53e4c7cfd0324a43a328264ac2be4`).

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
