# PFTF_alpha_dev

PFTF-guided local anisotropic alpha fields for robust point-cloud shape
reconstruction.

## Status

- Quality: **ToDo**
- Stage: G0-G3 complete plus a G4 conservative P2 fallback prototype,
  evaluation-only cell/boundary bridge diagnostics, a calibration-only penalty
  ablation, a rejected boundary-owner intervention, and a rejected connected
  boundary risk-region/cut audit, plus an exact-predicate construction-readiness
  preflight, a fail-closed optional exact-backend handoff validator, an
  evaluation-only validated-connectivity shadow path, a built-in small-panel
  exact construction backend, and an exact-rational simplex filtration-value
  audit, plus evaluation-only exact-resampling connectivity/value and exact-B3
  candidate-selection shadows
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
- a built-in small-panel exact 3D Delaunay backend that converts the rational
  protocol coordinates to one exact integer scale, enumerates every four-point
  candidate, and retains exact empty circumspheres without consuming
  SciPy/Qhull candidate connectivity; it caps inputs at 64 points, skips
  coplanar candidates, and fails closed on exact empty-cosphere ambiguity
  instead of selecting an undocumented symbolic perturbation;
- an exact-rational filtration-value audit over every simplex of host-validated
  connectivity. It solves each intrinsic affine-hull Gram system with
  `Fraction`, tests Gabriel emptiness against every input point, applies the
  exact minimum-immediate-coface rule for non-Gabriel simplices, and compares
  values, ties, critical counts, and ordering with the floating filtration. It
  stores a canonical exact-record SHA-256 per case and cannot change primary
  selection;
- an evaluation-only exact-connectivity shadow that retains only host-accepted
  cells, rebuilds the filtration through `AlphaFiltration.from_top_simplices`
  with floating circumsphere values, reruns the requested methods, and compares
  every non-runtime output plus the bridge-risk probe under declared numerical
  tolerances without changing primary benchmark cases or selection;
- an evaluation-only exact-rounded-value shadow that requires the accepted
  connectivity, exact-value audit, and same-connectivity floating shadow;
  recomputes and digest-verifies the exact rational records, rounds them to
  binary64 for the runtime filtration, and separates selected-alpha, objective,
  endpoint, and candidate-bookkeeping differences without changing selection;
- an evaluation-only exact critical-index audit that compares the ordered
  top-simplex birth groups, B2/B3 selected critical ranks, selected full
  complexes and regularized boundaries, and B3 signature/candidate/persistence
  paths across floating, exact-rounded, and exact-rational views without
  changing primary selection;
- an evaluation-only exact-selected-threshold resampling audit that holds the
  B3 full complex, full-surface samples, resampled point subsets, floating
  resampled connectivity/values, and sampling seeds fixed, then measures only
  the effect of the floating versus exact-rounded selected alpha;
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

Schema 18 adds `--evaluate-exact-connectivity-shadow`. Accepted backend cells
are retained only after the schema-17 host checks and are passed through
`AlphaFiltration.from_top_simplices`, which rejects non-integer shapes,
out-of-range or repeated vertices, duplicate cells, and incomplete point
coverage. The shadow then reruns the requested B0-P2 methods and compares every
method output except runtime, plus the bridge-risk probe, using relative
`1e-12` and absolute `1e-15` float tolerances. It never replaces the primary
case reports or changes selection. The frozen held-out run again supplied no
backend, so it produced zero accepted cases, zero shadow reports, zero shadow
differences, and six explicit null reports; the schema-17 config, predicate
audit, and all 48 primary non-runtime results remained exactly unchanged. A
successful Qhull-connectivity fixture exercises the injection and comparison
path, but its exactness attestation is test-only and is not evidence that Qhull
constructed the cells exactly. This remains exact connectivity with floating
filtration values, not an exact alpha complex or a deployed fallback. The fixed
artifact is
`benchmark-out/smoke_b0_p2_exact_connectivity_shadow_held_out.json`
(SHA-256 `eeaf25337689b79e25973ae1a87c037a4ae942463d2e38eea27e0beda2622dd2`).

Schema 19 adds `--exact-python-backend`, a built-in exact Euclidean Delaunay
constructor for small audit panels. It converts the protocol's binary64
rational coordinates to a common exact integer scale, enumerates all
four-point candidates, and retains precisely the exact empty-sphere
tetrahedra. It does not use SciPy/Qhull connectivity. Inputs are capped at 64
points; coplanar candidates are skipped and exact cospherical ambiguity fails
closed because no symbolic perturbation is claimed. On the frozen 48-point,
six-case held-out panel, the independent schema-17 host validator accepted all
six responses from `pftf_alpha_python_exact`; the retained tetrahedron counts
were 208, 198, 188, 187, 209, and 141. All six shadow filtrations matched the
primary connectivity and every non-runtime B0-P2 output, while the complete
primary non-runtime result remained exactly equal to schema 18. This is exact
Euclidean Delaunay connectivity followed by floating-point circumsphere
filtration values. It is neither CGAL, an exact alpha complex, a spatially
varying anisotropic triangulation, nor a deployed fallback. The fixed artifact
is `benchmark-out/smoke_b0_p2_exact_python_backend_shadow_held_out.json`
(438,966 bytes; SHA-256
`5ce3863eae74fcdec439bba26c142006823abdbb84bf940ca4fce50aef15f5db`).


Schema 20 adds `--evaluate-exact-filtration-values`. For every simplex in the
host-validated 3D connectivity it computes the intrinsic circumsphere, Gabriel
emptiness, and non-Gabriel coface minimum with exact rational arithmetic. The
frozen six-case held-out audit covered 5,430 simplices. Only 953 floating values
were bitwise equal to the correctly rounded exact value; 4,477 differed. The
worst absolute error was `3.0440274899232807e-4`, the worst relative error was
`1.70727232787347e-9`, and the largest ULP difference was 11,128,782. Despite
those value-level differences, the audit found zero Gabriel disagreements,
zero exact-tie splits, zero exact-versus-rounded or exact-versus-floating
critical-count mismatches, and zero adjacent exact-order violations. All six
connectivity shadows still had zero non-runtime output differences, and the
complete primary non-runtime payload remained exactly equal to schema 19.
These results establish ordering agreement only for this frozen panel; exact
values remain audit-only and are not deployed into selection. This is not CGAL
or an exact end-to-end alpha evaluation. The fixed artifact is
`benchmark-out/smoke_b0_p2_exact_filtration_audit_held_out.json`
(448,635 bytes; SHA-256
`7c9aa989d122b9b2ed736977a32267e3b2c76cb66a51395747500e32654fc54a`).

Schema 21 adds `--evaluate-exact-value-shadow`. It runs only after exact
construction, the exact filtration-value audit, and the same-connectivity
floating-value shadow. Each case recomputes the exact rational records, checks
its SHA-256 and simplex count against the audit, converts each value with the
correctly rounded `Fraction`-to-binary64 conversion, and reruns the requested
methods without replacing primary cases or selection. Missing, rejected,
arithmetically invalid, or digest-mismatched prerequisites fail closed.

On the same frozen six-case B0-P2 held-out panel, all six exact-value shadows ran.
Every case had non-runtime B2 and B3 differences versus both primary and the
same-connectivity floating-value shadow. Candidate-range bookkeeping differed in
all six cases; selected-alpha fields exceeded the declared `1e-12` relative /
`1e-15` absolute tolerance in two B2 cases (`u_concavity` and `sharp_crease`),
and B3 objective/stability fields differed in two cases (`u_concavity` and
`disconnected_parts`). No surface/topology endpoint or bridge-risk-probe output
changed. The schema-20 primary non-runtime payload, exact-value audit, and
floating-connectivity shadow all remained exactly unchanged. Thus unchanged
critical counts and ordering did not imply identical method reports, although
the frozen endpoint metrics remained stable.

Exact rational values are only the source for this evaluation shadow. Threshold,
objective, resampling, and surface evaluation arithmetic remains floating-point,
and the shadow is not deployed into primary selection. This is neither an
end-to-end exact alpha complex nor CGAL evidence. The fixed artifact is
`benchmark-out/smoke_b0_p2_exact_value_shadow_held_out.json` (614,286 bytes;
SHA-256 `ea0d83f2b0a63f83591b8faeb32a9e710f2799fc970aa79c0a17889814db8e0d`).

Schema 22 adds `--evaluate-exact-critical-index-audit` after the complete schema-21
chain. It compares floating, correctly rounded exact, and exact-rational critical
counts and ordered top-simplex birth groups, then audits the B2/B3 selected
critical rank, full-complex SHA-256, and regularized-boundary SHA-256. For B3 it
also compares the component/Euler signature sequence, budgeted candidate-index
sequence, normalized log-radius plateau persistence, and topology/stability
objective terms. The audit fails closed and cannot replace primary selection.

On the frozen six-case 48-point held-out panel, all critical counts and ordered
birth groups matched. B2 and B3 had zero selected-index, selected-complex, and
selected-boundary mismatches; B2 objectives and endpoint payloads also matched.
B3 persistence arrays differed bitwise in all six cases, but the selected
persistence exceeded tolerance only for `disconnected_parts`. B3 retained the
same selected complex while its objective differed for `u_concavity` and
`disconnected_parts`: the topology term differed in one case and the resampling
stability term in two. Thus schema-21 report differences on this panel came from
numeric-radius paths, not a combinatorial selection change. A focused 16-point
test does produce a selected-index/complex divergence, so this is a frozen-panel
finding rather than a general invariance claim.

The primary cases, exact-filtration audit, connectivity shadow, and value shadow
remain structure-identical to schema 21 after runtime fields are excluded. The
fixed artifact is
`benchmark-out/smoke_b0_p2_exact_critical_index_audit_held_out.json` (642,885
bytes; SHA-256
`5a104add183795ba179eb3b943cc00f9f8135c80c1b6e7481c8c8b3e715a0894`).
Exact-rational values are still rounded before runtime evaluation; resampling
filtrations remain floating-point. This is not deployment, CGAL evidence, or an
end-to-end exact alpha-complex evaluation.

Schema 23 adds `--evaluate-exact-resampling-threshold-audit` after the full
schema-22 chain and requires B3 to have the same selected critical rank, full
complex, regularized boundary, and budgeted-candidate position in both shadows.
It regenerates the deterministic B3 resamples once, holds their floating
connectivity and filtration values fixed, verifies identical full-surface sample
hashes, and applies the two selected alpha values to each shared resample. It
records resampled full-complex and boundary hashes plus per-repeat stability
losses, then verifies that their means reproduce both B3 reports.

All six frozen cases were audited and all six selected alpha values differed.
Only `u_concavity` and `disconnected_parts` changed a resampled complex and
boundary, each in one of two repeats. Those were exactly the two stability
differences: `0.00026695180449573357` and `0.010215245512115334`,
respectively. The other four cases changed neither resampled boundary nor
stability. All reported stability values were reproduced, with zero reproduction
failures, zero stability differences without a boundary change, and zero
boundary changes without a stability difference.

The controlled result identifies the schema-22 B3 stability differences as
binary64 selected-threshold crossings in otherwise shared floating resampling
filtrations. It does not construct exact connectivity or exact filtration values
for a resampled point set. Primary cases and every schema-22 prerequisite remain
structure-identical after runtime fields are excluded. The fixed artifact is
`benchmark-out/smoke_b0_p2_exact_resampling_threshold_audit_held_out.json`
(667,995 bytes; SHA-256
`0aa9302181cabd48c4cc99a6a7a3b52e458b133813b76236a95aab52df9fdff9`).
This remains evaluation-only and cannot support deployment, CGAL parity, or an
end-to-end exact-resampling claim.

Schema 24 adds `--evaluate-exact-resampling-filtration-audit`, gated on the
complete schema-23 chain. It regenerates the same deterministic B3 point
subsets, runs the exact backend independently on every resample, applies the
host validator, constructs correctly rounded exact filtration values, and
compares the selected complex, regularized boundary, and stability loss at the
same exact-rounded B3 threshold. Full-surface samples, point subsets, and all
sampling seeds remain fixed; primary cases and selection remain immutable.

All 12 resamples on the frozen six-case 48-point held-out panel were accepted.
Their exact connectivity matched SciPy/Qhull in every repeat. All 12 had one or
more floating-versus-correctly-rounded filtration-value differences, totaling
7,723 simplex values with a maximum difference of 32,653,561 ULP. The large ULP
count is recorded, not treated by itself as a failure. Six repeats across
`u_concavity`, `opposing_sheets`, `torus`, and `disconnected_parts` changed the
selected complex, regularized boundary, and stability; the other six changed
none of them. Mean stability changed by `0.00026695180449573357`,
`0.00030055016177662954`, `0.00012341684299446362`, and
`0.010775278523207324`, respectively.

The primary cases and every schema-23 prerequisite remain structure-identical
after runtime fields are excluded. Thus the frozen resampling connectivity is
not the source of the new differences; correctly rounded exact filtration
values cross the fixed selected threshold in four cases. The audit still rounds
exact rationals to binary64 and leaves threshold selection, objectives, surface
sampling, and deployment outside the exact kernel. It is not a general
false-safe certificate, an anisotropic exact construction, or CGAL parity. The
fixed artifact is
`benchmark-out/smoke_b0_p2_exact_resampling_filtration_audit_held_out.json`
(698,999 bytes; SHA-256
`190ccdd545e6a4fde31af53ed5706015e202f296684958ce5d2d99fab9d138b9`).

Schema 25 adds `--evaluate-exact-b3-selection-shadow`, gated on the complete
schema-24 chain. It holds the correctly rounded exact full filtration, budgeted
critical-index sequence, geometry sampling, and endpoint sampling fixed. Every
B3 candidate is then evaluated twice: once with the original floating resample
filtrations and once with the schema-24 host-validated exact resample
filtrations. The exact-value B3 reference must be reproduced first; missing or
nonidentical prerequisites fail the case closed.

All six frozen cases and all 60 budgeted candidates were evaluated. Exact
resampling changed candidate stability and objective values in 28 candidates
across five cases: 4 in `u_concavity`, 5 in `opposing_sheets`, 4 in `torus`, 7
in `disconnected_parts`, and 8 in `missing_patch`; `sharp_crease` was unchanged.
The selected objective changed numerically in the first four cases, but every
case retained the same selected critical index, alpha, full complex,
regularized boundary, and endpoint metrics. Thus exact-resampling perturbations
did not change the frozen-panel B3 argmin.

Primary cases and every schema-24 section remain structure-identical after
runtime fields are excluded. This shadow is not deployed, still rounds exact
rationals to binary64, and leaves objective aggregation and surface sampling in
floating point. It is not a general false-safe certificate, an anisotropic
exact construction, or CGAL parity. The fixed artifact is
`benchmark-out/smoke_b0_p2_exact_b3_selection_shadow_held_out.json` (765,053
bytes; SHA-256
`5d153c584c6400f3ca5ea8b4a62bbf6dc2069a43ecf3c784d0b8b146d22a61dc`).

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
