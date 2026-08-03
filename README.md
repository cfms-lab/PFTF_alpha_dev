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

### G5 frozen held-out robustness preflight

Run the separate deterministic preflight with:

```powershell
uv run python -m pftf_alpha.g5_validation
```

The command freezes the P2 confidence threshold and one B4/B5/P1/P2
multiplier on the six-case calibration panel, then leaves them unchanged for
four held-out profiles: base, two-thirds-density sparse, doubled-noise, and a
family-specific harder-geometry shift. The default artifact contains three
paired seeds per family and profile (72 held-out cases total). Dense references
and synthetic labels are evaluation-only.

The frozen default run selected confidence threshold `0.2632006823` with
calibration fallback fraction `0.2514417532`. Across profiles, P2 mean fallback
fractions ranged from `0.22717` to `0.28956`, and the B4 guard had zero score-level
violations. That invariant did not translate into endpoint promotion: P2 failed
the strict casewise B4/B5 endpoint envelope in all four profiles. Its mean
F-score margins were `-0.047907`, `+0.000172`, `-0.048930`, and `-0.049247` for
base, sparse, noisy, and hard geometry; its geometry-loss margins were negative
in every profile. Topology burden exceeded the envelope by one in the first
three profiles and tied it in hard geometry, while labeled false-bridge edges
still exceeded the envelope by 86, 55, 80, and 4.

Accordingly, `endpoint_preflight_supported` and `promotion_supported` are both
false. This is a synthetic G5 preflight, not higher-fidelity real held-out
evidence, and it does not close the undeployed exact/fail-closed G4 fallback.
The artifact is
`benchmark-out/g5_frozen_held_out_preflight.json` (1,078,057 bytes; SHA-256
`ca1fee74276635b4848ead5a09710d4857c4d232a42aa077b50f6251c62dde69`).

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

## Risk-localized reacquisition Phase 0

`pftf_alpha.reacquisition` runs a paired synthetic ROI-rescan experiment on the
held-out thin-gap case. It compares uniform and boundary-risk-targeted added
samples at exactly the same point budget while keeping component labels and the
evaluation reference out of selection.

```powershell
python -m pftf_alpha.reacquisition `
  --output benchmark-out/risk_targeted_reacquisition_phase0.json
```

The frozen 8-repeat panel did not pass: 1.25x--1.75x total density never removed
the false bridge. A separate exploratory sweep reduced targeted cross-component
kNN mixing to `0.0189` at 640 total points, but unchanged B5 still merged the
sheets in all three repeats. This falsifies the direct `risk -> rescan -> rerun
B5` path; it supports using reacquisition as a sampling-sufficiency diagnostic
followed by a fail-closed or connectivity-changing reconstruction. See
[docs/ACTIVE_REACQUISITION_PHASE0.md](docs/ACTIVE_REACQUISITION_PHASE0.md).

`pftf_alpha.sampling_gate` implements the next observed-only diagnostic. Across
the 40-case Phase 1 calibration panel and a disjoint 40-case Phase 1b held-out
panel, its inferred sufficient/insufficient sampling class matched the
evaluation labels in all 80 cases and it made zero false-safe accepts. Exact
three-way held-out routing accuracy was only `0.875`, however, because five
under-resolved cases failed closed as `unsupported` rather than
`rescan_required`. B5 produced no safe positive output, so acceptance coverage
cannot be tested and deployment remains unsupported. See
[docs/SAMPLING_SUFFICIENCY_GATE_PHASE1.md](docs/SAMPLING_SUFFICIENCY_GATE_PHASE1.md).

`pftf_alpha.two_layer_connectivity` supplies the first safe positive
reconstruction candidate for the declared parallel-sheet regime. It infers two
layers from observed coordinates, constructs one 2D Delaunay surface in each
layer tangent plane, and blocks cross-layer faces by construction. On a new
40-case synthetic held-out panel, all 16 sampling-sufficient cases were accepted
and truly safe (false-safe 0); mean F-score improved from B5 `0.3922` to `0.7564`,
while component error, labeled bridge edges, and Betti error fell to zero. The
result is parallel-sheet-specific and remains non-deployable; see
[docs/TWO_LAYER_CONNECTIVITY_PHASE2.md](docs/TWO_LAYER_CONNECTIVITY_PHASE2.md).

`pftf_alpha.two_layer_stress` tests that candidate on a separately frozen
48-case Phase-3 panel. All 32 positive rotated, shallow-curved, varying-gap, and
partial-overlap cases were accepted and truly safe, with mean F-score improving
from B5 `0.4714` to `0.8256` and Betti-error sum falling from `1159` to zero.
All 16 declared out-of-scope near-contact and crossing cases were rejected;
false-safe count was zero. This broadens the synthetic claim to globally
separable, approximately parallel two-layer surfaces, but real-scan and
deployment support remain false. See
[docs/TWO_LAYER_STRESS_PHASE3.md](docs/TWO_LAYER_STRESS_PHASE3.md).

`pftf_alpha.two_layer_boundary` then maps the operating boundary on a frozen
192-case Phase-4 sweep. Tilt and near-contact stresses fail closed, and every
tested overlap offset remains safe. Curvature is the critical blind spot:
curvature `0.36`, `0.48`, and `0.60` produce 6, 6, and 3 false-safe accepts.
The last fully reliable tested curvature is `0.24`; the current global
normal-coordinate gate does not detect the later layer-assignment failure.
Accordingly `phase4_diagnostic_supported=false` and the candidate remains
non-deployable. See
[docs/TWO_LAYER_BOUNDARY_PHASE4.md](docs/TWO_LAYER_BOUNDARY_PHASE4.md).

`pftf_alpha.curvature_guard` repairs that specific safety failure with an
observed-only orientation-tensor coherence guard. Threshold `0.82` was frozen
from Phase-4 calibration, then evaluated on a new 192-case seed. The base router
reproduced 14 false-safe accepts; the guard removed all 14 while retaining 116
of 117 safe accepts (`99.15%`). This supports a synthetic fail-closed safety
envelope, not a universal curvature certificate or PFTF-SPD claim. See
[docs/CURVATURE_GUARD_PHASE4B.md](docs/CURVATURE_GUARD_PHASE4B.md).

`pftf_alpha.guard_domain_shift` tests that fixed threshold under crossed point
density, noise, and curvature-shape shifts. The 360-case Phase-5 panel fails:
false-safe accepts fall from 57 to 12 but do not reach zero, and safe-accept
retention is only `79.03%`. Sparse 96-point safe cases are over-rejected, while
dense asymmetric-converging cases can exceed coherence `0.82` and remain
false-safe. A single global coherence cutoff therefore does not transfer; see
[docs/CURVATURE_GUARD_DOMAIN_SHIFT_PHASE5.md](docs/CURVATURE_GUARD_DOMAIN_SHIFT_PHASE5.md).

`pftf_alpha.local_order_guard` replaces the global cutoff with density-binned
normal coherence plus a density-normalized tangent-plane layer-order margin.
On the frozen 360-case Phase-6 held-out it retains 174/190 safe accepts
(`91.58%`) and removes 59/61 false-safe accepts, including every false-safe at
N=160 and N=256. Two sparse N=96 false-safe cases remain, however, so the
predeclared zero-false-safe gate fails and `phase6_supported=false`. Thresholds
were not retuned after inspection; see
[docs/LOCAL_ORDER_GUARD_PHASE6.md](docs/LOCAL_ORDER_GUARD_PHASE6.md).

`pftf_alpha.shared_trend_inference` addresses the upstream failure instead of
adding another rejection threshold. It removes one shared quadratic tangent-
plane trend, clusters the residual layer offsets, recomputes the sampling gate,
and triangulates the inferred layers. On a frozen 360-case Phase-7 seed it
retains all 186 base-safe accepts, reduces false-safe accepts from 60 to zero,
and repairs 58/60 base false-safe cases into accepted safe outputs (`96.67%`).
The other two fail closed as `rescan_required`. Thus `phase7_supported=true`
within the generator-matched synthetic two-layer regime, while deployment and
PFTF-SPD novelty remain unsupported. See
[docs/SHARED_TREND_INFERENCE_PHASE7.md](docs/SHARED_TREND_INFERENCE_PHASE7.md).

`pftf_alpha.sensor_stress` then freezes that candidate and tests 216 cases with
one-sided occlusion, 75:25 layer imbalance, anisotropic noise, 1-5% spatial
outliers, and sinusoidal/local-bump surfaces. Phase 8 fails: all 56 candidate
false-safe accepts are contaminated cases, showing that the current route has
no outlier policy. A useful operating boundary nevertheless appears: at N=160
and N=256 all 96 non-outlier stress cases are accepted safely, including every
nonquadratic case, while sparse N=96 coverage is only `43.75%`.
`phase8_supported=false`; see
[docs/SENSOR_STRESS_PHASE8.md](docs/SENSOR_STRESS_PHASE8.md).

`pftf_alpha.outlier_guard` adds a frozen MAD/leverage-studentized shared-trend
residual score multiplied by local-density anomaly. Phase 9 removes 54/58
outlier false-safe accepts, including every accepted 3% and 5% contamination
case, but four 1% cases remain and safe retention is only `88.70%`. The guard
also over-rejects localized nonquadratic bumps (9/22 retained), demonstrating
that a shared-quadratic residual is not a shape-agnostic outlier certificate.
`phase9_supported=false`; see
[docs/OUTLIER_GUARD_PHASE9.md](docs/OUTLIER_GUARD_PHASE9.md).

`pftf_alpha.local_surface_consensus` separates source provenance from realized
geometry/topology harm and replaces the global residual with a same-layer,
leave-one-out local tangent-plane score. On the once-frozen 216-case Phase-10
panel it reduces harmful-outlier false-safe accepts from 55 to 1, but retains
only 23/39 (`58.97%`) clean/local-bump safe accepts. The score distributions
overlap: a cutoff strict enough to remove every harmful case would retain only
`38.46%`, while a cutoff that retains 90% still leaves two harmful cases.
Thresholds were not retuned. `phase10_supported=false`, so trimmed
reconstruction and real-scan validation were not started; see
[docs/LOCAL_SURFACE_CONSENSUS_PHASE10.md](docs/LOCAL_SURFACE_CONSENSUS_PHASE10.md).

`pftf_alpha.multiscale_surface_consensus` replaces that representation with
leave-one-out local quadratic fits at 12, 18, and 24 neighbours and separates
threshold calibration from held-out evaluation. Calibration removes 53/53
harmful false-safe accepts while retaining 36/40 (`90.00%`) clean/local-bump
safe accepts. At the frozen threshold, however, the untouched held-out panel
leaves one of 54 harmful false-safes, despite retaining 41/42 (`97.62%`) safe
accepts. Tightening the score enough to remove that case would reduce
calibration retention to `87.50%`, so the threshold was not retuned.
`phase11_supported=false`; see
[docs/MULTISCALE_QUADRATIC_CONSENSUS_PHASE11.md](docs/MULTISCALE_QUADRATIC_CONSENSUS_PHASE11.md).

`pftf_alpha.local_insertion_influence` then measures the change in neighbour
surface predictions when each omitted point is inserted into a local quadratic
fit. A two-coordinate peak/support accept rectangle is selected on a separate
calibration seed. Calibration removes 52/52 harmful false-safes with 43/43
(`100%`) clean/local-bump retention, but the frozen rectangle leaves one of 52
harmful false-safes on held-out while retaining 42/42 safe focus cases. A
post-hoc search shows that stricter rectangles could separate both panels, so
the negative is calibration-margin transfer rather than demonstrated
representation overlap. Held-out thresholds were not retuned and
`phase12_supported=false`; see
[docs/LOCAL_INSERTION_INFLUENCE_PHASE12.md](docs/LOCAL_INSERTION_INFLUENCE_PHASE12.md).

`pftf_alpha.conservative_influence_calibration` keeps the Phase-12
representation fixed and selects one rectangle from two new calibration
cohorts by maximizing worst-cohort retention. Both calibration cohorts remove
all 105 harmful false-safes at just over 90% clean/local-bump retention, but the
frozen final held-out still leaves one of 55 harmful cases. Exhaustive
three-panel analysis finds no zero-harm rectangle retaining 90% on every panel;
the best minimum retention is only `43.18%`. This changes the diagnosis from a
calibration-margin problem to representation overlap for the peak/support
influence family. `phase13_supported=false`; see
[docs/CONSERVATIVE_INFLUENCE_CALIBRATION_PHASE13.md](docs/CONSERVATIVE_INFLUENCE_CALIBRATION_PHASE13.md).

`pftf_alpha.observed_identifiability` then tests a broader 14-feature
observed-only signature rather than proposing another guard. The signature
combines layer balance, normalized within/cross-layer spacing, gap and
thickness, local insertion influence, and multiscale quadratic residuals.
Nearest-class diagnosis still misses 3/54 harmful calibration cases and 4/51
harmful held-out cases; six of the seven misses are 1% contamination. Held-out
safe specificity is `97.50%`, but harmful recall is only `92.16%`, below the
predeclared 100% safety requirement. Thus `feature_identifiable=false` within
this signature family and `guard_supported=false`; see
[docs/OBSERVED_IDENTIFIABILITY_AUDIT_PHASE14.md](docs/OBSERVED_IDENTIFIABILITY_AUDIT_PHASE14.md).

`pftf_alpha.paired_scan_persistence` explicitly expands the information set to
one independently sampled repeat of the same synthetic surface/stress family.
It predicts each primary point from same-layer replicate local quadratic fits
and calibrates a peak/support persistence rectangle on two new cohorts. The
joint zero-harm rectangle removes all 56 harmful false-safes in each cohort,
but retains only 36/42 (`85.71%`) and 37/43 (`86.05%`) safe control/local-bump
accepts. Because both are below the predeclared 90% gate, the final held-out
panel was not opened and thresholds were not retuned. Thus
`phase15_supported=false`, including for paired synthetic support; real
registration, trimmed reconstruction, and deployment remain unsupported. See
[docs/PAIRED_SCAN_PERSISTENCE_PHASE15.md](docs/PAIRED_SCAN_PERSISTENCE_PHASE15.md).

`pftf_alpha.studentized_paired_scan` next replaces the fixed residual scale
with replicate-only quadratic LOO prediction error and query leverage. Two
fresh calibration cohorts still reach zero guarded harm, but their best joint
zero-harm rectangle retains only 18/43 (`41.86%`) and 11/42 (`26.19%`) safe
control/local-bump accepts. The predictive uncertainty suppresses some desired
curvature error but also suppresses the nonpersistent outlier residual itself.
Both calibrations fail, so the final held-out panel remains unopened and
`phase16_supported=false`. This score is not retuned; see
[docs/STUDENTIZED_PAIRED_SCAN_PHASE16.md](docs/STUDENTIZED_PAIRED_SCAN_PHASE16.md).

`pftf_alpha.matched_pair_consistency` then tests a stronger information model:
the simulator supplies exact one-to-one acquisition IDs for primary/repeat
returns, and the guard uses only robust axis-standardized matched displacement.
Fresh calibration A/B panels remove all 52/58 harmful false-safes with 43/43
(`100%`) focus retention in each. The frozen final held-out also removes all 55
harmful cases with 43/43 focus retention, so `phase17_supported=true` and
`exact_correspondence_synthetic_supported=true`. One harmless
source-provenance violation remains accepted, and the simulator uses hidden
truth only to construct matched returns. Therefore real correspondence,
registration, trimming, and deployment remain unsupported; see
[docs/MATCHED_PAIR_CONSISTENCY_PHASE17.md](docs/MATCHED_PAIR_CONSISTENCY_PHASE17.md).

`pftf_alpha.matched_pair_stress` then freezes that score and evaluates exact
pairs, 0.5-degree repeat-cloud rotation, 10% missing pairs, 2% cyclic pair-ID
mismatch, and their combined perturbation on two fresh calibration cohorts.
One common zero-harm rectangle retains 43/43 focus accepts in each cohort for
exact, rotation-only, and missing-only profiles. It retains 0/43 in both
`mismatch_02` profiles and only 0/43 and 1/43 for `combined`, because an
incorrect pair appears as a large physical displacement to the score. Both
calibrations therefore fail the predeclared all-profile gate, final held-out is
not opened, and `phase18_supported=false`. Phase 17 remains an exact-ID
synthetic upper bound; see
[docs/MATCHED_PAIR_STRESS_PHASE18.md](docs/MATCHED_PAIR_STRESS_PHASE18.md).

`pftf_alpha.tangential_pair_confidence` next robustly aligns the presented
pairs and tries to remove ID errors using local tangent-plane residual divided
by local spacing, while keeping the Phase-18 matched-displacement rectangle
frozen. The joint A/B cutoff removes all 2,738 truth-mismatched pairs, but also
retains only `71.74%` of truth-correct pairs versus the preregistered `99%`
requirement. Harmful false-safes remain 246/275 and 225/250, despite 100%
case-level safe-focus retention. A low-scoring wrong pair overlaps correct
pairs within the same local-bump case, so the failure is representation overlap
rather than threshold selection. Both calibrations fail and final remains
unopened; `phase19_supported=false`. See
[docs/TANGENTIAL_PAIR_CONFIDENCE_PHASE19.md](docs/TANGENTIAL_PAIR_CONFIDENCE_PHASE19.md).

`pftf_alpha.global_tangential_assignment` then removes the scalar cutoff and
solves a whole-set Hungarian assignment using the same aligned local-tangent
cost, again freezing the Phase-18 downstream rectangle. It restores zero harm
in both fresh calibration panels (`260→0`, `270→0`) and repairs over 97% of the
pure mismatch profile, but assignment accuracy is only `97.89%` and `97.81%`.
Even exact presented pairs are unnecessarily permuted, reducing safe-focus
retention to `83.18%` and `76.67%`; combined mismatch repair also remains below
90%. Both panels fail, final is not opened, and `phase20_supported=false`. See
[docs/GLOBAL_TANGENTIAL_ASSIGNMENT_PHASE20.md](docs/GLOBAL_TANGENTIAL_ASSIGNMENT_PHASE20.md).

`pftf_alpha.cycle_gated_assignment` next preserves the presented pairing by
default and applies only complete Hungarian permutation cycles whose relative
cost gain exceeds a truth-supervised cutoff frozen on the already-open Phase-20
A/B cohorts. The cutoff rejects all 3,275 development non-improving cycles, and
the exact, registration-only, and missing-only profiles retain 100% assignment
accuracy and focus acceptance. However, it also rejects 304/750 improving
cycles. `mismatch_02` repairs only `62.08%/64.44%`, `combined` repairs only
`43.80%/44.44%`, and the combined profiles remain below 99% accuracy with only
50% focus retention. Both development panels fail, so validation seeds
23600804/23700804 and final seed 23800804 remain unopened.
`phase21_supported=false`; a single observed cycle-gain threshold is closed for
this protocol. See
[docs/CYCLE_GATED_ASSIGNMENT_PHASE21.md](docs/CYCLE_GATED_ASSIGNMENT_PHASE21.md).

`pftf_alpha.multivariate_cycle_signature` then freezes 14 observed cycle
features spanning within-cycle tangent, normal, and total cost changes. A ridge
linear score trained only on fresh seed 23900804 accepts 276/343 strictly
correcting cycles and 0/1,524 unsafe cycles, so it introduces no mismatches.
However, `mismatch_02` repair is only `77.78%`, `combined` repair is `53.64%`,
and safe-focus retention is `89.27%`. One missing-only harmful case is still
accepted under 100% correct unchanged pairing because its matched-displacement
peak `9.2668` lies below the frozen `10.9226` limit. Training A fails, so seeds
24000804--24300804 remain unopened and `phase22_supported=false`. The next
bottleneck is no longer correspondence alone: the matched-displacement guard
also lacks fresh-seed transfer. See
[docs/MULTIVARIATE_CYCLE_SIGNATURE_PHASE22.md](docs/MULTIVARIATE_CYCLE_SIGNATURE_PHASE22.md).

`pftf_alpha.matched_guard_signature` then isolates the downstream exact-pair
guard on the exact, 0.5-degree registration, and 10%-missing profiles. A fixed
12-feature displacement-tail ridge score perfectly separates training A:
harmful 168 to 0 with 120/120 focus and 351/351 all-safe retention. On fresh
development B it retains all 123 focus-safe accepts but leaves one of 159
harmful false-safes in `missing_10pct`; exact and registration still pass. The
opened A/B scores retain a post-hoc focus-safe gap, so this is a cutoff-margin
transfer failure, not demonstrated focus overlap. Validation seeds
24600804/24700804 and final 24800804 remain unopened, and
`phase23_supported=false`. See
[docs/MATCHED_GUARD_SIGNATURE_PHASE23.md](docs/MATCHED_GUARD_SIGNATURE_PHASE23.md).

`pftf_alpha.split_cohort_guard_calibration` then keeps the same 12 features and
ridge score family but separates coefficient fitting (seed 24900804) from a
conservative quarter-gap cutoff cohort (seed 25000804). The score-fit cohort has
a positive harmful/focus gap, but the fresh cutoff cohort reverses the ordering:
minimum harmful `0.02888` versus maximum focus-safe `0.13806`. The limiting
missing-only harmful case lies inside every coordinate-wise focus-safe feature
range and at a typical focus-neighbor distance. Calibration therefore fails
closed before a deployable cutoff is formed; validation seeds 25100804--25300804
remain unopened and `phase24_supported=false`. The same global 12-feature
linear score should not be recalibrated again without new observed local/spatial
evidence. See
[docs/SPLIT_COHORT_GUARD_CALIBRATION_PHASE24.md](docs/SPLIT_COHORT_GUARD_CALIBRATION_PHASE24.md).

`pftf_alpha.matched_subset_reconstruction` then addresses the limiting case in
which missing-pair deletion removed the only harmful vertex from the observed
guard evidence. Missing profiles are reconstructed from retained primary IDs
before the frozen Phase-24 score is applied. The frozen cutoff calibration
reproduces, but the corrected design gate finds two originally safe
`missing_10pct/upper_occlusion` accepts that gain 10 and 5 clean cross-layer
faces after trimming. An initial evaluator bug omitted this no-new-harm gate
and improperly opened seeds 25400804--25600804; those results are contaminated
and provide no support. The corrected run stops at design score fit with
`design_gate_passed=false` and `phase25_supported=false`. Matched-subset
trimming therefore replaces one unobserved-harm failure with reconstruction-
induced topology harm; see
[docs/MATCHED_SUBSET_RECONSTRUCTION_PHASE25.md](docs/MATCHED_SUBSET_RECONSTRUCTION_PHASE25.md).

`pftf_alpha.frozen_partition_reconstruction` then keeps the full-primary
observed layer assignment fixed when 10% of pairs are missing and triangulates
only within each retained frozen layer. It eliminates Phase 25's newly created
endpoint harm and retains every fresh focus-safe accept. The run also corrects
freshness at the case level: sequential bases 25700804--25900804 overlap prior
case seeds, so the disjoint bases 27500804--27700804 are used instead.
Validation A/B pass with zero harm, but the final missing profile retains one
harmful outlier case: score `0.19257` lies just below cutoff `0.19784`.
Therefore `phase26_supported=false`; this is a cutoff-margin transfer failure,
not reconstruction-induced harm or demonstrated feature overlap. See
[docs/FROZEN_PARTITION_RECONSTRUCTION_PHASE26.md](docs/FROZEN_PARTITION_RECONSTRUCTION_PHASE26.md).

`pftf_alpha.focus_envelope_cutoff` then freezes a stricter cutoff one binary64
step above the maximum focus-safe score in the already-open Phase-26 A/B
cohorts. The design focus maximum `0.18181536` and routed-harm minimum
`0.28460336` reproduce with a positive gap, and every design panel passes. On
case-seed-disjoint Validation A, however, one missing-profile harmful case
scores `0.16754805` below the frozen cutoff while the fresh maximum focus score
is `0.19702923`, reversing the ranking gap to `-0.02948117`. Validation A has
harm 153 -> 1, focus 125/126, all-safe 362/366, and no introduced endpoint
harm. It fails, so Validation B and final remain unopened and
`phase27_supported=false`. A post-hoc lower cutoff could meet the 90% focus
gate on this opened panel, but is forbidden evidence-dependent retuning. See
[docs/FOCUS_ENVELOPE_CUTOFF_PHASE27.md](docs/FOCUS_ENVELOPE_CUTOFF_PHASE27.md).

`pftf_alpha.local_spatial_residual_guard` then adds genuinely spatial observed
evidence: each standardized matched displacement is compared with the
componentwise median displacement of its eight nearest primary-coordinate
neighbors, and the case uses the 95th-percentile local residual. A cutoff
frozen on opened development panels removes the Phase-27 residual case while
retaining 378/381 focus cases under the combined score/local route. On new
case-seed-disjoint Validation A/B and final panels, the combined route has harm
`171 -> 0`, `171 -> 0`, and `168 -> 0`, with 100% focus retention and zero
introduced endpoint harm. Thus `phase28_supported=true` for the preregistered
synthetic protocol. However, the predecessor also has zero harm on all three
fresh panels; the local guard's nine additional fresh rejections are all
non-focus safe cases. Fresh incremental rescue by the local feature is
therefore not established. See
[docs/LOCAL_SPATIAL_RESIDUAL_GUARD_PHASE28.md](docs/LOCAL_SPATIAL_RESIDUAL_GUARD_PHASE28.md).

`pftf_alpha.targeted_local_residual_challenge` then tests the marginal local
claim on a preregistered N=96 challenge containing 64 repeats each of control,
local-bump, and 1% outlier stress. Validation A contains one harmful
`outliers_01` primary case accepted by the predecessor in all three profiles,
so the panel is informative. The frozen q95 local guard rescues none: total
harm is `57 -> 3 -> 3` for original, predecessor, and combined routes. All
three residual q95 values lie below the frozen local cutoff, while their
maximum local residuals are above it. Validation A therefore fails and the
fixed Validation B/final seeds remain unopened; `phase29_supported=false`.
The maximum statistic is post-open diagnostic evidence only and is not a
Phase 29 retuning. See
[docs/TARGETED_LOCAL_RESIDUAL_CHALLENGE_PHASE29.md](docs/TARGETED_LOCAL_RESIDUAL_CHALLENGE_PHASE29.md).

`pftf_alpha.tail_sensitive_local_guard` then compares six observed-only tail
summaries on all nine opened development panels and freezes the
maximum-to-q95 local-residual ratio. Its strict cutoff
`1.6636368999089541` retains 1248/1281 development focus accepts and removes
all three Phase-29 residual rows. On case-seed-disjoint targeted bases
30500804--30700804, all panels pass with aggregate harm `171 -> 2 -> 0`, focus
`697/735`, and zero introduced endpoint harm. The two Phase-28 predecessor
residual rows occur only in final and are both rejected by tail ratios 1.9092
and 2.1968. Therefore `phase30_supported=true` for the preregistered synthetic
targeted protocol. Real correspondence and deployment remain unsupported. See
[docs/TAIL_SENSITIVE_LOCAL_GUARD_PHASE30.md](docs/TAIL_SENSITIVE_LOCAL_GUARD_PHASE30.md).

`pftf_alpha.open3d_real_pair_intake` then opens the first downloadable real
paired-scan intake using Open3D `DemoICPPointClouds`. The official archive is
verified by MD5/SHA-256 and parsed without an Open3D runtime dependency. After
checking the transformation-log direction, the evaluator forms 2 cm and 5 cm
reciprocal-NN candidates and applies the unchanged global, local-q95, and
tail-ratio observations to 32 deterministic 96-pair spatial patches. The full
observational stack passes 17/32 patches; the tail rule alone passes 20/32, so
the real coordinates exercise it nontrivially. This supports verified real
paired-scan **intake only**. Reciprocal NN does not prove physical point
identity, and there is no labeled reconstruction-harm endpoint, so real
correspondence, guard safety, reconstruction, and deployment remain
unsupported. See
[docs/OPEN3D_REAL_PAIR_INTAKE_PHASE31.md](docs/OPEN3D_REAL_PAIR_INTAKE_PHASE31.md).

`pftf_alpha.threedmatch_registration_guard` then evaluates the frozen observed
guard on the official 3DMatch `7-scenes-redkitchen` real-fragment registration
benchmark. All guard observations are materialized from fragment coordinates
and the external `3dmatch.log` predictions before `gt.log`/`gt.info` are read.
The official-label baseline has 383/531 correct predictions, 72.13% precision,
and 85.30% recall. The full global/local/tail route raises precision to 88.73%
at 2 cm and 77.78% at 5 cm, but retains only 16.45% and 10.97% of the correct
predictions. Tail evidence removes no additional incorrect prediction at 2 cm;
at 5 cm it removes four incorrect and seventeen correct predictions. Thus
`phase32_supported=false` and
`tail_sensitive_real_registration_supported=false`. The Phase-30 synthetic
result does not transfer to a useful real-registration guard. See
[docs/THREEDMATCH_REGISTRATION_GUARD_PHASE32.md](docs/THREEDMATCH_REGISTRATION_GUARD_PHASE32.md).

`pftf_alpha.threedmatch_transfer_audit` freezes the exact Phase-32 artifact and
repeats the unchanged route on the official SUN3D
`hotel_umd/maryland_hotel3` scene. Its baseline has 15/61 correct predictions
(24.59% precision, 57.69% recall). The full route accepts only 1/13 correct at
2 cm and 0/8 correct at 5 cm, so precision falls to 7.69% and 0%. Thus the
redkitchen failure transfers to a second independently labeled real scene:
`negative_transfer_reproduced=true`, `cross_scene_guard_supported=false`, and
deployment remains unsupported. The opened redkitchen and Maryland scenes must
not be reused as fresh validation after any redesign. See
[docs/THREEDMATCH_SECOND_SCENE_TRANSFER_PHASE33.md](docs/THREEDMATCH_SECOND_SCENE_TRANSFER_PHASE33.md).

`pftf_alpha.scene_relative_rotation_guard` then uses only the two opened scenes
to design a batch-relative spatial-transform observation. It converts each
external prediction's principal rotation angle to a within-scene empirical
midrank percentile and rejects the top 10%. On redkitchen it retains 346/383
correct predictions and rejects 16/148 incorrect predictions; on Maryland
hotel3 it retains 14/15 and rejects 5/46. Precision rises only from 72.13% to
72.38% and from 24.59% to 25.45%, so `phase34_design_supported=true` is a narrow
design result, not validation. The six remaining official scenes are frozen as
untouched Phase-35 validation, while `held_out_validation_supported=false` and
deployment remains unsupported. See
[docs/SCENE_RELATIVE_ROTATION_GUARD_PHASE34.md](docs/SCENE_RELATIVE_ROTATION_GUARD_PHASE34.md).

`pftf_alpha.scene_relative_rotation_validation` then opens all six frozen
3DMatch validation scenes as one panel without changing the p90 rule, midrank
ties, or gates. It verifies every evaluation ZIP exactly and materializes all
six blind decision sets from `3dmatch.log` before decoding any `gt.log` or
`gt.info`. All six scenes independently improve precision by 2.33--4.56
percentage points, retain 95.10--100% of correct predictions, and reject
13.04--23.21% of incorrect predictions. Thus
`held_out_validation_supported=true` and the narrow
`cross_scene_real_registration_supported=true` claim applies to the fixed
external 3DMatch predictor. The rule remains batch-relative, and real
correspondence identity, alpha-shape reconstruction, transfer to another
registration algorithm, and deployment remain unsupported. See
[docs/SCENE_RELATIVE_ROTATION_VALIDATION_PHASE35.md](docs/SCENE_RELATIVE_ROTATION_VALIDATION_PHASE35.md).

`pftf_alpha.independent_method_rotation_transfer` next applies the same frozen
p90 rule to the official toolbox's FPFH and Spin-Images predictions on four
ICL-NUIM synthetic scenes. All eight method-scene blocks pass independently:
precision rises by 1.95--2.63 percentage points, all 699 correct predictions are
retained, and each block rejects 12.20--13.19% of incorrect predictions. Thus
`independent_method_transfer_supported=true` and
`cross_benchmark_transfer_supported=true` for these fixed logs. FPFH and Spin
are distinct descriptors but share the toolbox's RANSAC registration pipeline;
the external generation was not rerun, so independent end-to-end pipeline
transfer, correspondence identity, alpha-shape reconstruction, and deployment
remain unsupported. See
[docs/INDEPENDENT_METHOD_TRANSFER_PHASE36.md](docs/INDEPENDENT_METHOD_TRANSFER_PHASE36.md).

`pftf_alpha.open3d_fgr_pipeline` and
`pftf_alpha.independent_pipeline_rotation_audit` close the precomputed-log gap
by locally generating all 2,341 nonconsecutive-pair predictions with Open3D
0.19.0 FPFH+FGR and then applying the unchanged p90 guard. Redkitchen retains
all 178 correct predictions and rejects 11.15% of incorrect predictions;
Maryland hotel3 retains all 8 correct predictions and rejects 10.13%. Precision
rises from 10.40% to 11.56% and from 1.27% to 1.41%, respectively, so
`independent_end_to_end_pipeline_transfer_supported=true` for this fixed
pipeline audit. A preliminary uncommitted matrix-direction mismatch was
invalidated and corrected from the official fragment2-to-fragment1 3DMatch
convention without changing parameters or gates. Because both scene labels had
already been opened, `fresh_label_blind_validation_supported=false`; Open3D
also supplies the algorithms, so independent algorithm implementation,
correspondence identity, alpha-shape reconstruction, and deployment remain
unsupported. See
[docs/INDEPENDENT_PIPELINE_TRANSFER_PHASE37.md](docs/INDEPENDENT_PIPELINE_TRANSFER_PHASE37.md).

Phase 38 preregisters a genuinely fresh external validation on the ETH
Mountain Plain real laser-scan sequence. The official 902,525,379-byte archive
matches its published MD5 and a locally frozen SHA-256. Only the ZIP directory
has been inspected: all 31 local Hokuyo scan member names and the separate
Leica-pose member name are present, but the pose member's values have not been
opened or decoded. `docs/FRESH_EXTERNAL_PROTOCOL_PHASE38.md` fixes all 435
nonconsecutive pairs, the unchanged Phase-37 Open3D pipeline, a strict
15-degree/0.30-m correctness rule, and the unchanged p90/gate requirements.
The full pre-label execution generated all 435 predictions after the
preregistration commit, and the hash-locked p90 decision artifact accepted 391
and rejected 44 before pose values were opened. The subsequent fixed audit is
negative: the unchanged cross-domain FPFH+FGR predictor has 0/435 registrations
within RRE < 15 degrees and RTE < 0.30 m. The guard rejects 10.11% of incorrect
predictions, but there are no correct predictions to retain and precision
stays at zero. Consequently `fresh_label_blind_validation_supported=false`
and `fresh_external_pipeline_transfer_supported=false`; see
[docs/FRESH_EXTERNAL_ROTATION_AUDIT_PHASE38.md](docs/FRESH_EXTERNAL_ROTATION_AUDIT_PHASE38.md).

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
