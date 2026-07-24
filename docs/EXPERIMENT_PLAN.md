# PFTF-alpha first benchmark plan

Status: B0-P2 frozen held-out smoke, exact-predicate readiness preflight,
fail-closed backend-handoff validation, and an evaluation-only exact-resampling
audit plus exact-B3 candidate-selection shadow complete; exact construction is
not applied to primary selection.

## Objective

Determine whether a PFTF-derived local SPD metric and confidence-aware fallback
improve alpha-shape reconstruction beyond established density-scaled and
normal-based anisotropic baselines.

## Minimal reproducible panel

| Family | Expected surface Betti | Variation | Required failure signal |
|---|---:|---|---|
| U/C concavity | (1, 1, 0) | opening width, density | convex-hull overfill |
| opposing sheets | (2, 0, 0) | gap / sample spacing | false bridge |
| torus | (1, 2, 1) | hole radius, noise | topology loss |
| disconnected parts | (2, 0, 2) | separation | false merge |
| sharp crease | (1, 0, 0) | angle, normal noise | webbing or hole |
| missing patch | (1, 0, 1) | missing area | unstable closure |

Every family uses fixed train/calibration/held-out seeds. Held-out conditions
include unseen density ratios, gap-to-spacing ratios, and noise levels.

## Methods

| ID | Method | Purpose |
|---|---|---|
| B0 | convex hull | over-smoothing reference |
| B1 | hand-picked global alpha | common manual workflow |
| B2 | exhaustive critical-alpha scan | global selection oracle |
| B3 | persistence + resampling stability | unlabeled automatic baseline |
| B4 | kNN density-scaled alpha | adaptive-scale baseline |
| B5 | normal/PCA anisotropic alpha | anisotropic prior-art baseline |
| P1 | PFTF local SPD metric | field contribution |
| P2 | P1 + confidence/B4 guard prototype | safety-path prototype |

## Current implementation status

The 2026-07-24 smoke runs all six synthetic families through B0-P2. It is a
pipeline and information-boundary check, not a promotion result:

- B2 exhaustively evaluates every top-simplex critical value and may use the
  dense synthetic reference as the declared global-selection oracle.
- B3 scans the full component/Euler topology sequence, then evaluates a fixed
  budget by input-point fit and 90% resampling stability. It does not use the
  dense reference or expected component count during selection. The terminal
  convex-hull plateau is not treated as a finite persistence interval,
  preventing the largest sliver alpha from winning automatically.
- B4 divides each Euclidean tetrahedral circumradius by the geometric mean kNN
  spacing of its vertices. Thresholded top cells receive complete downward
  closure before boundary extraction.
- B5 estimates a regularized local-PCA SPD metric, combines it with the B4
  density normalization, and applies a planarity-weighted normal penalty. It
  uses the same complete-closure extraction.
- B4/B5 currently score cells of one fixed Euclidean Delaunay triangulation.
  They are scale/rotation-invariant research surrogates, not exact anisotropic
  Delaunay or CGAL alpha-shape constructions.
- P1 forms a signed trace-free relation tensor from directed receiver/source
  kNN scale contrast and receiver-direction imbalance. It maps that relation
  through bounded log-eigenvalues to an SPD metric, then blends uncertain point
  metrics toward the density-scaled identity.
- P1 confidence combines relation strength, neighbor-distance regularity, and
  reciprocal-kNN support.
- P2 accepts an explicit threshold or freezes one without reference geometry.
  The automatic path pools P1 simplex confidence on the calibration panel and
  places the threshold between order statistics for a predeclared target
  fallback fraction, before P2 multiplier selection.
- For each low-confidence top cell P2 uses `max(P1 score, B4 score)`, so the cell
  must pass both tests under one frozen multiplier. Confident cells retain the
  P1 score. This is a conservative fixed-complex guard, not exact CGAL fallback.
- Schema 11 adds an evaluation-only geometric bridge-risk probe. The risk path
  fits kNN PCA normals and planarity, then routes by the largest eigenvalue of
  the mean unoriented normal outer product. A coherent field uses the mean
  planar parallel-normal edge signal; other fields use the second-longest edge
  normalized by the geometric mean endpoint kNN scale. Dividing by frozen
  route-specific thresholds makes `risk > 1` the diagnostic flag. The risk
  function reads neither references nor component labels; labels are applied
  only afterward for AUC, recall, and FPR.
- Schema 12 adds a zero-strength-exact multiplicative penalty
  `score * (1 + strength * max(risk - 1, 0))` and an optional
  `--evaluate-bridge-penalty` calibration curve. It records reference/label
  endpoints only for evaluation gates, never freezes a strength, and never
  modifies requested B0-P2 selection.
- Schema 13 adds `--evaluate-boundary-bridges` on a frozen P2 output. It scores
  each output boundary edge with the existing route-specific observed-geometry
  signal and assigns each boundary face its maximum incident-edge risk. The
  selected-cell dual graph is audited for articulation cells and bridge edges,
  but that weak cut signal is not fused into the geometric risk. Component
  labels enter only after localization for AUC, recall, and FPR; the diagnostic
  cannot change P2 selection.
- Schema 14 adds `--evaluate-boundary-intervention`, a calibration-only layered
  ablation. Each round removes every unique owner of a current boundary face
  with `risk > 1` and recomputes the boundary. Labels and reference geometry are
  absent from the intervention order and appear only in the objective, geometry,
  component, Betti, and strict labeled edge/face bridge promotion gate. No depth
  is frozen or applied to held-out P2 selection.
- Schema 15 adds `--evaluate-boundary-region-cuts`, a calibration-only structural
  audit with fixed `baseline`, `largest_risk_region`, and `safe_backbone_cut`
  strategies. Risk regions join flagged faces only through flagged edges. The
  safe-backbone strategy removes flagged edges, labels safe-edge vertex
  components, and retains only flagged edges crossing those components. Neither
  references nor component labels enter either candidate construction.
- Schema 16 adds `--evaluate-exact-predicates`, a requested-split readiness
  audit. It converts every finite binary64 coordinate to its exact rational
  value and evaluates exact 3D orientation and interior-facet in-sphere signs on
  supplied SciPy/Qhull connectivity. It changes neither construction nor
  selection, is not an exact alpha-complex or CGAL certificate, and keeps
  promotion blocked until an exact construction backend is integrated.
- Schema 17 adds `--evaluate-exact-construction` and an optional explicit
  backend executable. Canonical requests use exact rational coordinate pairs;
  responses must echo the request SHA-256 and attest backend name, version,
  kernel, and exact construction. The host independently checks point coverage,
  face incidence, convex-hull support, exact volume coverage, orientation, and
  in-sphere predicates. Missing, failed, or rejected backends fail closed, and
  validated cells are not yet used by benchmark selection.
- Schema 18 adds `--evaluate-exact-connectivity-shadow`. Only cells retained by
  the schema-17 host validator are eligible, and they are passed through
  `AlphaFiltration.from_top_simplices`, which enforces integer shape, bounds,
  unique vertices/cells, and complete point coverage. The shadow reruns the
  requested methods and compares all non-runtime outputs plus bridge risk under
  relative `1e-12` and absolute `1e-15` float tolerances. It cannot replace
  primary case reports or selection. Filtration values remain floating-point,
  so this is neither an exact alpha complex nor a deployed fallback.
- Schema 19 adds `--exact-python-backend`, a built-in small-panel exact
  Euclidean Delaunay constructor. It converts exact rational protocol
  coordinates to one integer scale, enumerates all four-point candidates, and
  retains exact empty circumspheres without SciPy/Qhull candidate connectivity.
  The implementation caps inputs at 64 points, skips coplanar candidates, and
  fails closed on exact cospherical ambiguity rather than claiming symbolic
  perturbation. Returned cells still pass the independent schema-17 host
  validator and enter only the schema-18 shadow. Alpha filtration values remain
  floating-point; primary case reports and selection remain immutable. This is
  not CGAL, an exact alpha complex, a spatially varying anisotropic
  triangulation, or a deployed fallback.
- Schema 20 adds `--evaluate-exact-filtration-values`. On every simplex of
  host-validated 3D connectivity it solves the intrinsic affine-hull Gram
  system with exact rational arithmetic, checks Gabriel emptiness against every
  input point, and assigns non-Gabriel simplices the minimum exact immediate-
  coface value. The audit compares correctly rounded values, exact Gabriel
  flags, critical counts, exact ties, and adjacent exact ordering with the
  floating filtration, and hashes the canonical exact records. A nonempty top
  sphere, invalid arithmetic, or missing accepted connectivity fails closed.
  Exact values remain audit-only: primary case reports, selection, and the
  schema-18 connectivity shadow are unchanged. This is not a deployed exact
  alpha evaluation or CGAL comparison.
- P1 uses the same fixed Euclidean Delaunay top-cell closure as B4/B5, so it is
  not an exact spatially varying anisotropic triangulation.
- `--calibrate-adaptive` first freezes P2 confidence without dense references,
  then pools the six calibration cases to minimize one mean geometry/topology/
  complexity objective per method and freezes one multiplier before evaluating
  the requested split. Its current topology term is component error only. The
  declared surface Betti targets and their errors are evaluation-only and do
  not enter adaptive multiplier or reference-free P2 confidence selection.
- The frozen held-out smoke used 48 input points, 512 reference points, 96
  surface samples, 12 requested calibration candidates, and seed 20260724. It
  selected B4 multiplier 1.1977526569935681, B5 multiplier 2.80293354289327,
  P1 multiplier 1.2076985596095746, and P2 multiplier
  1.2217738660639386.
- All 24 B4/B5/P1/P2 held-out records used the same corresponding frozen
  multiplier, declared `uses_reference_for_selection=false`, and evaluated one
  candidate.
- Across the six smoke cases, mean F-score was 0.15198 for B4, 0.15616 for B5,
  0.17634 for P1, and 0.16777 for P2. Mean normalized squared Chamfer was
  0.00919, 0.00957, 0.00966, and 0.01037 respectively; component-error sums
  were tied at 2. Thus this is not evidence of an overall P1 or P2 win.
- GF(2) surface Betti-error sums were B4 20, B5 27, P1 25, and P2 25. P1/P2
  reconstructed the torus as `(1,2,1)` exactly, but missed the other five
  declared targets; the missing-patch full-sphere target alone had error 14.
  The topology endpoint therefore exposes failures hidden by component count.
- Synthetic input vertices now retain evaluation-only ground-truth surface
  component labels. P2 produced 39 cross-sheet edges/39 mixed faces on opposing
  sheets and 18 cross-part edges/19 mixed faces on disconnected parts, so both
  multi-component cases remain labeled false-bridge failures.
- B2 reported the requested two components on opposing sheets while retaining
  46 cross-sheet edges/47 mixed faces. This is a concrete false negative of the
  old component-count proxy. The labeled endpoint did not change any selection
  parameter or prior endpoint.
- With frozen probe thresholds `(0.9, 0.02, 1.8)`, opposing sheets reached AUC
  0.99490, recall 0.87402, and FPR 0.02817; disconnected parts reached AUC 1.0,
  recall 1.0, and FPR 0.0. The four single-component cases still flagged
  6.25-32.98% of their cells. All 48 pre-existing non-runtime B0-P2 results
  matched the schema-10 artifact exactly, so the probe is informative but is
  not enabled as a hard guard.
- The frozen calibration-only strength curve
  `[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]` found no promotion-eligible
  point. Strength 0.01 improved mean geometry/objective but changed labeled
  false-bridge edges from 34 to 35. Strength 0.8 reduced edges to 25 but
  regressed geometry and Betti error. Selected flagged-cell count and selected
  mean risk each had Spearman correlation -0.26998 with output false-bridge
  edges. The schema-12 run again preserved all 48 pre-existing non-runtime
  B0-P2 results.
- The frozen schema-13 held-out localization inspected 550 P2 boundary faces
  and 807 boundary edges. At the unchanged `risk > 1` rule, face AUC/recall/FPR
  were 0.99523/0.96552/0.02439 and edge AUC/recall/FPR were
  0.98905/0.89474/0.00933. It found 19/19 mixed disconnected-parts faces and
  37/39 mixed opposing-sheets faces.
- The separate dual-bottleneck face score reached pooled AUC 0.58274. In
  particular, opposing sheets had no selected-dual articulation or bridge edge,
  so graph cuts alone cannot localize that failure mode. The dual signal remains
  audit-only and is not combined with the geometric boundary risk.
- All 48 pre-existing non-runtime B0-P2 results and the schema-11 cell probe
  again matched the schema-12 artifact.
- P1's maximum observed metric condition number was 1.64114 under the bound of
  9.0. Median point confidence ranged from 0.34275 to 0.43049 across cases, and
  numeric fallback fraction was zero.
- The reference-free target fallback fraction 0.25 selected threshold
  0.26868716232131434 from 1,099 calibration cells and achieved 0.25842.
  Held-out fallback ranged from 0.07487 to 0.48990 of all cells and 0.11009 to
  0.42308 of selected cells, so the nearly-all-fallback failure was removed.
- Score-level and selected-set B4 guard violations were zero. Every selected
  top-cell set reported complete downward closure and no face incidence above
  two. Nevertheless, the labeled endpoint finds false bridges in both P2
  multi-component cases. The score guard is relative to B4 and is not a
  topology-safety certificate; general exact fallback remains pending.
- Every result records the selected generic parameter, complete candidate
  range, selection objective, reference-use flag, runtime, surface endpoints,
  and method diagnostics. Alpha-specific fields remain populated for B1-B3.
- GF(2) surface Betti numbers and L1 target error are implemented for the
  triangular output complex, with Euler-Poincare consistency checked in tests.
- Labeled cross-component edge/face counts are implemented for the synthetic
  partition and are evaluation-only. They are direct discrete witnesses for
  declared multi-component surfaces, not general handle or CGAL certificates.
- Exact semantic hole/cavity localization, general exact false-safe,
  normal-consistency, volume, and memory endpoints remain pending. The existing
  `false_bridges`/`false_splits` fields remain component-count proxies.
- Surface-sampling sliver tetrahedra can produce very large critical values;
  these remain visible in the candidate range and are not silently removed.

The P2 conservative fallback path has run, but the project remains **ToDo**
because the selective fallback smoke does not improve P1 geometry or topology,
retains labeled false bridges, and lacks general exact fallback certification.

The per-cell soft penalty and the boundary-owner peeling intervention are both
rejected, while the evaluation-only boundary localizer remains diagnostic only.
Schema 14 tested depths `[0, 1, 2, 4]` on the calibration panel. One round
improved objective/geometry from 0.41890/0.20157 to 0.41450/0.19798 but worsened
labeled bridge edges/faces from 34/35 to 50/53. Four rounds removed 121 of 579
selected cells and reduced bridges to 11/12, but objective/geometry regressed to
0.45821/0.24757. No positive depth passed every promotion gate, so no depth was
frozen or deployed. The 48 held-out non-runtime B0-P2 results, schema-11 cell
probe, and schema-13 boundary localization remained unchanged.
Schema 15 evaluated the fixed `largest_risk_region` and `safe_backbone_cut`
strategies under the same gates. The calibration panel contained 13
flagged-edge-connected face regions, and the largest contained 25 faces. The
largest-region strategy removed 28 cells across five cases and improved Betti
error from 25 to 22, but regressed objective/geometry from 0.41018/0.19247 to
0.41441/0.19727 and increased labeled bridge edges/faces from 34/35 to 43/45.
Removing every flagged edge left seven safe-edge vertex components across the
panel, but no flagged edge connected distinct safe components. Consequently the
safe-backbone strategy generated no candidate and was exactly equal to the
baseline. No strategy passed all promotion gates or was frozen.

All 48 held-out non-runtime B0-P2 results, the schema-11 cell probe, and the
schema-13 boundary localization again remained unchanged. Per-cell penalties,
layered owner peeling, connected-region removal, and simple safe-backbone cuts
have now all failed the declared gates. The next safe implementation path is an
exact-construction fallback rather than another heuristic removal rule; it must
remain separately audited before promotion.

Schema 16 audited the frozen held-out split with exact rational predicates over
the binary64 inputs: 288 points, 1,131 supplied Qhull tetrahedra, and 2,094
interior facets. It found zero degenerate tetrahedra, exact cospherical facets,
local-Delaunay violations, non-manifold facets, or float/exact sign
disagreements. All 48 prior non-runtime B0-P2 results and the bridge probe were
exactly unchanged. This preflight certifies only predicate consistency of the
supplied connectivity on this dataset; it does not construct an exact
triangulation or alpha complex. No exact backend is integrated, so promotion
remains blocked.

Schema 17 froze the optional backend protocol without providing a backend.
Consequently `backend_requested=false`, zero cases were accepted, and
selection remained unchanged under `no_exact_construction_backend` blocking.
Protocol fixtures accepted one complete bipyramid and rejected request-hash,
missing-point, and exact local-Delaunay failures. The host structural validator
also accepted all six frozen-panel Qhull connectivities when injected as test
fixtures, including exact hull-volume equality; that is validator coverage, not
evidence that Qhull constructed them exactly. The schema-16 predicate result,
config, bridge probe, and all 48 non-runtime results remained exactly unchanged.

Schema 18 connected accepted backend cells to an evaluation-only filtration
application path. A Qhull-connectivity fixture passes the exact host validator,
is reconstructed through `AlphaFiltration.from_top_simplices`, and reproduces
the tested non-runtime method output within the declared tolerances. The fixture
attestation is test-only and does not establish exact Qhull construction. The
frozen held-out run still supplied no backend: it recorded zero accepted cases,
zero shadow runs, zero shadow differences, and six null shadow reports. Its
schema-17 config, predicate audit, and all 48 primary non-runtime outputs were
exactly unchanged. The shadow uses exact-validated connectivity but floating
circumsphere filtration values; a real exact backend must still be supplied and
evaluated before any separate deployment or promotion decision.

Schema 19 supplied the built-in exact backend on the frozen 48-point held-out
panel. The schema-17 host accepted all six responses and recorded exact-backend
tetrahedron counts 208, 198, 188, 187, 209, and 141 for `u_concavity`,
`opposing_sheets`, `torus`, `disconnected_parts`, `sharp_crease`, and
`missing_patch`, respectively. Every backend connectivity matched the primary
connectivity, all six shadows ran, and zero cases differed in any declared
non-runtime B0-P2 output or bridge-risk probe. The complete primary non-runtime
case payload remained exactly equal to schema 18. The artifact is
`benchmark-out/smoke_b0_p2_exact_python_backend_shadow_held_out.json`
(438,966 bytes; SHA-256
`5ce3863eae74fcdec439bba26c142006823abdbb84bf940ca4fce50aef15f5db`).
The result closes the missing-backend readiness gap for this small,
non-cospherical panel only. Promotion remains blocked because exact connectivity
is still shadow-only and filtration values are floating-point; deployment and a
CGAL/reference-stack comparison remain separate gates.


Schema 20 audited all 5,430 simplices in the frozen six-case held-out exact
connectivities. Of these, 953 floating values were bitwise equal to the
correctly rounded exact rational value and 4,477 differed. The panel maxima were
absolute error `3.0440274899232807e-4`, relative error
`1.70727232787347e-9`, and 11,128,782 ULP. The large ULP count is reported
without treating it as a failure because the declared combinatorial checks are
separate: exact versus floating Gabriel disagreements, exact-tie splits,
exact-versus-rounded critical-count mismatches, exact-versus-floating
critical-count mismatches, and adjacent exact-order violations were all zero.
All six schema-19 connectivity shadows again produced zero non-runtime output
differences, while the complete primary non-runtime B0-P2 payload remained
exactly equal to schema 19. The fixed artifact is
`benchmark-out/smoke_b0_p2_exact_filtration_audit_held_out.json`
(448,635 bytes; SHA-256
`7c9aa989d122b9b2ed736977a32267e3b2c76cb66a51395747500e32654fc54a`).
This removes the unmeasured filtration-value gap on the frozen panel, but not
the deployment gate: exact values are not used for thresholds or selection,
and CGAL/reference-stack comparison remains pending.

### Schema-21 exact-rounded value-shadow checkpoint (2026-07-24)

Schema 21 requires the complete exact evaluation chain: accepted exact backend
connectivity, an audited exact-filtration digest, and a completed floating-value
shadow on the same connectivity. `exact_rounded_filtration` recomputes every
exact rational record, verifies the audit SHA-256 and simplex count, then builds
an evaluation-only `AlphaFiltration` from correctly rounded binary64 values.
Any missing prerequisite, arithmetic failure, or digest mismatch produces no
shadow report. Primary cases and selection are immutable.

All six frozen 48-point held-out cases ran through the B0-P2 exact-value shadow.
Against the same-connectivity floating shadow, every case changed B2 and B3
non-runtime reports. Candidate minima/maxima differed in all six. Under relative
`1e-12` and absolute `1e-15` comparison tolerances, selected-alpha fields changed
for B2 on `u_concavity` and `sharp_crease`; B3 objective/stability fields changed
on `u_concavity` and `disconnected_parts`. Surface/topology endpoint payloads and
the bridge-risk probe changed in zero cases. The primary non-runtime payload,
schema-20 exact audit, and floating-connectivity shadow were byte-for-structure
equal to schema 20 after excluding runtime fields.

This result narrows the schema-20 conclusion: matching critical counts, ties, and
ordering is insufficient to guarantee identical candidate bookkeeping or
objectives. It does not show a frozen endpoint improvement or regression.
Exact-rational values are rounded to binary64 before use; thresholds, objectives,
resampling, and surface evaluation remain floating-point. Deployment into
selection, end-to-end exact alpha-complex evaluation, and a CGAL/reference-stack
comparison remain blocked. Artifact:
`benchmark-out/smoke_b0_p2_exact_value_shadow_held_out.json` (614,286 bytes;
SHA-256 `ea0d83f2b0a63f83591b8faeb32a9e710f2799fc970aa79c0a17889814db8e0d`).

### Schema-22 exact critical-index checkpoint (2026-07-24)

Schema 22 adds `--evaluate-exact-critical-index-audit`, gated on the complete
schema-21 exact-construction, filtration-audit, connectivity-shadow, and
value-shadow chain and on B2 or B3 being requested. It verifies critical counts
and ordered top-simplex birth-group identities across floating, exact-rounded,
and exact-rational views. It then compares selected critical ranks, full-complex
and regularized-boundary digests, objective/endpoints, and the B3 signature,
candidate-index, plateau-persistence, and objective-term paths. Missing
prerequisites fail closed, and primary selection remains immutable.

All six frozen 48-point held-out cases had identical critical counts, birth-group
identities, B3 topology-signature sequences, and B3 budgeted candidate-index
sequences. Across B2 and B3 there were zero selected-index, selected-complex, and
selected-boundary mismatches. B2 objective/endpoints matched. Every B3
persistence array had bitwise differences, while selected persistence crossed
tolerance only on `disconnected_parts`. With the selected complex held fixed,
B3 objective values differed on `u_concavity` and `disconnected_parts`: topology
changed in one case and resampling stability in two.

This isolates the frozen schema-21 B2/B3 report differences to numeric-radius
bookkeeping, plateau normalization, and floating resampling thresholds rather
than a combinatorial selection change. It is not a general invariant: the
focused 16-point test panel can select a different critical index and complex.
The primary non-runtime cases, exact-filtration audit, connectivity shadow, and
value shadow remained structure-identical to schema 21 after runtime fields were
excluded. Exact resampling, deployment, CGAL/reference-stack comparison, and an
end-to-end exact alpha-complex evaluation remain pending. Artifact:
`benchmark-out/smoke_b0_p2_exact_critical_index_audit_held_out.json` (642,885
bytes; SHA-256
`5a104add183795ba179eb3b943cc00f9f8135c80c1b6e7481c8c8b3e715a0894`).

### Schema-23 exact-selected-threshold resampling checkpoint (2026-07-24)

Schema 23 adds `--evaluate-exact-resampling-threshold-audit`, gated on the
complete schema-22 chain and B3 shared selected-index, full-complex, boundary,
and candidate-position identity. The controlled variables are the selected full
complex, deterministic full-surface samples, resampled point subsets, floating
resampled connectivity and filtration values, and all sampling seeds. The only
treatment is which selected binary64 alpha threshold is applied to each shared
floating resample. Missing or nonidentical prerequisites fail closed.

All six frozen 48-point held-out cases were audited. The two full-surface sample
hashes matched per case, and all six selected alpha values differed. Of 12 shared
resample evaluations, only one repeat in `u_concavity` and one in
`disconnected_parts` changed the full complex and regularized boundary. Exactly
those two cases changed mean stability, by `0.00026695180449573357` and
`0.010215245512115334`. The remaining four changed neither resampled boundary
nor stability.

Both reported B3 stability values were reproduced in every case. Reproduction
failures, stability differences without a resampled-boundary change, and
boundary changes without a stability difference were all zero. Primary cases,
the exact-filtration audit, connectivity shadow, value shadow, and critical-index
audit remained structure-identical to schema 22 after runtime fields were
excluded.

This closes the selected-threshold explanation for the frozen schema-22
stability differences. It does not construct exact resampled connectivity or
exact resampled filtration values; the resampling path remains SciPy/Qhull and
binary64. Deployment, exact resampling, end-to-end exact alpha evaluation, and
CGAL/reference-stack comparison remain pending. Artifact:
`benchmark-out/smoke_b0_p2_exact_resampling_threshold_audit_held_out.json`
(667,995 bytes; SHA-256
`0aa9302181cabd48c4cc99a6a7a3b52e458b133813b76236a95aab52df9fdff9`).

### Schema-24 exact-resampling filtration checkpoint (2026-07-24)

Schema 24 adds `--evaluate-exact-resampling-filtration-audit`, gated on the
complete schema-23 chain and the same reusable exact backend. It regenerates the
deterministic B3 resampled point subsets, independently constructs and
host-validates exact connectivity for each subset, computes exact rational
simplex filtration values, and evaluates their correctly rounded binary64 view
at the schema-23 exact-selected B3 threshold. The exact full-surface samples,
subsets, thresholds, and sampling seeds are held fixed. Missing schema-23
identity or any rejected resample fails the entire case closed.

All six cases and all 12 resamples were audited. Exact resampled connectivity
matched SciPy/Qhull in every repeat. All 12 resamples nevertheless contained
floating-versus-correctly-rounded filtration-value differences: 7,723 values in
total, with maximum difference 32,653,561 ULP. Six repeats across four cases
changed the selected full complex and regularized boundary. The same six
repeats changed stability, so there were no stability changes without boundary
changes and no boundary changes without stability changes.

Relative to schema 23's exact-threshold-on-floating-resampling value, mean
stability changed for `u_concavity` by `0.00026695180449573357`,
`opposing_sheets` by `0.00030055016177662954`, `torus` by
`0.00012341684299446362`, and `disconnected_parts` by
`0.010775278523207324`. `sharp_crease` and `missing_patch` were unchanged.
The primary cases, exact-predicate audit, exact construction, exact-filtration
audit, connectivity shadow, value shadow, critical-index audit, and schema-23
threshold audit all remained structure-identical after runtime fields were
excluded.

This closes exact connectivity and exact-filtration construction for the fixed
small-panel B3 resamples as an audit. It does not deploy exact resampling into
selection, retain rational arithmetic through thresholds/objectives/surface
evaluation, establish a general false-safe certificate, construct a spatially
varying anisotropic complex, or provide CGAL/reference-stack parity. Artifact:
`benchmark-out/smoke_b0_p2_exact_resampling_filtration_audit_held_out.json`
(698,999 bytes; SHA-256
`190ccdd545e6a4fde31af53ed5706015e202f296684958ce5d2d99fab9d138b9`).

### Schema-25 exact-B3 selection shadow checkpoint (2026-07-24)

Schema 25 adds `--evaluate-exact-b3-selection-shadow`, gated on the complete
schema-24 chain. It reuses the exact full filtration and the exact resample
filtrations retained by schema 24, reproduces the exact-value B3 reference with
floating resamples, and then re-evaluates every budgeted candidate with exact
resamples. Candidate indices, full-filtration terms, surface sample seeds, and
endpoint sample seeds are controlled; only the resampling filtration source
changes. Any missing schema-24 context or reference-reproduction failure fails
the case closed.

All six cases and all 60 candidates were evaluated. Candidate stability and
total objective changed for 28 candidates in five cases: 4 in `u_concavity`, 5
in `opposing_sheets`, 4 in `torus`, 7 in `disconnected_parts`, and 8 in
`missing_patch`. `sharp_crease` had no candidate differences. The selected
objective changed in `u_concavity`, `opposing_sheets`, `torus`, and
`disconnected_parts`, but all six cases retained the same selected critical
index, alpha, full complex, regularized boundary, and endpoint metrics.

The primary cases and every schema-24 section remained structure-identical
after runtime fields were excluded. On this frozen panel, the exact-resampling
changes therefore perturb objective values without changing the B3 argmin. The
shadow is not deployed; exact rationals are rounded to binary64 and objective
aggregation and surface evaluation remain floating-point. This is not a
general false-safe certificate, anisotropic exact construction, or
CGAL/reference-stack parity. Artifact:
`benchmark-out/smoke_b0_p2_exact_b3_selection_shadow_held_out.json` (765,053
bytes; SHA-256
`5d153c584c6400f3ca5ea8b4a62bbf6dc2069a43ecf3c784d0b8b146d22a61dc`).

## Primary endpoints

1. General false bridges per shape (labeled synthetic witnesses now exist).
2. Betti/component error.
3. Surface F-score and Hausdorff distance.
4. Stability under 10% subsampling and calibrated noise.
5. Non-manifold and singular-face counts.
6. Runtime, memory, and fallback rate.

## Required ablations

- PFTF relation field versus local PCA covariance.
- Isotropic versus anisotropic metric.
- Density only, normal only, relation only, combined.
- Hard gate versus soft calibration followed by hard exact evaluation.
- Confidence fallback on/off.

## Promotion rule

The initial B0-P2 end-to-end run gate is satisfied. The project remains
**ToDo** until a higher-fidelity frozen held-out evaluation shows value beyond
both B4 and B5 on declared geometry and topology endpoints, and an exact or
validated fallback demonstrates no unreported
false-safe cases. A mean F-score gain in this small smoke alone is insufficient
for promotion.
