# P3 candidate — resampling-consensus bridge excision (predeclared design)

Status: **predeclared design — core signal FALSIFIED by pre-flight probe; not built.**
Date: 2026-07-24 (Asia/Seoul).

> **Outcome (2026-07-24).** The resampling-consensus gate at the heart of this
> design was tested cheaply before implementation (`bridge_persistence_probe.py`)
> and **falsified**: per-cell resampling persistence separates labeled bridge
> from non-bridge boundary cells at AUC ≈ 0.50 (bridge tetrahedra re-form under
> subsampling as reliably as real-surface tets, because Delaunay connectivity is
> locally stable). Two follow-up calibration probes then showed multiplier
> backoff and a global reference-free fail-closed router also cannot pass the
> both-B4-and-B5 non-regression bar (backoff fixes large-gap `disconnected_parts`
> but not thin-gap `opposing_sheets`, and the reference-free signals that reveal
> the good backoff also silently favour destroying legitimate topology). This
> corroborated the handoff thesis that the promotion bottleneck is G4, and work
> pivoted to the deployed G4 fail-closed fallback
> (see `docs/G4_FAIL_CLOSED_DEPLOYMENT_DESIGN.md`). The design below is retained
> as the predeclaration it was, not as a live plan.

This document is the P1 deliverable required by the 2026-07-24 handoff: define
the new output-level topology / false-bridge intervention rule and argue why it
is structurally different from the already-rejected schemas 12, 14, and 15,
**before** any held-out outcome is read. All candidate strategies and strengths
are frozen here as a predeclaration. Implementation and the calibration-only
ablation follow in a separate step; the four-profile / three-seed G5 held-out
panel is only touched after a single configuration is frozen on calibration.

## 1. Problem restatement

P2 (`pftf_confidence_fallback_filtration`, `adaptive.py:1677`) is a *per-cell
score guard*: for a low-confidence tetrahedron it uses `max(P1 score, B4 score)`.
It has no representation of the output surface's global component structure, so
it inherits B4-like false bridges. The G5 preflight confirmed this: P2 keeps the
B4 score guard (zero violations) yet fails the strict casewise B4/B5 endpoint
envelope in all four profiles, with labeled false-bridge-edge excess of 86/55/80/4.

Two facts from the existing evidence frame the opportunity:

1. **Localization is already solved.** The schema-13 boundary localizer
   (`boundary_bridge_localization`, `adaptive.py:804`) scores each output
   boundary face by the reference/label-free geometric risk and reaches face
   AUC ≈ 0.995 / edge AUC ≈ 0.989. Finding *where* a bridge is, on the output
   boundary, is not the bottleneck.
2. **Intervention is the bottleneck, and it is a global-structure problem.** The
   selected-cell dual graph has, for opposing sheets, *no* articulation cell or
   bridge edge (EXPERIMENT_PLAN schema-13 note), because a false bridge is a
   *thick* web of parallel tetrahedra. Deleting flagged cells locally either
   exposes new one-owner faces (schema-14 shallow peeling worsened bridges to
   50/53) or over-removes and regresses geometry (schema-14 deep peeling,
   schema-15 largest-region).

The missing ingredient shared by every rejected attempt: a **label-free way to
decide which flagged structure is genuinely spurious**, and a removal rule tied
to **actual component separation** rather than to per-cell risk thresholds.

## 2. The new signal: per-cell resampling consensus

A genuine surface tetrahedron is reconstructed consistently when the point cloud
is resampled; a false bridge depends on a few points that happen to align across
a gap and disappears when they are resampled out. B3 already uses resampling,
but only as a *scalar global* stability term (`_stability_loss`, `baselines.py:638`,
mean Chamfer between full and resampled surface samples). It never asks, per
cell, "does this tetrahedron survive resampling?" No such per-cell / per-face
persistence exists anywhere in the codebase today.

We define, at the frozen P2 multiplier `m*`:

- Draw `R` deterministic resamples that keep a fraction `keep` of the points
  (reusing the deterministic draw of `_resampled_point_sets`, extended to also
  return the retained base indices, which it currently discards at
  `baselines.py:677`).
- For each resample `r`, rebuild the **same adaptive method** on the subset
  (`AlphaFiltration.from_points(subset)` → the P2 construction at the frozen
  confidence threshold) and select cells at the same frozen `m*`.
- A base tetrahedron `c` (4 base-point indices) is *evaluable* in resample `r`
  iff all 4 of its points were retained in `r`. It is *supported* iff it is also
  a selected cell of `r`'s reconstruction (mapped back through the retained
  indices).
- `persistence(c) = supported_count / evaluable_count`, defined only when
  `evaluable_count ≥ min_evaluations`. **Undefined persistence fails closed: the
  cell is never eligible for removal.**

Genuine surface cells → high persistence; spurious bridge fill → low
persistence. This is a reference-free, label-free, observed-geometry-only signal.

## 3. The intervention

At `m*`, per case:

1. Build base P2 `AdaptiveCellFiltration`; selected set `S = {score ≤ m*}`.
2. Localize with `boundary_bridge_localization` (reuse). Flagged faces
   `F = {boundary_face_risk > 1}`; candidate owner cells `O = unique owners of F`.
3. Compute `persistence(c)` for every `c ∈ O` (§2).
4. **Eligibility** = flagged **and** `persistence(c) ≤ τ`. The resampling gate is
   what rejects the geometric localizer's false positives: single-component
   families flag 6–33 % of cells, and those legitimate cells must have *high*
   persistence and be spared.
5. Apply one predeclared **strategy** to the eligible set, then recompute the
   regularized boundary by raising `scores[removed] = nextafter(m*, ∞)` and
   re-reading `surface_at(m*)` (reuse; preserves downward closure exactly as
   schemas 14/15 do).

### Predeclared strategies

- **`S0 baseline`** — no removal (control; must reproduce P2 exactly).
- **`S1 gated_peel`** — remove *all* eligible cells, recompute boundary once.
  Isolates the value of the resampling gate alone versus schema-14 peeling.
- **`S2 separation_cut`** — remove only a *minimal* eligible-cell subset whose
  removal **increases the output surface's connected-component count**, and keep
  the removal only if **each resulting side is itself resampling-persistent**
  (present as a connected piece in ≥ `rho` fraction of resamples). If no eligible
  subset raises the component count into persistent sides, **no-op** (fail
  closed to baseline). This is the global-separation variant that targets the
  thick-bridge failure the dual graph cannot localize.

### Predeclared sweep (frozen before any held-out read)

- Resample count `R = 8`, keep fraction `keep = 0.90`, `min_evaluations = 3`.
- Persistence threshold `τ ∈ {0.25, 0.50, 0.75}`.
- Side-persistence `rho ∈ {0.50, 0.75}` (S2 only).
- Strategy `∈ {S0, S1, S2}`.

### Predeclared freeze rule (calibration only)

On the six-case calibration panel, compute for every (strategy, τ, rho) the
labeled endpoints already used by the schema-14/15 gate. Select the single
configuration that:

1. **strictly reduces labeled false-bridge edges** versus the S0/P2 baseline, and
2. **does not regress against either B4 or B5** on the predeclared endpoints
   (normalized geometry loss, component error, surface Betti error, labeled
   bridge edges *and* faces, nonmanifold edges), and
3. does not increase runtime beyond a predeclared factor.

Ties broken by (largest bridge-edge reduction, then smallest geometry
regression, then simplest strategy S0 ≺ S1 ≺ S2). **If no configuration passes,
freeze `S0` and report that P1 produced no promotable candidate** — the same
honest outcome schemas 14 and 15 reached. Only the frozen configuration is then
run, unchanged, on the four-profile / three-seed G5 panel, with the existing
strict casewise B4/B5 envelope as the pass/fail test.

## 4. Information-boundary discipline

The eligibility and intervention order use **only** observed geometry (the
geometric risk), resampling consensus (observed points), and frozen calibration
constants. Synthetic component labels, expected Betti values, and dense
references never enter the risk, the persistence, the strategy, or the removal
order. Labels/references enter **only** the calibration freeze gate and the final
evaluation — identical to the schema-14/15 precedent
(`benchmark.py:862 "labeled_false_bridge_role": "evaluation_only"`). A run must
re-assert `promotion_supported = false`: passing this synthetic ablation is
necessary, not sufficient, for the promotion rule, which still also requires the
G4 exact / validated fail-closed fallback.

## 5. Why this is structurally different from schemas 12, 14, 15

- **vs Schema 12 (per-cell soft penalty).** Schema 12 multiplies the *selection
  score* by `(1 + strength·max(risk−1,0))`, changing which cells cross the
  threshold — purely local, single-snapshot, no output-structure awareness. Its
  risk↔bridge Spearman correlation was only ≈ −0.27. P3 never modifies selection
  scores; it operates on the *frozen* selected output and adds a cross-resample
  consensus vote plus a component-separation target that schema 12 has no analog
  of.

- **vs Schema 14 (fixed-depth owner peeling).** Schema 14 removes *every* flagged
  owner at a fixed depth `∈ {0,1,2,4}`; shallow depth *exposed* more bridges,
  deep depth regressed geometry, and no depth passed the gate. P3 (a) filters
  flagged owners by resampling persistence, so it never removes a persistent
  legitimate cell, and (b) in `S2` removes only a minimal *separating* subset
  judged by a global component-count objective, not a depth sweep.

- **vs Schema 15 (region cuts / safe backbone).** Schema 15's region is flagged
  faces joined through flagged edges (largest by risk mass), or a static
  safe-vertex-component backbone cut — both single-snapshot structural
  heuristics with no resampling. Its safe-backbone strategy found "no flagged
  edge connected distinct safe components" and produced no candidate at all. P3
  replaces that static safe-component test with a *resampling-consensus*
  definition of persistent sides, which can identify the two real components
  even when the full-complex dual graph exposes no articulation or bridge — the
  exact opposing-sheets failure mode schema 15 could not localize.

The common thread: schemas 12/14/15 are all deterministic single-snapshot
operations on the full-complex output. P3's decisive new ingredient — a per-cell
resampling-consensus vote fused with a component-separation objective — is absent
from all three, and directly targets the reason each of them failed.

## 6. Reused vs new code

Reused: `geometric_bridge_risk` / `boundary_bridge_localization`
(`adaptive.py:664`, `:804`); the `scores[c] = nextafter(m*, ∞)` removal +
`surface_at` recompute (`adaptive.py:372`); `_resampled_point_sets`
(`baselines.py:667`, extended to return retained indices);
`_labeled_false_bridge_counts` / `evaluate_surface` (`surface.py:327`, `:366`);
the calibration freeze pattern of `calibrate_adaptive_multiplier`
(`calibration.py:478`).

New: (1) a resample draw that returns base indices; (2) per-cell resampling
consensus (`persistence`); (3) the three strategies and the connected-component
separation test; (4) a calibration-only ablation driver and its frozen-artifact
schema, mirroring `g5_validation`'s conservative claim boundary.

## 7. Predeclared success / failure conditions

- **Promotable-candidate signal**: on calibration, some (strategy, τ, rho)
  strictly reduces labeled bridge edges with no B4/B5 regression, and that frozen
  configuration is then non-inferior to both B4 and B5 on the G5 held-out panel.
- **Honest negative**: no calibration configuration passes → freeze S0, record P3
  as another negative intervention (like 14/15), and leave the promotion
  bottleneck to G4. Either way `promotion_supported` stays false and the result
  is filed as a synthetic ablation, not paper evidence.
