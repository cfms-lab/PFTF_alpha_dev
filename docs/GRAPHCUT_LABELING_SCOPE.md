# Scope — Global graph-cut inside/outside tet labeling (headline-advantage candidate)

**Date:** 2026-07-25 · **Status:** feasibility scope (no code written) · **Trigger:** PFTF_alpha
paper is on hold ("장점 부각 실패") — the honest negative/limits result needs at least one
clear headline advantage. This document scopes the single untested path that could create one.

## 1. Why this path (and only this path)

Every method tested so far (B4 density, B5 normal-anisotropic, P1/P2 PFTF local SPD, M1
weighted alpha) makes a **local, per-cell** decision on a **fixed** triangulation. On the
hardest family the inter-sheet gap is smaller than the sample spacing (41 % of kNN edges cross
the two ground-truth sheets), so the information needed to separate the sheets is *absent from
local neighborhoods* — no local method can win there, by construction (see
`docs/P3_REAL_HELDOUT_DESIGN.md`, memory `m2-thin-gap-exhausted`).

A **global graph-cut** (Labatut–Pons–Keriven 2009) labels each Delaunay tetrahedron
inside/outside by a min-cut over the tet adjacency graph. It is the one approach whose decision
is *not* local: it can cut a false bridge without opening a hole because it enforces global
consistency (each sheet must bound a consistent inside/outside). It is therefore the only
candidate that could turn the negative result into a **positive contrast**:

> "All local anisotropic alpha methods (prior-art B5 included, and our PFTF P1/P2) fail the
> thin-gap held-out gate; a global graph-cut labeling clears it." — local vs. global.

## 2. Grounded integration map (from a full code read)

Package `src/pftf_alpha`, Python ≥3.12, deps **only** `numpy>=2.0`, `scipy>=1.14`.

### Reusable as-is (large — most infrastructure already exists)
- **Delaunay build** — `filtration.py:137` (`scipy.spatial.Delaunay`, Qhull); `AlphaFiltration`
  keeps a facet→incident-tet coface map `filtration.py:108` (2-coface = interior adjacency,
  1-coface = hull boundary).
- **All evaluation endpoints** — `surface.py:evaluate_surface` (`surface.py:366`) computes surface
  F-score (`surface.py:279`), geometric loss (normalized Chamfer + Hausdorff, `surface.py:279`),
  Betti error via GF(2) (`surface.py:218/421`), nonmanifold edges (`surface.py:182`), labeled
  false-bridge edges/faces (`surface.py:327/454`). **Zero new endpoint code needed.**
- **PCA normals + kNN spacing** — `local_neighborhood_geometry` (`adaptive.py:624`), normals as
  smallest eigenvector (`adaptive.py:698`); `knn_scales` (`adaptive.py:608`, cKDTree).
- **Tet dual-graph + bridge code** — `_dual_cut_structure` Tarjan pass (`adaptive.py:750`),
  face-owner/dual-adjacency construction (`adaptive.py:870-884`). Nearest existing "tet adjacency
  graph"; reuse for graph assembly.
- **Max-flow primitive** — `scipy.sparse.csgraph.maximum_flow` is available (scipy ≥1.14).
  **No new dependency required.** (Grep confirms no maxflow/mincut/visibility/labatut code exists
  yet anywhere.)
- **Output type** — `SurfaceMesh(vertices, faces)` (`surface.py:20`).

### Must be built new
1. **Full tet-adjacency graph with an outside/source node.** Currently only `.simplices` is kept
   (`filtration.py:140`); scipy's `Delaunay.neighbors` and hull-infinite cells (the natural
   "outside" source) are discarded. Re-run Delaunay keeping `.neighbors`, or lift the coface map.
   *Effort: small–moderate.*
2. **Data term (inside/outside preference) — the crux, see §3.** *Effort: the research risk.*
3. **Smoothness term** — cost of the surface passing through a shared facet (facet area ×
   dihedral / normal agreement). *Effort: small.*
4. **Max-flow solve + label→SurfaceMesh extraction** — build a CSR s/t capacity graph, call
   `maximum_flow`, take boundary facets between inside and outside labels as the mesh.
   *Effort: moderate.*
5. **Harness reconciliation (see §4)** — the whole harness selects surfaces by sweeping a scalar
   threshold over per-tet `scores` (`adaptive.py:372 surface_at`, `_adaptive_baseline`
   `baselines.py:840`, `_freeze_multiplier` `real_heldout.py:216`). A graph-cut is a *single
   binary labeling*, not a monotone sweep. *Effort: moderate; design decision.*
6. **Registration** — add a `BaselineID` (`baselines.py:41`), a builder branch in `_adaptive`
   (`real_heldout.py:153`), and entries in `METHODS`/`CANDIDATES` (`real_heldout.py:184/43`,
   `g5_validation.py:35/41`). *Effort: small.*

## 3. The dominant risk: a data term without visibility

Labatut's power comes from **sensor visibility** — rays from the scanner to each point imply the
traversed space is empty (outside). **We have no rays, no sensor/camera, no scan direction**
anywhere (confirmed by grep); inputs are bare clouds sampled from meshes (`real_heldout.py:120`),
and the PCA normals are **unoriented** (sign-ambiguous, `adaptive.py:698`).

So the visibility data term must be *synthesized* from what we have. Options, in rough order of
ambition:
- **(A) Hull/outside as source, interior seed as sink**, smoothness-dominated cut. Cheapest;
  tests whether global consistency alone separates the gap.
- **(B) Consistently oriented normals** (add Hoppe-style orientation propagation — MST over the
  kNN graph — since current normals are unoriented) to define an inside/outside preference per
  facet. More faithful to Labatut's normal-consistency term.
- **(C) Free-space heuristic** (mark large empty Delaunay balls as outside) as a visibility proxy.

**The make-or-break question:** B5 *already* uses local normals and still fails the thin gap
because its decision is local. Does enforcing the *same* normal information **globally** (closed,
orientable, consistent inside/outside) resolve the gap where local scoring could not? The
literature says global labeling helps on thin structures *when visibility is present*; without
visibility, the added signal is only global consistency. **Plausible, not guaranteed.** This is
exactly what the pilot (Phase 0) must answer before any further investment.

## 4. Harness reconciliation

Two ways to make a single-shot labeling fit a threshold-sweep harness:
- **(i) Score encoding** — inside tets → score 0, outside → +inf; then `surface_at(threshold>0)`
  returns the cut boundary unchanged. Minimal code, but `critical_values()` collapses to a
  degenerate 1-point spectrum, which `calibrate_adaptive_multiplier`/`_freeze_multiplier`
  (`real_heldout.py:216`) assume is a range — needs a guard.
- **(ii) Single-shot path** — bypass `surface_at`, build the `SurfaceMesh` from the cut boundary
  and call `evaluate_surface` directly; add a small branch in the held-out loop
  (`real_heldout.py:373-426`). Cleaner for a one-shot method; a little new plumbing.
Recommend **(ii)** — the method has no scale to freeze, so forcing it through the multiplier
machinery is dishonest; a dedicated single-shot branch is clearer and keeps the frozen-protocol
guarantees intact (still no held-out tuning).

## 5. Phased plan with a hard go/no-go gate

| Phase | Work | Output | Rough effort |
|---|---|---|---|
| **0 — Pilot (GO/NO-GO)** | tet-adjacency+source (build-new 1) + one data term (option A or B) + `scipy.maximum_flow` + label→mesh, on **one** synthetic thin-gap case | Does the cut separate the two sheets without opening a hole? (visual + Betti/false-bridge on 1 case) | **~1 week** |
| 1 — Productionize | data/smoothness terms, oriented-normal propagation if needed, robust label→`SurfaceMesh`, single-shot harness path (§4-ii) | a registered method that runs on the panel | ~1–1.5 weeks |
| 2 — Held-out eval | register in G5 + `real_heldout` frozen protocol, run calibration/held-out, add tests (`tests/test_<module>.py`, ~150–250 lines) | casewise envelope margins on synthetic + real | ~1 week |
| 3 — Write-up | if it clears the envelope, reframe the paper around the local-vs-global contrast | headline advantage | (paper work) |

**Total to a go/no-go answer: ~1 week. To a full held-out verdict: ~3–4 weeks.** Most of the
infrastructure (endpoints, normals, Delaunay, max-flow) is reused, so the new code is one focused
module (~400–600 lines) plus small harness edits — the cost is concentrated in the §3 data-term
research risk, which the Phase-0 pilot isolates deliberately.

## 6. Strategic caveat (read before committing)

A graph-cut success would save the **paper**, but note carefully what it does and does not
vindicate:
- It reframes the contribution as **"global labeling beats all local anisotropic alpha,
  including our PFTF local metric."** The headline advantage then belongs to the **graph-cut**,
  not to PFTF. P1/P2 still lose; the PFTF local SPD metric remains a documented negative.
- That is still a legitimate, publishable positive result (a clean local-vs-global boundary with
  a rigorous held-out protocol and a safety fallback), and it is arguably a *stronger* paper than
  a marginal PFTF win — but it is a **pivot of the paper's thesis away from PFTF**. Confirm this
  is acceptable before funding Phases 1–3.
- Claim boundary unchanged: the cut runs on floating Qhull connectivity (`filtration.py:137`); the
  exact backend (`g4_fallback.py`) certifies only base Delaunay, so any graph-cut result inherits
  the "non-exact" boundary.

## 7. Recommendation

**GO for Phase 0 only.** It is cheap (~1 week, reuses ~80 % of needed infrastructure), and it
answers the one make-or-break question (§3) before any larger commitment. Fund Phases 1–3 **only
if** the pilot separates the thin gap **and** the §6 thesis pivot (credit to graph-cut, not PFTF)
is acceptable. If Phase 0 fails to separate the gap with normals-only signal, the honest
conclusion is that thin-gap reconstruction needs true sensor visibility — a data-acquisition
change, not an algorithm change — and the paper stays a negative/limits result.
