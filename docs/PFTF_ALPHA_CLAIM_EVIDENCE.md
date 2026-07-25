# PFTF-alpha — claim / evidence matrix for the negative-limits paper (framing b)

Paper framing (confirmed 2026-07-25): **honest negative / limits result.** The
manuscript does not claim the PFTF local SPD metric beats prior-art anisotropic
alpha; it reports a rigorous frozen held-out evaluation whose primary finding is
negative, plus two real secondary contributions (a strictly better density
baseline, and a deployed fail-closed exact-construction fallback). Every claim
below is bounded by committed evidence; `promotion_supported` is false throughout.

## Primary claim (negative, the headline)

| Claim | Evidence | Boundary |
|---|---|---|
| On real higher-fidelity held-out meshes the PFTF local SPD metric (P1) and its confidence-fallback variant (P2) show **no value beyond** the established normal/PCA-anisotropic alpha baseline (B5). | P3 frozen real-data held-out (`pftf_alpha.real_heldout`, `pftf_alpha_p3_real_heldout/v1`): held-out mean F-score B5 0.868 vs P1 0.727, P2 0.728; P1/P2 casewise B4/B5 envelope F-margin ≈ −0.14, 0/40 cases cleared. | Local corpus (Thingi10K + CAD), not a licensed leaderboard; topology endpoints on complex shells are noisy (geometry-primary reporting). |
| The synthetic panel **over-stated** anisotropic-baseline weakness: B5 looks degenerate there only because of an under-resolved pathology, reversing on real data. | Synthetic G5 (B5 Betti 105 / nonmanifold 175) vs P3 real (B5 best). M2 probes: `opposing_sheets` gap < sampling spacing (43% of kNN edges cross sheets), so no local method can separate it. | The synthetic result is a preflight, not evidence of real superiority for any method. |

## Secondary contributions (positive, honestly bounded)

| Claim | Evidence | Boundary |
|---|---|---|
| **M1** (regular/weighted alpha with proper power-circumradius scoring) is a strictly better **density-adaptive** alpha than B4 — it changes connectivity rather than rescoring a fixed Delaunay. | M1 ablation (`pftf_alpha_m1_weighted_alpha_ablation/v1`): synthetic frozen `scale=0.375` strictly dominates B4 on all endpoints (F 0.365→0.384, geometry 0.180→0.173, bridges 48→46, equal topology); real held-out M1 F 0.774 > B4 0.728. | M1 does **not** beat B5; it is a better B4, not a new state of the art. Floating Qhull lift, not an exact weighted-alpha complex. |
| A **deployed exact / validated fail-closed fallback** (G4) turns the exact-construction work into a real selection path with a proven no-silent-false-safe invariant. | `pftf_alpha.g4_fallback` (`pftf_alpha_g4_fail_closed/v1`), 9 failure-mode tests; injects host-validated exact Euclidean Delaunay connectivity, else fails closed to a labeled conservative fallback. | Certifies only base Delaunay connectivity, **not** the anisotropic PFTF complex; 64-point cap. |
| **Removal-based** false-bridge interventions fail **universally** by exposing boundary; only a connectivity change helps. | Negative evidence: schemas 12/14/15, P1 resampling-persistence probe (AUC≈0.50), multiplier backoff, reference-free routing, M2 oriented-normal removal/offset/signed labeling — all regress. | A global graph-cut tetrahedron labeling (Labatut-style) remains untested (separate build). |
| A **frozen, predeclared held-out methodology** (synthetic G5 four-profile preflight + real P3), with information-boundary discipline and fail-closed claim tracking. | G5 (`pftf_alpha_g5_preflight/v1`), P3, and the predeclared design docs; every method-development step was probe-gated before build. | Methodology contribution, not a performance claim. |

## What the paper must NOT claim

- PFTF-alpha beats prior-art anisotropic alpha (P3 refutes it on real data).
- The PFTF complex is exact (G4 certifies only base Delaunay connectivity).
- The false-bridge problem is solved (thin-gap is under-resolved / unsolved).
- Any promotion: `promotion_supported=false` in every artifact.

## Suggested paper spine

1. Motivation: is a PFTF-derived local SPD metric worth the complexity over
   density (B4) and normal-anisotropic (B5) alpha baselines?
2. Methods: B0–P2, M1 weighted alpha, G4 fail-closed fallback; frozen protocols.
3. Synthetic preflight (G5) and its honest limitations (under-resolved thin-gap).
4. Real held-out (P3): the negative primary result (B5 wins; PFTF adds no value).
5. What *does* survive: M1 as a better density baseline; G4 safety; the
   comprehensive negative map of removal interventions.
6. Limits and the one untested path (global graph-cut labeling).
