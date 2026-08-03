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

## Separate Phase-43 bounded result (not the negative-paper headline)

| Claim | Evidence | Boundary |
|---|---|---|
| Continuous observed-confidence weighting can outperform fused B4 and binary point deletion on the frozen analytic multi-view panel. | Phase 43: held-out geometry 0.134240 versus 0.153141/0.145920, Betti error 1.444444 versus 2.611111/2.111111, and repeat stability 0.021446 versus 0.034078/0.043830. All preregistered gates pass. | Synthetic analytic surfaces only. The score is a closure-preserving tetrahedron filtration penalty, not P1/P2, a classical local-alpha complex, or a learned global alpha. It does not beat anchor B4 on combined objective, every calibration method hits the 0.84 grid boundary, and topology remains imperfect. |
| The Phase-43 advantage does **not transfer** under a reference-free complete critical-gap selector and new density/occlusion/local-warp stresses. | Phase 44: continuous objective 0.208318 versus fused B4 0.207237 and binary deletion 0.208194; only 4/27 joint casewise wins versus the frozen 18/27 gate. | Negative synthetic transfer evidence. Do not tune the opened panel or treat another fixed-complex score penalty as a distinct local-alpha construction. |
| Confidence-aware power weights change regular-triangulation connectivity but still do **not** establish value beyond simpler confidence routes. | Phase 45: connectivity changes in 27/27 held-out cases with zero fallback and best stability, but objective 0.211541 trails M1 0.211419, fixed-cell continuous 0.211105, and binary deletion 0.210246; joint wins are 6/27. | Floating-Qhull synthetic evidence only. The scalar confidence-to-power mapping is not a PFTF-trained alpha, exact weighted complex, or local-SPD metric construction. |
| One constant global SPD metric defines a coherent affine alpha-complex control. | Phase 46: identity, explicit-transform, affine-covariance, and constant-field controls reproduce canonical connectivity and filtration values; a rotating local field fails closed. All 5/5 frozen controls pass. | Construction-invariant evidence only. It does not establish a spatially varying local-SPD complex, exact predicates, reconstruction improvement, topology correctness, real transfer, or deployment. |
| One explicit globally injective nonlinear coordinate map defines a coherent spatially varying SPD alpha-complex control. | Phase 47: the quadratic shear has an exact inverse, determinant 1, varying induced metrics, and changes Delaunay connectivity by 133 symmetric-difference cells; all 8/8 frozen controls pass and the nonintegrable Jacobian field fails closed. | Analytic construction evidence only. It does not establish arbitrary point-local metrics, a PFTF-conditioned map, reconstruction advantage, exact predicates, topology correctness, real transfer, or deployment. A generic floating rotation preserves connectivity but misses the strict score tolerance. |
| Frozen PFTF summaries do **not** identify the coefficient of the declared invertible quadratic-shear family better than simple controls. | Phase 48: all 45 held-out maps pass the inverse/determinant audit, but PFTF coefficient MAE/RMS/Jaccard are `0.115118/0.098199/0.576518`, behind the TRAIN mean (`0.100000/0.085432/0.601368`) and non-PFTF geometry ridge (`0.100103/0.085439/0.598767`). | Negative synthetic coordinate-aligned recovery evidence. It does not support PFTF-conditioned map value, arbitrary local-SPD metrics, alpha selection, reconstruction/topology benefit, real transfer, exactness, or deployment. Do not retune the opened held-out panel. |

## Separate Phase-2 positive result (not the negative-paper headline)

| Claim | Evidence | Boundary |
|---|---|---|
| **Parallel two-layer constrained connectivity** is the first safe positive reconstruction result in its declared regime. | Phase 2 (`pftf_alpha_two_layer_connectivity_phase2/v1`): new synthetic held-out, 16/16 sampling-sufficient cases accepted and truly safe, false-safe 0, mean F 0.3922 -> 0.7564 vs B5, component/bridge/Betti errors all reduced to zero. | Specialized two-layer planar model; not PFTF-SPD evidence, not general alpha reconstruction, and not real-scan or deployment evidence. It belongs in a separate positive follow-up. |
| The candidate generalizes within a **globally separable, approximately parallel two-layer** synthetic regime and fails closed on tested violations. | Phase 3 (`pftf_alpha_two_layer_stress_phase3/v1`): 32/32 rotated, shallow-curved, varying-gap, and partial-overlap cases accepted and truly safe; mean F 0.4714 -> 0.8256 vs B5; Betti error 1159 -> 0. All 16 near-contact/crossing negatives rejected; false-safe 0. | Synthetic stress only. Curvature and tilt ranges are narrow, family declarations are evaluation-only, and there is no real-scan, visibility-aware, arbitrary-intersection, PFTF-SPD, or deployment evidence. |
| The Phase-3 candidate has a measured **curvature limit and an unsafe global-coordinate blind spot**. | Phase 4 (`pftf_alpha_two_layer_boundary_phase4/v1`): 192 cases. Curvature <=0.24 was 24/24 safe; curvature 0.36/0.48/0.60 produced 6/6/3 false-safe accepts. Tilt became rejection-dominant at span 0.55, contact at severity 0.60, and overlap offset remained safe through 2.50. | This is negative boundary evidence. It narrows rather than promotes the Phase-3 claim. The tested grid cannot establish an exact continuous threshold. `phase4_diagnostic_supported=false`. |
| An **observed-only normal-coherence guard** removes the measured curvature blind spot on a disjoint held-out seed. | Phase 4b (`pftf_alpha_curvature_guard_phase4b/v1`): threshold 0.82 frozen on Phase 4; new 192-case seed. Base false-safe 14 -> guarded 0; safe accepts 117 -> 116; retention 99.15%; all required anchors >=75% coverage. | Conventional kNN-PCA orientation tensor, not PFTF-SPD novelty. Synthetic generator and fixed density only; no universal threshold, real scan, or deployment evidence. |
| The fixed coherence guard **does not transfer across density and curvature shape**. | Phase 5 (`pftf_alpha_curvature_guard_domain_shift_phase5/v1`): 360 cases crossing n={96,160,256}, noise={0.005,0.010,0.025}, and five shapes. False-safe 57 -> 12, safe retention 79.03%; all 12 survivors are dense asymmetric-converging cases. | Negative transfer evidence. The 0.82 threshold is too conservative for sparse safe cases and too permissive for dense converging layers. `phase5_supported=false`. |
| A density-normalized local-order guard improves transfer but **does not certify safety**. | Phase 6 (`pftf_alpha_local_order_guard_phase6/v1`): frozen 360-case held-out. False-safe 61 -> 2 and safe retention 91.58%; N=160/256 false-safe 59 -> 0, but both N=96 false-safe survive. | Useful diagnostic progress, not a positive safety claim. The two sparse survivors violate the predeclared zero-false-safe gate; no post-hoc retuning. `phase6_supported=false`. |
| **Shared-trend residual layer inference** repairs the upstream global-coordinate assignment failure in the declared two-layer regime. | Phase 7 (`pftf_alpha_shared_trend_inference_phase7/v1`): frozen 360-case held-out. Base false-safe 60 -> candidate 0; all 186 base-safe accepts retained; 58/60 base false-safe cases become accepted safe outputs and two fail closed. | Positive model-based baseline, not PFTF-SPD novelty. The quadratic trend family matches the synthetic shapes; no real scan, occlusion/outlier stress, arbitrary surface, or deployment evidence. `phase7_supported=true`, `deployment_supported=false`. |
| The Phase-7 candidate transfers to tested non-outlier sensor stresses at N>=160, but is **outlier-blind** and sparse-conservative. | Phase 8 (`pftf_alpha_sensor_stress_phase8/v1`): 216 frozen cases. At N=160/256, all 96 non-outlier occlusion/imbalance/noise/nonquadratic cases are safe accepts. At N=96, non-outlier coverage is 43.75%. Spatial outliers produce 56 candidate false-safe accepts. | Positive bounded operating envelope plus decisive negative boundary. No outlier robustness, real scan, arbitrary corruption, or deployment support. `phase8_supported=false`. |
| A robust shared-trend residual guard catches moderate contamination but **does not transfer as a complete outlier certificate**. | Phase 9 (`pftf_alpha_outlier_guard_phase9/v1`): frozen 216-case held-out. False-safe 58 -> 4; all accepted 3%/5% contamination removed; safe retention 88.70%. Local-bump retention is only 9/22. | Residual score confounds localized nonquadratic shape with contamination, while near-surface 1% outliers can look like noisy inliers. No post-hoc retuning. `phase9_supported=false`. |

## What the paper must NOT claim

- PFTF-alpha beats prior-art anisotropic alpha (P3 refutes it on real data).
- The PFTF complex is exact (G4 certifies only base Delaunay connectivity).
- The general false-bridge problem is solved (Phases 2-3 cover only an explicit
  sampling-sufficient, globally separable two-layer synthetic regime).
- The current gate is safe for arbitrary smooth curvature (Phase 4 finds 15
  false-safe accepts once the paired-paraboloid curvature reaches 0.36).
- The coherence threshold 0.82 is universal (Phase 4b validates only a disjoint
  seed of the same synthetic generator and density).
- Phase 4b establishes density/shape transfer (Phase 5 refutes this directly).
- Phase 6 establishes a universal density-normalized safety certificate (two
  sparse false-safe cases remain).
- Phase 7 solves arbitrary multilayer reconstruction or establishes PFTF-alpha
  superiority (it validates a conventional shared-quadratic two-layer model).
- Phase 8 supports outlier robustness or sparse sensor deployment (56 outlier
  false-safe accepts and 43.75% N=96 non-outlier coverage refute this).
- Phase 9 supplies a universal shape-agnostic outlier certificate (four 1%
  false-safe cases remain and local-bump safe retention is 40.91%).
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
