# Two-layer generalization stress: Phase 3

## Frozen question and claim boundary

Phase 3 asks whether the Phase-2 constrained-connectivity baseline generalizes
beyond horizontal planar sheets while still failing closed on geometries that
violate its two-separated-layer assumption. It does not test PFTF local SPD and
cannot support deployment.

Construction and routing may use observed coordinates and inferred layer IDs
only. True layer labels, dense references, family names, and in/out-of-scope
declarations are evaluation-only.

## Frozen held-out panel

The seed is `20280803`, with eight unseen repeats per family, 160 observed
points, 4096 reference points, and 512 evaluation surface samples.

Positive, in-scope families:

1. `rotated_parallel`: gap 0.80 planar sheets under an arbitrary rigid rotation.
2. `curved_parallel`: gap 0.80 paired paraboloids with curvature 0.12.
3. `tilted_separation`: two sheets whose gap varies linearly from 0.55 to 1.05.
4. `partial_overlap`: gap 0.80 sheets with 50% overlap in the x extent.

Negative, declared out-of-scope families:

5. `near_contact`: gap varies from 0.04 to 0.80, creating a locally unresolved
   contact-like region.
6. `crossing`: sheets intersect along a line and therefore cannot be represented
   as two separated layers in a single global normal coordinate.

Observation noise is 0.01. The observed-only gate remains frozen at k=12,
cross-kNN <= 0.05, minimum cluster fraction 0.20, and separation SNR >= 3.
The reconstruction baseline remains frozen B5.

## Predeclared success gates

Phase 3 passes only if all of the following hold:

1. each positive family has at least four sampling-eligible cases and at least
   75% safe acceptance coverage among its eligible cases;
2. every accepted positive output has correct true component count and zero
   true labeled cross-layer edges/faces;
3. aggregate positive constrained F-score is within 0.01 of or above B5;
4. aggregate positive constrained Betti-error sum does not exceed B5;
5. neither negative family is ever accepted;
6. total false-safe count is zero.

Whatever the result, `deployment_supported` remains false. A failure is kept as
evidence that the Phase-2 scope must be narrowed or the observed-only geometry
gate must be strengthened; thresholds are not retuned on this panel.

## Run

```powershell
python -m pftf_alpha.two_layer_stress `
  --output benchmark-out/two_layer_stress_phase3.json
```

## Result

The frozen 48-case execution passed every predeclared gate.

| Family | Scope | Eligible | Accepted safe | B5 F | Constrained F | B5 Betti error | Constrained Betti error | Gate |
|---|:---:|---:|---:|---:|---:|---:|---:|:---:|
| rotated_parallel | in | 8/8 | 8/8 | 0.5454 | 0.8829 | 266 | 0 | pass |
| curved_parallel | in | 8/8 | 8/8 | 0.3976 | 0.7722 | 324 | 0 | pass |
| tilted_separation | in | 8/8 | 8/8 | 0.4428 | 0.7835 | 283 | 0 | pass |
| partial_overlap | in | 8/8 | 8/8 | 0.4998 | 0.8637 | 286 | 0 | pass |
| near_contact | out | 0/8 | 0/8 | 0.5109 | 0.7635 | 204 | 0 | pass (rejected) |
| crossing | out | 0/8 | 0/8 | 0.4320 | 0.7249 | 162 | 0 | pass (rejected) |

Across the 32 positive cases, all were sampling-eligible, accepted, and truly
safe. Safe acceptance coverage was 1.00, mean F-score improved from B5 `0.4714`
to constrained `0.8256`, and Betti-error sum fell from `1159` to `0`. Across
the 16 negative cases, no case was accepted. Total false-safe count was zero.

Artifact: `benchmark-out/two_layer_stress_phase3.json`.

`phase3_supported=true`, while `deployment_supported=false`. The evidence now
supports rigid-orientation invariance and limited smooth-curvature, varying-gap,
and partial-overlap generalization inside a globally separable two-layer model.
It also shows that the frozen cross-kNN gate rejects the tested near-contact and
crossing cases. It does not establish arbitrary-curvature, intersecting-surface,
real-scan, visibility-aware, PFTF-SPD, or production validity.
