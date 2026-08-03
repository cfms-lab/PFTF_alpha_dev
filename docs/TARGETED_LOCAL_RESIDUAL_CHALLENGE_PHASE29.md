# Targeted incremental local-residual challenge: Phase 29

## Question

Phase 28 transferred the combined global-score/local-residual route to three
fresh panels, but its global-score predecessor already removed every harmful
false-safe there. Can a challenge family declared before opening any new case
contain a fresh predecessor residual and show that the frozen local observation
removes it without weakening the Phase 28 safety and utility gates?

## Why a targeted challenge is needed

The Phase 28 fresh panels establish combined-route transfer, not marginal
benefit from `q95_local`: the local guard rejected nine additional safe,
non-focus cases and rescued no fresh harmful case. Phase 29 therefore separates
two questions:

1. does the unchanged combined route remain safe on every opened panel; and
2. is the fresh sample informative enough to contain at least one harmful case
   accepted by the frozen Phase 27 predecessor?

No predecessor residual is an uninformative result, not evidence that the local
guard failed. A residual left by the combined route is a negative result.

## Frozen challenge family

Before opening any fresh seed, freeze:

- point count: `N=96`;
- stresses: `control`, `local_bump`, and `outliers_01`;
- repeats: 64 per stress;
- profiles: unchanged exact, 0.5-degree registration, and 10%-missing views;
- 2048 clean-reference points and 256 endpoint surface samples; and
- 192 primary cases and 576 profile rows per panel.

This family is chosen from the already-declared Phase 28 stress grid because
the known Phase 27 residual occurred at N=96 under `outliers_01`. It is fixed as
a family, rather than by searching fresh seeds for a favorable outcome.

## Frozen route

The route, reconstruction, features, and strict inequalities are exactly those
from Phase 28:

`phase27_score < 0.18181536333942858`

and

`q95_local < 3.544133065251552`.

The reproduced development limiting value is
`3.5441330652515526`. Neighborhood size remains eight. No cutoff, stress,
repeat count, profile, endpoint, or gate may change after Validation A opens.

## Freshness and fixed seeds

Using

`case_seed = base + count_index*1000003 + stress_index*100003 + repeat*10007`,

freeze:

- Validation A: `30200804`;
- Validation B: `30300804`; and
- final held-out: `30400804`.

Each base produces 192 mutually distinct primary case seeds and has zero
case-seed overlap with all full-panel bases `20300804`--`25900804` and all
used or reserved bases `27500804`--`28300804`. Candidate bases
`28400804`--`30100804` collide with historical case seeds when expanded to 64
repeats and are rejected by the audit without running their cases. No alternate
seed will be tried if the fixed panels are uninformative or fail.

## Safety and incremental-information gates

Each opened panel must satisfy the unchanged Phase 28 safety gate across every
profile:

1. at least one original unguarded harmful false-safe is present;
2. the combined route reduces harmful false-safes to zero;
3. it retains at least 90% of original-safe control/local-bump accepts; and
4. it accepts no originally safe case whose routed endpoint becomes harmful.

Validation B opens only after Validation A passes this safety gate, and final
opens only after Validation B passes. The separate Phase 29 incremental gate
passes only if all three panels are opened and pass, the aggregate predecessor
residual harmful count is greater than zero, and the aggregate combined
residual harmful count is zero.

## Design-only result

The evaluator reproduced the complete Phase 28 design, including the score and
local cutoffs, and passed its design gate. The three targeted case-seed sets
passed the mutual and historical disjointness audit. With `--design-only`, zero
fresh panels were opened and `phase29_supported=false`, as required.

## Planned run

```powershell
python -m pftf_alpha.targeted_local_residual_challenge `
  --output benchmark-out/targeted_local_residual_challenge_phase29.json
```

The full Ruff, diff, and regression checks must pass before base `30200804` is
opened.

## Claim boundary

Even a full incremental pass supports only a synthetic targeted challenge of
the frozen matched-pair/local-neighborhood route. The clean reference,
correspondence truth, injected-outlier identity, stress/profile label, and
endpoint truth remain evaluation-only. `real_correspondence_supported`,
`real_paired_scan_supported`, `real_trimmed_reconstruction_supported`, and
`deployment_supported` remain false.

## Result

Ruff, the diff check, and all 278 tests passed before base `30200804` was
opened. The evaluator reproduced the Phase 28 design and opened Validation A.
Validation A failed its safety gate, so Validation B and final remained
unopened exactly as preregistered.

| profile | harm, original -> predecessor -> combined | focus, combined/original | all-safe, combined/original | introduced harm | gate |
|---|---:|---:|---:|---:|---|
| exact | 19 -> 1 -> 1 | 83/86 | 83/90 | 0 | fail |
| registration 0.5 deg | 19 -> 1 -> 1 | 85/86 | 86/90 | 0 | fail |
| missing 10% | 19 -> 1 -> 1 | 83/86 | 84/90 | 0 | fail |
| Validation A total | 57 -> 3 -> 3 | 251/258 | 253/270 | 0 | fail |

The three residual rows are the exact, registration, and missing views of one
`outliers_01`, N=96, repeat-59 primary case with case seed `30991223`. Their
predecessor scores are `0.08974054`, `0.14060191`, and `0.07348659`, all below
the score cutoff. Their `q95_local` values are `2.85526003`, `2.95746686`, and
`2.91324421`, also below the frozen local cutoff. Each routed endpoint retains
one harmful outlier vertex incident to five harmful faces. Thus the challenge
is informative, but the frozen local summary rescues zero of three residual
profile rows and the combined residual count remains three.

The same rows have maximum local residuals `4.75855119`, `4.92015100`, and
`4.86832205`, above the frozen q95 cutoff. This is post-open diagnostic evidence
only: it may motivate a separately preregistered tail-sensitive observation,
but it cannot be used to retune Phase 29 or claim a pass.

Therefore `phase29_supported=false` and
`incremental_local_rescue_synthetic_supported=false`. Real correspondence,
paired scans, trimmed reconstruction, and deployment remain unsupported.

Artifact: `benchmark-out/targeted_local_residual_challenge_phase29.json`,
SHA-256
`bdfea7cc2243b18147ad05d3963d2a46d66d7fdad4005c77e5fcd31c3823b1b6`.
