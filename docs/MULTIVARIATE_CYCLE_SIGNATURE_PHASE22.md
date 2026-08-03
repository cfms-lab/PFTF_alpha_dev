# Multivariate cycle signature: Phase 22

## Frozen question

Phase 21 showed that total Hungarian cycle gain alone cannot separate cycles
that improve correspondence from cycles that damage or fail to improve it. Can
a fixed multivariate description of the within-cycle cost changes transfer
across fresh synthetic cohorts while preserving already-correct presented
pairs?

The repository contains no measured acquisition timestamp, ray, track, or
hardware correspondence metadata. Phase 22 therefore remains an observed-
coordinate synthetic study. It does not claim real correspondence recovery.

## Frozen information boundary

Routing receives only:

1. the presented primary and repeat coordinate arrays;
2. their asserted row pairing for the unchanged robust alignment;
3. the Phase-20 tangent-only Hungarian candidate permutation; and
4. one linear cycle-signature model frozen on the Phase-22 training cohort.

Source IDs are used only to label training cycles and to evaluate opened
panels. Stress identity, clean surfaces, analytic heights, geometry/topology
endpoints, and case decisions are unavailable to feature construction,
training, and routing.

## Frozen candidate and cycle decomposition

Reuse without modification the Phase-20 three-iteration 80%-trimmed alignment,
12-neighbour primary tangent frames, tangent spacing floor, complete Hungarian
assignment, and deterministic tie perturbation. Decompose the resulting
permutation into nontrivial cycles using the Phase-21 rule.

For every primary row, compute tangent and absolute-normal assignment costs,
each divided by the same local tangent spacing. The total cost is the Euclidean
combination of those two coordinates. Normal cost is diagnostic input only; it
does not change the Hungarian candidate.

## Frozen 14-coordinate cycle signature

For each cycle, define row log ratios

`log((presented_cost + 1e-6) / (assigned_cost + 1e-6))`,

clipped to `[-20, 20]`, separately for tangent, normal, and total cost. Positive
values favor reassignment. The fixed signature, in order, is:

1. log cycle length;
2. Phase-21 aggregate tangent relative gain;
3. minimum tangent log ratio;
4. median tangent log ratio;
5. fraction of positive tangent log ratios;
6. minimum normal log ratio;
7. median normal log ratio;
8. fraction of positive normal log ratios;
9. minimum total-cost log ratio;
10. median total-cost log ratio;
11. fraction of positive total-cost log ratios;
12. `log1p` of median presented tangent cost;
13. `log1p` of maximum presented tangent cost; and
14. `log1p` of median assigned tangent cost.

No feature selection, interaction, polynomial expansion, per-profile feature,
or stress/case identifier is allowed.

## Frozen training label and linear model

A cycle is labelled `strictly_correcting` only when applying the entire cycle
makes every row in that cycle truth-correct and corrects at least one row.
Every other cycle, including partially improving cycles, is unsafe for model
training.

On training cohort A only:

1. standardize each feature by its training mean and population standard
   deviation, replacing a scale below `1e-12` by one;
2. fit one ridge linear least-squares score to targets one for strictly
   correcting and zero otherwise;
3. use ridge penalty `1.0` on feature coefficients and no penalty on the
   intercept; and
4. freeze the acceptance cutoff as the next larger binary64 value above the
   maximum unsafe-cycle training score.

The fit must contain at least one strictly correcting and one unsafe cycle.
At routing time a complete cycle is applied only when its frozen linear score
is at least the cutoff. Otherwise every row in that cycle keeps the presented
identity pairing. The model is never refit after training A is opened.

## Frozen sequential panels and seeds

Every panel uses the unchanged five Phase-18 correspondence profiles, nine
Phase-8 stresses, `N in {96,160,256}`, eight repeats, 2048 clean-reference
points, and 256 endpoint surface samples.

- Training A: seed `23900804`.
- Development transfer B: seed `24000804`, opened only if training A passes.
- Validation A: seed `24100804`, opened only if both development panels pass.
- Validation B: seed `24200804`, opened only if validation A passes.
- Final held-out: seed `24300804`, opened only if validation B passes.
- Prior opened or reserved Phase-20/21 seeds `23300804` through `23800804` are
  forbidden.

## Frozen panel gates

Every profile in every opened panel must satisfy:

1. at least 99% aggregate truth assignment accuracy;
2. at least 90% repair of presented mismatches for `mismatch_02` and
   `combined`;
3. at least one unguarded harmful-outlier false-safe;
4. zero guarded harmful-outlier false-safes; and
5. at least 90% safe control/local-bump retention.

Training A must pass before B is opened. B must also pass before validation A
is opened. The validations are sequential, and both must pass before final is
opened. No feature, label, penalty, cutoff, seed, or gate may change after
training A is observed.

## Frozen downstream guard and claim boundary

Reorder original unaligned repeat coordinates only for accepted cycles. Reuse
the Phase-17 matched-displacement evidence and the Phase-18 rectangle exactly:

`peak <= 10.922625244331805 AND support <= infinity`.

Even a pass supports only the declared supervised synthetic multivariate cycle
route. `real_correspondence_supported`, `real_paired_scan_supported`,
`trimmed_reconstruction_supported`, and `deployment_supported` remain false.

## Planned run

```powershell
python -m pftf_alpha.multivariate_cycle_signature `
  --output benchmark-out/multivariate_cycle_signature_phase22.json
```

## Result

The implementation and full regression suite passed before training seed
`23900804` was opened. The training fit contained 1,867 cycles: 343 were
strictly correcting and 1,524 were unsafe under the frozen label. The fitted
cutoff was `0.8124335749442713`.

The cutoff admitted 276/343 strictly correcting cycles, rejected 67/343, and
admitted 0/1,524 unsafe cycles. Consequently no new truth mismatch was
introduced in any profile. Training A nevertheless failed multiple frozen
gates:

- Overall assignment was 176,373/176,832 (`99.74%`). Harmful false-safes fell
  from 275 to 1, and safe-focus retention was 183/205 (`89.27%`).
- Exact, registration-only, and missing-only profiles preserved 100%
  assignment accuracy and introduced zero mismatches.
- `mismatch_02` repaired 560/720 (`77.78%`) presented mismatches, below 90%,
  and retained 35/41 (`85.37%`) safe-focus accepts.
- `combined` repaired 346/645 (`53.64%`), below 90%, and retained 25/41
  (`60.98%`) safe-focus accepts. Its assignment accuracy was `99.10%`.
- The missing-only profile retained one guarded harmful-outlier false-safe even
  though no cycle was accepted and assignment accuracy was 100%.

That remaining case is `missing_10pct`, `outliers_01`, N=96, repeat 3, case
seed `24330837`. It has one harmful outlier vertex and six harmful faces.
Presented pairing was exact, the signature route made zero changes, and the
matched-displacement peak was `9.266772085354779`, below the frozen Phase-18
limit `10.922625244331805`; the support value was
`3.114570793792814` against the frozen infinite support limit. This is a fresh-
seed downstream guard transfer failure independent of correspondence routing.

Training A failed, so development B seed `24000804`, validation seeds
`24100804`/`24200804`, and final seed `24300804` were not opened. Their artifact
fields are `null`. The feature set, ridge fit, cutoff, and guard were not
changed after training A was observed.

Therefore `phase22_supported=false` and
`multivariate_cycle_signature_synthetic_supported=false`. The richer signature
improves training precision but still lacks the recall needed for injected
mismatches; moreover, further correspondence-only refinement cannot address a
harmful accept that occurs under exact retained pairing. Before another
matcher phase, the matched-displacement guard itself needs a separately
preregistered transfer audit or a stronger acquisition information source.
`real_correspondence_supported=false`, `real_paired_scan_supported=false`,
`trimmed_reconstruction_supported=false`, and `deployment_supported=false`.

The reproducible artifact is
`benchmark-out/multivariate_cycle_signature_phase22.json` (gitignored,
3,835,681 bytes, SHA-256
`e5fefc3ed2465ea22c43108b39912fc535d3920084bab16287a5f6628931447c`).
