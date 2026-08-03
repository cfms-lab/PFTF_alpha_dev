# Local/spatial residual guard transfer: Phase 28

## Question

Phase 27 showed that the frozen twelve-feature global score does not preserve
harmful/focus ordering on a new panel. Can a genuinely spatial observation--a
matched displacement's disagreement with nearby matched displacements--remove
the known residual harm while transferring to untouched case-seed-disjoint
panels?

## Development evidence and claim boundary

All panels opened through Phase 27 are development evidence only. Phase 28
uses:

- Phase 24/25 score-fit seed `24900804` and cutoff-calibration seed `25000804`
  only to reproduce the predecessor score;
- Phase 26 A/B seeds `27500804` and `27600804` as local-feature design panels;
  and
- failed Phase 27 Validation A seed `27800804` as the local-cutoff calibration
  panel.

Phase 27's reserved bases `27900804` and `28000804` remain unopened, but are
forbidden as future fresh evidence. No result from the development panels can
support Phase 28.

## New observed local/spatial evidence

The global Phase 27 signature and cutoff `0.18181536333942858` remain frozen.
For each presented matched pair, Phase 28 additionally:

1. finds its eight nearest primary-coordinate neighbors in observed 3D space;
2. reuses the Phase 17 robust displacement location and axis scales;
3. forms the standardized displacement vector `z_i`;
4. computes
   `r_i = norm(z_i - componentwise_median(z_j for j in N8(i)))`; and
5. summarizes the case by `q95_local = percentile95(r_i)`.

The route may use ordered primary/repeat coordinates, presented pair order,
the primary-coordinate k-nearest-neighbor graph, the frozen Phase 27 score,
and the frozen full-primary layer partition. It may not use source labels,
injected-outlier identities, pairing-correctness truth, clean references,
stress/profile labels, or endpoint metrics.

The feature is invariant to a joint permutation of the matched pairs and is
zero for a shared translation after robust displacement centering. It is a
case-level summary of a local observed field, not another source-label or
endpoint-derived feature.

## Frozen development selection

On the already-open `275/276/278` panels, the development comparison included
maximum and second local residual, 95th-percentile local residual, maximum
local score excess, peak-neighbor support, and neighbor-radius summaries. This
comparison is truth-supervised design work, not validation.

Among the high-reject residual summaries that remove the one Phase 27 residual
harm, `q95_local` retained the most focus cases: 379/381 under the local rule
alone. With the unchanged Phase 27 score applied first, the combined route
retains 378/381 development focus cases.

The one predecessor residual harmful case is
`missing_10pct/outliers_01`, N=96, repeat 1, case seed `28210823`. Its
`q95_local` is `3.5441330652515526`. Freeze

`local_cutoff = nextafter(3.5441330652515526, -infinity)`

which equals `3.544133065251552`; the binary64 decrement is
`4.440892098500626e-16`.

The combined route accepts only when both strict tests pass:

`phase27_score < 0.18181536333942858`

and

`q95_local < 3.544133065251552`.

No coefficient, neighborhood size, local statistic, cutoff, reconstruction,
profile, endpoint, or gate may change after this document.

## Design-only reproduction

The implementation's `--design-only` run reproduced the predecessor failure,
the one residual harmful case, all fixed values above, and zero historical or
mutual case-seed overlap. The combined design panels are:

| panel | harm, original -> predecessor -> combined | focus, combined/original | all-safe, combined/original | introduced harm | gate |
|---|---:|---:|---:|---:|---|
| score fit `24900804` | 168 -> 0 -> 0 | 131/132 | 353/360 | 0 | pass |
| cutoff calibration `25000804` | 165 -> 0 -> 0 | 124/126 | 345/351 | 0 | pass |
| local design A `27500804` | 168 -> 0 -> 0 | 122/123 | 359/360 | 0 | pass |
| local design B `27600804` | 165 -> 0 -> 0 | 131/132 | 358/360 | 0 | pass |
| local calibration `27800804` | 153 -> 1 -> 0 | 125/126 | 362/366 | 0 | pass |

These results establish only that the implementation reproduces its declared
development target.

## Fresh case-seed-disjoint protocol

The new base seeds are:

- Validation A: `28100804`;
- Validation B: `28200804`, opened only if Validation A passes; and
- final held-out: `28300804`, opened only if Validation B passes.

Using

`case_seed = base + count_index*1000003 + stress_index*100003 + repeat*10007`,

each 216-case set is mutually disjoint and has zero overlap with every full
panel base from `20300804` through `25900804` and every used or reserved base
from `27500804` through `28000804`.

Each panel keeps the unchanged nine stresses, `N in {96,160,256}`, eight
repeats, 2048 clean-reference points, 256 endpoint surface samples, and exact,
0.5-degree registration, and 10%-missing profiles. Each opened panel therefore
audits 648 profile rows from 216 primary cases.

## Frozen gates and sequential opening

Before Validation A is opened, Ruff, the diff check, and the full regression
suite must pass, and the evaluator must reproduce every design value and gate.
Every profile in every opened fresh panel must:

1. contain at least one original unguarded harmful false-safe;
2. reduce routed harmful false-safes to zero;
3. retain at least 90% of original-safe control/local-bump accepts; and
4. accept zero originally safe cases whose routed endpoint becomes harmful.

Validation B opens only after Validation A passes. The final panel opens only
after Validation B passes. A failure closes all later panels. No opened seed
may alter the feature, cutoff, or route.

Even a complete pass supports only a synthetic matched-pair/local-neighborhood
guard under the presented-pair simulator. `real_correspondence_supported`,
`real_paired_scan_supported`, `real_trimmed_reconstruction_supported`, and
`deployment_supported` remain false.

## Planned run

```powershell
python -m pftf_alpha.local_spatial_residual_guard `
  --output benchmark-out/local_spatial_residual_guard_phase28.json
```

## Result

Ruff, the diff check, and all 274 tests passed before base `28100804` was
opened. The evaluator then reproduced every design value and opened the fresh
panels sequentially. All three passed:

| panel | harm, original -> predecessor -> combined | focus, combined/original | all-safe, combined/original | introduced harm | gate |
|---|---:|---:|---:|---:|---|
| Validation A `28100804` | 171 -> 0 -> 0 | 129/129 | 354/360 | 0 | pass |
| Validation B `28200804` | 171 -> 0 -> 0 | 126/126 | 361/363 | 0 | pass |
| final `28300804` | 168 -> 0 -> 0 | 129/129 | 347/354 | 0 | pass |

Every exact, registration, and missing profile has zero combined harmful
false-safes, 100% focus retention, zero introduced routed endpoint harm, and a
passing profile gate. Therefore `phase28_supported=true` and
`local_spatial_residual_guard_synthetic_supported=true` for the preregistered
synthetic combined route.

The marginal-evidence boundary is narrower. On all three fresh panels, the
Phase 27 predecessor already reduced harmful false-safes to zero. The local
guard additionally rejected 3, 2, and 4 predecessor accepts respectively, but
all nine were non-focus safe cases and none was a residual harmful case. Thus
the fresh run demonstrates transfer of the combined safety/utility protocol,
not a fresh incremental harmful-case rescue attributable to `q95_local`.

No feature, cutoff, model, reconstruction, profile, or gate was changed after
opening Validation A. Real correspondence, paired scans, trimmed
reconstruction, and deployment remain unsupported.

Artifact: `benchmark-out/local_spatial_residual_guard_phase28.json`, SHA-256
`2e64562de0d8a7094f8e5e2092c238c90d94649b166a0d9a04e9df597eacc198`.
