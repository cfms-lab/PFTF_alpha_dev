# Tail-sensitive local guard transfer: Phase 30

## Question

Phase 29 exposed a fresh harmful primary case whose three profile rows passed
both the global score and the q95 local-residual guard. Can a tail-sensitive
observation derived only from the same observed local residual field detect a
rare isolated peak while preserving the fixed safety and focus-retention gates?

## Evidence boundary

Every panel opened through Phase 29 is development evidence only. Phase 30
uses the five Phase 28 design panels, the three Phase 28 fresh panels, and
failed Phase 29 Validation A (`30200804`) to select one tail summary and one
cutoff. The previously reserved Phase 29 bases `30300804` and `30400804` are
forbidden as fresh evidence even though their cases were never evaluated.

The route sees only presented primary and repeat coordinates, their matched
order, the primary-coordinate eight-nearest-neighbor graph, the frozen global
score, and local residual summaries. Source labels, outlier identities, clean
references, stress/profile labels, and endpoint truth remain evaluation-only.

## Declared candidate comparison

For each case, let `r_max` be the maximum local residual, `r_2` the second
largest local residual, `r_q95` its 95th percentile, and `s_max` the maximum
local score excess. Before opening a Phase 30 seed, compare exactly:

- `r_max`;
- `r_2`;
- `s_max`;
- `r_max - r_q95`;
- `r_max / r_q95`; and
- `r_2 - r_q95`.

All values are observed-coordinate statistics. Every candidate is high-tail
rejecting. Its cutoff is one binary64 step below the smallest candidate value
among the three Phase 28-route residual harmful rows in Phase 29 Validation A.
A candidate is eligible only if it removes all three residual rows and every
development panel/profile retains the unchanged Phase 28 safety gate. Among
eligible candidates, maximize retained focus accepts, then all-safe accepts,
then use the declared order above.

## Frozen development selection

The design-only comparison is:

| candidate | cutoff | focus retained | all-safe retained | passing panels | eligible |
|---|---:|---:|---:|---:|---|
| maximum local residual | 4.758551187362967 | 1201/1281 | 2920/3144 | 5/9 | no |
| second local residual | 3.0211373260137844 | 64/1281 | 112/3144 | 0/9 | no |
| maximum local score excess | 3.3041164903146822 | 1238/1281 | 3040/3144 | 9/9 | yes |
| isolated tail gap | 1.903291156395975 | 1235/1281 | 2997/3144 | 9/9 | yes |
| isolated tail ratio | 1.6636368999089541 | 1248/1281 | 3027/3144 | 9/9 | yes |
| second-residual tail gap | 0.10789311632299102 | 6/1281 | 11/3144 | 0/9 | no |

Freeze the selected feature

`isolated_tail_ratio = maximum_local_residual / percentile95_local_residual`

and accept only if the Phase 28 route accepts and

`isolated_tail_ratio < 1.6636368999089541`.

The limiting development value is `1.6636368999089544`; the strict cutoff is
its preceding binary64 value. Across all nine development panels the selected
route changes harm from original -> Phase 28 predecessor -> Phase 30 combined
as follows:

| panel | harm | focus, combined/original | all-safe, combined/original | gate |
|---|---:|---:|---:|---|
| score fit `24900804` | 168 -> 0 -> 0 | 129/132 | 348/360 | pass |
| cutoff calibration `25000804` | 165 -> 0 -> 0 | 124/126 | 343/351 | pass |
| local design A `27500804` | 168 -> 0 -> 0 | 115/123 | 343/360 | pass |
| local design B `27600804` | 165 -> 0 -> 0 | 131/132 | 349/360 | pass |
| local calibration `27800804` | 153 -> 0 -> 0 | 121/126 | 357/366 | pass |
| Phase 28 Validation A `28100804` | 171 -> 0 -> 0 | 129/129 | 346/360 | pass |
| Phase 28 Validation B `28200804` | 171 -> 0 -> 0 | 125/126 | 357/363 | pass |
| Phase 28 final `28300804` | 168 -> 0 -> 0 | 127/129 | 336/354 | pass |
| Phase 29 Validation A `30200804` | 57 -> 3 -> 0 | 247/258 | 248/270 | pass |

This is truth-supervised development evidence and cannot support Phase 30.

## Fresh case-seed-disjoint protocol

Keep the Phase 29 targeted family unchanged: N=96; control, local-bump, and 1%
outlier stresses; 64 repeats; exact, 0.5-degree registration, and 10%-missing
profiles; 2048 clean-reference points; and 256 endpoint surface samples. Each
panel contains 192 primary cases and 576 profile rows.

Freeze:

- Validation A: `30500804`;
- Validation B: `30600804`; and
- final held-out: `30700804`.

The case-seed audit finds zero mutual overlap and zero overlap with every full
panel base `20300804`--`25900804`, every base `27500804`--`28300804`, and the
targeted Phase 29 bases `30200804`--`30400804`. No replacement seed may be
tried if the fixed panels fail or contain no Phase 28 residual.

## Frozen gates and sequential opening

Every opened panel and profile must:

1. contain at least one original harmful false-safe;
2. reduce Phase 30 combined harmful false-safes to zero;
3. retain at least 90% of original-safe control/local-bump accepts; and
4. accept zero originally safe cases whose routed endpoint becomes harmful.

Validation B opens only after Validation A passes; final opens only after
Validation B passes. The separate incremental gate requires all three panels
to open and pass, at least one aggregate harmful residual accepted by the
frozen Phase 28 score/q95 predecessor, and zero residual harmful rows under the
Phase 30 combined route. No predecessor residual is an uninformative result,
not positive support.

## Design-only result

The evaluator reproduced the complete Phase 28 pass and Phase 29 failure,
selected the expected ratio and cutoff, passed all nine development gates, and
passed the case-seed audit. Zero Phase 30 fresh panels were opened and
`phase30_supported=false`, as required.

## Planned run

```powershell
python -m pftf_alpha.tail_sensitive_local_guard `
  --output benchmark-out/tail_sensitive_local_guard_phase30.json
```

Ruff, the diff check, and the complete regression suite must pass before base
`30500804` is opened.

## Claim boundary

Even a complete pass supports only a synthetic targeted transfer result for
the frozen matched-pair simulator. It does not establish real correspondence,
paired-scan, trimmed-reconstruction, or deployment support.

## Result

Ruff, the diff check, and all 284 tests passed before base `30500804` was
opened. The first deterministic run numerically passed, but post-run case
inspection found that a tail-rejected harmful original accept was serialized
with `guarded_decision=accept` even though its guarded-accept and harm booleans
were correctly false. This decision-label branch did not affect the feature,
cutoff, panel opening, endpoint, or gate result. It was corrected to emit the
existing fail-closed `unsupported_geometry_fail_closed` decision, covered by a
regression test, and followed by a complete 285-test pass. The same fixed
305/306/307 seeds were then rerun without changing any scientific rule.

All three panels pass:

| panel | harm, original -> Phase 28 -> Phase 30 | focus, combined/original | all-safe, combined/original | introduced harm | gate |
|---|---:|---:|---:|---:|---|
| Validation A `30500804` | 51 -> 0 -> 0 | 222/234 | 222/252 | 0 | pass |
| Validation B `30600804` | 63 -> 0 -> 0 | 228/246 | 231/273 | 0 | pass |
| final `30700804` | 57 -> 2 -> 0 | 247/255 | 247/264 | 0 | pass |
| aggregate | 171 -> 2 -> 0 | 697/735 | 700/789 | 0 | pass |

Validation A and B establish safety/utility transfer but contain no Phase 28
residual. The final panel is informative. Its two residual rows are the
registration and missing views of one `outliers_01`, N=96, repeat-44 primary
case with case seed `31341118`. Their frozen score/q95 values are respectively
`0.11947032`/`2.73412811` and `0.15492914`/`2.78282436`, so the Phase 28 route
accepts both. Their maximum residuals are `5.21987698` and `6.11333623`, giving
tail ratios `1.90915596` and `2.19680994`. Both exceed the frozen ratio cutoff
and are fail-closed. Each evaluation-only routed endpoint contains one harmful
outlier vertex incident to four harmful faces.

Thus all three safety gates pass, two fresh predecessor residual rows are
observed, both are rescued, and the combined residual count is zero.
`phase30_supported=true` and
`tail_sensitive_local_guard_synthetic_supported=true` for this preregistered
synthetic targeted protocol. No feature, cutoff, seed, profile, or gate was
changed after Validation A was opened.

Real correspondence, paired scans, trimmed reconstruction, and deployment
remain unsupported. Artifact:
`benchmark-out/tail_sensitive_local_guard_phase30.json`, SHA-256
`84e169e0749daffd7465a91f0b90048267c2fdf7a84d2c7464d14d69b794690d`.
