# Focus-envelope cutoff transfer: Phase 27

## Question

Phase 26 eliminated reconstruction-induced endpoint harm and preserved every
fresh focus-safe accept, but its final missing-profile harmful score
`0.19257350714226332` fell just below the frozen quarter-gap cutoff
`0.19784302031484602`. Because every opened panel retained a positive
harmful/focus score gap, can a stricter cutoff defined only from Phase 26 A/B
focus evidence transfer to untouched, case-seed-disjoint panels?

## Design evidence and information boundary

The reconstruction, feature signature, and ridge coefficients are unchanged.
Phase 27 uses:

- Phase 24/25 score-fit seed `24900804` and cutoff-calibration seed `25000804`
  only to reproduce the predecessor score;
- Phase 26 validation A `27500804` and validation B `27600804` as cutoff-design
  cohorts; and
- Phase 26 final `27700804` only as diagnosis that motivated this phase.

No opened Phase 26 panel can support Phase 27. In particular, the fact that the
new cutoff would reject the known final failure is not validation evidence.

The route may use only full-primary coordinates and their observed frozen
shared-trend partition, the upstream decision, presented retained-pair IDs and
repeat coordinates, and the frozen twelve-coordinate score. Clean references,
source labels, source outlier identities, focus/harm labels, and endpoint
metrics are design/evaluation only.

## Frozen cutoff rule

Let `F_AB` contain all original-safe accepted control/local-bump cases from
Phase 26 validation A and B, across exact, registration, and missing profiles.
Freeze

`cutoff = nextafter(max(score(F_AB)), +infinity)`.

The strict `< cutoff` acceptance test therefore retains the observed maximum
focus case while adding only one binary64 step of margin. The frozen values are:

- A/B focus count: 255;
- A/B maximum focus-safe score: `0.18181536333942855`;
- A/B routed harmful count: 330;
- A/B minimum routed-harmful score: `0.28460336155814553`;
- separation gap: `0.10278799821871698`;
- cutoff: `0.18181536333942858`; and
- binary64 increment: `2.7755575615628914e-17`.

The cutoff uses truth-supervised focus membership only in the declared design
cohorts. This is a synthetic calibration claim, not a deployable unlabeled
calibration procedure.

Under this cutoff, each A/B profile has zero guarded harm, 100% focus
retention, 120/120 all-stress safe retention, and zero newly introduced
endpoint harm. The older 249/250 design panels also retain their Phase 26 gate
results. No coefficient, feature, reconstruction rule, endpoint definition,
cutoff rule, profile, or gate may change after this document.

## Case-seed-disjoint fresh protocol

The fresh base seeds are:

- Validation A: `27800804`;
- Validation B: `27900804`, opened only if validation A passes; and
- Final held-out: `28000804`, opened only if validation B passes.

Using

`case_seed = base + count_index*1000003 + stress_index*100003 + repeat*10007`,

their 216-case sets are mutually disjoint and have zero overlap with every
standard full-panel case set used through base `25600804` and the Phase 26 sets
from bases `27500804`--`27700804`.

Each panel keeps the unchanged nine stresses, `N in {96,160,256}`, eight
repeats, 2048 clean-reference points, 256 endpoint surface samples, and three
profiles, for 648 audited rows from 216 primary cases.

## Frozen gates

Before validation A is opened, the evaluator must reproduce the predecessor
model, Phase 26 A/B score envelope, new cutoff, case-seed audit, and all design
profile gates. Every profile in every opened fresh panel must:

1. contain at least one original unguarded harmful false-safe;
2. reduce routed harmful false-safes to zero;
3. retain at least 90% of original-safe control/local-bump accepts; and
4. accept zero originally safe cases whose routed endpoint becomes harmful.

Validation B opens only after validation A passes; final opens only after
validation B passes. A failure closes all later panels. No opened seed may be
used to alter the cutoff.

Even a complete pass supports only synthetic frozen-partition reconstruction
with a truth-supervised design cutoff under the presented-pair simulator.
`real_correspondence_supported`, `real_paired_scan_supported`,
`real_trimmed_reconstruction_supported`, and `deployment_supported` remain
false.

## Planned run

```powershell
python -m pftf_alpha.focus_envelope_cutoff `
  --output benchmark-out/focus_envelope_cutoff_phase27.json
```

## Result

The implementation and design-only reproduction passed before any fresh panel
was opened. Ruff, the diff check, and all 266 tests then passed; pytest emitted
only the existing Windows `.pytest_cache` cleanup warning.

The frozen design reproduced exactly:

- predecessor cutoff `0.19784302031484602`;
- A/B focus maximum `0.18181536333942855`;
- A/B routed-harm minimum `0.28460336155814553`;
- focus/harm gap `0.10278799821871698`;
- Phase 27 cutoff `0.18181536333942858`; and
- zero case-seed overlap in every audited comparison.

All four design panels passed. Validation A was then opened and produced:

- harmful false-safes: 153 -> 1;
- focus-safe accepts: 126 -> 125 (`99.21%`);
- all-stress safe accepts: 366 -> 362;
- introduced routed endpoint harm: 0; and
- panel gate: fail.

The exact profile passed with harm 51 -> 0, focus 42/42, and all-safe 121/122.
Registration passed with harm 51 -> 0, focus 41/42, and all-safe 120/122. The
missing profile failed with harm 51 -> 1 despite focus 42/42 and all-safe
121/122. Consequently Validation B and the final held-out panel remained
unopened, as preregistered.

The limiting case is `missing_10pct/outliers_01`, N=96, repeat 1, case seed
`28210823`, replicate seed `528210832`, and perturbation seed `930210836`.
One harmful outlier remains in six routed faces. Its score
`0.16754805218690128` is `0.01426731115252772` below the frozen cutoff, so the
guard accepts it.

This is a transfer failure of the Phase-26 A/B focus envelope. In Validation A,
the minimum harmful score is `0.16754805218690128` while the maximum focus-safe
score is `0.1970292265872296`, giving a negative ranking gap
`-0.0294811744003283`. Two of 126 focus-safe cases score at or above the
limiting harm (`0.17923386190820076` and `0.1970292265872296`). Thus the newly
opened panel no longer preserves the positive harmful/focus ordering observed
in the design cohorts.

A post-hoc cutoff at or below the limiting harmful score would still retain
124/126 focus cases overall, but this observation is diagnosis only. Changing
the cutoff after opening seed `27800804` would violate the frozen protocol and
cannot support Phase 27. No coefficient, feature, reconstruction rule, cutoff,
profile, or gate was retuned.

`phase27_supported=false` and
`focus_envelope_cutoff_synthetic_supported=false`. Base seed `27800804` is
opened; reserved bases `27900804` and `28000804` were not opened. None of these
three bases may be reused as fresh evidence. Real correspondence, paired scans,
trimmed reconstruction, and deployment remain unsupported.

Artifact: `benchmark-out/focus_envelope_cutoff_phase27.json`, SHA-256
`f44580020bb552fbec2d1288420d62c3a8faa90eb91f06e1bb6337e954e15c83`.
