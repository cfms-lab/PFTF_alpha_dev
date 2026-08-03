# Matched-pair correspondence stress: Phase 18

## Frozen question

Phase 17 passed only when every primary/repeat pair ID was correct, complete,
and perfectly registered. Does the same matched-displacement representation
retain its zero-harm and safe-focus performance under small, explicitly
declared correspondence and registration defects?

Phase 17's threshold is not reused or retuned. Phase 18 uses fresh primary
cohorts and selects one common rectangle across all stress profiles before a
conditional final held-out panel.

## Frozen base observation model

Primary cases, matched-repeat simulation, independent observation noise, and
independent repeat transient returns are unchanged from Phase 17. The repeat
seed remains

`primary_case_seed + 500000009`.

For each profile, a deterministic perturbation seed is

`primary_case_seed + 900000007 + profile_index * 1000003`.

The primary reconstruction and geometry/topology endpoint are computed once
per case and are identical across profiles. Only the evidence presented to the
matched-pair guard changes.

## Frozen perturbation profiles

Each primary case is evaluated under all five profiles:

1. `exact`: no additional perturbation;
2. `registration_0p5deg`: rotate the repeat cloud by exactly 0.5 degrees about
   its centroid and a deterministic random unit axis;
3. `missing_10pct`: delete `round(0.10 * N)` deterministic random pair IDs from
   both ordered arrays;
4. `mismatch_02`: select `round(0.02 * N)` pair IDs, with a minimum of two,
   and cyclically permute their repeat assignments; and
5. `combined`: apply the 0.5-degree rotation, then the 2% cyclic mismatch, then
   the 10% pair deletion.

No profile uses source labels, stress identity, clean references, or endpoints.
The mismatch is hidden from the guard: it trusts the presented pair order.
Missing IDs are simply absent. Rotation is not corrected before scoring.

Each artifact row records the retained/missing/mismatched pair counts, rotation
axis and angle, and a SHA-256 digest of the presented pair map.

## Frozen score

Use the unchanged Phase-17 robust matched-displacement score on the presented
pairs:

- median-center displacement independently by coordinate axis;
- scale each axis by the maximum of `1.4826 * MAD`, `0.002 * L_obs`, and
  machine epsilon; and
- retain the largest `peak` and second-largest `support` standardized
  displacements.

No scale, profile severity, operation order, or score may change after either
Phase-18 calibration cohort is observed.

## Frozen profile-aware dual calibration

Use one common rectangle

`peak <= peak_threshold AND support <= support_threshold`.

Candidate thresholds remain values immediately below harmful coordinates plus
positive infinity. Discard every rectangle accepting any harmful case in any
profile or calibration cohort. Rank the remaining rectangles by:

1. maximum worst retention across the ten calibration-cohort/profile
   control/local-bump groups;
2. maximum total retained focus-safe cases;
3. maximum worst retention across the ten all-safe groups;
4. maximum total retained all-safe cases;
5. largest peak threshold; and
6. largest support threshold.

Every profile in both cohorts must reproduce at least one unguarded harmful
false-safe, reduce guarded harmful false-safes to zero, and retain at least 90%
of safe control/local-bump accepts before final held-out is opened.

## Frozen three-panel protocol

Each primary panel uses the unchanged nine Phase-8 stresses,
`N in {96,160,256}`, eight repeats, 2048 clean-reference points, and 256 surface
endpoint samples. The five evidence profiles produce 1080 audited rows per
panel from 216 primary reconstructions.

- Calibration cohort A seed: `22700804`.
- Calibration cohort B seed: `22800804`.
- Conditional final held-out seed: `22900804`.
- Final held-out is not executed unless every A/B profile passes with the same
  rectangle.
- Final-held-out retuning is forbidden.

## Predeclared success gate

Phase 18 is supported only if every profile in all three panels has at least
one unguarded harmful false-safe, zero guarded harmful false-safes, and at least
90% safe control/local-bump retention.

Even a pass supports only the declared synthetic perturbation magnitudes. It
does not establish a real correspondence algorithm, scanner registration,
trimmed reconstruction, or deployment.

## Planned run

```powershell
python -m pftf_alpha.matched_pair_stress `
  --output benchmark-out/matched_pair_stress_phase18.json
```

## Result

The implementation tests and full regression suite passed before the frozen
panels were executed. The profile-aware dual-cohort optimizer selected

`peak <= 10.922625244331805 AND support <= infinity`.

The serialized peak threshold is the immediately preceding binary64 value
below the closest harmful calibration peak, `10.922625244331806`, which was a
calibration-B `registration_0p5deg` case.

Both panels preserved safety: every one of the 275 calibration-A and 280
calibration-B unguarded harmful false-safes across the five profiles was
reduced to zero. However, the profile gates separated sharply:

| Profile | Calibration A focus retention | Calibration B focus retention | Gate |
|---|---:|---:|---|
| `exact` | 43/43 (`100%`) | 43/43 (`100%`) | pass |
| `registration_0p5deg` | 43/43 (`100%`) | 43/43 (`100%`) | pass |
| `missing_10pct` | 43/43 (`100%`) | 43/43 (`100%`) | pass |
| `mismatch_02` | 0/43 (`0%`) | 0/43 (`0%`) | fail |
| `combined` | 0/43 (`0%`) | 1/43 (`2.33%`) | fail |

Thus both calibration panels failed the predeclared all-profile gate and final
held-out seed `22900804` was not opened. Thresholds and perturbation severities
were not retuned.

The failure is not caused by registration or missing pairs at the declared
magnitudes. Across both cohorts, the largest safe-focus peak was `5.1808055`
for the first three profiles. In contrast, the smallest `mismatch_02`
safe-focus peak was `14.7655707` in calibration A and `54.4210251` in
calibration B; median peaks were `147.5983` and `156.0771`. A cyclic mismatch
therefore looks like a large physical displacement to the unchanged robust
score. The same peak/support representation cannot distinguish a transient
surface return from an incorrect externally asserted pair while retaining the
declared safe focus cases.

Therefore `phase18_supported=false` and
`correspondence_stress_synthetic_supported=false`. The exact-ID Phase-17 result
remains an upper bound, while `real_correspondence_supported=false`,
`real_paired_scan_supported=false`, `trimmed_reconstruction_supported=false`,
and `deployment_supported=false`. A next phase, if pursued, must add an
explicit correspondence-confidence or robust pair-assignment mechanism and a
new preregistered protocol; it must not reuse this final seed or retune this
rectangle.

The reproducible artifact is
`benchmark-out/matched_pair_stress_phase18.json` (gitignored, 6,518,448 bytes,
SHA-256
`7458347ec62916634f4c4ecac53c027ebc5da686d07f84e491394444ce70092e`).
