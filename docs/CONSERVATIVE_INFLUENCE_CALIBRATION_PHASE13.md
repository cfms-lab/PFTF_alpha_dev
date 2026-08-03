# Conservative influence calibration: Phase 13

## Frozen question

Phase 12 showed that its local insertion-influence representation had
post-hoc separating rectangles, but a rectangle selected from one calibration
cohort did not transfer with zero harmful false-safes. Phase 13 keeps the
representation unchanged and asks whether two independent calibration cohorts
can select a conservative boundary that transfers to a third, untouched final
held-out panel.

This remains a synthetic safety audit. It does not establish PFTF-SPD novelty,
trimmed-reconstruction safety, real-scan support, or deployment support.

## Frozen observed-only representation

Phase 13 reuses Phase 12 without changing any feature hyperparameter:

- same inferred shared-trend layers;
- same leave-one-out local quadratic insertion influence;
- same neighbourhood sizes `{12, 18, 24}`;
- same `1.4826 * MAD` and `0.04 * tangent_radius` scale floor; and
- same peak/support two-coordinate case evidence.

The route continues to use observed coordinates and inferred layer labels
only. Stress identity, injected-source labels, and clean references remain
evaluation-only.

## Frozen dual-cohort calibration rule

Run both full calibration cohorts before selecting a rectangle. Candidate peak
and support thresholds are positive infinity and the largest floating-point
value strictly below each harmful case coordinate in the union of both
cohorts. Search every rectangular accept region

`peak <= peak_threshold AND support <= support_threshold`.

Discard every rectangle that accepts any harmful case in either calibration
cohort. Rank the remaining rectangles deterministically by:

1. maximum of the lower safe control/local-bump retention across the two
   calibration cohorts;
2. maximum total retained safe control/local-bump cases;
3. maximum of the lower all-stress safe retention across the cohorts;
4. maximum total retained all-stress safe cases;
5. largest peak threshold; and
6. largest support threshold.

The calibration stage passes only if each cohort reproduces at least one
unguarded harmful false-safe, the selected rectangle leaves zero guarded
harmful false-safes in each cohort, and safe control/local-bump retention is at
least 90% in each cohort.

## Frozen three-panel protocol

Every panel uses the unchanged nine Phase-8 stresses, `N in {96,160,256}`,
eight repeats, 2048 clean-reference points, and 256 surface endpoint samples.

- Calibration cohort A seed: `21300804`.
- Calibration cohort B seed: `21400804`.
- Final held-out seed: `21500804`.
- The final held-out panel is not executed or inspected unless both
  calibration cohorts pass with the same selected rectangle.
- If calibration passes, the rectangle is frozen and evaluated exactly once on
  the final held-out seed. No final-held-out retuning is permitted.

Phase-12 calibration or held-out cases are not reused for threshold selection.

## Predeclared success gate

Phase 13 is supported only if all three full panels reproduce at least one
unguarded harmful-outlier false-safe, guarded harmful-outlier false-safe count
is zero on every panel, and safe control/local-bump retention is at least 90%
on every panel.

Even a pass permits only a later, separately predeclared
trimmed-reconstruction study. It does not itself support trimming, real scans,
or deployment.

## Planned run

```powershell
python -m pftf_alpha.conservative_influence_calibration `
  --output benchmark-out/conservative_influence_calibration_phase13.json
```

## Result

The implementation passed the full 202-test suite before any new panel was
opened. After the run, a strict-JSON serialization test was added and the final
full 203-test suite passed. The dual-cohort search selected:

- peak threshold: `0.44327071014047154`;
- support threshold: positive infinity.

Thus the most conservative dual-cohort solution reduced to a peak-only gate.
Both calibration cohorts passed, so the final held-out panel was executed
exactly once.

| Panel | Harmful false-safe, unguarded | Harmful false-safe, guarded | Clean/local-bump safe retention | All-stress safe retention | Gate |
|---|---:|---:|---:|---:|---|
| Calibration A, seed `21300804` | 56 | **0** | **40/44 = 90.91%** | 102/117 = 87.18% | pass |
| Calibration B, seed `21400804` | 49 | **0** | **39/43 = 90.70%** | 104/122 = 85.25% | pass |
| Final held-out, seed `21500804` | 55 | **1** | **42/44 = 95.45%** | 108/123 = 87.80% | fail |

The remaining final-held-out failure is the 96-point 1% outlier case at seed
`21920830`. Its peak and support influence scores are only
`0.2359290560051831` and `0.1371204515389834`, both well inside the frozen
accept region. One harmful source-2 vertex is used by four output faces;
cross-layer faces, component error, and Betti error are all zero.

The split provenance diagnostic reports accepted source-provenance violations
of 56 to 0 on calibration A, 52 to 3 on calibration B, and 55 to 1 on final
held-out. These remain diagnostic rather than routing labels.

Unlike Phase 12, there is no post-hoc rectangular rescue across all three
Phase-13 panels. Exhaustive search found zero peak/support rectangles that
simultaneously remove every harmful case and retain at least 90% of safe
control/local-bump accepts on each panel. Under zero harm, the best achievable
minimum retention is only `43.18%`: 19/44 on calibration A, 24/43 on
calibration B, and 20/44 on final held-out.

This is evidence of representation overlap for the Phase-12 peak/support
insertion-influence family on these panels, not merely an unstable calibration
tie-break. Another threshold or additional calibration cohort should not be
tried on the same representation. A defensible next step is an observed-data
identifiability audit of the low-influence harmful case against safe local
features before proposing any new guard.

`phase13_supported=false`. Thresholds were not retuned, and trimmed
reconstruction, real-scan validation, and deployment remain unsupported and
were not started.

Artifact: `benchmark-out/conservative_influence_calibration_phase13.json`.
