# Local insertion influence: Phase 12

## Frozen question

Phase 11 improved clean/local-bump retention, but its scalar maximum-residual
boundary did not transfer with zero harmful false-safes. Phase 12 does not
retune that score. It asks whether a set-valued local influence representation
can distinguish points that merely depart from a local fit from points that
materially change the fitted neighbourhood.

This remains a synthetic safety audit. It does not establish PFTF-SPD novelty,
trimmed-reconstruction safety, real-scan support, or deployment support.

## Frozen observed-only influence representation

For each point, use only its inferred shared-trend layer and exclude the point
from an initial fit. At each feasible same-layer neighbourhood size in
`{12, 18, 24}`:

1. estimate a PCA tangent frame from the neighbours;
2. fit the Phase-11 quadratic height model to the neighbours;
3. append the omitted point in that fixed tangent frame and refit the same
   quadratic model;
4. evaluate both fits at every original neighbour; and
5. divide the RMS change in predicted neighbour height by the larger of
   `1.4826 * MAD` of the baseline fit residuals and `0.04 *` the median
   neighbour tangent radius.

The per-point influence score is the minimum standardized prediction shift
over the feasible neighbourhood sizes. Each case retains the descending
point-score set. The router uses the largest score `peak` and second-largest
score `support`; it does not collapse them to the Phase-11 residual score.

Stress identity, injected-source labels, and the dense clean reference remain
evaluation-only and are never inputs to the influence calculation.

## Frozen two-dimensional calibration rule

The Phase-10 geometry/topology harm endpoint remains unchanged. Calibration
considers only cases accepted by the unchanged Phase-7 sampling gate.

The accept region is the rectangle

`peak <= peak_threshold AND support <= support_threshold`.

Candidate values for each threshold are positive infinity and the largest
floating-point value strictly below each harmful calibration case's value on
that coordinate. Search every candidate pair and retain only pairs that accept
zero harmful calibration cases. Select deterministically by:

1. maximum retained safe control/local-bump accepts;
2. then maximum retained safe accepts over all stress families;
3. then the largest peak threshold; and
4. then the largest support threshold.

If calibration contains no accepted harmful case, no valid relevance check is
possible and calibration fails. Calibration passes only when its selected
rectangle removes every harmful false-safe and retains at least 90% of safe
control/local-bump accepts.

## Frozen two-panel protocol

Both panels use the unchanged nine Phase-8 stresses, `N in {96,160,256}`, eight
repeats, 2048 clean-reference points, and 256 surface endpoint samples.

- Calibration seed: `21100804`.
- Held-out seed: `21200804`.
- The held-out panel is not executed or inspected unless calibration passes.
- If calibration passes, both thresholds are frozen and evaluated exactly once
  on the held-out seed. No held-out retuning is permitted.

## Predeclared success gate

Phase 12 is supported only if both full panels reproduce at least one
unguarded harmful-outlier false-safe, guarded harmful-outlier false-safe count
is zero on both, and safe control/local-bump retention is at least 90% on both.

Even a pass permits only a later, separately predeclared
trimmed-reconstruction study. It does not itself support trimming, real scans,
or deployment.

## Planned run

```powershell
python -m pftf_alpha.local_insertion_influence `
  --output benchmark-out/local_insertion_influence_phase12.json
```

## Result

The implementation passed the full 198-test suite before either frozen panel
was opened. Calibration selected:

- peak threshold: `0.6715108036751242`;
- support threshold: `0.5445668538984977`.

Calibration passed, so the held-out panel was executed exactly once.

| Panel | Harmful false-safe, unguarded | Harmful false-safe, guarded | Clean/local-bump safe retention | All-stress safe retention | Gate |
|---|---:|---:|---:|---:|---|
| Calibration, seed `21100804` | 52 | **0** | **43/43 = 100%** | 114/116 = 98.28% | pass |
| Held-out, seed `21200804` | 52 | **1** | **42/42 = 100%** | 115/121 = 95.04% | fail |

The remaining held-out failure is the 160-point 3% outlier case at seed
`22770871`. Its peak and support scores are `0.5046093406788189` and
`0.4096165294090569`, both below the frozen thresholds. Four harmful source-2
vertices are used by 22 output faces; cross-layer faces, component error, and
Betti error are all zero.

The split provenance diagnostic reports accepted source-provenance violations
of 52 to 0 on calibration and 53 to 2 on held-out. The fixed rectangle rejects
no safe control/local-bump cases. Its other safe rejections are two calibration
sinusoidal cases and, on held-out, four sinusoidal plus two anisotropic-noise
cases.

A post-hoc exhaustive diagnostic found 98 rectangles that would remove all
harm on the union of both panels while retaining at least 90% of safe
control/local-bump accepts on each. For example, keeping the peak boundary and
placing the support boundary strictly below `0.394703787788857` would retain
43/43 calibration and 41/42 held-out focus-safe accepts. This does **not**
rescue Phase 12: that boundary was identified after opening held-out and cannot
replace the frozen calibration result.

The important negative is therefore calibration-margin transfer, not a proof
that the influence representation is inseparable. A future phase would need a
predeclared conservative margin or nested calibration procedure using new
seeds. It must not tune on this held-out panel.

`phase12_supported=false`. Thresholds were not retuned, and trimmed
reconstruction, real-scan validation, and deployment remain unsupported and
were not started.

Artifact: `benchmark-out/local_insertion_influence_phase12.json`.
