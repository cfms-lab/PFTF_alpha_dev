# Density-normalized local-order guard: Phase 6

## Calibration

Phase 5 is calibration-only. For each point, the guard projects coordinates to
the global normal/tangent frame inferred from observed kNN PCA normals. It finds
the nearest opposite inferred-layer point in tangent coordinates and divides
their normal-coordinate gap by the point's nearest same-layer tangent spacing.
The case score is

\[
m_{order}=\frac{Q_{0.05}(\Delta n_i/s_i)}{\sqrt{N}}.
\]

This normalization removes the leading density dependence of raw gap/spacing.
The guard combines this local-order margin with normal coherence. A small,
predeclared density bin is used because the k=12 normal estimate retains a
finite-sample bias:

| Observed N | Minimum coherence | Minimum local-order margin |
|---:|---:|---:|
| <=96 | 0.75 | 0.150 |
| 97-160 | 0.80 | 0.195 |
| >160 | 0.80 | 0.220 |

These settings maximize safe-accept retention subject to zero false-safe accepts
on Phase-5 calibration. They retain 163/186 calibration safe accepts (87.6%).
No threshold is adjusted on Phase 6.

## Frozen held-out

- Seed `20400804`, unseen by Phases 0-5.
- Same 9 density/noise profiles and 5 curvature shapes as Phase 5.
- Eight repeats per cell: 360 cases total.
- 2048 reference points and 256 evaluation surface samples.

Only observed coordinates and inferred labels enter the guard. Profile names,
shape names, true labels, and references are evaluation-only.

## Predeclared success gate

Phase 6 passes only if:

1. the base router reproduces at least one false-safe accept;
2. the guarded router has zero false-safe accepts;
3. every base false-safe accept is removed;
4. overall safe-accept retention is at least 85%; and
5. each point-count group with at least eight base-safe accepts retains at least
   75% of them.

Even a pass remains synthetic and non-deployable.

## Run

```powershell
python -m pftf_alpha.local_order_guard `
  --output benchmark-out/local_order_guard_phase6.json
```

## Result

The frozen held-out **did not pass**:

| Metric | Base router | Local-order guard | Required |
|---|---:|---:|---:|
| Safe accepts | 190 | 174 | - |
| Safe-accept retention | - | **91.58%** | >=85% |
| False-safe accepts | 61 | **2** | 0 |
| Removed false-safe accepts | - | 59/61 | 61/61 |

The density-stratified retention gates all passed:

| Observed N | Base safe | Guarded safe | Retention | Base false-safe | Guarded false-safe |
|---:|---:|---:|---:|---:|---:|
| 96 | 31 | 30 | 96.77% | 2 | **2** |
| 160 | 80 | 72 | 90.00% | 18 | 0 |
| 256 | 79 | 72 | 91.14% | 41 | 0 |

Both survivors are sparse 96-point cases: one `paraboloid_024` case at noise
`0.005` and one `paraboloid_036` case at noise `0.025`. Their observed
coherence/order pairs are `(0.8000, 0.2110)` and `(0.7544, 0.2436)`, so both
clear the frozen sparse thresholds `(0.75, 0.150)`.

`phase6_supported=false` because criteria 2 and 3 fail. The result still shows
that density normalization repairs the Phase-5 retention failure and removes
all 59 false-safe cases at N=160/256, but it is not a safety certificate. The
thresholds were not retuned after held-out inspection. A next experiment must
add a genuinely local convergence/order feature for sparse surfaces or change
the reconstruction route; tightening this seed's sparse threshold would be
post-hoc overfitting.

Artifact:
`benchmark-out/local_order_guard_phase6.json`.
