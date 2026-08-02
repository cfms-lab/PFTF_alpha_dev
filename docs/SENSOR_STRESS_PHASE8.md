# Sensor-style shared-trend stress: Phase 8

## Frozen question

Phase 7 passed on balanced, complete, clean samples whose five shapes were well
matched by a shared quadratic trend. Phase 8 changes no algorithm or threshold.
It asks whether that frozen candidate transfers to elementary sensor-like
sampling defects and shared surfaces outside the quadratic family.

## Frozen panel

For N in `{96,160,256}`, evaluate eight new seeds under each stress:

1. clean quadratic control;
2. upper-layer one-sided occlusion (`x >= -0.20`);
3. 75:25 layer sampling imbalance;
4. anisotropic noise (`sigma_xy=0.006`, `sigma_z=0.040`);
5. 1% spatial outliers;
6. 3% spatial outliers;
7. 5% spatial outliers;
8. shared sinusoidal surface; and
9. shared localized Gaussian bump.

This gives 216 cases at seed `20600804`, with 2048 clean reference points and
256 surface samples. Outliers receive a third evaluation-only source label, so
any accepted triangle connecting them to a surface counts as unsafe. The route
does not see stress names, source labels, or references.

## Predeclared success gate

Phase 8 passes only if:

1. the global-normal base reproduces at least one false-safe accept;
2. shared-trend inference has zero false-safe accepts over all 216 cases;
3. each non-outlier stress has at least 75% safe-accept coverage;
4. the combined sinusoidal/bump subset has at least 75% safe-accept coverage;
5. every outlier group has zero false-safe accepts; and
6. each point-count group has zero false-safe accepts and at least 75%
   non-outlier safe-accept coverage.

Even a pass remains synthetic and non-deployable. Failure is expected to expose
the next missing policy feature; no Phase-8 result may be used to retune the
already inspected panel.

## Run

```powershell
python -m pftf_alpha.sensor_stress `
  --output benchmark-out/sensor_stress_phase8.json
```

## Result

The frozen panel **did not pass**:

| Metric | Result | Required |
|---|---:|---:|
| Candidate false-safe | **56** | 0 |
| Non-outlier safe-accept coverage | 117/144 = **81.25%** | >=75% |
| Nonquadratic safe-accept coverage | 40/48 = **83.33%** | >=75% |
| Base false-safe reproduced | 63 | >0 |

The aggregate coverage values hide a sharp density boundary:

| Observed N | Non-outlier safe accepts | Coverage | Candidate false-safe |
|---:|---:|---:|---:|
| 96 | 21/48 | **43.75%** | 8 |
| 160 | 48/48 | **100%** | 24 |
| 256 | 48/48 | **100%** | 24 |

At N=160 and N=256, every non-outlier control, occlusion, imbalance,
anisotropic-noise, sinusoidal, and local-bump case was accepted safely. The
sparse N=96 sampling gate rejected all imbalance cases, seven of eight
sinusoidal cases, and most noisy/control cases. This is conservative rather
than unsafe, but it violates the per-density coverage gate.

All 56 candidate false-safe accepts are outlier cases: 18/24 at 1%, 19/24 at
3%, and 19/24 at 5%. The remaining sparse outlier cases fail closed as
`rescan_required`; every N=160/256 outlier case is silently accepted. Residual
trend RMSE increases with contamination, but it overlaps clean/noisy cases and
was not a frozen routing feature.

`phase8_supported=false`. The useful bounded result is a 100%-safe accepted
envelope for the tested non-outlier sensor stresses at N>=160. The decisive
missing capability is an observed-only outlier policy. Phase 8 must not be
retuned. A new calibration/held-out phase may add robust residual leverage or
trimmed inlier consensus, while preserving fail-closed routing.

Artifact:
`benchmark-out/sensor_stress_phase8.json`.
