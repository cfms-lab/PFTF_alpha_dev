# Curvature-guard domain shift: Phase 5

## Frozen question

Does the Phase-4b normal-coherence threshold `0.82` remain fail-closed when
point density, observation noise, and curvature shape change together? The
threshold and all Phase-1/2 routing settings remain frozen.

## Frozen panel

Seed `20300804`; 2048 reference points; 256 surface-evaluation samples.

Sampling profiles are the Cartesian product:

- observed point count `{96, 160, 256}`;
- isotropic observation noise `{0.005, 0.010, 0.025}`.

Geometries:

1. `paraboloid_024`: paired `0.24*(x^2+y^2)` sheets, gap 0.80.
2. `paraboloid_036`: paired `0.36*(x^2+y^2)` sheets, gap 0.80.
3. `asymmetric_converging`: lower curvature 0.36, upper curvature 0.12;
   gap decreases from 0.80 to 0.32 over the sampled square.
4. `saddle_024`: paired `0.24*(x^2-y^2)` sheets, gap 0.80.
5. `cylinder_036`: paired `0.36*x^2` sheets, gap 0.80.

Nine profiles x five geometries x eight unseen repeats = 360 cases. Profile,
geometry, true labels, and dense references are evaluation-only. Construction,
base routing, and the guard use observed coordinates only.

## Predeclared success gate

Phase 5 passes only if:

1. the unguarded router has at least one false-safe accept;
2. the guarded router has zero false-safe accepts across all 360 cases;
3. overall guarded safe-accept retention is at least 90%;
4. every sampling profile with at least four base-safe accepts retains at least
   75% of them; and
5. every false-safe accepted by the base router is removed.

This is a strict transfer test. A failure freezes the current `0.82` guard to
the Phase-4b generator and motivates a shape-aware model-adequacy test; the
threshold is not changed on this panel. `deployment_supported` remains false.

## Run

```powershell
python -m pftf_alpha.guard_domain_shift `
  --output benchmark-out/curvature_guard_domain_shift_phase5.json
```

## Result

The frozen domain-shift panel failed the transfer gate.

| Endpoint | Base router | Guarded router | Required |
|---|---:|---:|---:|
| Safe accepts | 186 | 147 | retention >=90% |
| Safe-accept retention | - | **79.03%** | >=90% |
| False-safe accepts | 57 | **12** | 0 |
| Removed false-safe accepts | - | 45/57 | 57/57 |

Geometry summaries:

| Geometry | Mean coherence | Base safe | Guarded safe | Retention | Base false safe | Guarded false safe |
|---|---:|---:|---:|---:|---:|---:|
| paraboloid_024 | 0.8360 | 58 | 50 | 86.2% | 0 | 0 |
| paraboloid_036 | 0.7396 | 19 | 0 | 0.0% | 31 | 0 |
| asymmetric_converging | 0.7295 | 1 | 0 | 0.0% | 26 | **12** |
| saddle_024 | 0.8444 | 55 | 49 | 89.1% | 0 | 0 |
| cylinder_036 | 0.8321 | 53 | 48 | 90.6% | 0 | 0 |

The 12 surviving false-safe cases are all high-density asymmetric-converging
cases: four at `n256/noise0.005`, six at `n256/noise0.010`, and two at
`n256/noise0.025`. Their coherence lies in `[0.8201, 0.8503]`, so the fixed
threshold accepts them.

The opposite failure occurs at 96 points. Mean coherence falls to roughly
0.706-0.723 across those profiles, and safe-accept retention is only 0-20%.
Thus normal coherence is strongly density-dependent: the same scalar threshold
is too conservative for sparse safe surfaces and too permissive for dense
asymmetric converging surfaces.

Artifact: `benchmark-out/curvature_guard_domain_shift_phase5.json`.

`phase5_supported=false` and `deployment_supported=false`. The Phase-4b result
remains valid only for its fixed generator and density. Retuning `0.82` cannot
solve both observed errors simultaneously. The next method must normalize for
sampling uncertainty and test local layer-order consistency or local gap
convergence, rather than relying on one global orientation-tensor eigenvalue.
