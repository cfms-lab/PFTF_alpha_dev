# Parallel two-layer constrained connectivity: Phase 2

## Scope

Phase 2 is a specialized parallel-sheet baseline, not a general replacement for
alpha complexes.  It is evaluated only after the observed-only sampling gate
identifies two layers.

## Frozen construction

1. Infer the dominant local-normal direction from observed points.
2. Run deterministic two-means on the normal coordinate.
3. Project each inferred layer independently into its PCA tangent plane.
4. Construct one 2D Delaunay triangulation per layer.
5. Return the union of both triangle sets, with no cross-layer face.

Construction and routing receive no true component labels or dense reference.
Those inputs are evaluation-only.

## Frozen held-out panel

- Seed: `20280802`, unseen by Phase 0/1/1b.
- Gaps: `{0.18, 0.40, 0.60, 0.80, 1.20}`.
- Eight seeds per gap, 96 observed points, 2048 reference points.
- Sampling gate: k=12, cross-kNN <= 0.05, cluster fraction >= 0.20,
  separation SNR >= 3.
- Baseline: frozen B5 multiplier `2.80293354289327`.
- Under-resolved or unsupported cases cannot be accepted.

Phase 2 passes only if, among sampling-sufficient cases:

1. every constrained output is truly bridge-free with correct component count;
2. safe-output acceptance coverage is at least 80%;
3. false-safe count is zero;
4. mean F-score is within 0.01 of or above B5;
5. Betti-error sum does not exceed B5; and
6. labeled bridge-edge sum is zero.

Even a pass remains synthetic and parallel-sheet-specific, so
`deployment_supported` stays false.

## Run

```powershell
python -m pftf_alpha.two_layer_connectivity `
  --output benchmark-out/two_layer_connectivity_phase2.json
```

## Result

The frozen held-out panel passed every Phase-2 gate.

| Gap | Sampling-sufficient | Accepted | True safe | False safe | B5 F | Constrained F | B5 bridge edges | Constrained bridge edges | B5 Betti error | Constrained Betti error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.18 | 0/8 | 0/8 | 6/8 | 0 | 0.6891 | 0.7128 | 270 | 21 | 35 | 0 |
| 0.40 | 0/8 | 0/8 | 3/8 | 0 | 0.5887 | 0.6336 | 262 | 189 | 54 | 0 |
| 0.60 | 0/8 | 0/8 | 8/8 | 0 | 0.5202 | 0.7401 | 333 | 0 | 80 | 0 |
| 0.80 | 8/8 | 8/8 | 8/8 | 0 | 0.3868 | 0.7513 | 714 | 0 | 164 | 0 |
| 1.20 | 8/8 | 8/8 | 8/8 | 0 | 0.3976 | 0.7615 | 486 | 0 | 216 | 0 |

Across the 16 sampling-sufficient cases:

- accepted: 16/16;
- true safe outputs: 16/16;
- false safe: 0;
- safe acceptance coverage: 1.00;
- mean F-score: B5 `0.3922` -> constrained `0.7564`;
- component-error sum: B5 `16` -> constrained `0`;
- labeled bridge-edge sum: B5 `1200` -> constrained `0`;
- Betti-error sum: B5 `380` -> constrained `0`.

The conservative sampling gate also behaved as intended.  It rejected every
gap-0.18/0.40/0.60 case, including the two unsafe gap-0.18 constructions and
all five unsafe gap-0.40 constructions.  It also rejected some genuinely safe
outputs, so its coverage outside the declared sufficient regime is deliberately
low rather than silently unsafe.

Artifact: `benchmark-out/two_layer_connectivity_phase2.json`.

`phase2_supported=true`, but `deployment_supported=false`.  This is the first
positive reconstruction result in the project, yet it depends on an explicit
parallel two-layer model, planar tangent projections, and synthetic data.  It
does not validate PFTF local SPD, solve general false bridges, or replace a
visibility-aware graph-cut method for arbitrary surfaces.
