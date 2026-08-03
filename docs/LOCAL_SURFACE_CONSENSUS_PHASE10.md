# Local surface consensus: Phase 10

## Frozen question

Phase 9 conflated every source-2 triangle with an unsafe reconstruction and its
shared-quadratic residual guard rejected coherent local bumps. Phase 10 asks a
narrower question: can an observed-only local surface-consensus guard reject
geometrically harmful contamination while retaining coherent clean and
localized-bump surfaces?

This is a synthetic safety audit. It does not establish PFTF-SPD novelty,
real-scan support, trimming safety, or deployment support.

## Frozen observed-only guard

For each point, use its inferred shared-trend layer and exclude the point from
the fit. Select up to 12 nearest same-layer neighbours, fit their PCA tangent
plane, and measure the omitted point's orthogonal residual. Normalize by the
larger of:

1. `1.4826 * MAD` of neighbour plane residuals; and
2. `0.04 *` the median neighbour tangent radius.

A case fails closed when any point has standardized leave-one-out residual
above `5.0`. The route never sees the stress identity, injected-source labels,
or dense clean reference.

These values are fixed before the Phase-10 seed is evaluated. They will not be
retuned after inspecting that panel.

## Split evaluation endpoints

The evaluation-only source labels and clean reference report two different
facts:

- **source-provenance violation:** any output triangle uses a source-2 point;
- **geometry/topology harm:** an output face uses a source-2 point farther than
  `0.025 * characteristic_length` from the clean generating reference, a face
  directly mixes clean layers 0 and 1, or the mesh differs from the expected
  two disk components in connected-component or GF(2) Betti counts.

Thus a near-surface source-2 point may remain a provenance violation without
being silently promoted to a geometry/topology failure. Neither endpoint is
available to the router.

## Frozen held-out protocol

- Phase-8 stress families and point counts remain unchanged: nine stresses,
  `N in {96,160,256}`, eight repeats, 216 cases.
- New seed: `20800804`, unseen by Phases 0-9.
- Clean reference count: 2048; surface endpoint samples: 256.
- Phase-7 shared-trend reconstruction and sampling gate remain unchanged.
- Phase 10 adds rejection only; it does not remove points or retriangulate.

## Predeclared success gate

Phase 10 passes only if:

1. the unguarded route reproduces at least one harmful-outlier false-safe;
2. guarded harmful-outlier false-safe count is exactly zero; and
3. safe-accept retention over the union of clean-control and local-bump cases
   is at least 90%.

Source-provenance violation accepts are reported but are not silently equated
with harmful false-safes. Even a pass only permits the next research phase to
test trimmed reconstruction and real scans; it does not itself support either.

## Run

```powershell
python -m pftf_alpha.local_surface_consensus `
  --output benchmark-out/local_surface_consensus_phase10.json
```

## Result

The frozen panel **did not pass**. Thresholds were not adjusted after the run.

| Metric | Unguarded | Guarded | Required |
|---|---:|---:|---:|
| Harmful-outlier false-safe | 55 | **1** | 0 |
| Source-provenance violation accepts | 57 | **1** | diagnostic only |
| Clean/local-bump safe accepts | 39 | **23** | retention >=90% |
| Clean/local-bump retention | - | **58.97%** | >=90% |

The one remaining harmful false-safe is a 96-point, 1% outlier case at seed
`21250851`. Its maximum standardized leave-one-out residual is `4.3593`, below
the frozen `5.0` cutoff. One harmful source-2 vertex is used by four output
faces; connected-component and Betti errors remain zero, so the split endpoint
correctly identifies geometric face harm without requiring a topology change.

The over-rejection is not confined to the shared-quadratic mismatch that Phase
10 was intended to avoid. It retains 15/17 safe control accepts but only 8/22
safe local-bump accepts. It also rejects most anisotropic-noise and every
sinusoidal accept, so a maximum single-point local-plane residual is not a
shape-agnostic certificate.

There is no scalar threshold rescue on this frozen score. Eliminating all 55
harmful cases would require a cutoff below the minimum harmful score `4.3593`,
which retains only 15/39 (`38.46%`) clean/local-bump safe accepts. Retaining at
least 90% requires a cutoff of at least `7.4204`, which leaves two harmful cases.
This overlap is reported as a representation failure, not a threshold-tuning
opportunity.

`phase10_supported=false`. In accordance with the predeclared sequence,
trimmed reconstruction, real-scan validation, and deployment remain
unsupported and were not started.

Artifact: `benchmark-out/local_surface_consensus_phase10.json`.
