# Phase 42: reserved anchor-relative validation protocol

## Frozen route

The selected route first applies Phase 41 (`support >= 2`, dispersion <=0.15 m)
and then accepts a target-only cell only when:

- nearest anchor-cell distance <=1.50 m;
- anchor local plane residual <=0.50 m;
- unsigned 12-NN PCA normal alignment >=0.75.

The global alpha remains 1.00 m. Calibration artifact SHA-256:
`ddf119166d119e376acc21b0d73ba078616d6356a9a0f2093944dd7b1fb2f16f`.

## Reserved panel

Development sources 0 and 17, all 17 Phase-40 sources, and all seven Phase-41
sources are excluded. Among remaining sources, the minimum-three-direct-pair
rule freezes:

| Source | Direct pairs | p90 accepted |
|---:|---:|---:|
| 25 | 5 | 5 |
| 26 | 4 | 4 |
| 27 | 3 | 3 |

These 12 predictions are all p90-accepted. None of the three source heldout
reconstruction endpoints was evaluated before this protocol was materialized.

## Frozen comparison and gate

Each source produces three meshes with matched sampling:

1. anchor-only fallback;
2. frozen Phase-41 local-support route;
3. selected Phase-42 anchor-relative route.

Support requires all nine meshes to materialize, every case to add at least one
Phase-42 cell, mean Phase-42 geometry loss below both baseline means, mean
F-score no more than 0.025 below anchor, and mean recall no more than 0.01 below
anchor. Topology remains descriptive.

Protocol artifact SHA-256:
`1b304d0c62251e1f572ab65295f5903b29e820ee1c9557edf8b6c54424b3efac`.

The evaluator may open exactly the 32 Hokuyo members after the protocol commit.
It must not consume previous endpoint panels, the Leica pose member, or any
registration correctness label.

The panel is small and late-sequence. Even a positive result supports only a
bounded anchor-relative fixed-alpha shadow, not a general local alpha field,
correct topology, deployed trimming, or deployment.
