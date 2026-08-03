# Phase 42: anchor-relative validation result

## Outcome

The preregistered three-source validation is complete and negative. All routes
and meshes execute, and the frozen F-score and recall tolerances pass, but the
anchor-relative route does not improve geometry over either baseline.

Across 12 p90-accepted direct predictions, Phase 41 supplies 648 candidate
target-only cells. The anchor-relative observation retains 135 and rejects 513.

| Mean endpoint | Anchor only | Phase 41 | Phase 42 |
|---|---:|---:|---:|
| Geometry loss | 0.147269 | 0.143611 | 0.148807 |
| F-score | 0.656327 | 0.583448 | 0.640541 |
| Recall | 0.684616 | 0.647052 | 0.676635 |
| Connected components | 14.0000 | 13.3333 | 13.3333 |
| Betti-1 | 12.6667 | 21.0000 | 14.0000 |
| Nonmanifold-edge fraction | 0.022023 | 0.025103 | 0.024931 |

Phase 42 restores much of the F-score and recall lost by Phase 41 and reduces
its Betti-1 inflation. Nevertheless, its geometry loss is 0.001539 above anchor
and 0.005197 above Phase 41. It beats anchor geometry in 0/3 cases and Phase-41
geometry in only 1/3.

Therefore `anchor_relative_shadow_supported=false`.

## Casewise result

| Source | Added cells | Anchor geometry | Phase-41 geometry | Phase-42 geometry |
|---:|---:|---:|---:|---:|
| 25 | 46 | 0.124855 | 0.127014 | 0.125896 |
| 26 | 46 | 0.174221 | 0.158908 | 0.174934 |
| 27 | 43 | 0.142730 | 0.144909 | 0.145592 |

Source 25 is the sole geometry improvement over Phase 41 but remains worse than
anchor. Source 26 shows the largest loss of useful Phase-41 geometry. Source 27
is worse than both baselines.

## Protocol integrity

The development grid was committed as `6c04859`. Calibration then selected
`anchor_d150_p050_n075`. Commit `a14474f` froze sources 25-27, the artifact
identities, the comparison, and all gates before their endpoints were opened.

- Calibration SHA-256:
  `ddf119166d119e376acc21b0d73ba078616d6356a9a0f2093944dd7b1fb2f16f`
- Validation protocol SHA-256:
  `1b304d0c62251e1f572ab65295f5903b29e820ee1c9557edf8b6c54424b3efac`
- Result SHA-256:
  `f23e441bbbfa29bcb6cc54eac1bb21a813ec0b25a3248760cf5f3a6526dd0aa8`

A second execution reproduces it byte for byte. Previous validation endpoints
and registration correctness labels were not accessed; exactly the 32 Hokuyo
members were opened.

## Interpretation

Nearest-anchor distance, anchor-plane residual, and local PCA-normal alignment
are more informative than target-to-target support alone: they recover coverage
and suppress much of the Phase-41 topology inflation. They still do not identify
which geometrically plausible cells improve the fixed-alpha surface endpoint.
A binary point gate can move between the anchor and Phase-41 trade-offs without
dominating both.

The small late-sequence panel is also a hard scope limit. It has only three
sources and 12 direct predictions and cannot support a general method claim.

Keep:

- `anchor_relative_shadow_supported=false`;
- `point_local_alpha_field_supported=false`;
- `topology_correctness_supported=false`;
- `real_trimmed_reconstruction_supported=false`;
- `deployment_supported=false`.

## Next-step boundary

Do not tune further on Gazebo. All source endpoints with at least three direct
predictions have now entered development or validation. The next scientifically
distinct step must freeze a new panel before method development, preferably one
with a reference surface or known simulated ground truth. It should test a
continuous confidence-to-alpha field or confidence-weighted reconstruction,
not another binary cell-deletion threshold. This returns the work to the core
alpha-selection question and makes geometry/topology correctness identifiable.
