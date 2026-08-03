# Phase 41: point-local multi-scan support validation result

## Outcome

The preregistered seven-source validation is complete and negative. The selected
`support02_dispersion0150mm` route is exercised in every case, but it does not
clear the frozen aggregate geometry, F-score, or recall requirements.

Across 84 p90-accepted direct scan inputs, the local observation adds 1,162
corroborated target-only cells and fails closed on 23,696 target-only cells. All
21 anchor, scan-fused, and local-support alpha meshes materialize.

| Mean endpoint | Anchor only | Scan fused | Local support |
|---|---:|---:|---:|
| Geometry loss | 0.124651 | 0.174167 | 0.130329 |
| F-score | 0.680672 | 0.190455 | 0.627225 |
| Recall | 0.706426 | 0.280927 | 0.690487 |
| Connected components | 12.0000 | 8.5714 | 13.0000 |
| Betti-1 | 16.7143 | 79.8571 | 22.2857 |
| Nonmanifold-edge fraction | 0.028347 | 0.021592 | 0.031650 |

The local route has lower geometry loss than anchor-only in 6/7 cases and lower
loss than scan-fused in 6/7 cases. These are different exceptions: source 17
loses to anchor-only, while source 22 narrowly loses to scan-fused. Aggregate
geometry remains worse than anchor-only by 0.005678.

The mean F-score tolerance requires at least `0.680672 - 0.025 = 0.655672`, but
the local mean is 0.627225. The recall tolerance requires at least
`0.706426 - 0.01 = 0.696426`, but the local mean is 0.690487. Therefore:

`local_support_shadow_supported=false`.

## Protocol integrity

The calibration grid was committed as `d66c3ff`. The selected calibration
artifact was then hash-locked, and commit `6561711` froze the validation source
set, candidate, endpoints, and gates before the seven reserved source references
were evaluated.

- Calibration SHA-256:
  `dfc37f2bbc011e89646bd5a9a89744b9e065d78026b1dcdb58f900f64b18ecae`
- Validation protocol SHA-256:
  `f3c48a60657a8877f4cde330a8f917b99bd1a3521ed5ffd7682565145fce60a8`
- Result SHA-256:
  `ab91f20245bcdce687b897469084e742030a837ca0c2538dc49ad4395b5c3ed9`
- Deterministic repeat: identical result SHA-256 byte for byte.
- Phase-40 validation references accessed: false.
- Registration correctness labels accessed: false.
- Opened archive members: exactly the 32 Hokuyo point-cloud members.

## Failure diagnosis

Source 17 is the decisive anchor-relative geometry failure:

| Source-17 endpoint | Anchor only | Local support |
|---|---:|---:|
| Geometry loss | 0.140771 | 0.227313 |
| Normalized Hausdorff | 0.140230 | 0.225938 |
| F-score | 0.705230 | 0.671912 |
| Recall | 0.741476 | 0.755115 |

Recall improves, yet Hausdorff worsens sharply. This is consistent with a
coherent group of transformed scans agreeing with one another inside local
voxels while remaining misplaced relative to the trusted anchor observation.
Distinct-scan support and within-cell dispersion measure target-to-target
agreement; they do not establish target-to-anchor agreement.

This diagnosis is post-validation and must not be used to revise the Phase-41
thresholds or rerun the reserved panel.

## Claim boundary and next step

- `local_support_shadow_supported=false`;
- `point_local_alpha_field_supported=false`;
- `topology_correctness_supported=false`;
- `real_trimmed_reconstruction_supported=false`;
- `deployment_supported=false`.

Phase 41 does provide a useful negative result: scan provenance plus local
dispersion is insufficient as a standalone reconstruction certificate. The
next distinct observation must be anchor-relative, such as local distance and
normal/tangent agreement with the source observation, and must fail closed when
target scans agree only with each other.

Do not tune that observation on the seven Phase-41 sources or the 17 Phase-40
sources. Source 17 may now be used only as opened failure-development evidence.
Any positive validation claim requires a separately frozen endpoint panel or a
different real scene.
