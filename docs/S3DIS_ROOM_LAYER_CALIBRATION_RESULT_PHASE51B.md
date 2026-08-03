# Phase 51B — S3DIS floor--ceiling calibration result

## Leakage boundary

Phase 51B was preregistered after the wall--board calibration failed its
topology/safety transfer and before any Area-5 point contents were opened. The
same building-disjoint split is retained: Areas 1/2/3/4/6 are calibration and
Area 5 is reserved.

The calibration-only extractor opened 525 floor/ceiling annotation files. The
ZIP central directory reports 144 target members in Area 5, but no Area-5
member was extracted, parsed, summarized, visualized, or evaluated. The intake
artifact reports `floor_ceiling_intake_supported=true` and
`reserved_content_opened=false`; SHA-256:
`d9fd169baab6d0238eab58dbad46cef0cd4ad06da35369c897390282ea39fbe8`.

## Calibration geometry

All 204 calibration rooms contain both floor and ceiling data after same-class
annotation fragments are merged. Across rooms:

- median normal angle: `0.161 degrees`;
- median projected bounding-box overlap: `0.982`;
- median physical gap: `2.715` S3DIS coordinate units;
- median gap / point-spacing: `110.3`;
- median gap / joint plane-residual RMS: `409.7`.

The calibration geometry artifact SHA-256 is
`f55094230c08b2452edc8aeff96d95d0205affb58ded23f63d9a4d036918f502`.

The calibration-only eligibility rule is:

- normal angle at most 5 degrees;
- projected bounding-box overlap at least 0.75;
- at least 592 common-footprint points on each layer;
- gap / joint plane-residual RMS at least 10; and
- p95 plane residual / median spacing at most 10 on each layer.

The case builder takes a deterministic disjoint coordinate-hash split with 80
observed and up to 512 reference points per layer. It normalizes by the common
footprint diagonal. Because quantized planar points can generate rank-deficient
3D Delaunay cells, all methods receive the same deterministic `1e-4` normalized
general-position joggle; reference points remain unchanged. This is below the
original S3DIS `0.001` coordinate quantization scale after room normalization.

## Calibration-only method result

The rule admits 179 rooms. The unchanged Phase-50 shared-trend candidate gives:

- safe accepts: 172/179 = `0.960894`;
- false-safe accepts: 0;
- mean F-score: `0.777477`;
- mean normalized geometry loss: `0.118253`;
- aggregate topology error: 1,122, confined to rejected cases for the strict
  false-bridge safety endpoint.

Against frozen comparators:

- B5 is constructible on 177/179 rooms and fails closed on 2; candidate mean
  paired F-score margin is `+0.344118`, with 174/177 casewise wins;
- M1 is available on all 179 rooms; candidate mean paired F-score margin is
  `+0.486853`, with 176/179 casewise wins;
- B5 and M1 mean geometry loss are `0.247555` and `0.263249`, both worse than
  the candidate;
- topology error sums are 14,529 for B5 and 35,209 for M1, versus 1,122 for the
  candidate.

The global-normal two-layer ablation also has 172/179 safe accepts, zero
false-safe, and mean F-score `0.776976`. Therefore Phase 51B does **not** support
shared-trend, PFTF, or local-SPD superiority. It supports a narrower hypothesis:
observed-only two-layer routing plus constrained per-layer reconstruction can
provide strong real long-gap floor--ceiling geometry and topology benefits over
unconstrained alpha baselines.

Calibration benchmark SHA-256:
`3dccb4c48cb97b73c432eca31f1d3d7f833d7df70895ab540dfd752b0f71798f`.

## Next checkpoint

Freeze these eligibility, observation, comparator, safety, geometry, topology,
and construction-availability rules in a separate final Area-5 protocol. Only
after that protocol is committed may Area-5 floor/ceiling point contents be
opened once as a building-disjoint panel.

At this checkpoint, `real_scan_supported=false`,
`held_out_validation_supported=false`, `pftf_superiority_supported=false`, and
`deployment_supported=false`.
