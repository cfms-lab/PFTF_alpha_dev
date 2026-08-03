# Phase 51C — S3DIS Area-5 final validation result

## Frozen external test

The final protocol was committed as `330b340` before any Area-5 point contents
were opened. The evaluator was committed as `7f5882d` before extraction. The
protocol SHA-256 is
`fbbb56cad2dff5127bf51677e8b1f520b8870353de23dc927cce9db314485807`.

The once-opened panel is Building-3 Area 5 of aligned S3DIS v1.2. The extractor
opened 68 floor files and 76 ceiling files, 144 files and 983,453,238
uncompressed bytes in total. It did not extract other classes or areas.

The automatic frozen eligibility rule admits 63 rooms, exceeding the minimum
panel size of 20. No room was manually included or excluded after opening.

## Result

The observed-XYZ-only two-layer candidate produces:

- safe accepts: 62/63 = `0.984127`;
- false-safe accepts: 0;
- one conservative `rescan_required` on `Area_5/hallway_5`; that output is
  topology-safe and has F-score `0.989640`;
- mean F-score: `0.805611`;
- mean normalized geometry loss: `0.116385`;
- aggregate topology error: 0.

Both frozen alpha comparators are available on all 63 rooms:

- B5 mean F-score `0.420983`, candidate paired margin `+0.384627`;
- M1 mean F-score `0.323764`, candidate paired margin `+0.481847`;
- candidate wins F-score 63/63 against B5 and 63/63 against M1;
- B5/M1 mean geometry loss `0.243356` / `0.228920`, both worse than the
  candidate;
- B5/M1 topology error 5,214 / 13,375, versus candidate 0.

The global-normal two-layer ablation has the same 62/63 safe coverage, zero
false-safe, and the same mean F-score at printed precision. The preregistered
ablation non-inferiority gate passes, but there is no shared-trend superiority.

Every frozen gate passes:

- protocol and calibration identity;
- minimum panel size;
- zero-false-safe and >=0.90 safe coverage;
- B5/M1 construction availability;
- F-score margin, casewise win, and geometry-loss efficacy;
- topology-error ratio; and
- global-normal ablation non-inferiority.

Therefore:

- `phase51c_supported=true`;
- `real_long_gap_two_layer_supported=true`;
- `real_scan_supported=true`;
- `held_out_validation_supported=true`.

The following remain false:

- `pftf_superiority_supported=false`;
- `local_spd_superiority_supported=false`;
- `shared_trend_superiority_supported=false`;
- `close_layer_transfer_supported=false`;
- `deployment_supported=false`.

## Reproducibility

- Area-5 geometry artifact SHA-256:
  `f30f6fa3ddb3c3cef376f99febac46184c88c90bdbc3e961a69a18b06313a528`;
- frozen core endpoint artifact SHA-256:
  `9202fd317b11029d586381daa0ce3290997eb36a2fa8bfcbfb88b03307ca05fb9`;
- final gate-audit artifact SHA-256:
  `14645b09c2c73c58ee242c370034afb109f19bc1b5c9e55a3f35fb9864ffd9b8`.

A second unchanged execution produced byte-identical geometry, identical 63
case endpoints, identical gates, and the same support flags. The wrapper file
hash differs only because the repeat output filenames are embedded.

## Paper-level conclusion

This is the first positive real/public held-out result in the project. The
defensible paper claim is narrow but affirmative: on building-disjoint S3DIS
rooms with sufficiently planar, overlapping, long-gap floor--ceiling layers,
observed-only two-layer routing and constrained per-layer reconstruction are
substantially more accurate and topologically safer than frozen anisotropic and
weighted alpha baselines.

It is not evidence that PFTF selects alpha, that shared-trend inference is
superior to a global-normal alternative, or that the method transfers to close
occluded wall--board layers. Phase 51A already provides the corresponding
negative close-layer boundary.
