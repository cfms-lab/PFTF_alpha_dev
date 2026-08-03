# Phase 51B — S3DIS floor--ceiling calibration protocol

Phase 51A found real-data F-score gains on S3DIS wall--board calibration pairs,
but zero topology-safe accepts. An opaque board hides the wall behind it, so
the two annotated surfaces do not supply the overlapping support assumed by the
Phase-50 method. Area 5 was not opened.

Phase 51B therefore starts a distinct calibration branch on real S3DIS floor
and ceiling annotations. These surfaces are approximately parallel and share a
room footprint, but their large separation makes this an easier regime. Any
positive result must be described as **real long-gap floor--ceiling transfer**,
not close-layer wall--board reconstruction.

The split remains building disjoint:

- calibration: Areas 1, 2, 3, 4, and 6;
- reserved held-out: Area 5.

Only calibration floor/ceiling contents may be extracted. Area-5 content must
remain unopened until a later committed protocol freezes the exact plane,
overlap, gap, point-count, crop, coordinate-hash split, metrics, gates, and case
enumeration.

The candidate remains the unchanged Phase-50 shared-trend method. B5, M1, and
the global-normal ablation remain frozen comparators. Reconstruction receives
combined observed XYZ only; semantic labels and references are extraction or
evaluation-only.

At this checkpoint, `real_scan_supported=false`,
`held_out_validation_supported=false`, `pftf_superiority_supported=false`, and
`deployment_supported=false`.
