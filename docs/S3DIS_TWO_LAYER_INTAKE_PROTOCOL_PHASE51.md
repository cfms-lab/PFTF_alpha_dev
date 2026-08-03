# Phase 51 — S3DIS real two-layer intake protocol

## Purpose

Phase 50 established a bounded positive result only on synthetic two-layer
surfaces. Phase 51 starts a separate external validation on the Stanford
Large-Scale Indoor Spaces 3D Dataset (S3DIS), using real annotated `board` and
`wall` point-cloud instances. This document freezes acquisition and information
boundaries before any reserved held-out coordinates are opened. It is not a
real-scan result.

## Corpus and split

- Corpus: S3DIS v1.2, globally aligned point-cloud archive from the ETH Zurich
  university mirror linked to the Stanford dataset.
- Calibration: Areas 1, 2, 3, 4, and 6.
- Reserved external held-out: Area 5.
- Rationale: the distributed README maps Areas 1/3/6 to Building 1, Areas 2/4
  to Building 2, and Area 5 to Building 3. The test is therefore building
  disjoint.
- Dataset terms boundary: the distributed README asks users to cite the dataset
  but states no license. Keep the archive local for research validation, cite
  the dataset, do not redistribute it, and make no broader rights claim.

## Frozen information boundary

Before a separate final-evaluation protocol is committed, Area 5 may be seen
only through ZIP central-directory metadata: member paths and compressed or
uncompressed sizes. Its point-cloud members must not be extracted, opened,
parsed, summarized, visualized, or used for any statistic.

Calibration areas may be used to freeze deterministic rules for:

1. associating each annotated board instance with a nearby wall instance;
2. robust plane-fit quality and minimum point counts;
3. plane parallelism, projected overlap, physical gap, and gap-to-spacing
   eligibility;
4. a local wall crop around the board footprint;
5. a coordinate-hash observed/reference split; and
6. sampling-sufficiency and fail-closed handling.

The eventual candidate receives only the combined observed XYZ coordinates.
RGB, semantic class names, instance IDs, room/area names, held-out reference
points, and baseline endpoints are extraction or evaluation-only information.

## Frozen method family and comparators

The candidate remains the unchanged Phase-50 shared-trend two-layer method.
The comparators remain the global-normal two-layer ablation, frozen B5
PCA-anisotropic alpha, and frozen M1 weighted power-alpha. Phase 51 may calibrate
only corpus eligibility, crop, and deterministic observation/reference rules;
it must not retune the candidate on Area 5.

## Required checkpoint before held-out opening

After calibration-only intake, a new committed protocol must freeze the exact
pair eligibility thresholds, Area-5 enumeration rule, coordinate-hash split,
metrics, gates, and error handling. Only then may Area-5 point contents be
opened once as a panel.

Until that checkpoint,
`external_archive_intake_supported=false`, `real_scan_supported=false`,
`held_out_validation_supported=false`, `pftf_superiority_supported=false`, and
`deployment_supported=false`.
