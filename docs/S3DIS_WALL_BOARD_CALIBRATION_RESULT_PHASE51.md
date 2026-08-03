# Phase 51A — S3DIS wall--board calibration result

## Leakage-controlled intake

The aligned S3DIS v1.2 archive was downloaded from the ETH Zurich university
mirror and audited after the Phase-51 protocol and extractor were committed.
The archive contains 11,257 ZIP members and has SHA-256
`f16db73310983e6b4df9b3d5290d20315e0cdf56c14f0d795f96d372378fce94`.

Only `board*.txt` and `wall*.txt` members from calibration Areas 1, 2, 3, 4,
and 6 were opened and extracted: 1,300 members in total. Area 5 remains inside
the archive. Its 385 wall/board member names and sizes were counted from the ZIP
central directory, but no Area-5 point coordinates were extracted, parsed,
summarized, visualized, or evaluated.

The intake artifact reports
`external_archive_intake_supported=true` and
`reserved_content_opened=false`. Artifact SHA-256:
`0bf4883ea34cad27f284143b388c4b643e97f4ba0cbaaa1047c5e09997049823`.

## Calibration geometry

There are 95 calibration board instances. Every board could be associated with
a nearly parallel wall in the same room. The selected normal angle has median
`0.378 degrees`; the physical gap has median `0.0115` in S3DIS coordinate units.
However, the regime differs materially from the Phase-50 generator:

- median gap / point-spacing is only `0.655`;
- only 22/95 pairs have gap / spacing at least 1;
- median wall support inside the projected board footprint is `0.344` on an
  8x8 grid;
- only 2/95 pairs have projected support at least 0.5.

This is the expected physical-occlusion pattern of a board mounted on a wall:
the wall behind the board is usually not observed. The calibration geometry
artifact SHA-256 is
`069004f9c98f772c266bba1820207b86a45ca5be922cbbe1a6175f84fb627a0e`.

## Method-development result

The initial calibration-only rule selected 33 pairs using support at least
0.20, at least 320 wall points inside the board footprint, gap / spacing at
least 0.50, angle at most 15 degrees, and per-plane p95 residual / spacing at
most 2. The observation/reference split is deterministic and disjoint, with 80
observed points per layer and up to 512 reference points per layer.

The frozen Phase-50 candidate shows a useful geometry signal:

- mean F-score: candidate `0.7813`, B5 `0.7289`, M1 `0.6100`;
- mean F-score margin: `+0.0524` over B5 and `+0.1713` over M1;
- casewise F-score wins: 29/33 over B5 and 32/33 over M1.

But this does **not** produce a safe real two-layer result. Candidate faces mix
the evaluation-only board and wall labels in all 33 cases. The observed-only
gate therefore accepts none: 26 cases request rescanning and 7 fail closed as
unsupported geometry. Tightening gap / spacing to 1.0 or 1.5 still gives zero
safe accepts. Benchmark artifact SHA-256:
`d3edfa629681dc70294b6bf4068b27ef6d0e49dc93bdd38bf5b7266c713e082f`.

## Decision

Do not open Area-5 wall/board coordinates and do not claim real wall--board
support. Phase 51A establishes a positive real-geometry F-score tendency but a
negative topology/safety transfer caused by occluded, non-overlapping support.

The next two-layer calibration branch should use S3DIS floor--ceiling pairs.
They are real, building-disjoint, parallel surfaces with overlapping room
footprints and therefore match the declared two-surface observation regime
without pretending that the wall behind an opaque board was measured. This is
an easier long-gap regime and must be reported separately from near-layer
wall--board reconstruction. Area 5 must remain unopened until a new
floor--ceiling protocol, calibration rule, and final evaluation protocol are
committed.
