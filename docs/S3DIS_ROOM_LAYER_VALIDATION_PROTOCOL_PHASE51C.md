# Phase 51C — S3DIS Area-5 final room-layer validation protocol

Area 5 is Building 3 and has not been opened at the point of this protocol.
Member names and byte sizes were counted from the ZIP central directory, but no
Area-5 point coordinate has been extracted, parsed, summarized, visualized, or
evaluated.

The final panel is enumerated automatically from every lexicographically sorted
Area-5 room containing both `floor*.txt` and `ceiling*.txt` annotations. All
same-class fragments in a room are merged. The frozen calibration eligibility
rule is:

- normal angle <= 5 degrees;
- projected bounding-box overlap >= 0.75;
- at least 592 common-footprint points per layer;
- gap / joint plane-residual RMS >= 10; and
- p95 plane residual / median spacing <= 10 per layer.

Each case uses 80 coordinate-hash-selected observed points and up to 512
disjoint reference points per layer, an exact common-footprint crop, footprint
normalization, and the same deterministic `1e-4` normalized general-position
joggle for every method. Routing receives only combined observed XYZ.

The panel must contain at least 20 eligible rooms. Candidate false-safe accepts
must be zero and safe-accept coverage at least 0.90. B5/M1 availability must be
at least 0.90/0.95. On paired available cases, candidate F-score margins must be
at least +0.20/+0.30, casewise win rates at least 0.85, geometry loss strictly
lower, and topology-error sums at most 0.25 of each comparator. The candidate
must remain non-inferior to the global-normal ablation within 0.02 coverage and
0.01 mean F-score.

All gates must pass without changing the panel or parameters. A positive result
supports only real, building-disjoint, long-gap, approximately parallel
floor--ceiling reconstruction with annotation-defined corpus extraction and
observed-XYZ-only routing. PFTF, local-SPD, shared-trend, close-layer, and
deployment claims remain false.
