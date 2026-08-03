# Phase 48: learned PFTF-conditioned coordinate-map protocol

## Motivation

Phase 47 established one analytic, globally invertible nonlinear coordinate map
whose Jacobian induces a coherent spatial SPD alpha complex. It did not show
that observed PFTF evidence can identify such a map. Phase 48 asks that narrower
question without reopening alpha selection or reconstruction endpoints.

## Frozen synthetic recovery task

Each latent point cloud is centered and divided by its RMS radius. A hidden
coefficient `s` then corrupts it with

`Phi_-s(x,y,z) = (x, y - s*x^2, z)`.

The learner sees only the corrupted coordinates. It predicts one scalar and
applies

`Phi_s(x,y,z) = (x, y + s*x^2, z)`.

For every prediction this map is globally invertible, has explicit inverse
`Phi_-s`, and has determinant one. The task therefore learns a member of a
safe map family; it does not learn an unconstrained pointwise metric field.

The panel contains torus, disconnected-parts, and sharp-crease families with
64 points per case. Its disjoint partitions are:

- TRAIN: 60 cases from four seeds and strengths `0, 0.08, ..., 0.32`;
- CALIBRATION: 30 cases from two new seeds and interleaved strengths
  `0.04, 0.12, ..., 0.36`;
- HELD_OUT: 45 cases from three further seeds and strengths
  `0.02, 0.10, ..., 0.34`.

## Frozen learners and comparators

The candidate uses seven observed-only summaries from the PFTF relation field:
signed `xy` relation mean and spread, relation-strength median and 90th
percentile, mean confidence, mean reciprocity, and log-scale spread.

A standardized linear ridge model is fit on TRAIN only. CALIBRATION chooses
one penalty from `0, 1e-4, 1e-2, 1`; exact ties choose the larger penalty.
Predictions are clipped to the declared safe family `[0, 0.40]`. All
preprocessing, coefficients, penalty, and clipping are frozen before HELD_OUT.

The fixed comparisons are:

1. identity/no correction;
2. the TRAIN-mean coefficient;
3. an equally trained ridge model using only non-PFTF global covariance and
   moment features;
4. the true coefficient as an evaluation-only oracle ceiling.

## Frozen endpoints and gate

The three held-out mean endpoints are coefficient MAE, corrected-coordinate
RMS error to the latent cloud, and Delaunay top-cell Jaccard against the latent
alpha complex. PFTF conditioning is supported only if it strictly beats both
the TRAIN-mean and non-PFTF geometry models on all three endpoints. Every
predicted map must also remain within its declared bounds, retain determinant
one, and pass inverse roundtrip error `<=1e-12`.

Held-out labels, latent points, and alpha complexes are evaluation targets
only. They cannot alter model selection after the protocol is committed.

## Claim boundary

A positive result supports only recovery of one coordinate-aligned scalar
quadratic-shear family on this synthetic panel. It does not establish an
arbitrary local-SPD field, a general nonlinear map learner, alpha selection,
reconstruction or topology advantage, exact predicates, real-scan transfer,
or deployment.
