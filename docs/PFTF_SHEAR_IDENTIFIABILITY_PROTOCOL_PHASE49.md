# Phase 49: PFTF shear-signal identifiability protocol

## Motivation

Phase 48 showed that a PFTF-conditioned quadratic-shear decoder is worse than
simple controls on its frozen held-out panel. That panel is now closed. Phase
49 does not add features or retry it. Instead, it uses only the earlier TRAIN
and CALIBRATION partitions to ask why the decoder failed.

The diagnostic separates two questions:

1. Within one fixed surface family and seed, does an observed PFTF summary
   respond consistently as synthetic shear strength changes?
2. Is that response strong enough to identify absolute shear from a single
   observation when family and seed are unknown?

Passing the first question alone means that a repeated-measurement signal
exists. It is not a deployable identification result.

## Frozen information boundary

- Reuse Phase-48 TRAIN: 12 family/seed blocks, five strengths each.
- Reuse Phase-48 CALIBRATION: six new family/seed blocks, five interleaved
  strengths each.
- Prohibit all Phase-48 held-out seeds, points, labels, predictions, and
  endpoint values from feature selection and Phase-49 gates.
- Consume no reference surface, alpha threshold, or reconstruction endpoint.

One block is one family/seed cloud observed at all five strengths. True
strength organizes this diagnostic repeated-measurement experiment only.

## Frozen feature selection

The seven Phase-48 PFTF summaries and four non-PFTF geometry summaries remain
unchanged. On TRAIN only, select one feature from each group by maximizing, in
order:

1. median within-block linear `R2`;
2. block-slope sign consistency;
3. standardized feature change across the declared strength span.

Exact ties keep the earlier frozen feature order. The median TRAIN slope sign
is frozen before CALIBRATION.

## Frozen CALIBRATION gates

A stable within-block PFTF signal requires all of:

- median within-block `R2 >= 0.75`;
- at least `5/6` block slopes match the TRAIN direction;
- all three family-median slopes match the TRAIN direction;
- median slope-span effect is at least `0.25` TRAIN feature standard deviations.

PFTF specificity additionally requires its CALIBRATION median within-block
`R2` to strictly exceed the separately TRAIN-selected geometry feature.

Standalone identification fits one clipped univariate affine decoder on TRAIN
only. Its CALIBRATION coefficient MAE must be below the geometry decoder and
below `0.75` times the TRAIN-mean baseline MAE.

New representation development requires stable and PFTF-specific signals. A
new held-out panel additionally requires the standalone gate.

## Confounding diagnostics and claim boundary

For each selected feature, report pooled `R2`, block variance fraction, partial
strength `R2` after block intercepts, and the ratio of block to strength
explained sums of squares. These describe whether family/seed offsets dominate
the shear response; they do not alter any gate.

A positive within-block result supports only a response to one known,
coordinate-aligned synthetic shear under repeated measurements. It does not
establish standalone identifiability, PFTF reconstruction value, alpha
selection, real transfer, exactness, or deployment.
