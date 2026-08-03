# 3DMatch real-fragment registration guard benchmark: Phase 32

## Question and frozen claim boundary

Phase 32 tests whether the frozen Phase-27 global signature, Phase-28 local-q95
guard, and Phase-30 isolated-tail guard can safely filter independently produced
real-fragment registration predictions.

This is the first phase with official real-scene registration labels. It does
not train a new registration method and does not tune any guard threshold. The
external `3dmatch.log` predictions and fragment coordinates are used first to
materialize every observed guard decision. Only after that blind materialization
does the evaluator read `gt.log` and `gt.info` to score correctness.

## Official benchmark and provenance

The 3DMatch geometric registration benchmark uses real RGB-D fragments. The
official page states that every fragment is a surface point cloud fused from 50
depth frames and asks algorithms to predict transformations for non-consecutive
fragment pairs:
<https://3dmatch.cs.princeton.edu/>.

The selected scene is `7-scenes-redkitchen`. The official page marks 7-Scenes
data as non-commercial-use only. Phase 32 uses:

- fragments:
  <https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/7-scenes-redkitchen.zip>;
- evaluation files:
  <https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/7-scenes-redkitchen-evaluation.zip>;
- official evaluation driver:
  <https://github.com/andyzeng/3dmatch-toolbox/blob/master/evaluation/geometric-registration/evaluate.m>;
- official precision/recall and transformation-error implementation:
  <https://github.com/andyzeng/3dmatch-toolbox/blob/master/evaluation/geometric-registration/external/ElasticReconstruction/mrEvaluateRegistration.m>.

Locally frozen archive identities are:

| Archive | Bytes | MD5 | SHA-256 |
|---|---:|---|---|
| fragments | 39,950,111 | `2b7ba86ec2a370a9a5b989b63cc515f2` | `7cb9a1c9236e6833e910692b1d3f572b970c3fc3493e7641c28f1a45841fa51c` |
| evaluation | 140,319 | `20dc1a06956e01886f378109eb6df3bc` | `ff3eaa243025a0cdf6dd1ca5364a726acf7c08b36444e49c685e1f014bc4f16e` |

The safe extractor accepts exactly 60 binary little-endian XYZ-only PLY files
and exactly `3dmatch.log`, `gt.log`, and `gt.info`. It checks both hashes, ZIP
member CRCs, exact member allowlists, and path safety.

## Frozen protocol

The protocol was fixed before official labels were joined to observed guard
results:

1. Exclude consecutive pairs exactly as the official evaluator does.
2. Deterministically retain at most 10,000 order-spanning points per fragment.
3. Apply each external source-to-target prediction transform without using
   `gt.log`.
4. Form reciprocal 1-NN candidate pairs at both 2 cm and 5 cm. Neither distance
   is selected after observing labels.
5. Require at least 96 candidate pairs; otherwise fail closed.
6. Select four unique farthest spatial anchors and take a 96-pair neighbourhood
   around each.
7. A prediction passes a route only when all four patches pass that route:
   global score `< 0.18181536333942858`; then local q95
   `< 3.544133065251552`; then isolated-tail ratio
   `< 1.6636368999089541`.
8. After all 531 eligible blind predictions are materialized, join the 449
   eligible ground-truth overlap pairs and reproduce the official normalized
   quadratic transformation-error threshold `error <= 0.04`.

The predeclared threshold gate requires all of:

- guarded precision strictly above the unguarded external predictions;
- at least 90% of the unguarded correct predictions retained;
- at least 10% of unguarded incorrect predictions rejected; and
- the tail step retains at least 90% of predecessor-correct predictions while
  rejecting at least one predecessor-incorrect prediction.

Both 2 cm and 5 cm must pass for `phase32_supported=true`.

## Official-label baseline

`3dmatch.log` contains 588 raw predictions and 531 official non-consecutive
predictions. The ground truth contains 449 eligible overlap pairs. Reproducing
the official evaluator gives:

- correct predictions: 383;
- incorrect predictions: 148, comprising 138 non-overlap predictions and 10
  inaccurate transformations on overlap pairs;
- precision: 383/531 = **72.13%**; and
- recall: 383/449 = **85.30%**.

## Frozen-guard results

### Route comparison

| Distance | Route | Accepted | Correct | Incorrect | Precision | Recall |
|---:|---|---:|---:|---:|---:|---:|
| 2 cm | base | 531 | 383 | 148 | 72.13% | 85.30% |
| 2 cm | global | 72 | 64 | 8 | 88.89% | 14.25% |
| 2 cm | global + local | 72 | 64 | 8 | 88.89% | 14.25% |
| 2 cm | global + local + tail | 71 | 63 | 8 | 88.73% | 14.03% |
| 5 cm | base | 531 | 383 | 148 | 72.13% | 85.30% |
| 5 cm | global | 131 | 107 | 24 | 81.68% | 23.83% |
| 5 cm | global + local | 75 | 59 | 16 | 78.67% | 13.14% |
| 5 cm | global + local + tail | 54 | 42 | 12 | 77.78% | 9.35% |

At 2 cm only 72 predictions have enough reciprocal support for four 96-pair
patches; all 72 already pass both global and local criteria. Support availability,
not the local statistic, therefore dominates. The tail step removes one correct
prediction and no incorrect prediction.

At 5 cm, 189 predictions have sufficient support. The global step raises
precision, but the local and tail stages progressively lower it. The tail step
retains only 42/59 = 71.19% of predecessor-correct predictions while removing
four predecessor-incorrect predictions and seventeen predecessor-correct ones.

### Gate decision

| Distance | Full correct retention | Full incorrect rejection | Tail predecessor-correct retention | Tail incorrect removals | Gate |
|---:|---:|---:|---:|---:|---|
| 2 cm | 16.45% | 94.59% | 98.44% | 0 | fail |
| 5 cm | 10.97% | 91.89% | 71.19% | 4 | fail |

The high incorrect-rejection fractions do not establish a useful guard because
correct prediction retention collapses far below 90%. Precision gains at the
global step are obtained largely by abstention, and the added local/tail evidence
does not align with official registration correctness.

## Decision

- `real_registration_labels_supported=true`: the official real-scene
  ground-truth and transformation-error protocol are reproduced.
- `phase32_supported=false`: neither distance passes the frozen gate.
- `tail_sensitive_real_registration_supported=false`: the tail step has no
  useful incremental transfer at 2 cm and is strongly correctness-adverse at
  5 cm.
- `real_correspondence_supported=false`: reciprocal NN pairs remain candidate
  geometric matches, not physical point-identity labels.
- `real_trimmed_reconstruction_supported=false` and
  `deployment_supported=false`.

The synthetic Phase-30 success must not be presented as a real-registration
guard. Redkitchen is now opened and cannot be used to tune the thresholds. A
future learned or redesigned real-registration observation must use separate
3DMatch design scenes and reserve untouched scenes for validation; alternatively,
the unchanged negative route may be tested on a second scene only as a transfer
audit.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.threedmatch_redkitchen `
  --data-root benchmark-data/3dmatch_redkitchen --download

.\.venv\Scripts\python.exe -m pftf_alpha.threedmatch_registration_guard `
  --data-root benchmark-data/3dmatch_redkitchen `
  --phase28-artifact benchmark-out/local_spatial_residual_guard_phase28.json `
  --output benchmark-out/threedmatch_registration_guard_phase32.json
```

Artifact SHA-256:
`b7653adda0f0b93f14fda54bb57a4559c4a00863e4f22702b4bc14650442cb4d`.
