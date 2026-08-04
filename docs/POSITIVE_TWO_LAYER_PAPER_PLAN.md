# Positive two-layer paper: frozen claim and artifact map

Date: 2026-08-04

## One-sentence claim

On sufficiently planar, overlapping, long-gap two-surface point clouds,
observed-XYZ-only two-layer routing followed by constrained per-layer
triangulation is substantially more accurate and topologically safer than the
frozen B5 PCA-anisotropic and M1 weighted power-alpha comparators.

## Claim ownership

- **Core novelty:** change admissible connectivity by inferring two layers,
  failing closed when the observed separation/sampling evidence is inadequate,
  and triangulating the accepted layers independently.
- **Synthetic extension:** shared-quadratic trend removal repairs six
  global-normal false-safe cases in the untouched Phase-50 panel.
- **Real-data result:** the core layer-first construction transfers to
  building-disjoint S3DIS Area 5, but shared trend does not outperform the
  simpler global-normal ablation there.
- **Not claimed:** PFTF or local-SPD superiority, learned/global alpha
  selection, close-layer transfer, arbitrary multi-surface reconstruction,
  automatic semantic extraction, exact predicates, or deployment.

## Main evidence table

| Panel | Candidate F | B5 F | M1 F | Candidate wins | Candidate topology | B5 topology | M1 topology |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 50 synthetic, n=144 | 0.898536 | 0.515603 | 0.620140 | 144/144 vs each | 0 | 45,606 | 11,925 |
| Phase 51C S3DIS Area 5, n=63 | 0.805611 | 0.420983 | 0.323764 | 63/63 vs each | 0 | 5,214 | 13,375 |

Safety: Phase 50 candidate safe accepts 144/144 with false-safe 0; Phase 51C
safe accepts 62/63 with false-safe 0. The sole Phase-51C rescan output is itself
topology-safe, but remains a rejection under the frozen route.

## Source-to-claim map

| Manuscript element | Primary local evidence |
|---|---|
| Phase-50 protocol and gates | `docs/TWO_LAYER_CONFIRMATORY_PROTOCOL_PHASE50.md` and `benchmark-out/two_layer_confirmatory_protocol_phase50.json` |
| Phase-50 result | `docs/TWO_LAYER_CONFIRMATORY_RESULT_PHASE50.md` and `benchmark-out/two_layer_confirmatory_phase50.json` |
| S3DIS calibration boundary | `docs/S3DIS_ROOM_LAYER_CALIBRATION_RESULT_PHASE51B.md` |
| Phase-51C protocol | `docs/S3DIS_ROOM_LAYER_VALIDATION_PROTOCOL_PHASE51C.md` and its JSON artifact |
| Phase-51C result | `docs/S3DIS_ROOM_LAYER_VALIDATION_RESULT_PHASE51C.md` and `benchmark-out/s3dis_room_layer_validation_phase51c.json` |
| Close-layer negative boundary | `docs/S3DIS_WALL_BOARD_CALIBRATION_RESULT_PHASE51.md` |
| Method implementation | `sampling_gate.py`, `shared_trend_inference.py`, `two_layer_connectivity.py` |

## Paper assets

- English peer-blind source: `draft/paper_en.tex` (target: at most 4,000 main-text words).
- Korean internal-review source: `draft/paper_kr.tex`.
- Concept figure: `draft/pics/two_layer_workflow.svg`.
- Confirmatory result figure: `draft/pics/two_layer_results.svg`.
- Deterministic generator: `scripts/make_two_layer_paper_figures.py`.

No PDF or DOCX is generated at this intermediate stage.
