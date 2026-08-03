# 3DMatch second-scene frozen transfer audit: Phase 33

## Question and claim boundary

Phase 33 asks whether the unchanged Phase-32 real-registration guard behaves
usefully on a second independently labeled 3DMatch scene. It is a transfer
audit of an already negative route, not a threshold-selection or model-training
phase.

The second scene is SUN3D `hotel_umd/maryland_hotel3`. It was selected after
redkitchen only because its official fragment download is the smallest of the
remaining seven scenes. No scene result, label count, or guard statistic was
used in that choice.

## Official provenance and archive boundary

The official 3DMatch page lists eight real-scene registration sets, including
`hotel_umd/maryland_hotel3`, and states that each fragment is a surface point
cloud fused from 50 RGB-D depth frames:
<https://3dmatch.cs.princeton.edu/>.

Phase 33 uses the official files:

- fragments:
  <https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel3.zip>;
- evaluation:
  <https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel3-evaluation.zip>.

The official SUN3D page identifies the source dataset and requests citation,
but the accessed 3DMatch and SUN3D pages do not state an explicit SUN3D data
license. The archives therefore remain ignored local inputs and must not be
redistributed: <https://sun3d.cs.princeton.edu/>.

| Archive | Bytes | Files | MD5 | SHA-256 |
|---|---:|---:|---|---|
| fragments | 32,849,425 | 37 PLY | `6edd29d020fa164939141ad218973aba` | `2dd600fad0cfd98968b9ff1684430f3647f241b3690004154c56d1c058c6f5bc` |
| evaluation | 18,445 | 3 | `ad9128fae730de3c29ccf76575577a7a` | `180bf3749c7353f5e0a0a17220f9760944ed86613aaf787ab1e841e7007912b7` |

The scene-spec verifier checks both hashes, CRCs, exact member allowlists, and
path safety before extraction. The 37 fragment indices are exactly 0--36 and
all PLY files use the same binary little-endian XYZ-only schema as redkitchen.

## Frozen protocol identity

The evaluator refuses to start unless the opened Phase-32 artifact has the
expected schema, redkitchen identity, Phase-28 model hash, and every exact
setting below:

- reciprocal 1-NN distances: 2 cm and 5 cm;
- at most 10,000 deterministic order-spanning points per fragment;
- four unique farthest patches of 96 candidate pairs;
- global score `< 0.18181536333942858`;
- local q95 `< 3.544133065251552`;
- isolated-tail ratio `< 1.6636368999089541`;
- official transformation error `<= 0.04`; and
- the unchanged Phase-32 precision, retention, rejection, and incremental-tail
  gates.

The Phase-32 reference artifact SHA-256 is
`b7653adda0f0b93f14fda54bb57a4559c4a00863e4f22702b4bc14650442cb4d`.
The canonical frozen-protocol SHA-256 is
`a29abba7b0088b8a0467cd0cbb9cc324b59c3b87c139cb0377054a70b7dc2fd6`.

As in Phase 32, the evaluator materializes every coordinate-only guard
observation from the external prediction transforms before reading `gt.log` or
`gt.info`.

## Official-label baseline

The archive contains 88 raw predictions. The official non-consecutive filter
leaves 61 predictions and 26 ground-truth overlap pairs:

- correct predictions: 15;
- incorrect predictions: 46, comprising 41 non-overlap predictions and 5 bad
  transforms on overlap pairs;
- precision: 15/61 = **24.59%**; and
- recall: 15/26 = **57.69%**.

## Frozen-route results

| Distance | Route | Accepted | Correct | Incorrect | Precision | Recall |
|---:|---|---:|---:|---:|---:|---:|
| 2 cm | base | 61 | 15 | 46 | 24.59% | 57.69% |
| 2 cm | global | 13 | 1 | 12 | 7.69% | 3.85% |
| 2 cm | global + local | 13 | 1 | 12 | 7.69% | 3.85% |
| 2 cm | global + local + tail | 13 | 1 | 12 | 7.69% | 3.85% |
| 5 cm | base | 61 | 15 | 46 | 24.59% | 57.69% |
| 5 cm | global | 13 | 1 | 12 | 7.69% | 3.85% |
| 5 cm | global + local | 11 | 0 | 11 | 0.00% | 0.00% |
| 5 cm | global + local + tail | 8 | 0 | 8 | 0.00% | 0.00% |

At 2 cm, only 13 predictions support all four patches. All 13 pass every
guard, but twelve are incorrect. At 5 cm, 21 predictions support the patches;
the global step accepts 13, the local step removes the only correct prediction,
and the tail step removes three additional incorrect predictions from an
already zero-correct predecessor.

| Distance | Full correct retention | Full incorrect rejection | Precision improved | Tail incorrect removals | Gate |
|---:|---:|---:|---|---:|---|
| 2 cm | 6.67% | 73.91% | no | 0 | fail |
| 5 cm | 0.00% | 82.61% | no | 3 | fail |

Unlike redkitchen, apparent precision does not improve. The frozen score ranks
the second-scene correct predictions below many incorrect predictions. This is
stronger evidence against a useful cross-scene guard than a mere
support-availability limitation.

## Decision

- `phase33_audit_completed=true`;
- `negative_transfer_reproduced=true`;
- `second_scene_guard_supported=false`;
- `second_scene_tail_supported=false`;
- `cross_scene_guard_supported=false`;
- `tail_sensitive_real_registration_supported=false`; and
- correspondence identity, trimmed reconstruction, and deployment remain
  unsupported.

Testing the unchanged route on more scenes is no longer the highest-value next
step. Any further real-registration work should redesign the observation using
separate design scenes and reserve untouched scenes for validation. The two
opened scenes must never be used as fresh validation after such redesign.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.threedmatch_transfer_audit `
  --download `
  --data-root benchmark-data/3dmatch_maryland_hotel3 `
  --phase28-artifact benchmark-out/local_spatial_residual_guard_phase28.json `
  --reference-phase32-artifact benchmark-out/threedmatch_registration_guard_phase32.json `
  --output benchmark-out/threedmatch_transfer_audit_phase33.json
```

Artifact SHA-256:
`961773176cd10cd41e6054b2b898f02af2c9c357a28401911c05189b2bedd5fa`.
