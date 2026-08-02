# PFTF-alpha Codex handoff (2026-08-03)

## 오늘 종료 시점

PFTF-alpha의 기존 negative/limits 논문과 분리할 수 있는 긍정적 후속
방향을 검증했다. 현재 가장 방어 가능한 결과는 **sampling-sufficient,
globally separable, approximately parallel two-layer synthetic regime**에
한정된 constrained connectivity이다. 일반적인 alpha reconstruction,
PFTF-SPD 우월성, 실제 스캔 또는 배포 성능은 아직 지지되지 않는다.

Phase 0부터 Phase 9까지 구현, 문서화, 회귀 테스트가 저장소에 포함되어
있다. `benchmark-out/`의 JSON은 재생성 가능한 로컬 산출물이며 Git에는
포함하지 않는다.

## 단계별 결론

| Phase | 핵심 결과 | 판정 |
|---|---|---|
| 0 | Active reacquisition을 검토했지만 promotion 근거를 만들지 못함 | negative |
| 1 | Sampling sufficiency gate를 도입 | diagnostic |
| 2 | 16/16 sampling-sufficient cases에서 안전한 2층 constrained connectivity, B5 대비 mean F-score 0.3922 -> 0.7564 | positive, specialized |
| 3 | globally separable/approximately parallel synthetic stress 32/32 safe; near-contact/crossing 16/16 fail-closed | positive, bounded |
| 4 | curvature 0.36 이상에서 false-safe가 나타나는 global-coordinate blind spot 확인 | `phase4_diagnostic_supported=false` |
| 4b | observed-only normal-coherence guard가 별도 seed에서 false-safe 14 -> 0, safe retention 99.15% | `phase4b_supported=true`, fixed regime only |
| 5 | density/shape shift에서 false-safe 57 -> 12, retention 79.03% | `phase5_supported=false` |
| 6 | density-normalized local-order guard로 false-safe 61 -> 2, retention 91.58% | `phase6_supported=false` |
| 7 | shared-trend residual inference로 base false-safe 60 -> 0; safe accepts 186/186 유지; 58/60 repair, 2 fail-closed | `phase7_supported=true`, deployment false |
| 8 | N=160/256의 non-outlier stress 96/96 safe; N=96 coverage 43.75%; outlier false-safe 56 | `phase8_supported=false` |
| 9 | robust residual outlier guard로 false-safe 58 -> 4; 3%/5% contamination 전부 제거; safe retention 88.70% | `phase9_supported=false` |

Phase 7의 shared-quadratic mixture regression과 Phase 4b의 kNN-PCA
coherence는 conventional baseline이다. 이를 PFTF-SPD의 새 알고리즘으로
주장하면 안 된다.

## Phase 9 해석상 주의점

- 현재 strict invariant는 source label 2가 연결된 모든 triangle을
  unsafe로 센다. 주입점이 실제 표면 가까이에 있어도 provenance 위반이면
  실패로 분류된다.
- 남은 네 개의 1% outlier는 generating surface로부터 약 0.029--0.109
  거리에 있다. 다음 단계에서는 **source-provenance violation**과 실제
  **geometry/topology harm**를 분리해 보고해야 한다.
- residual guard는 local bump도 outlier처럼 거부한다. local-bump safe
  retention은 9/22이므로 shape-agnostic certificate가 아니다.
- Phase 8/9 패널을 본 뒤 같은 패널에 맞춰 threshold를 재조정하지 않는다.

## 다음 작업: Phase 10

오늘은 실행하지 않았다. 다음 세션에서는 아래 순서로 진행한다.

1. local tangent-plane leave-one-out surface consensus를 구현한다.
2. strict source-provenance 위반과 실제 geometry/topology harm를 별도
   endpoint로 기록한다.
3. 사전 성공 기준을 `harmful-outlier false-safe = 0` 및
   `clean/local-bump retention >= 90%`로 고정한다.
4. 위 기준을 통과한 뒤에만 trimmed reconstruction과 real-scan 검증으로
   이동한다.

## 재현 명령

저장소 루트 `D:\__PFTF_Projects(2026)\PFTF_alpha_dev`에서 실행한다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q

.\.venv\Scripts\python.exe -m pftf_alpha.shared_trend_inference --output benchmark-out/shared_trend_inference_phase7.json
.\.venv\Scripts\python.exe -m pftf_alpha.sensor_stress --output benchmark-out/sensor_stress_phase8.json
.\.venv\Scripts\python.exe -m pftf_alpha.outlier_guard --output benchmark-out/outlier_guard_phase9.json
```

이전 단계의 세부 명령과 frozen protocol은 다음 문서에 있다.

- `docs/ACTIVE_REACQUISITION_PHASE0.md`
- `docs/SAMPLING_SUFFICIENCY_GATE_PHASE1.md`
- `docs/TWO_LAYER_CONNECTIVITY_PHASE2.md`
- `docs/TWO_LAYER_STRESS_PHASE3.md`
- `docs/TWO_LAYER_BOUNDARY_PHASE4.md`
- `docs/CURVATURE_GUARD_PHASE4B.md`
- `docs/CURVATURE_GUARD_DOMAIN_SHIFT_PHASE5.md`
- `docs/LOCAL_ORDER_GUARD_PHASE6.md`
- `docs/SHARED_TREND_INFERENCE_PHASE7.md`
- `docs/SENSOR_STRESS_PHASE8.md`
- `docs/OUTLIER_GUARD_PHASE9.md`

## 논문 분리 원칙

- 기존 PFTF-alpha 논문: PFTF local SPD metric의 negative/limits result,
  M1 density baseline, G4 fail-closed fallback을 중심으로 유지한다.
- 긍정적 후속 논문 후보: Phase 2--9의 two-layer reconstruction을 별도
  축으로 발전시킨다. 현재는 synthetic bounded claim이며 Phase 10과
  real-scan 검증 전에는 투고 수준의 일반화 주장을 하지 않는다.
- 모든 산출물에서 `promotion_supported=false` 경계를 유지한다.
