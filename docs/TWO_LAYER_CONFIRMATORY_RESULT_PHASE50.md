# Phase 50: two-layer positive confirmatory result

## 결론

Phase 50은 결과 확인 전에 고정한 모든 gate를 통과했다.

`phase50_supported=true`는 다음의 제한된 양성 결론을 지지한다.

> 충분한 sampling과 전역 분리가 가능한 non-outlier synthetic two-layer
> 표면에서, shared-trend layer inference와 layer-constrained connectivity는
> frozen B5 PCA-anisotropic 및 M1 weighted-alpha comparator보다 geometry와
> topology를 일관되게 개선했다.

이는 PFTF/local-SPD 우월성 결과가 아니다. Candidate는 관측 좌표에 적용하는
conventional shared-quadratic mixture model이며, 모든 일반화 주장은 아래 경계로
제한된다.

## Frozen provenance

- protocol commit: `940219376d7fe3c50233fbe44cdef6c33a4890a7`;
- implementation commit: `5543743`;
- protocol artifact SHA-256:
  `7615721e347647def8589cbf9204723ba000c529487a32dbfd7dd2d1a6839c76`;
- result artifact SHA-256:
  `236699aa3ba0bb00a7572ff46df7cd74c94fc7fe918ca2fcefed1de18c685b64`;
- deterministic repeat: byte-identical with the same SHA-256;
- untouched base seed: `35000804`;
- panel: N=160/256 × non-outlier stress 6종 × 12 repeats = 144 cases;
- pose: 매 case seed에서 만든 proper 3D rotation을 observed/reference에 함께
  적용;
- protocol identity: pass;
- M1 construction availability: 144/144.

Phase-50 point, label, reference 또는 endpoint를 보고 candidate, baseline,
threshold, case set 또는 gate를 변경하지 않았다.

## Safety와 ablation

| Endpoint | Global-normal base | Shared-trend candidate | Frozen gate |
|---|---:|---:|---:|
| Safe accepts | 138/144 | **144/144** | candidate coverage ≥95% |
| False-safe accepts | 6 | **0** | 0 |
| Repaired base false-safe | - | **6/6** | ≥1 |
| Minimum subgroup coverage | - | **100%** | ≥90% |

모든 12개 point-count × stress subgroup가 12/12 safe accept였고, candidate의
aggregate topology error와 nonmanifold-edge count도 모두 0이었다.

## Geometry efficacy

| Method | Mean F-score | Mean normalized geometry loss |
|---|---:|---:|
| B5 PCA-anisotropic | 0.515603 | 0.140552 |
| M1 weighted alpha | 0.620140 | 0.137152 |
| **Shared-trend two-layer** | **0.898536** | **0.126043** |

- mean F-score margin versus B5: **+0.382933**;
- mean F-score margin versus M1: **+0.278396**;
- casewise F-score wins versus B5: **144/144 (100%)**;
- casewise F-score wins versus M1: **144/144 (100%)**;
- frozen mean-margin gate: 각각 최소 +0.10;
- frozen casewise-win gate: 각각 최소 75%.

가장 약한 subgroup margin도 양수였다.

- B5 기준: N=256 upper occlusion, **+0.252639**;
- M1 기준: N=160 control, **+0.210241**;
- 최저 candidate subgroup mean F-score: N=160 sinusoidal, **0.865520**.

## Topology efficacy

사전등록 topology error는 component error, Betti error, true-layer bridge edges와
bridge faces의 합이다.

| Method | Aggregate topology error | Candidate nonmanifold edges |
|---|---:|---:|
| B5 PCA-anisotropic | 45,606 | - |
| M1 weighted alpha | 11,925 | - |
| **Shared-trend two-layer** | **0** | **0** |

이 큰 차이는 일반 alpha surface가 두 개의 가까운 층 사이에 cell/face를 만들 수
있는 반면, candidate가 관측 기반으로 두 층을 먼저 추론하고 각 층 안에서만
연결성을 구성하기 때문이다. Ground-truth layer label은 이 구성이나 route에
사용되지 않고 평가에만 사용됐다.

## Gate audit

| Gate | Result |
|---|---|
| Protocol/config identity | pass |
| Safety | pass |
| Geometry/casewise efficacy | pass |
| Topology | pass |
| Shared-trend ablation | pass |
| **Phase 50** | **pass** |

## 논문상 의미

Phase 50은 two-layer 방법을 별도의 positive manuscript로 전환할 수 있는 첫
강한 confirmatory result다. Phase 2/3의 초기 F-score 향상, Phase 7의 inference
repair, Phase 8의 작동 범위를 독립 seed, arbitrary poses, 더 큰 reference/sample
budget, B5와 M1 동시 비교, casewise 판정으로 한 번에 재확인했다.

논문의 핵심 기여는 “PFTF가 alpha를 고른다”가 아니라 다음으로 잡아야 한다.

1. alpha threshold 조정만으로 해결하기 어려운 두 층 간 false connectivity를
   layer inference와 constrained connectivity로 분리한다;
2. sampling sufficiency를 observed-only fail-closed route로 명시한다;
3. geometry와 topology를 함께 평가하고, 강한 adaptive/anisotropic comparator에
   대해 untouched casewise improvement를 보인다;
4. sparse sampling과 outlier에 대한 실패 경계를 숨기지 않는다.

## 주장 경계와 다음 검증

계속 `false`로 유지한다.

- `promotion_supported=false`;
- `pftf_superiority_supported=false`;
- `real_scan_supported=false`;
- `deployment_supported=false`.

Phase 50은 spatial outlier, N<160, intersecting/non-separable surface, arbitrary
surface family, real scan, exact predicates를 다루지 않는다. 다음 독립 단계는
candidate를 다시 조정하는 것이 아니라, 두 표면의 ground-truth 분리가 가능한
실측 또는 공개 real two-layer point-cloud corpus를 사전등록해 외적 타당성을
검사하는 것이다.

## 재현 명령

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.two_layer_confirmatory `
  --output benchmark-out\two_layer_confirmatory_phase50.json
```

Final repository verification:

- `ruff check src tests examples`: pass;
- `pytest -q -p no:cacheprovider`: **440 passed**.
