# Phase 50: two-layer positive confirmatory protocol

## 목적

Phase 49까지의 결과는 임의 local-SPD 또는 PFTF-conditioned alpha complex의
우월성을 지지하지 않았다. 반면 Phase 2, 3, 7은 globally separable two-layer
표면에서 층별 연결성 구성이 geometry와 topology를 개선할 수 있음을 보였고,
Phase 8은 `N >= 160`인 non-outlier sensor stress 96/96에서 안전한 수용을
확인했다.

Phase 50은 이 이미 선언된 작동 범위를 개발에 사용하지 않은 새 seed와 강한
alpha-shape 비교군으로 확인한다. 결과를 본 뒤 범위나 threshold를 바꾸지 않는다.

## Untouched panel

- base seed: `35000804`;
- observed point counts: `160`, `256`;
- stress: control, upper occlusion, 75/25 imbalance, anisotropic noise,
  sinusoidal surface, local bump;
- repeat: 각 point-count × stress마다 12회;
- 총 사례: `2 × 6 × 12 = 144`;
- dense reference: 4096 points;
- reconstructed-surface samples: 1024;
- F-score 거리 threshold: characteristic length의 `0.025`.

각 사례는 case seed로부터 하나의 proper 3D rotation을 만들고 observed/reference에
같이 적용한다. 따라서 latent stress family는 유지하면서 world-coordinate 정렬은
제거한다. Stress identity, true layer label, dense reference와 baseline endpoint는
평가에만 사용한다.

Spatial outlier, `N < 160`, 교차하거나 전역 분리가 불가능한 표면은 이 양성
패널에서 제외한다. 이는 Phase 8에서 이미 실패 경계로 확인된 범위를 숨기는 것이
아니라, 별도의 제한된 confirmatory claim을 검사하기 위한 사전 범위다.

## Frozen methods

### Candidate

Phase 7의 shared-quadratic trend fit, residual two-means, observed-only sampling
gate, inferred layer별 2D Delaunay를 변경 없이 사용한다. Gate parameter는
`k=12`, minimum cluster fraction `0.20`, separation SNR `3.0`, cross-kNN
fraction `0.05`이다.

### Ablation

Phase 2의 global-normal two-layer inference와 같은 층별 Delaunay 구성을 사용한다.
이는 shared-trend inference가 실제로 필요한지를 확인하는 ablation이다.

### Strong alpha-shape comparators

- B5 PCA-anisotropic: scale multiplier `2.80293354289327`, maximum normal
  penalty `4.0`;
- M1 regular weighted alpha: weight scale `0.375`, scale multiplier
  `2.5009326930224836`.

M1 값은 독립 calibration artifact
`benchmark-out/m1_weighted_alpha_ablation.json`의 B4-dominating design point이며,
그 artifact의 SHA-256은
`78831ffaf2a43409fbc17ef4e79447041eb8c946a9bb48e465626fa64e799c66`이다.
이 값은 M1의 B5 우월성을 의미하지 않으며 Phase 50에서 재보정하지 않는다.

## Frozen gates

모든 144 cases를 포함해 다음을 동시에 요구한다.

1. **Safety:** candidate false-safe 0; 전체 safe-accept coverage ≥95%; 모든
   point-count × stress subgroup coverage ≥90%.
2. **Geometry:** candidate mean F-score가 B5와 M1 각각보다 최소 0.10 높고,
   mean normalized Chamfer+Hausdorff loss가 각각보다 작다.
3. **Casewise efficacy:** candidate가 B5 및 M1 각각에 대해 최소 75%의 cases에서
   더 높은 F-score를 보인다.
4. **Topology:** candidate의 aggregate component/Betti/true-layer bridge error와
   nonmanifold edge가 0이고, B5와 M1의 aggregate topology error는 각각 0보다
   크다.
5. **Ablation:** candidate safe accept 수가 global-normal base보다 적지 않고,
   base false-safe 중 최소 1건을 candidate safe accept로 복구한다.

하나라도 실패하면 `phase50_supported=false`이다. 결과를 확인한 뒤 threshold,
case set, candidate 또는 baseline parameter를 조정하지 않는다.

## 주장 경계

통과 시 허용되는 주장은 다음 한 문장으로 제한한다.

> 충분한 sampling과 전역 분리가 가능한 non-outlier synthetic two-layer
> 표면에서 shared-trend layer inference와 layer-constrained connectivity가
> frozen B5/M1 alpha-shape comparator보다 geometry와 topology를 개선했다.

통과하더라도 outlier robustness, arbitrary surface, general alpha selection,
PFTF/local-SPD superiority, real-scan transfer, exactness 또는 deployment는
지지하지 않는다.
