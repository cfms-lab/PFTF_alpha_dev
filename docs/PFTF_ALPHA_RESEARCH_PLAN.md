# PFTF-alpha 연구 설계

작성일: 2026-07-24
상태: ToDo / 설계선
프로젝트: `PFTF_alpha_dev`

## 1. 문제 정의

Convex hull은 모든 표본을 포함하는 안정적인 외피를 제공하지만 오목부, 내부 빈
공간, 분리된 구조를 표현하지 못한다. Alpha shape은 Delaunay triangulation의
부분 complex를 선택하여 이를 개선하지만, 결과가 scale parameter \(\alpha\)에
민감하다.

CGAL의 3차원 정의에서는 Delaunay simplex \(\tau\)가 빈 circumsphere를 가지며
그 squared radius가 \(\alpha\) 이하일 때 alpha complex에 들어간다.

\[
\tau\in K_\alpha
\quad\Longleftrightarrow\quad
r_\tau^2\le\alpha.
\]

라이브러리에 따라 \(\alpha\)를 radius, squared radius, 또는 inverse scale로
정의할 수 있으므로 모든 benchmark는 convention을 명시하고 물리 길이 단위로
환산해야 한다.

## 2. 핵심 판단

### 2.1 전역 alpha 선택

Alpha shapes는 실수 \(\alpha\) 전체에 대해 정의되지만 실제 complex는 유한한
critical value에서만 변한다. 따라서 전역 alpha 선택은 다음 유한 후보 문제로
두는 것이 정직하다.

\[
\alpha^*
=
\arg\min_{\alpha\in\mathcal A_{\mathrm{critical}}}
J(\alpha).
\]

\[
J(\alpha)=
w_gE_{\mathrm{geometry}}
+w_tE_{\mathrm{topology}}
+w_sE_{\mathrm{stability}}
+w_cE_{\mathrm{complexity}}.
\]

- \(E_{\mathrm{geometry}}\): held-out points 또는 ground-truth surface에 대한
  Chamfer/Hausdorff/F-score
- \(E_{\mathrm{topology}}\): target component, hole, cavity 또는 Betti-number
  오류
- \(E_{\mathrm{stability}}\): subsampling, jitter, noise, outlier 변화에 대한
  민감도
- \(E_{\mathrm{complexity}}\): singular face, non-manifold element, 불필요한
  triangle 수

Ground truth가 없을 때는 topology persistence와 resampling stability를
selection evidence로 사용한다. 대상이 여러 component를 의도하는 경우
“한 component가 되는 최소 alpha”를 정답으로 사용하지 않는다.

### 2.2 PFTF 적용점

전역 \(\alpha\)는 방향 \(n\)이 아니라 하나의 스칼라이므로 PFTF의
Rayleigh-contraction 문제와 직접 동형이 아니다. PFTF는 다음 local field
문제에서 사용한다.

\[
M_i=L_iL_i^T+\epsilon I,\qquad M_i\succ0,
\]

\[
d_{M_i}(p_i,p_j)^2
=(p_j-p_i)^TM_i(p_j-p_i).
\]

PFTF field는 local neighbor의 방향, 거리, normal, density, curvature,
uncertainty 및 source-to-receiver relation을 누적하여 \(M_i\), local scale,
confidence를 제안한다.

- sparse tangent direction: 필요한 연결 범위를 완화
- surface normal direction: 가까운 opposing sheet 사이의 false bridge 억제
- high curvature 또는 noisy normal: confidence 감소
- unreliable region: global-alpha 또는 exact baseline으로 fail closed

## 3. PFTF 구성요소 매핑

| PFTF 요소 | Alpha-shape 역할 | 현재 판단 |
|---|---|---|
| Rayleigh tensor | ellipsoidal local metric 및 방향별 scale | 핵심 후보 |
| M2 gate relaxation | simplex inclusion의 soft calibration | 직접 적용 가능 |
| M3 typed field | density, normal, curvature, uncertainty 유지 | 핵심 후보 |
| M1 virtual boundary | open boundary 또는 missing-data penalty | 선택적 |
| verify/escalate | exact alpha construction과 fallback | 필수 |

Hard inclusion은 calibration 동안 다음과 같이 완화할 수 있다.

\[
g_\tau(\alpha,T)
=\sigma\left(\frac{\alpha-r_{\tau,M}^{\,2}}{T}\right).
\]

Tie가 없는 점에서 \(T\to0^+\)이면 hard gate로 수렴한다. 이 결과는 simplex
포함 여부의 gate relaxation에 관한 것이며, arbitrary nearest-receiver routing의
정당성을 자동으로 주지 않는다.

## 4. 수학·구현 제약

1. Alpha metric은 최소한 symmetric positive-definite여야 한다.
2. PFTF의 antisymmetric part는 Rayleigh scalar에 기여하지 않으며 거리로 직접
   해석하지 않는다.
3. 안정적인 parameterization은 \(M_i=L_iL_i^T+\epsilon I\), matrix exponential,
   또는 bounded eigenvalue decomposition을 사용한다.
4. CGAL weighted alpha shape의 scalar point weight는 arbitrary per-point SPD
   metric과 같지 않다.
5. 공간적으로 변하는 anisotropic metric은 custom anisotropic triangulation,
   affine diagram, 또는 검증 가능한 local approximation이 필요할 수 있다.
6. 서로 다른 local test를 단순히 이어 붙이면 global simplicial-complex
   consistency가 깨질 수 있으므로 closure와 adjacency를 별도로 검사한다.

## 5. 선행 연구와 novelty 경계

Edelsbrunner–Mücke의 alpha-shape family와 CGAL 구현은 critical alpha spectrum,
filtration, regularized/general complex를 제공한다. 따라서 critical-value
enumeration 자체는 기여가 아니다.

Teichmann–Capps(1998)는 이미 다음을 제안했다.

- local sampling density에 따른 alpha scaling;
- point normal에 따른 anisotropic alpha-ball;
- 가까운 surface 사이 false connection 완화.

따라서 다음은 baseline이다.

- \(k\)-NN 거리 기반 local alpha;
- local PCA/normal 기반 ellipsoid;
- density와 normal을 결합한 heuristic.

PFTF의 신규성 후보는 다음으로 제한한다.

1. directed source-to-receiver relation을 metric 또는 confidence에 반영;
2. topology-aware objective와 PFTF field의 공동 calibration;
3. soft gate의 hard-limit consistency 및 exact evaluation 분리;
4. failure confidence와 conservative fallback;
5. frozen held-out에서 prior adaptive/anisotropic baseline보다 우수한 결과.

## 6. 검증 설계

### 6.1 비교군

- B0: convex hull
- B1: fixed global alpha
- B2: critical-alpha exhaustive scan
- B3: persistence/stability-selected global alpha
- B4: \(k\)-NN density-scaled alpha
- B5: normal-based anisotropic alpha
- P1: PFTF local SPD metric
- P2: PFTF metric + confidence + conservative B4 guard; exact fallback pending

### 6.2 데이터 조건

Synthetic family:

- concave U/C shapes;
- torus와 multi-hole shapes;
- thin opposing sheets;
- narrow gap와 sharp crease;
- deliberately disconnected components;
- uniform/non-uniform sampling;
- Gaussian noise, outlier, missing patch.

Real family:

- 공개 CAD/scan point cloud;
- 서로 다른 sensor density 또는 resampling level;
- ground-truth mesh가 있는 표본을 우선 사용.

### 6.3 측정값

- Chamfer distance, Hausdorff distance, surface F-score
- normal consistency
- component, hole, cavity/Betti error
- false bridge와 false hole 수
- watertightness, non-manifold count
- volume/area error
- subsampling/noise stability
- runtime와 memory
- fallback rate와 false-safe count

### 6.4 공정성

- 동일 Delaunay/geometry backend와 exact predicate 설정
- candidate budget와 metric fitting budget 분리 보고
- 모든 weight와 threshold는 held-out 평가 전에 동결
- density-adaptive 및 anisotropic baseline을 생략하지 않음
- global alpha만 이기는 결과를 PFTF 우월성으로 해석하지 않음

## 7. 단계별 게이트

### G0 — 정의와 convention

- 라이브러리별 alpha convention 표 작성
- general/regularized mode와 singular face 정책 고정
- 대상 topology prior를 데이터별로 명시

### G1 — 전역 기준선

- critical-alpha spectrum 열거
- geometry/topology/stability objective 구현
- exhaustive scan을 전역 alpha oracle로 고정

### G2 — 기존 adaptive 기준선

- \(k\)-NN density scaling 구현
- normal/PCA anisotropic alpha 구현 또는 재현
- synthetic thin-gap과 density-shift failure atlas 생성

### G3 — PFTF field

- relation feature와 SPD mapping 정의
- eigenvalue bound와 confidence contract 검증
- soft gate finite-difference와 sharp-limit 수치 검증

### G4 — exact integration

- anisotropic test와 global complex consistency 검사
- exact construction 또는 conservative fallback 연결
- false-safe 0 또는 명시한 fail-closed 정책 확인

### G5 — frozen held-out

- weight/threshold 동결
- unseen shape, density, noise regime 평가
- B4/B5 대비 통계와 실패 사례를 함께 보고

## 8. 성공·중단 기준

성공:

- B4/B5보다 thin-gap false bridge 또는 topology error를 줄임;
- geometry accuracy를 악화시키지 않음;
- noise/density shift에서 안정성 향상;
- confidence fallback이 false-safe를 만들지 않음.

중단 또는 재정의:

- 개선이 fixed global alpha에만 존재;
- local density 또는 PCA normal baseline과 동률;
- SPD projection 후 PFTF 관계 정보가 사라짐;
- custom triangulation 비용이 reconstruction 이득보다 큼;
- topology 개선이 특정 target prior를 누설한 결과임.

## 9. 현재 claim boundary

현재는 연구 설계이며 다음을 주장하지 않는다.

- PFTF가 보편적으로 최적 alpha를 찾는다.
- 하나의 alpha가 물체의 참 형상을 유일하게 정의한다.
- soft complex가 exact alpha complex를 대체한다.
- local anisotropy 자체가 신규 기여다.
- 실제 scan 또는 산업 데이터에서 우월성이 검증됐다.


### 9.1 구현 체크포인트 (2026-07-24)

- G0-G3와 conservative P2 guard 프로토타입은 Python/SciPy 경로에서
  end-to-end smoke를 통과했다. exact G4 integration은 아직 남아 있다.
- P1은 source scale `h_i`와 receiver scale `h_j`의 bounded signed message
  `tanh(log(h_j/h_i)/s)` 및 receiver 방향 불균형을 trace-free relation
  tensor로 집계한다.
- signed relation은 거리로 직접 사용하지 않는다. 고유값을 제한한
  log-eigenvalue mapping으로 SPD를 만들고, unlabeled confidence에 따라
  density-scaled identity 쪽으로 연속 완화한다.
- P2 confidence threshold는 명시적으로 주거나 reference geometry를 읽지
  않는 calibration-only 분위수로 먼저 동결한다. 목표 fallback 0.25는
  calibration cell 1,099개에서 threshold 0.268687과 실제 비율 0.25842를
  만들었다.
- low-confidence top cell에는 `max(P1 score, B4 score)`를 사용하여 P1과
  trusted B4를 모두 통과해야 포함되도록 했다. 이는 fixed Euclidean complex
  위의 conservative guard이며 exact CGAL fallback은 아니다.
- held-out guard 대상은 전체 cell의 7.49-48.99%, 선택된 cell의
  11.01-42.31%였다. score-level 및 selected-set B4 guard 위반은 0건이었고
  complete downward closure와 face incidence 검사도 모두 통과했다. 이는
  exact false-safe 0을 뜻하지 않으며 그 검증은 여전히 G4 과제다.
- 작은 frozen held-out smoke에서 평균 F-score는 P1 0.17634, P2 0.16777였고
  normalized squared Chamfer는 각각 0.00966, 0.01037이었다. P2는 P1보다
  개선되지 않았고 component-error 합도 B4/B5/P1/P2 모두 2였다. 따라서
  현재 결과는 구현 체크포인트이지 우월성 또는 논문 승격 근거가 아니다.
- 출력 삼각 복합체의 GF(2) surface Betti number `(beta_0, beta_1, beta_2)`와
  합성군별 선언 target에 대한 L1 error를 evaluation-only endpoint로 추가했다.
  이 target은 B2/adaptive calibration 또는 reference-free P2 confidence
  selection에 사용하지 않는다. smoke의 Betti-error 합은 B4 20, B5 27,
  P1 25, P2 25였고, P1/P2는 torus만 `(1,2,1)`로 정확히 복원했다. 기존
  component 지표가 놓친 위상 실패가 확인되었으므로 이 결과 역시 승격
  근거가 아니다.
- 합성 관측점의 선언 surface-component label을 보존하고, 출력 mesh에서 서로
  다른 label을 잇는 고유 edge와 혼합 face 수를 evaluation-only endpoint로
  추가했다. P2는 opposing sheets에서 39 edge/39 face, disconnected parts에서
  18 edge/19 face를 남겨 두 multi-component case 모두 false-bridge witness가
  관측됐다. B2도 opposing sheets의 component 수는 2였지만 cross-sheet edge
  46개를 남겼으므로 기존 component proxy의 false negative가 확인됐다. 라벨을
  바꾸어도 adaptive multiplier와 P2 confidence threshold는 동일했고, 이전
  schema-9 smoke와 selection 및 기존 endpoint가 완전히 일치했다. 따라서 현재
  guard는 topology safety certificate가 아니며 general exact G4는 남아 있다.

### Schema-11 bridge-risk checkpoint (2026-07-24)

- Added an evaluation-only per-cell geometric bridge-risk probe. Its risk path
  consumes observed points and fixed Delaunay cells only; references and
  synthetic labels cannot affect selection or risk.
- Globally coherent unoriented normals route to a planar parallel-normal signal;
  other fields route to a density-normalized second-longest-edge signal.
- At frozen thresholds `(0.9, 0.02, 1.8)`, held-out opposing sheets achieved
  AUC 0.99490/recall 0.87402/FPR 0.02817 and disconnected parts achieved
  AUC 1.0/recall 1.0/FPR 0.0.
- All 48 pre-existing non-runtime B0-P2 results matched schema 10, but
  single-component cell flag rates reached 32.98%. The probe therefore remains
  diagnostic and is not a hard guard.
- The calibration-only aggregation/soft-penalty test is now complete; see the
  schema-12 checkpoint below.

### Schema-12 bridge-penalty ablation checkpoint (2026-07-24)

- Added a zero-strength-exact, label-free soft penalty and calibration-only
  audit. The audit never freezes or deploys a strength and leaves requested
  B0-P2 selection unchanged.
- The frozen strength curve `[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]` found
  no promotion-eligible point. Strength 0.01 improved mean geometry/objective
  but increased labeled false-bridge edges from 34 to 35; strength 0.8 reduced
  edges to 25 but regressed geometry and Betti error.
- Selected flagged-cell count and selected mean risk each had Spearman
  correlation -0.26998 with output false-bridge edges. Independent cell-risk
  suppression therefore does not certify boundary bridge suppression.
- The boundary/dual-connectivity localization diagnostic is now complete; see
  the schema-13 checkpoint below.

### Schema-13 boundary bridge-localization checkpoint (2026-07-24)

- Added an evaluation-only frozen-P2 localizer. Boundary edges use the
  route-specific observed-geometry signal and boundary faces use their maximum
  incident-edge risk. Neither references nor component labels enter risk.
- At the unchanged `risk > 1` rule, the frozen held-out panel reached face
  AUC/recall/FPR 0.99523/0.96552/0.02439 over 550 faces and edge
  AUC/recall/FPR 0.98905/0.89474/0.00933 over 807 edges.
- The localizer found all 19 mixed disconnected-parts faces and 37 of 39 mixed
  opposing-sheets faces. Labels were applied only afterward for evaluation.
- Selected-cell dual articulation/bridge structure is recorded separately. Its
  pooled face AUC was only 0.58274, and opposing sheets had no dual cut signal,
  so dual bottlenecks are audit-only and are not fused into geometric risk.
- All 48 pre-existing non-runtime B0-P2 results and the schema-11 cell probe
  matched schema 12. This is localization evidence, not a deployed repair or
  an exact false-safe certificate.
- Next work should test a calibration-only boundary-owner intervention that
  recomputes the boundary after each change and passes geometry, component,
  Betti, and labeled-bridge non-regression gates. Exact integration is pending.

### Schema-14 boundary-owner intervention checkpoint (2026-07-24)

- Added a calibration-only layered intervention. Each round removes all unique
  owners of current boundary faces with `risk > 1` and then recomputes the
  boundary. References and component labels are used only by the promotion gate.
- Depths `[0, 1, 2, 4]` started from objective/geometry 0.41890/0.20157, Betti
  error 25, and labeled false-bridge edges/faces 34/35.
- One round improved objective/geometry to 0.41450/0.19798 and Betti error to 23,
  but exposed 50/53 labeled bridge edges/faces. Two rounds regressed both
  objective/geometry and Betti while still exceeding baseline bridge counts.
- Four rounds removed 121 of 579 selected cells and reduced bridges to 11/12,
  but objective/geometry regressed to 0.45821/0.24757. No positive depth passed
  all objective, geometry, component, Betti, and strict bridge-improvement gates.
- No intervention depth was frozen or deployed. All 48 held-out non-runtime
  B0-P2 results, the schema-11 cell probe, and schema-13 boundary localization
  remained exactly unchanged.
- This rejects local boundary-owner peeling as the next repair. The connected
  region/cut hypotheses are evaluated in the schema-15 checkpoint below.

### Schema-15 boundary region/cut checkpoint (2026-07-24)

- Added fixed calibration-only `baseline`, `largest_risk_region`, and
  `safe_backbone_cut` strategies. Flagged faces join only through flagged edges;
  labels and reference geometry are absent from candidate construction.
- The calibration panel contained 13 connected risk regions, with a largest
  region of 25 faces. The largest-region candidate removed 28 cells across five
  cases and improved Betti error from 25 to 22.
- That candidate nevertheless regressed objective/geometry from
  0.41018/0.19247 to 0.41441/0.19727 and increased labeled bridge edges/faces
  from 34/35 to 43/45, so it failed the global promotion gate.
- Removing flagged edges left seven safe-edge vertex components in total, but no
  flagged edge connected distinct safe components. The safe-backbone strategy
  therefore generated no candidate and remained exactly equal to baseline.
- No strategy was frozen or deployed. All 48 held-out non-runtime B0-P2 results,
  the schema-11 cell probe, and schema-13 boundary localization remained exactly
  unchanged.
- Per-cell, layered-owner, connected-region, and simple safe-backbone-cut
  heuristics have now failed the declared gates. Exact-construction fallback is
  the next implementation path and must remain separately audited before use.

### Schema-16 exact-predicate readiness checkpoint (2026-07-24)

- Added `--evaluate-exact-predicates`, which converts each binary64 coordinate
  to its exact rational value and audits exact 3D orientation and interior-facet
  in-sphere signs on supplied SciPy/Qhull connectivity.
- The frozen held-out audit covered 288 points, 1,131 tetrahedra, and 2,094
  interior facets. It found zero exact degeneracies, cospherical facets,
  local-Delaunay violations, non-manifold facets, or float/exact sign conflicts.
- This is a predicate-consistency preflight for the supplied connectivity, not
  an exact triangulation, alpha-complex construction, or CGAL certificate.
- The JSON records `exact_construction_backend_integrated=false`; promotion is
  blocked until that backend exists. All 48 prior non-runtime B0-P2 results and
  the bridge probe remained exactly unchanged.

### Schema-17 exact-backend handoff checkpoint (2026-07-24)

- Added a versioned stdin/stdout JSON protocol for an explicitly supplied exact
  backend. Requests carry exact rational coordinate pairs; responses echo the
  canonical-request SHA-256 and attest backend identity, kernel, and exactness.
- Host validation requires all input points, manifold face incidence,
  convex-hull supporting boundary facets, exact cell/boundary volume equality,
  and exact orientation/in-sphere consistency.
- The frozen held-out run supplied no backend, accepted zero cases, changed no
  selection, and recorded `no_exact_construction_backend`. Schema-16 exact
  predicate results and all 48 non-runtime B0-P2 results were unchanged.
- Protocol fixtures cover a valid complete bipyramid plus request-binding,
  missing-point, and local-Delaunay rejection. Injected Qhull cells are test
  fixtures only and are not evidence of exact Qhull construction.
- Validated backend cells are not yet applied to B0-P2; promotion stays blocked.

## 10. 근거

- H. Edelsbrunner and E. P. Mücke, “Three-dimensional alpha shapes,”
  *ACM Transactions on Graphics*, 1994.
  <https://pub.ista.ac.at/~edels/Papers/1994-04-3DAlphaShapes.pdf>
- CGAL, “3D Alpha Shapes.”
  <https://doc.cgal.org/latest/Alpha_shapes_3/index.html>
- M. Teichmann and M. Capps, “Surface reconstruction with anisotropic
  density-scaled alpha shapes,” *IEEE Visualization*, 1998.
  <https://doi.org/10.1109/VISUAL.1998.745286>
- PFTF theory core:
  `D:\__PFTF_Projects(2026)\PFTF_dev\draft\PFTF_theory_core.tex`
