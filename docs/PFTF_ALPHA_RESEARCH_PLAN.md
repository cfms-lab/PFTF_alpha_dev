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

### Schema-18 exact-connectivity shadow checkpoint (2026-07-24)

- Added `--evaluate-exact-connectivity-shadow`, which accepts only cells retained
  by the schema-17 host validator and rebuilds an evaluation-only B0-P2
  filtration through `AlphaFiltration.from_top_simplices`.
- The factory preserves caller cell ordering for primary-result compatibility but
  rejects invalid integer shape, bounds, repeated vertices, duplicate cells, and
  incomplete input-point coverage before constructing any filtration values.
- Requested methods are rerun only in the shadow. Non-runtime outputs and the
  bridge-risk probe use relative `1e-12` and absolute `1e-15` comparison
  tolerances; primary case reports and selection cannot be replaced.
- The frozen held-out run supplied no backend and therefore produced zero
  accepted cases, zero shadow runs/differences, and six null shadow reports. The
  schema-17 config, exact-predicate audit, and all 48 primary non-runtime outputs
  were exactly unchanged.
- A successful Qhull-connectivity fixture covers the injection path, but its
  exactness attestation is test-only and is not evidence of exact Qhull behavior.
- This path has exact-validated connectivity but floating circumsphere values;
  it is not an exact alpha complex or a deployed fallback. An actual exact backend
  remained the next gated step at schema 18.

### Schema-19 built-in exact-backend checkpoint (2026-07-24)

- Added `--exact-python-backend`, which invokes
  `python -m pftf_alpha.exact_python_backend` through the schema-17 protocol.
  It maps every exact rational binary64 coordinate to one common integer scale,
  enumerates every four-point candidate, and retains cells with exact empty
  circumspheres. No SciPy/Qhull connectivity is consumed by construction.
- The backend is intentionally limited to at most 64 points. Coplanar
  candidates are skipped, while an exact empty-cosphere ambiguity fails closed
  because this implementation does not claim a symbolic perturbation policy.
- On the frozen six-case, 48-point held-out panel, the independent host accepted
  all six responses. Exact backend tetrahedron counts were 208, 198, 188, 187,
  209, and 141. The schema-16 predicate totals remained unchanged.
- All six validated connectivities matched the primary connectivity. Six shadow
  filtrations ran with zero non-runtime output differences, and the entire
  primary non-runtime case payload was exactly equal to schema 18.
- The exact-predicate artifact now records
  `exact_construction_backend_integrated=true`, so the missing-backend readiness
  blocker is removed for this run. Promotion remains false: connectivity is
  shadow-only, selection is unchanged, and the exact shadow is not deployed.
- The claim is limited to exact Euclidean Delaunay connectivity on the tested
  non-cospherical small panel. Alpha filtration values remain floating-point;
  this is not CGAL, an exact alpha complex, or an anisotropic Delaunay complex.
- Artifact: `benchmark-out/smoke_b0_p2_exact_python_backend_shadow_held_out.json`
  (438,966 bytes; SHA-256
  `5ce3863eae74fcdec439bba26c142006823abdbb84bf940ca4fce50aef15f5db`).

### Schema-20 exact filtration-value checkpoint (2026-07-24)

- Added `--evaluate-exact-filtration-values` over host-validated exact
  connectivity. Every 0D-3D simplex receives an exact rational intrinsic
  circumsphere and Gabriel empty-ball test; non-Gabriel values are the minimum
  exact immediate-coface value, preserving the alpha-filtration rule.
- Exact computation is independent of the floating circumsphere kernel. It
  rejects structurally invalid connectivity, a nonempty top circumsphere,
  degenerate exact Gram systems, and non-finite float conversion. Canonical
  numerator/denominator/Gabriel records are SHA-256 bound per case.
- The frozen held-out audit covered 5,430 simplices. Floating values matched the
  correctly rounded exact value for 953 simplices and differed for 4,477. The
  maximum absolute error was `3.0440274899232807e-4`, maximum relative error
  `1.70727232787347e-9`, and maximum ULP difference 11,128,782.
- Exact versus floating Gabriel disagreements, exact-tie splits,
  exact-versus-rounded critical-count mismatches, exact-versus-floating
  critical-count mismatches, and adjacent exact-order violations were all zero.
  Thus the frozen panel preserves filtration ordering despite widespread
  bit-level value differences; this is not a general error bound.
- All six connectivity shadows retained zero non-runtime output differences,
  and every primary non-runtime case result remained exactly equal to schema 19.
- Exact values remain audit-only and cannot replace thresholds or selection.
  Promotion remains blocked by nondeployment and the absent CGAL/reference-
  stack comparison; this is not an exact end-to-end alpha evaluation.
- Artifact: `benchmark-out/smoke_b0_p2_exact_filtration_audit_held_out.json`
  (448,635 bytes; SHA-256
  `7c9aa989d122b9b2ed736977a32267e3b2c76cb66a51395747500e32654fc54a`).

### Schema-21 exact-rounded value-shadow checkpoint (2026-07-24)

- Added `--evaluate-exact-value-shadow`, which is legal only with exact
  construction, the exact-filtration audit, and the exact-connectivity shadow.
  This makes the same-connectivity floating shadow the explicit control for
  isolating filtration-value effects.
- For every accepted case, the exact records are recomputed and must match the
  schema-20 audit SHA-256 and simplex count. Correctly rounded binary64 values and
  exact Gabriel flags populate a separate `AlphaFiltration`; primary reports are
  never mutated. Missing, rejected, invalid, or digest-mismatched prerequisites
  fail closed.
- The frozen six-case B0-P2 panel produced non-runtime differences in B2 and B3
  for all six cases. Candidate bookkeeping changed in six cases. Selected-alpha
  fields crossed the declared comparison tolerance in two B2 cases
  (`u_concavity`, `sharp_crease`), while objective/stability fields changed in
  two B3 cases (`u_concavity`, `disconnected_parts`).
- Endpoint and bridge-risk-probe differences were zero. The complete primary
  non-runtime payload, exact-filtration audit, and floating-connectivity shadow
  remained exactly equal to schema 20. This overturns only the overly broad
  inference that stable critical counts/order imply identical method reports;
  it does not demonstrate a held-out endpoint change.
- Values are exact rational only before correctly rounded conversion. Runtime
  thresholds, objectives, resampling, and surface evaluation remain binary64.
  The shadow cannot support promotion and is not CGAL, an end-to-end exact alpha
  complex, an anisotropic exact complex, or a deployed fallback.
- Artifact: `benchmark-out/smoke_b0_p2_exact_value_shadow_held_out.json`
  (614,286 bytes; SHA-256
  `ea0d83f2b0a63f83591b8faeb32a9e710f2799fc970aa79c0a17889814db8e0d`).

### Schema-22 exact critical-index checkpoint (2026-07-24)

- Added `--evaluate-exact-critical-index-audit`, gated on the accepted exact
  backend, exact-filtration audit, floating-connectivity shadow, exact-value
  shadow, and B2 or B3. It fails closed and never mutates primary selection.
- The audit compares floating, exact-rounded, and exact-rational critical counts
  and ordered top-simplex birth-group cell identities. For B2/B3 it records the
  selected critical rank, full-complex digest, regularized-boundary digest, and
  objective/endpoints. For B3 it additionally audits the component/Euler
  signature sequence, budgeted candidate indices, normalized log-radius plateau
  persistence, and topology/stability terms.
- On the frozen six-case 48-point panel, critical-count, birth-group, B3
  signature, and B3 candidate-index mismatch counts were all zero. B2/B3 had
  zero selected-index, selected-complex, and selected-boundary mismatches, and
  B2 objectives/endpoints matched.
- B3 persistence arrays differed bitwise in all six cases; selected persistence
  crossed tolerance only on `disconnected_parts`. With selection fixed, B3
  objectives differed on `u_concavity` and `disconnected_parts`, with one
  topology-term and two stability-term differences. The schema-21 differences
  therefore arise from numeric-radius paths on this panel, not different chosen
  complexes.
- This is not a general invariance result: the focused 16-point test panel can
  select a different critical rank and complex. Exact values are rounded before
  runtime evaluation and resampled filtrations remain floating-point. Promotion,
  exact resampling, CGAL/reference-stack comparison, and end-to-end exact alpha
  evaluation remain blocked.
- The primary non-runtime cases, exact-filtration audit, connectivity shadow,
  and value shadow remained structure-identical to schema 21 after runtime
  fields were excluded.
- Artifact:
  `benchmark-out/smoke_b0_p2_exact_critical_index_audit_held_out.json` (642,885
  bytes; SHA-256
  `5a104add183795ba179eb3b943cc00f9f8135c80c1b6e7481c8c8b3e715a0894`).

### Schema-23 exact-selected-threshold resampling checkpoint (2026-07-24)

- Added `--evaluate-exact-resampling-threshold-audit`, gated on the complete
  schema-22 chain and shared B3 selected critical rank, full complex,
  regularized boundary, and budgeted-candidate position. Nonidentical or missing
  prerequisites fail closed.
- It holds full-surface samples, resampled point subsets, SciPy/Qhull resampled
  connectivity, floating filtration values, and sampling seeds fixed. The sole
  treatment is the floating-selected versus exact-rounded-selected binary64
  alpha threshold applied to each shared resample.
- All six frozen cases were audited and had identical full-surface sample hashes.
  All six selected thresholds differed. Only `u_concavity` and
  `disconnected_parts` changed a resampled full complex and boundary, each in
  one of two repeats.
- Those same two cases were the only stability differences, with absolute mean
  changes `0.00026695180449573357` and `0.010215245512115334`. The other four
  changed neither resampled boundary nor stability.
- Both reported B3 stability values were reproduced in every case. Reproduction
  failures, stability differences without boundary changes, and boundary
  changes without stability differences were all zero.
- Primary cases and every schema-22 prerequisite remained structure-identical
  after runtime fields were excluded. The result isolates binary64 threshold
  crossings on the frozen shared floating resamples.
- No exact resampled connectivity or exact resampled filtration was constructed.
  The audit is not deployment, CGAL parity, end-to-end exact alpha evaluation,
  or evidence that resampling itself is exact.
- Artifact:
  `benchmark-out/smoke_b0_p2_exact_resampling_threshold_audit_held_out.json`
  (667,995 bytes; SHA-256
  `0aa9302181cabd48c4cc99a6a7a3b52e458b133813b76236a95aab52df9fdff9`).

### Schema-24 exact-resampling filtration checkpoint (2026-07-24)

- Added `--evaluate-exact-resampling-filtration-audit`, gated on the complete
  schema-23 identity chain and a reusable exact backend. Each deterministic B3
  resample is reconstructed independently, host-validated, assigned exact
  rational filtration values, then evaluated through correctly rounded values
  at the fixed schema-23 exact-selected threshold.
- All six frozen cases and all 12 resamples were audited. Exact connectivity
  matched SciPy/Qhull in every repeat, while all 12 resamples contained
  floating-versus-exact-rounded filtration differences: 7,723 simplex values
  in total and a panel maximum of 32,653,561 ULP.
- Six repeats in four cases changed the selected complex, regularized boundary,
  and stability. Mean changes were `0.00026695180449573357` for `u_concavity`,
  `0.00030055016177662954` for `opposing_sheets`,
  `0.00012341684299446362` for `torus`, and `0.010775278523207324` for
  `disconnected_parts`. `sharp_crease` and `missing_patch` were unchanged.
- The primary cases and every schema-23 prerequisite remained
  structure-identical after runtime fields were excluded. On this frozen panel,
  the newly observed differences therefore come from exact-rounded resampled
  filtration values rather than connectivity changes.
- The audit remains binary64 after exact rounding and does not deploy exact
  resampling into B3 selection or make objectives and surface evaluation exact.
  It is not a general false-safe certificate, anisotropic exact construction,
  or CGAL/reference-stack parity.
- Artifact:
  `benchmark-out/smoke_b0_p2_exact_resampling_filtration_audit_held_out.json`
  (698,999 bytes; SHA-256
  `190ccdd545e6a4fde31af53ed5706015e202f296684958ce5d2d99fab9d138b9`).

### Schema-25 exact-B3 selection shadow checkpoint (2026-07-24)

- Added `--evaluate-exact-b3-selection-shadow`, gated on the complete schema-24
  chain. It holds the exact full filtration, budgeted candidate indices, and
  sampling seeds fixed, reproduces the exact-value B3 reference with floating
  resamples, then evaluates the same candidates with schema-24 exact resamples.
- All six cases and all 60 candidates were evaluated. Stability and objective
  changed for 28 candidates across five cases: 4 in `u_concavity`, 5 in
  `opposing_sheets`, 4 in `torus`, 7 in `disconnected_parts`, and 8 in
  `missing_patch`. `sharp_crease` was unchanged.
- The selected objective changed in `u_concavity`, `opposing_sheets`, `torus`,
  and `disconnected_parts`, but every case retained the same selected critical
  index, alpha, full complex, regularized boundary, and endpoint metrics.
- The primary cases and every schema-24 result section remained
  structure-identical after runtime fields were excluded. The frozen-panel B3
  argmin is therefore robust to the observed exact-resampling perturbations.
- This remains an evaluation-only binary64 shadow. It does not deploy exact
  selection, make objective or surface arithmetic exact, establish a general
  false-safe certificate, construct an anisotropic exact complex, or provide
  CGAL/reference-stack parity.
- Artifact:
  `benchmark-out/smoke_b0_p2_exact_b3_selection_shadow_held_out.json` (765,053
  bytes; SHA-256
  `5d153c584c6400f3ca5ea8b4a62bbf6dc2069a43ecf3c784d0b8b146d22a61dc`).

### G5 frozen held-out robustness preflight (2026-07-24)

- Added `python -m pftf_alpha.g5_validation` as a separate deterministic
  `pftf_alpha_g5_preflight/v1` artifact. It freezes the reference-free P2
  confidence threshold and one B4/B5/P1/P2 multiplier on calibration before
  any held-out profile is evaluated.
- The declared profiles are base, two-thirds-density sparse, doubled-noise, and
  family-geometry parameter times 0.75. Three paired seeds across six families
  give 72 held-out cases. References and component labels remain
  evaluation-only, and held-out tuning is prohibited.
- The default calibration froze threshold `0.2632006823297974`, achieved
  fallback fraction `0.25144175317185696`, and selected multipliers B4
  `72.1450536742484`, B5 `1.91928528060946`, and P1/P2
  `71.93897148061905`.
- P2 preserved the B4 score guard with zero violations and profile-mean
  fallback fractions 0.27815, 0.28956, 0.22717, and 0.25652. However, it did
  not match the strict per-endpoint B4/B5 envelope in any profile. Mean
  F-score margins were -0.047907, +0.000172, -0.048930, and -0.049247;
  geometry-loss margins were negative in all four profiles. Topology burden
  excess was 1, 1, 1, and 0, and labeled false-bridge-edge excess was 86, 55,
  80, and 4.
- Therefore `endpoint_preflight_supported=false` and
  `promotion_supported=false`. The result is a synthetic robustness preflight,
  not a substitute for real higher-fidelity held-out data, confirmatory
  uncertainty estimates, or the exact/validated fail-closed G4 fallback.
- Artifact: `benchmark-out/g5_frozen_held_out_preflight.json` (1,078,057
  bytes; SHA-256
  `ca1fee74276635b4848ead5a09710d4857c4d232a42aa077b50f6251c62dde69`).

### P1 output-topology candidate — negative result (2026-07-24)

- Goal: an output-level false-bridge intervention beating B4/B5 on the G5
  envelope. Localization was never the blocker (schema-13 boundary risk AUC≈0.99);
  intervention was. Three cheap, predeclared calibration-only probes were run
  before any full build, and all three were negative:
  - Per-cell resampling persistence does not separate labeled bridge from
    non-bridge boundary cells (instability AUC≈0.50): Delaunay connectivity is
    locally stable, so a bridge tetrahedron re-forms whenever its points survive
    subsampling. Probe: `pftf_alpha.bridge_persistence_probe`.
  - Conservative multiplier backoff is family-dependent: it fixes large-gap
    `disconnected_parts` entirely (bridge 28→0, Betti 2→0, F 0.37→0.62) but never
    removes the thin-gap `opposing_sheets` bridge. The single frozen global
    multiplier is badly miscalibrated per-family.
  - A global reference-free fail-closed backoff router cannot pass the both-B4-and-B5
    bar: the reference-free signals that reveal the good `disconnected_parts`
    backoff (unlabeled fit and boundary risk) also silently favour destroying
    legitimate single-component topology (torus Betti 2→15).
- Conclusion: synthetic output heuristics keep hitting the same wall as schemas
  12/14/15. Predeclaration `docs/P3_BRIDGE_EXCISION_DESIGN.md` (annotated with the
  falsification). `promotion_supported` stays false.

### G4 deployed exact/validated fail-closed fallback (2026-07-24)

- Added `pftf_alpha.g4_fallback`: the first path that actually routes the
  selection filtration, unlike the schema 16-25 shadows (all `selection_effect:
  none`). `route_case_filtration` runs the built-in exact Euclidean Delaunay
  backend under host validation and, on acceptance, injects the validated
  connectivity through `run_case_benchmarks(filtration=...)`.
- Observed-data-only trigger (3D, 4≤n≤64). On any refusal — over the 64-point
  cap, exact cospherical ambiguity, host rejection, external backend
  timeout/malformed/nonzero — it fails closed to a floating Qhull construction
  explicitly labeled non-exact, recording the reason. `is_exact_certified` is
  unreachable without full validation; nine failure-mode tests prove no silent
  false-safe. Artifact schema `pftf_alpha_g4_fail_closed/v1`.
- G4 certifies only the base Delaunay connectivity B4/B5/P1/P2 score, not the
  anisotropic PFTF complex, so it does not describe the PFTF complex as exact. The
  64-point cap means the default 96-point panel routes entirely to the
  conservative fallback by design. `promotion_supported` stays false: promotion
  still additionally requires frozen higher-fidelity held-out value beyond both B4
  and B5, which the P1 probes indicate is not reachable with synthetic output
  heuristics. Design: `docs/G4_FAIL_CLOSED_DEPLOYMENT_DESIGN.md`.

### M1 weighted / regular alpha complex (2026-07-25)

- Added `pftf_alpha.weighted_alpha`, the first method that changes the
  **connectivity** rather than rescoring one fixed Delaunay triangulation. It
  builds the regular (weighted / power) triangulation via a 4D lift with density
  weights `w_i = (s * spacing_i)^2`, scores cells by the proper weighted-power
  circumradius over kNN spacing, and selects with the existing adaptive
  machinery. `s = 0` reproduces B4 exactly.
- Manifold refinement was required: scoring the weighted connectivity with B4's
  ordinary circumradius produced nonmanifold / Betti regressions; the proper
  weighted-power circumradius removed them (e.g. `missing_patch` Betti 36 -> 11).
- Predeclared calibration ablation over `s in {0, 0.125, 0.25, 0.375, 0.5}` with a
  reference-free frozen multiplier. At `s = 0.375` M1 **strictly dominates B4 on
  every declared endpoint** (F 0.365 -> 0.384, geometry loss 0.1802 -> 0.1730,
  labeled bridge edges 48 -> 46, identical component error 2, Betti error 9, and
  nonmanifold edges 0) — the first method improvement over B4 in the
  investigation (`m1_dominates_b4 = true`).
- It does not clear the strict "beyond both B4 and B5 on every endpoint" bar: B5
  has 0.007 higher mean F-score and one lower component error while being far
  worse on Betti error (105 vs 9) and manifoldness (175 vs 0). A post-hoc sweep
  confirmed M1 cannot close the F gap without sacrificing its topology advantage
  (`s = 0.45` reaches F 0.421 but Betti 36 / nonmanifold 22), so the residual gap
  is left to M2 (oriented normals). `promotion_supported` stays false. Design:
  `docs/M1_WEIGHTED_ALPHA_DESIGN.md`.

### M2 oriented normals and the thin-gap limit (2026-07-25)

M2 (oriented normals) and the `opposing_sheets` thin-gap false bridge were
explored with cheap predeclared probes; all negative:

- At the default density (80 points) the gap (0.26) is smaller than the median
  kNN spacing (0.39): 43% of kNN edges cross between the two sheets, so
  oriented-normal MST propagation bridges the gap and assigns both sheets the same
  orientation. No **local** method (B5, P1, M1, M2) can separate locally
  interleaved sheets, so `opposing_sheets` at this density is under-resolved.
- Denser sampling resolves the gap at the kNN level (640 points: spacing 0.16,
  cross-fraction 0.04), but plain B4 Delaunay-alpha still bridges it — resolution
  is necessary, not sufficient.
- At the resolved density every tried reconstruction still fails to separate the
  sheets: oriented-normal cell removal reproduced the schema-14 boundary-exposure
  failure (bridge 42 -> 225), normal-offset connectivity exploded Betti error
  (3 -> 64), and per-tet signed inside/outside labeling produced a non-manifold
  mess (bridge 677, Betti 176).

Across the whole investigation, removal / deletion interventions fail universally
by exposing new boundary; only a principled **connectivity change** works, and M1
is the achievable improvement while being powerless on the thin-gap case. The one
untested path is a global graph-cut tetrahedron inside/outside labeling
(Labatut-style) — a separate multi-day build with uncertain payoff.
`promotion_supported` stays false.

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
