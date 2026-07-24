# G4 — deployed exact / validated fail-closed fallback (predeclared design)

Status: **predeclared design; implementation follows in the same change.**
Date: 2026-07-24 (Asia/Seoul).

This is the P2/G4 deliverable from the handoff: turn the exact-construction work
from shadow evidence into a **deployed runtime policy** that actually routes the
selection filtration, fails closed on every backend failure mode, and never
labels an uncertified result as exact/safe. It is written before implementation
so the claim boundary is fixed up front.

Context established by code review: schemas 16-25 built a full exact-construction
+ validation + shadow chain, but every path is `selection_effect: none` /
`changes_benchmark_selection: false`. The primary loop
(`benchmark.py:686`) always builds the filtration with SciPy/Qhull
(`AlphaFiltration.from_points`) and never injects. `run_case_benchmarks` already
accepts a `filtration=` argument validated against `case.points`
(`baselines.py:1012, 1024-1030`) — the single seam a deployed router attaches to.

## 1. What G4 does and does not claim

The only available exact backend is the built-in **exact Euclidean Delaunay**
constructor (`exact_python_backend.exact_delaunay_tetrahedra`,
`MAX_EXACT_POINT_COUNT = 64`). It is **not** an anisotropic construction. Per the
handoff, when exact anisotropic construction is unavailable we must use an
explicitly declared conservative fallback and must not call the PFTF complex
exact.

Therefore G4 certifies exactly one thing: **the base Delaunay connectivity that
B4/B5/P1/P2 score was produced and host-validated by exact rational arithmetic**,
rather than by floating Qhull. It does **not** certify the anisotropic PFTF alpha
complex, the multiplier selection, the surface endpoints, or any false-safe
property of the output. `promotion_supported` stays false.

## 2. Routing policy (per case)

`route_case_filtration(case_id, points, *, backend_command=None,
timeout_seconds, max_point_count=64)` returns `(AlphaFiltration, G4CaseRouting)`.

Observed-data-only trigger (never reads labels/reference/Betti): the exact path
is *attempted* iff the points are 3D and `4 ≤ n ≤ max_point_count`. Otherwise the
trigger declines with a recorded reason. Duplicate and non-finite inputs are
refused earlier by the shared `as_point_array` contract (neither the exact
backend nor the floating fallback can be built from them), so those inputs raise
rather than producing any result — an intentional refusal, not a certified pass.

- **Exact path attempted:**
  - In-process backend: `exact_construction_request` → `exact_backend_response`
    (may raise `ExactPythonBackendError`) → `validate_exact_construction_response`.
  - Optional external backend: `run_exact_construction_backend(command, …,
    timeout_seconds)` (already fails closed on timeout / nonzero exit / oversize
    response / invalid JSON / non-object).
  - **Accepted** (`ExactConstructionCaseResult.accepted` and
    `validated_top_simplices is not None`): build
    `AlphaFiltration.from_top_simplices(points, validated_top_simplices)`.
    `is_exact_certified = True`, provenance `exact_validated_connectivity`.
  - **Rejected / raised / failed:** record the specific reason and fall through
    to the conservative fallback.
- **Conservative fallback (fail closed):** build
  `AlphaFiltration.from_points(points)` (floating Qhull). `is_exact_certified =
  False`, provenance `conservative_floating_fallback`, `failure_reason` set to the
  exact refusal reason (e.g. `point_count_exceeds_exact_backend_limit`,
  `exact_empty_cospherical_ambiguity_not_supported`,
  `duplicate_points_not_supported`, `backend_timeout`, `not_all_input_points_used`,
  any host `rejection_reasons[0]`). The result is explicitly **not exact**.

Because the built-in backend caps at 64 points, the default 96-point G5 panel
routes entirely to the conservative fallback. That is the intended, recorded
behavior, not a bug: the fail-closed path is the common path and must be proven
safe.

## 3. The invariant (what the tests must prove)

For every case exactly one of two disjoint outcomes holds, and it is impossible
to reach the certified outcome without full validation:

1. `is_exact_certified = True` **iff** a backend response was host-validated as
   `accepted` and the deployed filtration's top simplices are exactly its
   `validated_top_simplices`.
2. Otherwise `is_exact_certified = False`, a non-empty `failure_reason` is
   recorded, and the deployed filtration is the floating fallback.

There is no code path that sets `is_exact_certified = True` from a caught
exception, a rejected validation, an over-cap input, or a missing backend. This
is the "no unreported false-safe" property: an uncertified construction is always
labeled uncertified.

## 4. Deployment surface

`run_g4_routed_case(case, *, config, methods, backend_command=None)` routes the
filtration and calls `run_case_benchmarks(case, …, filtration=routed)`. The
per-case `G4CaseRouting` is recorded alongside the benchmark. A panel driver
`evaluate_g4_deployment_panel(...)` emits artifact schema
`pftf_alpha_g4_fail_closed/v1` recording, per case: provenance,
`is_exact_certified`, `exact_backend_requested`, `failure_reason`,
`point_count`, `top_simplex_count`, and panel-level counts
(`exact_certified_case_count`, `fail_closed_case_count`) plus the invariant flag
`no_uncertified_result_labeled_exact = True` and the fixed claim boundary.

On accepted small cases the exact connectivity equals the Qhull connectivity
(schema 19 showed zero differences), so deployment changes *provenance and
certification*, not the numbers — an honest safety deployment, not a result
improvement.

## 5. Predeclared failure-mode tests

Each proves the fallback path is taken and nothing is silently certified:

1. small general-position case (n≤64) → certified, deployed connectivity equals
   the validated exact cells, benchmark runs;
2. over-cap (n>64) → not certified, reason `point_count_exceeds_exact_backend_limit`,
   floating fallback deployed;
3. exact cospherical config (cube corners) → not certified, reason
   `exact_empty_cospherical_ambiguity_not_supported`;
4. duplicate points → refused by the shared input contract before routing (raises);
5. malformed / nonzero external backend command → not certified, floating fallback;
6. invariant sweep: certified ⟺ validated cells deployed; uncertified ⟹ reason +
   floating fallback;
7. artifact schema present and `promotion_supported = false`;
8. determinism.

## 6. Claim boundary that must remain intact

G4 deployment gives a real, tested fail-closed selection path with exact-validated
Euclidean connectivity when available and an explicitly conservative floating
fallback otherwise. It does **not** provide an exact anisotropic PFTF complex,
CGAL/reference-stack parity, a general zero-false-safe surface certificate, or
higher-fidelity held-out evidence. It satisfies the "exact or validated fallback
with no unreported false-safe cases" half of the promotion rule only for the base
Delaunay construction; promotion still additionally requires frozen
higher-fidelity held-out value beyond B4 and B5.
