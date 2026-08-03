# PFTF-alpha Codex handoff (2026-08-03)

## Latest continuation: Phase 32

Phase 31 was committed on `main` as `72e4d0d` (`Add Open3D real scan intake
phase 31`) and was not pushed.

Phase 32 in `docs/THREEDMATCH_REGISTRATION_GUARD_PHASE32.md` uses the official
3DMatch `7-scenes-redkitchen` real-fragment geometric-registration benchmark.
The verified 39,950,111-byte fragment archive contains 60 binary XYZ PLY files;
the 140,319-byte evaluation archive contains `3dmatch.log`, `gt.log`, and
`gt.info`. Their SHA-256 values are respectively
`7cb9a1c9236e6833e910692b1d3f572b970c3fc3493e7641c28f1a45841fa51c`
and `ff3eaa243025a0cdf6dd1ca5364a726acf7c08b36444e49c685e1f014bc4f16e`.
The official page marks 7-Scenes data as non-commercial-use only.

The evaluator enforces a label-blind order: it first materializes every
reciprocal-pair, four-patch, global/local/tail observation from external
`3dmatch.log` predictions and fragment coordinates. Only afterward does it read
and join the official ground-truth transform and information files. The official
non-consecutive filter leaves 531 predictions and 449 ground-truth overlap
pairs. The reproduced baseline has 383 correct and 148 incorrect predictions,
72.13% precision, and 85.30% recall.

The frozen full route accepts 71 predictions at 2 cm: 63 correct and 8 incorrect,
for 88.73% precision but only 14.03% recall and 16.45% base-correct retention.
At 5 cm it accepts 54: 42 correct and 12 incorrect, for 77.78% precision, 9.35%
recall, and 10.97% retention. Both fail the predeclared >=90% correct-retention
gate. Tail evidence removes one correct and zero incorrect predictions at 2 cm;
at 5 cm it removes seventeen correct and four incorrect predictions relative to
the global/local predecessor. Therefore `phase32_supported=false` and
`tail_sensitive_real_registration_supported=false`.

The observed global/local/tail stack is too conservative for real fragment
registration, and the local/tail ordering does not align with official
correctness. Do not tune thresholds on opened redkitchen. If work continues,
use separate 3DMatch scenes for design and untouched scenes for validation, or
run the unchanged negative route on a second scene only as a transfer audit.

Implementation: `src/pftf_alpha/threedmatch_redkitchen.py` and
`src/pftf_alpha/threedmatch_registration_guard.py`; tests:
`tests/test_threedmatch_registration_guard.py`; artifact:
`benchmark-out/threedmatch_registration_guard_phase32.json`, SHA-256
`b7653adda0f0b93f14fda54bb57a4559c4a00863e4f22702b4bc14650442cb4d`.
No push has been performed.

## Latest continuation: Phase 31

Phase 30 was committed on `main` as `d9d40c7` (`Add tail-sensitive local guard
phase 30`) and was not pushed.

Phase 31 opens the first downloadable real paired-scan intake in
`docs/OPEN3D_REAL_PAIR_INTAKE_PHASE31.md`. It uses the official Open3D
`DemoICPPointClouds` archive: 10,829,466 bytes, official MD5
`596cffe5f9c587045e7397ad70754de9`, local SHA-256
`b94e0146c1d48c5edfc11af71b4af39ffca604485668c55a127c3b43203a6bd5`,
and the exact three-PCD-plus-`init.log` member allowlist. The dataset is stated
as CC BY 3.0 by the official Open3D API. Dataset files are ignored under
`benchmark-data/open3d_demo_icp/`.

The dependency-light loader validates the archive, parses the exact binary PCD
schema and transform log, and extracts only safe top-level members. Direction
diagnostics confirm that the inverse log matrix is the source-to-target map for
both pairs: median target-NN distance is 0.027855 m versus 0.454854 m for pair
0--1 and 0.014228 m versus 0.184901 m for pair 1--2.

At 2 cm the reciprocal-NN intake contains 43,342 and 42,596 candidate pairs; at
5 cm it contains 49,086 and 46,989. Eight deterministic 96-pair spatial patches
per pair and threshold produce 32 frozen observations. The unchanged Phase-27
global score passes 30/32, Phase-28 local q95 passes 24/32, Phase-30 tail ratio
passes 20/32, and the full observed stack passes 17/32. The tail-ratio range is
1.06748--4.34164, so the new tail observation remains nontrivial on real scan
coordinates.

This is not labeled correspondence or safety validation. The candidates are
metadata-aligned reciprocal nearest neighbours, physical point identity is
unknown, and no geometry/topology harm endpoint is available. Therefore only
`real_paired_scan_intake_supported=true`; real correspondence, real paired-scan
guard support, trimmed reconstruction, and deployment remain false. Opened
DemoICP patches must not tune any frozen threshold.

Implementation: `src/pftf_alpha/open3d_demo_icp.py` and
`src/pftf_alpha/open3d_real_pair_intake.py`; tests:
`tests/test_open3d_real_pair_intake.py`; artifact:
`benchmark-out/open3d_real_pair_intake_phase31.json`, SHA-256
`d74498640e6032ee5f82726f93f3cf449f06cd8d687d791fda5aa4548510922e`.

The next unresolved step is a real fragment benchmark with independently
defined correspondence positives/negatives or scene-level registration ground
truth. Run a preregistered label-blind evaluation with every Phase-27/28/30
threshold frozen. No push has been performed.

## Latest continuation: Phase 30

Phase 29 was committed on `main` as `f913333` (`Add targeted local residual
challenge phase 29`) and was not pushed. Its fixed challenge established that
q95 local evidence misses a rare isolated harmful peak.

Phase 30 in `docs/TAIL_SENSITIVE_LOCAL_GUARD_PHASE30.md` compares six
observed-only tail summaries across the five Phase 28 design panels, three
Phase 28 fresh panels, and failed Phase 29 Validation A. A candidate must
remove all three Phase-29 residual rows and pass every panel/profile safety
gate. The deterministic focus-first selection freezes

`isolated_tail_ratio = maximum_local_residual / percentile95_local_residual`

with strict cutoff `1.6636368999089541`. It retains 1248/1281 development
focus accepts and passes all nine development panels. Phase-30 bases
`30500804`, `30600804`, and `30700804` are mutually disjoint and have zero
case-seed overlap with all prior full panels and targeted bases 302--304.

Ruff, `git diff --check`, and 284 tests passed before opening Validation A.
All three fresh panels then pass:

- Validation A: harm 51 -> 0 -> 0, focus 222/234, all-safe 222/252;
- Validation B: harm 63 -> 0 -> 0, focus 228/246, all-safe 231/273; and
- final: harm 57 -> 2 -> 0, focus 247/255, all-safe 247/264.

The arrows are original -> Phase 28 score/q95 predecessor -> Phase 30 route.
All introduced-harm counts are zero. The final panel contains two residual
profile rows from one `outliers_01`, N=96, repeat-44 case seed `31341118`.
Tail ratios 1.90915596 and 2.19680994 reject both. Aggregate harm is
171 -> 2 -> 0 and focus retention is 697/735, so
`phase30_supported=true` and the fresh incremental gate passes.

Post-run case inspection found a decision-label-only bug: tail-rejected
harmful original accepts had correct false accept/harm booleans but serialized
as `guarded_decision=accept`. The branch was fixed, a regression test added,
all 285 tests passed, and the exact fixed seeds were deterministically rerun
without changing any scientific rule. Final rejected rows now serialize as
`unsupported_geometry_fail_closed`.

Implementation: `src/pftf_alpha/tail_sensitive_local_guard.py`; tests:
`tests/test_tail_sensitive_local_guard.py`; artifact:
`benchmark-out/tail_sensitive_local_guard_phase30.json`, SHA-256
`84e169e0749daffd7465a91f0b90048267c2fdf7a84d2c7464d14d69b794690d`.

The next unresolved step is external validity: real paired-scan correspondence
or a downloadable benchmark with correspondence noise is required before the
tail-sensitive route can support a real-data or deployment claim. Phase 30 is
the next local checkpoint; no push has been performed.

## Latest continuation: Phase 29

Phase 28 was verified with Ruff, `git diff --check`, and 17 targeted tests,
then committed on `main` as `f7c9cbe` (`Add local spatial residual guard phase
28`). It was not pushed.

Phase 29 tests whether the frozen q95 local observation has fresh incremental
benefit, rather than only combined-route transfer. The challenge protocol in
`docs/TARGETED_LOCAL_RESIDUAL_CHALLENGE_PHASE29.md` fixes N=96, stresses
control/local-bump/outliers-01, 64 repeats, and the unchanged three profiles.
Bases `28400804`--`30100804` collide with historical case seeds under this
expanded repeat count and were rejected without execution. The first three
mutually and historically disjoint bases are fixed as `30200804`, `30300804`,
and `30400804`; no seed search or replacement is allowed.

The evaluator reproduced every Phase 28 design value and gate. Ruff, the diff
check, and all 278 tests passed before Validation A was opened. Validation A is
informative but fails:

- total harm is 57 -> 3 -> 3 for original -> predecessor -> combined;
- focus retention is 251/258, all-safe retention is 253/270, and introduced
  routed endpoint harm is zero; and
- the three predecessor/combined residual rows are the exact, registration,
  and missing views of one N=96, `outliers_01`, repeat-59 case seed `30991223`.

Its three q95 local residuals (`2.85526003`, `2.95746686`, `2.91324421`) remain
below the frozen `3.544133065251552` cutoff, so the local guard rescues none.
Validation B and final remain unopened. The same rows' maximum local residuals
are above the cutoff, but that is post-open diagnosis only and cannot retune
Phase 29. Therefore `phase29_supported=false` and fresh incremental q95-local
rescue is disproved on this fixed challenge. Real correspondence, paired
scans, trimmed reconstruction, and deployment remain unsupported.

Implementation:
`src/pftf_alpha/targeted_local_residual_challenge.py`; tests:
`tests/test_targeted_local_residual_challenge.py`; artifact:
`benchmark-out/targeted_local_residual_challenge_phase29.json`, SHA-256
`bdfea7cc2243b18147ad05d3963d2a46d66d7fdad4005c77e5fcd31c3823b1b6`.

The tail-sensitive synthetic question is addressed by Phase 30 above. Phase 29
is committed as `f913333` and remains unpushed.

## Latest continuation: Phase 28

The complete Phase 10--27 worktree was verified with Ruff, `git diff --check`,
and 266 tests, then committed on `main` as `956731c` (`Add observed-coordinate
validation through phase 27`). It was not pushed.

Phase 27's global score-ordering failure was followed by the preregistered
local/spatial residual guard in
`docs/LOCAL_SPATIAL_RESIDUAL_GUARD_PHASE28.md`. The new observed feature builds
an eight-nearest-neighbor graph from primary 3D coordinates, robustly
standardizes each matched displacement, compares it with the componentwise
median neighbor displacement, and summarizes the local residual field by its
95th percentile. Source labels, injected-outlier IDs, clean references, and
endpoint truth remain unavailable to the route.

Opened development seeds `27500804`, `27600804`, and failed Phase-27
Validation A `27800804` were used only for feature/cutoff design. The one
predecessor residual harmful case has local q95 `3.5441330652515526`; the
frozen strict cutoff is the preceding binary64 value
`3.544133065251552`. The combined route requires both the unchanged Phase-27
score test and this local test. Design-only reproduction passed all five
development panels.

Fresh bases `28100804`--`28300804` have zero actual-case-seed overlap with all
full panels from `20300804`--`25900804` and used/reserved bases
`27500804`--`28000804`. Ruff, diff check, and all 274 tests passed before the
sequential fresh run. Results:

- Validation A: harm 171 -> 0, focus 129/129, all-safe 354/360, introduced
  harm 0; pass;
- Validation B: harm 171 -> 0, focus 126/126, all-safe 361/363, introduced
  harm 0; pass; and
- final: harm 168 -> 0, focus 129/129, all-safe 347/354, introduced harm 0;
  pass.

`phase28_supported=true` and
`local_spatial_residual_guard_synthetic_supported=true` for the preregistered
combined synthetic route. The claim boundary matters: the Phase-27 predecessor
also had zero fresh harm in all three panels. The local guard additionally
rejected 3/2/4 predecessor accepts, all of them non-focus safe cases. Therefore
fresh marginal harmful-case rescue by the local feature is not established.
Real correspondence, paired scans, trimmed reconstruction, and deployment
remain unsupported.

Implementation:
`src/pftf_alpha/local_spatial_displacement.py`,
`src/pftf_alpha/local_spatial_residual_guard.py`, plus local evidence plumbing
in `src/pftf_alpha/matched_pair_stress.py` and
`src/pftf_alpha/frozen_partition_reconstruction.py`. Tests:
`tests/test_local_spatial_displacement.py` and
`tests/test_local_spatial_residual_guard.py`. Artifact:
`benchmark-out/local_spatial_residual_guard_phase28.json`, SHA-256
`2e64562de0d8a7094f8e5e2092c238c90d94649b166a0d9a04e9df597eacc198`.

That unresolved incremental-evidence question is addressed by Phase 29 above.
Phase 28 is committed as `f7c9cbe` and remains unpushed.

## Latest continuation: Phase 27

Phase 26's final cutoff-margin failure was followed by the preregistered
focus-envelope cutoff transfer in
`docs/FOCUS_ENVELOPE_CUTOFF_PHASE27.md`. The twelve features, ridge
coefficients, frozen-partition reconstruction, and endpoint gates remain
unchanged. Phase 26 validation A/B (`27500804`/`27600804`) serve only as cutoff
design evidence; Phase 26 final `27700804` is diagnosis only.

The frozen rule is
`nextafter(maximum Phase-26-A/B focus-safe score, +infinity)`. It reproduced
focus count 255, maximum focus score `0.18181536333942855`, routed-harm count
330, minimum routed-harm score `0.28460336155814553`, gap
`0.10278799821871698`, and cutoff `0.18181536333942858`. All design panels and
the actual-case-seed disjointness audit passed. Fresh bases `27800804`,
`27900804`, and `28000804` have zero mutual or historical overlap.

Ruff, the diff check, and all 266 tests passed before execution. The sequential
run stopped at Validation A, as required:

- Validation A: harmful 153 -> 1, focus 125/126, all-safe 362/366, introduced
  harm 0; fail;
- Validation B `27900804`: unopened; and
- final `28000804`: unopened.

Exact passes with harm 51 -> 0 and focus 42/42. Registration passes with harm
51 -> 0 and focus 41/42. `missing_10pct` fails with harm 51 -> 1 despite focus
42/42. The limiting `missing_10pct/outliers_01`, N=96, repeat 1 case has seed
`28210823`, replicate `528210832`, and perturbation `930210836`; one harmful
outlier remains in six routed faces. Score `0.16754805218690128` is
`0.01426731115252772` below the cutoff.

Validation A reverses the design ordering: its harmful minimum is
`0.16754805218690128`, focus maximum is `0.1970292265872296`, and gap is
`-0.0294811744003283`. Two fresh focus cases score at or above the limiting
harm. A post-hoc lower cutoff would retain 124/126 focus cases and might pass
the declared utility gate, but it is forbidden retuning on an opened panel and
provides no support. No model or protocol element was changed.

`phase27_supported=false` and
`focus_envelope_cutoff_synthetic_supported=false`. Treat base `27800804` as
opened and bases `27900804`/`28000804` as reserved; never reuse any of them as
fresh evidence. The next unresolved decision is whether to stop this global
twelve-feature scalar-cutoff line or preregister genuinely new observed local/
spatial evidence on new case-seed-disjoint cohorts. Real correspondence,
paired scans, trimmed reconstruction, and deployment remain unsupported.

Implementation: `src/pftf_alpha/focus_envelope_cutoff.py`. Tests:
`tests/test_focus_envelope_cutoff.py`. Artifact:
`benchmark-out/focus_envelope_cutoff_phase27.json`, SHA-256
`f44580020bb552fbec2d1288420d62c3a8faa90eb91f06e1bb6337e954e15c83`.
Run with:

```powershell
python -m pftf_alpha.focus_envelope_cutoff `
  --output benchmark-out/focus_envelope_cutoff_phase27.json
```

Phase 10--27 changes remain uncommitted and unpushed in the existing dirty
worktree.

## Latest continuation: Phase 26

Phase 25's subset re-clustering failure was followed by the preregistered
frozen-partition route in
`docs/FROZEN_PARTITION_RECONSTRUCTION_PHASE26.md`. For missing pairs it keeps
the full-primary observed shared-trend layer IDs fixed, restricts them to the
retained primary IDs, and triangulates only within each frozen layer before
applying the unchanged twelve-feature score.

A deletion-only subcomplex was first rejected on design evidence: exact Betti
preservation retained only 1/44 and 1/42 missing-profile focus cases. The
frozen-partition design instead reproduced cutoff `0.19784302031484602`, zero
guarded harm, 100% focus retention, and zero newly introduced endpoint harm in
both design panels.

The continuation also found that sequential base seeds `25700804`--`25900804`
would each duplicate 128 actual case seeds from prior panels. Phase 26 therefore
froze the first three consecutive 216-case sets that are mutually and
historically disjoint: `27500804`, `27600804`, and `27700804`. The artifact's
seed audit confirms every overlap count is zero.

Ruff, diff check, and all 262 tests passed before validation A was opened. The
sequential result was:

- validation A: harmful 168 -> 0, focus 123/123, all-safe 360/360, introduced
  harm 0; pass;
- validation B: harmful 165 -> 0, focus 132/132, all-safe 360/360, introduced
  harm 0; pass;
- final: harmful 165 -> 1, focus 132/132, all-safe 360/360, introduced harm 0;
  exact and registration pass, `missing_10pct` fails.

The final limiting `missing_10pct/outliers_01`, N=96, repeat 1 case has seed
`28110823`, replicate `528110832`, and perturbation `930110836`. Its retained
harmful outlier still participates in five faces. Score `0.19257350714226332`
falls only `0.00526951317258270` below cutoff, so it is accepted. Every opened
panel still has a positive harmful/focus score gap; final's is
`0.05454504822839301`. This is cutoff-margin transfer failure rather than
demonstrated representation overlap. No cutoff was changed.

`phase26_supported=false` and
`frozen_partition_reconstruction_synthetic_supported=false`. Do not reuse
seeds `27500804`--`27700804` as fresh evidence. The next unresolved decision is
whether to preregister a stricter score cutoff using A/B only as design
evidence, then validate on new actual-case-seed-disjoint panels. Real
correspondence, paired scans, trimming, and deployment remain unsupported.

Implementation: `src/pftf_alpha/frozen_partition_reconstruction.py` plus the
evaluator-only frozen endpoint in `src/pftf_alpha/matched_pair_stress.py`.
Tests: `tests/test_frozen_partition_reconstruction.py`. Artifact:
`benchmark-out/frozen_partition_reconstruction_phase26.json`, SHA-256
`051be081dbc56929ece47644b4b776d092c2d3b034d424b1c8e76ab3c84ab1b1`.
Run with:

```powershell
python -m pftf_alpha.frozen_partition_reconstruction `
  --output benchmark-out/frozen_partition_reconstruction_phase26.json
```

Phase 10--26 changes remain uncommitted and unpushed in the existing dirty
worktree.

## Latest continuation: Phase 25

Phase 24's limiting missing-pair case omitted the only harmful source vertex
from the retained-pair evidence. Phase 25 therefore preregistered matched-subset
reconstruction: reconstruct missing profiles from retained primary IDs, then
apply the unchanged Phase-24 twelve-feature score and cutoff. The design
calibration reproduced the routed harmful count 164, focus-safe count 126, and
cutoff `0.19784302031484602`.

An evaluator bug initially omitted the preregistered zero-new-endpoint-harm
gate. It incorrectly opened seeds `25400804`, `25500804`, and `25600804`; their
apparently passing harmful/focus counts are audit-invalid and cannot support
Phase 25. The final panel exposed the bug directly: one accepted originally
safe `missing_10pct/upper_occlusion`, N=96, repeat 6 case (case seed `25760849`,
replicate `525760858`, perturbation `927760862`, score `0.0152761`) acquired
four clean cross-layer faces after trimming.

The corrected implementation and regression test enforce zero
`introduced_routed_endpoint_harm_accept` cases per profile and panel. The
corrected Ruff and diff checks passed and all 257 tests passed. The correct run
then fails closed at design score fit, before opening any later panel in that
run:

- `missing_10pct/upper_occlusion`, N=96, repeat 0, case seed `25000807`, score
  `0.00471408690994185`: 10 newly introduced clean cross-layer faces;
- the same profile/stress/N at repeat 3, case seed `25030828`, score
  `-0.014959621371691989`: 5 newly introduced clean cross-layer faces;
- both are accepted below the frozen cutoff;
- `design_reproduced=true`, `design_gate_passed=false`;
- corrected `validation_a`, `validation_b`, and `final_held_out` are null; and
- `phase25_supported=false` and
  `matched_subset_reconstruction_synthetic_supported=false`.

Seeds `25400804`--`25600804` are nevertheless opened/contaminated by the buggy
run and must never be reused as fresh evidence. Do not retune the score,
cutoff, subset rule, or gate on Phase 25 seeds. The next unresolved decision is
whether to preregister a directly observed clean-cross-layer/topology guard,
develop a reconstruction rule with a no-new-harm invariant, or obtain external
paired acquisition evidence. Real correspondence, paired scans, trimming, and
deployment remain unsupported.

Implementation: `src/pftf_alpha/matched_subset_reconstruction.py` plus the
evaluator-only retained-subset endpoint in
`src/pftf_alpha/matched_pair_stress.py`. Tests:
`tests/test_matched_subset_reconstruction.py`. Artifact:
`benchmark-out/matched_subset_reconstruction_phase25.json`, SHA-256
`e11ddb915c497e9663768be2bffaa4817e35977f4c5597415eae3f2d0a872162`.
Run with:

```powershell
python -m pftf_alpha.matched_subset_reconstruction `
  --output benchmark-out/matched_subset_reconstruction_phase25.json
```

Phase 10--25 changes remain uncommitted and unpushed in the existing dirty
worktree.

## Latest continuation: Phase 24

Phase 23's cutoff-margin failure was followed by the preregistered split-cohort
audit in `docs/SPLIT_COHORT_GUARD_CALIBRATION_PHASE24.md`. It keeps the exact
same twelve observed matched-displacement features and ridge penalty, fits only
the score on seed `24900804`, and reserves seed `25000804` solely for the frozen
quarter-gap cutoff rule.

- Full Ruff, diff check, and 252 pytest tests passed before either full seed was
  opened.
- Score-fit accepts: 528 total, 168 harmful, 360 safe. Its minimum harmful
  score was `0.5934101` and maximum focus-safe score was `0.1192945`.
- Cutoff-calibration accepts: 516 total, 165 harmful, 351 safe, including 126
  focus-safe cases.
- Calibration reversed the required order: minimum harmful `0.02888193`,
  maximum focus-safe `0.13806208`, gap `-0.10918015`.
- Calibration was invalid, so the model used the finite fail-closed cutoff and
  rejected all candidates. This is not a tuned threshold.
- The limiting `missing_10pct/outliers_01` case is N=160, repeat 0, case seed
  `26400819`, with one harmful vertex, five harmful faces, and score
  `0.02888193`.
- All twelve of its features lie within calibration focus-safe coordinate
  ranges. Its nearest focus-safe distance is `0.53698` standardized units,
  inside the focus within-class nearest-neighbor IQR `0.36400--0.60938`.
- `prevalidation_gate_passed=false`, so validations `25100804`/`25200804` and
  final `25300804` were not opened.
- `phase24_supported=false` and
  `split_cohort_guard_calibration_synthetic_supported=false`.

This is stronger than another cutoff-margin miss: a fresh mild harmful case is
locally embedded in the frozen twelve-coordinate focus-safe representation.
Do not recalibrate the same global linear score on seeds `24900804` or
`25000804`, and do not reuse reserved seeds `25100804`--`25300804`. The next
unresolved decision is whether to preregister a genuinely local/spatial
observed-harm signature on entirely fresh seeds or obtain external acquisition
and correspondence evidence. Real correspondence, paired scans, trimming, and
deployment remain unsupported.

Implementation: `src/pftf_alpha/split_cohort_guard_calibration.py`. Tests:
`tests/test_split_cohort_guard_calibration.py`. Artifact:
`benchmark-out/split_cohort_guard_calibration_phase24.json`, SHA-256
`1814fe4a4bc56bd872aa2244565abcdfe7f27ead368aac578d0ee127f522ec30`.
Run with:

```powershell
python -m pftf_alpha.split_cohort_guard_calibration `
  --output benchmark-out/split_cohort_guard_calibration_phase24.json
```

Phase 10--24 changes remain uncommitted and unpushed in the existing dirty
worktree.

## Latest continuation: Phase 23

Phase 22's exact-pair downstream failure was followed by the preregistered
matched-displacement guard transfer audit in
`docs/MATCHED_GUARD_SIGNATURE_PHASE23.md`. It isolates the `exact`,
`registration_0p5deg`, and `missing_10pct` profiles and replaces the Phase-18
rectangle with a fixed 12-coordinate tail/physical-displacement signature. One
ridge score and rejection cutoff are fitted only on fresh training seed
`24400804`.

- Training accepts: 519 total, 168 harmful, 351 safe.
- Frozen cutoff: `0.49009791476937975`.
- Training A: harmful 168 -> 0, focus 120/120, all safe 351/351; all three
  profiles pass.
- Development B seed `24500804`: harmful 159 -> 1, focus 123/123, all safe
  348/351. Exact and registration pass; missing-only fails.
- The limiting `missing_10pct/outliers_01` case is N=256, repeat 0, seed
  `26900822`, with two harmful vertices, twelve harmful faces, and model score
  `0.4460047` below the frozen cutoff.
- Development B failed, so validations `24600804`/`24700804` and final
  `24800804` were not opened.
- `phase23_supported=false` and
  `matched_guard_signature_synthetic_supported=false`.

Post-hoc diagnosis, not a permitted retune, shows the minimum harmful A/B score
is `0.4460047` while the maximum focus-safe score is `0.1739441`. Thus the
opened panels retain a `0.2720606` focus separation and the failure is
single-cohort cutoff-margin transfer, not demonstrated focus-safe overlap. Do
not lower the Phase-23 cutoff on development B and do not reuse reserved seeds
24600804--24800804. A future Phase 24, if pursued, should preregister separate
fresh cohorts for linear-score fitting and conservative cutoff calibration,
then use untouched validation cohorts. Real correspondence, paired scans,
trimming, and deployment remain unsupported.

Implementation: `src/pftf_alpha/matched_guard_signature.py`. Tests:
`tests/test_matched_guard_signature.py`. Artifact:
`benchmark-out/matched_guard_signature_phase23.json`, SHA-256
`7564ed416cb1f1ce30e8690f6291eb6de5c6ab8797146261353fae343db0d84a`.
Run with:

```powershell
python -m pftf_alpha.matched_guard_signature `
  --output benchmark-out/matched_guard_signature_phase23.json
```

Full Ruff and 248 pytest tests passed before execution. Phase 10--23 changes
remain uncommitted and unpushed in the existing dirty worktree.

## Latest continuation: Phase 22

With no measured acquisition metadata in the repository, Phase 21 was followed
by the preregistered observed-coordinate audit in
`docs/MULTIVARIATE_CYCLE_SIGNATURE_PHASE22.md`. It uses a fixed 14-coordinate
signature of within-cycle tangent, normal, and total assignment-cost changes,
then fits one ridge linear score on fresh training seed `23900804`. The cutoff
is immediately above the maximum score of every training cycle that is not
fully correcting.

- Training cycles: 1,867 total, 343 strictly correcting, 1,524 unsafe.
- Cutoff: `0.8124335749442713`.
- Accepted: 276/343 strictly correcting and 0/1,524 unsafe; no profile gained a
  newly introduced mismatch.
- Training A assignment: 176,373/176,832 = `99.74%`.
- Harmful false-safes: 275 -> 1; focus: 183/205 = `89.27%`.
- `mismatch_02` repair: 560/720 = `77.78%`; focus 35/41 = `85.37%`.
- `combined` repair: 346/645 = `53.64%`; focus 25/41 = `60.98%`.
- Training A failed, so development B `24000804`, validations
  `24100804`/`24200804`, and final `24300804` were not opened.
- `phase22_supported=false` and
  `multivariate_cycle_signature_synthetic_supported=false`.

The decisive new boundary is not correspondence alone. In training A's
`missing_10pct/outliers_01`, N=96, repeat 3, seed `24330837`, the presented
pairing is 100% correct and the route applies no cycle, yet one harmful vertex
and six harmful faces are accepted. The matched-displacement peak is `9.26677`,
below the frozen Phase-18 limit `10.92263`. Thus the frozen downstream guard
itself fails fresh-seed transfer under exact retained pairing.

Do not retune the 14 features, ridge model, cutoff, or Phase-18 rectangle on
seed 23900804, and do not reuse reserved seeds 24000804--24300804. The next
unresolved decision is whether to obtain external acquisition correspondence
metadata or preregister a new guard-transfer phase with entirely fresh seeds.
Further correspondence-only refinement cannot resolve the exact-pair harmful
accept. Real correspondence, paired scans, trimming, and deployment remain
unsupported.

Implementation: `src/pftf_alpha/multivariate_cycle_signature.py`. Tests:
`tests/test_multivariate_cycle_signature.py`. Artifact:
`benchmark-out/multivariate_cycle_signature_phase22.json`, SHA-256
`e5fefc3ed2465ea22c43108b39912fc535d3920084bab16287a5f6628931447c`.
Run with:

```powershell
python -m pftf_alpha.multivariate_cycle_signature `
  --output benchmark-out/multivariate_cycle_signature_phase22.json
```

Full Ruff and 244 pytest tests passed before execution. Phase 10--22 changes
remain uncommitted and unpushed in the existing dirty worktree.

## Latest continuation: Phase 21

Phase 20's unnecessary global reassignments were followed by a preregistered
cycle-gated preserve-versus-reassign audit in
`docs/CYCLE_GATED_ASSIGNMENT_PHASE21.md`. The Hungarian permutation is split
into disjoint cycles; a complete cycle is applied only if its relative cost
gain reaches a joint truth-supervised cutoff frozen on the already-open Phase-20
A/B cohorts. The Phase-18 downstream rectangle remains unchanged.

- Frozen cutoff: `0.9830303574551544`, immediately above the maximum
  non-improving gain `0.9830303574551543`.
- Across development A/B, all 3,275 non-improving cycles were rejected, but so
  were 304/750 truth-improving cycles; 446 improving cycles were accepted.
- Development A seed `23300804`: assignment 176,196/176,832 = `99.64%`,
  harmful 260 -> 0, focus 183/220 = `83.18%`.
- Development B seed `23400804`: assignment 176,217/176,832 = `99.65%`,
  harmful 270 -> 0, focus 176/210 = `83.81%`.
- Exact, registration-only, and missing-only profiles preserve 100% assignment
  and focus retention in both panels.
- `mismatch_02` repair is only `62.08%/64.44%`; `combined` repair is only
  `43.80%/44.44%`. Both combined profiles remain below 99% assignment accuracy
  and retain only 50% of safe-focus accepts.
- Both development screens failed. Validation seeds `23600804`, `23700804`
  and final seed `23800804` were not opened.
- `phase21_supported=false` and
  `cycle_gated_assignment_synthetic_supported=false`.

The limiting issue is scalar cycle-gain overlap: excluding every development
non-improving cycle also excludes 40.53% of improving cycles. Do not retune this
cutoff or route on seeds 23300804/23400804, and do not reuse reserved unopened
Phase-21 seeds 23600804--23800804 in a follow-up. The next unresolved decision
is whether independent acquisition metadata are available; without them, a
future synthetic Phase 22 would need a preregistered richer cycle signature and
entirely fresh development/validation seeds. Real correspondence, paired scans,
trimming, and deployment remain unsupported.

Implementation: `src/pftf_alpha/cycle_gated_assignment.py`. Tests:
`tests/test_cycle_gated_assignment.py`. Artifact:
`benchmark-out/cycle_gated_assignment_phase21.json`, SHA-256
`ce869cc421140caca289d14d546f78a8ae97a92b90cf066b4a428735da5a1eb6`.
Run with:

```powershell
python -m pftf_alpha.cycle_gated_assignment `
  --output benchmark-out/cycle_gated_assignment_phase21.json
```

Full Ruff and 240 pytest tests passed before execution. The first full pytest
attempt used an inaccessible system temp directory and produced 225 passes plus
15 setup errors; rerunning with an explicit workspace `--basetemp` passed all
240 tests. The temporary directory was removed. Phase 10--21 changes remain
uncommitted and unpushed in the existing dirty worktree.

## Latest continuation: Phase 20

Phase 19's scalar confidence cutoff was replaced by a preregistered whole-set
Hungarian correspondence audit in `docs/GLOBAL_TANGENTIAL_ASSIGNMENT_PHASE20.md`.
It reused the three-iteration 80%-trimmed alignment and 12-neighbour tangent
frame, minimized tangent-only one-to-one assignment cost, and kept the Phase-18
matched-displacement rectangle frozen.

- Calibration A seed `23300804`: assignment 173,097/176,832 = `97.89%`,
  harmful 260 -> 0, focus 183/220 = `83.18%`.
- Calibration B seed `23400804`: assignment 172,968/176,832 = `97.81%`,
  harmful 270 -> 0, focus 161/210 = `76.67%`.
- `mismatch_02` repair passed at `97.78%/98.33%`, but `combined` repair was
  only `85.24%/86.85%`.
- Even exact profiles introduced 751/780 new wrong pairs and stayed below 99%
  assignment accuracy.
- Both panels failed; final seed `23500804` was not opened.
- `phase20_supported=false` and
  `global_tangential_assignment_synthetic_supported=false`.

The strongest safe-focus failure is calibration-A combined local bump, N=96,
repeat 5, seed `24150863`: five changed rows, four introduced mismatches,
`94.19%` assignment accuracy, and matched-displacement peak `183.5362`. The
tangent-only global objective cannot decide when an already-correct presented
pairing should be preserved. Do not tune its cost or assignment using the
opened A/B cohorts. A future phase needs a preregistered preserve-versus-reassign
confidence signal or external acquisition metadata. Real correspondence,
paired scans, trimming, and deployment remain unsupported.

Implementation: `src/pftf_alpha/global_tangential_assignment.py`. Tests:
`tests/test_global_tangential_assignment.py`. Artifact:
`benchmark-out/global_tangential_assignment_phase20.json`, SHA-256
`7476a839b076bf16fe8c84093fab1aa23e2d7402352ecbb671f58e8bf802f397`.
Full Ruff and 236 pytest tests passed before execution. Phase 10--20 changes
remain uncommitted and unpushed in the existing dirty worktree.

## Latest continuation: Phase 19

Phase 18's pair-ID mismatch failure was followed by a preregistered synthetic
pair-confidence audit. `docs/TANGENTIAL_PAIR_CONFIDENCE_PHASE19.md` froze a
three-iteration 80%-trimmed Kabsch alignment, 12-neighbour primary local
tangent score, joint truth-supervised A/B cutoff, and the unchanged Phase-18
matched-displacement rectangle before fresh seeds were observed.

- Selected pair cutoff: `0.05577587737222289`.
- Across A/B it removed all 2,738 truth-mismatched pairs, but retained only
  251,749/350,926 truth-correct pairs (`71.74%`; required at least 99%).
- Calibration A seed `23000804`: harmful 275 -> 246, focus 220/220.
- Calibration B seed `23100804`: harmful 250 -> 225, focus 210/210.
- Every profile failed the pair-retention and zero-harm requirements.
- Both panels failed, so final seed `23200804` was not opened.
- `phase19_supported=false` and
  `tangential_pair_confidence_synthetic_supported=false`.

The limiting mismatch is calibration-B `local_bump`, N=160, repeat 5, seed
`24950866`: wrong-pair score `0.0557758773722229` versus a same-case correct
maximum `0.11296911439256035`. This is direct score overlap, not a cutoff
tie-break. Do not retune this scalar score, 12-neighbour definition, Kabsch trim,
or cutoff on the opened A/B cohorts. A later phase needs a materially different
correspondence representation or externally measured correspondence metadata;
real correspondence, paired scans, trimming, and deployment remain unsupported.

Implementation: `src/pftf_alpha/tangential_pair_confidence.py`. Tests:
`tests/test_tangential_pair_confidence.py`. Artifact:
`benchmark-out/tangential_pair_confidence_phase19.json`, SHA-256
`a5464935f294975b7219ee81ade20523b741c5ba80f281070f45aa6222babb24`.
Full Ruff and 232 pytest tests passed before execution. Phase 10--19 changes
remain uncommitted and unpushed in the existing dirty worktree.

## Latest continuation: Phase 18

Phase 17's exact externally supplied pair-ID upper bound was stressed without
retuning its robust matched-displacement representation. The protocol was
frozen in `docs/MATCHED_PAIR_STRESS_PHASE18.md` before fresh seeds were run.
Every primary case was evaluated under exact pairs, 0.5-degree registration
rotation, 10% missing pairs, 2% cyclic mismatch, and a combined profile.

- Common rectangle: peak `10.922625244331805`, support `infinity`.
- Calibration A seed `22700804`: harmful 275 -> 0 across profiles.
- Calibration B seed `22800804`: harmful 280 -> 0 across profiles.
- Exact, registration-only, and missing-only each retained 43/43 safe focus
  accepts in both cohorts.
- `mismatch_02` retained 0/43 in both cohorts; `combined` retained 0/43 and
  1/43.
- Both panel gates failed, so final seed `22900804` was not opened.
- `phase18_supported=false` and
  `correspondence_stress_synthetic_supported=false`.

The limiting issue is representation, not a threshold tie-break: 2% wrong IDs
produce the same extreme matched-displacement signature as transient returns.
Do not retune the Phase-18 rectangle or reuse its unopened final seed in a
follow-up. A new phase should preregister a correspondence-confidence or robust
pair-assignment mechanism before using new seeds. Real correspondence, paired
scans, trimming, and deployment remain unsupported.

Implementation: `src/pftf_alpha/matched_pair_stress.py`. Tests:
`tests/test_matched_pair_stress.py`. Artifact:
`benchmark-out/matched_pair_stress_phase18.json`, SHA-256
`7458347ec62916634f4c4ecac53c027ebc5da686d07f84e491394444ce70092e`.
Full Ruff and 225 pytest tests passed. Phase 10--18 changes remain uncommitted
and unpushed in the existing dirty worktree.

## Latest continuation: Phase 17

Phase 15/16의 independent-sampling ambiguity를 제거하는 synthetic 상한선으로
externally supplied exact pair ID를 쓰는 matched-repeat consistency audit을
구현했다. Phase 0는 추가 ROI sample로 B5를 다시 실행한 정책 비교였으므로
중복되지 않는다. 프로토콜은 fresh seed 실행 전에
`docs/MATCHED_PAIR_CONSISTENCY_PHASE17.md`에 고정했다.

- 선택 rectangle: peak `5.493421266362053`, support `infinity`.
- calibration A seed `22400804`: harmful 52 -> 0, focus 43/43 = 100%, 통과.
- calibration B seed `22500804`: harmful 58 -> 0, focus 43/43 = 100%, 통과.
- 조건부 final seed `22600804`: harmful 55 -> 0, focus 43/43 = 100%, 통과.
- `phase17_supported=true`,
  `exact_correspondence_synthetic_supported=true`.

가장 좁은 calibration margin은 B의 max safe-focus peak `5.16830` 대 min
harmful peak `5.49342`이다. final에서는 min harmful `15.07169`, max
safe-focus `4.70490`으로 더 넓었다.

단, 이것은 exact pair ID, zero registration error, no missing correspondence를
가정하는 강한 simulator 결과이다. Simulator는 source/stress truth로 matched
repeat를 생성하지만 guard에는 ordered primary/repeat 좌표만 전달한다.
final provenance violation accept는 57 -> 1이며, 남은 `outliers_01`, N=96,
repeat 6, seed `23060858`은 source outlier vertex를 사용했지만 실제
geometry/topology harm은 0이었다. 따라서 strict source 제거를 주장하지
않는다.

`real_correspondence_supported=false`, `real_paired_scan_supported=false`,
`trimmed_reconstruction_supported=false`, `deployment_supported=false`이다.
다음 단계는 threshold 재튜닝이 아니라 correspondence error, missing pair,
registration perturbation을 넣는 별도 stress audit이어야 한다.

구현은 `src/pftf_alpha/matched_pair_consistency.py`, 테스트는
`tests/test_matched_pair_consistency.py`, 재생 산출물은
`benchmark-out/matched_pair_consistency_phase17.json`에 있다. 전체 Ruff와
219개 pytest가 통과했다. Phase 10--17 변경은 아직 commit/push하지 않은
작업 트리 변경이다.

## Latest continuation: Phase 16

Phase 15의 safe rejection을 분석해 replicate 내부 LOO 예측오차와 query
leverage로 residual을 studentize하는 별도 Phase 16을 구현했다. 프로토콜은
새 seed 실행 전에 `docs/STUDENTIZED_PAIRED_SCAN_PHASE16.md`에 고정했고,
Phase 15 A/B는 개발 진단으로만 취급했다.

- 선택 rectangle: peak `1.2689290645115756`, support
  `0.9295682875345601`.
- calibration A seed `22100804`: harmful 55 -> 0, focus retention
  18/43 = 41.86%, 실패.
- calibration B seed `22200804`: harmful 54 -> 0, focus retention
  11/42 = 26.19%, 실패.
- 두 calibration이 90% 기준에 실패했으므로 final held-out seed
  `22300804`는 열지 않았다.

제한 사례는 calibration A의 `outliers_03`, N=96, repeat 6, seed
`22660861`이다. evaluation-only harmful point의 raw residual `0.09630`이
predictive scale `0.10359`로 나뉘어 score `0.92957`까지 낮아졌다. 따라서
LOO/leverage uncertainty가 safe 곡률 오차뿐 아니라 검출해야 할 outlier
신호까지 흡수했다. zero harm 제약에서 최저 focus retention을 최대화한
결과가 26.19%이므로 threshold tie-break 문제가 아니다.

`phase16_supported=false`, `paired_synthetic_supported=false`이다. 같은
studentized score의 LOO quantile, leverage factor, rectangle을 이 cohort에
다시 맞추지 않는다. real paired-scan registration, trimming, deployment는
계속 시작하지 않는다.

구현은 `src/pftf_alpha/studentized_paired_scan.py`, 테스트는
`tests/test_studentized_paired_scan.py`, 재생 산출물은
`benchmark-out/studentized_paired_scan_phase16.json`에 있다. 전체 Ruff와
215개 pytest가 통과했다. Phase 10--16 변경은 아직 commit/push하지 않은
작업 트리 변경이다.

## Latest continuation: Phase 15

Phase 14의 single-scan 한계를 넘기 위해 정보 집합을 독립 반복 스캔 한
개로 명시적으로 확장한 paired-scan persistence audit을 구현했다.
프로토콜은 새 seed 실행 전에
`docs/PAIRED_SCAN_PERSISTENCE_PHASE15.md`에 고정했다.

- 선택 rectangle: peak `3.866421328274521`, support `infinity`.
- calibration A seed `21800804`: harmful 56 -> 0, focus retention
  36/42 = 85.71%, 실패.
- calibration B seed `21900804`: harmful 56 -> 0, focus retention
  37/43 = 86.05%, 실패.
- 사전 기준은 두 패널 모두 zero harm와 focus retention 90% 이상이다.
- 두 calibration이 실패했으므로 final held-out seed `22000804`는 열지
  않았다.

zero harm인 두-cohort rectangle 중 선택된 경계가 최저 focus retention을
최대화한 결과이므로, 같은 peak/support 표현의 threshold 재조정으로 두
패널을 모두 90%까지 올릴 수 없다. 가장 제한적인 harmful 사례는
calibration A의 `outliers_01`, N=96, repeat 5, seed `22250851`이며 peak
`3.8664213282745212`, support `1.7440799339827457`이다.

따라서 `phase15_supported=false`, `paired_synthetic_supported=false`이다.
반복 synthetic 관측은 분리를 개선했지만 선언한 safety/retention gate를
통과하지 못했다. real paired-scan registration, correlated artifact,
trimming, deployment는 시작하지 않는다. 같은 두 score의 threshold나
calibration cohort를 더 늘리는 것도 다음 단계가 아니다.

구현은 `src/pftf_alpha/paired_scan_persistence.py`, 테스트는
`tests/test_paired_scan_persistence.py`, 재생 산출물은
`benchmark-out/paired_scan_persistence_phase15.json`에 있다. 전체 Ruff와
211개 pytest가 통과했다. Phase 10--15 변경은 아직 commit/push하지 않은
작업 트리 변경이다.

## Latest continuation: Phase 14

Phase 13 뒤에 또 다른 threshold guard를 만들지 않고, 관측 좌표와 inferred
layer에서 얻는 14-feature signature의 identifiability audit을 구현했다.
프로토콜은 새 seed 실행 전에
`docs/OBSERVED_IDENTIFIABILITY_AUDIT_PHASE14.md`에 고정했다.

- calibration seed `21600804`, leave-one-out: harmful 51/54 correct =
  94.44% recall, safe-focus 42/42 = 100% specificity, 실패.
- held-out seed `21700804`: harmful 47/51 correct = 92.16% recall,
  safe-focus 39/40 = 97.50% specificity, 실패.
- 누락 harmful 7개 중 6개는 1% contamination이다.
- held-out safe false alarm은 N=96 local bump seed `22520842` 한 건이다.

spacing, layer imbalance, gap/thickness, insertion influence, multiscale
residual을 함께 사용해도 일부 harmful 사례가 calibration safe-focus
사례에 더 가까웠다. 따라서 `feature_identifiable=false`이고
`guard_supported=false`이다. 이는 절대적 비식별성 증명은 아니지만,
선언한 single-scan 14-feature 경로는 닫는다.

같은 좌표에서 feature나 threshold를 더 늘리지 않는다. 다음 단계가
필요하면 반복 스캔 persistence 또는 sensor confidence처럼 정보 집합을
명시적으로 늘리는 별도 Phase 15를 먼저 사전 등록해야 한다. trimming,
real scan, deployment는 계속 시작하지 않는다.

구현은 `src/pftf_alpha/observed_identifiability.py`, 테스트는
`tests/test_observed_identifiability.py`, 재생 산출물은
`benchmark-out/observed_identifiability_phase14.json`에 있다. 전체 Ruff와
207개 pytest가 통과했다. Phase 10--14 변경은 아직 commit/push하지 않은
작업 트리 변경이다.

## Latest continuation: Phase 13

Phase 12의 insertion-influence 표현은 고정하고, 두 calibration cohort의
최저 유지율을 우선하는 보수적 경계를 선택하는 Phase 13을 구현했다.
프로토콜은 새 seed 실행 전에
`docs/CONSERVATIVE_INFLUENCE_CALIBRATION_PHASE13.md`에 고정했다.

- 선택 rectangle: peak `0.44327071014047154`, support `infinity`.
- calibration A seed `21300804`: harmful 56 -> 0, focus retention
  40/44 = 90.91%, 통과.
- calibration B seed `21400804`: harmful 49 -> 0, focus retention
  39/43 = 90.70%, 통과.
- 조건부 final held-out seed `21500804`: harmful 55 -> 1, focus retention
  42/44 = 95.45%, 실패.
- 남은 사례: `outliers_01`, N=96, repeat 2, seed `21920830`, peak
  `0.2359290560051831`, support `0.1371204515389834`, harmful vertex 1개와
  harmful face 4개.

세 Phase-13 패널을 합친 사후 전수 탐색에서도 zero harm와 각 패널 90%
focus retention을 함께 만족하는 peak/support rectangle은 0개였다. zero
harm에서 가능한 최선의 최저 retention은 43.18%뿐이다. 따라서 이번
실패는 calibration tie-break가 아니라 insertion-influence peak/support
표현의 관측 중첩이다.

`phase13_supported=false`이며 같은 표현에서 threshold나 calibration
cohort를 더 추가하지 않는다. 다음 단계가 필요하면 새 guard를 바로
만들기 전에, 저영향 harmful 사례와 safe local feature 사이의
observed-data identifiability audit을 별도 Phase 14로 사전 등록해야 한다.
trimming, real scan, deployment는 계속 시작하지 않는다.

구현은 `src/pftf_alpha/conservative_influence_calibration.py`, 테스트는
`tests/test_conservative_influence_calibration.py`, 재생 산출물은
`benchmark-out/conservative_influence_calibration_phase13.json`에 있다.
전체 Ruff와 203개 pytest가 통과했다. Phase 10--13 변경은 아직
commit/push하지 않은 작업 트리 변경이다.

## Latest continuation: Phase 12

Phase 11의 scalar residual cutoff을 다시 맞추지 않고, 각 점을 국소
이차곡면에 삽입했을 때 이웃 예측이 변하는 정도를 peak/support 두 축으로
보존하는 Phase 12를 구현했다. 프로토콜은 새 seed 실행 전에
`docs/LOCAL_INSERTION_INFLUENCE_PHASE12.md`에 고정했다.

- calibration seed `21100804`: harmful false-safe 52 -> 0,
  clean/local-bump retention 43/43 = 100%, 통과.
- 고정 rectangle: peak `0.6715108036751242`, support
  `0.5445668538984977`.
- 조건부로 한 번 실행한 held-out seed `21200804`: harmful false-safe
  52 -> 1, clean/local-bump retention 42/42 = 100%, 실패.
- 남은 사례: `outliers_03`, N=160, repeat 7, seed `22770871`, peak
  `0.5046093406788189`, support `0.4096165294090569`, harmful vertex 4개와
  harmful face 22개.

두 패널을 합친 사후 진단에는 zero harm와 각 패널 focus-safe retention
90%를 함께 만족하는 rectangle이 98개 있었다. 따라서 표현의 완전한
분리 실패라기보다 calibration margin이 held-out으로 전이되지 않은
실패다. 하지만 이 경계들은 held-out을 본 뒤 얻었으므로 Phase 12를
구제하는 데 사용하지 않았다.

`phase12_supported=false`이며 trimming, real scan, deployment는 시작하지
않는다. 다음 단계가 필요하면 Phase 12 held-out을 재사용해 threshold를
맞추지 말고, 새 seed를 사용하는 conservative margin 또는 nested
calibration Phase 13을 먼저 사전 등록해야 한다.

구현은 `src/pftf_alpha/local_insertion_influence.py`, 테스트는
`tests/test_local_insertion_influence.py`, 재생 산출물은
`benchmark-out/local_insertion_influence_phase12.json`에 있다. 전체 Ruff와
198개 pytest가 통과했다. Phase 10--12 변경은 아직 commit/push하지 않은
작업 트리 변경이다.

## Latest continuation: Phase 11

Phase 10의 단일 국소 평면 점수를 재조정하지 않고, 표현을 다중 규모
leave-one-out 국소 이차곡면으로 바꾼 Phase 11을 구현하고 검증했다.
프로토콜은 실행 전에
`docs/MULTISCALE_QUADRATIC_CONSENSUS_PHASE11.md`에 고정했다.

- 보정 seed `20900804`: harmful false-safe 53 -> 0,
  clean/local-bump retention 36/40 = 90.00%, 통과.
- 보정으로 고정된 threshold: `7.203925635649806`.
- 조건부로 한 번 실행한 held-out seed `21000804`: harmful false-safe
  54 -> 1, clean/local-bump retention 41/42 = 97.62%, 실패.
- 남은 사례: `outliers_03`, N=96, repeat 1, seed `21510826`, score
  `7.087467354965811`, harmful vertex 2개와 harmful face 12개.
- held-out harm를 없애는 사후 cutoff는 calibration retention을 35/40 =
  87.50%로 낮추므로 두 패널을 동시에 만족하는 scalar cutoff는 없다.

따라서 `phase11_supported=false`이며 threshold를 다시 조정하지 않았다.
trimmed reconstruction, real scan, deployment는 여전히 시작하지 않는다.
다음 단계가 필요하면 동일 점수의 재튜닝이 아니라, 한 사례의 최대값을
넘어서는 set-valued/influence evidence처럼 표현 자체가 다른 Phase 12를
별도 calibration/held-out 프로토콜로 먼저 고정해야 한다.

구현은 `src/pftf_alpha/multiscale_surface_consensus.py`, 테스트는
`tests/test_multiscale_surface_consensus.py`, 재생 산출물은
`benchmark-out/multiscale_surface_consensus_phase11.json`에 있다. 전체
Ruff와 194개 pytest가 통과했다. 현재 Phase 10과 Phase 11 변경은 아직
commit/push하지 않은 작업 트리 변경이다.

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

## Phase 10 후속 실행 결과

Phase 10을 seed `20800804`의 새 216-case 패널에서 한 번 실행했다.
local tangent-plane leave-one-out consensus와 provenance/harm 분리 endpoint를
구현했지만 사전 게이트는 실패했다.

- harmful-outlier false-safe: 55 -> 1 (요구값 0)
- source-provenance violation accept: 57 -> 1 (별도 진단값)
- clean/local-bump safe retention: 23/39 = 58.97% (요구값 90% 이상)
- 모든 harmful case를 제거하는 cutoff에서는 retention이 38.46%뿐이며,
  retention 90% cutoff에서는 harmful case 두 건이 남는다.

따라서 동일 score의 threshold를 재조정하지 않는다.
`phase10_supported=false`, `trimmed_reconstruction_supported=false`,
`real_scan_supported=false`, `deployment_supported=false`이다. 사전 순서에
따라 trimmed reconstruction과 real-scan 검증은 시작하지 않았다. 상세
프로토콜과 결과는 `docs/LOCAL_SURFACE_CONSENSUS_PHASE10.md`에 있다.

다음 연구 단계가 필요하다면 이 패널을 다시 튜닝하지 말고, 별도
calibration/held-out seed로 local quadratic 또는 multi-neighbourhood
consensus처럼 표현 자체가 다른 가설을 먼저 등록해야 한다.

## 재현 명령

저장소 루트 `D:\__PFTF_Projects(2026)\PFTF_alpha_dev`에서 실행한다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q

.\.venv\Scripts\python.exe -m pftf_alpha.shared_trend_inference --output benchmark-out/shared_trend_inference_phase7.json
.\.venv\Scripts\python.exe -m pftf_alpha.sensor_stress --output benchmark-out/sensor_stress_phase8.json
.\.venv\Scripts\python.exe -m pftf_alpha.outlier_guard --output benchmark-out/outlier_guard_phase9.json
.\.venv\Scripts\python.exe -m pftf_alpha.local_surface_consensus --output benchmark-out/local_surface_consensus_phase10.json
.\.venv\Scripts\python.exe -m pftf_alpha.multiscale_surface_consensus --output benchmark-out/multiscale_surface_consensus_phase11.json
.\.venv\Scripts\python.exe -m pftf_alpha.local_insertion_influence --output benchmark-out/local_insertion_influence_phase12.json
.\.venv\Scripts\python.exe -m pftf_alpha.conservative_influence_calibration --output benchmark-out/conservative_influence_calibration_phase13.json
.\.venv\Scripts\python.exe -m pftf_alpha.observed_identifiability --output benchmark-out/observed_identifiability_phase14.json
.\.venv\Scripts\python.exe -m pftf_alpha.paired_scan_persistence --output benchmark-out/paired_scan_persistence_phase15.json
.\.venv\Scripts\python.exe -m pftf_alpha.studentized_paired_scan --output benchmark-out/studentized_paired_scan_phase16.json
.\.venv\Scripts\python.exe -m pftf_alpha.matched_pair_consistency --output benchmark-out/matched_pair_consistency_phase17.json
.\.venv\Scripts\python.exe -m pftf_alpha.matched_pair_stress --output benchmark-out/matched_pair_stress_phase18.json
.\.venv\Scripts\python.exe -m pftf_alpha.tangential_pair_confidence --output benchmark-out/tangential_pair_confidence_phase19.json
.\.venv\Scripts\python.exe -m pftf_alpha.global_tangential_assignment --output benchmark-out/global_tangential_assignment_phase20.json
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
- `docs/LOCAL_SURFACE_CONSENSUS_PHASE10.md`
- `docs/MULTISCALE_QUADRATIC_CONSENSUS_PHASE11.md`
- `docs/LOCAL_INSERTION_INFLUENCE_PHASE12.md`
- `docs/CONSERVATIVE_INFLUENCE_CALIBRATION_PHASE13.md`
- `docs/OBSERVED_IDENTIFIABILITY_AUDIT_PHASE14.md`
- `docs/PAIRED_SCAN_PERSISTENCE_PHASE15.md`
- `docs/STUDENTIZED_PAIRED_SCAN_PHASE16.md`
- `docs/MATCHED_PAIR_CONSISTENCY_PHASE17.md`

## 논문 분리 원칙

- 기존 PFTF-alpha 논문: PFTF local SPD metric의 negative/limits result,
  M1 density baseline, G4 fail-closed fallback을 중심으로 유지한다.
- 긍정적 후속 논문 후보: Phase 2--9의 two-layer reconstruction을 별도
  축으로 발전시킨다. 현재는 synthetic bounded claim이며 Phase 10과
  real-scan 검증 전에는 투고 수준의 일반화 주장을 하지 않는다.
- 모든 산출물에서 `promotion_supported=false` 경계를 유지한다.
