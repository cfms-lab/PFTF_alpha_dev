# Cycle-gated global assignment: Phase 21

## Frozen question

Phase 20's tangent-only Hungarian assignment restored zero harm but changed
about 2% of already-correct exact pairs, reducing safe-focus retention. Can the
global permutation be decomposed into independent cycles and only
high-confidence cost-improving cycles be applied, preserving the presented
pairing otherwise?

This is a new preserve-versus-reassign representation. It does not change the
Phase-20 global cost or the frozen Phase-18 downstream guard.

## Frozen information boundary

Routing receives only the presented primary/repeat coordinate sets, their row
pairing for robust alignment, and one scalar cycle-gain cutoff frozen from
prior development cohorts. It receives no pair-source truth, source labels,
stress identity, clean surface, analytic height, or endpoint.

Pair-source IDs are used only on the already-open Phase-20 A/B cohorts to freeze
the development cutoff and on later panels for evaluation. This makes Phase 21
an optimistic supervised synthetic study, not a deployable matcher.

## Frozen candidate assignment

Reuse Phase 20 without modification:

1. three 80%-trimmed Kabsch iterations on presented rows;
2. 12-neighbour primary local normals and tangent spacing with the
   `0.002 * L_obs` floor;
3. the complete tangent-only cost matrix; and
4. deterministic SciPy Hungarian one-to-one assignment.

The identity mapping is the presented pairing. Decompose the candidate
permutation into disjoint nontrivial cycles, visiting the smallest unvisited row
first.

For cycle `C`, define

`gain(C) = (identity_cost(C) - assigned_cost(C)) / max(identity_cost(C), eps)`.

Reported costs exclude the deterministic tie perturbation. Negative numerical
gains are clamped to zero.

## Frozen development cutoff

Use only the already-open Phase-20 calibration seeds `23300804` and `23400804`
as development cohorts. For each candidate cycle, compare the number of
truth-correct pairs before and after applying that cycle. A cycle is
`non-improving` when the after count is less than or equal to the before count.

Freeze the cutoff as the next larger binary64 value above the maximum gain of
all non-improving development cycles. If no non-improving cycle exists, use
zero. At routing time, apply a cycle only when `gain >= cutoff`; otherwise keep
all rows in that cycle at their presented identity mapping.

No endpoint, case decision, stress identity, focus label, or fresh validation
result may affect cutoff selection. No per-profile cutoff, candidate pruning,
post-assignment repair, or second threshold is allowed.

## Frozen development screen

Before fresh seeds are opened, both Phase-20 development panels must satisfy in
every profile:

1. at least 99% aggregate truth assignment accuracy;
2. at least 90% repair of presented mismatches for `mismatch_02` and `combined`;
3. at least one unguarded harmful-outlier false-safe;
4. zero guarded harmful-outlier false-safes; and
5. at least 90% safe control/local-bump retention.

If this screen fails, Phase 21 stops and none of its fresh validation seeds are
opened.

## Frozen downstream guard

Reorder original, unaligned repeat coordinates only for accepted cycles. Then
compute the unchanged Phase-17 robust matched-displacement evidence and reuse
the Phase-18 rectangle exactly:

`peak <= 10.922625244331805 AND support <= infinity`.

All primary reconstruction, simulator, endpoint, profile, severity, and seed
offset rules remain unchanged.

## Frozen fresh validation protocol

If and only if the development screen passes, run the same 1080-row five-profile
panel on:

- Validation cohort A seed: `23600804`.
- Validation cohort B seed: `23700804`.
- Conditional final held-out seed: `23800804`.
- Phase-20's unopened final seed `23500804` is not reused.

Both validation panels must pass the same five per-profile gates before final
held-out is opened. No cutoff or rule may change after the development cycles
are observed.

Even a pass supports only the declared supervised synthetic cycle filter.
`real_correspondence_supported`, `real_paired_scan_supported`,
`trimmed_reconstruction_supported`, and `deployment_supported` remain false.

## Planned run

```powershell
python -m pftf_alpha.cycle_gated_assignment `
  --output benchmark-out/cycle_gated_assignment_phase21.json
```

## Result

The implementation tests and full regression suite passed before the frozen
development screen was executed. The joint development cutoff was
`0.9830303574551544`, the next binary64 value above the largest non-improving
cycle gain (`0.9830303574551543`). This admitted no development-labelled
non-improving cycle, but it also rejected 304/750 truth-improving cycles:

- 4,025 nontrivial candidate cycles were observed across development A/B;
- 3,275 were non-improving and all were rejected;
- 446/750 truth-improving cycles were accepted; and
- 304/750 truth-improving cycles were rejected by the same scalar cutoff.

The preserve-by-default rule fixed Phase 20's needless changes in the exact,
registration-only, and missing-only profiles: those profiles retained their
presented identity pairing at 100% accuracy and 100% safe-focus retention in
both development panels. It did not repair enough of the mismatch-bearing
profiles:

- Development A seed `23300804` reached 176,196/176,832 (`99.64%`) overall
  assignment accuracy, reduced harmful false-safes from 260 to zero, and
  retained 183/220 (`83.18%`) safe-focus accepts.
- Its `mismatch_02` profile repaired 447/720 (`62.08%`) presented mismatches,
  with `99.25%` assignment accuracy and 29/44 (`65.91%`) focus retention. Its
  `combined` profile repaired 279/637 (`43.80%`), reached only `98.91%`
  accuracy, and retained 22/44 (`50.00%`) focus accepts.
- Development B seed `23400804` reached 176,217/176,832 (`99.65%`) overall,
  reduced harmful false-safes from 270 to zero, and retained 176/210
  (`83.81%`) focus accepts.
- Its `mismatch_02` profile repaired 464/720 (`64.44%`), with `99.29%`
  assignment accuracy and 29/42 (`69.05%`) focus retention. Its `combined`
  profile repaired 284/639 (`44.44%`), reached `98.93%` accuracy, and retained
  21/42 (`50.00%`) focus accepts.

Both development panels therefore failed the frozen per-profile gates.
Validation seeds `23600804` and `23700804` and final seed `23800804` were not
opened; their artifact fields are `null`. No cutoff or routing rule was changed
after the development results were observed.

This is observed cycle-gain overlap, not a tie-break failure. A cutoff strict
enough to exclude every development non-improving cycle necessarily rejects
40.53% of development truth-improving cycles. Therefore the single
cycle-relative-gain preserve-versus-reassign route is closed for this protocol;
it must not be retuned on these opened cohorts. A later phase needs independent
acquisition metadata or a preregistered richer cycle signature and fresh seeds.

Thus `phase21_supported=false` and
`cycle_gated_assignment_synthetic_supported=false`.
`real_correspondence_supported=false`, `real_paired_scan_supported=false`,
`trimmed_reconstruction_supported=false`, and `deployment_supported=false`.

The reproducible artifact is
`benchmark-out/cycle_gated_assignment_phase21.json` (gitignored, 7,665,315
bytes, SHA-256
`ce869cc421140caca289d14d546f78a8ae97a92b90cf066b4a428735da5a1eb6`).
