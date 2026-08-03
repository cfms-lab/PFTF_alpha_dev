# Global tangential assignment: Phase 20

## Frozen question

Phase 19 showed that independently thresholding a scalar tangential pair score
cannot remove every incorrect pair while retaining 99% of correct pairs and the
normal-direction transient evidence needed by the safety guard. Can a global
one-to-one assignment use competition across the whole presented point set to
recover correspondence without a pair-rejection cutoff?

This phase tests a synthetic set-level correspondence representation. It keeps
the downstream Phase-18 matched-displacement rectangle fixed and does not
retune any Phase-19 scalar cutoff.

## Frozen information boundary

The route receives only the presented primary/repeat coordinate arrays and the
asserted row pairing used for initial robust alignment. It receives no source
labels, stress identity, clean surface, analytic height, endpoint, missing-ID
truth, or pair-source truth.

The perturbation generator's primary and repeat source IDs are used only to
evaluate assignment accuracy and mismatch repair after routing. Geometry and
topology endpoints remain evaluation-only.

## Frozen robust alignment and local frame

Reuse the Phase-19 observed-coordinate preparation without modification:

1. three Kabsch iterations;
2. retain the 80% smallest current presented-pair residuals for the next fit;
3. estimate each primary normal from its 12 nearest primary neighbours; and
4. use median tangent-projected neighbour distance, floored at
   `0.002 * L_obs`, as local spacing.

The alignment may use the presented row pairs, but the subsequent assignment
does not preserve them.

## Frozen global assignment

For every primary point `i` and aligned repeat point `j`, define

`cost(i,j) = ||tangent_i(primary_i - aligned_repeat_j)|| / spacing_i`.

The normal residual is intentionally excluded so that a transient depth return
can still match its acquisition location. Solve the square minimum-cost
one-to-one assignment with SciPy's `linear_sum_assignment`.

For reproducible exact ties, add less than one floating spacing unit of
deterministic nonlinear row-major perturbation across the complete matrix
before solving. The perturbation is used only to choose among numerically equal
costs; reported costs are the unperturbed values.

No pair cutoff, source/stress truth, candidate pruning, unmatched penalty, or
post-assignment repair is permitted.

## Frozen downstream guard

Reorder the original, unaligned presented repeat coordinates according to the
global assignment. Compute the unchanged Phase-17 robust matched-displacement
peak/support evidence and reuse the Phase-18 rectangle exactly:

`peak <= 10.922625244331805 AND support <= infinity`.

The primary reconstruction, unguarded decision, geometry/topology endpoint,
matched-repeat simulator, five perturbation profiles, severity/order, and seed
offsets are unchanged from Phases 18 and 19.

## Frozen three-panel protocol

Each panel uses the unchanged nine Phase-8 stresses, `N in {96,160,256}`, eight
repeats, 2048 clean-reference points, and 256 surface endpoint samples. Five
profiles produce 1080 audited rows per panel from 216 primary reconstructions.

- Calibration cohort A seed: `23300804`.
- Calibration cohort B seed: `23400804`.
- Conditional final held-out seed: `23500804`.
- Phase-19's unopened seed `23200804` is not reused.
- Final held-out is not executed unless both calibration panels pass without
  changing the assignment or downstream guard.
- No alignment, cost, assignment, tie-break, or gate may change after either
  fresh calibration cohort is observed.

## Predeclared success gate

Every profile in every opened panel must satisfy:

1. at least 99% aggregate truth assignment accuracy;
2. for `mismatch_02` and `combined`, at least 90% recovery of the originally
   mismatched presented primary rows;
3. at least one unguarded harmful-outlier false-safe;
4. zero guarded harmful-outlier false-safes; and
5. at least 90% retention of safe control/local-bump accepts.

Both mismatch-bearing profiles must contain presented mismatches. Both
calibration panels must pass before final held-out is opened. Final retuning is
forbidden.

Even a pass supports only this declared synthetic set assignment and
perturbation family. `real_correspondence_supported`,
`real_paired_scan_supported`, `trimmed_reconstruction_supported`, and
`deployment_supported` remain false.

## Planned run

```powershell
python -m pftf_alpha.global_tangential_assignment `
  --output benchmark-out/global_tangential_assignment_phase20.json
```

## Result

The implementation tests and full regression suite passed before the fresh
calibration cohorts were executed. Global assignment repaired most injected
mismatches and restored the zero-harm safety result, but it also reassigned
already-correct rows and failed the predeclared assignment and retention gates.

- Calibration A assigned 173,097/176,832 pairs correctly (`97.89%`), below
  99%. Harmful false-safes fell from 260 to zero, while safe-focus retention was
  183/220 (`83.18%`).
- Calibration B assigned 172,968/176,832 pairs correctly (`97.81%`). Harmful
  false-safes fell from 270 to zero, while focus retention was 161/210
  (`76.67%`).
- `mismatch_02` repair was `97.78%` and `98.33%`, above its 90% gate.
- `combined` repair was only `85.24%` and `86.85%`, below 90%.
- Every profile had zero guarded harmful false-safes, but no profile reached
  the required 99% assignment accuracy in either calibration cohort.

The failure is present even when the supplied pairing is already exact.
Global assignment introduced 751 and 780 wrong pairs in the A/B exact profiles,
giving `97.96%` and `97.88%` accuracy. Exact-profile focus retention was
`88.64%` in A and `76.19%` in B. Thus the tangent-only cost cannot know when a
low-cost global permutation should leave a correct presented pairing unchanged.

The most severe safe-focus example was calibration-A `combined`, `local_bump`,
N=96, repeat 5, seed `24150863`. Assignment changed five rows, introduced four
new mismatches, reached only `94.19%` correctness, and inflated the frozen
matched-displacement peak to `183.5362`, causing fail-closed rejection.

Both calibration panels failed, so final held-out seed `23500804` was not
opened and no assignment rule was changed. Therefore
`phase20_supported=false` and
`global_tangential_assignment_synthetic_supported=false`. Pure tangent-only
global reassignment is closed for this protocol. A follow-up would need a new
confidence-qualified preserve-versus-reassign signal or externally measured
acquisition metadata; tuning this cost on the opened cohorts is forbidden.
`real_correspondence_supported=false`, `real_paired_scan_supported=false`,
`trimmed_reconstruction_supported=false`, and `deployment_supported=false`.

The reproducible artifact is
`benchmark-out/global_tangential_assignment_phase20.json` (gitignored,
8,095,046 bytes, SHA-256
`7476a839b076bf16fe8c84093fab1aa23e2d7402352ecbb671f58e8bf802f397`).
