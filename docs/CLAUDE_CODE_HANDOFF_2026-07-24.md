# Claude Code Handoff — PFTF_alpha_dev

Snapshot date: 2026-07-24 (Asia/Seoul)

Repository: `D:\__PFTF_Projects(2026)\PFTF_alpha_dev`

## Read this first

The worktree is intentionally dirty. Do not reset, clean, checkout, or otherwise
discard the local G5 work listed below.

The last published commit is:

- branch: `main`
- local `HEAD`: `b15d25438134014a126b89d5d73d790856cca058`
- `origin/main`: `b15d25438134014a126b89d5d73d790856cca058`
- commit title: `Complete exact alpha audit chain through schema 25`
- ahead/behind at the start of the G5 work: `0/0`

That commit was tested and pushed successfully. Everything in the next section
was created after that push and has not been committed or pushed yet.

## Current uncommitted work

Modified tracked files:

- `README.md`
- `docs/EXPERIMENT_PLAN.md`
- `docs/PFTF_ALPHA_RESEARCH_PLAN.md`
- `src/pftf_alpha/synthetic.py`
- `tests/test_synthetic.py`

New untracked files:

- `src/pftf_alpha/g5_validation.py`
- `tests/test_g5_validation.py`
- `CLAUDE_CODE_HANDOFF_2026-07-24.md` (this file)

Run `git status --short --branch` before doing anything. The G5 source and test
files are untracked, so ordinary `git diff` does not show their contents.

## What the completed G5 work does

`src/pftf_alpha/g5_validation.py` implements a separate deterministic artifact
schema, `pftf_alpha_g5_preflight/v1`. It deliberately does not bump the main
benchmark schema beyond 25.

The flow is:

1. Build the six-family `calibration` panel.
2. Freeze a reference-free P2 confidence threshold.
3. Freeze one multiplier for each of B4, B5, P1, and P2 on calibration only.
4. Evaluate those values unchanged on four `held_out` profiles:
   - `base`: ordinary held-out geometry, density, and noise;
   - `sparse`: two-thirds observed point count, rounded with a floor of 16;
   - `noisy`: held-out observation noise multiplied by 2;
   - `hard_geometry`: the family-specific geometry parameter multiplied by 0.75.
5. Use the same seed for each profile's corresponding repeat/family pair.
6. Aggregate geometry, topology, labeled false bridges, runtime, P2 fallback,
   and matched profile degradation.
7. Compare P1 and P2 against a strict casewise, per-endpoint B4/B5 envelope.

Dense references and synthetic component labels are evaluation-only. Held-out
tuning is explicitly prohibited and checked by the result contract.

`src/pftf_alpha/synthetic.py` now accepts a scoped
`variation_overrides` mapping. Only the current family's geometry variable and
`noise` are legal. The old behavior is unchanged when no override is provided.

## Verification already completed

The last full verification passed:

```powershell
uv run ruff check .
# All checks passed!

uv run pytest -q
# 141 passed in 42.54s

git diff --check
# passed; only the repository's usual LF -> CRLF warnings were printed
```

Focused G5/synthetic tests also passed: 7 tests in 2.83 seconds before the full
suite was run.

The new tests verify:

- variation overrides are deterministic and family-scoped;
- all four profiles are frozen as declared;
- calibration happens before held-out evaluation;
- all adaptive multipliers are frozen;
- held-out methods never use the reference for selection;
- the result contains 24 cases for the one-repeat test panel;
- P2 guard violations remain zero;
- smoke/preflight results can never claim paper promotion;
- the CLI writes the declared artifact schema.

## Default G5 artifact

Reproduction command:

```powershell
uv run python -m pftf_alpha.g5_validation
```

Default configuration:

- observed points: 96 (`sparse` uses 64)
- dense reference points: 4096
- endpoint surface samples: 512
- calibration candidate budget: 24 (25 candidates are evaluated when 1.0 is
  inserted into the declared range)
- adaptive kNN: 12
- repeats: 3
- families: 6
- held-out cases: 72
- methods per case: B4, B5, P1, P2
- seed: 20260724

Local ignored artifact:

- path: `benchmark-out/g5_frozen_held_out_preflight.json`
- size: 1,078,057 bytes
- SHA-256:
  `ca1fee74276635b4848ead5a09710d4857c4d232a42aa077b50f6251c62dde69`
- completed around 2026-07-24 22:10 KST

`benchmark-out/` is ignored. A rerun overwrites the artifact, and runtime fields
can change the file hash even when substantive results are reproduced. Preserve
the current file if its exact hash is needed.

Frozen calibration result:

- P2 confidence threshold: `0.2632006823297974`
- achieved calibration fallback fraction: `0.25144175317185696`
- B4 multiplier: `72.1450536742484`
- B5 multiplier: `1.91928528060946`
- P1 multiplier: `71.93897148061905`
- P2 multiplier: `71.93897148061905`

## G5 result and interpretation

Top-level result:

- `endpoint_preflight_supported = false`
- `promotion_supported = false`
- P2 fallback/B4 guard violations: 0 in every profile

P2 aggregate endpoints (each row covers 18 cases):

| profile | mean F-score | mean normalized Chamfer^2 | mean normalized Hausdorff | component error sum | Betti error sum | false-bridge edges/faces | mean fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.529804 | 0.005685 | 0.170668 | 6 | 28 | 146 / 146 | 0.278146 |
| sparse | 0.502268 | 0.006005 | 0.174927 | 6 | 27 | 123 / 123 | 0.289562 |
| noisy | 0.513194 | 0.005788 | 0.170719 | 6 | 27 | 140 / 140 | 0.227166 |
| hard_geometry | 0.538345 | 0.005833 | 0.175690 | 6 | 28 | 149 / 149 | 0.256515 |

P2 margins against the strict casewise B4/B5 endpoint envelope:

| profile | mean F-score margin | mean geometry-loss margin | topology burden excess | false-bridge edge excess | result |
|---|---:|---:|---:|---:|---|
| base | -0.047907 | -0.039344 | +1 | +86 | fail |
| sparse | +0.000172 | -0.010943 | +1 | +55 | fail |
| noisy | -0.048930 | -0.039375 | +1 | +80 | fail |
| hard_geometry | -0.049247 | -0.042294 | 0 | +4 | fail |

Margin definitions matter:

- F-score margin = candidate minus the casewise maximum of B4/B5; positive is
  better.
- geometry-loss margin = the casewise minimum B4/B5 geometry loss minus the
  candidate geometry loss; positive is better.
- The envelope may use a different baseline for different endpoints. It is a
  deliberately strict test of being beyond both mandatory adaptive baselines,
  not a comparison of aggregate means alone.

The guard invariant is therefore intact, but it is only a score/selected-cell
invariant. It does not certify the topology or false-bridge behavior of the
output surface. Increasing fallback frequency alone is not the next research
task.

## Claim boundary that must remain intact

Do not present this result as G5 completion or paper evidence.

The current artifact is only a deterministic synthetic robustness preflight.
It does not provide:

- a real higher-fidelity held-out CAD/scan evaluation;
- confirmatory confidence intervals or uncertainty estimates;
- a deployed exact or validated fail-closed G4 fallback;
- an exact spatially varying anisotropic alpha complex;
- a general zero-false-safe certificate;
- CGAL/reference-stack parity.

`promotion_supported` must remain false until both of these independent gates
are satisfied:

1. a frozen higher-fidelity held-out evaluation shows value beyond B4 and B5 on
   declared geometry and topology endpoints; and
2. an exact or validated fail-closed fallback has no unreported false-safe
   cases.

## Previously tested and rejected repair directions

Do not repeat these as if they were new proposals:

1. Schema 12, independent per-cell bridge soft penalty:
   - a weak penalty improved geometry slightly but increased false bridges;
   - a strong penalty reduced bridges but regressed geometry/Betti error;
   - risk-versus-output-bridge Spearman correlation was about -0.27.
2. Schema 14, layered boundary-owner peeling:
   - shallow peeling exposed more bridge edges/faces;
   - deep peeling reduced bridges but badly regressed geometry/objective;
   - no depth passed every frozen gate.
3. Schema 15, connected boundary-risk region removal and simple
   safe-backbone cuts:
   - neither balanced geometry, topology, and strict bridge improvement;
   - no intervention was frozen or deployed.
4. Schemas 16-25, exact numerical audit chain:
   - useful for predicate/connectivity/value/index/resampling evidence;
   - still evaluation-only and mostly binary64 after exact rounding;
   - schema 25 did not deploy exact B3 selection or close G4.

The next method should not be “schema 26” merely to add another narrow exact
shadow. It should directly address the promotion bottleneck below.

## Remaining work, in priority order

### P0 — Preserve and review the current G5 milestone

1. Inspect `git status`, the five modified files, and both new G5 files.
2. Do not discard the untracked source/test files.
3. Re-run Ruff, the full test suite, and `git diff --check` after any edit.
4. Commit/push directly to `main` only if the user's next instruction explicitly
   authorizes publishing the new G5 milestone. The earlier push authorization
   was already consumed by commit `b15d254`.

### P1 — Develop an output-level topology/false-bridge candidate

The present P2 often inherits B4-like false-bridge behavior even though its
score guard is valid. The next candidate must act on globally consistent output
structure, not just independent cell scores or a fixed amount of owner peeling.

Required experimental contract:

1. Define the new intervention/selection rule using observed geometry,
   filtration structure, resampling stability, and frozen calibration data
   only. Synthetic labels, expected Betti values, and dense references may not
   affect selection or intervention order.
2. Start as a calibration-only ablation. Predeclare all candidate strengths or
   discrete strategies before reading held-out outcomes.
3. Preserve downward closure and recompute the actual regularized boundary after
   every structural change.
4. Measure geometry, component error, full surface Betti error, labeled
   false-bridge edges/faces, nonmanifold edges, runtime, and fallback rate.
5. Freeze the selected candidate on calibration, then evaluate it unchanged on
   the existing four-profile, three-seed G5 panel.
6. Require non-regression against both B4 and B5. Do not claim success from one
   mean F-score or one family.

Candidate design work should first explain why it is structurally different
from schemas 12, 14, and 15. A promising direction must couple boundary risk to
global complex/resampling structure or candidate rejection/fail-closed routing;
plain per-cell penalties and fixed risk-region deletion are already negative
evidence.

### P2 — Convert G4 from shadow evidence into deployed fail-closed behavior

Existing reusable modules include:

- `src/pftf_alpha/exact_backend.py`
- `src/pftf_alpha/exact_python_backend.py`
- `src/pftf_alpha/exact_filtration.py`
- `src/pftf_alpha/exact_shadow.py`
- `src/pftf_alpha/exact_value_shadow.py`
- `src/pftf_alpha/exact_index_audit.py`
- `src/pftf_alpha/exact_resampling_audit.py`
- `src/pftf_alpha/exact_resampling_filtration.py`
- `src/pftf_alpha/exact_b3_shadow.py`

The missing work is not another comparison report. It is a runtime policy and
selection path:

1. Define an observed-data-only trigger and a frozen fallback target.
2. Integrate the host-validated exact backend into an actual selection/fallback
   path.
3. Fail closed on backend timeout, malformed responses, nonidentical inputs,
   failed boundary validation, or unsupported anisotropic construction.
4. Add tests for every failure mode and prove that none silently proceeds as a
   safe result.
5. Keep the claim conservative: if exact anisotropic construction is not
   available, use an explicitly declared conservative global fallback rather
   than describing the PFTF complex as exact.

This work should be attempted after a method candidate shows synthetic endpoint
value, unless the user explicitly prioritizes G4 integration first.

### P3 — Higher-fidelity frozen held-out confirmation

Only after P1 produces a credible frozen candidate:

1. Select a public CAD/scan dataset with a usable reference mesh.
2. Predeclare shape, density, noise/missing-data splits and all exclusions.
3. Freeze weights, thresholds, fallback triggers, and calibration seeds before
   the held-out run.
4. Compare B4, B5, P1, P2, and the new candidate with paired statistics and
   uncertainty intervals.
5. Report failures by shape/regime, not only pooled means.
6. Keep the real-data artifact separate from synthetic preflight evidence.

Dataset selection/download is not yet present in this repository and may need
the user's direction or approval.

## Useful commands

```powershell
# Inspect without changing anything
git status --short --branch
git diff -- README.md docs src tests

# Verify current work
uv run ruff check .
uv run pytest -q
git diff --check

# Small focused tests
uv run pytest -q tests/test_synthetic.py tests/test_g5_validation.py

# Reproduce the default ignored G5 artifact
uv run python -m pftf_alpha.g5_validation

# Inspect CLI options for a cheaper smoke run
uv run python -m pftf_alpha.g5_validation --help
```

## Files to read first

1. `CLAUDE_CODE_HANDOFF_2026-07-24.md`
2. `docs/EXPERIMENT_PLAN.md`, especially the promotion rule and schemas 12-15
3. `docs/PFTF_ALPHA_RESEARCH_PLAN.md`, especially G4/G5 and the latest checkpoint
4. `src/pftf_alpha/g5_validation.py`
5. `src/pftf_alpha/calibration.py`
6. `src/pftf_alpha/adaptive.py`
7. `src/pftf_alpha/exact_backend.py`
8. `tests/test_g5_validation.py`

There is an existing `graphify-out/graph.json`, but it primarily reflects the
schema-25 state and should not be assumed to include the uncommitted G5 module.
Use the live source and this handoff as the authority for the current snapshot.
