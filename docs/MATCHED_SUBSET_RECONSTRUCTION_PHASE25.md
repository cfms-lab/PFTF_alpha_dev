# Matched-subset reconstruction guard: Phase 25

## Question

Phase 24's limiting `missing_10pct` case deleted the pair for the only harmful
source outlier vertex while the upstream surface was still reconstructed from
the full primary cloud. Can a route that reconstructs missing-pair profiles
from the observed matched primary subset, then applies the frozen observed
displacement score, remove this information mismatch on untouched seeds?

## Design evidence and information boundary

The Phase 24 score-fit seed `24900804` and cutoff-calibration seed `25000804`
are design-only cohorts in this phase. They may define the route but cannot
support Phase 25. The limiting calibration case had source outlier indices 158
and 159. Pair deletion removed index 158, which was the only harmful vertex;
the retained index 159 was below the harmful distance threshold. This source
identity check is diagnosis-only and is never available to the route.

The route may use:

- the full primary acquisition;
- the presented retained-pair map and retained primary IDs;
- the presented repeat coordinates;
- the frozen Phase 24 twelve-coordinate observed displacement signature and
  ridge coefficients; and
- the upstream candidate decision.

For a profile with missing pairs, the route reconstructs the surface using only
the retained primary points. Exact and registration-only profiles use the
unchanged full primary reconstruction. Clean reference points, source labels,
and geometry/topology endpoints are evaluation-only.

## Frozen design rule

1. Reproduce the Phase 24 ridge coefficients by fitting the unchanged twelve
   features with penalty `1.0` on score-fit seed `24900804` only.
2. Apply matched-subset reconstruction to missing-pair cases in design seed
   `25000804` before defining harmful calibration labels.
3. Let `s_focus_max` be the maximum score of original-safe control/local-bump
   accepts and `s_harm_min` the minimum score of routed-endpoint harmful
   accepts.
4. Freeze the same conservative quarter-gap rule:

   `cutoff = s_focus_max + 0.25 * (s_harm_min - s_focus_max)`.

The frozen design cohort gives:

- routed harmful count: 164;
- focus-safe count: 126;
- `s_harm_min = 0.3771858494633348`;
- `s_focus_max = 0.13806207726534975`;
- separation gap `0.23912377219798503`; and
- cutoff `0.19784302031484602`.

The exact, registration, and missing profile gaps are respectively
`0.31726984076584674`, `0.23912377219798503`, and
`0.28099267692719687`. No feature, coefficient, cutoff, endpoint definition,
subset rule, profile, gate, or fresh seed may change after this document.

## Frozen fresh-seed protocol

Every panel uses the unchanged nine Phase-8 stresses, `N in {96,160,256}`,
eight repeats, 2048 clean-reference points, and 256 endpoint surface samples.
Each panel contains 648 audited rows from 216 primary reconstructions.

- Validation A: seed `25400804`.
- Validation B: seed `25500804`, opened only if validation A passes.
- Final held-out: seed `25600804`, opened only if validation B passes.
- Every opened, design, or reserved Phase 20--24 seed from `23300804` through
  `25300804` is forbidden as fresh evidence.

## Frozen gates

Every profile in every opened fresh panel must:

1. reproduce at least one original unguarded harmful false-safe;
2. reduce routed harmful false-safes to zero; and
3. retain at least 90% of original-safe control/local-bump accepts without
   introducing routed endpoint harm.

The design model and calibration must reproduce their recorded valid values
before validation A is opened. Each later seed is opened only after the prior
fresh panel passes. A failed panel closes every later seed and may not be used
for retuning.

Even a complete pass supports only synthetic matched-subset reconstruction
under the presented-pair simulator. `real_correspondence_supported`,
`real_paired_scan_supported`, `real_trimmed_reconstruction_supported`, and
`deployment_supported` remain false.

## Planned run

```powershell
python -m pftf_alpha.matched_subset_reconstruction `
  --output benchmark-out/matched_subset_reconstruction_phase25.json
```

## Result

Phase 25 is **not supported**. The implementation first passed Ruff, the diff
check, and all 256 tests that existed before execution. It reproduced the
design calibration exactly: routed harmful count 164, focus-safe count 126,
and cutoff `0.19784302031484602`.

The first implementation mistakenly enforced zero accepted harmful-outlier
endpoints and 90% focus retention, but omitted the separately preregistered
requirement that subset reconstruction must not introduce endpoint harm into
an originally safe accepted case. Because of that implementation error, it
incorrectly opened validation A (`25400804`), validation B (`25500804`), and
the final seed (`25600804`). The invalid run appeared to pass the incomplete
gate:

- design score fit: harmful 168 -> 0, focus 132/132, all-safe 352/360;
- design cutoff calibration: harmful 165 -> 0, focus 126/126, all-safe
  351/351;
- validation A: harmful 156 -> 0, focus 129/129, all-safe 366/366;
- validation B: harmful 162 -> 0, focus 123/123, all-safe 354/354; and
- final held-out: harmful 165 -> 0, focus 132/132, all-safe 368/372.

Those fresh-panel values are recorded only as an audit trail. They are not
valid support evidence and must not be reused: the sequential opening rule was
violated by the evaluator bug. In particular, the final panel contained an
accepted originally safe `missing_10pct/upper_occlusion`, N=96, repeat 6 case
(case seed `25760849`, replicate seed `525760858`, perturbation seed
`927760862`, score `0.0152761`) for which subset reconstruction introduced four
clean cross-layer faces.

The corrected gate added an explicit
`introduced_routed_endpoint_harm_accept == 0` requirement at both profile and
panel levels, with a regression test. After the correction, Ruff and the diff
check passed and all 257 tests passed. Re-execution then stopped at the design
score-fit panel, as the frozen protocol required. Two originally safe accepted
`missing_10pct/upper_occlusion`, N=96 cases became harmful after subset
reconstruction:

- repeat 0, case seed `25000807`, replicate seed `525000816`, perturbation
  seed `927000820`, score `0.00471408690994185`: 10 clean cross-layer faces;
- repeat 3, case seed `25030828`, replicate seed `525030837`, perturbation
  seed `927030841`, score `-0.014959621371691989`: 5 clean cross-layer faces.

Both scores lie below the frozen cutoff and are therefore accepted. The exact
and registration profiles pass, but the missing profile and design panel fail;
`design_reproduced=true`, `design_gate_passed=false`, and all three fresh-panel
fields in the corrected artifact are null. Consequently
`phase25_supported=false` and
`matched_subset_reconstruction_synthetic_supported=false`. Seeds
`25400804`--`25600804` are opened and contaminated despite being absent from
the corrected fail-closed artifact and may never be treated as fresh evidence.

Matched-subset trimming closes the missing-harmful-pair information gap but can
itself create clean cross-layer surface faces that the frozen displacement
score does not observe. Continuing this route requires a preregistered
observed guard for reconstruction-induced clean cross-layer topology, a safer
reconstruction rule with a proved invariant, or external acquisition evidence;
the present route must not be retuned on any Phase 25 seed. Real
correspondence, paired scans, trimming, and deployment remain unsupported.
