# Sampling-sufficiency fail-closed gate: Phase 1

## Motivation

Phase 0 falsified `risk localization -> local rescan -> unchanged B5`.  At 640
total points, targeted cross-component kNN mixing fell to `0.0189`, but B5 still
merged both sheets.  Reacquisition can therefore remove an information deficit
without removing the reconstruction failure.

Phase 1 asks whether an observed-only gate can distinguish:

1. `rescan_required`: the two layers remain mixed in local neighborhoods;
2. `algorithm_failure_fail_closed`: the layers are sampling-resolved but the
   output retains risky boundary connections; and
3. `accept`: sampling is resolved and the output contains no residual risky
   boundary edge or face.

Unsupported or non-identifiable geometry fails closed.

## Frozen estimator

- Estimate the global sheet normal with the smallest PCA eigenvector.
- Run deterministic two-means on the normal coordinate.
- Require both inferred layers to contain at least 20% of the points.
- Require layer-separation SNR >= 4.
- Compute k=12 cross-cluster kNN mixing using the inferred labels.
- Declare sampling sufficient when inferred mixing <= 0.05.

The estimator receives observed coordinates only.  Ground-truth component
labels are used solely to measure sampling-classification and routing accuracy.

## Frozen evaluation panel

- Held-out opposing sheets with gaps `{0.18, 0.40, 0.60, 0.80, 1.20}`.
- 96 observed points and 2048 evaluation-reference points.
- Eight paired seeds per gap.
- Frozen B5 multiplier `2.80293354289327`.
- Gate success requires at least 95% sampling-regime accuracy, at least 95%
  route accuracy, both rescan and algorithm-failure routes, and zero false-safe
  accepts.
- Deployment additionally requires at least one empirically safe accepted case.

## Run

```powershell
python -m pftf_alpha.sampling_gate `
  --output benchmark-out/sampling_sufficiency_gate_phase1.json
```

## Result

### Phase 1 frozen panel

| Gap | Estimated cross-kNN | True cross-kNN | Sampling accuracy | Routing accuracy | Rescan | Algorithm failure | Unsupported | False safe |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.18 | 0.4581 | 0.4581 | 1.00 | 1.00 | 8 | 0 | 0 | 0 |
| 0.40 | 0.2737 | 0.3065 | 1.00 | 0.50 | 4 | 0 | 4 | 0 |
| 0.60 | 0.1229 | 0.1229 | 1.00 | 1.00 | 8 | 0 | 0 | 0 |
| 0.80 | 0.0420 | 0.0420 | 1.00 | 1.00 | 2 | 6 | 0 | 0 |
| 1.20 | 0.0011 | 0.0011 | 1.00 | 1.00 | 0 | 8 | 0 | 0 |

Overall sampling-regime accuracy is 1.00 and no unsafe output is accepted.
Exact three-way routing accuracy is 0.90, below the frozen 0.95 gate, because
four gap-0.40 cases fail closed as `unsupported` rather than `rescan_required`.
Consequently `phase1_diagnostic_supported=false` and
`deployment_supported=false`.

Artifact: `benchmark-out/sampling_sufficiency_gate_phase1.json`.

### Phase 1b held-out amendment

The Phase 1 panel is now calibration-only.  It identified one over-conservative
parameter: `minimum_separation_snr=4`.  Phase 1b changes only this threshold to
`3`; the cross-kNN threshold (`0.05`), cluster-balance threshold (`0.20`), risk
rule, gaps, point count, eight repeats, and 0.95/zero-false-safe gates remain
unchanged.

Phase 1b uses a new seed (`20270802`) and is not allowed to tune again on its
result.  Diagnostic promotion still does not imply deployment: at least one
safe accepted reconstruction remains necessary for deployment support.

### Phase 1b held-out result

| Gap | Estimated cross-kNN | True cross-kNN | Sampling accuracy | Routing accuracy | Rescan | Algorithm failure | Unsupported | False safe |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.18 | 0.4577 | 0.4743 | 1.00 | 0.75 | 6 | 0 | 2 | 0 |
| 0.40 | 0.2776 | 0.3130 | 1.00 | 0.625 | 5 | 0 | 3 | 0 |
| 0.60 | 0.1343 | 0.1343 | 1.00 | 1.00 | 8 | 0 | 0 | 0 |
| 0.80 | 0.0340 | 0.0340 | 1.00 | 1.00 | 0 | 8 | 0 | 0 |
| 1.20 | 0.0005 | 0.0005 | 1.00 | 1.00 | 0 | 8 | 0 | 0 |

Overall sampling-regime accuracy remains 1.00 and false-safe count remains zero.
Exact route accuracy is 0.875, below the 0.95 gate.  The lower SNR threshold
does not generalize: five under-resolved cases still route to the conservative
`unsupported` state.  There are no accepts because frozen B5 fails every case.

Artifact: `benchmark-out/sampling_sufficiency_gate_phase1b_heldout.json`.

## Decision

- The observed-only two-layer estimate is useful: its sufficient/insufficient
  classification matched the evaluation labels in all 80 Phase 1/1b cases.
- The router produced zero silent false-safe accepts.
- The exact three-way diagnostic gate is not promoted because the held-out
  route accuracy is only 0.875.
- Deployment is impossible to evaluate because B5 supplied zero safe positive
  outputs; `deployment_supported=false` is therefore mandatory.
- No more threshold tuning is allowed on these panels.

The next experiment must first supply a connectivity-changing candidate that
occasionally produces a truly safe two-sheet surface.  Only then can acceptance
coverage and false-safe behavior be tested.  Until that exists, the gate is an
audit tool that separates clear sampling deficits from clear algorithm failures,
not a production certificate.
