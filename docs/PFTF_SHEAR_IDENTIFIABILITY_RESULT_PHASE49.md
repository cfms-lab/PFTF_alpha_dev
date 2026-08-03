# Phase 49: PFTF shear-signal identifiability result

## Outcome

The TRAIN/CALIBRATION-only audit finds no stable, PFTF-specific shear signal
that justifies another representation or held-out panel. The TRAIN-selected
PFTF feature responds inconsistently on CALIBRATION, and family/seed block
variation overwhelmingly dominates the shear contribution.

Therefore all of the following remain false:

- `stable_within_block_pftf_signal_supported`;
- `pftf_specific_within_block_signal_supported`;
- `standalone_pftf_identifiability_supported`;
- `new_representation_development_justified`;
- `new_held_out_panel_justified`;
- `phase49_identifiability_supported`.

## Frozen protocol and implementation

- Protocol commit: `5b2dddb`.
- Protocol SHA-256:
  `7256ca43884d7e808f8e9e3f186a24f510e59810f06b928f6f84f3d8cc34a1bc`.
- Implementation commit: `62275ec`.
- TRAIN/CALIBRATION cases: `60/30`; held-out consumed: `0`.
- TRAIN/CALIBRATION blocks: `12/6`, each containing five shear strengths for
  one fixed family and seed.

The implementation treats a zero-variance response as `R2=0`, because a
constant feature contains no identification signal. A regression test fixes
this convention; otherwise the shear-invariant `x_skewness` coordinate would
be incorrectly ranked as a perfect response.

## Selected-feature results

TRAIN selected `log_scale_std` from the seven PFTF summaries and
`normalized_covariance_xy` from the four non-PFTF geometry summaries.

| Endpoint | PFTF `log_scale_std` | Geometry covariance |
|---|---:|---:|
| TRAIN median within-block `R2` | `0.924162` | `0.977365` |
| TRAIN slope-sign consistency | `7/12` | `6/12` |
| CALIBRATION median within-block `R2` | `0.491280` | `0.992884` |
| CALIBRATION direction consistency | `3/6` | `5/6` |
| CALIBRATION family direction | `3/3` | `2/3` |
| CALIBRATION standardized span effect | `0.008540` | `0.136756` |
| CALIBRATION pooled `R2` | `0.000234` | `0.002122` |
| CALIBRATION block variance fraction | `0.997299` | `0.991420` |
| CALIBRATION partial strength `R2` | `0.086759` | `0.247314` |
| Block/strength explained-SS ratio | `4255.57` | `467.21` |
| TRAIN-only standalone CALIBRATION MAE | `0.103968` | `0.103960` |

The frozen PFTF thresholds were `R2 >= 0.75`, direction consistency at least
`5/6`, all family directions, and standardized effect at least `0.25`. Only
the family-direction condition passes. The TRAIN-mean standalone MAE is
`0.104000`, so the required `0.75` fraction is `0.078000`; neither learned
feature approaches it.

The most important result is not merely the failed threshold. For the selected
PFTF summary, surface/seed blocks explain `99.73%` of CALIBRATION variance,
while the pooled strength relation is essentially zero. The observed summary
is dominated by what object was sampled, not by how much synthetic shear was
applied.

The result is byte-identical across repeated writes. Result SHA-256:
`f6479406becf127301f290e37f8d0890fb1c7816bfc8e3513f75e6f68bb51975`.

## Interpretation and next boundary

Do not add another PFTF summary, reopen Phase-48 held-out data, or create a new
held-out shear panel. Phase 49 closes the learned PFTF coordinate-map branch:
the prerequisite observed identifiability signal was not found even before a
new performance claim was attempted.

This does not invalidate the Phase-47 analytic spatial-SPD construction. It
shows that the current PFTF relation summaries do not identify which member of
that safe map family to use.

The next project-level decision should be a paper pivot rather than Phase-50
feature tuning: consolidate the honest negative/limits paper around P3, M1,
G4, and the Phase-43--49 boundary evidence, or move the specialized positive
two-layer result into a separate manuscript.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.pftf_shear_identifiability_protocol --output benchmark-out\pftf_shear_identifiability_protocol_phase49.json
.\.venv\Scripts\python.exe -m pftf_alpha.pftf_shear_identifiability --output benchmark-out\pftf_shear_identifiability_phase49.json
```
