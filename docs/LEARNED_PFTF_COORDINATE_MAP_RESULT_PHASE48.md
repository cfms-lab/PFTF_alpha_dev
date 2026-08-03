# Phase 48: learned PFTF-conditioned coordinate-map result

## Outcome

The declared quadratic-shear family remains globally invertible for all 45
held-out predictions, but the PFTF-conditioned learner fails the frozen value
gate. It is worse than both the TRAIN-mean predictor and the non-PFTF global
geometry learner on coefficient recovery, coordinate recovery, and Delaunay
top-cell recovery.

Therefore `pftf_conditioning_value_supported=false` and
`learned_invertible_quadratic_shear_supported=false`. The latter means that the
learned PFTF route is not supported; the analytic map-family construction
check itself passes.

## Frozen protocol and implementation

- Protocol commit: `38312af`.
- Protocol SHA-256:
  `4428aff85db9242bdd25880032cedc50cc7f538316b1eb60279cc3fe8ee90085`.
- Implementation commit: `c4ecc2f`.
- TRAIN/CALIBRATION/HELD_OUT cases: `60/30/45` with disjoint seeds.
- Map family: `Phi_s(x,y,z)=(x,y+s*x^2,z)`, `s` clipped to `[0,0.40]`.
- Candidate features: seven PFTF relation/confidence/scale summaries.
- Comparator features: four non-PFTF covariance/moment summaries.
- Learner: training-standardized ridge regression; calibration chooses the
  penalty before the single held-out evaluation.

Calibration selected PFTF penalty `0.01` and geometry penalty `0`.

## Held-out results

| Method | Coefficient MAE | Coordinate RMS | Top-cell Jaccard |
|---|---:|---:|---:|
| Identity/no correction | `0.180000` | `0.153777` | `0.493423` |
| TRAIN-mean coefficient | **`0.100000`** | **`0.085432`** | **`0.601368`** |
| Non-PFTF geometry ridge | `0.100103` | `0.085439` | `0.598767` |
| PFTF ridge | `0.115118` | `0.098199` | `0.576518` |
| Oracle coefficient | `0` | `0` | `1` |

All three PFTF value sub-gates fail. The simple TRAIN mean is also marginally
better than the geometry ridge, showing that the current observed summaries do
not identify the hidden shear reliably across surface families and new seeds.
The benchmark does show that both learned routes improve on identity, but that
is insufficient under the preregistered PFTF-specific gate.

## Construction audit

- bounded predictions: `45/45`;
- maximum inverse roundtrip error: `2.220446e-16`;
- minimum/maximum Jacobian determinant: `1/1`;
- construction gate: pass.

Thus the failure is not loss of invertibility. It is lack of predictive value
in the frozen PFTF summaries for this hidden coordinate deformation.

The result is byte-identical across repeated writes. Result SHA-256:
`f024f0609da9f4ce09f71f4b97a085a66bb3a9ab0166409d45032ab15e0ad3eb`.

## Claim boundary and next step

Keep the following false:

- `pftf_conditioning_value_supported`;
- `learned_invertible_quadratic_shear_supported`;
- `arbitrary_point_local_spd_complex_supported`;
- `general_nonlinear_map_learner_supported`;
- `point_local_alpha_field_supported`;
- `global_alpha_selection_supported`;
- reconstruction/topology/real-scan/exact/deployment support.

Do not tune the opened Phase-48 held-out panel or add features and rerun it as
though it were new evidence. A distinct next phase should first audit
identifiability on TRAIN/CALIBRATION only: separate within-family shear signal
from between-family variation, then preregister a new feature representation
and new held-out seeds if a stable observed signal exists.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.learned_pftf_coordinate_map_protocol --output benchmark-out\learned_pftf_coordinate_map_protocol_phase48.json
.\.venv\Scripts\python.exe -m pftf_alpha.learned_pftf_coordinate_map --output benchmark-out\learned_pftf_coordinate_map_phase48.json
```
