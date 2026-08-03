# Phase 40: ETH Gazebo real alpha-reconstruction shadow result

## Outcome

The preregistered real-data fixed-alpha shadow is complete and passes its
geometry gate. Across 17 untouched validation sources, the frozen Phase-39 p90
decision removes 45 of 336 direct registered scan inputs. All 34 baseline and
guarded alpha meshes materialize.

| Endpoint | Unguarded | p90-routed | Change |
|---|---:|---:|---:|
| Mean geometry loss | 0.245788 | 0.241750 | -0.004039 |
| Mean F-score | 0.044418 | 0.059588 | +0.015170 |
| Mean recall | 0.074709 | 0.103817 | +0.029108 |
| Mean connected components | 7.4118 | 7.1765 | -0.2353 |
| Mean Betti-1 | 60.7647 | 67.1765 | +6.4118 |
| Mean Betti-2 | 46.8235 | 47.2353 | +0.4118 |
| Mean nonmanifold-edge fraction | 0.015981 | 0.016680 | +0.000699 |

The guarded mesh has lower geometry loss in 11/17 cases. F-score and recall
improve in 16/17 cases; source 2 is the only small regression. The aggregate
gate passes because mean geometry loss decreases, mean F-score increases, and
mean recall does not regress.

`geometry_shadow_supported=true`.

## Protocol integrity

Source 0 was opened only as a development/runtime case and excluded before the
validation source set was frozen. Commit `a489402` preregistered the 17-source
panel, heldout row split, ROI, fixed alpha, sampling, endpoints, and gate before
the validation references were evaluated.

- Protocol SHA-256:
  `cd829c3e1c1d9585ccef5c6fa98311e6a62507d9f57fb55e960405dd53ba635b`
- Final result SHA-256:
  `aa6e4cdf6ce30e554bd5945a475ce9f81ba7b8564a5d27021db3be88b00e0f20`
- Deterministic repeat: a second full execution reproduced the same result
  SHA-256 byte for byte.
- Registration-label access by the Phase-40 evaluator: false.
- Opened archive members: exactly the 32 Hokuyo point-cloud members.
- Validation reference access: true, only after protocol commit.

The construction consumes the Phase-39 pre-label prediction and decision
artifacts. It does not read the Phase-39 post-label registration audit or the
Gazebo Leica pose member.

## Interpretation

This is the first real point-cloud alpha reconstruction execution directly
using the validated observation. The positive result supports a narrow claim:
for this fixed 1.00 m alpha and source-view heldout endpoint, scan-level p90
input routing improves aggregate geometric consistency.

The topology evidence is mixed. Mean component count decreases slightly, but
Betti-1, Betti-2, and the nonmanifold-edge fraction increase. Without a
full-scene reference mesh, none of these directions identifies correct
topology. Therefore:

- `real_alpha_reconstruction_shadow_executed=true`;
- `geometry_shadow_supported=true`;
- `topology_endpoint_comparison_executed=true`;
- `topology_correctness_supported=false`;
- `real_trimmed_reconstruction_supported=false`;
- `deployment_supported=false`.

The very low absolute F-scores also matter: the heldout source view samples a
large outdoor scan, while the 0.75 m fused cloud and 1.00 m alpha are coarse.
The result is a paired improvement, not a high-fidelity scene reconstruction.

## Next distinct step

Do not retune alpha or p90 on these validation references. The next useful
phase is a preregistered point-local spatial-support shadow: derive per-voxel
multi-scan support/dispersion from transformed observed points only, use it to
route a local alpha field or fail closed, and keep the current 17-source panel
out of development. A new development/validation partition or a different
real scene is required before any new threshold is evaluated.
