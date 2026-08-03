# Open3D real paired-scan evidence intake: Phase 31

## Question and claim boundary

Phase 31 asks whether the frozen Phase-27/28/30 observed guard stack can be
executed on downloadable real scan pairs, and whether its local/tail summaries
remain nontrivial outside the synthetic generator.

This is an **intake and diagnostic phase**, not a safety validation. Open3D's
transform log supplies alignment metadata, but the evaluated pairs are
distance-gated reciprocal nearest-neighbour candidates. They do not establish
that two samples are the same physical point. The benchmark also has no clean
surface or topology endpoint for this route. Consequently this phase may set
only `real_paired_scan_intake_supported=true`; it keeps real correspondence,
real paired-scan guard, reconstruction, and deployment support false.

## Frozen data provenance

The selected intake is Open3D `DemoICPPointClouds`:

- official API: <https://www.open3d.org/docs/latest/python_api/open3d.data.DemoICPPointClouds.html>;
- official dataset declaration and CC BY 3.0 notice:
  <https://www.open3d.org/docs/0.19.0/cpp_api/classopen3d_1_1data_1_1_demo_i_c_p_point_clouds.html>;
- official source declaration:
  <https://github.com/isl-org/Open3D/blob/main/cpp/open3d/data/dataset/DemoICPPointClouds.cpp>;
- frozen archive URL:
  <https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/DemoICPPointClouds.zip>;
- archive size: 10,829,466 bytes;
- official MD5: `596cffe5f9c587045e7397ad70754de9`; and
- locally recorded SHA-256:
  `b94e0146c1d48c5edfc11af71b4af39ffca604485668c55a127c3b43203a6bd5`.

The verified archive contains exactly three binary PCD files and `init.log`.
The point counts are 198,835, 137,833, and 191,397. Dataset files are kept in
the ignored `benchmark-data/open3d_demo_icp/` directory rather than committed.

## Frozen intake protocol

1. Verify both archive hashes, the exact four-member allowlist, member CRCs,
   and safe top-level paths before extraction.
2. Parse only the observed PCD schema: eight little-endian float32 fields
   `x y z rgb normal_x normal_y normal_z curvature`, retaining XYZ only.
3. Parse each `init.log` pair header and finite invertible 4x4 matrix.
4. Check transform direction without changing the log. For an `i j` entry,
   compare direct and inverse matrix application; use the inverse only when it
   has both lower median target-NN distance and higher 2 cm coverage.
5. In the target frame, form one-to-one candidate pairs by reciprocal 1-NN at
   fixed maximum distances 2 cm and 5 cm.
6. At each scan pair and distance, choose eight deterministic spatial anchors
   by farthest-point traversal and take the 96 nearest reciprocal candidates
   around each anchor.
7. Apply the unchanged frozen thresholds to each patch:
   global signature score `< 0.18181536333942858`, local q95 residual
   `< 3.544133065251552`, and isolated-tail ratio
   `< 1.6636368999089541`.

The Phase-23 signature function was factored so the same formula can consume
observed evidence and pair counts without constructing a simulator-only raw
case. No feature, model coefficient, cutoff, neighbour count, or tail rule was
changed.

## Results

The inverse log convention is supported for both supplied pairs:

| Pair | Direct median NN | Inverse median NN | Direct within 2 cm | Inverse within 2 cm |
|---|---:|---:|---:|---:|
| 0 -> 1 | 0.454854 m | 0.027855 m | 3.41% | 42.02% |
| 1 -> 2 | 0.184901 m | 0.014228 m | 7.47% | 61.39% |

Reciprocal candidate intake is substantial but incomplete:

| Pair | Maximum distance | Candidate pairs | Source coverage | Median | p95 |
|---|---:|---:|---:|---:|---:|
| 0 -> 1 | 2 cm | 43,342 | 21.80% | 6.35 mm | 16.57 mm |
| 0 -> 1 | 5 cm | 49,086 | 24.69% | 6.98 mm | 28.99 mm |
| 1 -> 2 | 2 cm | 42,596 | 30.90% | 5.83 mm | 16.89 mm |
| 1 -> 2 | 5 cm | 46,989 | 34.09% | 6.31 mm | 23.21 mm |

The fixed observed guard produces mixed decisions rather than a trivial all-pass
or all-reject outcome:

| Pair | Maximum distance | Global pass | Local-q95 pass | Tail pass | All three |
|---|---:|---:|---:|---:|---:|
| 0 -> 1 | 2 cm | 7/8 | 5/8 | 5/8 | 4/8 |
| 0 -> 1 | 5 cm | 7/8 | 7/8 | 4/8 | 4/8 |
| 1 -> 2 | 2 cm | 8/8 | 6/8 | 6/8 | 5/8 |
| 1 -> 2 | 5 cm | 8/8 | 6/8 | 5/8 | 4/8 |
| **Total** | both | **30/32** | **24/32** | **20/32** | **17/32** |

Across all 32 patches the tail-ratio range is 1.06748--4.34164 and its median
is 1.43236. The real scan therefore exercises the tail guard materially: the
tail rule rejects 12 patches, including seven that pass both the global score
and local-q95 rule. This is evidence that the tail observation is operational
on real scan coordinates, not evidence that its rejections are correct.

## Decision

- `real_paired_scan_intake_supported=true`: archive integrity, PCD/log parsing,
  direction checks, reciprocal matching, and frozen observed-feature execution
  all succeeded on both supplied pairs.
- `real_correspondence_supported=false`: reciprocal NN candidates are not
  physical point-identity ground truth.
- `real_paired_scan_guard_supported=false`: no labeled safe/harmful endpoint is
  available to measure false accepts or safe retention.
- `real_trimmed_reconstruction_supported=false` and
  `deployment_supported=false`: no reconstruction route was run.

The next external-validity step is a real fragment benchmark with independently
defined correspondence positives/negatives or scene-level registration ground
truth, followed by a preregistered label-blind guard evaluation. Thresholds
must remain frozen; these opened DemoICP observations must not be used to tune
them.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.open3d_demo_icp `
  --data-root benchmark-data/open3d_demo_icp --download

.\.venv\Scripts\python.exe -m pftf_alpha.open3d_real_pair_intake `
  --data-root benchmark-data/open3d_demo_icp `
  --phase28-artifact benchmark-out/local_spatial_residual_guard_phase28.json `
  --output benchmark-out/open3d_real_pair_intake_phase31.json
```

Artifact SHA-256:
`d74498640e6032ee5f82726f93f3cf449f06cd8d687d791fda5aa4548510922e`.
