# bench — experiment harness for the z-shift results section

Every table and figure in the paper is generated from a CSV in `results/`.
No number is typed by hand.

## Layout

```
bench/
  instrument.py    P1  stage timing + peak RSS, plus the environment capture
  metrics.py       P2  chamfer / hausdorff-95 / precision-recall-F / normal consistency
  fixtures.py      P3  controlled mesh damage, labelled motion video, clock-offset streams
  meshes.py            mesh-size and component-count ladders shared by A1–A4
  csvio.py             result rows -> results/<exp_id>.csv, and reading them back
  gt_align.py          Umeyama / ICP alignment to ground truth (Tier B)
  tier_b_common.py     scene manifests, reconstruct-and-score loop (Tier B)
  harness.py           shared CLI scaffolding
  exp_a*.py            CPU-only experiments — no MASt3R, no GPU
  exp_b*.py            experiments requiring MASt3R weights and a GPU
  run_tier_a.py        runs all of Tier A in the protocol's order
  fetch_dtu.py         pulls DTU scans + reference clouds by HTTP byte range
  run_tier_b.py        runs Tier B locally, one subprocess per (experiment, scene)
  plots.py             regenerates every figure from the CSVs
  generate_case_study_plots.py
                       optional visual cross-check of Table II's case-study
                       numbers (hardcoded, not read from a CSV); not used by
                       the paper -- Table II already prints the same numbers
  results/*.csv        committed results
  results/fig7-8-*.png outputs of generate_case_study_plots.py
  scenes/example.json  Tier B scene-manifest template
```

`StageLog` itself lives in `src/spatial_ingestion/instrumentation.py`, not here,
so the pipelines can emit timings into their manifests without depending on
this development-only tree. `bench.instrument` re-exports it.

## Running

```bash
# everything CPU-only (about 45 min on a 12-core laptop)
uv run python -m bench.run_tier_a

# one experiment
uv run python -m bench.exp_a5_frame_budget -v

# reduced grids, for checking the harness works
uv run python -m bench.run_tier_a --quick

# figures from whatever CSVs exist
uv run python -m bench.plots
```

Tier B needs a scene manifest (see `scenes/example.json`) and a GPU:

```bash
uv run python -m bench.exp_b4_frame_budget_ablation --manifest bench/scenes/dtu.json
```

On a local CUDA GPU (no notebook), one command sets up the environment,
fetches the DTU scans by byte range, runs a smoke lane and then B2, B7, B4 and
B6 once per scan, and merges the per-scan CSVs into `results/`. It is
resumable: finished scans are skipped on a re-run. Environment, data and logs
live under `~/zshift` on the Linux side. `--tier-a` also re-runs Tier A and
regenerates the figures afterwards.

```bash
bash scripts/run-review-experiments.sh            # Linux or inside WSL2
bash scripts/run-review-experiments.sh --tier-a   # and every CPU result too
```

What each Tier B experiment runs on DTU:

- B2: 10 views and all 49, spread evenly over the rig (`view_selection`). The
  first ten views by number are one short arc of the robot arm's path, and
  score as a narrow-baseline capture. Pairing is complete while its pairs fit
  `--max-pairs` (`tier_b_common.bounded_pairing`), so 10 views are complete
  and 49 use the widest non-cyclic sliding window that fits; the 49-view
  complete graph (2352 pairs) needs ~18 GB and ran out of memory on an 8 GB
  card. Phase 2 also moves MASt3R's weights off the GPU once the pairs have
  been through the network (`reconstruction.alignment`), which is ~2.7 GB more
  for the scene optimizer.
- B4: every scan, three seeds a cell. MASt3R's TSDF depth refinement samples
  with `torch.randn`, so the seed is a real source of variation. V3 takes
  exactly `budget` frames, evenly spaced; the budget-60 point takes all 49.
- B6: the same 10 spread views as B2.
- B7: the first three scans, as in the paper.
- B9 (`--case-study`): the CO3D teddybear trace behind Table II, rebuilt from
  CO3D's single-sequence chunk (sequence `187_20215_38541`, the one holding
  `paper/outputs/teddy.jpeg`; 8 evenly spaced frames, complete pairing, seed
  42, object-mode refinement, biped rig). Refinement runs with `split_bodies`
  on the CPU, as submitted, and then with `labels` on the same raw mesh.
  `--n-images 24 --isolate-object --filters labels --output-root <dir>` runs the same capture
  at 24 views (complete pairing, 552 pairs) and adds its rows next to the
  8-view ones; `split_bodies` is left out there because it scales with the
  component count of a mesh three times the size.

A1 and A2 time both component filters (`filter_method`): `split_bodies` on the
CPU, which is what the submitted paper's Fig. 9 and stage table measured, and
the batched `labels` path that replaced it. The figure scripts plot the
`split_bodies` rows.

```powershell
# Windows: runs the same script inside WSL2 and keeps the machine awake.
# Native Windows Python is not used: Smart App Control blocks uv's venv
# launcher and the unsigned DLLs torch needs.
powershell -ExecutionPolicy Bypass -File scripts\run-review-experiments.ps1
```

The two Python steps it wraps are `bench.fetch_dtu` and `bench.run_tier_b`.
Setup also installs nvcc at torch's CUDA version into `~/zshift/cuda-<ver>`
(`scripts/install-cuda-toolkit.sh`, no sudo) and compiles MASt3R's RoPE CUDA
kernel; preflight refuses to start if MASt3R would fall back to the PyTorch
RoPE.

Refinement's component filter and watertight check run on CUDA through torch
(`src/spatial_ingestion/refinement/gpu.py`); B6 records the device in
`refine_device`. The output matches the old `split_bodies` path (tests in
`tests/test_refinement.py`), which stays available as
`MeshCleaningConfig(component_filter="split_bodies")`. Nearest-neighbour
queries (ICP, Chamfer, F-score, colour transfer) stay on a CPU KD-tree using
every core: on 500k x 500k points it takes 0.08-0.28 s, where an exact GPU
brute force needs about 2.5e11 distance evaluations per query.

### DTU scoring protocol

DTU scenes are scored the way DTU's own evaluation does it, adapted to a
method that estimates its own cameras (`dtu_eval.py`):

1. Phase 2 writes `cameras.json`, each image's estimated pose.
2. Those camera centres are matched to DTU's calibration
   (`rect_NNN_*` to `pos_NNN.txt`), and a similarity transform fitted to the
   pairs places the reconstruction in DTU's frame. Its scale, over the 1000x
   metre-to-mm factor, is `mast3r_scale_ratio`: 1.0 means MASt3R's metric
   output was exactly right.
3. Accuracy is scored on points inside the scan's observation mask, and
   completeness on ground-truth points above the table plane. The
   `dtu_accuracy` / `dtu_completeness` / `dtu_overall` means ignore distances
   over 20 mm, as DTU does; `chamfer_l1` is `dtu_overall` here. The
   `*_outlier_frac` columns give the share beyond 20 mm. `f_score` uses
   tau = 2 mm.
4. A rigid ICP refinement is reported in `*_icp` columns, but the headline
   scores use the camera fit alone. ICP minimises the distances being scored,
   so on a sparse reconstruction it can find a pose that scores better and is
   wrong. On the first 4-view smoke run it rotated 14 degrees away from a
   camera fit with a 2 mm residual. Large `icp_rotation_deg` / `icp_shift_mm`
   values flag this.
5. The camera fit is trimmed: a camera whose residual is over both 10 mm and
   3x the median is left out and the fit redone (`camera_outlier_views`).
   Untrimmed, one lost view (B4 had single views 100-280 mm off) drags the
   whole placement, which both fails the run and skews its scores.
   `camera_align_rmse_all_mm` keeps the untrimmed residual.
6. `pose_failure` is true when the trimmed fit's residual is over 10 mm, or
   more than 20% of the cameras were left out. Working runs on DTU sit at
   3-6 mm; above 10 mm MASt3R's poses have broken, and the row's scores
   measure that failure rather than the surface. Report those rows as
   failures, not as low accuracy.

`bench.rescore_dtu` re-scores finished B2, B4 and B6 runs from their saved
outputs (no MASt3R, no GPU beyond B6's refinement), keeping each row's timing
and env columns and stamping `rescored_git_commit`. Use it after a scoring
change instead of reconstructing again.

ICP from scratch (the fallback when a manifest has no `dtu_eval` entry) does
not work on DTU. The reconstruction includes table and backdrop that the
ground truth lacks, so every seed settles hundreds of millimetres off.
`bench.fetch_dtu` downloads the calibration and masks (a few MB from
SampleSet.zip) and adds them to the manifest.

## Ground rules, and how they are enforced

| Rule | Where |
|---|---|
| n ≥ 3 repeats for anything timed, mean ± std | `REPEATS` in A1/A2; `mean_std` uses ddof=1 |
| Machine, OS, Python, library versions recorded | `env_metadata()` stamps `env_*` columns on **every row** |
| Seeds fixed and logged | `seed` column on every row |
| One CSV per experiment | `ResultWriter(EXP_ID)` |
| Figures regenerate from CSVs | `bench/plots.py` reads only `results/` |
| Discard the first MASt3R run | `discard_first` in B2 and B7 |
| Clear `cache/` between independent runs | `tier_b_common.clear_alignment_cache`, called in B7 and every Tier B run; `cache_cleared` recorded |
| τ stated for every F-score, units for every distance | `tau` and `distance_unit` columns |

## Deviations from the protocol, and why

- **Peak RSS is not `resource.getrusage`.** That module is Unix-only and the
  authoring machine is Windows. `spatial_ingestion.instrumentation` reads
  `GetProcessMemoryInfo` there and `getrusage` elsewhere (handling the
  KiB/bytes difference between Linux and macOS). A silent 0 would have made
  every memory column meaningless, so `tests/test_bench.py` asserts a non-zero
  reading on whatever platform it runs on.

- **`corrupt_mesh` removes connected face patches, not scattered faces.** The
  protocol's `rng.choice(len(faces), n_holes * 20)` produces `n_holes * 20`
  single-triangle punctures, not `n_holes` holes — a much easier repair
  problem than the one hole filling claims to solve.

- **`normal_consistency` defaults to a KD-tree estimator.**
  `trimesh.proximity.closest_point` needs the optional `rtree` package and runs
  a Python-level query per point, which is untenable across A3's 648-cell grid.
  The default samples the GT surface 4× more densely and takes the nearest
  sample's normal; it is pinned against an analytic sphere reference in the
  tests. `exact=True` still selects the textbook path and raises a clear error
  if `rtree` is missing.

- **A1 sweeps component count and the two axes together, not just triangle
  count.** A pilot run showed triangle count alone cannot reproduce Table I's
  reported cost: a connected 2.5M-triangle mesh refines in ~11 s, not ~7 min.
  Component count is a second, independent axis, and the two are strongly
  super-additive, so A1 has three ladders (`triangles`, `components`,
  `interaction`). The `interaction` ladder runs only in Table I's own
  configuration (smoothing off, watertightness skipped) so its numbers are
  directly comparable to the figure the paper already prints.

- **A6 adds a motion-noise axis.** `derive_offset_stream` as specified clones
  the anchor exactly, so every camera's motion signature is byte-identical and
  `MOTION_MATCH_TOLERANCE` — one of the constants A6 exists to justify — is
  never exercised. `motion_noise` perturbs per-camera scores.

- **The motion-video fixture pans the whole frame.** A moving subject's motion
  score is bounded by its own area and never reaches the sampler's 0.18
  high-motion threshold at any speed, so the high-motion branch would have gone
  untested. `camera_pan=True` (default) scrolls a textured backdrop, which is
  also what a handheld capture does.

- **`fragmented_mesh` builds pieces from UV spheres, not decimated icospheres.**
  `decimate_pro` tears a closed icosphere into several shells at some
  reductions even with `preserve_topology=True`, which silently multiplied the
  component count the experiment exists to control.

## Findings that pin current behaviour

`tests/test_bench.py` contains three tests that assert what the code does
today, not what the paper says. They exist so that changing the behaviour
fails a test and forces the paper text to change with it:

- `test_object_mode_keeps_every_fragment_it_is_given`
- `test_hole_filling_is_skipped_on_sheet_like_input`
- `test_cap_frames_does_not_restore_capture_order`

If one of these starts failing, the code was changed deliberately — update the
corresponding experiment's commentary and the Section IV/V text together.

## Experiment index

| id | claim it supports | needs GPU |
|---|---|---|
| A1 | §V-C refinement scalability (replaces Table I) | no |
| A2 | per-stage attribution of refinement cost | no |
| A3 | §V-B component filtering / hole filling / smoothing / normals | no |
| A4 | §V-B vertex-colour transfer | no |
| A5 | §IV-C frame ordering and pair adjacency | no |
| A6 | §IV-A multi-camera sync | no |
| A7 | §V-B motion-adaptive frame selection | no |
| A8 | §IV-F auto-rigging weight quality | no |
| A9 | §IV-E routing completeness | no |
| B1 | §V-A end-to-end functional validation | yes |
| B2 | §V-E reconstruction accuracy baseline | yes |
| B3 | pairing strategy cost/quality slope | yes |
| B4 | frame-budget ablation (closes A5) | yes |
| B5 | §V-C/D TSDF fallback characterisation | yes |
| B6 | refinement effect on reconstruction quality | yes |
| B7 | §V-D determinism bound | yes |
| B8 | §IV-C EXIF intrinsics initialisation | yes |
| B9 | §V-D CO3D teddybear end-to-end trace (Table II, fig13) | yes |

## Reviewer-response experiments (B10-B13)

One command runs the experiments added for the second review round, from the
repository root inside WSL2 (it needs the environment and DTU scans the Tier B
runner sets up, and B6's outputs):

```bash
bash scripts/run-reviewer-response.sh                 # tests, B10-B13, then a summary
bash scripts/run-reviewer-response.sh --only "b12 b13"
```

```powershell
wsl.exe -d Ubuntu --cd /mnt/c/Users/theof/projects/personal/z-shift --exec bash scripts/run-reviewer-response.sh
```

- B10 (`exp_b10_real_mesh_refinement`): refinement on the raw DTU meshes B6
  saved. `part=accuracy` scores raw, object, room and object-with-isolation
  (`isolate_object`) against DTU's ground truth, with the share of faces kept and
  `normal_consistency` (`dtu_eval.mesh_normal_consistency`: mean |cos| against
  normals estimated from the structured-light points; not a standard DTU metric).
  `part=timing` times `labels` and `split_bodies` with repeats; `split_bodies` runs
  in a child process under `--split-budget-s`, so a hung cell is a `timeout` row.
- B11 (`exp_b11_co3d_accuracy`): the CO3D teddybear runs scored against CO3D's own
  `pointcloud.ply`. That cloud is a COLMAP reconstruction, so it is a
  pseudo-reference, and the single-sequence release has no cameras, so the
  reconstruction is registered by a fixed RMS-radius scale and then ICP
  (`co3d_eval.register`); the registration residual is on every row. Distances are
  fractions of the reference's bounding-box diagonal.
- B12 (`exp_b12_dtu_sync`): the synchroniser on the real DTU rig images. Motion
  scores come from real consecutive frames and a second camera's from perturbed
  pixels; only the clocks are injected, since DTU has none.
- B13 (`exp_b13_backend_swap`): DUSt3R behind `reconstruction/backends.py` on the
  same five scans and views as B2's 10-view rows, scored identically, with
  `contract_ok` (the stage's artifacts and a manifest naming the backend) and
  `refine_ok` (refinement consumed the mesh unchanged) per run.

A reconstruction backend is a `ReconstructionBackend` (`align`, `export`) in
`spatial_ingestion/reconstruction/backends.py`, selected by
`ReconstructionJob.backend_name` or `zshift-image-to-3d --backend`;
`tests/test_backends.py` states what one must produce.

