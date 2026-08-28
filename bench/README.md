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
  plots.py             regenerates every figure from the CSVs
  results/*.csv        committed results
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

## Ground rules, and how they are enforced

| Rule | Where |
|---|---|
| n ≥ 3 repeats for anything timed, mean ± std | `REPEATS` in A1/A2; `mean_std` uses ddof=1 |
| Machine, OS, Python, library versions recorded | `env_metadata()` stamps `env_*` columns on **every row** |
| Seeds fixed and logged | `seed` column on every row |
| One CSV per experiment | `ResultWriter(EXP_ID)` |
| Figures regenerate from CSVs | `bench/plots.py` reads only `results/` |
| Discard the first MASt3R run | `discard_first` in B2 |
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
