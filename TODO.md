# TODO — what's left before submission

Companion to [`bench/FINDINGS.md`](bench/FINDINGS.md), which records *what was
measured*. This file records *what to do about it*.

Ordered by what blocks what. P0 items block every long run; do them first.

---

## The constraint that drives the schedule

Tier B (B1–B8) is written and imports cleanly, but **this machine has no CUDA GPU**:

```
torch 2.11.0+cu128  torch.cuda.is_available() = False    nvidia-smi: not found
GPU  Intel Iris Xe (integrated)
CPU  i7-1255U, 12 logical cores       RAM 15.7 GB       Disk 16 GB free of 953 GB (99% full)
```

Measured CPU cost of a MASt3R pair at 512 px: **~62 s**, from the forward-pass
cache timestamps in `data/reconstruction/cache/forward` (4 images, complete
pairing, 6 unique pairs, 22:44:06 → 22:49:17).

What that implies for B4, the experiment to run first:

| | |
|---|---|
| frames per run | 40 |
| pairing | `swin`, winsize 3, symmetrised → ~120 unique pairs |
| forward passes alone | ~120 × 62 s ≈ **2.1 h per run** |
| runs in B4 | 3 variants + 4-point budget sweep = **7** on one GT scene |
| forward passes, B4 total | **~15 h**, before sparse global alignment and TSDF |
| realistic B4 wall clock on CPU | 1.5–3 days |
| realistic B1–B8 on CPU | a week of continuous compute |

**Do not run Tier B on this laptop.** Rent an A5000/4090-class box for a day
(~$0.30–0.80/h on RunPod, Lambda, or Vast); the whole tier is a few hours and
$10–30 there, and B4 drops to well under an hour a run.

---

## P0 — blockers before any long run

- [ ] **Wire `--quick` into the eight Tier B modules.** `experiment_parser` adds
      the flag, every `exp_a*.py` reads it, **no `exp_b*.py` does** —
      `bench/exp_b4_frame_budget_ablation.py:230` builds the parser and never
      passes `args.quick` to `run()`. Today you would type the smoke-test
      command and silently launch the full multi-day grid. This is the single
      most dangerous defect for an unattended run.
- [ ] **Free disk.** 16 GB free, down from the 22 GB this file was written
      against. Each run's alignment cache is ~1.5 GB (240 forward tensors ×
      6.3 MB); `clear_alignment_cache` stops it
      accumulating across runs, but `data/bench/` outputs do accumulate, and a
      DTU download needs room on top.
- [ ] **Get a dataset with ground truth.** `bench/scenes/` holds only
      `example.json`, and nothing named `dtu` or `scan24` exists in the tree.
      B2/B4/B6/B8 skip scenes with `gt_path=None`, so as things stand they
      produce **zero rows**. The one 72-frame sequence you have
      (`data/normalized/ingest_74464a2b…`) has no ground truth — a smoke test,
      not a result. Attaching a Kaggle dataset is now the whole job: §8 of
      `notebooks/tier_b_gpu.ipynb` builds the manifest from whatever is mounted
      and §10 tests it before any GPU time is spent. Neither can invent ground
      truth — a mount of images with no `.ply` still yields zero rows.
- [ ] **Test the orchestration before renting a GPU.** `SceneSet`,
      `load_gt_points`, `clear_alignment_cache`, `build_job` and
      `align_to_reference` have tests. `run_reconstruction`, `score_against_gt`,
      `_run_variant` and all eight `run()` functions do not. A three-day run
      dying at row 1 on a `KeyError` is the predictable outcome. The notebook's
      §10 covers the *data* half of this; these two cover the code half:
  - [ ] monkeypatch `run_phase2` and `point_cloud_from_output` so `_run_variant`
        is exercised end-to-end on CPU in seconds
  - [ ] add a `dry_run=True` walk — `Mast3rRunParams` already supports it — so
        each B module's scene loading, CLI and manifest path can be validated
        without touching MASt3R
- [ ] **Fix the summary print in `exp_b4…main()`.** It indexes `row['f_score']`
      unconditionally, which takes down a *completed* run on any partial row.

---

## P1 — paper decisions you deferred

- [x] **Finding 2 (object mode keeps every fragment).** Reversed the call
      below — took option (a) instead, fixed the code.
      `keep_object_components` now keeps only the largest component by
      `n_cells`, matching what §IV-D already said. Renamed
      `test_multi_sheet_object_mode_keeps_all_components` to
      `..._keeps_only_the_largest_component` and rewrote its assertion rather
      than leaving it pinning the old behaviour. A1/A2's component-count
      ladders relied on object mode keeping every fragment to hold triangle
      count fixed while sweeping component count; they now run in room mode
      with `min_cell_count=0` instead — see the comments in
      `bench/exp_a1_refine_scaling.py` and `bench/exp_a2_stage_profile.py`.
      A1/A2/A3's CSVs were regenerated against the fixed code.
      ~~Take option (b), change the text. The behaviour is deliberate and
      already asserted by `test_multi_sheet_object_mode_keeps_all_components`;
      option (a) breaks a test that encodes a design decision, in order to
      make one sentence true. §IV-D/§V-B/Fig. 3 should say component
      filtering is a **room-mode** operation and that object mode
      deliberately preserves all components.~~
- [ ] **Finding 5 (`_cap_frames` discards capture order).** Still deferred, as
      planned. Fix it in code —
      `src/spatial_ingestion/reconstruction/jobs.py:102`, re-sort by index after
      the cap — but **after B4 runs**. A5/B4 implement V1 and V2 independently of
      the shipped code, so measurement doesn't depend on current behaviour, and
      "found, fixed, quantified" is a far stronger §IV-C than either half alone.
      Flipping `test_cap_frames_does_not_restore_capture_order` is part of the fix.
- [ ] **Table I / §V-C rewrite.** Still unedited at `paper/paper.tex:463-465`
      ("~2.6M triangles", "~7 minutes") and `paper/paper.tex:489`. FINDINGS §1
      has the replacement numbers: report cost as a function of **component
      count** with the measured exponents (1.16–1.28 in triangles, ~0.5 in
      components, super-additive by 5.5×), name `split_bodies` /
      `merge_components` / `finalize_mesh` with their per-regime shares, and
      drop the generic "GPU acceleration" future-work item in favour of the
      batched-merge fix (already landed in code; FINDINGS §1's prose and
      numbers still describe the pre-fix cost and need re-measuring against
      the regenerated `a1_refine_scaling.csv`/`a2_stage_profile.csv`).
- [x] **Finding 3 (inverted hole-fill guard).** Fixed as planned:
      `fill_mesh_holes` now takes a `guard_mesh` argument and `clean_mesh`
      passes the raw, pre-filter mesh, so `is_sheet_like` no longer depends on
      what component filtering happened to leave behind. Re-ran A3 against the
      fixed code; see `bench/FINDINGS.md` #3 for the before/after.
- [ ] **§VII additions:** `compute_priority_score` is recorded but unconsumed;
      `_patch_tsdf_cuda_hardcode` patches `torch.Tensor.cuda` process-wide.

---

## P2 — CPU-only work worth doing now

These need no GPU and close real gaps.

- [ ] **Exercise `MIN_MOTION_VARIANCE`.** It fired in 0 of A6's 4455 trials, so
      it is currently recorded as untested rather than validated. Add a
      low-variance stream axis to A6 so the constant is actually hit. Minutes of
      compute.
- [ ] **Batched merge in `merge_components`.** It chains `merged.merge(piece)` in
      a Python loop, copying the accumulated mesh once per piece — ~35% of
      `component_filter`, which is itself 98.5% of runtime in the
      1000-component regime. A1/A2 already give you the "before", so the "after"
      is one re-run and a genuine optimisation result for §V-C.
- [ ] **Relax or justify `tolerance_ms=120`.** A6 found no breakdown anywhere in
      ±640 ms, so the constant is currently defended by intuition when it could
      be defended by data.
- [ ] **Tune the 0.18 / 0.055 motion thresholds.** A7 shows concentration is set
      by the thresholds, not the intervals, and the adaptive advantage collapses
      to ~1.2× on slow motion because scores never clear them.

---

## Figures still needed — these require Tier B (GPU)

- [ ] **The same before/after plot on a real reconstruction** (B6). `fig8` uses
      synthetic injected fragments; the paper needs stray-fragment removal
      measured on actual MASt3R pointmap output, where a fused four-view
      pointmap can have far more than 1000 islands after confidence masking.
      This is the figure a reviewer will actually want.
- [ ] **V1 vs V2 vs V3 reconstruction quality** (B4) — the headline. A5 already
      established the ordering defect on CPU (Kendall τ ≈ −0.04, median pair gap
      184 → 27 after re-sorting); B4 turns pair adjacency into F-score and
      Chamfer.
- [ ] **Quality vs frame budget curve** (B4 sweep, {10, 20, 40, 60}) — what
      turns "bounds reconstruction cost" into a defensible number.
- [ ] **TSDF fallback characterisation** (B5), including how often
      `_patch_tsdf_cuda_hardcode` was applied.
- [ ] **Determinism bound** (B7) — §V-D currently claims reproducibility with
      manifest backing but no measured run-to-run spread.

---

## Suggested order

1. P0 `--quick` wiring + orchestration tests (half a day, no GPU).
2. P1 CPU fixes: hole-fill guard, A6 variance axis, batched merge — each is a
   re-run of an existing experiment, so each produces a before/after result.
3. Attach the dataset on Kaggle and let §8 of the notebook build the manifest,
   then rent a GPU for one day. §10 must come back green before you start the
   meter. Run **B4 first**; it closes A5 and is the paper's strongest claim.
   Then B6 (gives the real `fig8`), B2, B7, and the rest.
4. Rewrite Table I / §V-C, §V-B and §IV-C against the finished CSVs.
5. Apply the `_cap_frames` fix and flip its pinning test, reporting V1 and V2
   side by side.
