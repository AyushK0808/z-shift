# Z-Shift — Remaining Tasks (as of 2026-07-20)

> Snapshot after the PR pass on **#3** (Phase 2), **#5** (Phase 4), **#6** (Phase 3). See [ROADMAP.md](ROADMAP.md) for the full findings and per-item PR status. This file tracks only **what's left**, grouped by owner.
>
>
> Effort tags: `[S]` hours · `[M]` days · `[L]` weeks.

---

## Cross-cutting (blocks the paper) — **Siddhant + Shardul**

- [ ] **X1 · Pipeline orchestrator** `[L]` — a `pipeline/orchestrator.py` that chains Phase 2 → Phase 3 → Phase 4 into one "image folder in, deliverable out" command. PR #3 added an in-process `Mast3rBackend.execute()` and PR #6 made Phase 3 importable, but **nothing wires the phases together** and Phase 4 is still fed mocks. Everything in the evaluation plan (§8) depends on this.

---

## Shardul — Phase 1 (Ingestion)

Phase 1 code is largely settled (PRs #1). Remaining work is paper-facing, not bug-fixing.

- [ ] **X2 · Phase 1 normalization ablation** `[M]` — "does normalization help or hurt Chamfer distance?" Run once the orchestrator exists (§8.1). Currently an unexamined assumption; evidence suggests it hurts.
- [ ] **Ablate the motion-adaptive sampler** `[M]` — uniform vs. adaptive sampling at the *same frame budget*, measured by downstream reconstruction quality. Thresholds (0.18/0.055) are hand-picked and unvalidated; the "intelligent sampling" claim depends on this.
- [ ] **Make priority scores do something or cut the claim** `[S–M]` — the compute-priority score is consumed by nothing. A minimal priority queue in the orchestrator (X1) rescues "Latency-Aware Resource Tagging"; otherwise delete the section.
- [ ] **Per-stage latency + failure telemetry** `[S]` — feeds the systems-numbers table (§8.4).
- [ ] Declare limitations (not fix): MIME/extension-only classification, per-process in-memory state, `file://` URIs tie the pipeline to one machine.

---

## Siddhant — Phase 2 (Reconstruction)

PR #3 closed most of §3.1. Remaining:

**Fix before merging PR #3**
- [ ] **X3(b) · Sync-aware pairing is broken** `[M]` — `_build_sync_pairs()` returns `list[tuple[int,int]]`, but `sparse_global_alignment` expects `make_pairs`-style image-dict pairs. Will fail at runtime; the test never runs alignment. Fix the pair construction **and** add an integration test that actually reconstructs, or scope the claim as WIP. Then ablate (reconstruction quality with vs. without sync-aware pairing, §8.2).
- [ ] **CLI `-o` bypasses the job-id dir** `[S]` — `main()` passes raw `args.output` into `metadata["output_path"]`; the job-id-adding `resolve_output_path` is now dead in the runtime path. Two scenes with the same `-o` still clobber. Route the explicit-output path through the same job-id logic (or thread `resolve_output_path` back in).
- [ ] **Frame cap can break sync groups** `[S]` — `SYNCHRONIZED_VIEWS` flattens then keeps top-N by motion, which can drop one camera from a group. Cap per-group instead of globally.

**Still open (not touched by PR #3)**
- [ ] **Intrinsics as priors** `[M]` — PR #3 threads `camera_intrinsics` through `HandoffFrame` but never passes them into `sparse_global_alignment`. Either feed them as focal priors (measurable accuracy win + a real ablation) or delete the EXIF path.
- [ ] **Single-image backend** `[L]` — the core prerequisite for Phase 5. Pick for license cleanliness (TRELLIS/TripoSR MIT). Slots into the existing backend abstraction.
- [ ] **Reproducibility polish** `[S]` — add mast3r/dust3r commit hash to the manifest and `torch.use_deterministic_algorithms`; PR #3 records versions + seed but not these.
- [ ] Declare limitations: MASt3R weights CC BY-NC-SA 4.0; dense meshes are view-sheets not fused surfaces; ASCII PLY export size.

---

## Rakshit — Phase 3 (Refinement)

PR #6 is a strong rewrite but **blocked on a merge conflict** and incomplete on the hard parts.

**Unblock PR #6**
- [ ] **Resolve the merge conflict** `[S]` — PR #6 adds `scripts/refinement.py` which already exists on `main` (add/add conflict); `uv.lock` likely conflicts too. Rebase onto `main` and resolve.
- [ ] **Handle the `clean_ai_mesh` → `clean_mesh` rename** `[S]` — keep a compat alias or update every caller/doc referencing the old name.
- [ ] Give `clean_mesh` a `config=None` default (currently a required positional; `clean_mesh(mesh)` errors).

**Finish X4 for real**
- [ ] **True geometric fusion** `[M–L]` — PR #6's `merge_components` concatenates per-view sheets (`merge_points=False`); it stops the deletion bug but doesn't stitch. Object-mode output stays multi-sheet and non-watertight, so the "watertight mesh" claim is still unmet. Implement point-cloud fusion + Poisson (or equivalent).
- [ ] **End-to-end color path** `[M]` — `clean_mesh` preserves colors *if present*, but the actual X4 loss is at the OBJ→PyVista read boundary (VTK drops trimesh's `v x y z r g b`). Adopt PLY-with-vertex-colors or a trimesh→pyvista converter as the interchange and prove color survives Phase 2 → Phase 3 → Phase 4.

**Still open**
- [ ] **CLI / `--refine` flag** `[S]` — PR #6 makes refinement importable but adds no runnable entry point; it can't be driven end-to-end yet.
- [ ] **Retopology / UV / texture optimization** `[L or cut]` — doesn't exist (decimation ≠ retopology). Needed for Phase 5 skinning quality, or strike from the paper.

---

## Jhanvi — Phase 4 (Outcomes) + 2D Translation & Palettes

PR #5 cleared all the small/honesty items. Remaining:

**Phase 4 — confirm & test PR #5**
- [ ] **Confirm PR #5's base branch** `[S]` — it targets `roadmap`, not `main` (unlike #3/#6). Retarget or agree merge order.
- [ ] **Add tests** `[S]` — the new `DeliverableResult`/error types are directly testable; PR #5 adds none. Cover each track + invalid combos.

**Phase 4 — Medium (unstarted)**
- [ ] **De-mock (X1)** `[M]` — consume real Phase 2/3 artifact paths; delete the `get_phase3_*` mocks.
- [ ] **Split into a package** `[M]` — router / exporters / validation / delivery modules per the plan's architecture.
- [ ] **Pre-delivery validation gate** `[M]` — file loads in trimesh, manifold/watertight flags threaded from Phase 3, polycount limit, glTF (Khronos) validation.
- [ ] **Cross-track metadata schema** `[M]` — a Pydantic contract (source, capture conditions, confidence, LOD tier) referencing the Phase 1 schema.
- [ ] **FBX export with bones** `[M]` — lands on the critical path the moment Phase 5 produces a skeleton (GLB skins also acceptable).

**2D Translation & Palettes (§6) — nothing committed yet**
- [ ] **Multi-angle 2D extraction** `[M]` — render the final model from arbitrary cameras (trimesh/pyrender headless or Blender `bpy`).
- [ ] **Palette preservation** `[M]` — extract source palette, quantify drift (ΔE in Lab), enforce/correct at render time. **Blocked on X4** — no colors to preserve until the Phase 2→3 color path exists.
- [ ] **Report ΔE distributions** `[S once built]` — numbers, not adjectives, if the claim stays in the paper.

---

## TBD lead — Phase 5 (Auto-Rigging) + Evaluation

The actual contribution. Nothing exists yet; sequencing protects the time it needs.

- [ ] **Positioning / related-work pass first** `[S]` — RigNet/TARig, Make-It-Animatable, MagicPony/3D-Fauna/LASSIE, Magic-Articulate-class 2025 work, commercial auto-riggers. Confirm the gap (single image → rigged arbitrary category + benchmark) is still open **now**.
- [ ] **Method work** `[L]` — single-image→mesh backend (shared with Siddhant), category/archetype inference, skeleton prediction (template-retrieval baseline first), skinning weights, base-animation retargeting. Stylization is demo garnish, not a claim.
- [ ] **Evaluation assets** `[L]` — 150–300 objects across ≥5 categories with reference rigs (check licenses); RigNet-standard metrics + animator-rated plausibility; baselines (RigNet/TARig, template-fitting, one commercial tool) and ablations.
- [ ] **Systems + user-study eval** `[M–L]` — per-phase latency/GPU tables (§8.4), animator study N≥15 with Holm–Bonferroni (file IRB early), artifact packaging with fixed seeds + pinned deps.

---