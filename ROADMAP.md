# Z-Shift — Paper Readiness Roadmap

> **Reviewed:** 2026-07-13 · All findings below were verified against the code at commit `965df6d` (+ working tree). Every file:line reference was read, not inferred.
> **Effort tags:** `[S]` = hours · `[M]` = days · `[L]` = weeks (these are the paper's actual workload).
> **Owners:** **Shardul** = Phase 1 · **Siddhant** = Phase 2 · **Rakshit** = Phase 3 · **Jhanvi** = Phase 4 + 2D/palettes · Phase 5 + evaluation lead = **TBD**.

---

## 0. Verdict & Contribution Framing

Be honest about what exists: **the repo is well-organized plumbing around MASt3R, and MASt3R does 100% of the 3D work.** Wrapping `sparse_global_alignment` behind FastAPI and Pydantic schemas is engineering, not a contribution. Worse, of the five "genuinely novel" components claimed in the project plan, **two demonstrably have no effect on any output** (the multi-source sync map is computed and then discarded before reconstruction — §1·X3; the compute-priority score is consumed by nothing — §2.3), **two are unvalidated heuristics** (motion-adaptive sampling with hand-picked thresholds; a "decision matrix" that is a MIME/extension if-else), and **one is a mocked demo script** (deliverable routing, §5). A reviewer who reads the code — and artifact evaluators do — will find this in an afternoon.

**The defensible novelty is Phase 5**: single arbitrary image → 3D mesh with **inferred skeleton, skinning weights, and base animations** for non-humanoid categories, exportable as a standard rig. Nothing in the repo implements it yet, but nothing published covers the full image→animatable-arbitrary-category path either (RigNet rigs *meshes*, MagicPony/3D-Fauna handle *animals only*, Mixamo is humanoid-only — verify against 2025–26 literature before freezing claims, this space moves fast). Phases 1–4 become the substrate and the "system" section.

**A second citable contribution is within reach:** an evaluation benchmark of arbitrary-category objects (pets, toys, doodles, robots) with reference rigs + an animator-rated protocol — no such benchmark exists, and it doubles as the paper's eval set.

**What is NOT a contribution and should never be framed as one:** the ingestion gateway, the unified schema, MASt3R invocation, GLB/PLY export, or the mesh-cleaning pipeline (standard VTK filters composed in the obvious order).

| Venue | Fit | Condition |
|---|---|---|
| **3DV / Eurographics** (primary) | Method + system + benchmark | Phase 5 core works; eval plan in §8 executed |
| **SIGGRAPH Asia (TC) / TOG** | Rigging/animation contribution | Skeleton+skinning quality competitive with RigNet-class baselines; animator study |
| **CVPR / ICCV** | Single-image → animatable 3D | Only if the method beats published baselines on public metrics — highest bar |
| **ACM MM** | System + demo | Weakest claim; fallback if Phase 5 underdelivers; requires the end-to-end system to actually run (§1·X1) |
| **WACV / BMVC** | Solid method, lower novelty bar | Realistic fallback with partial Phase 5 results |

---

## 1. Show-Stoppers — fix before running ANY experiment

The paper describes a pipeline: ingest → reconstruct → refine → deliver. **That pipeline does not exist as an executable path in this repository.** These four findings are structural, not polish. Any result produced before fixing them measures a system that isn't the one described.

### X1 · Nothing executes the pipeline — four disconnected islands `[L]` — **Siddhant + Shardul**

- `ReconstructionBackendRegistry.resolve_for_job()` / `Mast3rBackend.plan()` produce a `BackendExecutionPlan` containing a shell command that **no code ever runs** — the only callers are the tests ([test_reconstruction_mast3r.py:151-152](tests/test_reconstruction_mast3r.py#L151-L152)). There is no executor module. The entire Phase 1 → handoff → job → backend chain is dead weight at runtime.
- The actual entry point, `zshift-image-to-3d` ([cli.py:35-65](src/spatial_ingestion/reconstruction/cli.py#L35-L65)), reads a **raw folder from disk**, bypassing Phase 1 normalization, the handoff, the job builder, and the registry entirely. So the two ways to run the system are: an API that stops at normalized JPEGs, and a CLI that never sees them.
- Phase 3 (`clean_ai_mesh`) is called by **nothing** — no CLI, no import outside its own file, no test.
- Phase 4's router fabricates its inputs in-memory ([engine.py:11-25](src/spatial_ingestion/outcomes_engine/engine.py#L11-L25)) and never reads a Phase 2/3 artifact.

**Number of phases actually connected end-to-end: zero.** Fix: a `pipeline/orchestrator.py` that executes `BackendExecutionPlan` via `subprocess` (the plan format already exists — use it), then feeds `mesh.obj`/GLB into Phase 3, then Phase 3 output into Phase 4. One command: image folder in, deliverable out. Everything in §8 depends on this.

### X2 · Phase 1 normalization actively corrupts Phase 2 inputs `[M]` — **Shardul**

Two different geometric crimes, one per media type:

- **Images are letterboxed onto a 1024×1024 black canvas** ([image_processor.py:36-43](src/spatial_ingestion/batch_normalization/image_processor.py#L36-L43)). MASt3R then resizes to 512 and matches features; the black borders waste resolution, generate spurious border geometry, and shift the effective principal point. MASt3R wants the original images — it does its own resizing ([runners/mast3r.py:160](src/spatial_ingestion/reconstruction/runners/mast3r.py#L160)).
- **Video frames are anisotropically stretched** to 1024×1024 via `cv2.resize` with no aspect preservation ([video_processor.py:36](src/spatial_ingestion/batch_normalization/video_processor.py#L36)). Non-uniform scaling changes fx/fy — this *geometrically corrupts* any reconstruction from video. The image path letterboxes, the video path stretches: the two paths are also inconsistent with each other.
- Both paths **re-encode to JPEG quality 92**, adding compression artifacts to the features MASt3R matches, and **discard the originals** (uploads live in a `TemporaryDirectory`, [api.py:76-84](src/spatial_ingestion/ingestion_gateway/api.py#L76-L84)) — the loss is permanent and unre-doable. The `ObjectStore` built to preserve originals ([object_store.py](src/spatial_ingestion/storage/object_store.py)) is **imported by nothing** — dead code.

Nobody has noticed because of X1: reconstruction has never actually consumed Phase 1 output. **Fix:** preserve aspect ratio everywhere (resize longest side, no canvas, no stretch), keep originals via the ObjectStore, pass originals (not re-encoded copies) to reconstruction, and make "does Phase 1 normalization help or hurt Chamfer distance?" an ablation (§8.1) — it's currently an unexamined assumption that it helps, and the evidence says it hurts.

### X3 · The multi-source sync contribution is void — twice over `[M–L]` — **Shardul (a), Siddhant (b)**

The plan's "Multi-Source Sync Logic" novel contribution has zero effect on any output, for two independent reasons:

- **(a) The timestamps being aligned are fictional.** Frame timestamps are `frame_index / fps` ([video_sampler.py:63](src/spatial_ingestion/batch_normalization/video_sampler.py#L63)) — relative to each file's own start. The syncer ([multi_source.py:31-56](src/spatial_ingestion/sync/multi_source.py#L31-L56)) therefore aligns "1.2 s into camera A" with "1.2 s into camera B", which is only correct if every camera started recording at the same instant. There is no cross-stream offset estimation (no audio correlation, no timecode, no visual sync). For real multi-camera footage the sync map is wrong by an arbitrary constant. Also: `frame_index/fps` is wrong for VFR video — use `CAP_PROP_POS_MSEC`.
- **(b) Even if the sync map were correct, it is discarded.** `Mast3rBackend.plan()` for `SYNCHRONIZED_VIEWS` flattens all frames from all sources into one MASt3R call and its only concession to synchronization is `--pairing-strategy swin` ([backends/mast3r.py:65-67](src/spatial_ingestion/reconstruction/backends/mast3r.py#L65-L67)). The sync groups painstakingly rebuilt by the handoff ([assembler.py:60-90](src/spatial_ingestion/generation_handoff/assembler.py#L60-L90)) never influence pairing, alignment, or anything else.

**Fix or cut.** Either implement real offset estimation (audio cross-correlation is the standard cheap method) *and* make sync groups drive pair construction (pair within-timestamp across cameras), then ablate it (reconstruction quality with vs. without sync-aware pairing) — or delete the claim from the paper. A claimed contribution that a reviewer can trace to a no-op is a rejection, not a weakness.

### X4 · Phase 3 destroys Phase 2's output — and the palette claim dies with it `[M]` — **Rakshit + Jhanvi** (+ **Siddhant** for the Phase 2 export side)

- **Largest-component filtering deletes most of the scene.** MASt3R's export builds **one disconnected grid-mesh per view** and concatenates them ([runners/mast3r.py:226-239](src/spatial_ingestion/reconstruction/runners/mast3r.py#L226-L239)); the components are never stitched. Phase 3's default `object` mode keeps only the single largest connected component ([refinement.py:86-91](scripts/refinement.py#L86-L91)) — i.e., **one view's sheet out of N**. Running the advertised Phase 2 → Phase 3 flow discards most of the reconstruction by design.
- **Hole-filling is wrong for this geometry.** Per-view depth meshes are open 2.5D sheets; `fill_holes` with the auto size of 0.5× the bounding diagonal ([refinement.py:71-73](scripts/refinement.py#L71-L73), [119-124](scripts/refinement.py#L119-L124)) will attempt to seal the entire open boundary with garbage faces. The watertight framing assumes closed object scans that Phase 2 never produces.
- **All appearance data is annihilated at the phase boundary.** Phase 2's only appearance channel is vertex colors. They survive into GLB, but Phase 3 reads the **OBJ** via PyVista — VTK's OBJ reader does not read trimesh's non-standard `v x y z r g b` extension — and every subsequent filter (`fill_holes`, `clean`, `decimate_pro`, `compute_normals`) operates on geometry only. Phase 3's output is **colorless**, so Phase 4 exports colorless GLBs, so the project-plan claim of "palette preservation throughout the pipeline" is dead at the first real handoff. Jhanvi's palette scripts (not in the repo — §6) have nothing upstream to preserve.
- Two mesh stacks (trimesh in Phases 2/4, PyVista in Phase 3) with no conversion layer is the root cause. Pick a lossy-free interchange (**PLY with vertex colors** or GLB via trimesh→pyvista conversion), thread colors through every filter, and make Phase 3's component filter fuse per-view sheets (e.g., point-cloud fusion + Poisson reconstruction) instead of deleting them.

---

## 2. Phase 1 — Ingestion (`src/spatial_ingestion/`) — **Shardul**

The best-engineered part of the repo, and also the part with the least paper relevance. It works for the happy path; its claimed intelligence is unvalidated; its security posture is a stub pretending otherwise.

### 2.1 Changes

**Small `[S]`**
- [ ] **WebSocket frames endpoint has no auth** — `/v1/ingest/uploads` and `/streams/connect` require `auth_context`; the WS endpoint ([api.py:106-110](src/spatial_ingestion/ingestion_gateway/api.py#L106-L110)) accepts anyone, and **auto-creates a stream for any unknown `stream_id`** — so any client can also push frames into any *existing* stream (hijack) or mint unlimited buffers.
- [ ] **Unbounded memory, twice** — streams are never removed from `LiveStreamManager._streams` ([manager.py:22-24](src/spatial_ingestion/live_stream/manager.py#L22-L24)); each buffer holds up to 64 decoded frames (~6 MB each at 1080p ≈ 400 MB/stream); and `push_encoded_frame` runs `cv2.imdecode` on arbitrary-size payloads with no cap ([buffer.py:36-41](src/spatial_ingestion/live_stream/buffer.py#L36-L41)). Add stream close-on-disconnect, a payload size limit, and a stream count limit.
- [ ] **Auth stub collapses the world into two subjects** — any non-empty `Authorization` header is "authenticated-client", everyone else is "anonymous" ([auth.py:15-18](src/spatial_ingestion/ingestion_gateway/auth.py#L15-L18)), so the rate limiter ([rate_limit.py](src/spatial_ingestion/ingestion_gateway/rate_limit.py)) gives the entire anonymous internet one shared 60 req/min budget. One abuser starves everyone. Key the limiter on client identity or admit in the paper that this is single-tenant.
- [ ] **Junk files pass classification and then 500** — `classify_static` discards `UNKNOWN` kinds ([router.py:40-41](src/spatial_ingestion/media_classifier/router.py#L40-L41)), so `[photo.jpg, notes.txt]` classifies as `IMAGE_FOLDER`; the normalizer then calls `Image.open` on the `.txt` and the request dies as an unhandled 500. Reject payloads containing unknown items, or skip them with a warning in the schema.
- [ ] **Dead branch** — [router.py:47-50](src/spatial_ingestion/media_classifier/router.py#L47-L50): `elif len(media_kinds) > 1` and the `else` both return `UNKNOWN`. Collapse it.
- [ ] **Uploads read whole files into RAM** with no size limit ([api.py:81](src/spatial_ingestion/ingestion_gateway/api.py#L81)). Stream to disk with a cap.
- [ ] **`ffprobe` timeout is uncaught** — `subprocess.run(..., timeout=20)` ([ffmpeg_tools.py:30-36](src/spatial_ingestion/batch_normalization/ffmpeg_tools.py#L30-L36)) raises `TimeoutExpired` → 500. Catch it, return `{"available": True, "error": "timeout"}`.
- [ ] **Dead conditional in `_first_intrinsics`** — both branches return `intrinsics` ([normalizer.py:112-114](src/spatial_ingestion/batch_normalization/normalizer.py#L112-L114)). Also the *first* image's EXIF is applied to the **entire folder** — wrong the moment two cameras contribute. Make intrinsics per-frame (`FrameReference` has no intrinsics field; `CameraIntrinsics` sits at payload level in [schema.py:57](src/spatial_ingestion/metadata/schema.py#L57)).
- [ ] **The API reaches into private state** — `state.live_streams._streams` ([api.py:109](src/spatial_ingestion/ingestion_gateway/api.py#L109)). Add a public `has_stream()`.
- [ ] **VFR-wrong timestamps** — see X3(a); switch the sampler to `CAP_PROP_POS_MSEC`.
- [ ] **`estimated_frames` is never passed** by any caller of `LatencyAwareResourceTagger.score` ([priority.py:29](src/spatial_ingestion/resource_tagging/priority.py#L29)) — half the scoring logic is unreachable.

**Medium `[M]`**
- [ ] **WebRTC and RTSP are fiction** — `/streams/connect` accepts `rtsp_url` and `webrtc_offer_sdp`, stores them in a metadata dict ([api.py:97-103](src/spatial_ingestion/ingestion_gateway/api.py#L97-L103)), and **nothing ever connects to the RTSP URL or answers the SDP offer**. Only WebSocket push works. Either implement one of them (`aiortc` for WebRTC, OpenCV/FFmpeg for RTSP) or delete the parameters and the claim — the README and plan both oversell this.
- [ ] **Wire up the ObjectStore** (X2) — originals must survive normalization.
- [ ] **No HTTP-level tests at all** — `httpx` is a dev dependency and is imported by zero tests; every test in [test_phase1.py](tests/test_phase1.py) calls components directly. The FastAPI layer (auth dependency, 415 paths, WS loop) has never been executed by the suite. Add `TestClient` tests including the WS endpoint and the junk-payload 500 above.

### 2.2 Additions for the paper
- [ ] **Ablate the motion-adaptive sampler or stop calling it intelligent** `[M]` — thresholds 0.18/0.055 and intervals 24/12/4 ([video_sampler.py:92-97](src/spatial_ingestion/batch_normalization/video_sampler.py#L92-L97)) are hand-picked and never validated. The experiment is cheap and the claim depends on it: uniform sampling vs. adaptive at the *same frame budget*, measured by downstream reconstruction quality (§8.1). If adaptive doesn't win, it's not a contribution, it's a config choice.
- [ ] **Make priority scores do something or cut the claim** `[S–M]` — the score is computed ([priority.py](src/spatial_ingestion/resource_tagging/priority.py)) and consumed by **nothing**: there is no queue, no scheduler, no worker. "Latency-Aware Resource Tagging" is currently a float in a JSON blob. A minimal priority queue in the orchestrator (X1) rescues it; otherwise delete the section from the paper.
- [ ] Per-stage latency + failure telemetry (feeds §8.4). `[S]`

### 2.3 Limitations to declare (not fix)
- Classification trusts client MIME + extension — no content sniffing; a `.jpg`-named MP4 routes wrong.
- All state (streams, rate limits) is per-process and in-memory; a restart loses everything. Single-tenant research prototype — say so.
- `file://` URIs in `FrameReference` tie the pipeline to one machine.

---

## 3. Phase 2 — Reconstruction (`reconstruction/`) — **Siddhant**

The MASt3R integration itself is competent. Everything around it — job identity, artifact management, reproducibility — is not paper-grade.

### 3.1 Changes

**Small `[S]`**
- [ ] **Every default-output run overwrites the last one** — the CLI's default output is `data/reconstruction/<folder>.obj` ([cli.py:93](src/spatial_ingestion/reconstruction/cli.py#L93)), which makes `output_dir` = `data/reconstruction/` for **every run** — so `run_manifest.json`, `camera_poses.json`, `point_cloud.ply`, and the alignment `cache/` are shared and clobbered across runs of *different scenes* ([runners/mast3r.py:67](src/spatial_ingestion/reconstruction/runners/mast3r.py#L67), [162](src/spatial_ingestion/reconstruction/runners/mast3r.py#L162)). Give every run a job-id directory. This silently invalidates any experiment batch run today.
- [ ] **Job output dirs collide by camera name** — `_job_stem` builds the directory from the first three `source_id`s ([backends/mast3r.py:83-88](src/spatial_ingestion/reconstruction/backends/mast3r.py#L83-L88)); two different scenes shot with `cam_a`/`cam_b` overwrite each other's artifacts.
- [ ] **`-o model.glb` creates a directory named `model.glb`** — `resolve_output_path` treats any non-`.obj` suffix as a directory ([cli.py:85-90](src/spatial_ingestion/reconstruction/cli.py#L85-L90)) → `model.glb/mesh.obj`. Accept `.glb`/`.ply` or error loudly.
- [ ] **Zero reproducibility metadata** — the manifest ([runners/mast3r.py:107-133](src/spatial_ingestion/reconstruction/runners/mast3r.py#L107-L133)) records no mast3r/dust3r commit, no torch/CUDA version, no seed; nothing seeds torch. Reviewers will run your artifact twice and get different point clouds. Log all of it; set seeds and deterministic flags.
- [ ] **`min_conf_thr=2.0` hardcoded** ([runners/mast3r.py:196](src/spatial_ingestion/reconstruction/runners/mast3r.py#L196)) — the single most quality-relevant threshold isn't a CLI flag while `--tsdf-thresh` is. Expose it; it's an ablation axis.
- [ ] **TSDF failure prints to stdout** ([runners/mast3r.py:218](src/spatial_ingestion/reconstruction/runners/mast3r.py#L218)) instead of logging, and the fallback isn't recorded in the manifest — a run that silently degraded is indistinguishable from one that didn't.
- [ ] **No frame cap** — `complete` pairing is O(N²); a video-folder job flattens *all* sampled frames ([jobs.py:37-51](src/spatial_ingestion/reconstruction/jobs.py#L37-L51)) — the 74-frame sample already in `data/normalized/` means ~2,700 symmetrized pairs, which will exhaust any GPU. Cap frames (select by motion score — it's already in the schema and currently used for nothing) or force `swin` above a threshold.

**Medium `[M]`**
- [ ] **X1 executor** — run `BackendExecutionPlan`, capture logs, verify `expected_artifacts` exist afterwards (the field exists and is checked by nothing).
- [ ] **X3(b) sync-aware pairing** — or cut the claim.
- [ ] **Intrinsics are extracted and thrown away** — Phase 1 mines EXIF focal lengths; `GenerationHandoff` has **no intrinsics field** ([generation_handoff/models.py:34-43](src/spatial_ingestion/generation_handoff/models.py#L34-L43)), and MASt3R estimates focals itself. Either thread known intrinsics into the sparse alignment as priors (supported upstream, measurable accuracy win — a real ablation) or stop extracting EXIF and delete the dead code path.

**Large `[L]`**
- [ ] **Single-image backend** — the plan's core pitch, `jobs.py` rejects it ([jobs.py:16-17](src/spatial_ingestion/reconstruction/jobs.py#L16-L17)), and Phase 5 cannot exist without it. Pick by license as much as quality: **TRELLIS / TripoSR (MIT)** are clean; **Hunyuan3D** has a community license with field-of-use and region restrictions — check before it becomes load-bearing. Slots into the existing backend abstraction, which is genuinely ready for it.
- [ ] **Video-sequence mode** ([jobs.py:18-19](src/spatial_ingestion/reconstruction/jobs.py#L18-L19) rejects it) — sampled frames are just multi-view images; the cheap version is routing `VIDEO_SEQUENCE` to the mast3r backend with `swin` pairing + a frame cap. Days, not weeks, and it unlocks all video inputs.

### 3.2 Limitations to declare
- MASt3R weights are **CC BY-NC-SA 4.0** — fine for the paper, blocks commercial use; artifact statement must say so.
- Dense per-pixel meshes are view-sheets, not fused surfaces (this is *why* X4 happens); if the paper shows meshes, say how they were fused.
- ASCII PLY export ([runners/mast3r.py:289-310](src/spatial_ingestion/reconstruction/runners/mast3r.py#L289-L310)) is 10–50× larger than binary; fine for sparse clouds, say nothing or switch to binary.

---

## 4. Phase 3 — Refinement (`scripts/refinement.py`) — **Rakshit**

The cleaning code itself is defensively written (step wrapping, diagnostics, config validation — good). Its problems are placement, integration, and wrong assumptions about its input.

- [ ] **X4 fixes first** — component fusion instead of largest-component deletion; sheet-aware hole policy; color preservation. Nothing else in Phase 3 matters until then. `[M]`
- [ ] **`pyvista` is not in `pyproject.toml`** — a fresh `uv sync` cannot run Phase 3 at all. One line. `[S]`
- [ ] **Move out of `scripts/` into `src/spatial_ingestion/refinement/`** and give it a CLI (or a `--refine` flag on the reconstruction CLI). It is currently driven by a README code snippet. `[S]`
- [ ] **Zero tests** — the only phase with no test coverage whatsoever. Synthetic fixtures are trivial here (a sphere with holes, two disjoint sheets, a noisy icosphere). Add: object mode, room mode, the X4 multi-sheet case (this test would have caught it), NaN rejection, decimation. `[M]`
- [ ] **Watertight check is boundary-edges only** ([refinement.py:76-83](scripts/refinement.py#L76-L83)) — non-manifold edges pass silently, and the Phase 4 plan claims "manifold checks". Extend or reword. `[S]`
- [ ] **Retopology and UV/texture optimization** from the plan don't exist (decimation ≠ retopology). Either implement (instant-meshes-style quad remesh is `[L]`) or strike from the paper. For Phase 5, decent topology is not optional — skinning quality depends on it. `[L or cut]`
- [ ] Nit: `clean_ai_mesh` constructs the config before rejecting the config+overrides combination ([refinement.py:159-161](scripts/refinement.py#L159-L161)) — validate first. `[S]`

---

## 5. Phase 4 — Outcomes Engine (`outcomes_engine/engine.py`) — **Jhanvi**

Currently a 105-line demo script, and the distance between it and the plan's described architecture is the largest in the repo. Brutal but necessary list:

**Small `[S]`**
- [ ] **It prints instead of returning** — `deliverable_router` returns `None` on every path and reports errors via `print` ([engine.py:82](src/spatial_ingestion/outcomes_engine/engine.py#L82)). Nothing can programmatically consume it, including tests. Return a result object; raise on invalid routing.
- [ ] **Vocabulary mismatch with Phase 1** — the router keys on `"video"`, `"folder"`, `"single_image"`, `"live_stream"` ([engine.py:71-77](src/spatial_ingestion/outcomes_engine/engine.py#L71-L77)); Phase 1's `SourceType` emits `"single_video"`, `"image_folder"`, `"video_folder"`. Three of five values don't match — the moment Phase 4 receives a real Phase 1 schema, routing silently falls through to the error branch. Import `SourceType`; share the enum. (This is the classic cross-team handoff bug: two vocabularies, no contract, caught only because this review diffed them.)
- [ ] **`editing` ignores `input_type` entirely** — `live_stream` + `editing` happily exports a GLB ([engine.py:65](src/spatial_ingestion/outcomes_engine/engine.py#L65)). Validate the combination matrix.
- [ ] **Deliverables are written into the source tree** — `src/.../deliverables/` ([engine.py:33](src/spatial_ingestion/outcomes_engine/engine.py#L33), [44](src/spatial_ingestion/outcomes_engine/engine.py#L44)), which is *not* gitignored (only `data/` is) — one `uv run` away from committing binary artifacts. Move to `data/deliverables/`.
- [ ] Remove `time.sleep(1)` demo pauses; move the `__main__` block to a test or example script.
- [ ] **README fix** — it claims `trimesh` "is not yet declared in the project dependencies"; it is ([pyproject.toml:23](pyproject.toml#L23)).

**Medium `[M]`**
- [ ] **De-mock (X1)** — consume Phase 3's cleaned mesh and Phase 2's point cloud from real artifact paths, delete `get_phase3_cleaned_mesh`/`get_phase3_point_cloud`.
- [ ] **Split into a package** — router / exporters / validation / delivery modules matching the plan's architecture, with unit tests (currently zero).
- [ ] **Validation gate** — the plan's "Pre-Delivery Validation Gate" doesn't exist in any form. Minimum viable: file exists + loads in trimesh, manifold/watertight flags from Phase 3 diagnostics threaded through, polycount limit, glTF validation (Khronos validator).
- [ ] **Cross-track metadata schema** — the plan's unified metadata contract (source, capture conditions, confidence, LOD tier) is entirely absent; Phase 4 currently receives no metadata at all. Define it as a Pydantic model referencing the Phase 1 schema; this is cheap and it's one of the few plan claims that can be made real quickly.
- [ ] **FBX export with bones** — Phase 5's deliverable is a rigged FBX; GLB also supports skins. Neither is supported (GLB export exists but only for static geometry). This lands on the critical path the moment Phase 5 produces a skeleton.

**Honesty items**
- [ ] **Rename or implement `package_4d_gaussian`** — it writes a single static PLY of points ([engine.py:42-51](src/spatial_ingestion/outcomes_engine/engine.py#L42-L51)). There are no Gaussians (no covariances, no SH coefficients, no opacity) and no time dimension. As named, it's a claim the code contradicts. The real temporal-bundling scheme is `[L]` and depends on Phase 2 emitting splats, which it doesn't (§3.1) — likely post-submission; rename it `export_point_cloud` until then.
- [ ] **Track C is a print statement** ([engine.py:77-79](src/spatial_ingestion/outcomes_engine/engine.py#L77-L79)). No WebRTC, no WebSocket, no stream. Cut it from the paper or implement it; do not describe it in present tense.

---

## 6. 2D Translation & Palettes — **Jhanvi** `[M–L]`

Listed in the plan as a *primary* novel component; **zero lines of it exist in the repo** (verified: no palette/render/POV code anywhere in `src/`). If the scripts exist elsewhere, commit them this week — uncommitted work is unreviewable and unciteable.

- [ ] Multi-angle 2D extraction: render the final model from arbitrary cameras (trimesh/pyrender headless, or Blender `bpy` batch). `[M]`
- [ ] Palette preservation: extract source-image palette, quantify drift through the pipeline (ΔE in Lab space between source palette and rendered-output palette), and enforce/correct at render time. **Note: until X4 is fixed there are no colors to preserve** — this work is blocked on the Phase 2→3 color path. `[M]`
- [ ] If palette preservation stays in the paper, it needs a number: report ΔE distributions across the eval set, not adjectives. `[S once built]`

---

## 7. Phase 5 — Auto-Rigging (the actual contribution) — **lead TBD, all hands**

Everything above is substrate. This section is the paper. Nothing here exists yet; the sequencing in §10 exists to protect the time this needs.

### 7.1 Method work `[L]`
- [ ] **Single-image → mesh backend first** (§3.1) — hard prerequisite; choose for license cleanliness (TRELLIS/TripoSR MIT) as much as mesh quality, because Phase 5's training/eval sits on top of its outputs.
- [ ] **Category/part inference** — coarse kinematic archetype (biped/quadruped/winged/wheeled/amorphous) from image + mesh, conditioning skeleton prediction. Start with a fixed small taxonomy; genuinely arbitrary topology is the stretch goal, not the MVP.
- [ ] **Skeleton prediction** — RigNet-style graph prediction vs. template-retrieval + deformation: implement template-retrieval first (it's the baseline anyway), then decide if learned prediction beats it. That comparison is a required table regardless.
- [ ] **Skinning weights** — geodesic-distance initialization + learned refinement; evaluate against artist weights where available.
- [ ] **Base animation retargeting** — idle/walk/gesture onto inferred skeletons with unpredictable bone counts. This is a real subproblem, not plumbing; scope to 2–3 clips and a fixed retargeting recipe, or it eats the schedule.
- [ ] **Stylization is demo garnish, not a claim** — cartoon/claymation/low-poly variants dilute the rigging contribution and are largely solved elsewhere. Build one (low-poly = decimation + palette quantization, nearly free given Phases 3/6) for the video; do not evaluate it.

### 7.2 Positioning `[S, do first]`
- [ ] Related-work pass **before any engineering**: RigNet/TARig (mesh→rig), Make-It-Animatable, MagicPony/3D-Fauna/LASSIE (single-image articulated animals), Magic Articulate-class 2025 work, commercial (Anything World, Tripo/Meshy auto-rig, Mixamo). The defensible gap is *single image → rigged arbitrary category, end-to-end, with a benchmark*. If a 2025–26 paper already covers it, find out **now**, not at rebuttal.

### 7.3 Evaluation assets `[L]`
- [ ] **Benchmark**: 150–300 objects across ≥5 categories (pets, toys, doodles, robots, figurines), each with a reference rig (commission or curate from licensed rigged model sets — **check licenses**: game-ripped datasets like Models-Resource-derived RigNet data are legally murky for redistribution; this matters for the artifact).
- [ ] **Metrics**: RigNet-standard joint metrics (J2J/J2B/B2B chamfer), skinning L1 vs. reference, deformation quality under canonical poses; plus animator-rated plausibility (the headline where no reference rig exists).
- [ ] **Baselines**: RigNet/TARig applied to your generated meshes; template-fitting; one commercial tool (report as-is, no tuning). Ablations: category conditioning on/off, skeleton representation, mesh quality (raw Phase 2 vs. Phase 3-refined input — this ties the system into the paper).

---

## 8. Evaluation Plan (the results section, enumerated)

1. **Reconstruction system eval** — needed even though MASt3R isn't the claim, because the substrate must be credible: GSO (Google Scanned Objects) renders → reconstruct → Chamfer/F-score vs. ground truth. Ablations: **Phase 1 normalization on/off (X2 — expect "off" wins; report it honestly)**, adaptive vs. uniform sampling at equal budget (§2.2), pairing strategy, TSDF on/off, `min_conf_thr` sweep, refinement on/off (Chamfer + normal consistency + watertightness + polycount).
2. **Sync ablation** (only if X3 is fixed rather than cut): multi-camera reconstruction with vs. without sync-aware pairing.
3. **Rigging eval** (§7.3) — the headline tables.
4. **Systems numbers**: per-phase latency (ingest → normalize → reconstruct → refine → export), GPU memory vs. frame count, artifact sizes. One table; reviewers of systems-flavored venues expect it, and none of it is currently measured.
5. **User study**: animator plausibility ratings, N ≥ 15, pairwise vs. baselines, Holm–Bonferroni; needs IRB/ethics clearance — file early.
6. **Artifact**: code + benchmark + fixed seeds + pinned deps + model revisions. Current state fails artifact evaluation on X1 alone (the described pipeline can't be run).

---

## 9. Ethics & Licensing (required section)

- **People in photos**: single-photo → rigged, animatable 3D of *a person* is deepfake-adjacent. Either scope people out explicitly (recommended — the contribution is non-humanoid anyway) or add consent requirements and a misuse paragraph. Reviewers will ask; decide before submission, not in rebuttal.
- **License matrix, written down**: MASt3R + DUSt3R **CC BY-NC-SA 4.0** (non-commercial, share-alike); single-image backend TBD (prefer MIT: TRELLIS/TripoSR; Hunyuan3D community license has restrictions); rigged-model datasets for §7.3 (Models-Resource-derived sets are not redistributable — budget for licensed/commissioned alternatives); animation clips (Mixamo's terms restrict redistribution — check before bundling clips in the artifact).
- **Uploaded media**: the gateway keeps normalized copies of user media forever under `data/normalized/` with no retention policy (X2 makes originals vanish while derivatives persist). One paragraph + a cleanup policy.

---



**One-sentence summary:** four phases of clean-looking code currently form zero working pipelines — the sync and priority "contributions" provably do nothing, normalization damages the reconstruction it feeds, and refinement deletes most of what it receives — so the path to a publishable paper is: make the substrate true (X1–X4), cut every claim the code contradicts, and spend the recovered time on Phase 5 and its benchmark, which is the only part of this project a top venue will accept as new.
