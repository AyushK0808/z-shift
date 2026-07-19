# Z-Shift — Paper Readiness Roadmap

> **Reviewed:** 2026-07-13 · All findings below were verified against the code at commit `965df6d` (+ working tree). Every file:line reference was read, not inferred.
> **PR pass 2026-07-20:** Diffs of open PRs **#3** (Phase 2, Siddhant), **#5** (Phase 4, Jhanvi), **#6** (Phase 3, Rakshit) reviewed line-by-line and folded into the checkboxes below. None are merged yet; #6 has an unresolved merge conflict. Remaining work is tracked per person in [ROADMAP_20-07.md](ROADMAP_20-07.md). Status tags used below: **✅ PR #N** = done, **⚠️ PR #N** = attempted but incomplete/unverified, **⛔ PR #N** = blocked (e.g. merge conflict).
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

> **Status (2026-07-20):** Pieces moved, seam still open. PR #3 gave the Mast3r backend an in-process `execute()` (no more dead `BackendExecutionPlan`) and routes the CLI through the registry; PR #6 gave Phase 3 an importable `clean_mesh` package; PR #5 made Phase 4 return a structured `DeliverableResult`. But **nothing chains Phase 2 → Phase 3 → Phase 4** — there is still no orchestrator, and Phase 4 is still fed in-memory mocks (§5). The single "image folder in, deliverable out" command does not exist. Still `[L]`, still the gate for §8.

### X2 · Phase 1 normalization actively corrupts Phase 2 inputs `[M]` — **Shardul** — ✅ **Fixed in PR #1**

Two different geometric crimes, one per media type:

- **Images are letterboxed onto a 1024×1024 black canvas** ([image_processor.py:36-43](src/spatial_ingestion/batch_normalization/image_processor.py#L36-L43)). MASt3R then resizes to 512 and matches features; the black borders waste resolution, generate spurious border geometry, and shift the effective principal point. MASt3R wants the original images — it does its own resizing ([runners/mast3r.py:160](src/spatial_ingestion/reconstruction/runners/mast3r.py#L160)).
- **Video frames are anisotropically stretched** to 1024×1024 via `cv2.resize` with no aspect preservation ([video_processor.py:36](src/spatial_ingestion/batch_normalization/video_processor.py#L36)). Non-uniform scaling changes fx/fy — this *geometrically corrupts* any reconstruction from video. The image path letterboxes, the video path stretches: the two paths are also inconsistent with each other.
- Both paths **re-encode to JPEG quality 92**, adding compression artifacts to the features MASt3R matches, and **discard the originals** (uploads live in a `TemporaryDirectory`, [api.py:76-84](src/spatial_ingestion/ingestion_gateway/api.py#L76-L84)) — the loss is permanent and unre-doable. The `ObjectStore` built to preserve originals ([object_store.py](src/spatial_ingestion/storage/object_store.py)) is **imported by nothing** — dead code.

Nobody has noticed because of X1: reconstruction has never actually consumed Phase 1 output. **Fix:** preserve aspect ratio everywhere (resize longest side, no canvas, no stretch), keep originals via the ObjectStore, pass originals (not re-encoded copies) to reconstruction, and make "does Phase 1 normalization help or hurt Chamfer distance?" an ablation (§8.1) — it's currently an unexamined assumption that it helps, and the evidence says it hurts.

> **Status:** Images no longer letterboxed onto a black canvas (original resolution preserved); video now resizes via aspect-preserving `_resize_longest_side` instead of anisotropic stretch; both paths switched JPEG→PNG; originals are copied into the `ObjectStore` and referenced via `original_uri` (with cleanup-on-failure so a rejected batch doesn't orphan files). The Chamfer-distance ablation (§8.1) is still outstanding — that's a Phase 2 evaluation task, not a Phase 1 code fix.

### X3 · The multi-source sync contribution is void — twice over `[M–L]` — **Shardul (a) ✅, Siddhant (b)**

The plan's "Multi-Source Sync Logic" novel contribution has zero effect on any output, for two independent reasons:

- **(a) The timestamps being aligned are fictional.** ~~Frame timestamps are `frame_index / fps` ([video_sampler.py:63](src/spatial_ingestion/batch_normalization/video_sampler.py#L63)) — relative to each file's own start.~~ **Fixed in PR #1**: the sampler now reads `CAP_PROP_POS_MSEC` (with the old `frame_index/fps` calc kept only as a fallback for the timestamp==0 edge case), and `multi_source.py` estimates a real per-source clock offset via motion-signature cross-correlation (bucketed median match, gated on a minimum motion-variance threshold) before aligning frames — it no longer assumes every camera started recording at the same instant. The syncer ([multi_source.py:31-56](src/spatial_ingestion/sync/multi_source.py#L31-L56)) previously aligned "1.2 s into camera A" with "1.2 s into camera B" outright; that no-op is gone. Note: the offset estimator is a heuristic (motion-tolerance match + coarse 100ms bucketing), not audio cross-correlation — fine as a first pass, but worth flagging as a limitation rather than a validated method until evaluated.
- **(b) Even if the sync map were correct, it is discarded.** `Mast3rBackend.plan()` for `SYNCHRONIZED_VIEWS` flattens all frames from all sources into one MASt3R call and its only concession to synchronization is `--pairing-strategy swin` ([backends/mast3r.py:65-67](src/spatial_ingestion/reconstruction/backends/mast3r.py#L65-L67)). The sync groups painstakingly rebuilt by the handoff ([assembler.py:60-90](src/spatial_ingestion/generation_handoff/assembler.py#L60-L90)) never influence pairing, alignment, or anything else.
  - **⚠️ PR #3 (attempted, not working):** the runner now builds cross-camera pairs from `sync_view_groups` via `_build_sync_pairs()` and threads them into `run_sparse_alignment`. **But the pairs are the wrong type** — `_build_sync_pairs` returns `list[tuple[int, int]]` (frame indices), while `sparse_global_alignment` consumes `make_pairs`-style pairs of image *dicts* (verified against `origin/phase-2`). Fed integer tuples, it will crash or silently misbehave. The only test asserts the helper's return shape and never runs alignment, so CI is green on a broken feature. **X3(b) is not closed** — fix the pair construction and add an integration test that actually reconstructs, or scope the claim as WIP.

**Fix or cut.** Either implement real offset estimation (audio cross-correlation is the standard cheap method) *and* make sync groups drive pair construction (pair within-timestamp across cameras), then ablate it (reconstruction quality with vs. without sync-aware pairing) — or delete the claim from the paper. A claimed contribution that a reviewer can trace to a no-op is a rejection, not a weakness.

### X4 · Phase 3 destroys Phase 2's output — and the palette claim dies with it `[M]` — **Rakshit + Jhanvi** (+ **Siddhant** for the Phase 2 export side)

- **Largest-component filtering deletes most of the scene.** MASt3R's export builds **one disconnected grid-mesh per view** and concatenates them ([runners/mast3r.py:226-239](src/spatial_ingestion/reconstruction/runners/mast3r.py#L226-L239)); the components are never stitched. Phase 3's default `object` mode keeps only the single largest connected component ([refinement.py:86-91](scripts/refinement.py#L86-L91)) — i.e., **one view's sheet out of N**. Running the advertised Phase 2 → Phase 3 flow discards most of the reconstruction by design.
- **Hole-filling is wrong for this geometry.** Per-view depth meshes are open 2.5D sheets; `fill_holes` with the auto size of 0.5× the bounding diagonal ([refinement.py:71-73](scripts/refinement.py#L71-L73), [119-124](scripts/refinement.py#L119-L124)) will attempt to seal the entire open boundary with garbage faces. The watertight framing assumes closed object scans that Phase 2 never produces.
- **All appearance data is annihilated at the phase boundary.** Phase 2's only appearance channel is vertex colors. They survive into GLB, but Phase 3 reads the **OBJ** via PyVista — VTK's OBJ reader does not read trimesh's non-standard `v x y z r g b` extension — and every subsequent filter (`fill_holes`, `clean`, `decimate_pro`, `compute_normals`) operates on geometry only. Phase 3's output is **colorless**, so Phase 4 exports colorless GLBs, so the project-plan claim of "palette preservation throughout the pipeline" is dead at the first real handoff. Jhanvi's palette scripts (not in the repo — §6) have nothing upstream to preserve.
- Two mesh stacks (trimesh in Phases 2/4, PyVista in Phase 3) with no conversion layer is the root cause. Pick a lossy-free interchange (**PLY with vertex colors** or GLB via trimesh→pyvista conversion), thread colors through every filter, and make Phase 3's component filter fuse per-view sheets (e.g., point-cloud fusion + Poisson reconstruction) instead of deleting them.

> **Status (2026-07-20):** ⚠️ PR #6 (open, **merge-conflicted** — adds `scripts/refinement.py` which already exists on `main`; rebase required before any of this lands). It substantially rewrites Phase 3 in the right direction: `keep_object_components` now **merges every component** instead of keeping only the largest (the sheet-deletion bug is gone); hole-filling is sheet-aware (`is_sheet_like` skips open 2.5D sheets); `preserve_data_arrays` transfers vertex colors/point-data through the filters; the watertight check now also counts non-manifold edges. **Two gaps remain:** (1) the "fusion" is `merge(merge_points=False)` — concatenation, not geometric stitching (no point-cloud/Poisson fusion), so object-mode output stays multi-sheet and effectively never watertight on real Phase 2 input; the README "watertight mesh" claim is still unmet. (2) The color path is only proven *inside* `clean_mesh` — the actual X4 loss point (VTK's OBJ reader dropping trimesh's `v x y z r g b`) is untouched because there is no loader/CLI yet, so **end-to-end color preservation is still unverified.**

---

## 2. Phase 1 — Ingestion (`src/spatial_ingestion/`) — **Shardul**

The best-engineered part of the repo, and also the part with the least paper relevance. It works for the happy path; its claimed intelligence is unvalidated; its security posture is a stub pretending otherwise.

### 2.1 Changes

> **Status (2026-07-14):** All items below were resolved in [PR #1](https://github.com/AyushK0808/z-shift/pull/1) (`codex/phase-1-ingestion` → `main`), verified line-by-line against the actual diff, not just the PR description.

**Small `[S]`**
- [x] **WebSocket frames endpoint has no auth** — `/v1/ingest/uploads` and `/streams/connect` require `auth_context`; the WS endpoint ([api.py:106-110](src/spatial_ingestion/ingestion_gateway/api.py#L106-L110)) accepts anyone, and **auto-creates a stream for any unknown `stream_id`** — so any client can also push frames into any *existing* stream (hijack) or mint unlimited buffers. — **Fixed:** WS endpoint now authenticates, checks `has_stream()` + `is_owner()` before accepting, closes with `WS_1008_POLICY_VIOLATION` on failure, and `open_stream()` raises `StreamOwnershipError` (409) instead of silently overwriting on stream-id collision.
- [x] **Unbounded memory, twice** — streams are never removed from `LiveStreamManager._streams` ([manager.py:22-24](src/spatial_ingestion/live_stream/manager.py#L22-L24)); each buffer holds up to 64 decoded frames (~6 MB each at 1080p ≈ 400 MB/stream); and `push_encoded_frame` runs `cv2.imdecode` on arbitrary-size payloads with no cap ([buffer.py:36-41](src/spatial_ingestion/live_stream/buffer.py#L36-L41)). Add stream close-on-disconnect, a payload size limit, and a stream count limit. — **Fixed:** `close_stream()` called in a `finally` block on WS disconnect; new `MAX_LIVE_STREAMS`, `MAX_LIVE_STREAMS_PER_SUBJECT`, `MAX_LIVE_FRAME_BYTES` config caps enforced (`StreamLimitExceeded` → 429).
- [x] **Auth stub collapses the world into two subjects** — any non-empty `Authorization` header is "authenticated-client", everyone else is "anonymous" ([auth.py:15-18](src/spatial_ingestion/ingestion_gateway/auth.py#L15-L18)), so the rate limiter ([rate_limit.py](src/spatial_ingestion/ingestion_gateway/rate_limit.py)) gives the entire anonymous internet one shared 60 req/min budget. One abuser starves everyone. Key the limiter on client identity or admit in the paper that this is single-tenant. — **Fixed:** subject is now `auth:{sha256(token)[:16]}` for authenticated clients or `anonymous:{client_host}` otherwise — no more shared bucket.
- [x] **Junk files pass classification and then 500** — `classify_static` discards `UNKNOWN` kinds ([router.py:40-41](src/spatial_ingestion/media_classifier/router.py#L40-L41)), so `[photo.jpg, notes.txt]` classifies as `IMAGE_FOLDER`; the normalizer then calls `Image.open` on the `.txt` and the request dies as an unhandled 500. Reject payloads containing unknown items, or skip them with a warning in the schema. — **Fixed:** classifier now returns `UNKNOWN` with a reason naming the offending file(s); endpoint responds 415 instead of 500.
- [x] **Dead branch** — [router.py:47-50](src/spatial_ingestion/media_classifier/router.py#L47-L50): `elif len(media_kinds) > 1` and the `else` both return `UNKNOWN`. Collapse it. — **Fixed:** dead branch removed.
- [x] **Uploads read whole files into RAM** with no size limit ([api.py:81](src/spatial_ingestion/ingestion_gateway/api.py#L81)). Stream to disk with a cap. — **Fixed:** `_stream_upload_to_disk` streams in 1 MB chunks, raises `UploadTooLargeError` → 413 past `MAX_UPLOAD_FILE_BYTES`.
- [x] **`ffprobe` timeout is uncaught** — `subprocess.run(..., timeout=20)` ([ffmpeg_tools.py:30-36](src/spatial_ingestion/batch_normalization/ffmpeg_tools.py#L30-L36)) raises `TimeoutExpired` → 500. Catch it, return `{"available": True, "error": "timeout"}`. — **Fixed** as specified.
- [x] **Dead conditional in `_first_intrinsics`** — both branches return `intrinsics` ([normalizer.py:112-114](src/spatial_ingestion/batch_normalization/normalizer.py#L112-L114)). Also the *first* image's EXIF is applied to the **entire folder** — wrong the moment two cameras contribute. Make intrinsics per-frame (`FrameReference` has no intrinsics field; `CameraIntrinsics` sits at payload level in [schema.py:57](src/spatial_ingestion/metadata/schema.py#L57)). — **Fixed:** `FrameReference` now carries its own `camera_intrinsics`, extracted per image in `_normalize_images()`.
- [x] **The API reaches into private state** — `state.live_streams._streams` ([api.py:109](src/spatial_ingestion/ingestion_gateway/api.py#L109)). Add a public `has_stream()`. — **Fixed** as specified.
- [x] **VFR-wrong timestamps** — see X3(a); switch the sampler to `CAP_PROP_POS_MSEC`. — **Fixed** (see X3(a) above).
- [x] **`estimated_frames` is never passed** by any caller of `LatencyAwareResourceTagger.score` ([priority.py:29](src/spatial_ingestion/resource_tagging/priority.py#L29)) — half the scoring logic is unreachable. — **Resolved by removal:** the dead `estimated_frames` param was cut from `score()` rather than wired up. Reasonable given X1 doesn't exist yet, but confirm this matches how "Latency-Aware Resource Tagging" gets framed in the paper (§2.2 below still applies — the score itself is still consumed by nothing).

**Medium `[M]`**
- [x] **WebRTC and RTSP are fiction** — `/streams/connect` accepts `rtsp_url` and `webrtc_offer_sdp`, stores them in a metadata dict ([api.py:97-103](src/spatial_ingestion/ingestion_gateway/api.py#L97-L103)), and **nothing ever connects to the RTSP URL or answers the SDP offer**. Only WebSocket push works. Either implement one of them (`aiortc` for WebRTC, OpenCV/FFmpeg for RTSP) or delete the parameters and the claim — the README and plan both oversell this. — **Resolved by cutting the claim:** `rtsp_url`/`webrtc_offer_sdp` params removed from `StreamConnectRequest`; `classify_stream` now only accepts `"websocket"`; README's "Implemented Scope" section documents WebSocket-only live ingestion and explicitly notes RTSP/WebRTC are not implemented.
- [x] **Wire up the ObjectStore** (X2) — originals must survive normalization. — **Fixed** (see X2 above); includes cleanup-on-failure via `delete_uri()`.
- [x] **No HTTP-level tests at all** — `httpx` is a dev dependency and is imported by zero tests; every test in [test_phase1.py](tests/test_phase1.py) calls components directly. The FastAPI layer (auth dependency, 415 paths, WS loop) has never been executed by the suite. Add `TestClient` tests including the WS endpoint and the junk-payload 500 above. — **Fixed:** new `TestClient`/WebSocket tests cover original-preservation, 413 oversized upload, 415 junk payload, cross-token WS hijack rejection, WS payload-too-large, and 429 per-subject stream cap.

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

> **Status (2026-07-14):** [PR #2](https://github.com/AyushK0808/z-shift/pull/2) ("Phase 2 cleanup", merged `5aeddb9`) is a **refactor**, not a fix pass — it replaced the `third_party/mast3r` submodule with `scripts/setup-mast3r.sh`, extracted `write_json`/`write_ply`/array-serialization helpers into a new `runners/_io.py`, and collapsed the `generation_handoff` package into `ReconstructionJobBuilder`. Verified line-by-line against the diff (`965df6d..5aeddb9`): only one item below is actually resolved. Everything else in this list is untouched, and one **new regression** was introduced (noted below) — do not check these off on the strength of the PR title alone.

> **Status (2026-07-20):** [PR #3](https://github.com/AyushK0808/z-shift/pull/3) (Phase 2, open, mergeable-clean) is the real fix pass and closes most of §3.1 — verified line-by-line against the diff. Resolved: job-id output dirs, camera-name collision, reproducibility metadata + `--seed`, `--min-conf-thr`, TSDF logging + manifest flag, frame cap, video-sequence mode, intrinsics threading, and the X1 `execute()` executor. **Two items are only partially done** (sync-aware pairing is broken — see X3(b); the CLI `-o` path bypasses the new job-id dir), and **single-image + intrinsics-as-priors remain open.** Individual boxes updated below.

**Small `[S]`**
- [x] **Every default-output run overwrites the last one** — the CLI's default output is `data/reconstruction/<folder>.obj` ([cli.py:93](src/spatial_ingestion/reconstruction/cli.py#L93)), which makes `output_dir` = `data/reconstruction/` for **every run** — so `run_manifest.json`, `camera_poses.json`, `point_cloud.ply`, and the alignment `cache/` are shared and clobbered across runs of *different scenes*. Give every run a job-id directory. — **✅ PR #3 (default path):** default runs now land in a `{stem}_{job_id}` dir, and `_output_dir`/`_job_stem` hashes the frame set. **⚠️ Remaining gap:** the explicit `-o` path does *not* get a job-id — `main()` passes the raw `args.output` straight into `metadata["output_path"]`, and the job-id-adding `resolve_output_path` is now **dead in the runtime path** (only tests call it), so two scenes run with the same `-o` still clobber. Reconcile before relying on it for experiment batches.
- [x] **Job output dirs collide by camera name** — `_job_stem` built the directory from the first three `source_id`s; two different scenes shot with `cam_a`/`cam_b` overwrote each other's artifacts. — **✅ PR #3:** `_job_stem` now appends a SHA-256 of the sorted frame-id set, so distinct scenes get distinct dirs.
- [x] **`-o model.glb` creates a directory named `model.glb`** — `resolve_output_path` treats any non-`.obj` suffix as a directory ([cli.py:85-90](src/spatial_ingestion/reconstruction/cli.py#L85-L90)) → `model.glb/mesh.obj`. Accept `.glb`/`.ply` or error loudly. — **Fixed:** `resolve_output_path` now accepts `.obj`/`.glb` explicitly and only falls back to `.../mesh.obj` for a bare directory path; `mast3r.py`'s exporter also honors the requested suffix instead of always writing both `.obj` and `.glb`. `.ply` explicit output still isn't special-cased, but that's a minor gap, not the reported bug.
- [x] **Zero reproducibility metadata** — the manifest recorded no torch/CUDA version, no seed; nothing seeded torch. Reviewers will run your artifact twice and get different point clouds. — **✅ PR #3:** `--seed` seeds Python/NumPy/PyTorch RNGs; `_reproducibility_metadata()` records torch/CUDA/numpy versions into the manifest. (Still missing: mast3r/dust3r commit hash and `torch.use_deterministic_algorithms` — minor, folded into the remaining-tasks list.)
- [x] **`min_conf_thr=2.0` hardcoded** — the single most quality-relevant threshold wasn't a CLI flag while `--tsdf-thresh` was. — **✅ PR #3:** `--min-conf-thr` exposed on the CLI and threaded through `run()` → `export_sparse_scene_to_path` and into the manifest. It's now an ablation axis.
- [x] **TSDF failure prints to stdout** instead of logging, and the fallback wasn't recorded in the manifest — a silently-degraded run was indistinguishable from a clean one. — **✅ PR #3:** now `logger.warning(...)`, and `export_sparse_scene_to_path` returns a `tsdf_fell_back` flag recorded as `tsdf_fallback` in the manifest.
- [x] **No frame cap** — `complete` pairing is O(N²); a video-folder job flattened *all* sampled frames — ~2,700 symmetrized pairs from a 74-frame sample, which exhausts any GPU. — **✅ PR #3:** `_cap_frames` caps at `MAX_RECONSTRUCTION_FRAMES` (40, highest-motion selection) and multi-view jobs above `SWIN_PAIRING_THRESHOLD` (20) auto-switch to `swin`. **⚠️ Caveat:** for `SYNCHRONIZED_VIEWS` the cap flattens then top-N by motion, which can drop one camera out of a sync group and undermine cross-camera pairing — cap per-group instead (see remaining tasks).

**Medium `[M]`**
- [x] **X1 executor** — run the plan, capture logs, verify `expected_artifacts` exist afterwards. — **✅ PR #3:** `ReconstructionBackend` gained an abstract `execute()`; `Mast3rBackend.execute()` resolves image URIs, calls the runner in-process (not subprocess), and `logger.warning`s any `expected_artifacts` missing on disk afterward. The `plan()` regression from PR #2 is also fixed — `expected_artifacts` now correctly lists only `run_manifest.json` + `mesh.obj`/`.glb` (no more phantom `point_cloud.ply`/`camera_poses.json`). Note: this is an in-process executor, not the standalone `pipeline/orchestrator.py` that chains all four phases (still open under X1).
- [ ] **X3(b) sync-aware pairing** — or cut the claim. — **⚠️ PR #3 attempted but broken:** `_build_sync_pairs()` now builds cross-camera pairs from `sync_view_groups` and the runner threads them into `run_sparse_alignment`, so `job.mode` influences pairing again. **But it returns `list[tuple[int,int]]` where `sparse_global_alignment` expects `make_pairs`-style image-dict pairs** — wrong type, will fail at runtime, and the test never runs alignment (see X3(b) above). **Still open** until the pair format is fixed and covered by an integration test.
- [ ] **Intrinsics are extracted and thrown away** — Phase 1 mines EXIF focal lengths; MASt3R estimates focals itself. Either thread known intrinsics into the sparse alignment as priors (measurable accuracy win — a real ablation) or delete the dead code path. — **⚠️ PR #3 (half):** `HandoffFrame` gained a `camera_intrinsics` field and `_to_handoff_frame` now copies it from `FrameReference`, so intrinsics survive the handoff. **But they are still never passed into `sparse_global_alignment` as priors** — threaded through, then dropped at the model call. Same end result (unused), one step later. Still open.

**Large `[L]`**
- [ ] **Single-image backend** — the plan's core pitch, `jobs.py` rejects it ([jobs.py:16-17](src/spatial_ingestion/reconstruction/jobs.py#L16-L17)), and Phase 5 cannot exist without it. Pick by license as much as quality: **TRELLIS / TripoSR (MIT)** are clean; **Hunyuan3D** has a community license with field-of-use and region restrictions — check before it becomes load-bearing. Slots into the existing backend abstraction, which is genuinely ready for it.
- [x] **Video-sequence mode** — sampled frames are just multi-view images; the cheap version routes `VIDEO_SEQUENCE` to the mast3r backend with `swin` pairing + a frame cap. — **✅ PR #3:** `Mast3rBackend.supports()` now accepts `VIDEO_SEQUENCE`, the job builder routes it with forced `swin` pairing and the frame cap, and it's no longer rejected in `jobs.py`.

### 3.2 Limitations to declare
- MASt3R weights are **CC BY-NC-SA 4.0** — fine for the paper, blocks commercial use; artifact statement must say so.
- Dense per-pixel meshes are view-sheets, not fused surfaces (this is *why* X4 happens); if the paper shows meshes, say how they were fused.
- ASCII PLY export ([runners/mast3r.py:289-310](src/spatial_ingestion/reconstruction/runners/mast3r.py#L289-L310)) is 10–50× larger than binary; fine for sparse clouds, say nothing or switch to binary.

---

## 4. Phase 3 — Refinement (`scripts/refinement.py`) — **Rakshit**

The cleaning code itself is defensively written (step wrapping, diagnostics, config validation — good). Its problems are placement, integration, and wrong assumptions about its input.

> **Status (2026-07-20):** ⛔ [PR #6](https://github.com/AyushK0808/z-shift/pull/6) (Phase 3, open) rewrites most of this section — but it has an **unresolved merge conflict** (adds `scripts/refinement.py`, which already exists on `main`; `uv.lock` likely conflicts too). **Nothing here lands until it's rebased onto `main`.** It also renames the public entry point `clean_ai_mesh` → `clean_mesh`, which breaks callers/docs referencing the old name — keep a compat alias or update every reference.

- [x] **X4 fixes first** — component fusion instead of largest-component deletion; sheet-aware hole policy; color preservation. — **⚠️ PR #6 (partial, blocked on conflict):** largest-component deletion replaced with a merge of all components; `is_sheet_like` gates hole-filling; `preserve_data_arrays` carries colors through the filters. **But** fusion is concatenation not geometric stitching (no Poisson), so object-mode output stays multi-sheet/non-watertight, and the OBJ-reader color-loss point is untouched (no loader). See X4 status note above. `[M]`
- [x] **`pyvista` is not in `pyproject.toml`** — a fresh `uv sync` cannot run Phase 3 at all. — **✅ PR #6:** `pyvista>=0.44.0` added to `dependencies` (and `uv.lock` updated).
- [~] **Move out of `scripts/` into `src/spatial_ingestion/refinement/`** and give it a CLI (or a `--refine` flag on the reconstruction CLI). — **⚠️ PR #6 (half):** moved into `src/spatial_ingestion/refinement/` (`core.py` + `__init__.py`, with a `scripts/refinement.py` compat wrapper). **CLI / `--refine` flag still missing** — it's importable but has no runnable entry point, so it still can't be driven end-to-end. `[S]`
- [x] **Zero tests** — the only phase with no test coverage whatsoever. — **✅ PR #6:** `tests/test_refinement.py` covers object mode + color preservation, room mode, the X4 multi-sheet case, NaN rejection, and decimation. `[M]`
- [x] **Watertight check is boundary-edges only** — non-manifold edges passed silently. — **✅ PR #6:** `count_topology_issues` now extracts both boundary and non-manifold edges and reports both counts; `is_watertight` requires both to be zero.
- [ ] **Retopology and UV/texture optimization** from the plan don't exist (decimation ≠ retopology). Either implement (instant-meshes-style quad remesh is `[L]`) or strike from the paper. For Phase 5, decent topology is not optional — skinning quality depends on it. — **Not addressed by PR #6.** `[L or cut]`
- [x] Nit: config was constructed before rejecting the config+overrides combination — validate first. — **✅ PR #6:** `clean_mesh` raises on `config is not None and overrides` before building `cfg`. `[S]`

---

## 5. Phase 4 — Outcomes Engine (`outcomes_engine/engine.py`) — **Jhanvi**

Currently a 105-line demo script, and the distance between it and the plan's described architecture is the largest in the repo. Brutal but necessary list:

> **Status (2026-07-20):** [PR #5](https://github.com/AyushK0808/z-shift/pull/5) (Phase 4, open, mergeable-clean) is the cleanest of the three PRs and clears every **Small** and both **Honesty** items. **Caveats:** it targets base branch `roadmap`, not `main` (unlike #3/#6 — confirm the intended base / merge order), and despite making the router trivially testable it **adds no tests.** The **Medium** items (de-mock, package split, validation gate, cross-track metadata, FBX-with-bones) are all still open.

**Small `[S]`**
- [x] **It prints instead of returning** — `deliverable_router` returned `None` on every path and reported errors via `print`. — **✅ PR #5:** returns a frozen `DeliverableResult` dataclass; raises `InvalidRoutingError` on bad combos and `TrackNotImplementedError` for unbuilt tracks. Nothing is printed.
- [x] **Vocabulary mismatch with Phase 1** — the router keyed on `"video"`/`"folder"`/... while Phase 1 emits `SourceType` values like `"video_folder"`. — **✅ PR #5:** imports and validates against the shared `spatial_ingestion.metadata.schema.SourceType`; `_coerce_source_type` rejects unknown values. The two-vocabulary handoff bug is gone.
- [x] **`editing` ignores `input_type` entirely** — `live_stream` + `editing` happily exported a GLB. — **✅ PR #5:** explicit per-use-case valid-input sets (`_EDITING_INPUT_TYPES` etc.); mismatches raise `InvalidRoutingError`.
- [x] **Deliverables are written into the source tree** — `src/.../deliverables/` isn't gitignored. — **✅ PR #5:** default output root moved to `<repo>/data/deliverables/` (overridable via `output_root`).
- [x] Remove `time.sleep(1)` demo pauses; move the `__main__` block to a test or example script. — **✅ PR #5:** `__main__` block extracted to `examples/demo_outcomes_engine.py`; sleeps removed; module is import-safe.
- [x] **README fix** — it claimed `trimesh` "is not yet declared in the project dependencies"; it is. — **✅ PR #5:** README corrected.

**Medium `[M]`**
- [ ] **De-mock (X1)** — consume Phase 3's cleaned mesh and Phase 2's point cloud from real artifact paths, delete `get_phase3_cleaned_mesh`/`get_phase3_point_cloud`. — **Not done (acknowledged in PR #5, blocked on X1 orchestrator).**
- [ ] **Split into a package** — router / exporters / validation / delivery modules matching the plan's architecture, with unit tests (currently zero). — **Not done:** PR #5 keeps a single `engine.py` and adds **no tests** despite the new result/error types being directly testable.
- [ ] **Validation gate** — the plan's "Pre-Delivery Validation Gate" doesn't exist in any form. Minimum viable: file exists + loads in trimesh, manifold/watertight flags from Phase 3 diagnostics threaded through, polycount limit, glTF validation (Khronos validator). — **Not done** (routing validation ≠ delivery validation).
- [ ] **Cross-track metadata schema** — the plan's unified metadata contract (source, capture conditions, confidence, LOD tier) is entirely absent; Phase 4 currently receives no metadata at all. Define it as a Pydantic model referencing the Phase 1 schema. — **Not done.**
- [ ] **FBX export with bones** — Phase 5's deliverable is a rigged FBX; GLB also supports skins. Neither is supported (GLB export exists but only for static geometry). This lands on the critical path the moment Phase 5 produces a skeleton. — **Not done.**

**Honesty items**
- [x] **Rename or implement `package_4d_gaussian`** — it wrote a single static PLY of points; no Gaussians, no time dimension, so the name overstated it. — **✅ PR #5:** renamed to `export_point_cloud`, with a docstring stating exactly what it does/doesn't produce and to rename back once real splat export exists.
- [x] **Track C is a print statement** — no WebRTC, no WebSocket, no stream. — **✅ PR #5:** Track C now raises `TrackNotImplementedError` instead of claiming a stream was established; README documents it as not implemented.

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
