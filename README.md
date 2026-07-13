# Spatial Ingestion

A 2D-to-3D generation pipeline. **Phase 1** ingests and normalizes heterogeneous 2D
media into a single unified schema; **Phase 2** takes that normalized output across a
generation-handoff boundary and reconstructs 3D geometry with MASt3R; **Phase 3** refines
that raw reconstruction into a clean, watertight mesh ready for downstream use; **Phase 4**
routes the finished geometry to a use-case-specific deliverable — an editable interchange
file, a packaged point cloud, or a real-time stream.

## Repo Layout

```text
src/spatial_ingestion/
  ingestion_gateway/          Phase 1 — HTTP + WebSocket API, auth, rate limiting
  media_classifier/           Phase 1 — MIME/extension decision matrix + routing
  batch_normalization/        Phase 1 — image/video normalization, EXIF, frame sampling
  live_stream/                Phase 1 — real-time stream buffering + backpressure
  sync/                       Phase 1 — multi-source timestamp alignment
  resource_tagging/           Phase 1 — compute-priority scoring
  metadata/                   Phase 1 — UnifiedSpatialIngestionSchema
  generation_handoff/         Phase 2 — normalized payload -> generation-ready view
  reconstruction/             Phase 2 — job builder, backends, MASt3R runner, CLI
  outcomes_engine/            Phase 4 — use-case router + deliverable packaging/export
scripts/refinement.py         Phase 3 — mesh cleaning / refinement (clean_ai_mesh)
scripts/setup-mast3r.sh        clones upstream MASt3R into third_party/mast3r
data/normalized/              Phase 1 normalized media outputs
data/reconstruction/          Phase 2 reconstruction artifacts
src/spatial_ingestion/outcomes_engine/deliverables/   Phase 4 packaged deliverables
```

`third_party/` is the intended place for upstream reconstruction repos. The runner code
prefers a local `third_party/mast3r` checkout before falling back to globally installed
packages. Run `bash scripts/setup-mast3r.sh` to clone it at the pinned commit.

---

## Phase 1 — Ingestion

The data ingestion and pre-processing service. It accepts heterogeneous 2D media —
single images, image folders, single videos, video folders, and live streams —
classifies each input, routes it to the appropriate processing track, normalizes it, and
emits a single **unified spatial ingestion schema** (`UnifiedSpatialIngestionSchema`) that
downstream 3D-reconstruction stages can consume regardless of where the media came from.

The service is a FastAPI gateway that turns any supported 2D input into one consistent,
structured document. Under the hood it handles the full ingestion path:

- **Ingestion gateway** (`ingestion_gateway/`) — HTTP + WebSocket endpoints with pluggable
  auth and an in-memory, per-subject rate limiter (stub interfaces that mirror what a
  production auth/limiter would expose).
- **Media classifier & router** (`media_classifier/`) — a decision matrix that inspects
  MIME types and file extensions to classify each payload as a single image, image folder,
  single video, video folder, or live stream, and picks a processing track. Mixed or
  unrecognized payloads are rejected as `unknown`.
- **Two processing tracks:**
  - **Track A — Batch** (`batch_normalization/`) for uploaded files. Images are normalized
    and their EXIF camera intrinsics (make, model, focal length, etc.) are extracted;
    videos are probed with FFmpeg and sampled into frames using a motion-adaptive
    frame-diff strategy.
  - **Track B — Live** (`live_stream/`) for real-time streams over WebSocket, WebRTC, or
    RTSP, with a bounded frame buffer and backpressure handling (accept / drop decisions)
    so a fast producer can't overwhelm the service.
- **Multi-source sync** (`sync/`) — for video folders, aligns frames across sources by
  nearest timestamp within a tolerance, producing a sync map so multi-camera captures stay
  temporally coherent.
- **Latency-aware resource tagging** (`resource_tagging/`) — assigns each input a
  normalized compute-priority score at ingestion time (live streams get top priority;
  batch scores vary by source type and payload size).
- **Unified metadata schema** (`metadata/`) — every path, batch or live, returns the same
  Pydantic model: source type, track, resolution, frame count, camera intrinsics, priority
  score, frame references, and any sync map.

The result is that everything downstream sees one schema, whether the input was a photo, a
folder of clips, or a live RTSP feed.

### API surface

- `GET  /health` — liveness probe.
- `POST /v1/ingest/uploads` — multipart upload of one or more image/video files; returns
  the unified schema for the batch.
- `POST /v1/ingest/streams/connect` — open a live stream (WebSocket / WebRTC / RTSP) and
  get back a stream handle plus the unified schema.
- `WS   /v1/ingest/streams/{stream_id}/frames` — push encoded frames; each frame gets a
  backpressure decision (accepted, action, dropped-frame count) in reply.

### Run

```bash
uv python install 3.11
uv sync --dev
uv run uvicorn spatial_ingestion.main:app --reload
```

---

## Phase 2 — Generation Handoff & Reconstruction

Phase 2 consumes Phase 1's normalized output and produces 3D geometry. It is split into a
handoff boundary and a reconstruction stage.

### Generation handoff (`generation_handoff/`)

The `GenerationHandoffBuilder` turns a `UnifiedSpatialIngestionSchema` into a
generation-ready `GenerationHandoff`. It:

- Orders and rewrites normalized frames into `HandoffFrame`s (each carrying its normalized
  asset URI, index, source, timestamp, motion score, and resolution).
- Maps each source type to a `GenerationMode` (single-view, multi-view, video-sequence,
  synchronized-views, or live-stream).
- Rebuilds per-timestamp `SyncViewGroup`s for video-folder captures from the Phase 1 sync
  map.
- Flags whether the payload is reconstruction-ready. Live-stream payloads are explicitly
  **not** reconstruction-ready in Phase 2 and are returned with a warning.

### Reconstruction (`reconstruction/`)

- **Job builder** (`jobs.py`) — the `ReconstructionJobBuilder` converts a reconstruction-ready
  handoff into a `ReconstructionJob`. Multi-view and synchronized-view modes are supported;
  single-view, video-sequence, and live-stream modes are intentionally rejected for now.
- **Backend abstraction** (`backends/`) — a `ReconstructionBackend` interface (`supports` /
  `plan`) produces a `BackendExecutionPlan`. The default multi-view backend is `mast3r`.
- **MASt3R runner** (`runners/mast3r.py`) — aligns the views, exports camera poses and a
  point cloud, then exports the dense point-map mesh as OBJ.

### Reconstruction CLI

Reconstruction is MASt3R-only. It requires a folder containing at least two images of the
same subject from different views. Single-image reconstruction is intentionally not
supported.

The direct image-to-3D command is:

```bash
uv python install 3.11
uv sync --dev
uv run zshift-image-to-3d path/to/folder/of/images
```

Use `-o` to control the final `.obj` path:

```bash
uv run zshift-image-to-3d path/to/folder -o ./output/object.obj
```

Useful flags:

- `--device` — `cuda`, `cpu`, or `auto` (default `auto`).
- `--model` — MASt3R model id or local checkpoint path (default
  `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`).
- `--pairing-strategy` — `complete` or `swin` (default `complete`).
- `--image-size` — MASt3R image size (default `512`).
- `--tsdf-thresh` — TSDF fusion threshold; `0` disables it, `0.1`–`0.5` recommended but
  expensive.
- `--dry-run` — validate routing without running the models.

### Outputs

Reconstruction artifacts are written under `data/reconstruction/`:

- `run_manifest.json`
- `camera_poses.json`
- `point_cloud.ply`
- `mesh.obj`

---

## Phase 3 — Mesh Refinement

Phase 2 produces geometry straight from the reconstruction models, which is typically
noisy: disconnected floating fragments, open holes, rough surfaces, and inconsistent
normals. Phase 3 (`scripts/refinement.py`) cleans that raw mesh into a polished,
optionally watertight result ready for rendering, simulation, or export.

The entry point is `clean_ai_mesh(mesh, config=None, **overrides)`. It operates on a
[PyVista](https://docs.pyvista.org/) `DataSet` **already in memory** — it performs no file
I/O — and returns the cleaned mesh plus diagnostics. The pipeline runs these steps, each
wrapped so any VTK/PyVista failure is reported with the failing step name:

1. **Validate** — reject empty meshes or meshes with NaN/Inf coordinates.
2. **Component filter** — the behaviour depends on the mode:
   - `object` (default) — keep only the single largest connected component, discarding
     stray fragments.
   - `room` — split into bodies and keep every component larger than `min_cell_count`,
     then merge them (for scenes made of multiple legitimate pieces).
3. **Fill holes** — close boundary holes up to `hole_size` (auto-sized to the mesh's
   bounding diagonal when not specified).
4. **Smooth** — Taubin smoothing (shrink-free). In `room` mode, feature edges and
   boundaries are preserved using `feature_angle`.
5. **Finalize** — merge coincident points, triangulate, optionally decimate
   (`decimate_target_reduction`), and recompute consistent, outward-facing normals.
6. **Watertight check** (optional) — count open boundary edges and flag the mesh as
   watertight; a non-watertight result is reported as a warning rather than an error.

### Configuration

Behaviour is controlled by `MeshCleaningConfig` (or the same fields passed as keyword
overrides):

- `mode` — `object` or `room` (default `object`).
- `smoothing_iters` — Taubin smoothing iterations (default `15`; `0` disables smoothing).
- `pass_band` — Taubin pass-band (default `0.1`).
- `hole_size` — max hole size to fill; `None` auto-sizes to the model scale.
- `min_cell_count` — `room` mode: drop components at/below this size (default `500`).
- `feature_angle` — `room` mode: sharp-edge preservation threshold (default `45.0`).
- `merge_tolerance` — relative tolerance for duplicate-point merging (default `1e-5`).
- `decimate_target_reduction` — e.g. `0.5` drops ~50% of triangles; `None` keeps all.
- `verify_watertight` — run the open-edge watertight check (default `True`).

### Run

`clean_ai_mesh` is a library function rather than a CLI, so it is driven from Python. It
requires `pyvista`, which is not yet declared in the project dependencies — install it into
the environment first:

```bash
uv pip install pyvista
```

Then load a Phase 2 mesh, clean it, and write the result:

```python
import pyvista as pv
from scripts.refinement import clean_ai_mesh, MeshCleaningConfig

raw = pv.read("data/reconstruction/mesh.obj")

result = clean_ai_mesh(raw, MeshCleaningConfig(mode="object", smoothing_iters=15))
# or with keyword overrides: clean_ai_mesh(raw, mode="object", smoothing_iters=15)

print(result["output_point_count"], "points, watertight:", result["is_watertight"])
result["mesh"].save("data/reconstruction/mesh_refined.obj")
```

The returned dict includes `mesh`, `mode`, input/output point and cell counts,
`is_watertight`, `open_edge_count`, and any `warnings`.

---

## Phase 4 — Outcomes & Deliverables Engine

Phases 1–3 turn 2D media into clean 3D geometry; Phase 4 (`outcomes_engine/`) decides what
that geometry should *become*. Rather than producing one fixed output, it routes each job to
a use-case-specific delivery track and packages the result into the right format.

The core is `deliverable_router(input_type, use_case)`, which assigns a job id and selects a
track:

- **Track A — Editing** (`use_case="editing"`) — exports a Blender-ready `.glb` interchange
  file via `export_blender_ready`, for jobs that will be edited in a DCC tool.
- **Track B — Viewing** (`use_case="viewing"` with a `video` or `folder` input) — packages
  the point cloud / Gaussian-splat data into a `.ply` via `package_4d_gaussian`, for
  web-based viewing of dynamic 3D scenes.
- **Track C — Live** (`use_case="live"` or a `live_stream` input) — establishes a real-time
  delivery layer (WebRTC / WebSocket) instead of writing a file.

Any other combination is rejected as an invalid routing request.

Packaged deliverables are written under
`src/spatial_ingestion/outcomes_engine/deliverables/`:

- `blender_ready/<job_id>_model.glb`
- `4d_gaussians/<job_id>_splat.ply`

### Run

The current module is a proof of concept: it mocks the Phase 3 handoff with in-memory
geometry (`get_phase3_cleaned_mesh`, `get_phase3_point_cloud`) so the routing and packaging
paths can be exercised without running the upstream models. It requires `trimesh`, which is
not yet declared in the project dependencies — install it into the environment first:

```bash
uv pip install trimesh
```

Then run the proof-of-concept, which fires one request through each track:

```bash
uv run python src/spatial_ingestion/outcomes_engine/engine.py
```

Or drive the router directly from Python:

```python
from spatial_ingestion.outcomes_engine.engine import deliverable_router

deliverable_router(input_type="single_image", use_case="editing")   # Track A -> .glb
deliverable_router(input_type="video", use_case="viewing")          # Track B -> .ply
deliverable_router(input_type="live_stream", use_case="live")       # Track C -> stream
```

---

## Team Setup

Clone the upstream MASt3R source into `third_party/` (not a submodule):

```bash
bash scripts/setup-mast3r.sh
uv python install 3.11
uv sync --dev
```

The expected layout is:

```bash
third_party/mast3r/
third_party/mast3r/dust3r/
```

The project dependency set includes the MASt3R runtime requirements needed by the local
runner. Do not install the upstream requirements separately; use `uv sync --dev` from this
repository.

MASt3R and its bundled DUSt3R source are licensed under CC BY-NC-SA 4.0. Confirm that this
is compatible with the intended use before distributing or deploying the reconstruction
feature.

## Test Harness

```bash
uv run python scripts/test_harness.py
uv run pytest
```

The harness creates a synthetic image, synthetic video, and mock live stream, then verifies
routing and unified schema output for each. The `tests/` suite covers Phase 1 ingestion
(`test_phase1.py`), the generation handoff (`test_generation_handoff.py`), and the Phase 2
reconstruction CLI, runner, and MASt3R integration.
