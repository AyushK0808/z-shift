# z-shift — 2D-to-3D Generation Pipeline

A 2D-to-3D generation pipeline. **Phase 1** ingests and normalizes heterogeneous 2D media
into a single unified schema; **Phase 2** converts that schema into reconstruction jobs and
produces 3D geometry with MASt3R; **Phase 3** refines the raw reconstruction into a clean
mesh (preserving vertex colors); **Phase 4** routes the finished geometry to a
use-case-specific deliverable — an editable `.glb`, a packaged point cloud, or a real-time
stream; **Phase 5** (WIP) fits a skeleton and skinning weights.

## Implemented Scope

- Static uploads: single image, image folder, single video, video folder.
- Live ingestion: authenticated WebSocket frame push after `/v1/ingest/streams/connect`.
- Originals are preserved in the local object-store stub under `data/object_store/`;
  normalized PNG derivatives are written under `data/normalized/`.
- Image/video normalization preserves aspect ratio and never pads to a square canvas or
  stretches frames.
- Live stream, auth, and rate-limit state are in-memory and single-process for this
  research prototype.

RTSP and WebRTC are intentionally rejected until real transport handlers are added.

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
  reconstruction/             Phase 2 — job builder, alignment, pairing, export, CLI
  refinement/                 Phase 3 — mesh cleaning, GLB-aware mesh IO
  outcomes_engine/            Phase 4 — use-case router + deliverable packaging/export
  auto_rigging/               Phase 5 — skeleton / skinning-weights MVP (see AUTO_RIGGING.md)
  final_pipeline/             Orchestration — Phase 1→2→3 (→4) end-to-end entry point
  storage/                    Object-store stub for gateway originals
  test_harness/               Synthetic media factory + harness
data/normalized/              Phase 1 normalized media outputs
data/reconstruction/          Phase 2 raw mesh + manifests (one dir per job)
data/deliverables/            Phase 4 packaged deliverables
scripts/setup-mast3r.sh       clones upstream MASt3R into third_party/mast3r (macOS/Linux/WSL)
scripts/setup-mast3r.ps1      same, for Windows PowerShell
scripts/refinement.py         legacy wrapper for the Phase 3 API (prefer the package)
```

`third_party/` is the intended place for upstream reconstruction repos. The runner code
prefers a local `third_party/mast3r` checkout before falling back to globally installed
packages. Run the setup script (see [Team Setup](#team-setup)) to clone it at the pinned
commit.

---

## Phase 1 — Ingestion

The data ingestion and pre-processing service. It accepts heterogeneous 2D media —
single images, image folders, single videos, video folders, and live streams —
classifies each input, routes it to the appropriate processing track, normalizes it, and
emits a single **unified spatial ingestion schema** (`UnifiedSpatialIngestionSchema`) that
downstream 3D-reconstruction stages can consume regardless of where the media came from.

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
  - **Track B — Live** (`live_stream/`) for real-time streams over WebSocket, with a
    bounded frame buffer and backpressure handling (accept / drop decisions)
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

### API surface

- `GET  /health` — liveness probe.
- `POST /v1/ingest/uploads` — multipart upload of one or more image/video files; returns
  the unified schema for the batch.
- `POST /v1/ingest/streams/connect` — open a live WebSocket stream and get back a stream
  handle plus the unified schema.
- `WS   /v1/ingest/streams/{stream_id}/frames` — push encoded frames; each frame gets a
  backpressure decision (accepted, action, dropped-frame count) in reply.

### Run

```bash
uv run uvicorn spatial_ingestion.main:app --reload
```

Upload a batch and save the returned schema JSON for later replay (`--from-schema`):

```bash
curl -H "authorization: Bearer dev-token" \
     -F "files=@a.jpg" -F "files=@b.jpg" \
     http://localhost:8000/v1/ingest/uploads -o payload.json
```

---

## End-to-end pipeline — `zshift-final-pipeline`

The main entry point. By default it runs **Phase 1 (ingest) → Phase 2 (MASt3R) → Phase 3
(refine)**; with `--use-case` it also runs **Phase 4**.

```bash
# Image folder → raw GLB + refined GLB (default output format is .glb)
uv run zshift-final-pipeline path/to/views

# Explicit outputs + Phase 4 editing deliverable (Blender-ready .glb)
uv run zshift-final-pipeline path/to/views -o out/mesh.glb \
     --refined-output out/mesh_refined.glb --use-case editing

# Replay a gateway payload (e.g. a video-folder capture with sync map)
uv run zshift-final-pipeline ignored-folder --from-schema payload.json -o out/mesh.glb
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--from-schema <json>` | — | Skip ingestion; consume a Phase 1 gateway payload |
| `-o, --output` | `data/reconstruction/<name>_<id>/<name>.glb` | Raw Phase 2 mesh path (`.glb`, `.obj`, `.ply`) |
| `--refined-output` | `<raw>_refined<ext>` next to the raw mesh | Phase 3 cleaned mesh path |
| `--use-case` | — | `editing` (→ `.glb` deliverable), `viewing` (→ `.ply` point cloud), `live` (not implemented) |
| `--input-type` | Phase 1 classified type | SourceType for Phase 4 routing |
| `--device` | `auto` | `cuda`, `cpu`, `mps` |
| `--model` | `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` | model id or local checkpoint |
| `--pairing-strategy` | auto | `complete` or `swin`; auto picks `swin` for videos and >20-frame image sets (an explicit flag overrides the auto choice) |
| `--image-size` | 512 | MASt3R input resolution |
| `--tsdf-thresh` | 0 | TSDF fusion threshold; `0.1`–`0.5` recommended but expensive |
| `--min-conf-thr` | 1.5 | Minimum confidence for point filtering |
| `--seed` | — | Random seed for reproducible runs |
| `--refinement-mode` | `object` | `object` or `room` |
| `--smoothing-iters` | 15 | Taubin iterations (`0` disables — see performance note) |
| `--pass-band` / `--hole-size` / `--min-cell-count` / `--feature-angle` / `--merge-tolerance` / `--decimate-target-reduction` | 0.1 / auto / 500 / 45.0 / 1e-5 / — | Phase 3 tuning |
| `--no-watertight-check` | off | Skip the open-edge watertight check |

### Outputs

- Raw mesh: `data/reconstruction/<label>_<job_id>/<label>.glb` + `run_manifest.json`
  (model, device, pairing, seed, TSDF config, reproducibility metadata) +
  `point_cloud.ply` (dense, confidence-masked point cloud used by `--use-case viewing`) +
  `cache/` (MASt3R alignment cache, reused between runs on the same frames).
- Refined mesh + `refinement_manifest.json` next to the raw mesh.
- Deliverables under `data/deliverables/`: `blender_ready/<JOB>_model.glb` (`editing`),
  `point_clouds/<JOB>_points.ply` (`viewing`).

### Known limitations

- `--use-case viewing` packages the Phase 2 `point_cloud.ply` as a `.ply` deliverable under
  `data/deliverables/point_clouds/`.
- `--use-case live` raises `TrackNotImplementedError` by design.
- Phase 3 is CPU-bound and slow on large reconstructions — `split_bodies()` dominates. A
  ~2.6M-triangle mesh takes ~7 minutes even with `--smoothing-iters 0
  --no-watertight-check`; the default 15 smoothing iterations take substantially longer.
  Plan accordingly or run the raw mesh directly.
- Vertex colors are preserved through refinement (nearest-point transfer), but the first
  MASt3R run downloads the checkpoint (~2.5 GB) and the RoPE "slow pytorch version"
  warning is expected without a CUDA toolkit installed.

---

## Phase 2 — Reconstruction

Phase 2 consumes Phase 1's normalized output and produces 3D geometry. `reconstruction/`
contains the job builder, MASt3R inference/pairing/alignment, mesh export, and CLIs.

### Reconstruction job builder (`reconstruction/jobs.py`)

The `ReconstructionJobBuilder` converts a `UnifiedSpatialIngestionSchema` into a
`ReconstructionJob`. It:

- Orders and rewrites normalized frames into `HandoffFrame`s (each carrying its normalized
  asset URI, index, source, timestamp, motion score, resolution, and camera intrinsics).
- Maps each source type to a `ReconstructionMode` (multi-view, video-sequence, or
  synchronized-views). Single-view and live-stream modes are rejected.
- Caps frames at `MAX_RECONSTRUCTION_FRAMES` (default 40), selecting the highest-motion
  frames when the limit is exceeded. Recommends `swin` pairing for videos and multi-view
  jobs with more than `SWIN_PAIRING_THRESHOLD` frames (default 20).
- Rebuilds per-timestamp `SyncViewGroup`s for video-folder captures from the Phase 1 sync
  map.

### Pipeline (`reconstruction/pipeline.py`)

`run(job)` resolves image URIs, resolves the device (CUDA/CPU/MPS), runs sparse global
alignment (`alignment.py`), then exports a mesh (`export.py`) with vertex colors:

1. **Reproducible seeding** — when `--seed` is provided, seeds Python / NumPy / PyTorch
   RNGs and records torch/CUDA/numpy versions in the manifest.
2. **Pairing** — `complete` or `swin` scene-graph pairing (`pairing.py`); for
   `SYNCHRONIZED_VIEWS` jobs, `sync_view_groups` drive cross-camera pairs within each sync
   group, falling back to the configured strategy if no sync pairs are generated.
3. **Sparse alignment** — runs `sparse_global_alignment` through MASt3R's pipeline.
4. **TSDF fusion** — optionally fuses depth with a configurable threshold. On failure
   (memory or runtime), falls back to dense point-map export and records `tsdf_fallback:
   true` in the manifest.
5. **Export** — writes a mesh (GLB by default; `.obj`/`.ply` supported) with configurable
   `min_conf_thr` point filtering. GLB and PLY preserve vertex colors.

### Reconstruction CLI (`zshift-image-to-3d`)

Standalone Phase 2: requires a folder containing at least two images of the same subject
from different views (single-image reconstruction is intentionally not supported).

```bash
uv run zshift-image-to-3d path/to/folder/of/images          # -> .glb by default
uv run zshift-image-to-3d path/to/folder -o ./output.obj    # explicit format
```

---

## Phase 3 — Mesh Refinement

Phase 2 produces geometry straight from the reconstruction models, which is typically
noisy: disconnected floating fragments, open holes, rough surfaces, and inconsistent
normals. Phase 3 (`refinement/`) cleans that raw mesh into a polished, optionally
watertight result ready for rendering, simulation, or export.

The entry point is `clean_mesh(mesh, config=None, **overrides)` (alias `clean_ai_mesh`).
It operates on a [PyVista](https://docs.pyvista.org/) `DataSet` — use `load_mesh_file` /
`write_mesh_file` for file I/O (GLB is supported in both directions; pyvista cannot write
GLB natively, so it is routed through trimesh). The pipeline runs these steps, each
wrapped so any VTK/PyVista failure is reported with the failing step name:

1. **Validate** — reject empty meshes or meshes with NaN/Inf coordinates; unwrap
   MultiBlock/GLB containers (warns when a multi-primitive scene is truncated to the first
   mesh).
2. **Component filter** — depends on the mode:
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
6. **Transfer colors** — vertex color arrays are carried onto the cleaned mesh via
   nearest-point transfer so colors survive smoothing/decimation.
7. **Watertight check** (optional) — count open boundary edges and flag the mesh as
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

```python
from spatial_ingestion.refinement import (
    MeshCleaningConfig,
    clean_mesh,
    load_mesh_file,
    write_mesh_file,
)

raw = load_mesh_file("data/reconstruction/mesh.glb")
result = clean_mesh(raw, MeshCleaningConfig(mode="object", smoothing_iters=15))
# or with keyword overrides: clean_mesh(raw, mode="object", smoothing_iters=15)

print(result["output_point_count"], "points, watertight:", result["is_watertight"])
write_mesh_file(result["mesh"], "data/reconstruction/mesh_refined.glb")
```

The returned dict includes `mesh`, `mode`, input/output point and cell counts,
`is_watertight`, `open_edge_count`, and any `warnings`.

For a standalone CLI (`.glb`, `.obj`, `.ply`, `.stl`, `.vtk` inputs and outputs are
supported):

```bash
uv run spatial-ingestion-refine --refine path/to/input.glb --output path/to/output.glb
```

---

## Phase 4 — Outcomes & Deliverables Engine

Phases 1–3 turn 2D media into clean 3D geometry; Phase 4 (`outcomes_engine/`) decides what
that geometry should *become*. `deliverable_router(input_type, use_case, mesh=..., ...)`
assigns a job id and selects a track:

- **Track A — Editing** (`use_case="editing"`) — exports a Blender-ready `.glb` via
  `export_blender_ready`, for jobs that will be edited in a DCC tool. Valid for image /
  image-folder / video-folder inputs.
- **Track B — Viewing** (`use_case="viewing"` with a `single_video`, `video_folder`, or
  `image_folder` input) — packages point/splat-center data into a `.ply` via
  `export_point_cloud`. (Renamed from `package_4d_gaussian`: it currently exports a plain
  colored point cloud — no covariances, spherical-harmonic coefficients, opacity, or time
  dimension — so the old name overstated what it produces.)
- **Track C — Live** (`use_case="live"` with a `live_stream` input) — intended to
  establish a real-time delivery layer (WebRTC / WebSocket). **Not implemented**: calling
  it raises `TrackNotImplementedError` rather than claiming a stream was established.

`input_type` is validated against the shared `SourceType` enum
(`spatial_ingestion.metadata.schema.SourceType`), and each use case is only valid for a
specific subset of source types (e.g. `editing` is not valid for `live_stream`). Any other
combination raises `InvalidRoutingError`.

Deliverables are written under `data/deliverables/`:

- `blender_ready/<job_id>_model.glb`
- `point_clouds/<job_id>_points.ply`

### Run

The integrated pipeline (`zshift-final-pipeline --use-case editing`) feeds the *real*
refined mesh into the router, so Track A is fully wired end-to-end. The in-memory mocks
(`get_phase3_cleaned_mesh`, `get_phase3_point_cloud`) remain as fallbacks so routing and
packaging can still be exercised without running the upstream models.

```python
from spatial_ingestion.outcomes_engine.engine import deliverable_router

deliverable_router(input_type="image_folder", use_case="editing")   # Track A -> .glb
deliverable_router(input_type="video_folder", use_case="viewing")   # Track B -> .ply
deliverable_router(input_type="live_stream", use_case="live")       # Track C -> raises
```

---

## Phase 5 — Auto-Rigging (WIP)

`auto_rigging/` fits a template skeleton and skinning weights to a mesh. It is **not yet
wired into the final pipeline** (it runs standalone, not on `zshift-final-pipeline`
outputs), and the research behind it lives in [AUTO_RIGGING.md](AUTO_RIGGING.md).

```bash
# Fit a biped skeleton + skinning weights, exporting metadata JSON
uv run zshift-auto-rig out/mesh_refined.glb --articulation biped --output-dir out/rig
```

Flags: `--articulation` (`static`, `biped`, `quadruped`, `winged`; default `static`),
`--max-influences` (default 4), `--no-normalize` (skip unit-scale bounding-box fit),
`--no-export` (return rig data without writing JSON), `--output-dir` (required for
exports; without it no files are written).

---

## Team Setup

Clone the upstream MASt3R source into `third_party/` (not a submodule).
Run `uv sync` first, then the setup script, so the editable installs survive
the environment reconciliation:

```bash
uv python install 3.11
uv sync --dev
bash scripts/setup-mast3r.sh          # macOS / Linux / WSL
```

On Windows, run the PowerShell equivalent:

```powershell
uv python install 3.11
uv sync --dev
./scripts/setup-mast3r.ps1            # PowerShell 5.1+
```

Notes:

- `uv sync` removes packages installed outside the lockfile, including the editable
  `mast3r`/`dust3r` installs. If you run `uv sync` again later, re-run the setup script
  afterwards.
- On Windows, `pyproject.toml` pins torch/torchvision to the CUDA 12.8 wheels
  (`2.11.0+cu128`) via a `pytorch-cu128` uv index; other platforms resolve the CPU
  wheels. This keeps GPU support on Windows without affecting CI.
- Do not install the upstream `requirements.txt` files separately (they force numpy 2.x);
  the project dependency set already includes the MASt3R runtime requirements.
- The expected layout is `third_party/mast3r/` containing `dust3r/` as a submodule,
  pinned to commit `f5209afc300cec36239a7ac992263f36847bbba0`.

MASt3R and its bundled DUSt3R source are licensed under CC BY-NC-SA 4.0. Confirm that this
is compatible with the intended use before distributing or deploying the reconstruction
feature.

## Linting & Type Checking

[ruff](https://docs.astral.sh/ruff/) handles linting (replacing flake8/isort/bandit) and
formatting. [ty](https://github.com/astral-sh/ty) handles static type checking. Both are
configured in `pyproject.toml` under `[tool.ruff]` and `[tool.ty]`.

```bash
uv run ruff check .          # lint
uv run ruff format .         # auto-format
uv run ty check              # type-check
```

Both run automatically on commit via [pre-commit](https://pre-commit.com/). Install the
git hook once per clone:

```bash
uv run pre-commit install
```

CI runs the same two checks (`.github/workflows/lint.yml`) on every push and pull request.

## Tests

```bash
uv run pytest
```

The suite covers the Phase 1 handoff and job building (`test_phase1_handoff.py`), the
Phase 2→3→4 integration (`test_phase2_phase3_pipeline.py`), reconstruction CLI / runner /
manifest (`test_reconstruction_cli.py`, `test_reconstruction_mast3r.py`, `test_mast3r_runner.py`),
refinement incl. GLB color preservation (`test_refinement.py`), and auto-rigging
(`test_auto_rigging.py`). One test (`real_pipeline`) skips unless a user-provided image
directory is present on the machine.
