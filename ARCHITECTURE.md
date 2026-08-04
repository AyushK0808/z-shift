# z-shift Architecture — How the Four Phases Fit Together

This document maps how the 2D-to-3D pipeline is organized: what each phase
does, its public entry points, the data contract it passes to the next phase,
and the naming vocabulary the whole codebase shares.

The pipeline is one coherent flow: **Phase 1 ingests 2D media → Phase 2
reconstructs 3D geometry → Phase 3 refines it → Phase 4 packages a
deliverable.** Everything is orchestrated by `final_pipeline/`; Phase 5
(auto-rigging) is research code not yet wired in.

```
                UnifiedSpatialIngestionSchema          ReconstructionJob
  ┌─────────────┐  (payload JSON, file:// URIs)  ┌──────────────────┐
  │  PHASE 1    │ ─────────────────────────────▶ │  PHASE 2         │
  │  Ingestion  │   data/payloads/*.json         │  Reconstruction  │
  └─────────────┘                                └────────┬─────────┘
      gateway API             zshift-image-to-3d          │ mesh, point_cloud.ply,
      batch_normalization     (standalone Phase 2 CLI)    │ run_manifest.json
      media_classifier                                      ▼
      live_stream / sync / resource_tagging         ┌──────────────┐
      metadata schema                               │  PHASE 3      │
                                                    │  Refinement   │
  zshift-final-pipeline ── Phase 1→2→3 (→4) ──────▶ │              │
                                                    └──────┬───────┘
                                                           │ refined mesh,
                                                           │ refinement_manifest.json
                                                           ▼
                                                    ┌──────────────┐
                                                    │  PHASE 4      │
                                                    │  Outcomes     │
                                                    │  data/deliverables/
                                                    └──────────────┘
```

## The shared vocabulary (one concept, one name)

| Concept | Name | Where |
|---|---|---|
| Media origin kind | `SourceType` (`single_image`, `image_folder`, `single_video`, `video_folder`, `live_stream`) | `metadata/schema.py` |
| Ingestion lane | `Track` (`batch` = "track_a_batch", `live` = "track_b_live") | `metadata/schema.py` |
| Phase 1 output | `UnifiedSpatialIngestionSchema` (a "payload" on disk) | `metadata/schema.py` |
| Phase 2 unit of work | `ReconstructionJob` (with `params: Mast3rRunParams`) | `reconstruction/models.py` |
| Phase 2 run params | `Mast3rRunParams` | `reconstruction/models.py` |
| Reconstruction mode | `ReconstructionMode` (`multi_view`, `video_sequence`, `synchronized_views`) | `reconstruction/models.py` |
| Normalized frame | `FrameReference` (same model in Phase 1 and Phase 2) | `metadata/schema.py` |
| Sync group | `SyncMapEntry` (Phase 1) → `SyncViewGroup` (Phase 2) | schema / models |
| Run identifier | `job_id` (12-char hex, threaded through every phase and artifact) | `reconstruction/models.py` |
| Phase 4 route | `use_case` (`editing`, `viewing`, `live`) | `outcomes_engine/engine.py` |
| Deliverable lane | `DeliverableResult.track` = the `use_case` value | `outcomes_engine/engine.py` |
| Output roots | constants in `spatial_ingestion/config.py` | — |

Deliberate renames for coherence: `RoutingDecision.source_type` (was
`input_type`), CLI flag `--source-type` (was `--input-type`, kept as hidden
alias), `ReconstructionRunResult` from `pipeline.run` (was an exit code).

## Phase 1 — Ingestion

**Job:** accept heterogeneous 2D media (single image, image folder, single
video, video folder, live stream), classify it, normalize it to PNG frames,
and emit one `UnifiedSpatialIngestionSchema`.

**Entry points**

- `uv run uvicorn spatial_ingestion.main:app` — FastAPI gateway.
  - `POST /v1/ingest/uploads` → schema (also persisted to
    `data/payloads/`, stamped with `metadata.payload_uri`)
  - `POST /v1/ingest/streams/connect` + `WS /v1/ingest/streams/{id}/frames`
    (in-memory only; live has no downstream handoff yet)
- `ingest_batch(paths)` in `final_pipeline/handoff.py` — in-process
  classify + normalize (the final pipeline's Phase 1).

**Internal modules:** `ingestion_gateway/` (HTTP/WS, auth, rate limit),
`media_classifier/` (extension/MIME decision matrix),
`batch_normalization/` (image/video normalization, EXIF, motion-adaptive
sampling), `live_stream/`, `sync/` (multi-camera timestamp alignment),
`resource_tagging/` (compute-priority score), `metadata/` (the schema).

**Handoff contract → Phase 2:** the schema. `frames[].uri` are `file://`
URIs of normalized PNGs that must exist on disk; `sync_map` entries must
reference `(source_id, index)` pairs that exist in `frames`. A payload can
be replayed via `zshift-final-pipeline --from-schema <path>`.

## Phase 2 — Reconstruction (MASt3R)

**Job:** turn normalized frames into raw 3D geometry with MASt3R: pairing →
sparse global alignment → optional TSDF fusion → mesh + dense point cloud.

**Entry points**

- `ReconstructionJobBuilder().build(payload)` in `reconstruction/jobs.py`
  (schema → job: mode mapping, 40-frame cap by motion score, sync groups,
  swin-pairing default for videos / large sets).
- `run(job)` in `reconstruction/pipeline.py` (exported as `run_pipeline`
  and `run_reconstruction`) → `ReconstructionRunResult` with
  `output_path`, `output_dir`, `point_cloud_path`, `manifest_path`.
- `zshift-image-to-3d` (standalone Phase 2 CLI, image folder only).

**Internal modules:** `pairing.py` (complete/swin scene graphs + sync-aware
cross-camera pairs + EXIF intrinsics priors), `alignment.py` (MASt3R sparse
global alignment), `inference.py` (model/image loading with cache),
`export.py` (dense points → mesh, `point_cloud.ply`, run manifest),
`device.py` (device resolution, seeding, reproducibility metadata),
`paths.py` (job-id-aware output path resolution), `input.py` (image folder
collection), `_deps.py` (unified MASt3R-not-installed error).

**Artifacts (in the job's output dir):** raw mesh (`.glb`/`.obj`/`.ply`),
`point_cloud.ply`, `run_manifest.json` (params + `job_id` + `mode` +
`label` + `provenance`), `cache/`. The output folder embeds `job_id`
(`<stem>_<job_id>/`).

## Phase 3 — Refinement

**Job:** clean the raw mesh (validate, component filter, hole fill, Taubin
smoothing, finalize, vertex-color transfer, optional watertight check).

**Entry points**

- `clean_mesh(mesh, config)` in `refinement/core.py` → diagnostics dict
  (also `clean_ai_mesh` alias).
- `refine_mesh_file(input_path, output_path, config)` — load → clean →
  write in one call; `default_refined_path()` names the default output
  `<stem>_refined<suffix>`.
- `spatial-ingestion-refine <mesh>` (standalone CLI; positional input).

**Config:** `MeshCleaningConfig` (mode object/room, smoothing, hole size,
decimation, watertight check). The same knobs are exposed by both CLIs via
`refinement/options.py`.

**Handoff → Phase 4:** refined mesh file + `refinement_manifest.json`
(diagnostics) written next to the raw mesh. For `editing` the refined mesh
is loaded and routed; for `viewing` Phase 4 uses Phase 2's
`point_cloud.ply` instead.

## Phase 4 — Outcomes & Deliverables

**Job:** route the finished geometry to a use-case deliverable.

**Entry points**

- `deliverable_router(source_type, use_case, *, job_id, mesh | point_cloud)`
  in `outcomes_engine/engine.py` → `DeliverableResult`.
  - `editing` → `blender_ready/<job_id>_model.glb`
  - `viewing` → `point_clouds/<job_id>_points.ply`
  - `live` → `TrackNotImplementedError` (not built yet)
- `validate_routing(source_type, use_case)` — rejects invalid combinations
  before any compute runs.

## Orchestration — `final_pipeline/`

- `handoff.py`: `ingest_batch`, `load_schema`, `build_job` (schema → job +
  params merge + job-id-aware output resolution), `run_from_schema`,
  `run_ingested_pipeline`.
- `core.py`: `run_phase2_phase3_pipeline` (run → validate artifact →
  refine → manifests) and `run_full_pipeline` (+ Phase 4 routing).
- `cli.py`: `zshift-final-pipeline [folder|--from-schema] [-o out] 
  [--refined-output] [--use-case] [--source-type] ...`
- `FinalPipelineResult` carries every artifact path; `FullPipelineResult`
  adds the `DeliverableResult`.

## Accepted frictions (known and intentionally left)

- `Track` enum values (`track_a_batch`/`track_b_live`) keep their legacy
  string values so persisted payloads under `data/payloads/` still load.
- Phase 5 (`auto_rigging/`) is not wired into the final pipeline; its
  `ReconstructionArtifactKind` rig values are declarations only.
- Live ingestion is a sink: nothing from Track B is persisted or
  reconstructable yet (`--use-case live` fails fast).
- `scripts/refinement.py` and `scripts/test_harness.py` are legacy
  sys.path wrappers kept for compatibility.
- `data/normalized/` and `data/payloads/` grow unboundedly (no GC yet).
