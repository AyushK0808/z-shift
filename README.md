# Spatial Ingestion Phase 1

Phase 1 of a 2D-to-3D generation pipeline: data ingestion and pre-processing.

## Implemented Scope

- Static uploads: single image, image folder, single video, video folder.
- Live ingestion: authenticated WebSocket frame push after `/v1/ingest/streams/connect`.
- Originals are preserved in the local object-store stub under `data/object_store/`; normalized PNG derivatives are written under `data/normalized/`.
- Image/video normalization preserves aspect ratio and never pads to a square canvas or stretches frames.
- Live stream, auth, and rate-limit state are in-memory and single-process for this research prototype.

RTSP and WebRTC are intentionally rejected until real transport handlers are added.

## Run

```bash
uv sync --dev
uv run uvicorn spatial_ingestion.main:app --reload
```

## Test Harness

```bash
uv run python scripts/test_harness.py
uv run pytest
```

The harness creates a synthetic image, synthetic video, and mock live stream, then verifies routing and unified schema output for each.
