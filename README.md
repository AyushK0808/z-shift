# Spatial Ingestion Phase 1

Phase 1 of a 2D-to-3D generation pipeline: data ingestion and pre-processing.

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
