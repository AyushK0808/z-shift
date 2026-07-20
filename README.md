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

## Refinement

```bash
uv run spatial-ingestion-refine --refine path/to/input.obj --output path/to/output.obj
```

`clean_mesh` defaults to object mode when no config is supplied, and `clean_ai_mesh` remains available as a compatibility alias. The refinement stage is a cleanup pass, not a geometric fusion step, so watertight output depends on the input mesh and settings.
