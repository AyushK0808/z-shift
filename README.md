# Spatial Ingestion Phase 1

Phase 1 of a 2D-to-3D generation pipeline: data ingestion and pre-processing.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
uvicorn spatial_ingestion.main:app --reload
```

## Test Harness

```bash
python scripts/test_harness.py
pytest
```

The harness creates a synthetic image, synthetic video, and mock live stream, then verifies routing and unified schema output for each.

