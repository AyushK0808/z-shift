# Spatial Ingestion

Phase 1 ingestion plus a MASt3R-based multi-view reconstruction boundary for a 2D-to-3D pipeline.

## Repo Layout

```text
src/spatial_ingestion/        application code
third_party/mast3r/           upstream MASt3R checkout or submodule
data/normalized/              Phase 1 normalized media outputs
data/reconstruction/          Phase 2 reconstruction artifacts
```

`third_party/` is the intended place for upstream reconstruction repos. The runner code prefers a local
`third_party/mast3r` checkout before falling back to globally installed packages.

## Run

```bash
uv python install 3.11
uv sync --dev
uv run uvicorn spatial_ingestion.main:app --reload
```

## Team Setup

Recommended setup keeps upstream reconstruction code inside this repo:

```bash
git submodule update --init --recursive
uv python install 3.11
uv sync --dev
```

If submodules are not added yet, the expected layout is:

```bash
third_party/mast3r/
third_party/mast3r/dust3r/
```

The project dependency set includes the MASt3R runtime requirements needed by the local runner. Do not install
the upstream requirements separately; use `uv sync --dev` from this repository.

MASt3R and its bundled DUSt3R source are licensed under CC BY-NC-SA 4.0. Confirm that this is compatible with
the intended use before distributing or deploying the reconstruction feature.

## Reconstruction

Reconstruction is MASt3R-only. It requires a folder containing at least two images of the same subject from
different views. Single-image reconstruction is intentionally not supported.

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

The reconstruction runners currently produce:

- `run_manifest.json`
- `camera_poses.json`
- `point_cloud.ply`
- `mesh.obj`

MASt3R aligns the views, exports camera poses and a point cloud, then exports the dense point-map mesh as OBJ.

## Test Harness

```bash
uv run python scripts/test_harness.py
uv run pytest
```

The harness creates a synthetic image, synthetic video, and mock live stream, then verifies routing and unified schema output for each.
