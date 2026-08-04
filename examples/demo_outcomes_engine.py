"""Manual demo of the Phase 4 deliverable router.

Run: uv run python examples/demo_outcomes_engine.py

This replaces the old `if __name__ == "__main__":` block that lived inside
`engine.py` — router logic should be import-safe with no demo side effects,
and this script is where ad-hoc exercising of it belongs instead.
"""

import numpy as np
import trimesh

from spatial_ingestion.outcomes_engine.engine import (
    InvalidRoutingError,
    TrackNotImplementedError,
    deliverable_router,
)


def _synthetic_mesh() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh.visual.vertex_colors = [100, 150, 255, 255]
    return mesh


def _synthetic_point_cloud() -> trimesh.PointCloud:
    rng = np.random.default_rng(0)
    points = rng.random((1000, 3)) * 10
    colors = (rng.random((1000, 3)) * 255).astype(np.uint8)
    return trimesh.PointCloud(vertices=points, colors=colors)


def main() -> None:
    print("==================================================")
    print("  PHASE 4: OUTCOMES & DELIVERABLES ENGINE (demo)  ")
    print("==================================================")

    # Scenario 1: single image -> edit result in Blender.
    result = deliverable_router(
        input_type="single_image",
        use_case="editing",
        job_id="demo_job_1",
        mesh=_synthetic_mesh(),
    )
    print(f"[{result.job_id}] Track A success: {result.output_path}")

    # Scenario 2: video -> view the dynamic 3D scene on the web.
    result = deliverable_router(
        input_type="video_folder",
        use_case="viewing",
        job_id="demo_job_2",
        point_cloud=_synthetic_point_cloud(),
    )
    print(f"[{result.job_id}] Track B success: {result.output_path}")

    # Scenario 3: live camera feed -> not implemented yet, raises.
    try:
        deliverable_router(
            input_type="live_stream",
            use_case="live",
            job_id="demo_job_3",
        )
    except TrackNotImplementedError as exc:
        print(f"Track C: {exc}")

    # Scenario 4: invalid combination -> raises instead of silently failing.
    try:
        deliverable_router(
            input_type="live_stream",
            use_case="editing",
            job_id="demo_job_4",
            mesh=_synthetic_mesh(),
        )
    except InvalidRoutingError as exc:
        print(f"Rejected: {exc}")

    print("\n==================================================")
    print("Demo complete. Check data/deliverables/ for generated files.")


if __name__ == "__main__":
    main()
