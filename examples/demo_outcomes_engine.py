"""Manual demo of the Phase 4 deliverable router.

Run: uv run python examples/demo_outcomes_engine.py

This replaces the old `if __name__ == "__main__":` block that lived inside
`engine.py` — router logic should be import-safe with no demo side effects,
and this script is where ad-hoc exercising of it belongs instead.
"""

from spatial_ingestion.outcomes_engine.engine import (
    InvalidRoutingError,
    TrackNotImplementedError,
    deliverable_router,
)


def main() -> None:
    print("==================================================")
    print("  PHASE 4: OUTCOMES & DELIVERABLES ENGINE (demo)  ")
    print("==================================================")

    # Scenario 1: single image -> edit result in Blender.
    result = deliverable_router(input_type="single_image", use_case="editing")
    print(f"[{result.job_id}] Track A success: {result.output_path}")

    # Scenario 2: video -> view the dynamic 3D scene on the web.
    result = deliverable_router(input_type="video_folder", use_case="viewing")
    print(f"[{result.job_id}] Track B success: {result.output_path}")

    # Scenario 3: live camera feed -> not implemented yet, raises.
    try:
        deliverable_router(input_type="live_stream", use_case="live")
    except TrackNotImplementedError as exc:
        print(f"Track C: {exc}")

    # Scenario 4: invalid combination -> raises instead of silently failing.
    try:
        deliverable_router(input_type="live_stream", use_case="editing")
    except InvalidRoutingError as exc:
        print(f"Rejected: {exc}")

    print("\n==================================================")
    print("Demo complete. Check data/deliverables/ for generated files.")


if __name__ == "__main__":
    main()
