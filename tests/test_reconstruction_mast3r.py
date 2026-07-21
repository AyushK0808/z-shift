from pathlib import Path

from spatial_ingestion.metadata.schema import (
    FrameReference,
    SourceType,
    SyncMapEntry,
    Track,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.reconstruction import (
    ReconstructionJobBuilder,
    ReconstructionMode,
)
from spatial_ingestion.reconstruction.models import ReconstructionJob


def test_mast3r_job_builder_maps_image_folder_to_multi_view() -> None:
    payload = UnifiedSpatialIngestionSchema(
        source_type=SourceType.IMAGE_FOLDER,
        track=Track.BATCH,
        resolution=(1024, 1024),
        frame_count=2,
        is_stream=False,
        compute_priority_score=0.5,
        frames=[
            FrameReference(
                frame_id="view_0",
                uri="file:///tmp/view_0.jpg",
                index=0,
                source_id="front",
                resolution=(1024, 1024),
            ),
            FrameReference(
                frame_id="view_1",
                uri="file:///tmp/view_1.jpg",
                index=1,
                source_id="side",
                resolution=(1024, 1024),
            ),
        ],
    )

    job = ReconstructionJobBuilder().build(payload)

    assert job.backend_name == "mast3r"
    assert job.mode == ReconstructionMode.MULTI_VIEW
    assert job.image_uris == ["file:///tmp/view_0.jpg", "file:///tmp/view_1.jpg"]


def test_mast3r_job_builder_flattens_synchronized_views() -> None:
    payload = UnifiedSpatialIngestionSchema(
        source_type=SourceType.VIDEO_FOLDER,
        track=Track.BATCH,
        resolution=(1024, 1024),
        frame_count=4,
        is_stream=False,
        compute_priority_score=0.6,
        sync_group_id="sync_123",
        frames=[
            FrameReference(
                frame_id="cam_a_0",
                uri="file:///tmp/cam_a_0.jpg",
                index=0,
                source_id="cam_a",
                timestamp_ms=0.0,
                resolution=(1024, 1024),
            ),
            FrameReference(
                frame_id="cam_b_0",
                uri="file:///tmp/cam_b_0.jpg",
                index=0,
                source_id="cam_b",
                timestamp_ms=5.0,
                resolution=(1024, 1024),
            ),
            FrameReference(
                frame_id="cam_a_1",
                uri="file:///tmp/cam_a_1.jpg",
                index=1,
                source_id="cam_a",
                timestamp_ms=100.0,
                resolution=(1024, 1024),
            ),
            FrameReference(
                frame_id="cam_b_1",
                uri="file:///tmp/cam_b_1.jpg",
                index=1,
                source_id="cam_b",
                timestamp_ms=103.0,
                resolution=(1024, 1024),
            ),
        ],
        sync_map=[
            SyncMapEntry(
                sync_group_id="sync_123",
                anchor_timestamp_ms=0.0,
                aligned_frames={"cam_a": 0, "cam_b": 0},
                offsets_ms={"cam_b": 5.0},
            ),
            SyncMapEntry(
                sync_group_id="sync_123",
                anchor_timestamp_ms=100.0,
                aligned_frames={"cam_a": 1, "cam_b": 1},
                offsets_ms={"cam_b": 3.0},
            ),
        ],
    )

    job = ReconstructionJobBuilder().build(payload)

    assert job.mode == ReconstructionMode.SYNCHRONIZED_VIEWS
    assert job.image_uris == [
        "file:///tmp/cam_a_0.jpg",
        "file:///tmp/cam_b_0.jpg",
        "file:///tmp/cam_a_1.jpg",
        "file:///tmp/cam_b_1.jpg",
    ]
    assert len(job.sync_view_groups) == 2


def test_pipeline_dry_run(tmp_path: Path) -> None:
    from spatial_ingestion.reconstruction.pipeline import run as pipeline_run

    image_a = tmp_path / "front.jpg"
    image_b = tmp_path / "side.jpg"
    image_a.write_bytes(b"front")
    image_b.write_bytes(b"side")

    output_path = tmp_path / "artifacts" / "mesh.obj"
    job = ReconstructionJob(
        mode=ReconstructionMode.MULTI_VIEW,
        image_uris=[str(image_a), str(image_b)],
        output_path=str(output_path),
        metadata={"dry_run": True},
    )

    exit_code = pipeline_run(job)

    assert exit_code == 0
    assert (tmp_path / "artifacts" / "run_manifest.json").exists()


def test_pipeline_rejects_single_view() -> None:
    from spatial_ingestion.reconstruction.pipeline import run as pipeline_run

    job = ReconstructionJob(
        mode=ReconstructionMode.SINGLE_VIEW,
        image_uris=["file:///tmp/view.jpg"],
    )

    try:
        pipeline_run(job)
    except ValueError as exc:
        assert "at least two images" in str(exc)
    else:
        raise AssertionError("expected single-view to be rejected")
