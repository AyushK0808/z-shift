from spatial_ingestion.metadata.schema import (
    FrameReference,
    SourceType,
    SyncMapEntry,
    Track,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.reconstruction import (
    Mast3rBackend,
    ReconstructionBackendRegistry,
    ReconstructionArtifactKind,
    ReconstructionJobBuilder,
    ReconstructionMode,
)


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


def test_mast3r_backend_builds_execution_plan() -> None:
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
    backend = ReconstructionBackendRegistry([Mast3rBackend()]).resolve_for_job(job)
    plan = backend.plan(job)

    assert plan.backend_name == "mast3r"
    assert [artifact.kind for artifact in plan.expected_artifacts] == [
        ReconstructionArtifactKind.POINT_CLOUD,
        ReconstructionArtifactKind.POSES,
        ReconstructionArtifactKind.RUN_MANIFEST,
        ReconstructionArtifactKind.MESH,
    ]
