from spatial_ingestion.generation_handoff import GenerationHandoffBuilder, GenerationMode
from spatial_ingestion.metadata.schema import (
    FrameReference,
    SourceType,
    SyncMapEntry,
    Track,
    UnifiedSpatialIngestionSchema,
)


def test_single_image_maps_to_single_view() -> None:
    payload = UnifiedSpatialIngestionSchema(
        source_type=SourceType.SINGLE_IMAGE,
        track=Track.BATCH,
        resolution=(1024, 1024),
        frame_count=1,
        is_stream=False,
        compute_priority_score=0.4,
        frames=[
            FrameReference(
                frame_id="frame_1",
                uri="file:///tmp/frame_1.jpg",
                index=0,
                source_id="image_a",
                resolution=(1024, 1024),
            )
        ],
    )

    handoff = GenerationHandoffBuilder().build(payload)

    assert handoff.mode == GenerationMode.SINGLE_VIEW
    assert handoff.supports_reconstruction is True
    assert [frame.frame_id for frame in handoff.primary_frames] == ["frame_1"]


def test_image_folder_maps_to_multi_view_in_source_order() -> None:
    payload = UnifiedSpatialIngestionSchema(
        source_type=SourceType.IMAGE_FOLDER,
        track=Track.BATCH,
        resolution=(1024, 1024),
        frame_count=2,
        is_stream=False,
        compute_priority_score=0.5,
        frames=[
            FrameReference(
                frame_id="frame_b",
                uri="file:///tmp/frame_b.jpg",
                index=1,
                source_id="view_b",
                resolution=(1024, 1024),
            ),
            FrameReference(
                frame_id="frame_a",
                uri="file:///tmp/frame_a.jpg",
                index=0,
                source_id="view_a",
                resolution=(1024, 1024),
            ),
        ],
    )

    handoff = GenerationHandoffBuilder().build(payload)

    assert handoff.mode == GenerationMode.MULTI_VIEW
    assert [frame.frame_id for frame in handoff.primary_frames] == ["frame_a", "frame_b"]


def test_video_folder_builds_synchronized_view_groups() -> None:
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

    handoff = GenerationHandoffBuilder().build(payload)

    assert handoff.mode == GenerationMode.SYNCHRONIZED_VIEWS
    assert len(handoff.sync_view_groups) == 2
    assert handoff.sync_view_groups[0].frames_by_source["cam_a"].frame_id == "cam_a_0"
    assert handoff.sync_view_groups[0].frames_by_source["cam_b"].frame_id == "cam_b_0"


def test_live_stream_is_marked_not_ready_for_reconstruction() -> None:
    payload = UnifiedSpatialIngestionSchema(
        source_type=SourceType.LIVE_STREAM,
        track=Track.LIVE,
        resolution=None,
        frame_count=None,
        is_stream=True,
        compute_priority_score=1.0,
        live_stream_handle="stream_1",
    )

    handoff = GenerationHandoffBuilder().build(payload)

    assert handoff.mode == GenerationMode.LIVE_STREAM
    assert handoff.supports_reconstruction is False
    assert handoff.warnings == ["live stream payloads are not reconstruction-ready in Phase 2"]
