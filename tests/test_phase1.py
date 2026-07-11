from pathlib import Path

from spatial_ingestion.batch_normalization.normalizer import BatchNormalizer
from spatial_ingestion.live_stream.manager import LiveStreamManager
from spatial_ingestion.media_classifier.router import MediaClassifierRouter, MediaItemDescriptor
from spatial_ingestion.metadata.schema import SourceType, Track, UnifiedSpatialIngestionSchema
from spatial_ingestion.sync.multi_source import MultiSourceSyncer
from spatial_ingestion.test_harness.media_factory import (
    create_live_frame,
    create_sample_image,
    create_sample_video,
)


def test_image_routing_and_schema(tmp_path: Path) -> None:
    image = create_sample_image(tmp_path / "sample.jpg")
    router = MediaClassifierRouter()
    decision = router.classify_static([MediaItemDescriptor(image.name, "image/jpeg")])
    payload = BatchNormalizer().normalize([image], decision)

    assert decision.input_type == SourceType.SINGLE_IMAGE
    assert decision.track == Track.BATCH
    assert payload.source_type == SourceType.SINGLE_IMAGE
    assert payload.frame_count == 1
    assert payload.frames[0].resolution == (1024, 1024)


def test_video_motion_adaptive_sampling(tmp_path: Path) -> None:
    video = create_sample_video(tmp_path / "sample.mp4")
    router = MediaClassifierRouter()
    decision = router.classify_static([MediaItemDescriptor(video.name, "video/mp4")])
    payload = BatchNormalizer().normalize([video], decision)

    assert payload.source_type == SourceType.SINGLE_VIDEO
    assert payload.frame_count is not None
    assert 1 < payload.frame_count < 48
    assert payload.metadata["sampling"] == "motion_adaptive_frame_diff"
    assert max(frame.motion_score or 0 for frame in payload.frames) > 0.05


def test_live_stream_buffer_schema() -> None:
    manager = LiveStreamManager()
    payload = manager.open_stream("mock-live")
    for index in range(70):
        manager.push_frame("mock-live", create_live_frame(index))

    buffer = manager.get_buffer("mock-live")
    assert payload.source_type == SourceType.LIVE_STREAM
    assert payload.compute_priority_score == 1.0
    assert payload.is_stream
    assert buffer.latest() is not None
    assert buffer.dropped_frames > 0


def test_video_folder_sync_map(tmp_path: Path) -> None:
    video_a = create_sample_video(tmp_path / "cam_a.mp4")
    video_b = create_sample_video(tmp_path / "cam_b.mp4")
    router = MediaClassifierRouter()
    decision = router.classify_static(
        [
            MediaItemDescriptor(video_a.name, "video/mp4"),
            MediaItemDescriptor(video_b.name, "video/mp4"),
        ]
    )
    payload = BatchNormalizer().normalize([video_a, video_b], decision)

    assert payload.source_type == SourceType.VIDEO_FOLDER
    assert payload.sync_group_id is not None
    assert payload.sync_map
    assert all(len(entry.aligned_frames) == 2 for entry in payload.sync_map)


def test_unified_schema_contract_minimum() -> None:
    schema = UnifiedSpatialIngestionSchema(
        source_type=SourceType.LIVE_STREAM,
        track=Track.LIVE,
        resolution=(160, 120),
        frame_count=None,
        is_stream=True,
        compute_priority_score=1.0,
        live_stream_handle="stream-test",
    )
    dumped = schema.model_dump()
    for key in [
        "source_type",
        "resolution",
        "frame_count",
        "is_stream",
        "camera_intrinsics",
        "compute_priority_score",
        "timestamp",
        "sync_group_id",
    ]:
        assert key in dumped

