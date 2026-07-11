from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from spatial_ingestion.batch_normalization.normalizer import BatchNormalizer
from spatial_ingestion.live_stream.manager import LiveStreamManager
from spatial_ingestion.media_classifier.router import MediaClassifierRouter, MediaItemDescriptor
from spatial_ingestion.metadata.schema import SourceType, Track
from spatial_ingestion.test_harness.media_factory import (
    create_live_frame,
    create_sample_image,
    create_sample_video,
)


def run_harness() -> None:
    router = MediaClassifierRouter()
    normalizer = BatchNormalizer()
    live = LiveStreamManager()

    with TemporaryDirectory(prefix="spatial_harness_") as temp_dir:
        temp = Path(temp_dir)
        image = create_sample_image(temp / "sample.jpg")
        video = create_sample_video(temp / "sample.mp4")

        image_decision = router.classify_static(
            [MediaItemDescriptor(filename=image.name, mime_type="image/jpeg")]
        )
        image_payload = normalizer.normalize([image], image_decision)
        assert image_decision.input_type == SourceType.SINGLE_IMAGE
        assert image_payload.track == Track.BATCH
        assert image_payload.frame_count == 1

        video_decision = router.classify_static(
            [MediaItemDescriptor(filename=video.name, mime_type="video/mp4")]
        )
        video_payload = normalizer.normalize([video], video_decision)
        assert video_decision.input_type == SourceType.SINGLE_VIDEO
        assert video_payload.track == Track.BATCH
        assert video_payload.frame_count and video_payload.frame_count > 1
        assert any((frame.motion_score or 0) > 0 for frame in video_payload.frames)

        stream_decision = router.classify_stream("websocket", "mock-live")
        live_payload = live.open_stream("mock-live")
        for index in range(5):
            live.push_frame("mock-live", create_live_frame(index))
        assert stream_decision.input_type == SourceType.LIVE_STREAM
        assert live_payload.is_stream
        assert live.get_buffer("mock-live").latest() is not None

        print("Phase 1 harness passed")
        print(f"image: {image_payload.model_dump_json(indent=2)}")
        print(f"video_frames: {video_payload.frame_count}")
        print(f"live_handle: {live_payload.live_stream_handle}")

