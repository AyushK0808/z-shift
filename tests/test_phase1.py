from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from spatial_ingestion.ingestion_gateway.api import create_app
from spatial_ingestion.batch_normalization.normalizer import BatchNormalizer
from spatial_ingestion.live_stream.manager import LiveStreamManager
from spatial_ingestion.media_classifier.router import MediaClassifierRouter, MediaItemDescriptor
from spatial_ingestion.metadata.schema import (
    FrameReference,
    SourceType,
    Track,
    UnifiedSpatialIngestionSchema,
)
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
    assert payload.frames[0].resolution == (320, 240)
    assert payload.frames[0].camera_intrinsics is not None


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
    assert all(frame.resolution == (320, 240) for frame in payload.frames)


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


def test_syncer_estimates_constant_timestamp_offset() -> None:
    syncer = MultiSourceSyncer()
    group_id = UnifiedSpatialIngestionSchema.new_sync_group_id()
    cam_a = [
        FrameReference(frame_id=f"a{i}", index=i, timestamp_ms=i * 500.0, motion_score=score)
        for i, score in enumerate([0.01, 0.2, 0.04, 0.3])
    ]
    cam_b = [
        FrameReference(frame_id=f"b{i}", index=i, timestamp_ms=(i * 500.0) + 500.0, motion_score=score)
        for i, score in enumerate([0.01, 0.2, 0.04, 0.3])
    ]

    sync_map = syncer.build_sync_map({"cam_a": cam_a, "cam_b": cam_b}, group_id)

    assert sync_map
    assert sync_map[0].offsets_ms["cam_b"] == -500.0


def test_classifier_rejects_unknown_mixed_payload() -> None:
    router = MediaClassifierRouter()
    decision = router.classify_static(
        [
            MediaItemDescriptor("photo.jpg", "image/jpeg"),
            MediaItemDescriptor("notes.txt", "text/plain"),
        ]
    )
    assert decision.input_type == SourceType.UNKNOWN
    assert "notes.txt" in decision.reason


def test_stream_router_rejects_unimplemented_transports() -> None:
    router = MediaClassifierRouter()
    decision = router.classify_stream("rtsp", "camera-1")
    assert decision.input_type == SourceType.UNKNOWN


def test_upload_endpoint_preserves_original_and_rejects_junk(tmp_path: Path) -> None:
    image = create_sample_image(tmp_path / "sample.jpg")
    app = create_app()
    client = TestClient(app)

    with image.open("rb") as handle:
        response = client.post(
            "/v1/ingest/uploads",
            files=[("files", ("sample.jpg", handle, "image/jpeg"))],
            headers={"Authorization": "Bearer upload-test"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["frames"][0]["original_uri"].startswith("file:///")
    assert payload["metadata"]["originals_preserved"] is True

    bad_response = client.post(
        "/v1/ingest/uploads",
        files=[
            ("files", ("sample.jpg", image.read_bytes(), "image/jpeg")),
            ("files", ("notes.txt", b"not an image", "text/plain")),
        ],
        headers={"Authorization": "Bearer upload-test"},
    )
    assert bad_response.status_code == 415


def test_websocket_requires_owned_precreated_stream() -> None:
    app = create_app()
    client = TestClient(app)
    headers = {"Authorization": "Bearer stream-owner"}

    connect = client.post(
        "/v1/ingest/streams/connect",
        json={"transport": "websocket", "stream_id": "owned-stream"},
        headers=headers,
    )
    assert connect.status_code == 200

    ok, encoded = cv2.imencode(".jpg", create_live_frame(0))
    assert ok

    with client.websocket_connect(
        "/v1/ingest/streams/owned-stream/frames",
        headers=headers,
    ) as websocket:
        websocket.send_bytes(encoded.tobytes())
        ack = websocket.receive_json()
        assert ack["accepted"] is True

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/ingest/streams/missing-stream/frames"):
            pass


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
